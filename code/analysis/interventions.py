"""E16 steering dose-response and E21 recourse summaries (+ E17 quantization invariance)."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import *  # noqa
from stats import fast_cluster_mean_ci


def steering(model="Q9"):
    p = ACTS / model / "steer" / "steer.jsonl"
    if not p.exists():
        return None
    S = pd.DataFrame(read_jsonl(p))
    base = S[S.alpha == 0].set_index(["item_id", "probe", "w_adv", "w_qual", "w_prov"])[["m_adv"]].rename(columns={"m_adv": "m0"})
    S = S.join(base, on=["item_id", "probe", "w_adv", "w_qual", "w_prov"])
    S["dm"] = S.m_adv - S.m0
    rows = []
    for (probe, mode, alpha), g in S.groupby(["probe", "mode", "alpha"]):
        per = g.groupby("item_id").agg(dm=("dm", "mean"), kl=("kl", "mean"), mass=("label_mass", "mean"))
        est, lo, hi, p_, n = fast_cluster_mean_ci(per.dm.to_numpy(), per.index.to_numpy())
        rows.append(dict(probe=probe, mode=mode, alpha=alpha, d_m_adv=est, lower=lo, upper=hi, kl=per.kl.mean(),
                         label_mass=per.mass.mean(), n_items=n))
    T = pd.DataFrame(rows)
    # monotonicity + direction specificity: slope of d_m_adv on alpha per mode
    slopes = []
    for (probe, mode), g in T.groupby(["probe", "mode"]):
        g = pd.concat([g, pd.DataFrame([dict(alpha=0, d_m_adv=0.0)])]) if 0 not in set(g.alpha) else g
        g = g.sort_values("alpha")
        slope = np.polyfit(g.alpha, g.d_m_adv, 1)[0]
        mono = bool(np.all(np.diff(g.d_m_adv.to_numpy()) > 0) or np.all(np.diff(g.d_m_adv.to_numpy()) < 0))
        slopes.append(dict(probe=probe, mode=mode, slope=slope, monotone=mono))
    T.to_csv(TABLES / f"steer_{model}.csv", index=False)
    pd.DataFrame(slopes).to_csv(TABLES / f"steer_slopes_{model}.csv", index=False)
    return T, pd.DataFrame(slopes)


def recourse(model="Q9"):
    p = ACTS / model / "recourse" / "recourse.jsonl"
    if not p.exists():
        return None
    R = pd.DataFrame(read_jsonl(p))
    before = R[R.route == "before"].set_index(["item_id", "scenario"])[["m12", "p_correct"]].add_suffix("_before")
    R = R.join(before, on=["item_id", "scenario"])
    R["d_m12"] = R.m12 - R.m12_before
    R["d_pcorrect"] = R.p_correct - R.p_correct_before
    rows = []
    for (scen, route), g in R.groupby(["scenario", "route"]):
        est, lo, hi, p_, n = fast_cluster_mean_ci(g.d_pcorrect.to_numpy(), g.item_id.to_numpy())
        rows.append(dict(scenario=scen, route=route, d_p_correct=est, lower=lo, upper=hi, p_correct=g.p_correct.mean(),
                         d_m12=g.d_m12.mean(), kl=g.kl_vs_before.mean(), label_mass=g.label_mass.mean(), n=n,
                         repaired_rate=float((g.p_correct > 0.5).mean())))
    T = pd.DataFrame(rows)
    T.to_csv(TABLES / f"recourse_{model}.csv", index=False)
    return T


def quantization(model="Q9", split="release_600"):
    base = ACTS / model / f"cpt_vllm_{split}" / "worlds.jsonl"
    if not base.exists():
        return None
    B = pd.DataFrame(read_jsonl(base))
    key = ["item_id", "template", "adv", "qual", "prov", "prior"]
    out = []
    for q in ("fp8", "bitsandbytes"):
        p = ACTS / model / f"cpt_vllm_{split}_{q}" / "worlds.jsonl"
        if not p.exists():
            continue
        Q = pd.DataFrame(read_jsonl(p))
        M = B.merge(Q, on=key, suffixes=("_bf16", f"_{q}"))
        d = M[f"m12_{q}"] - M["m12_bf16"]
        out.append(dict(quant=q, n=len(M), mean_abs_d_m12=float(d.abs().mean()), corr_m12=float(np.corrcoef(M.m12_bf16, M[f"m12_{q}"])[0, 1]),
                        argmax_agree=float((M.argmax_label_bf16 == M[f"argmax_label_{q}"]).mean())))
    T = pd.DataFrame(out)
    T.to_csv(TABLES / f"quantization_{model}.csv", index=False)
    return T


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    for f in (steering, recourse, quantization):
        r = f()
        if r is not None:
            print(r if not isinstance(r, tuple) else r[0].round(3).to_string() + "\n" + r[1].round(3).to_string())
