"""V3-C1'' gate evaluation: status vs stated reliability in behaviour and across depth (PLAN_V2.md addendum A5).

For each grid layer, the decision-token direction u_l = mean(expert worlds) - mean(anon worlds) (unit norm; fit on the evaluated
worlds with expert/anon labels only). Projections of the six status x record descriptor worlds give status and record effects
(disagreement worlds, averaged per item); status share = mean status effect / (mean status effect + mean record effect).
Behaviour uses the adoption margin m_adv. Item-cluster bootstrap (B=500) for shares and the mid - late difference.
Usage: python analysis/desc_depth.py --model Q9 --dir cpt_desc_release_600
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import *  # noqa

CORE = {r["id"]: r for r in load_core()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dir", default="cpt_desc_release_600")
    ap.add_argument("--B", type=int, default=500)
    a = ap.parse_args()
    d = ACTS / a.model / a.dir
    W = pd.DataFrame(read_jsonl(d / "worlds.jsonl"))
    meta = json.load(open(d / "meta.json"))
    grid = meta["grid"]
    n_layers = {"Q9": 32, "Q4": 32, "G4": 42}.get(a.model, max(grid) + 1)
    A = np.load(d / "acts.npy", mmap_mode="r")
    W["cluster"] = W.item_id.map(lambda i: CORE[i]["meta"].get("analysis_group") if CORE[i]["stratum"] == "persuasion" else CORE[i]["source_item_id"])
    exp_m, anon_m = (W.prov == "expert").to_numpy(), (W.prov == "anon").to_numpy()
    cols = []
    for gi, li in enumerate(grid):
        X = np.asarray(A[:, 0, gi], np.float32)
        u = X[exp_m].mean(0) - X[anon_m].mean(0)
        u /= np.linalg.norm(u)
        W[f"L{li}"] = X @ u
        cols.append(f"L{li}")
    dis = W[W.adv != W.prior]
    per = dis.groupby(["item_id", "cluster", "prov"])[["m_adv"] + cols].mean().unstack("prov")

    def shares(P):
        out = {}
        for c in ["m_adv"] + cols:
            m = P[c].mean()
            st = ((m.prof_bad - m.ord_bad) + (m.prof_good - m.ord_good)) / 2
            rc = ((m.ord_good - m.ord_bad) + (m.prof_good - m.prof_bad)) / 2
            out[c] = dict(status=st, record=rc, share=st / (st + rc), prof_bad_vs_none=m.prof_bad - m.none)
        return out

    est = shares(per)
    clusters = per.index.get_level_values("cluster").unique().to_numpy()
    groups = {c: per.xs(c, level="cluster", drop_level=False) for c in clusters}
    rng = np.random.default_rng(240091)
    boots = [shares(pd.concat([groups[c] for c in rng.choice(clusters, len(clusters))])) for _ in range(a.B)]
    rows = []
    for c in ["m_adv"] + cols:
        for k in ("status", "record", "share", "prof_bad_vs_none"):
            v = np.array([b[c][k] for b in boots])
            rows.append(dict(representation=c, depth=(np.nan if c == "m_adv" else int(c[1:]) / (n_layers - 1)), metric=k,
                             est=est[c][k], lower=np.quantile(v, .025), upper=np.quantile(v, .975)))
    T = pd.DataFrame(rows)
    mid = min(cols, key=lambda c: abs(int(c[1:]) / (n_layers - 1) - 0.39))
    late = min(cols, key=lambda c: abs(int(c[1:]) / (n_layers - 1) - 0.71))
    diff = np.array([b[mid]["share"] - b[late]["share"] for b in boots])
    closer = np.array([abs(b[late]["share"] - b["m_adv"]["share"]) < abs(b[mid]["share"] - b["m_adv"]["share"]) for b in boots])
    beh = T[(T.representation == "m_adv") & (T.metric == "share")].iloc[0]
    pbn = T[(T.representation == "m_adv") & (T.metric == "prof_bad_vs_none")].iloc[0]
    gate = dict(model=a.model, dir=a.dir, mid=mid, late=late,
                behaviour_share=beh[["est", "lower", "upper"]].to_dict(), prof_bad_vs_none=pbn[["est", "lower", "upper"]].to_dict(),
                mid_share=est[mid]["share"], late_share=est[late]["share"],
                mid_minus_late=dict(est=est[mid]["share"] - est[late]["share"], lower=float(np.quantile(diff, .025)), upper=float(np.quantile(diff, .975))),
                late_closer_to_behaviour_frac=float(closer.mean()))
    gate["PASS_i"] = bool(beh.upper < 0.30)
    gate["PASS_ii"] = bool(pbn.upper < 0)
    gate["PASS_iii"] = bool(gate["mid_minus_late"]["est"] > 0.20 and gate["mid_minus_late"]["lower"] > 0 and closer.mean() > 0.975)
    tag = f"{a.model}_{a.dir}"
    T.to_csv(TABLES / f"v3_desc_depth_{tag}.csv", index=False)
    json.dump(gate, open(TABLES / f"v3_desc_depth_gate_{tag}.json", "w"), indent=1, default=float)
    print(json.dumps(gate, indent=1, default=float))
    print(T[T.metric == "share"].round(3).to_string())


if __name__ == "__main__":
    main()
