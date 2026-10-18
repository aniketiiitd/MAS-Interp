"""CPT patching: recovery maps and interaction localization (feeds F5).

Orientation: s = +1 when the donor is the expert (provenance) or strong (content) world, so effects are expert-minus-anon.
recovery(layer, group, direction) = mean_s[s*(m_patched - m_recip)] / mean_s[s*(m_donor - m_recip)] over item pairs.
direction "suff" = expert->anon (is the expert state sufficient?), "nec" = anon->expert (is it necessary?).
Interaction localization: I_patch = s*dPatch(strong) - s*dPatch(weak); I_total analogous; recovery = mean I_patch / mean I_total.
CIs: item bootstrap (clusters = items; hero items are distinct source units).
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa


def boot_ratio(num_by_item, den_by_item, B=2000, seed=240091):
    items = num_by_item.index.intersection(den_by_item.index)
    n = num_by_item[items].to_numpy()
    d = den_by_item[items].to_numpy()
    if len(items) == 0 or abs(d.mean()) < 1e-9:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(items), size=(B, len(items)))
    r = n[idx].mean(1) / d[idx].mean(1)
    return n.mean() / d.mean(), *np.quantile(r, [0.025, 0.975])


def load(model, split="hero_48"):
    df = pd.DataFrame(read_jsonl(ACTS / model / f"patch_{split}" / "patch.jsonl"))
    df["donor_prov"] = df.donor.map(lambda d: d["prov"])
    df["donor_qual"] = df.donor.map(lambda d: d["qual"])
    df["adv"] = df.donor.map(lambda d: d["adv"])
    df["qual_ctx"] = df.recipient.map(lambda d: d["qual"])
    clean = df[df["mode"] == "clean"][["item_id", "kind", "donor_prov", "donor_qual", "adv", "qual_ctx", "madv_patched"]].rename(
        columns={"madv_patched": "madv_donor"})
    df = df[df["mode"] != "clean"].merge(clean, on=["item_id", "kind", "donor_prov", "donor_qual", "adv", "qual_ctx"], how="left")
    df["s"] = np.where(df.kind == "prov", np.where(df.donor_prov == "expert", 1, -1), np.where(df.donor_qual == "strong", 1, -1))
    df["direction"] = np.where(df.s > 0, "suff", "nec")
    df["d_patch"] = df.s * (df.madv_patched - df.madv_recip)
    df["d_total"] = df.s * (df.madv_donor - df.madv_recip)
    return df


def recovery_table(df):
    rows = []
    for (kind, group, layer, mode, direction), g in df.groupby(["kind", "group", "layer", "mode", "direction"]):
        num = g.groupby("item_id").d_patch.mean()
        den = g.groupby("item_id").d_total.mean()
        est, lo, hi = boot_ratio(num, den)
        rows.append(dict(kind=kind, group=group, layer=layer, mode=mode, direction=direction, recovery=est, lower=lo, upper=hi,
                         mean_d_patch=num.mean(), mean_d_total=den.mean(), n_items=len(num)))
    both = []
    for (kind, group, layer, mode), g in df.groupby(["kind", "group", "layer", "mode"]):
        num = g.groupby("item_id").d_patch.mean()
        den = g.groupby("item_id").d_total.mean()
        est, lo, hi = boot_ratio(num, den)
        both.append(dict(kind=kind, group=group, layer=layer, mode=mode, direction="both", recovery=est, lower=lo, upper=hi,
                         mean_d_patch=num.mean(), mean_d_total=den.mean(), n_items=len(num)))
    return pd.DataFrame(rows + both)


def interaction_table(df):
    p = df[(df.kind == "prov")]
    rows = []
    for (group, layer, mode), g in p.groupby(["group", "layer", "mode"]):
        per = g.groupby(["item_id", "qual_ctx"])[["d_patch", "d_total"]].mean().unstack("qual_ctx")
        if ("d_patch", "strong") not in per or ("d_patch", "weak") not in per:
            continue
        num = per[("d_patch", "strong")] - per[("d_patch", "weak")]
        den = per[("d_total", "strong")] - per[("d_total", "weak")]
        est, lo, hi = boot_ratio(num.dropna(), den.dropna())
        rows.append(dict(group=group, layer=layer, mode=mode, i_patch=num.mean(), i_total=den.mean(), recovery=est, lower=lo, upper=hi))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--split", default="hero_48")
    a = ap.parse_args()
    df = load(a.model, a.split)
    rt = recovery_table(df)
    it = interaction_table(df)
    rt.to_csv(TABLES / f"patch_recovery_{a.model}_{a.split}.csv", index=False)
    it.to_csv(TABLES / f"patch_interaction_{a.model}_{a.split}.csv", index=False)
    # onset: first layer at which answer-token patching recovers >50% of the provenance effect (both directions)
    ans = rt[(rt.kind == "prov") & (rt.group == "answer") & (rt["mode"] == "patch") & (rt.direction == "both")].sort_values("layer")
    src = rt[(rt.kind == "prov") & (rt.group == "src") & (rt["mode"] == "patch") & (rt.direction == "both")].sort_values("layer")
    onset = ans[ans.recovery > 0.5].layer.min() if len(ans) else None
    handoff = src[src.recovery < 0.5].layer.min() if len(src) else None
    summ = dict(total_prov_effect=float(df[(df.kind == "prov")].groupby("item_id").d_total.mean().mean()),
                total_qual_effect=float(df[(df.kind == "qual")].groupby("item_id").d_total.mean().mean()),
                answer_onset_layer=None if onset is None or pd.isna(onset) else int(onset),
                src_handoff_layer=None if handoff is None or pd.isna(handoff) else int(handoff))
    json.dump(summ, open(TABLES / f"patch_summary_{a.model}_{a.split}.json", "w"), indent=1)
    print(summ)
    print(rt[(rt["mode"] == "patch") & (rt.direction == "both")].pivot_table(index="layer", columns=["kind", "group"], values="recovery").round(2).to_string())
