"""Controls for "the answer is already fixed before the reasoning is written" (interp/commit_lit.py --mode filler).

The truncation result reads the forced answer after 0% of the agent's own reasoning and finds it already equals the answer
the agent finally writes in 67% of debate answer changes. Two things could produce that without the decision being made:

  filler      the same round message followed by neutral text of the SAME token length ("Let me work through this
              carefully." repeated), then the forced answer. If merely occupying that many tokens moves the read-out,
              the truncation curve measures length, not commitment.
  cross_item  the same round message followed by ANOTHER item's reasoning by the same agent in the same family, then the
              forced answer. If the read-out still lands on this item's final answer, the agent's own reasoning is not
              what carries it.

Both are compared against the matched fraction-0 and fraction-1.0 rows of the commit run on the same episodes.
Outputs: outputs/tables/lit_filler_<M>_<dataset>_{summary,by_group}.csv
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import *  # noqa
from stats import fast_cluster_mean_ci

CORE = {r["id"]: r for r in load_core()}
cluster_of = lambda i: CORE[i]["meta"].get("analysis_group") if CORE[i]["stratum"] == "persuasion" else CORE[i]["source_item_id"]
KEY = ["item_id", "cond", "agent", "round"]


def ci(x, c):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 5:
        return dict(est=np.nan, lower=np.nan, upper=np.nan, n_clusters=0, n=int(ok.sum()))
    e, lo, hi, p, n = fast_cluster_mean_ci(x[ok], np.asarray(c)[ok])
    return dict(est=e, lower=lo, upper=hi, n_clusters=n, n=int(ok.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="lit1200")
    a = ap.parse_args()
    # a comma-separated dataset must not become a comma in a filename; pooled runs use the shared "full2400" name
    tag = f"{a.model}_{a.dataset}" if "," not in a.dataset else f"{a.model}_full2400"
    # this analysis reads two sources; both have to pool, and the commit one was missed on the first pass
    parts = [x for x in a.dataset.split(",") if x]
    C = pd.concat([pd.DataFrame(read_jsonl(ACTS / a.model / f"lit_commit_{p2}" / "rows.jsonl")) for p2 in parts],
                  ignore_index=True)
    # rows.jsonl is record-keyed, so comma-separated halves union safely; a single name behaves as before.
    F = pd.concat([pd.DataFrame(read_jsonl(ACTS / a.model / f"lit_filler_{p2}" / "rows.jsonl")) for p2 in parts],
                  ignore_index=True)
    if len(parts) > 1:
        print(f"[filler] pooled {len(parts)} halves -> {F.item_id.nunique()} distinct items", flush=True)
    hit = lambda d: [float(max(lp, key=lp.get) == ans) for lp, ans in zip(d.lp_letters, d.answer)]
    C["is_final"], F["is_final"] = hit(C), hit(F)

    arms = {}
    for f, nm in ((0.0, "frac0"), (1.0, "frac1")):
        g = C[np.isclose(C.frame.astype(float), f)]
        arms[nm] = g.set_index(KEY).is_final
    for nm in ("filler", "cross_item"):
        g = F[F.frame == nm]
        arms[nm] = g.set_index(KEY).is_final

    D = pd.DataFrame(arms)
    meta = C[np.isclose(C.frame.astype(float), 0.0)].set_index(KEY)[["changed", "reasoning_chars"]]
    D = D.join(meta, how="inner").reset_index()
    D = D[D.filler.notna()]  # the filler run covers the same episode selection, but join on what is actually present
    D["cluster"] = D.item_id.map(cluster_of)
    D["domain"] = D.item_id.map(lambda i: CORE[i]["primary_domain"])
    D["family"] = D.item_id.map(lambda i: family_of(CORE[i]))
    print(f"[filler] {len(D)} matched episodes ({int(D.changed.sum())} answer changes), "
          f"{D.cluster.nunique()} clusters; cross_item present for {int(D.cross_item.notna().sum())}")
    D.to_csv(TABLES / f"lit_filler_{tag}_episodes.csv", index=False)

    rows = []
    for cond, dc in (("debate", D[D.cond == "debate"]), ("reflect", D[D.cond == "reflect"])):
        for chg, dg in (("changed", dc[dc.changed]), ("kept", dc[~dc.changed])):
            if not len(dg):
                continue
            for arm in ("frac0", "filler", "cross_item", "frac1"):
                rows.append(dict(cond=cond, group=chg, arm=arm, **ci(dg[arm], dg.cluster)))
            for nm, x in (("filler - frac0", dg.filler - dg.frac0),
                          ("cross_item - frac0", dg.cross_item - dg.frac0),
                          ("frac1 - cross_item", dg.frac1 - dg.cross_item)):
                rows.append(dict(cond=cond, group=chg, arm=nm, **ci(x, dg.cluster)))
    S = pd.DataFrame(rows)
    S.to_csv(TABLES / f"lit_filler_{tag}_summary.csv", index=False)
    print(S.round(3).to_string(index=False))

    grp = []
    dchg = D[(D.cond == "debate") & D.changed]
    for lvl, col in (("domain", "domain"), ("family", "family")):
        for v, g in dchg.groupby(col):
            for arm in ("frac0", "filler", "cross_item", "frac1"):
                grp.append(dict(level=lvl, group=v, arm=arm, **ci(g[arm], g.cluster)))
    pd.DataFrame(grp).to_csv(TABLES / f"lit_filler_{tag}_by_group.csv", index=False)
    print("[done] lit_filler")


if __name__ == "__main__":
    main()
