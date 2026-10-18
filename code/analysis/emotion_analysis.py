"""E19 emotion transfer to debate (exploratory unless replicated): debate-minus-pool change per concept at each turn,
and whether pre-registered pressure-related directions predict revision at the next turn (item-clustered logistic GEE)."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import *  # noqa
from stats import bh_fdr, fast_cluster_mean_ci


def main(model="Q9", replay="replay_core_release_600", dataset="core"):
    P = pd.read_parquet(TABLES / f"emotion_debate_{model}_{replay}.parquet")
    V = pd.read_csv(TABLES / f"emotion_validation_{model}.csv")
    tax = json.load(open(ROOT / "data/emotion/taxonomy.json"))
    passed = set(V[V.passes].concept)
    concepts = [c for g in tax["taxonomy"].values() for c in g]
    core = {r["id"]: r for r in load_core()}
    P["cluster"] = P.item_id.map(lambda i: core[i]["source_item_id"])
    last = P.turn.max()
    # per concept: debate - pool at final turn, averaged over agents
    agg = P[P.turn == last].groupby(["item_id", "cluster", "cond"])[[f"emo_{c}" for c in concepts]].mean().reset_index()
    wide = agg.pivot_table(index=["item_id", "cluster"], columns="cond").reset_index()
    rows = []
    for c in concepts:
        d = (wide[(f"emo_{c}", "debate")] - wide[(f"emo_{c}", "pool")]).to_numpy()
        est, lo, hi, p, n = fast_cluster_mean_ci(d, wide["cluster"].to_numpy().ravel())
        rows.append(dict(concept=c, group=next(g for g, cs in tax["taxonomy"].items() if c in cs), passes_validation=c in passed,
                         debate_minus_pool=est, lower=lo, upper=hi, p=p, n=n))
    F = pd.DataFrame(rows)
    F["q"] = bh_fdr(F.p)
    F.to_csv(TABLES / f"emotion_forest_{model}.csv", index=False)
    # revision prediction: emotion z at turn t predicts verdict change at t+1 (debate), controls: turn, agreement at t
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    D = P[P.cond.isin(["t0", "debate"])].copy()
    D = D.sort_values(["item_id", "agent", "turn"])
    D["next_verdict"] = D.groupby(["item_id", "agent"]).verdict.shift(-1)
    D = D[D.next_verdict.notna() & D.verdict.notna()]
    D["revise_next"] = (D.next_verdict != D.verdict).astype(int)
    grp = D.groupby(["item_id", "turn"]).verdict.agg(lambda s: s.value_counts().iloc[0] / 3.0).rename("agreement")
    D = D.join(grp, on=["item_id", "turn"])
    mods = []
    for c in tax["prereg_debate"]:
        if c not in passed:
            mods.append(dict(concept=c, passes_validation=False))
            continue
        D["z"] = D[f"emo_{c}"]
        try:
            m = smf.gee("revise_next ~ z + C(turn) + agreement", groups="cluster", data=D, family=sm.families.Binomial()).fit()
            mods.append(dict(concept=c, passes_validation=True, coef=m.params["z"], lower=m.conf_int().loc["z", 0],
                             upper=m.conf_int().loc["z", 1], p=m.pvalues["z"], n=len(D)))
        except Exception as e:  # noqa: BLE001
            mods.append(dict(concept=c, passes_validation=True, error=str(e)[:100]))
    M = pd.DataFrame(mods)
    if "p" in M:
        M["q"] = bh_fdr(M.p)
    M.to_csv(TABLES / f"emotion_revision_{model}.csv", index=False)
    print(F.sort_values("p").head(12).round(3).to_string())
    print(M.round(3).to_string())


if __name__ == "__main__":
    main()
