"""E14 directed edges (A <- B) and E15 heterogeneous triads; feeds F7.

Per edge: influence = mean P(adopt) with the full message; content = full - verdict_only; provenance = expert - anon;
self-normalized influence = influence(A<-B) - influence(A<-A). Reciprocity = influence(A<-B) - influence(B<-A) on the shared
item set. Item-cluster bootstrap CIs.
E15: per model, P(final verdict = its own turn-0 verdict | disagreement at turn 0) and adoption of other models' turn-0
verdicts, split by slot position; permutation-averaged.
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import *  # noqa
from stats import fast_cluster_mean_ci


def edges():
    fs = sorted(glob.glob(str(ACTS / "edges" / "edges_*.jsonl")))
    if not fs:
        return None
    E = pd.concat([pd.DataFrame(read_jsonl(f)) for f in fs], ignore_index=True)
    core = {r["id"]: r for r in load_core()}
    E["cluster"] = E.item_id.map(lambda i: core[i]["meta"].get("analysis_group") if core[i]["stratum"] == "persuasion" else core[i]["source_item_id"])
    W = E.pivot_table(index=["receiver", "sender", "item_id", "cluster", "domain"], columns="variant", values="p_adopt").reset_index()
    W["content"] = W["full"] - W["verdict_only"]
    W["provenance"] = W["expert"] - W["anon"]
    rows = []
    for (a, b), g in W.groupby(["receiver", "sender"]):
        for c in ("full", "content", "provenance"):
            est, lo, hi, p, n = fast_cluster_mean_ci(g[c].to_numpy(), g.cluster.to_numpy())
            rows.append(dict(receiver=a, sender=b, component=c, estimate=est, lower=lo, upper=hi, n_items=len(g)))
    T = pd.DataFrame(rows)
    selfinf = T[(T.receiver == T.sender) & (T.component == "full")].set_index("receiver").estimate
    T["self_baseline"] = T.receiver.map(selfinf)
    # reciprocity on shared items
    rec = []
    full = W.set_index(["receiver", "sender", "item_id"])["full"]
    models = sorted(set(W.receiver))
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            try:
                ab = full.loc[(a, b)]
                ba = full.loc[(b, a)]
            except KeyError:
                continue
            shared = ab.index.intersection(ba.index)
            if len(shared) < 20:
                continue
            d = (ab[shared] - ba[shared]).to_numpy()
            cl = W.set_index("item_id").cluster.groupby(level=0).first().reindex(shared).to_numpy()
            est, lo, hi, p, n = fast_cluster_mean_ci(d, cl)
            rec.append(dict(a=a, b=b, a_from_b_minus_b_from_a=est, lower=lo, upper=hi, n_items=len(shared)))
    T.to_csv(TABLES / "edges_influence.csv", index=False)
    pd.DataFrame(rec).to_csv(TABLES / "edges_reciprocity.csv", index=False)
    return T


def triads():
    out = []
    for d in glob.glob(str(RAW / "multi" / "*" / "hero_48")):
        recs = []
        for f in glob.glob(d + "/t*_*.jsonl"):
            recs += read_jsonl(f)
        if not recs:
            continue
        R = pd.DataFrame(recs)
        last = R.turn.max()
        t0 = R[R.turn == 0].set_index(["item_id", "perm", "agent"])
        fin = R[R.turn == last].set_index(["item_id", "perm", "agent"])
        for (iid, perm), g in t0.groupby(level=[0, 1]):
            v0 = g.verdict.to_dict()
            if len({x for x in v0.values() if x}) < 2:
                continue
            for (i_, p_, slot), row in g.iterrows():
                fv = fin.loc[(iid, perm, slot)].verdict if (iid, perm, slot) in fin.index else None
                others = [v0[k] for k in v0 if k[2] != slot]
                out.append(dict(triad=d.split("/")[-2], item_id=iid, perm=perm, slot=slot, model=row.model, t0=row.verdict, final=fv,
                                kept=(fv == row.verdict), adopted_other=(fv in others and fv != row.verdict)))
    if not out:
        return None
    T = pd.DataFrame(out)
    S = T.groupby(["triad", "model", "slot"])[["kept", "adopted_other"]].mean().reset_index()
    S.to_csv(TABLES / "triads_slot_summary.csv", index=False)
    T.groupby(["triad", "model"])[["kept", "adopted_other"]].mean().reset_index().to_csv(TABLES / "triads_model_summary.csv", index=False)
    return S


if __name__ == "__main__":
    e = edges()
    t = triads()
    pd.set_option("display.width", 220)
    if e is not None:
        print(e[e.component == "full"].pivot_table(index="receiver", columns="sender", values="estimate").round(2).to_string())
    if t is not None:
        print(t.round(3).to_string())
