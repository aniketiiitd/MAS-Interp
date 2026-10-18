"""Debate dynamics turn by turn: what happens to a group of agents as the disagreement persists.

Everything here is about the multi-agent system, resolved by turn, and every quantity has its matched no-communication
control computed on the same items with the same number of turns. Works with any number of rounds, so the same script
serves the 3-round runs and the 12-round run.

Per turn k, over items where the agents were not already unanimous at round 0:
  switch_rate        share of agents whose answer differs from their own previous answer
  unanimity          share of items where all three agents agree
  wrong_unanimity    share of items unanimous on an answer that is not the evaluation target
  spread             mean number of distinct answers held across the group
  churn_after_unan   share of agents who move away from an answer their group had already reached unanimously
  destab             mean destabilisation-dimension score at the reading site (if concept vectors exist)

Outputs: outputs/tables/lit_turns_<M>_<dataset>.csv
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
from stats import fast_cluster_mean_ci

CORE = {r["id"]: r for r in load_core()}
cluster_of = lambda i: CORE[i]["meta"].get("analysis_group") if CORE[i]["stratum"] == "persuasion" else CORE[i]["source_item_id"]
RAW_LIT = ROOT / "outputs/raw_lit"


def ci(x, c):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 5:
        return dict(est=np.nan, lower=np.nan, upper=np.nan, n=int(ok.sum()))
    e, lo, hi, p, n = fast_cluster_mean_ci(x[ok], np.asarray(c)[ok])
    return dict(est=e, lower=lo, upper=hi, n=int(ok.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="lit1200")
    ap.add_argument("--tag", default="", help="raw_lit subdir suffix, e.g. _t12")
    a = ap.parse_args()
    d = RAW_LIT / a.model / (a.dataset + a.tag)
    meta = json.load(open(d / "meta.json"))
    R = meta["rounds"]
    ans = {}
    for f in d.glob("*.jsonl"):
        for x in read_jsonl(f):
            ans[(x["item_id"], x["cond"], x["agent"], x["round"], x.get("query"))] = x["answer"]
    items = sorted({k[0] for k in ans})
    tgt = {i: item_view(CORE[i])["target"] for i in items}
    print(f"[turns] {a.model} {a.dataset}{a.tag}: {len(items)} items, {R} rounds", flush=True)

    # items where the group was NOT already unanimous at round 0 -- the only ones where debate has anything to do
    contested = []
    for i in items:
        a0 = [ans.get((i, "r0", g, 0, None)) for g in range(3)]
        if None in a0 or len(set(a0)) == 1:
            continue
        contested.append(i)
    print(f"  contested at round 0: {len(contested)}/{len(items)}", flush=True)

    # Both scopes go in the table. "contested" is where debate has anything to do; "all" is what a system that runs
    # debate on everything actually gets, and a reader will ask for it.
    SCOPES = {"contested": contested, "all": items}
    rows = []
    for scope_name, scope_items in SCOPES.items():
      for cond in ("debate", "reflect"):
        for k in range(1, R + 1):
            sw, un, wu, sp, ch, cl = [], [], [], [], [], []
            for i in scope_items:
                cur = [ans.get((i, cond, g, k, None)) for g in range(3)]
                prv = [ans.get((i, "r0" if k == 1 else cond, g, k - 1, None)) for g in range(3)]
                if any(x is None for x in cur) or any(x is None for x in prv):
                    continue
                c_ = cluster_of(i)
                for g in range(3):
                    sw.append(float(cur[g] != prv[g]))
                    cl.append(c_)
                u = len(set(cur)) == 1
                un.append(float(u))
                sp.append(float(len(set(cur))))
                if tgt[i] is not None:
                    wu.append(float(u and cur[0] != tgt[i]))
                # did the group break a unanimity it had already reached?
                if len(set(prv)) == 1:
                    ch.append(float(not u))
            base = dict(cond=cond, round=k, scope=scope_name)
            rows.append(dict(base, metric="switch_rate", **ci(sw, cl)))
            ucl = [cluster_of(i) for i in scope_items][: len(un)]
            rows.append(dict(base, metric="unanimity", **ci(un, ucl)))
            rows.append(dict(base, metric="spread", **ci(sp, ucl)))
            if wu:
                rows.append(dict(base, metric="wrong_unanimity", **ci(wu, ucl[: len(wu)])))
            if ch:
                rows.append(dict(base, metric="churn_after_unanimity", **ci(ch, ucl[: len(ch)])))
    T = pd.DataFrame(rows)
    T.to_csv(TABLES / f"lit_turns_{a.model}_{a.dataset}{a.tag}.csv", index=False)
    for sc in T.scope.unique():
        print(f"\n--- {sc} ---")
        print(T[T.scope == sc].pivot_table(index="round", columns=["metric", "cond"], values="est").round(3).to_string())
    print(f"[done] outputs/tables/lit_turns_{a.model}_{a.dataset}{a.tag}.csv")


if __name__ == "__main__":
    main()
