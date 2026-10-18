"""How many independent opinions a group of three debating agents is actually worth.

A majority vote corrects errors only when the voters fail independently. Debate makes the three agents agree far more
often, so the question for any system that votes over debating agents is how much independence is left after they have
talked.

For every question and turn the three agents' answers are scored right or wrong, and the correlation between agents
within the same question is estimated as a one-way intra-class correlation. Three voters correlated at rho carry the
error-correcting weight of

    n_eff = 3 / (1 + 2 * rho)

independent voters, which is 3 when they are independent and 1 when they always agree. The same quantity is computed
in the no-communication condition on the same questions and the same number of turns, so what is reported is what
talking costs, not what thinking longer costs.

Out: outputs/tables/lit_independence_<model>_<dataset>.csv
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

RAW_LIT = ROOT / "outputs/raw_lit"


def icc1(groups):
    """One-way ICC on equally sized groups: the share of total variance that is between questions rather than within.

    Groups are the three agents' correctness on one question. With a binary outcome this is the ordinary intra-class
    correlation, and it is exactly the pairwise correlation that the n_eff formula needs.
    """
    Y = np.asarray(groups, float)
    n, k = Y.shape
    if n < 2:
        return np.nan
    gm = Y.mean()
    msb = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1)
    msw = ((Y - Y.mean(1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    d = msb + (k - 1) * msw
    return float((msb - msw) / d) if d > 0 else np.nan


def boot(groups, fn, n=2000, seed=0):
    r = np.random.default_rng(seed)
    G = np.asarray(groups, float)
    out = [fn(G[r.integers(0, len(G), len(G))]) for _ in range(n)]
    out = np.asarray([x for x in out if np.isfinite(x)])
    if len(out) < 50:
        return np.nan, np.nan
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="lit1200")
    a = ap.parse_args()

    # Comma-separated halves pool into one full-benchmark run; a single name behaves exactly as before.
    parts = [x for x in a.dataset.split(",") if x]
    tag = a.dataset if len(parts) == 1 else "full2400"
    rows = []
    for f in sorted(f2 for p2 in parts for f2 in (RAW_LIT / a.model / p2).glob("*.jsonl")):
        for x in read_jsonl(f):
            if x.get("correct") is None:
                continue
            rows.append(dict(item_id=x["item_id"], cond=x["cond"], agent=int(x["agent"]),
                             round=int(x["round"]), correct=float(str(x["correct"]).lower() == "true"),
                             answer=x.get("answer")))
    D = pd.DataFrame(rows)
    if not len(D):
        print("[independence] no scored rows")
        return
    R = int(D["round"].max())
    # Questions where the three agents already disagreed before anyone spoke. Rho is inflated by questions being easy
    # or hard, and this subset removes both extremes: it is also exactly the set where a vote could have helped.
    r0 = D[D["round"] == 0]
    split = {i for i, g in r0.groupby("item_id") if g.answer.nunique() > 1}
    print(f"[independence] {a.model} {a.dataset}: {len(D)} scored answers, {R} turns; "
          f"{len(split)} of {r0.item_id.nunique()} questions start split", flush=True)

    out = []
    for k in range(0, R + 1):
        conds = ("r0",) if k == 0 else ("debate", "reflect")
        # Both conditions are scored on exactly the same questions. Unparseable answers are not evenly distributed
        # across the arms, and on Llama that left the two conditions with different item sets at the same turn.
        shared = None
        for cond in conds:
            w = D[(D["round"] == k) & (D.cond == cond)].pivot_table(index="item_id", columns="agent",
                                                                    values="correct").dropna()
            shared = set(w.index) if shared is None else (shared & set(w.index))
        for cond in conds:
          for scope in ("all", "split_at_start"):
            g = D[(D["round"] == k) & (D.cond == cond)]
            if scope == "split_at_start":
                g = g[g.item_id.isin(split)]
            if not len(g):
                continue
            W = g.pivot_table(index="item_id", columns="agent", values="correct").dropna()
            W = W.loc[sorted(shared & set(W.index))]
            if len(W) < 30 or W.shape[1] != 3:
                continue
            G = W.to_numpy()
            rho = icc1(G)
            rl, rh = boot(G, icc1)
            ne = 3 / (1 + 2 * rho) if np.isfinite(rho) and (1 + 2 * rho) != 0 else np.nan
            nl, nh = boot(G, lambda x: 3 / (1 + 2 * icc1(x)))
            out.append(dict(round=k, cond=cond, scope=scope, n_items=len(W), accuracy=float(G.mean()),
                            rho=rho, rho_lo=rl, rho_hi=rh, n_eff=ne, n_eff_lo=nl, n_eff_hi=nh))
            print(f"  turn {k} {cond:8s} {scope:14s} n={len(W):5d}  agents agree on being right/wrong rho={rho:.3f} "
                  f"[{rl:.3f},{rh:.3f}]   worth {ne:.2f} independent agents [{nh:.2f},{nl:.2f}]", flush=True)

    T = pd.DataFrame(out)
    T.to_csv(TABLES / f"lit_independence_{a.model}_{tag}.csv", index=False)
    print("\n  turn   independent agents, debate vs no communication")
    for k in range(1, R + 1):
        for scope in ("all", "split_at_start"):
            d = T[(T["round"] == k) & (T.cond == "debate") & (T.scope == scope)]
            s = T[(T["round"] == k) & (T.cond == "reflect") & (T.scope == scope)]
            if len(d) and len(s):
                print(f"   {k}  {scope:14s}  {d.n_eff.iloc[0]:.2f}  vs  {s.n_eff.iloc[0]:.2f}")
    print(f"\n[done] outputs/tables/lit_independence_{a.model}_{tag}.csv")


if __name__ == "__main__":
    main()
