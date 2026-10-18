"""When a group of agents stops changing its answers, and how much of a long debate is therefore spent.

Multi-agent systems set a number of rounds as a hyperparameter, usually upward. This measures the turn after which
nothing more happens: for every question, the last turn at which any of the three agents changed its answer. The same
quantity is computed in the matched no-communication condition, so what is reported is what talking settles rather than
what thinking longer settles.

Reported per turn: the share of groups that have stopped changing by that turn, with an item bootstrap. A group that
never changes after turn 0 is frozen at turn 0.

Out: outputs/tables/lit_freeze_<model>_<dataset>.csv
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import *  # noqa

RAW_LIT = ROOT / "outputs/raw_lit"


def last_change(seq):
    """The 1-indexed turn of the last change in a sequence of per-turn answer triples; 0 if it never changes."""
    lc = 0
    for i in range(1, len(seq)):
        if seq[i] != seq[i - 1]:
            lc = i + 1
    return lc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="release_600_t12")
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args()

    parts = [x for x in a.dataset.split(",") if x]
    tag = a.dataset if len(parts) == 1 else "full2400"
    dirs = [RAW_LIT / a.model / p2 for p2 in parts]
    rows = []
    for cond in ("debate", "reflect"):
        ans = defaultdict(dict)
        for f in sorted(f2 for d in dirs for f2 in d.glob(f"{cond}_r*.jsonl")):
            for x in read_jsonl(f):
                ans[x["item_id"]][(int(x["round"]), int(x["agent"]))] = x["answer"]
        if not ans:
            continue
        R = max(r for v in ans.values() for r, _ in v)
        last = []
        for it, v in ans.items():
            seq = [tuple(v.get((r, ag)) for ag in range(3)) for r in range(1, R + 1)]
            last.append(last_change(seq))
        last = np.asarray(last)
        rng = np.random.default_rng(0)
        idx = [rng.integers(0, len(last), len(last)) for _ in range(a.n_boot)]
        for t in range(1, R + 1):
            frozen = float((last <= t).mean())
            b = np.array([float((last[i] <= t).mean()) for i in idx])
            rows.append(dict(cond=cond, turn=t, frozen=frozen,
                             lower=float(np.quantile(b, 0.025)), upper=float(np.quantile(b, 0.975)),
                             n_items=len(last), turns_run=R))
        mb = np.array([last[i].mean() for i in idx])
        rows.append(dict(cond=cond, turn=-1, frozen=float(last.mean()),
                         lower=float(np.quantile(mb, 0.025)), upper=float(np.quantile(mb, 0.975)),
                         n_items=len(last), turns_run=R))
        print(f"[freeze] {cond:8s} n={len(last)} over {R} turns; mean turn of the last change "
              f"{last.mean():.2f} [{np.quantile(mb, 0.025):.2f}, {np.quantile(mb, 0.975):.2f}]", flush=True)

    T = pd.DataFrame(rows)
    T.to_csv(TABLES / f"lit_freeze_{a.model}_{tag}.csv", index=False)
    print("\n  turn   frozen: debate / no communication")
    for t in sorted(T[T.turn > 0].turn.unique()):
        dd = T[(T.cond == "debate") & (T.turn == t)]
        rr = T[(T.cond == "reflect") & (T.turn == t)]
        if len(dd) and len(rr):
            print(f"   {t:2d}     {100 * dd.frozen.iloc[0]:5.1f}%  /  {100 * rr.frozen.iloc[0]:5.1f}%")
    print(f"\n[done] outputs/tables/lit_freeze_{a.model}_{tag}.csv")


if __name__ == "__main__":
    main()
