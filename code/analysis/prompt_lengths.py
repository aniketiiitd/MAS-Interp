"""How long are the prompts, relative to Gemma's 512-token sliding window?

Claim 3 states that 35 of Gemma-4-E4B's 42 layers attend only within a 512-token window, "against a median
prompt here of 597" -- so for most items the peer's message sits outside the window in those layers and its
influence has to travel through the seven global ones. That 597 was computed by hand and lived only in the
ledger, the same defect as claim 7's AUROC: a number in a claim that no table backs.

This writes the distribution down for every model, so the architectural argument rests on a table.

Out: outputs/tables/prompt_lengths_<M>_<dataset>.csv
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="G4")
    ap.add_argument("--dataset", default="core")
    ap.add_argument("--window", type=int, default=512, help="sliding-window size to report coverage against")
    a = ap.parse_args()

    rd = ACTS / a.model / f"lit_replay_{a.dataset}"
    I = pd.DataFrame(read_jsonl(rd / "index.jsonl"))
    rows = []
    for cond, g in [("all_debate", I[(I.cond == "debate") & (I["round"] > 0)])] + list(I.groupby("cond")):
        n = g.n_tok.dropna().to_numpy()
        if len(n) < 30:
            continue
        rows.append(dict(model=a.model, dataset=a.dataset, cond=cond, n=len(n),
                         median=float(np.median(n)), p25=float(np.quantile(n, .25)),
                         p90=float(np.quantile(n, .90)), maximum=float(n.max()),
                         window=a.window, frac_over_window=float((n > a.window).mean())))
        r = rows[-1]
        print(f"  {cond:12s} n={r['n']:6d}  median {r['median']:7.1f}  p90 {r['p90']:7.1f}  "
              f"max {r['maximum']:7.0f}  over {a.window}: {100 * r['frac_over_window']:5.1f}%", flush=True)

    T = pd.DataFrame(rows)
    out = TABLES / f"prompt_lengths_{a.model}_{a.dataset}.csv"
    T.to_csv(out, index=False)
    print(f"[done] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
