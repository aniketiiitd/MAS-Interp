"""Does the peer's answer and the peer's reasoning add up to the peer's whole message?

The carrier experiment measures, on the same episode, how far the reader moves when it is given

  full             the peer's message as written, answer and reasoning together
  answer_only      the peer's answer with the reasoning removed
  reasoning_only   the peer's reasoning with the answer removed

If the two parts are independent channels, then answer_only + reasoning_only should equal full on the same episode.
That is a paired quantity and has to be tested as one: comparing the three published point estimates cannot tell an
additive decomposition from two parts that happen to average out.

Reported per model and per scope: the share of the whole message each part carries, and the residual
(answer_only + reasoning_only - full) with an item-clustered bootstrap interval. A residual whose interval covers zero
is an additive split.

Out: outputs/tables/lit_carrier_additivity_<model>.csv
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


def cluster_boot(df, fn, n=4000, seed=0):
    """Bootstrap resampling whole items, since the three agents on one item are not independent."""
    r = np.random.default_rng(seed)
    items = df.item_id.unique()
    idx = {k: g.index.to_numpy() for k, g in df.groupby("item_id")}
    out = []
    for _ in range(n):
        pick = r.choice(items, len(items), replace=True)
        rows = np.concatenate([idx[k] for k in pick])
        out.append(fn(df.loc[rows]))
    out = np.asarray([x for x in out if np.isfinite(x)])
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Q9,L31")
    ap.add_argument("--min_items", type=int, default=25, help="scopes with fewer items are not reported")
    a = ap.parse_args()

    for m in a.models.split(","):
        E = pd.read_csv(TABLES / f"lit_carrier_{m}_episodes.csv").reset_index(drop=True)
        E["residual"] = E.answer_only + E.reasoning_only - E["full"]
        rows = []
        scopes = [("all", E)] + [(d, g) for d, g in E.groupby("domain")]
        for name, g in scopes:
            g = g.reset_index(drop=True)
            if g.item_id.nunique() < a.min_items:
                continue
            f, an, rs, res = (g[c].mean() for c in ("full", "answer_only", "reasoning_only", "residual"))
            rl, rh = cluster_boot(g, lambda d: d.residual.mean())
            # The reasoning share is a ratio of two noisy means, so its interval is bootstrapped whole rather than
            # propagated from the two parts.
            sl, sh = cluster_boot(g, lambda d: d.reasoning_only.mean() / (d.answer_only.mean() + d.reasoning_only.mean()))
            share = rs / (an + rs)
            # A share is a ratio, and a ratio whose denominator straddles zero is not a number worth printing however
            # tight it looks. On Llama's values questions the whole effect is negative and the "reasoning share" came
            # out at 66%; on safety the interval was [-227%, 323%]. Suppress it rather than leave it to be misread.
            denom_lo, denom_hi = cluster_boot(g, lambda d: d.answer_only.mean() + d.reasoning_only.mean())
            if denom_lo * denom_hi <= 0:
                share, sl, sh = float("nan"), float("nan"), float("nan")
                print(f"[{m}] {name}: share suppressed, the two parts sum to an interval covering zero "
                      f"[{denom_lo:+.2f}, {denom_hi:+.2f}]", flush=True)
            rows.append(dict(model=m, scope=name, n_items=g.item_id.nunique(), n=len(g),
                             full=f, answer_only=an, reasoning_only=rs,
                             residual=res, res_lo=rl, res_hi=rh, additive=bool(rl <= 0 <= rh),
                             reasoning_share=share, share_lo=sl, share_hi=sh))
            print(f"[{m}] {name:24s} full {f:5.2f}   answer {an:5.2f}   reasoning {rs:5.2f}   "
                  f"residual {res:+5.2f} [{rl:+.2f},{rh:+.2f}] {'additive' if rl <= 0 <= rh else 'NOT additive'}   "
                  f"reasoning share {100 * share:.0f}% [{100 * sl:.0f},{100 * sh:.0f}]", flush=True)
        pd.DataFrame(rows).to_csv(TABLES / f"lit_carrier_additivity_{m}.csv", index=False)
        print(f"[done] outputs/tables/lit_carrier_additivity_{m}.csv\n")


if __name__ == "__main__":
    main()
