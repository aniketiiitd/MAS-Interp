"""The answer-before-argument gap, paired within item.

The headline comparison averages over episodes in each arm separately, so an item that contributes many changed
episodes to one arm and few to the other pulls the average around. Pairing within item removes that: each item
contributes one difference, computed from its own episodes in both arms, and items that appear in only one arm are
dropped rather than averaged in.

This is a robustness check on the same estimand, not a different claim. It is written as a script because the number
was previously produced by an ad-hoc analysis with no table behind it, which meant nothing verified it afterwards.

Out: outputs/tables/lit_commit_paired_<tag>.csv
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


def argmax_is_final(r):
    lp = r.lp_letters
    if not isinstance(lp, dict) or not lp or r.answer is None:
        return np.nan
    return float(max(lp, key=lp.get) == r.answer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="lit1200")
    ap.add_argument("--n_boot", type=int, default=4000)
    a = ap.parse_args()
    parts = [x for x in a.dataset.split(",") if x]
    tag = f"{a.model}_" + (parts[0] if len(parts) == 1 else "full2400")

    D = pd.concat([pd.DataFrame(read_jsonl(ACTS / a.model / f"lit_commit_{p}" / "rows.jsonl")) for p in parts],
                  ignore_index=True)
    D = D[D.frame == 0.0].copy()
    D["hit"] = D.apply(argmax_is_final, axis=1)
    C = D[(D.changed == True) & D.hit.notna()]          # noqa: E712

    per = C.groupby(["item_id", "cond"]).hit.mean().unstack("cond")
    both = per.dropna(subset=["debate", "reflect"])
    diff = (both["debate"] - both["reflect"]).to_numpy()
    rng = np.random.default_rng(0)
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(a.n_boot)])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    # a sign test as well, since the mean of a per-item difference is sensitive to items with few episodes
    pos, neg = int((diff > 0).sum()), int((diff < 0).sum())

    rows = [dict(model=a.model, scope="item_paired", n_items=len(diff),
                 est=float(diff.mean()), lower=float(lo), upper=float(hi),
                 items_favouring_debate=pos, items_favouring_solo=neg),
            dict(model=a.model, scope="unpaired", n_items=int(per.shape[0]),
                 est=float(C[C.cond == "debate"].hit.mean() - C[C.cond == "reflect"].hit.mean()),
                 lower=np.nan, upper=np.nan, items_favouring_debate=np.nan, items_favouring_solo=np.nan)]
    T = pd.DataFrame(rows)
    T.to_csv(TABLES / f"lit_commit_paired_{tag}.csv", index=False)
    print(f"[paired] {a.model} {a.dataset}: {len(diff)} items appear in both arms "
          f"(of {per.shape[0]} with any episode)")
    print(f"  item-paired gap {100 * diff.mean():+.1f} pts [{100 * lo:+.1f}, {100 * hi:+.1f}]")
    print(f"  items favouring debate {pos}, favouring no communication {neg}")
    print(f"  unpaired gap for comparison {100 * rows[1]['est']:+.1f} pts")
    print(f"[done] outputs/tables/lit_commit_paired_{tag}.csv")


if __name__ == "__main__":
    main()
