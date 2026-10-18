"""Is a source's stated credential legible inside the model, and where?

Contribution two rests on a contrast: the credential is readable in the state yet barely moves the decision. The
readable half was quoted at 0.85 AUROC from a one-off analysis that left no table, so nothing verified it afterwards.
This computes it properly and writes one.

The probe separates a described status (a tenured professor against an ordinary member of the public) from the state,
fit on one half of the items and scored on the other, so the number is held out rather than in-sample. It is reported at
every captured position, because the value depends entirely on where you read: at the source description itself the
label is lexically present and any probe scores near one, which says nothing. The decision point is the only position
where the question is interesting, and it is the one the claim should quote.

Both estimators are reported. A difference-of-means direction is the conservative one and a logistic probe the
permissive one; quoting the conservative number is the honest choice for a claim that the credential *is* legible.

Out: outputs/tables/lit_credential_probe_<model>.csv
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


def auroc(y, s):
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logistic_dir(X, y, iters=300, lr=0.5):
    """Small logistic fit, standardised inputs, no package dependency beyond numpy."""
    w = np.zeros(X.shape[1], np.float32)
    b = 0.0
    yy = y.astype(np.float32)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        g = X.T @ (p - yy) / len(yy)
        w -= lr * (g + 1e-3 * w)
        b -= lr * float((p - yy).mean())
    return w, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--run", default="cptlit_core_n600")
    a = ap.parse_args()
    d = ACTS / a.model / a.run
    meta = json.load(open(d / "meta.json"))
    grid, positions = meta["grid"], meta["positions"]
    W = pd.DataFrame(read_jsonl(d / "worlds.jsonl"))
    A = np.load(d / "acts.npy", mmap_mode="r")

    # only worlds that actually describe a source carry a status to read
    W = W.reset_index(drop=True)
    W["row"] = W.index
    S = W[W.status.isin(["prof", "ord"])].copy()
    if not len(S):
        S = W[W.status.astype(str).str.lower().isin(["professor", "public", "prof", "ord"])].copy()
    if not len(S):
        raise SystemExit(f"[credential] no rows with a described status; values seen: {sorted(W.status.unique())[:8]}")
    y = (S.status.astype(str).str.startswith("prof")).to_numpy()
    items = S.item_id.to_numpy()
    uniq = pd.unique(items)
    # held out by ITEM, not by row: the same item appears in many worlds and splitting rows would leak it across halves
    half = set(uniq[: len(uniq) // 2])
    tr = np.array([i in half for i in items])
    print(f"[credential] {a.model}: {len(S)} described-source worlds, {len(uniq)} items, "
          f"{int(y.sum())} professor / {int((~y).sum())} ordinary", flush=True)

    rows = []
    for pi, pos in enumerate(positions):
        for gi, L in enumerate(grid):
            X = np.asarray(A[S.row.to_numpy(), pi, gi], np.float32)
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            Xs = (X - mu) / sd
            dm = Xs[tr & y].mean(0) - Xs[tr & ~y].mean(0)
            a_dm = auroc(y[~tr], Xs[~tr] @ dm)
            w, b = logistic_dir(Xs[tr], y[tr])
            a_lr = auroc(y[~tr], Xs[~tr] @ w + b)
            rows.append(dict(model=a.model, position=pos, layer=L, auroc_diff_of_means=a_dm, auroc_logistic=a_lr,
                             n_train=int(tr.sum()), n_test=int((~tr).sum())))
    T = pd.DataFrame(rows)
    T.to_csv(TABLES / f"lit_credential_probe_{a.model}.csv", index=False)
    print("\nheld-out AUROC by position, best layer for each:")
    for pos, g in T.groupby("position", sort=False):
        b = g.loc[g.auroc_diff_of_means.idxmax()]
        print(f"  {pos:10s} difference of means {b.auroc_diff_of_means:.3f}   logistic {b.auroc_logistic:.3f}   "
              f"(layer {int(b.layer)})")
    print(f"\n[done] outputs/tables/lit_credential_probe_{a.model}.csv")


if __name__ == "__main__":
    main()
