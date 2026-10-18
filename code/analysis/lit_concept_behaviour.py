"""Does a concept's internal state, read before the agent writes, predict what it is about to do?

This replaces an earlier version of this analysis whose null was isotropic random directions. That null is invalid here:
the reading-site residuals have effective rank ~25-61 of 4,096, so a random direction lands inside the occupied subspace
and reads its dominant structure, and because effective rank grows with depth the baseline drifted with depth for purely
geometric reasons.

The null used here is the set of **permuted-label refits** produced alongside the real vectors: the same stories, the
same pooling, the same within-topic centring, with concept labels shuffled within topic. Each permutation yields a full
set of concept vectors that carry no concept information but have the same geometry and the same relationship to the
residual stream. Projecting debate states onto them gives the AUROC distribution to beat.

Outcome: does this agent change its answer this round (debate, and the matched no-communication condition).

Outputs: outputs/tables/lit_concept_behaviour_<tag>_<M>.csv
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import *  # noqa

CORE = {r["id"]: r for r in load_core()}
RAW_LIT = ROOT / "outputs/raw_lit"


def auroc_cols(y, X):
    """AUROC of every column of X against boolean y, vectorised over columns (average ranks for ties).

    The permutation null needs 20 x 171 columns scored per layer per condition; ranking column-by-column in python is
    ~1e6 calls and does not finish. scipy.stats.rankdata with axis=0 does the whole matrix at once.
    """
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.full(X.shape[1], np.nan)
    R = rankdata(X, axis=0)
    return (R[y].sum(0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def fam_adjusted(df, X, label="switch"):
    """AUROC within each task family, pooled by number of positives."""
    num = np.zeros(X.shape[1])
    den = 0.0
    y_all = df[label].to_numpy(bool)
    for _, idx in df.groupby("family").indices.items():
        y = y_all[idx]
        if y.all() or (~y).all() or len(idx) < 20:
            continue
        a = auroc_cols(y, X[idx])
        w = int(y.sum())
        num += np.nan_to_num(a, nan=0.5) * w
        den += w
    return num / den if den else np.full(X.shape[1], np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="lit1200")
    ap.add_argument("--tag", default="emotion_stories")
    a = ap.parse_args()
    cd = ACTS / a.model / a.tag
    meta = json.load(open(cd / "meta.json"))
    concepts = meta["concepts"]
    V = np.load(cd / "vectors.npz")
    VN = np.load(cd / "null_vectors.npz")

    rd = ACTS / a.model / f"lit_replay_{a.dataset}"
    rmeta = json.load(open(rd / "meta.json"))
    grid = rmeta["grid"]
    si = rmeta["sites"].index("reading")
    I = pd.DataFrame(read_jsonl(rd / "index.jsonl"))
    A = np.load(rd / "acts.npy", mmap_mode="r")

    d = RAW_LIT / a.model / a.dataset
    ans = {}
    for f in d.glob("*.jsonl"):
        for x in read_jsonl(f):
            ans[(x["item_id"], x["cond"], x["agent"], x["round"], x.get("query"))] = x["answer"]
    rows = []
    for r in I.itertuples():
        if isinstance(r.query, str) or r.cond not in ("debate", "reflect") or r.round == 0:
            continue
        prev = ans.get((r.item_id, "r0" if r.round == 1 else r.cond, r.agent, r.round - 1, None))
        cur = ans.get((r.item_id, r.cond, r.agent, r.round, None))
        if prev is None or cur is None:
            continue
        rows.append(dict(row=r.row, cond=r.cond, round=r.round, switch=float(prev != cur),
                         family=family_of(CORE[r.item_id])))
    D = pd.DataFrame(rows)
    print(f"[behaviour] {a.tag} on {a.model}: {len(D)} responses, {int(D.switch.sum())} answer changes", flush=True)

    out = []
    for L in grid:
        if f"L{L}" not in V:
            continue
        gi = grid.index(L)
        Vr = V[f"L{L}"]
        Vn = VN[f"L{L}"]                       # [n_perm, n_concepts, d]
        for cond in ("debate", "reflect"):
            sub = D[D.cond == cond].reset_index(drop=True)
            if sub.switch.sum() < 25:
                continue
            H = np.asarray(A[sub.row.to_numpy(), si, gi], np.float32)
            H = (H - H.mean(0)) / (H.std(0) + 1e-6)
            real = fam_adjusted(sub, H @ Vr.T)
            null = np.concatenate([fam_adjusted(sub, H @ Vn[b].T) for b in range(Vn.shape[0])])
            dev_r, dev_n = np.abs(real - 0.5), np.abs(null - 0.5)
            thr = np.nanquantile(dev_n, 0.95)
            n_beat = int(np.nansum(dev_r > thr))
            for j, c in enumerate(concepts):
                out.append(dict(layer=L, cond=cond, concept=c, auroc=real[j],
                                pct_of_null=float(np.nanmean(dev_n < dev_r[j])),
                                beats_null_p95=bool(dev_r[j] > thr)))
            print(f"  L{L:2d} {cond:8s} real |AUROC-.5| median {np.nanmedian(dev_r):.4f}   "
                  f"permuted-label null median {np.nanmedian(dev_n):.4f} (95th {thr:.4f})   "
                  f"concepts above null: {n_beat}/{len(concepts)}  (chance {0.05*len(concepts):.0f})", flush=True)
    T = pd.DataFrame(out)
    T.to_csv(TABLES / f"lit_concept_behaviour_{a.tag}_{a.model}.csv", index=False)
    deb = T[T.cond == "debate"]
    print(f"\n[summary] debate: {int(deb.beats_null_p95.sum())} of {len(deb)} concept-layer cells beat the permuted "
          f"null at p95 (chance {0.05*len(deb):.0f})")
    print(f"[done] outputs/tables/lit_concept_behaviour_{a.tag}_{a.model}.csv")


if __name__ == "__main__":
    main()
