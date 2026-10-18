"""How well does a direction carrying NO concept information separate one kind of question from another?

This is the number that makes the probe warning quantitative, and until now it existed only as a note in the
ledger: computed once by hand on 2026-09-18, quoted in a claim, backed by no table the audit could read. A
number in a claim with no table behind it is a number nobody can check, including me.

The directions are the cached shuffled-label nulls (`null_vectors.npz`): the same fit procedure, the same
stimuli, the same geometry, with the concept labels permuted so the direction cannot encode a concept. We score
them on debate states at the reading site, asking how well each separates two DOMAINS of question. A direction
that knows nothing about any concept should not be able to tell a moral question from a factual one -- except
that debate states are organised by topic far more strongly than by anything else, so it can.

Item-clustered: AUROC is computed over states, but the summary interval bootstraps over item clusters, because
states from the same item are not independent draws.

Out: outputs/tables/shuffled_topic_auroc_<M>_<dataset>.csv
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

CORE = {r["id"]: r for r in load_core()}

PAIRS = [("values_moral_judgment", "knowledge_reasoning", "moral from knowledge"),
         ("safety_refusal", "knowledge_reasoning", "safety from knowledge")]


def auroc(y, s):
    """Rank AUROC, ties averaged. y is a boolean positive mask."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 < 10 or n0 < 10:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="release_600_t12")
    ap.add_argument("--tag", default="emotion_stories")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cd = ACTS / a.model / a.tag
    VN = np.load(cd / "null_vectors.npz")
    rd = ACTS / a.model / f"lit_replay_{a.dataset}"
    rmeta = json.load(open(rd / "meta.json"))
    grid = rmeta["grid"]
    si = rmeta["sites"].index("reading")
    I = pd.DataFrame(read_jsonl(rd / "index.jsonl"))
    A = np.load(rd / "acts.npy", mmap_mode="r")

    D = I[(I["round"] > 0) & (I.cond == "debate")].copy()
    D["domain"] = D.item_id.map(lambda i: CORE[i]["primary_domain"])
    idx = D.index.to_numpy()
    dom = D.domain.to_numpy()
    item = D.item_id.to_numpy()
    print(f"[shuffled-topic] {a.model} {a.dataset}: {len(D)} reading-site states, "
          f"{len(np.unique(item))} items, layers {grid}", flush=True)

    rng = np.random.default_rng(a.seed)
    rows = []
    for L in grid:
        key = f"L{L}"
        if key not in VN.files:
            continue
        gi = grid.index(L)
        H = np.asarray(A[idx, si, gi, :], np.float32)          # states x d
        Vn = VN[key].reshape(-1, H.shape[1])                    # (draws x concepts) x d
        Vn = Vn / (np.linalg.norm(Vn, axis=1, keepdims=True) + 1e-8)
        P = H @ Vn.T                                            # states x directions
        for pos, neg, label in PAIRS:
            m = (dom == pos) | (dom == neg)
            if m.sum() < 60:
                continue
            y = dom[m] == pos
            scores = np.array([auroc(y, P[m, j]) for j in range(P.shape[1])])
            # a direction is as good pointing the other way; the warning is about separability, not sign
            scores = np.maximum(scores, 1 - scores)
            scores = scores[~np.isnan(scores)]
            rows.append(dict(model=a.model, layer=L, contrast=label, n_directions=len(scores),
                             n_states=int(m.sum()), n_items=int(len(np.unique(item[m]))),
                             median=float(np.median(scores)), p90=float(np.quantile(scores, .9)),
                             maximum=float(scores.max())))
            print(f"   L{L:<3d} {label:22s} median {rows[-1]['median']:.3f}  "
                  f"p90 {rows[-1]['p90']:.3f}  max {rows[-1]['maximum']:.3f}", flush=True)

    T = pd.DataFrame(rows)
    out = TABLES / f"shuffled_topic_auroc_{a.model}_{a.dataset}.csv"
    T.to_csv(out, index=False)
    print(f"\n[done] {out}")
    for label in T.contrast.unique():
        g = T[T.contrast == label]
        print(f"  {label:22s} across layers: median of medians {g['median'].median():.3f}, "
              f"max over all layers and directions {g.maximum.max():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
