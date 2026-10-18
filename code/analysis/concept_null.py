"""What does the permutation null score? The number that makes an AUROC interpretable.

A concept vector's out-of-fold AUROC only means something against the AUROC the same pipeline reaches when the labels
carry no information. Isotropic random directions cannot answer that: the states are effectively low rank, so a random
direction still reads their dominant structure. Instead the emotion (or moral) labels are permuted WITHIN topic and the
whole fit-and-score procedure is repeated, K-fold and all. Everything is preserved -- stimuli, pooling, within-topic
centring, geometry, fold structure, class sizes -- except the label mapping.

Runs on the cached pooled states, so it needs no GPU.

Outputs: outputs/tables/concept_null_<tag>_<M>.csv
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


def fit_vectors(H, lab, topic, concepts):
    Hc = H.copy()
    for t in np.unique(topic):
        m = topic == t
        Hc[m] -= Hc[m].mean(0, keepdims=True)
    V = np.stack([Hc[lab == c].mean(0) if (lab == c).sum() else np.zeros(H.shape[1], np.float32) for c in concepts])
    return (V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def oof_auroc(H, lab, topic, concepts, folds):
    OOF = np.full((len(H), len(concepts)), np.nan, np.float32)
    for f in folds:
        te = np.isin(topic, f)
        tr = ~te
        V = fit_vectors(H[tr], lab[tr], topic[tr], concepts)
        Ht = H[te].copy()
        for t in np.unique(topic[te]):
            m = topic[te] == t
            Ht[m] -= Ht[m].mean(0, keepdims=True)
        OOF[te] = Ht @ V.T
    out = np.full(len(concepts), np.nan)
    for j, c in enumerate(concepts):
        y = lab == c
        if y.all() or (~y).all():
            continue
        r = pd.Series(OOF[:, j]).rank().to_numpy()
        out[j] = (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--tag", default="emotion_stories")
    ap.add_argument("--stories", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--n_perm", type=int, default=20)
    a = ap.parse_args()
    d = ACTS / a.model / a.tag
    meta = json.load(open(d / "meta.json"))
    concepts = meta["concepts"]
    stories = a.stories or meta["stimuli"]
    label = a.label or meta.get("label_field", "emotion")
    df = pd.DataFrame([json.loads(l) for l in open(stories)]).rename(columns={label: "label"})
    lab, topic = df.label.to_numpy(), df.topic_idx.to_numpy()
    z = np.load(d / "pooled.npz")
    layers = sorted(int(k[1:]) for k in z.files)
    rng = np.random.default_rng(11)
    folds = np.array_split(rng.permutation(np.unique(topic)), meta.get("folds", 4))

    rows = []
    for L in layers:
        H = z[f"L{L}"]
        real = oof_auroc(H, lab, topic, concepts, folds)
        nulls = []
        for b in range(a.n_perm):
            r2 = np.random.default_rng(5000 + b)
            perm = lab.copy()
            for t in np.unique(topic):
                m = topic == t
                perm[m] = r2.permutation(perm[m])
            nulls.append(oof_auroc(H, perm, topic, concepts, folds))
        N = np.stack(nulls)
        thr = np.nanquantile(N, 0.95)
        for j, c in enumerate(concepts):
            rows.append(dict(layer=L, concept=c, auroc=real[j],
                             null_median=float(np.nanmedian(N[:, j])), null_p95=float(np.nanquantile(N[:, j], 0.95)),
                             beats_own_null=bool(real[j] > np.nanquantile(N[:, j], 0.95))))
        n_beat = int(np.nansum(real > np.nanquantile(N, 0.95, axis=0)))
        print(f"  L{L:2d}  real median {np.nanmedian(real):.3f}   permuted-label median {np.nanmedian(N):.3f} "
              f"(95th pct {thr:.3f})   concepts above their own null: {n_beat}/{len(concepts)}", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(TABLES / f"concept_null_{a.tag}_{a.model}.csv", index=False)
    best = T.loc[T.groupby("concept").auroc.idxmax()]
    print(f"\n[summary] {a.tag}: median best-layer AUROC {best.auroc.median():.3f}, "
          f"permuted-label median {T.null_median.median():.3f}")
    print(f"          {int(best.beats_own_null.sum())}/{len(best)} concepts beat their own permuted null at the best layer")
    print(f"[done] outputs/tables/concept_null_{a.tag}_{a.model}.csv")


if __name__ == "__main__":
    main()
