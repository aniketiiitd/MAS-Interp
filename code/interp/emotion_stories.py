"""Emotion vectors refit on topic-crossed stories, with a permutation null.

Two things are wrong with the previous fit and are corrected here.

**Stimuli.** The old set had 8 stories per concept with no topic variable, so the only way to remove story content was to
subtract the grand mean over all emotions -- which also subtracts whatever content happens to correlate with emotion.
The new set (6,510 stories, 171 emotions x ~50 topics) crosses emotion with topic, so content can be removed exactly:

    v_e = mean over stories of e of ( h_story - mean over emotions of h for that SAME topic )

Every emotion appears on many topics and every topic carries many emotions, so the within-topic centring cancels topic
content and leaves the emotion contrast. Validation holds out whole TOPICS, not stories, so a vector has to generalise to
content it never saw.

**Null.** Isotropic random directions are not a null here. The residual states we score have an effective rank of roughly
25-60 out of 4096, so a "random" 4096-dimensional direction still lands substantially inside the subspace the data
occupies and reads its dominant structure; worse, the effective rank grows with depth, so the random baseline drifts with
depth for reasons that have nothing to do with emotion. The null used here instead **refits the entire pipeline with the
emotion labels permuted within topic**. That preserves the stimuli, the pooling, the centring, the geometry and the
effective rank exactly, and destroys only the emotion-label mapping -- which is the thing under test.

Outputs: ACTS/<M>/emotion_stories/{vectors.npz, null_vectors.npz, index.csv, meta.json}
         outputs/tables/emotion_stories_<M>_validation.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "interp"))
from common import *  # noqa
from cpt import Receiver
from run_cpt import layer_grid

STORIES = ROOT / "data/emotion/stories.jsonl"
SKIP = 25          # story tokens dropped before pooling, so the vector is not a prompt-onset artefact
N_PERM = 20        # permuted-label refits; each yields a full set of 171 null vectors


def pool_states(R, texts, layers, batch=8, skip=SKIP):
    """Mean-pooled residual per story at each layer, [n_layers, n_stories, d]."""
    out = {L: [] for L in layers}
    t0 = time.time()
    for b0 in range(0, len(texts), batch):
        chunk = texts[b0: b0 + batch]
        ids = [R.tok(t, add_special_tokens=False)["input_ids"][:1024] for t in chunk]
        store = {}
        pos = [list(range(min(skip, max(1, len(x) - 1)), len(x))) for x in ids]
        for L in layers:
            R.add_capture_mean(L, pos, store)
        R.forward(ids)
        R.clear()
        for L in layers:
            out[L].extend([np.asarray(x, np.float32) for x in store[L]])
        if (b0 // batch) % 50 == 0:
            el = time.time() - t0
            print(f"[pool] {b0+len(chunk)}/{len(texts)}  {el/60:.1f} min", flush=True)
    return {L: np.stack(out[L]) for L in layers}


def fit_vectors(H, emo, topic, concepts):
    """Within-topic centred emotion means. H [n, d] -> V [n_concepts, d]."""
    Hc = H.copy()
    for t in np.unique(topic):
        m = topic == t
        Hc[m] -= Hc[m].mean(0, keepdims=True)
    V = np.stack([Hc[emo == c].mean(0) if (emo == c).sum() else np.zeros(H.shape[1], np.float32) for c in concepts])
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    return V.astype(np.float32)


def auroc(y, s):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return np.nan
    r = pd.Series(np.asarray(s, float)).rank().to_numpy()
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n_perm", type=int, default=N_PERM)
    ap.add_argument("--stories", default=STORIES, help="jsonl with {label, topic_idx, text}")
    ap.add_argument("--label", default="emotion", help="field holding the concept label")
    ap.add_argument("--tag", default="emotion_stories", help="output directory and table suffix")
    ap.add_argument("--folds", type=int, default=4, help="topic folds for out-of-fold validation")
    ap.add_argument("--repool", action="store_true", help="ignore the cached pooled states and redo the forward passes")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.stories)]
    df = pd.DataFrame(rows).rename(columns={a.label: "label"})
    concepts = sorted(df.label.unique())
    print(f"[stories] {len(df)} stories, {len(concepts)} {a.label} concepts, {df.topic_idx.nunique()} topics", flush=True)

    out = ACTS / a.model / a.tag
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "pooled.npz"
    if cache.exists() and not a.repool:
        z = np.load(cache)
        layers = [int(k[1:]) for k in z.files]
        S = {int(k[1:]): z[k] for k in z.files}
        print(f"[cache] pooled states loaded from {cache} (layers {layers}); no GPU needed", flush=True)
    else:
        R = Receiver(a.model)
        layers = layer_grid(R.n_layers)
        print(f"[stage] {a.model}: layers {layers}", flush=True)
        S = pool_states(R, df.text.tolist(), layers, batch=a.batch)
        np.savez_compressed(cache, **{f"L{L}": S[L] for L in layers})
        print(f"[cache] pooled states written to {cache}", flush=True)

    emo = df.label.to_numpy()
    topic = df.topic_idx.to_numpy()
    # K-fold over TOPICS. A single random topic holdout left 13 of 171 concepts with no held-out story at all, so those
    # concepts had no validation score; cross-validating over topic folds gives every story an out-of-fold score and
    # therefore every concept a held-out AUROC, while still never scoring a story whose topic was seen in fitting.
    tv = np.unique(topic)
    rng = np.random.default_rng(11)
    perm_t = rng.permutation(tv)
    folds = np.array_split(perm_t, a.folds)
    print(f"[split] {a.folds}-fold over {len(tv)} topics ({[len(f) for f in folds]} topics per fold)", flush=True)

    V_all, V_null, val = {}, {}, []
    for L in layers:
        H = S[L]
        OOF = np.full((len(H), len(concepts)), np.nan, np.float32)
        for f in folds:
            te = np.isin(topic, f)
            tr = ~te
            Vf = fit_vectors(H[tr], emo[tr], topic[tr], concepts)
            Ht = H[te].copy()
            for t in np.unique(topic[te]):
                m = topic[te] == t
                Ht[m] -= Ht[m].mean(0, keepdims=True)
            OOF[te] = Ht @ Vf.T
        for j, c in enumerate(concepts):
            val.append(dict(layer=L, concept=c, n_stories=int((emo == c).sum()),
                            auroc_oof_topic=auroc(emo == c, OOF[:, j])))
        # final vectors use every story; the folds above are only for validation
        V_all[f"L{L}"] = fit_vectors(H, emo, topic, concepts)
        # permutation null: identical pipeline, labels shuffled WITHIN topic, fit on the same full data
        nulls = []
        for b in range(a.n_perm):
            r2 = np.random.default_rng(1000 + b)
            perm = emo.copy()
            for t in np.unique(topic):
                m = topic == t
                perm[m] = r2.permutation(perm[m])
            nulls.append(fit_vectors(H, perm, topic, concepts))
        V_null[f"L{L}"] = np.stack(nulls).astype(np.float32)
        med = np.nanmedian([v["auroc_oof_topic"] for v in val if v["layer"] == L])
        print(f"[fit] L{L:2d}  median out-of-fold AUROC {med:.3f}  ({a.n_perm} permuted refits stored)", flush=True)

    np.savez_compressed(out / "vectors.npz", **V_all)
    np.savez_compressed(out / "null_vectors.npz", **V_null)
    df[["label", "topic", "topic_idx", "n_chars"]].to_csv(out / "index.csv", index=False)
    json.dump(dict(model=a.model, sha=R.sha, concepts=concepts, layers=layers, skip=SKIP, label_field=a.label,
                   n_stories=len(df), n_topics=int(df.topic_idx.nunique()), 
                   n_perm=a.n_perm, stimuli=a.stories, folds=a.folds,
                   centring="within-topic", validation="out-of-fold over topic folds",
                   null="labels permuted within topic, full refit on all data"),
              open(out / "meta.json", "w"), indent=1)
    T = pd.DataFrame(val)
    T.to_csv(TABLES / f"{a.tag}_{a.model}_validation.csv", index=False)
    ok = T.dropna(subset=["auroc_oof_topic"])
    n_missing = T.concept.nunique() - ok.concept.nunique()
    if n_missing:
        print(f"[warn] {n_missing} concepts have no out-of-fold score at any layer", flush=True)
    best = ok.loc[ok.groupby("concept").auroc_oof_topic.idxmax()]
    print(f"\n[done] {len(best)}/{T.concept.nunique()} concepts scored; median best-layer out-of-fold AUROC "
          f"{best.auroc_oof_topic.median():.3f}; {(best.auroc_oof_topic >= 0.75).sum()}/{len(best)} at or above 0.75")


if __name__ == "__main__":
    main()
