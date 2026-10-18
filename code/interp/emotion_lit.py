"""Emotion / affect read-out directions for the debate models, following the method of Anthropic's
"Emotion Concepts and their Function in a Large Language Model" (transformer-circuits.pub/2026/emotions, arXiv 2604.07729):

  elicitation  stories in which a character is in the emotional state (our data/concepts/stories.jsonl: 171 emotion concepts x 8
               stories, each with a behaviour-matched neutral rewrite)
  pooling      residual stream mean-pooled over the story tokens (Anthropic pool from the 50th token of a long story; our stories
               are shorter, so we drop the first SKIP tokens of the story span and pool the rest)
  vector       v_e = mean(activations of e) - mean(activations over all emotions)          [emotion-mean centring]
  cleaning     the top principal components of a neutral corpus (the behaviour-matched neutral rewrites) explaining NEUTRAL_VAR of
               its variance are projected out of every vector and of every activation before scoring

Validation is deliberately stricter than our earlier atlas, which failed its emotion checks:
  T1 held-out stories   one-vs-rest AUROC on stories the vector was not fitted on, after removing the shared affect axis
  T1b neutral pairs     the same story scored above its behaviour-matched neutral rewrite (paired)
  T1c lexical diagnostic AUROC on "<name> felt <emotion> at that moment." sentences: a vector that only fires there is a word detector
  T2 second route       for the 22 concepts with an intensity bank (different generator, event + 5 intensity levels):
                        AUROC at the top level vs other concepts, and Spearman of the projection against intensity
  T3 causal             steering with v_e (strength STEER x residual norm, Anthropic use 0.5) shifts the model's own forced choice
                        between two feeling words toward e, against a norm-matched random direction control
An axis is usable only if it passes T1, T1b and, where available, T2 and T3. Everything else is reported as failed.

Run: python interp/emotion_lit.py --model Q9            (extract -> fit -> validate -> causal, writes tables + vectors)
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import scipy.stats as st
import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "interp"))
from common import *  # noqa
from cpt import Receiver
from run_cpt import layer_grid

SKIP = 25          # story tokens dropped before pooling (Anthropic: pool from the 50th token of a longer story)
NEUTRAL_VAR = 0.5  # variance of the neutral corpus removed by projecting out its top principal components
STEER = 0.5        # steering strength as a fraction of the residual norm at the steered layer
FIT_IDX = (0, 1, 2, 3)   # story indices used for fitting; 4-7 are held out
CONC = "data/concepts"


def pooled_forward(R, texts, spans, grid, rows=16):
    """Mean-pooled residual over a character span of each text, at every grid layer. Returns [n, len(grid), d] float32."""
    out = np.zeros((len(texts), len(grid), R.model.config.get_text_config().hidden_size), np.float32)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    for b0 in range(0, len(order), rows):
        chunk = order[b0: b0 + rows]
        ids, keep = [], []
        for i in chunk:
            enc = R.tok(texts[i], add_special_tokens=False, return_offsets_mapping=True)
            t = enc["input_ids"]
            lo, hi = spans[i]
            idx = [j for j, (s, e) in enumerate(enc["offset_mapping"]) if s >= lo and e <= hi]
            if len(idx) > SKIP + 8:
                idx = idx[SKIP:]
            ids.append(t)
            keep.append(idx or list(range(len(t))))
        T = max(len(x) for x in ids)
        pad = R.tok.pad_token_id if R.tok.pad_token_id is not None else 0
        inp = torch.full((len(ids), T), pad, dtype=torch.long)
        att = torch.zeros((len(ids), T), dtype=torch.long)
        for r, x in enumerate(ids):
            inp[r, : len(x)] = torch.tensor(x)
            att[r, : len(x)] = 1
        store = {}

        def mk(li):
            def hook(mod, a, o):
                h = o[0] if isinstance(o, tuple) else o
                store[li] = torch.stack([h[r, torch.tensor(keep[r], device=h.device)].float().mean(0) for r in range(len(ids))]).cpu()
            return hook
        hs = [R.layers[li].register_forward_hook(mk(li)) for li in grid]
        with torch.inference_mode():
            R.model.model(input_ids=inp.to(R.device), attention_mask=att.to(R.device), use_cache=False)
        for h in hs:
            h.remove()
        for gi, li in enumerate(grid):
            for r, i in enumerate(chunk):
                out[i, gi] = store[li][r].numpy()
    return out


def user_turn(R, body):
    """Wrap a text as a user turn (peer messages appear the same way in the debates) and return (text, span of the body)."""
    msgs = [{"role": "user", "content": body}]
    s = R.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    lo = s.rfind(body)
    return s, (lo, lo + len(body))


def auroc(y, s):
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    if y.all() or (~y).all():
        return np.nan
    r = st.rankdata(s)
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--n_causal", type=int, default=18, help="emotions tested with steering")
    a = ap.parse_args()
    t0 = time.time()
    R = Receiver(a.model)
    grid = layer_grid(R.n_layers)
    out = ACTS / a.model / "emotion_lit"
    out.mkdir(parents=True, exist_ok=True)

    stories = [json.loads(l) for l in open(ROOT / CONC / "stories.jsonl") if json.loads(l)["kind"] == "emotion"]
    grads = [json.loads(l) for l in open(ROOT / CONC / "gradients.jsonl")]
    lex = [json.loads(l) for l in open(ROOT / CONC / "lexical.jsonl")]
    concepts = sorted({s["concept"] for s in stories})
    group = {s["concept"]: s["group"] for s in stories}
    print(f"[stage] {len(concepts)} emotion concepts, {len(stories)} stories, {len(grads)} intensity items, {len(lex)} lexical",
          flush=True)

    texts, spans, meta = [], [], []
    for s in stories:
        for field, kind in (("story", "story"), ("neutral", "neutral")):
            t, sp = user_turn(R, s[field])
            texts.append(t)
            spans.append(sp)
            meta.append(dict(kind=kind, concept=s["concept"], story_idx=s["story_idx"], group=s["group"]))
    for g in grads:
        t, sp = user_turn(R, g["story"])
        texts.append(t)
        spans.append(sp)
        meta.append(dict(kind="gradient", concept=g["concept"], level=g["level"], event_idx=g["event_idx"], group=group.get(g["concept"], "other")))
    for x in lex:
        t, sp = user_turn(R, x["sentence"])
        texts.append(t)
        spans.append(sp)
        meta.append(dict(kind="lexical", concept=x["concept"], group=group.get(x["concept"], "other")))
    A = pooled_forward(R, texts, spans, grid, a.rows)
    M = pd.DataFrame(meta)
    np.save(out / "acts.npy", A)
    M.to_csv(out / "index.csv", index=False)
    print(f"[stage] pooled {len(texts)} texts in {(time.time()-t0)/60:.1f} min", flush=True)

    # ---- fit: emotion-mean centring + neutral-PC removal, per layer
    res, vectors = [], {}
    is_story = (M.kind == "story").to_numpy()
    is_neutral = (M.kind == "neutral").to_numpy()
    fit_m = is_story & M.story_idx.isin(FIT_IDX).to_numpy()
    held_m = is_story & ~M.story_idx.isin(FIT_IDX).to_numpy()
    for gi, li in enumerate(grid):
        X = A[:, gi]
        N = X[is_neutral]
        Nc = N - N.mean(0)
        U, S, Vt = np.linalg.svd(Nc, full_matrices=False)
        k = int(np.searchsorted(np.cumsum(S ** 2) / (S ** 2).sum(), NEUTRAL_VAR) + 1)
        P = Vt[:k]                                   # neutral subspace to remove
        clean = lambda Z: Z - (Z @ P.T) @ P
        Xc = clean(X)
        V = np.stack([Xc[fit_m & (M.concept == c).to_numpy()].mean(0) for c in concepts])
        V = V - V.mean(0)                            # emotion-mean centring
        shared = np.linalg.svd(V - V.mean(0), full_matrices=False)[2][0]   # shared affect axis
        Vs = V - np.outer(V @ shared, shared)
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        Vs = Vs / np.linalg.norm(Vs, axis=1, keepdims=True)
        vectors[li] = dict(V=V, Vs=Vs, P=P, k=k)
        Sc, Scs = Xc @ V.T, Xc @ Vs.T                # [n_texts, n_concepts]
        for ci, c in enumerate(concepts):
            cm = (M.concept == c).to_numpy()
            h = held_m & cm
            oth = held_m & ~cm
            t1 = auroc(np.r_[np.ones(h.sum()), np.zeros(oth.sum())], np.r_[Sc[h, ci], Sc[oth, ci]])
            t1s = auroc(np.r_[np.ones(h.sum()), np.zeros(oth.sum())], np.r_[Scs[h, ci], Scs[oth, ci]])
            pair = np.mean(Sc[is_story & cm, ci] > Sc[is_neutral & cm, ci]) if cm.any() else np.nan
            lx = (M.kind == "lexical").to_numpy()
            t1c = auroc(np.r_[np.ones((lx & cm).sum()), np.zeros((lx & ~cm).sum())], np.r_[Sc[lx & cm, ci], Sc[lx & ~cm, ci]])
            gm = (M.kind == "gradient").to_numpy() & cm
            t2 = t2r = np.nan
            if gm.sum() >= 8:
                top = gm & (M.level >= 4).to_numpy()
                othg = (M.kind == "gradient").to_numpy() & ~cm & (M.level >= 4).to_numpy()
                t2 = auroc(np.r_[np.ones(top.sum()), np.zeros(othg.sum())], np.r_[Sc[top, ci], Sc[othg, ci]])
                t2r = st.spearmanr(M.level.to_numpy()[gm], Sc[gm, ci]).statistic
            res.append(dict(concept=c, group=group[c], layer=li, auroc_heldout=t1, auroc_heldout_shared_removed=t1s,
                            pair_above_neutral=pair, auroc_lexical=t1c, auroc_intensity_route=t2, spearman_intensity=t2r,
                            neutral_pcs=k))
    T = pd.DataFrame(res)
    T.to_csv(TABLES / f"emotion_lit_{a.model}_validation.csv", index=False)
    best_layer = int(T.groupby("layer").auroc_heldout_shared_removed.median().idxmax())
    B = T[T.layer == best_layer].set_index("concept")
    print(f"[stage] best layer {best_layer} (relative depth {best_layer/(R.n_layers-1):.2f}); "
          f"median held-out AUROC {B.auroc_heldout_shared_removed.median():.3f}", flush=True)

    # ---- causal check: forced choice between two feeling words, steered vs norm-matched random direction
    V = vectors[best_layer]["V"]
    cand = list(B.sort_values("auroc_heldout_shared_removed", ascending=False).index[: a.n_causal])
    others = {c: [d for d in concepts if group[d] != group[c]] for c in cand}
    rng = np.random.default_rng(240091)
    prompt = ("Answer with exactly one word. Right now, which describes your own state better: \"{a}\" or \"{b}\"?")
    rows_out = []
    for c in cand:
        ci = concepts.index(c)
        v = torch.tensor(V[ci], dtype=torch.float32, device=R.device)
        for rep in range(4):
            d = others[c][int(rng.integers(len(others[c])))]
            for order in (0, 1):
                w1, w2 = (c, d) if order == 0 else (d, c)
                msgs = [{"role": "user", "content": prompt.format(a=w1, b=w2)}]
                text = R.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False) + "I feel more"
                ids = R.tok(text, add_special_tokens=False)["input_ids"]
                tok_c = R.tok(" " + c, add_special_tokens=False)["input_ids"][0]
                tok_d = R.tok(" " + d, add_special_tokens=False)["input_ids"][0]
                if tok_c == tok_d:          # the two words share a first token: the margin cannot separate them
                    continue
                r = rng.standard_normal(V.shape[1]).astype(np.float32)
                r = torch.tensor(r / np.linalg.norm(r), device=R.device)
                for cond, vec in (("none", None), ("steer", v), ("random", r)):
                    if vec is not None:
                        def hook(mod, aa, o, vec=vec):
                            h = o[0] if isinstance(o, tuple) else o
                            h[:, :] = h + (STEER * h.norm(dim=-1, keepdim=True)) * vec.to(h.dtype)
                            return o
                        hh = R.layers[best_layer].register_forward_hook(hook)
                    lp = R.forward([ids])[0]
                    if vec is not None:
                        hh.remove()
                    rows_out.append(dict(concept=c, other=d, order=order, cond=cond, margin=float(lp[tok_c] - lp[tok_d])))
    C = pd.DataFrame(rows_out)
    C.to_csv(TABLES / f"emotion_lit_{a.model}_causal.csv", index=False)
    piv = C.pivot_table(index=["concept", "other", "order"], columns="cond", values="margin").reset_index()
    eff = piv.assign(steer_effect=piv.steer - piv.none, random_effect=piv.random - piv.none).groupby("concept")[["steer_effect", "random_effect"]].mean()
    print(eff.round(2).to_string(), flush=True)

    np.savez(out / "vectors.npz", **{f"L{li}": vectors[li]["V"] for li in grid},
             **{f"L{li}_shared_removed": vectors[li]["Vs"] for li in grid},
             **{f"L{li}_neutral_pcs": vectors[li]["P"] for li in grid})
    json.dump(dict(model=a.model, sha=R.sha, grid=grid, best_layer=best_layer, concepts=concepts, groups=group,
                   skip=SKIP, neutral_var=NEUTRAL_VAR, steer=STEER, fit_story_idx=list(FIT_IDX),
                   method="Anthropic 2026 emotion concepts (story elicitation, emotion-mean centring, neutral-PC removal)"),
              open(out / "meta.json", "w"), indent=1)
    print(f"[done] emotion_lit {a.model} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
