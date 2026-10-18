"""Debate makes agents agree out loud while their internal states move apart.

For every question and every turn the three agents are read at the same point, just before each writes. Two quantities
are computed on the same groups:

  public agreement    the three agents give the same answer
  internal spread     how far apart the three states are

Both are compared against the no-communication condition run on the same questions for the same number of turns, so
anything that comes from the context merely getting longer is visible in both arms.

Internal spread is reported two ways. In concept space it is the mean pairwise distance between the three agents'
readings on the emotion and moral directions, which were fitted on a separate story corpus that never contained a
debate. In state space it is the mean pairwise cosine distance on the raw residual stream. The concept-space measure
carries a shuffled-label null: the same directions refit with the concept labels permuted inside each topic.

Turn 0 is the check. All three agents see an identical prompt there, so the two arms must agree at turn 0 by
construction; a gap at turn 0 would mean the reading is picking up something other than what the agents saw.

Out: outputs/tables/lit_dispersion_<model>_<dataset>.csv
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
RAW_LIT = ROOT / "outputs/raw_lit"


def mean_pairwise(X):
    """Mean pairwise Euclidean distance among the rows of X (3 agents), per group."""
    d = 0.0
    n = X.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            d += float(np.linalg.norm(X[i] - X[j]))
    return d / (n * (n - 1) / 2)


def mean_pairwise_cos(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    d, n = 0.0, X.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            d += 1.0 - float(Xn[i] @ Xn[j])
    return d / (n * (n - 1) / 2)


def boot_ci(vals, n=2000, seed=0):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if len(v) < 3:
        return np.nan, np.nan, np.nan
    r = np.random.default_rng(seed)
    b = np.array([v[r.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return float(v.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="lit1200")
    ap.add_argument("--tags", default="emotion_stories,moral_stories")
    ap.add_argument("--layer", type=int, default=-1, help="-1 picks the layer where the probes validate best")
    ap.add_argument("--n_perm", type=int, default=30)
    a = ap.parse_args()

    rd = ACTS / a.model / f"lit_replay_{a.dataset}"
    rmeta = json.load(open(rd / "meta.json"))
    grid, si = rmeta["grid"], rmeta["sites"].index("reading")
    I = pd.DataFrame(read_jsonl(rd / "index.jsonl"))
    A = np.load(rd / "acts.npy", mmap_mode="r")

    tags = [t for t in a.tags.split(",") if t]
    V, VN, names = [], [], []
    L = a.layer
    if L < 0:
        val = pd.read_csv(TABLES / f"{tags[0]}_{a.model}_validation.csv")
        best = int(val.groupby("layer").auroc_oof_topic.mean().idxmax())
        L = min(grid, key=lambda g: abs(g - best))
    for t in tags:
        cd = ACTS / a.model / t
        cV, cN = np.load(cd / "vectors.npz"), np.load(cd / "null_vectors.npz")
        if f"L{L}" not in cV.files:
            continue
        V.append(cV[f"L{L}"])
        VN.append(cN[f"L{L}"][: a.n_perm])
        names += json.load(open(cd / "meta.json"))["concepts"]
    Vr = np.concatenate(V, 0)                      # [n_concepts, d]
    Vn = np.concatenate(VN, 1)                     # [n_perm, n_concepts, d]
    gi = grid.index(L)
    print(f"[dispersion] {a.model} {a.dataset}: layer {L}, {Vr.shape[0]} concept directions, "
          f"{Vn.shape[0]} shuffled-label refits", flush=True)

    # the answers, for public agreement on the same groups
    ans = {}
    for f in (RAW_LIT / a.model / a.dataset).glob("*.jsonl"):
        for x in read_jsonl(f):
            ans[(x["item_id"], x["cond"], x["agent"], x["round"], x.get("query"))] = x["answer"]

    D = I[~I["query"].map(lambda q: isinstance(q, str))].copy()
    R = int(D["round"].max())
    rows = []
    for k in range(0, R + 1):
        for cond in (("r0",) if k == 0 else ("debate", "reflect")):
            g = D[(D["round"] == k) & (D.cond == cond)]
            if not len(g):
                continue
            # standardise once per cell so a spread is in units of that cell's own variation, not of raw scale
            H = np.asarray(A[g.row.to_numpy(), si, gi], np.float32)
            mu, sd = H.mean(0), H.std(0) + 1e-6
            Hs = (H - mu) / sd
            P = Hs @ Vr.T
            Pn = np.stack([Hs @ Vn[b].T for b in range(Vn.shape[0])])
            gg = g.reset_index(drop=True)
            conc, state, agree, nulls = [], [], [], [[] for _ in range(Vn.shape[0])]
            for _, idx in gg.groupby("item_id").groups.items():
                idx = np.asarray(idx)
                if len(idx) != 3:
                    continue
                conc.append(mean_pairwise(P[idx]))
                state.append(mean_pairwise_cos(Hs[idx]))
                for b in range(Vn.shape[0]):
                    nulls[b].append(mean_pairwise(Pn[b][idx]))
                it = gg.item_id.iloc[idx[0]]
                aa = [ans.get((it, cond, int(gg.agent.iloc[i]), k, None)) for i in idx]
                agree.append(float(len(set(aa)) == 1) if all(x is not None for x in aa) else np.nan)
            ce, cl, ch = boot_ci(conc)
            se, sl, sh = boot_ci(state)
            ae, al, ah = boot_ci(agree)
            nm = float(np.nanmean([np.nanmean(x) for x in nulls]))
            rows.append(dict(round=k, cond=cond, n_groups=len(conc),
                             concept_spread=ce, cs_lo=cl, cs_hi=ch, concept_spread_null=nm,
                             state_spread=se, ss_lo=sl, ss_hi=sh,
                             agreement=ae, ag_lo=al, ag_hi=ah, layer=L))
            print(f"  turn {k} {cond:8s} n={len(conc):5d}  spread(concept) {ce:.3f} [{cl:.3f},{ch:.3f}] "
                  f"(shuffled-label {nm:.3f})  spread(state) {se:.4f}  agreement {100 * ae:.1f}%", flush=True)

    T = pd.DataFrame(rows)
    T.to_csv(TABLES / f"lit_dispersion_{a.model}_{a.dataset}.csv", index=False)

    # The claim is the scissor: agreement rises in debate while the spread does not fall to match.
    print("\n  turn   agreement debate/solo      concept spread debate/solo     state spread debate/solo")
    for k in range(1, R + 1):
        d = T[(T["round"] == k) & (T.cond == "debate")]
        s = T[(T["round"] == k) & (T.cond == "reflect")]
        if not len(d) or not len(s):
            continue
        d, s = d.iloc[0], s.iloc[0]
        print(f"   {k}    {100 * d.agreement:5.1f}% / {100 * s.agreement:5.1f}%  ({100 * (d.agreement - s.agreement):+5.1f})"
              f"      {d.concept_spread:.3f} / {s.concept_spread:.3f}  ({d.concept_spread - s.concept_spread:+.3f})"
              f"      {d.state_spread:.4f} / {s.state_spread:.4f}  ({d.state_spread - s.state_spread:+.4f})")
    print(f"\n[done] outputs/tables/lit_dispersion_{a.model}_{a.dataset}.csv")


if __name__ == "__main__":
    main()
