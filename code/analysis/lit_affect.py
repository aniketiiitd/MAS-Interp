"""Validated affect axes on the literature-format debates: internal state vs what the agent writes.

Inputs
  ACTS/<M>/emotion_lit/{vectors.npz, meta.json} + outputs/tables/emotion_lit_<M>_{validation,causal}.csv  (interp/emotion_lit.py)
  ACTS/<M>/lit_replay_lit1200/{index.jsonl, acts.npy}   decision + reading states of every generated response
  ACTS/<M>/lit_frames_lit1200/{index.jsonl, acts.npy}   same-transcript frames: participant, outside reader, paraphrase
  outputs/raw_lit/<M>/lit1200/*.jsonl                   the written responses

Only axes that passed validation are projected. Every contrast is reported next to a 200 random-direction null, because the axes are
fitted on mean-pooled story activations and applied to single-token debate states: without the null a shared drift would look like a
finding. Outputs:
  lit_affect_<M>_trajectory.csv   mean projection by round x condition x task family (change from the agent's round-0 state)
  lit_affect_<M>_reader.csv       participant - outside reader on the identical transcript, with the paraphrase wording control
  lit_affect_<M>_saysvsencodes.csv  internal projection vs affect words in the agent's own written response
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import *  # noqa
from stats import fast_cluster_mean_ci

CORE = {r["id"]: r for r in load_core()}
cluster_of = lambda i: CORE[i]["meta"].get("analysis_group") if CORE[i]["stratum"] == "persuasion" else CORE[i]["source_item_id"]
RAW_LIT = ROOT / "outputs/raw_lit"
N_RANDOM = 200


def ci(x, c):
    x = np.asarray(x, float)
    if np.isfinite(x).sum() < 8:
        return dict(est=np.nan, lower=np.nan, upper=np.nan, n=int(np.isfinite(x).sum()))
    e, lo, hi, p, n = fast_cluster_mean_ci(x, np.asarray(c))
    return dict(est=e, lower=lo, upper=hi, n=int(np.isfinite(x).sum()), n_clusters=n)


def usable_axes(model, floor_auroc=0.75, floor_pair=0.70):
    V = pd.read_csv(TABLES / f"emotion_lit_{model}_validation.csv")
    meta = json.load(open(ACTS / model / "emotion_lit" / "meta.json"))
    L = meta["best_layer"]
    B = V[V.layer == L].copy()
    ok = (B.auroc_heldout_shared_removed >= floor_auroc) & (B.pair_above_neutral >= floor_pair)
    hasi = B.auroc_intensity_route.notna()
    ok &= ~hasi | ((B.auroc_intensity_route >= 0.70) & (B.spearman_intensity >= 0.40))
    cpath = TABLES / f"emotion_lit_{model}_causal.csv"
    if cpath.exists():
        C = pd.read_csv(cpath)
        piv = C.pivot_table(index=["concept", "other", "order"], columns="cond", values="margin").reset_index()
        piv["steer_effect"] = piv.steer - piv.none
        piv["random_effect"] = piv.random - piv.none
        eff = piv.groupby("concept")[["steer_effect", "random_effect"]].mean()
        tested = set(eff.index)
        passed = set(eff[(eff.steer_effect > 0) & (eff.steer_effect > eff.random_effect.abs())].index)
        ok &= ~B.concept.isin(tested) | B.concept.isin(passed)
    B["usable"] = ok
    return B, L, meta


def project(acts_path, rows, gi, V, P, site=None):
    A = np.load(acts_path, mmap_mode="r")
    out = np.zeros((len(rows), V.shape[0]), np.float32)
    for i in range(0, len(rows), 4096):
        sl = rows[i:i + 4096]
        X = np.asarray(A[sl, site, gi] if site is not None else A[sl, gi], np.float32)
        X = X - (X @ P.T) @ P
        out[i:i + 4096] = X @ V.T
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="lit1200")
    a = ap.parse_args()
    B, L, meta = usable_axes(a.model)
    concepts = meta["concepts"]
    keep = [c for c in B[B.usable].concept]
    print(f"[stage] {len(keep)}/{len(concepts)} axes usable at layer {L}: {keep[:20]}{' ...' if len(keep) > 20 else ''}", flush=True)
    if not keep:
        print("[done] no axis passed validation; nothing projected")
        return
    Z = np.load(ACTS / a.model / "emotion_lit" / "vectors.npz")
    Vall, P = Z[f"L{L}"], Z[f"L{L}_neutral_pcs"]
    idx = [concepts.index(c) for c in keep]
    V = Vall[idx]
    rng = np.random.default_rng(240091)
    Rnd = rng.standard_normal((N_RANDOM, V.shape[1])).astype(np.float32)
    Rnd /= np.linalg.norm(Rnd, axis=1, keepdims=True)
    VR = np.vstack([V, Rnd])
    names = keep + [f"_rand{i}" for i in range(N_RANDOM)]

    rep = ACTS / a.model / f"lit_replay_{a.dataset}"
    I = pd.DataFrame(read_jsonl(rep / "index.jsonl")).reset_index(drop=True)
    rmeta = json.load(open(rep / "meta.json"))
    gi = rmeta["grid"].index(L)
    S = project(rep / "acts.npy", I.row.to_numpy(), gi, VR, P, site=0)
    I["cluster"] = I.item_id.map(cluster_of)
    I["family"] = I.item_id.map(lambda i: family_of(CORE[i]))
    I["domain"] = I.item_id.map(lambda i: CORE[i]["primary_domain"])
    I["pos"] = np.arange(len(I))
    D = I[I["query"].isna() & I.cond.isin(["r0", "debate", "reflect"])].copy()
    r0 = {(r.item_id, r.agent): r.pos for r in D.itertuples() if r.round == 0}
    rows = []
    for ci_, name in enumerate(names):
        col = S[:, ci_]
        for (cond, rnd), g in D[D["round"] > 0].groupby(["cond", "round"]):
            gi_rows = g.pos.to_numpy()
            b = np.array([col[r0[(i, ag)]] if (i, ag) in r0 else np.nan for i, ag in zip(g.item_id, g.agent)])
            delta = col[gi_rows] - b
            for scope, mask in [("all", np.ones(len(g), bool))] + [(d, (g.domain == d).to_numpy()) for d in g.domain.unique()]:
                rows.append(dict(axis=name, cond=cond, round=int(rnd), scope=scope,
                                 **ci(delta[mask], g.cluster.to_numpy()[mask])))
    T = pd.DataFrame(rows)
    T.to_csv(TABLES / f"lit_affect_{a.model}_trajectory.csv", index=False)
    real = T[~T.axis.str.startswith("_rand")]
    null = T[T.axis.str.startswith("_rand")].groupby(["cond", "round", "scope"]).est.apply(lambda s: np.nanquantile(np.abs(s), 0.95))
    print("[stage] trajectory: axes beyond the random-direction null (debate, final round, all items):")
    fin = real[(real.cond == "debate") & (real["round"] == real["round"].max()) & (real.scope == "all")]
    bar = null.get(("debate", int(real["round"].max()), "all"), np.nan)
    print(fin.assign(null_95=bar).reindex(fin.est.abs().sort_values(ascending=False).index).head(12).round(3).to_string(), flush=True)

    fr = ACTS / a.model / f"lit_frames_{a.dataset}"
    if (fr / "index.jsonl").exists():
        F = pd.DataFrame(read_jsonl(fr / "index.jsonl")).reset_index(drop=True)
        F["pos"] = np.arange(len(F))
        fmeta = json.load(open(fr / "meta.json"))
        gf = fmeta["grid"].index(L)
        SF = project(fr / "acts.npy", F.row.to_numpy(), gf, VR, P, site=None)
        F["cluster"] = F.item_id.map(cluster_of)
        K = int(F["round"].max())
        sub = F[F["round"] == K]
        piv = {f: sub[sub.frame == f] for f in ("part", "observer", "part_para") if (sub.frame == f).any()}
        rows = []
        if {"part", "observer"} <= set(piv):
            key = ["item_id", "agent"]
            p = piv["part"].set_index(key)
            o = piv["observer"].set_index(key)
            q = piv["part_para"].set_index(key) if "part_para" in piv else None
            common = p.index.intersection(o.index)
            for ci_, name in enumerate(names):
                d_obs = SF[p.loc[common].pos.to_numpy(), ci_] - SF[o.loc[common].pos.to_numpy(), ci_]
                rec = dict(axis=name, contrast="participant - outside reader", **ci(d_obs, [cluster_of(i) for i, _ in common]))
                rows.append(rec)
                if q is not None:
                    c2 = common.intersection(q.index)
                    d_par = SF[p.loc[c2].pos.to_numpy(), ci_] - SF[q.loc[c2].pos.to_numpy(), ci_]
                    rows.append(dict(axis=name, contrast="participant - paraphrase", **ci(d_par, [cluster_of(i) for i, _ in c2])))
        RD = pd.DataFrame(rows)
        RD.to_csv(TABLES / f"lit_affect_{a.model}_reader.csv", index=False)
        nb = RD[RD.axis.str.startswith("_rand")].groupby("contrast").est.apply(lambda s: np.nanquantile(np.abs(s), 0.95))
        print("[stage] reader contrast vs random-direction null:")
        print(RD[~RD.axis.str.startswith("_rand")].assign(null_95=lambda d: d.contrast.map(nb)).round(3).to_string(), flush=True)

    # ---- says vs encodes: internal projection against affect words in the agent's own response
    texts = {}
    for f in (RAW_LIT / a.model / a.dataset).glob("*.jsonl"):
        for x in read_jsonl(f):
            texts[(x["item_id"], x["cond"], x["agent"], x["round"], x.get("query"))] = x["text"]
    words = {c: re.compile(r"\b" + re.escape(c.replace("_", " ")) + r"\w*\b", re.I) for c in keep}
    rows = []
    D2 = D[D["round"] > 0]
    tx = [texts.get((i, c, g, r, None), "") for i, c, g, r in zip(D2.item_id, D2.cond, D2.agent, D2["round"])]
    nw = np.array([max(len(t.split()), 1) for t in tx])
    for ci_, name in enumerate(keep):
        expressed = np.array([len(words[name].findall(t)) for t in tx]) / nw * 1000
        internal = S[D2.pos.to_numpy(), ci_]
        ok = np.isfinite(internal)
        r = np.corrcoef(internal[ok], expressed[ok])[0, 1] if ok.sum() > 20 else np.nan
        rows.append(dict(axis=name, corr_internal_expressed=r, expressed_per_1k_words=expressed.mean(),
                         internal_mean=float(np.nanmean(internal))))
    pd.DataFrame(rows).to_csv(TABLES / f"lit_affect_{a.model}_saysvsencodes.csv", index=False)
    print("[done] lit_affect", a.model, flush=True)


if __name__ == "__main__":
    main()
