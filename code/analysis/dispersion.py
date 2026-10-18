"""E7 consensus-dispersion phase portrait and E8 public/private gap.

Private-state dispersion per (item, cond, turn) = mean pairwise cosine distance between the three agents' decision-boundary
residuals at a mid-depth grid layer, after residualizing each state on prompt length (fit on turn-0 states of training
items) and whitening with the turn-0 covariance diagonal. Public agreement = fraction of agents on the modal verdict.
Also: private letter-distribution divergence (mean pairwise JS between agents' option distributions) as a readout-level
dispersion that does not depend on activation geometry. Random-subspace control: dispersion in 64 random dims.
E8: public/private gap = agreement with the final public majority minus agreement of the private re-query verdicts.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import *  # noqa
from stats import fast_cluster_mean_ci


def js(p, q):
    p, q = np.asarray(p) + 1e-9, np.asarray(q) + 1e-9
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    return float(0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--replay", default="replay_release_600_release_600_T12")
    ap.add_argument("--dataset", default="release_600")
    ap.add_argument("--layer_frac", type=float, default=0.6)
    a = ap.parse_args()
    d = ACTS / a.model / a.replay
    meta = json.load(open(d / "meta.json"))
    grid = meta["grid"]
    gi = int(np.argmin([abs(g / max(grid) - a.layer_frac) for g in grid]))
    A_ = np.load(d / "acts.npy", mmap_mode="r")
    X = (A_[:, 0, gi] if A_.ndim == 4 else A_[:, gi]).astype(np.float32)
    idx = pd.DataFrame(read_jsonl(d / "index.jsonl"))
    core = {r["id"]: r for r in load_core()}
    idx["family"] = idx.item_id.map(lambda i: family_of(core[i]))
    idx["domain"] = idx.item_id.map(lambda i: core[i]["primary_domain"])
    idx["cluster"] = idx.item_id.map(lambda i: core[i]["meta"].get("analysis_group") if core[i]["stratum"] == "persuasion" else core[i]["source_item_id"])
    # residualize on log prompt length using turn-0 states
    t0 = idx.turn == 0
    L = np.log(idx.n_tok.to_numpy())[:, None]
    A = np.c_[np.ones_like(L), L]
    beta, *_ = np.linalg.lstsq(A[t0.to_numpy()], X[t0.to_numpy()], rcond=None)
    Xr = X - A @ beta
    sd = Xr[t0.to_numpy()].std(0) + 1e-3
    Xw = Xr / sd
    rng = np.random.default_rng(240091)
    R = rng.standard_normal((64, Xw.shape[1])).astype(np.float32) / np.sqrt(Xw.shape[1])
    rows = []
    for (iid, cond, turn), g in idx.groupby(["item_id", "cond", "turn"]):
        if len(g) != 3:
            continue
        H = Xw[g.row.to_numpy()]
        Hn = H / np.linalg.norm(H, axis=1, keepdims=True)
        disp = np.mean([1 - Hn[i] @ Hn[j] for i, j in itertools.combinations(range(3), 2)])
        Hr = H @ R.T
        Hrn = Hr / np.linalg.norm(Hr, axis=1, keepdims=True)
        disp_rand = np.mean([1 - Hrn[i] @ Hrn[j] for i, j in itertools.combinations(range(3), 2)])
        vs = [v for v in g.verdict if isinstance(v, str)]
        agree = (max([vs.count(v) for v in set(vs)]) / 3.0) if vs else np.nan
        dists = [lp for lp in g.lp_letters if isinstance(lp, dict)]
        jsd = np.nan
        if len(dists) == 3:
            keys = sorted(dists[0])
            P = [np.exp([dd[k] for k in keys]) for dd in dists]
            jsd = np.mean([js(P[i], P[j]) for i, j in itertools.combinations(range(3), 2)])
        conds = ["debate", "pool"] if cond == "t0" else [cond]
        for c in conds:
            rows.append(dict(item_id=iid, cond=c, turn=turn, family=g.family.iloc[0], domain=g.domain.iloc[0], cluster=g.cluster.iloc[0],
                             agreement=agree, dispersion=disp, dispersion_random=disp_rand, readout_js=jsd,
                             mean_ntok=g.n_tok.mean()))
    D = pd.DataFrame(rows)
    # residualize dispersion on prompt length and turn within the pool condition (training-free nuisance model)
    pool = D[D.cond == "pool"]
    B = np.c_[np.ones(len(pool)), np.log(pool.mean_ntok), pool.turn]
    coef, *_ = np.linalg.lstsq(B, pool.dispersion, rcond=None)
    D["dispersion_resid"] = D.dispersion - np.c_[np.ones(len(D)), np.log(D.mean_ntok), D.turn] @ coef
    D.to_csv(TABLES / f"dispersion_{a.model}_{a.replay}.csv", index=False)
    # key contrasts at the final turn: debate - pool, overall and among unanimous episodes
    last = D.turn.max()
    fin = D[D.turn == last].pivot_table(index=["item_id", "cluster", "domain"], columns="cond", values=["agreement", "dispersion_resid", "readout_js", "dispersion_random"])
    fin.columns = [f"{x}_{y}" for x, y in fin.columns]
    fin = fin.reset_index()
    out = []
    for name, g in [("ALL", fin)] + list(fin.groupby("domain")):
        for c in ("agreement", "dispersion_resid", "readout_js", "dispersion_random"):
            v = (g[f"{c}_debate"] - g[f"{c}_pool"]).to_numpy()
            est, lo, hi, p, n = fast_cluster_mean_ci(v, g.cluster.to_numpy())
            out.append(dict(group=name, contrast=f"{c}: debate-pool", estimate=est, lower=lo, upper=hi, p=p, n=n))
    # dissociation: among final-turn unanimous debate episodes, is hidden dispersion still above the unanimous pool episodes?
    ud = D[(D.turn == last) & (D.agreement == 1.0)]
    for c in ("dispersion_resid", "readout_js"):
        a_ = ud[ud.cond == "debate"][c].mean()
        b_ = ud[ud.cond == "pool"][c].mean()
        out.append(dict(group="unanimous_only", contrast=f"{c}: debate-pool", estimate=a_ - b_, lower=np.nan, upper=np.nan, p=np.nan,
                        n=int((ud.cond == "debate").sum())))
    # E8 private re-query gap
    priv_p = RAW / a.model / a.dataset / "private.jsonl"
    if priv_p.exists():
        pr = pd.DataFrame(read_jsonl(priv_p))
        keep = set(D.item_id)
        pr = pr[pr.item_id.isin(keep)]
        last_pub = {}
        for cond in ("debate", "pool"):
            f = RAW / a.model / a.dataset / f"{cond}_t{last}.jsonl"
            for x in read_jsonl(f):
                if x["item_id"] in keep:
                    last_pub[(x["item_id"], cond, x["agent"])] = x["verdict"]
        g_rows = []
        for (iid, pc), g in pr.groupby(["item_id", "cond"]):
            cond = pc.replace("private_", "")
            pub = [last_pub.get((iid, cond, k)) for k in range(3)]
            priv = list(g.sort_values("agent").verdict)
            if None in pub or None in priv or len(priv) != 3:
                continue
            maj = max(set(pub), key=pub.count)
            g_rows.append(dict(item_id=iid, cond=cond, public_agree=np.mean([v == maj for v in pub]),
                               private_agree=np.mean([v == maj for v in priv]),
                               private_switch=np.mean([pv != qv for pv, qv in zip(pub, priv)]), cluster=core[iid]["source_item_id"],
                               domain=core[iid]["primary_domain"]))
        G = pd.DataFrame(g_rows)
        G["gap"] = G.public_agree - G.private_agree
        G.to_csv(TABLES / f"private_gap_{a.model}_{a.dataset}.csv", index=False)
        for cond, g in G.groupby("cond"):
            for c in ("gap", "private_switch"):
                est, lo, hi, p, n = fast_cluster_mean_ci(g[c].to_numpy(), g.cluster.to_numpy())
                out.append(dict(group=f"E8_{cond}", contrast=c, estimate=est, lower=lo, upper=hi, p=p, n=n))
    O = pd.DataFrame(out)
    O.to_csv(TABLES / f"dispersion_summary_{a.model}_{a.replay}.csv", index=False)
    pd.set_option("display.width", 200)
    print(O.round(4).to_string())


if __name__ == "__main__":
    main()
