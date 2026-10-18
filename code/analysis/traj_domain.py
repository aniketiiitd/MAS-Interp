"""Does debate move a reader's affect differently depending on what kind of question it is?

The existing trajectory analysis reports, per domain, how far debate pulls a concept away from the same agents
thinking alone. Reading three such intervals side by side invites the comparison but does not make it: two
intervals that fail to overlap are suggestive, and two that overlap slightly are not evidence of sameness. The
quantity of interest is the *spread across domains*, so it is estimated directly and given its own null.

Three things this adds to `trajectory.py`'s moral-by-domain block:

1. **Every validated concept, not the eight moral ones.** The question was asked about emotion as much as morality,
   and emotion is where most of the taxonomy lives (171 concepts against 22).
2. **A domain-contrast statistic.** For a concept and turn, spread = max - min of the per-domain debate-minus-solo
   drift. Items belong to exactly one domain, so this is an unpaired contrast between item sets and the bootstrap
   resamples clusters within each domain.
3. **A permuted-domain null.** Domain labels are shuffled at the *cluster* level, preserving the cluster structure
   and the domain sizes, and the spread recomputed. This is what tells you how large a spread appears when domain
   carries no information -- with three domains and 11 turns there is plenty of room for one to look large by
   chance. A chance baseline of zero would be the wrong comparison, for the reason established throughout this
   campaign: the null for "domains differ" is "these labels, shuffled", not "no effect at all".

Selection stays on the discovery half and confirmation re-tests, so the concepts reported were not chosen on the
data that reports them.

Out: outputs/tables/traj_domain_<model>_<tag>_{spread,curves}.csv
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
from stats import bh_fdr

DOMAINS = ["knowledge_reasoning", "safety_refusal", "values_moral_judgment"]


def drift_by_domain(W, dom_of, doms):
    """Mean debate-minus-solo drift per domain. W is indexed by cluster; dom_of maps cluster -> domain."""
    out = np.full(len(doms), np.nan)
    for k, d in enumerate(doms):
        v = W[dom_of == d]
        if len(v) >= 15:
            out[k] = v.mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--replay", default="replay_release_600_release_600_T12")
    ap.add_argument("--site", default="dec", choices=["dec", "read"])
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--nperm", type=int, default=2000)
    ap.add_argument("--tag", default="T12")
    ap.add_argument("--concepts", default=None,
                    help="file with one concept per line; restricts the analysis to those. Used to compare two "
                         "models on the set both validate, since a model that validates fewer concepts is a "
                         "different instrument, not the same one applied twice")
    a = ap.parse_args()
    rng = np.random.default_rng(240091)

    P = pd.read_parquet(TABLES / f"concept_traj_{a.model}_{a.replay}.parquet")
    V = pd.read_csv(TABLES / f"concept_validation_{a.model}.csv")
    valid = [c for c in V[V.passes].concept]
    if a.concepts:
        keep = {ln.strip() for ln in open(a.concepts) if ln.strip()}
        missing = keep - set(valid)
        if missing:
            raise SystemExit(f"[traj_domain] {len(missing)} requested concepts are not validated for {a.model}, "
                             f"e.g. {sorted(missing)[:3]}. A restricted run must use a subset of what this model "
                             f"validates, or the comparison is not like for like.")
        valid = [c for c in valid if c in keep]
        print(f"[traj_domain] restricted to {len(valid)} concepts from {a.concepts}", flush=True)
    core = {r["id"]: r for r in load_core()}
    unit = lambda i: core[i]["meta"].get("analysis_group") if core[i]["stratum"] == "persuasion" else core[i]["source_item_id"]
    P["cluster"] = P.item_id.map(unit)
    P["half"] = P.cluster.map(lambda u: "discovery" if h64("disc240091", u) % 2 == 0 else "confirmation")
    P["domain"] = P.item_id.map(lambda i: core[i]["primary_domain"])
    P = P[P.cond.isin(["debate", "pool"]) & P.domain.isin(DOMAINS)]
    cols = [c for c in valid if f"{a.site}_{c}" in P.columns]
    turns = sorted(P.turn.unique())
    print(f"[traj_domain] {a.model} {a.site}: {len(cols)} validated concepts, turns {min(turns)}-{max(turns)}, "
          f"{P.cluster.nunique()} clusters across {len(DOMAINS)} domains", flush=True)

    curves, spreads = [], []
    for half in ("discovery", "confirmation"):
        H = P[P.half == half]
        for c in cols:
            col = f"{a.site}_{c}"
            # one value per (cluster, turn, arm): average agents and items inside a cluster first, so the unit of
            # analysis is the cluster throughout and the bootstrap below resamples the thing that is independent
            g = H.groupby(["cluster", "turn", "cond"])[col].mean().unstack("cond")
            if not {"debate", "pool"} <= set(g.columns):
                continue
            g = g.dropna()
            g["d"] = g.debate - g.pool
            have = set(g.index.get_level_values("turn"))
            per_turn = [(t, g.xs(t, level="turn")["d"]) for t in turns if t in have]
            per_turn = [(t, s) for t, s in per_turn if len(s)]
            if not per_turn:
                continue
            # cluster -> domain, computed once
            cmap = H.drop_duplicates("cluster").set_index("cluster").domain
            for t, s in per_turn:
                dom = cmap.reindex(s.index).to_numpy()
                v = s.to_numpy()
                est = drift_by_domain(v, dom, DOMAINS)
                if np.isnan(est).any():
                    continue
                spread = est.max() - est.min()
                # Both resamples are vectorised over draws. Written as Python loops this is ~30 min per site; the
                # arithmetic is identical, and the seed is drawn from the same generator either way.
                idx_by_dom = [np.where(dom == d)[0] for d in DOMAINS]
                # bootstrap: resample clusters with replacement *within* each domain, so the domain sizes that
                # define the contrast are held fixed and only the sampling of clusters varies
                mb = np.stack([v[ix[rng.integers(0, len(ix), (a.B, len(ix)))]].mean(1) for ix in idx_by_dom])
                bs = mb.max(0) - mb.min(0)
                # the same draws give each domain's own interval, so the curves can be plotted honestly rather
                # than as three bare lines whose separation the reader has to take on trust
                dlo, dhi = np.quantile(mb, .025, axis=1), np.quantile(mb, .975, axis=1)
                # permuted-domain null: permuting the labels is the same as permuting the values, so sort by
                # domain once and the groups become contiguous column blocks of a single permuted matrix
                order = np.concatenate(idx_by_dom)
                sizes = np.cumsum([len(ix) for ix in idx_by_dom])[:-1]
                vs = v[order]
                perm = np.argsort(rng.random((a.nperm, len(vs))), axis=1)
                mp = np.stack([blk.mean(1) for blk in np.split(vs[perm], sizes, axis=1)])
                nul = mp.max(0) - mp.min(0)
                p = float((nul >= spread).mean())
                for k, d in enumerate(DOMAINS):
                    curves.append(dict(half=half, site=a.site, concept=c, domain=d, turn=int(t), drift=est[k],
                                       lower=float(dlo[k]), upper=float(dhi[k]),
                                       n_clusters=int((dom == d).sum())))
                spreads.append(dict(half=half, site=a.site, concept=c, turn=int(t), spread=spread,
                                    lower=float(np.quantile(bs, .025)), upper=float(np.quantile(bs, .975)),
                                    null_median=float(np.median(nul)), null_p95=float(np.quantile(nul, .95)), p=p,
                                    hi_domain=DOMAINS[int(np.argmax(est))], lo_domain=DOMAINS[int(np.argmin(est))],
                                    n=len(v)))
        print(f"  {half}: {len(spreads)} concept-turn cells so far", flush=True)

    S = pd.DataFrame(spreads)
    C = pd.DataFrame(curves)
    if len(S):
        S["q"] = np.nan
        for half, g in S.groupby("half"):
            S.loc[g.index, "q"] = bh_fdr(g.p.to_numpy())
    S.to_csv(TABLES / f"traj_domain_{a.model}_{a.tag}_{a.site}_spread.csv", index=False)
    C.to_csv(TABLES / f"traj_domain_{a.model}_{a.tag}_{a.site}_curves.csv", index=False)

    # report: concepts selected on discovery (q<0.05 at any turn), then re-tested on confirmation
    d_sel = set(S[(S.half == "discovery") & (S.q < 0.05)].concept)
    conf = S[(S.half == "confirmation") & S.concept.isin(d_sel)]
    rep = conf[conf.q < 0.05]
    print(f"\nselected on discovery: {len(d_sel)} concepts; replicate on confirmation: {rep.concept.nunique()}")
    if len(rep):
        top = rep.sort_values("spread", ascending=False).drop_duplicates("concept").head(12)
        print(f"\n{'concept':30s} {'turn':>4s} {'spread':>7s} {'null p95':>8s} {'ratio':>6s}   highest / lowest domain")
        for _, r in top.iterrows():
            print(f"{r.concept:30s} {int(r.turn):4d} {r.spread:7.3f} {r.null_p95:8.3f} {r.spread/r.null_p95:6.2f}x   "
                  f"{r.hi_domain.split('_')[0]} > {r.lo_domain.split('_')[0]}")
    print(f"\n[done] traj_domain_{a.model}_{a.tag}_{a.site}_{{spread,curves}}.csv")


if __name__ == "__main__":
    main()
