"""Cluster bootstrap, BH-FDR, and small helpers used by every analysis table."""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG_SEED = 240091


def cluster_bootstrap(df, stat_fn, cluster="cluster", B=2000, seed=RNG_SEED, alpha=0.05):
    """Percentile CI for stat_fn(df) resampling whole clusters with replacement. Returns (est, lo, hi, p_two_sided_vs_0)."""
    est = stat_fn(df)
    groups = {k: g for k, g in df.groupby(cluster, sort=False)}
    keys = np.array(list(groups))
    rng = np.random.default_rng(seed)
    boots = np.empty(B)
    for b in range(B):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        boots[b] = stat_fn(pd.concat([groups[keys[i]] for i in pick], ignore_index=True))
    boots = boots[np.isfinite(boots)]
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return est, lo, hi, min(1.0, max(p, 1.0 / len(boots)))


def fast_cluster_mean_ci(values, clusters, B=4000, seed=RNG_SEED, alpha=0.05):
    """Bootstrap CI for the mean of per-cluster means weighted by cluster size (vectorized)."""
    s = pd.DataFrame(dict(v=values, c=clusters)).dropna()
    agg = s.groupby("c")["v"].agg(["sum", "count"])
    sums, cnts = agg["sum"].to_numpy(), agg["count"].to_numpy()
    n = len(sums)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boots = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1)
    est = sums.sum() / cnts.sum()
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return est, lo, hi, min(1.0, max(p, 1.0 / B)), n


def bh_fdr(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    if len(pv) == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.minimum(ranked, 1)
    q[ok] = out
    return q


def ece(conf, correct, bins=10):
    conf = np.asarray(conf, float) / 100.0
    correct = np.asarray(correct, float)
    ok = np.isfinite(conf) & np.isfinite(correct)
    conf, correct = conf[ok], correct[ok]
    if len(conf) == 0:
        return np.nan
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b) if a > 0 else (conf >= a) & (conf <= b)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return e
