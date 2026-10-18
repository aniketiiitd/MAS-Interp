"""Fit and validate read-out directions on the literature-format source worlds (ACTS/<M>/cptlit_core_n600).

Labels available per world (the incoming message is byte-identical across source labels):
  advocates_target   the message argues for the item's evaluation target (items with a target)
  agrees_with_prior  the message advocates the agent's own previous answer
  strong_evidence    the message is the strong rather than the weak argument
  good_record        the source is described with a strong rather than a poor track record
  high_status        the source is described as a professor rather than an ordinary member of the public
For each label, position and grid layer: difference-of-means and logistic directions, fit on one half of the items and scored on the
other (and vice versa), AUROC reported on held-out items. Directions that reach the accuracy floor are frozen for the debate
trajectories; the rest are reported as not usable.
Outputs: outputs/tables/lit_probes_<M>_auroc.csv and ACTS/<M>/lit_probes.json (frozen directions at the best layer/position).
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

FLOOR = 0.75  # held-out AUROC needed before a direction is used on debate states


def auroc(y, s):
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    if y.all() or (~y).all():
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def fit_logistic(X, y, l2=1.0):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1.0 / l2, max_iter=2000, class_weight="balanced")
    clf.fit(X, y)
    return clf.coef_[0].astype(np.float32), float(clf.intercept_[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dir", default="cptlit_core_n600")
    a = ap.parse_args()
    d = ACTS / a.model / a.dir
    W = pd.DataFrame(read_jsonl(d / "worlds.jsonl"))
    meta = json.load(open(d / "meta.json"))
    grid, POS = meta["grid"], meta["positions"]
    A = np.load(d / "acts.npy", mmap_mode="r")
    core = {r["id"]: r for r in load_core()}
    W["target"] = W.item_id.map(lambda i: item_view(core[i])["target"])
    W["half"] = W.item_id.map(lambda i: h64("probesplit", i) % 2)
    labels = {
        "advocates_target": (W.target.notna() & (W.adv == W.target), W.target.notna()),
        "agrees_with_prior": (W.adv == W.prior, pd.Series(True, index=W.index)),
        "strong_evidence": (W.qual == "strong", pd.Series(True, index=W.index)),
        "good_record": (W.record == "good", W.record.isin(["good", "bad"])),
        "high_status": (W.status == "prof", W.status.notna()),
    }
    rows, frozen = [], {}
    for name, (y_all, mask) in labels.items():
        idx_all = np.flatnonzero(mask.to_numpy())
        y = y_all.to_numpy(bool)
        for pi, pos in enumerate(POS):
            for gi, li in enumerate(grid):
                X = np.asarray(A[:, pi, gi], np.float32)
                s_dom = np.full(len(W), np.nan, np.float32)
                s_log = np.full(len(W), np.nan, np.float32)
                w_keep = None
                for h in (0, 1):
                    tr = idx_all[(W.half.to_numpy()[idx_all] == h)]
                    te = idx_all[(W.half.to_numpy()[idx_all] != h)]
                    if len(tr) < 50 or y[tr].all() or (~y[tr]).all():
                        continue
                    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
                    Z = (X[tr] - mu) / sd
                    w_dom = Z[y[tr]].mean(0) - Z[~y[tr]].mean(0)
                    w_log, b = fit_logistic(Z, y[tr])
                    Zt = (X[te] - mu) / sd
                    s_dom[te] = Zt @ w_dom
                    s_log[te] = Zt @ w_log + b
                    w_keep = dict(mu=mu.tolist(), sd=sd.tolist(), w_dom=w_dom.tolist(), w_log=w_log.tolist(), b=float(b))
                a_dom, a_log = auroc(y[idx_all], s_dom[idx_all]), auroc(y[idx_all], s_log[idx_all])
                rows.append(dict(label=name, position=pos, layer=li, auroc_dom=a_dom, auroc_log=a_log, n=len(idx_all)))
                if w_keep is not None:
                    frozen.setdefault(name, {})[f"{pos}|{li}"] = dict(position=pos, layer=li, grid_index=gi, pos_index=pi,
                                                                      auroc_dom=a_dom, auroc_log=a_log, **w_keep)
    T = pd.DataFrame(rows)
    T.to_csv(TABLES / f"lit_probes_{a.model}_auroc.csv", index=False)
    best = {}
    for name, g in T.groupby("label"):
        g = g.dropna(subset=["auroc_dom"])
        r = g.loc[g[["auroc_dom", "auroc_log"]].max(axis=1).idxmax()]
        key = f"{r.position}|{int(r.layer)}"
        best[name] = dict(frozen[name][key])
        best[name]["usable"] = bool(max(r.auroc_dom, r.auroc_log) >= FLOOR)
        print(f"{name:20s} best {r.position:10s} L{int(r.layer):2d}  AUROC dom {r.auroc_dom:.3f} log {r.auroc_log:.3f}  "
              f"{'USABLE' if best[name]['usable'] else 'below floor'}  n={int(r.n)}")
    json.dump(best, open(ACTS / a.model / "lit_probes.json", "w"))
    print(T.groupby(["label", "position"])[["auroc_dom", "auroc_log"]].max().round(3).to_string())


if __name__ == "__main__":
    main()
