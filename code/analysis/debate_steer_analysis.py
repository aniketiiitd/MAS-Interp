"""C3 gate evaluation for natural-debate steering (confirmation half).
Per direction x control (dir/random/orth) x episode (wrong_majority/right_majority/normative) x alpha: mean change in the
adoption margin toward the united majority, item-cluster bootstrap CIs. Selectivity = change(wrong) - change(right).
Pre-registered gate: for a direction, (i) dose-response slope on wrong-majority episodes with CI excluding 0,
(ii) that slope's CI does not overlap the random and orthogonal controls' slope CIs, (iii) selectivity at the protective
extreme dose (the alpha that lowers wrong adoption) has CI excluding 0 in the protective direction, (iv) mean label mass >= 0.95."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import *  # noqa
from stats import fast_cluster_mean_ci


def boot_slope(df, B=1000, seed=240091):
    by = {i: g for i, g in df.groupby("item_id")}
    items = np.array(list(by))
    rng = np.random.default_rng(seed)
    est = float(np.polyfit(df.alpha, df.d_m_adopt, 1)[0])
    bs = []
    for _ in range(B):
        d = pd.concat([by[i] for i in rng.choice(items, len(items))])
        bs.append(np.polyfit(d.alpha, d.d_m_adopt, 1)[0])
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return est, float(lo), float(hi)


def main(model="Q9", tag=""):
    S = pd.DataFrame(read_jsonl(ACTS / model / f"debate_steer{tag}" / "steer.jsonl"))
    S = S[S.ctrl != "base"]
    cells, slopes = [], []
    for (dirn, ctrl, ep, al), g in S.groupby(["direction", "ctrl", "episode", "alpha"]):
        per = g.groupby("item_id").d_m_adopt.mean()
        est, lo, hi, p, n = fast_cluster_mean_ci(per.to_numpy(), per.index.to_numpy())
        cells.append(dict(direction=dirn, ctrl=ctrl, episode=ep, alpha=al, d_m_adopt=est, lower=lo, upper=hi, n_items=n,
                          label_mass=g.label_mass.mean(), kl=g.kl.mean()))
    C = pd.DataFrame(cells)
    for (dirn, ctrl, ep), g in S.groupby(["direction", "ctrl", "episode"]):
        est, lo, hi = boot_slope(g)
        slopes.append(dict(direction=dirn, ctrl=ctrl, episode=ep, slope=est, lower=lo, upper=hi))
    Sl = pd.DataFrame(slopes)
    gate = []
    for dirn in S.direction.unique():
        w = Sl[(Sl.direction == dirn) & (Sl.episode == "wrong_majority")].set_index("ctrl")
        if "dir" not in w.index:
            continue
        d = w.loc["dir"]
        sig = d.lower > 0 or d.upper < 0
        sep = all((d.lower > w.loc[c].upper) or (d.upper < w.loc[c].lower) for c in ("random", "orth") if c in w.index)
        prot_alpha = min(S.alpha) if d.slope > 0 else max(S.alpha)  # the dose that lowers wrong-majority adoption
        sel = []
        g = S[(S.direction == dirn) & (S.ctrl == "dir") & (S.alpha == prot_alpha)]
        piv = g.pivot_table(index="item_id", columns="episode", values="d_m_adopt", aggfunc="mean")
        if {"wrong_majority", "right_majority"} <= set(piv.columns):
            # items differ across episode types, so selectivity CI uses independent bootstraps of the two means
            wv = piv.wrong_majority.dropna().to_numpy()
            rv = piv.right_majority.dropna().to_numpy()
            rng = np.random.default_rng(240091)
            bs = [rng.choice(wv, len(wv)).mean() - rng.choice(rv, len(rv)).mean() for _ in range(2000)]
            sel = [float(wv.mean() - rv.mean()), *np.quantile(bs, [0.025, 0.975])]
        protective = bool(sel) and sel[2] < 0
        lm = C[(C.direction == dirn) & (C.ctrl == "dir")].label_mass.min()
        gate.append(dict(direction=dirn, wrong_slope=d.slope, slope_ci=(d.lower, d.upper), slope_sig=sig, beats_controls=sep,
                         protective_alpha=prot_alpha, selectivity=sel, protective=protective, min_label_mass=lm,
                         PASS=bool(sig and sep and protective and lm >= 0.95)))
    C.to_csv(TABLES / f"debate_steer_cells_{model}{tag}.csv", index=False)
    Sl.to_csv(TABLES / f"debate_steer_slopes_{model}{tag}.csv", index=False)
    G = pd.DataFrame(gate)
    G.to_csv(TABLES / f"debate_steer_gate_{model}{tag}.csv", index=False)
    pd.set_option("display.width", 250)
    print(G.to_string())
    print(Sl[Sl.episode == "wrong_majority"].round(3).to_string())


if __name__ == "__main__":
    main(tag=sys.argv[1] if len(sys.argv) > 1 else "")
