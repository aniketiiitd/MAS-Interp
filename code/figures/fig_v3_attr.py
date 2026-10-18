"""V3 attribution-counterfactual figures (analysis/attribution.py tables).

  own: whose words — Delta_own by ground-truth regime (discovery hollow, confirmation filled) + order/names controls + families
  aud: who will hear — private/public x context/transcript-free switch rates and debate - pool gaps by regime
Usage: python figures/fig_v3_attr.py own Q9 [release_600,core]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import *  # noqa
import style
from style import plt
from stats import fast_cluster_mean_ci

style.apply()
FIG = ROOT / "outputs/figures"
REG = ["knowledge_reasoning", "values_moral_judgment", "safety_refusal"]
RLAB = {"knowledge_reasoning": "Knowledge\n(objective target)", "values_moral_judgment": "Values\n(no target)",
        "safety_refusal": "Safety\n(behavioural target)"}
DSLAB = {"release_600": "release-600 (discovery)", "core": "core (confirmation)"}


def ci_h(ax, y, est, lo, hi, color, marker="o", filled=True, size=30, lw=1.3, label=None):
    ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="round", zorder=2)
    ax.scatter([est], [y], marker=marker, s=size, facecolor=color if filled else "white", edgecolor=color, lw=1.2, zorder=3, label=label)


def fig_own(model, datasets):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.0, 1.25]))
    ax = axes[0]
    for di, ds in enumerate(datasets):
        p = TABLES / f"v3_own_summary_{model}_{ds}.csv"
        if not p.exists():
            continue
        S = pd.read_csv(p)
        filled = ds == "core" or len(datasets) == 1
        for ri, reg in enumerate(REG):
            y0 = len(REG) - 1 - ri
            off = 0.16 if di == 0 else -0.16
            for metric, c, m, dy in (("d_own", style.DOMAIN_COLOR[reg], "o", 0.0), ("d_order", style.MUTED, "s", 0.1),
                                     ("d_names", style.MUTED, "D", -0.1)):
                r = S[(S.scope == f"domain={reg}") & (S.metric == metric)]
                if not len(r):
                    continue
                r = r.iloc[0]
                size = 34 if metric == "d_own" else 14
                ci_h(ax, y0 + off + (dy if metric != "d_own" else 0), r.est, r.lower, r.upper, c, m, filled, size=size,
                     lw=1.3 if metric == "d_own" else 0.8)
    ax.axvline(0, color=style.SLATE, lw=0.8)
    ax.set_yticks(range(len(REG)))
    ax.set_yticklabels([RLAB[r] for r in REG][::-1])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(r"$\Delta_{own}$: adoption when the text is labelled 'yours'" "\nminus when the same text is a peer's (logits)")
    ax.set_title("a  Identical text, different author label", loc="left")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=style.INK, markerfacecolor="white" if len(datasets) > 1 else style.INK,
                label=DSLAB.get(datasets[0], datasets[0]))]
    if len(datasets) > 1:
        h.append(Line2D([], [], marker="o", ls="", color=style.INK, label=DSLAB.get(datasets[1], datasets[1])))
    h += [Line2D([], [], marker="s", ls="", color=style.MUTED, markersize=4, label="order control"),
          Line2D([], [], marker="D", ls="", color=style.MUTED, markersize=4, label="name control")]
    fig.legend(handles=h, loc="outside lower left", ncols=len(h), fontsize=7.5, handletextpad=0.3)
    # b: families (last dataset)
    ax = axes[1]
    ds = datasets[-1]
    E = pd.read_csv(TABLES / f"v3_own_episodes_{model}_{ds}.csv")
    rows = []
    for (reg, fam), g in E.groupby(["domain", "family"]):
        if g.cluster.nunique() < 3:
            continue
        e, lo, hi, p, n = fast_cluster_mean_ci(g.d_own.to_numpy(), g.cluster.to_numpy())
        rows.append(dict(reg=reg, fam=fam, est=e, lo=lo, hi=hi, n=n))
    F = pd.DataFrame(rows)
    F["ri"] = F.reg.map({r: i for i, r in enumerate(REG)})
    F = F.sort_values(["ri", "est"], ascending=[False, True]).reset_index(drop=True)
    for i, r in F.iterrows():
        ci_h(ax, i, r.est, r.lo, r.hi, style.DOMAIN_COLOR[r.reg], size=22, lw=1.0)
        ax.text(ax.get_xlim()[1] if False else 1.0, i, f" n={r.n}", transform=ax.get_yaxis_transform(), va="center", fontsize=7,
                color=style.SLATE)
    ax.axvline(0, color=style.SLATE, lw=0.8)
    ax.set_yticks(range(len(F)))
    ax.set_yticklabels([f.replace("_", " ") for f in F.fam], fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(r"$\Delta_{own}$ (logits)")
    ax.set_title(f"b  By task family, {DSLAB.get(ds, ds)}", loc="left")
    style.finish(fig, axes, FIG / f"V3_C2_ownership_{model}_{'_'.join(datasets)}")


FRAMES = [("priv_ctx", "private,\npeers visible"), ("pub_ctx", "public,\npeers visible"), ("priv_self", "private,\ntranscript-free"),
          ("pub_self", "public,\ntranscript-free")]


def fig_aud(model, dataset):
    W = pd.read_csv(TABLES / f"v3_aud_episodes_{model}_{dataset}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.0, 1.25]))
    ax = axes[0]
    U = W[W.unanimous == True]
    for ci_, (cond, c) in enumerate((("debate", style.TERRACOTTA), ("pool", style.TEAL))):
        g = U[U.cond == cond]
        for fi, (f, lab) in enumerate(FRAMES):
            e, lo, hi, p, n = fast_cluster_mean_ci(100 * g[f"switch_{f}"].to_numpy(), g.cluster.to_numpy())
            ci_h(ax, len(FRAMES) - 1 - fi + (0.14 if ci_ == 0 else -0.14), e, lo, hi, c,
                 label=("debate (12 turns)" if cond == "debate" else "independent pool") if fi == 0 else None)
    ax.set_yticks(range(len(FRAMES)))
    ax.set_yticklabels([l for _, l in FRAMES][::-1], fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Final answer differs from own public answer (%)")
    ax.set_title("a  Re-query framing (unanimous groups)", loc="left")
    ax.legend(loc="lower right", fontsize=7.5)
    ax = axes[1]
    for ri, reg in enumerate(["ALL"] + REG):
        g = U if reg == "ALL" else U[U.domain == reg]
        A = g.groupby(["item_id", "cluster", "cond"])[[f"switch_{f}" for f, _ in FRAMES]].mean().unstack("cond").dropna()
        cl = np.asarray(A.index.get_level_values("cluster"))
        for fi, (f, lab) in enumerate(FRAMES[:3]):
            d = 100 * (A[(f"switch_{f}", "debate")] - A[(f"switch_{f}", "pool")])
            e, lo, hi, p, n = fast_cluster_mean_ci(d.to_numpy(), cl)
            y = 3 - ri + 0.22 - 0.22 * fi
            ci_h(ax, y, e, lo, hi, [style.PLUM, style.SLATE, style.MUTED][fi], ["o", "s", "D"][fi], fi == 0,
                 label=lab.replace("\n", " ") if ri == 0 else None)
    ax.axvline(0, color=style.SLATE, lw=0.8)
    ax.set_yticks(range(4))
    ax.set_yticklabels(["All items"] + [RLAB[r] for r in REG][::-1] if False else ["All items", *[RLAB[r] for r in REG]][::-1], fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Debate − pool difference (percentage points)")
    ax.set_title(f"b  Where 'private dissent' comes from, {DSLAB.get(dataset, dataset)}", loc="left")
    ax.legend(loc="lower right", fontsize=7.5)
    style.finish(fig, axes, FIG / f"V3_C4_audience_{model}_{dataset}")


if __name__ == "__main__":
    kind, model = sys.argv[1], sys.argv[2]
    ds = sys.argv[3].split(",") if len(sys.argv) > 3 else ["release_600"]
    if kind == "own":
        fig_own(model, ds)
    else:
        fig_aud(model, ds[0])
