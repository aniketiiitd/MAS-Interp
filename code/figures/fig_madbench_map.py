"""MAD Bench breakdown figure: the social-influence effects resolved across the 15 task families (grouped by answer regime).

Columns (each a forest over families; filled = 95% CI excludes 0):
  1 unanimity gain      final unanimity, debate - no communication (all families)
  2 accuracy gain       final majority correct, debate - majority vote of initial answers (families with a target)
  3 lock-in             ends unanimous on the wrong answer given a wrong initial majority, debate - no communication (target families)
  4 privacy effect      private - public re-query deviation after debate (all families)
  5 debate x privacy    private deviation, debate - no communication (all families)
Also writes outputs/tables/madbench_map_<M>.csv with every family estimate (used by the cross-hero heatmap).
Usage: python figures/fig_madbench_map.py Q9
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
CORE = {r["id"]: r for r in load_core()}
ORDER = [("knowledge_reasoning", ["verifiable_knowledge", "disciplinary_reasoning", "misconception_resistance", "calibrated_abstention"]),
         ("safety_refusal", ["direct_harmful_request", "unsafe_request", "contextual_safety_boundary", "over_refusal", "strategic_deception"]),
         ("values_moral_judgment", ["clear_moral_norms", "ethical_framework", "moral_tradeoffs", "public_policy", "social_influence"])]
FLAB = {"verifiable_knowledge": "verifiable knowledge", "disciplinary_reasoning": "disciplinary reasoning",
        "misconception_resistance": "misconception resistance", "calibrated_abstention": "calibrated abstention",
        "direct_harmful_request": "direct harmful requests", "unsafe_request": "unsafe-request contrasts",
        "contextual_safety_boundary": "contextual safety boundaries", "over_refusal": "over-refusal controls",
        "strategic_deception": "strategic deception", "clear_moral_norms": "clear moral norms", "ethical_framework": "ethical frameworks",
        "moral_tradeoffs": "moral trade-offs", "public_policy": "public-policy debate", "social_influence": "persuasion"}
DLAB = {"knowledge_reasoning": "Knowledge", "safety_refusal": "Safety", "values_moral_judgment": "Values"}


def ci(x, c):
    x = np.asarray(x, float)
    if np.isfinite(x).sum() < 8:
        return np.nan, np.nan, np.nan, 0
    e, lo, hi, p, n = fast_cluster_mean_ci(x, np.asarray(c))
    return 100 * e, 100 * lo, 100 * hi, n


def main(m):
    I = pd.read_csv(TABLES / f"lit_debate_{m}_lit1200_items.csv")
    Q = pd.read_csv(TABLES / f"lit_debate_{m}_lit1200_requery.csv")
    Q["family"] = Q.item_id.map(lambda i: family_of(CORE[i]))
    I = I[I.kind != "numeric"].copy()
    I["unan_gain"] = I.debate_unan.astype(float) - I.reflect_unan.astype(float)
    tgt = I.target.notna()
    I.loc[tgt, "acc_gain"] = (I.loc[tgt, "debate_maj"] == I.loc[tgt, "target"]).astype(float) - (I.loc[tgt, "vote"] == I.loc[tgt, "target"]).astype(float)
    wrong0 = tgt & I.vote.notna() & (I.vote != I.target)
    I.loc[wrong0, "lockin"] = ((I.debate_unan.astype(bool) & (I.debate_maj != I.target)).astype(float)
                               - (I.reflect_unan.astype(bool) & (I.reflect_maj != I.target)).astype(float))[wrong0]
    U = Q[Q.unanimous.astype(bool)]
    deb = U[U.cond == "debate"].copy()
    deb["priv_minus_pub"] = deb.dev_private - deb.dev_public
    A = U.groupby(["item_id", "cluster", "family", "cond"]).dev_private.mean().unstack("cond").dropna().reset_index()
    A["debate_x_priv"] = A.debate - A.reflect
    cols = [("unan_gain", I, "Unanimity\n(debate − no comm.)"), ("acc_gain", I, "Accuracy\n(debate − vote)"),
            ("lockin", I, "Wrong-majority lock-in\n(debate − no comm.)"), ("priv_minus_pub", deb, "Private − public\nanswer change"),
            ("debate_x_priv", A, "Private change\n(debate − no comm.)")]
    rows, table = [], []
    for dom, fams in ORDER:
        for f in fams:
            rows.append((dom, f))
    fig, axes = plt.subplots(1, len(cols), figsize=(10.2, 5.2), sharey=True, constrained_layout=True)
    for k, (col, df, title) in enumerate(cols):
        ax = axes[k]
        for ri, (dom, f) in enumerate(rows):
            g = df[df.family == f]
            e, lo, hi, n = ci(g[col], g.cluster) if col in g else (np.nan,) * 4
            table.append(dict(model=m, family=f, domain=dom, metric=col, est=e, lower=lo, upper=hi, n_clusters=n))
            y = len(rows) - 1 - ri
            if not np.isfinite(e):
                ax.text(0.5, y, "–", transform=ax.get_yaxis_transform(), ha="center", va="center", color=style.MUTED, fontsize=8)
                continue
            c = style.DOMAIN_COLOR[dom]
            sig = lo > 0 or hi < 0
            ax.plot([lo, hi], [y, y], color=c, lw=1.2, solid_capstyle="round")
            ax.scatter([e], [y], s=26, facecolor=c if sig else "white", edgecolor=c, lw=1.2, zorder=3)
        ax.axvline(0, color=style.SLATE, lw=0.8)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("percentage points", fontsize=8)
        # domain separators
        ys = np.cumsum([len(f) for _, f in ORDER])[:-1]
        for yb in ys:
            ax.axhline(len(rows) - yb - 0.5, color=style.GRID, lw=0.8)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([FLAB[f] for _, f in rows][::-1], fontsize=8.5)
    for (dom, fams), y0 in zip(ORDER, [0, 4, 9]):
        ymid = len(rows) - 1 - (y0 + (len(fams) - 1) / 2)
        axes[0].annotate(DLAB[dom], xy=(0, ymid), xycoords=("axes fraction", "data"), xytext=(-150, 0), textcoords="offset points",
                         rotation=90, va="center", ha="center", fontsize=9, color=style.DOMAIN_COLOR[dom])
    fig.suptitle("Where debate changes answers across MAD Bench task families (Qwen3.5-9B; filled: 95% CI excludes 0)", fontsize=10, x=0.02, ha="left")
    style.finish(fig, axes, ROOT / "outputs/figures" / f"MADBench_map_behaviour_{m}")
    pd.DataFrame(table).to_csv(TABLES / f"madbench_map_{m}.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Q9")
