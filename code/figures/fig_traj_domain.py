"""What debate does to a reader's affect depends on the kind of question, in all three models; the dynamics do not agree.

Rewritten 2026-09-20 after Gemma-4-E4B landed and cost the earlier version two of its three findings. The previous
figure showed Qwen alone, and its lead example was `pressure_from_numbers`, a concept Gemma's probes cannot
validate at all -- so the illustration was not merely unreplicated, it was unmeasurable in the second model.

Everything here is on the 104 concepts BOTH models validate. Gemma validates 105 against Qwen's 158, and a smaller,
differently selected concept set is a different instrument rather than the same one applied twice.

  (a) one concept, both models: the domain dependence itself, which replicates
  (b) reading site, spread across domains over twelve turns: Qwen converges, Gemma diverges
  (c) decision site, the same: Qwen diverges, Gemma is flat and twice as high

Panels b and c carry the shuffled-domain null as a shaded band, because a spread is a max minus a min and is
positive even when nothing is happening.

Out: outputs/figures/{,paper/}V3_traj_domain.{pdf,png}
"""
from __future__ import annotations

import sys

import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import *  # noqa
import style
import matplotlib.pyplot as plt

SHORT = {"knowledge_reasoning": "knowledge", "safety_refusal": "safety", "values_moral_judgment": "values"}
ORDER = ["knowledge_reasoning", "safety_refusal", "values_moral_judgment"]
# (model, table tag, display name). Qwen must use the shared-set run, not its own 158-concept one.
MODELS = [("Q9", "T12shared3", "Qwen"), ("L31", "T12shared3", "Llama"), ("G4", "T12shared3", "Gemma")]
# One encoding for the whole figure: COLOUR is the domain, LINE STYLE is the model. The first version coloured
# the models teal and terracotta in panels b and c, which are the knowledge and safety colours in panel a -- the
# same two hues carrying two different meanings three inches apart.
MSTYLE = {"Q9": dict(ls="-", marker="o"), "L31": dict(ls=(0, (1, 1.4)), marker="^"),
          "G4": dict(ls=(0, (5, 2)), marker="s")}
CONCEPT = "despondent"   # largest decision-site spread among the 99 all three models validate


SOURCES: list[str] = []


def table(name):
    """Read a table and record that we read it.

    The audit used to check which tables a figure reads by grepping its source, which works only while the
    filename is a literal. This one builds names by interpolation, so the grep passed without testing anything.
    A figure that writes down what it actually opened cannot be checked vacuously.
    """
    SOURCES.append(name)
    return pd.read_csv(TABLES / name)


def head(ax, title, sub):
    ax.text(0, 1.155, title, transform=ax.transAxes, fontsize=10.5, color=style.INK, va="baseline")
    ax.text(0, 1.035, sub, transform=ax.transAxes, fontsize=8.4, color=style.SLATE, va="baseline")


def confirmed(m, tag, site):
    S = table(f"traj_domain_{m}_{tag}_{site}_spread.csv")
    sel = set(S[(S.half == "discovery") & (S.q < 0.05)].concept)
    return S[(S.half == "confirmation") & S.concept.isin(sel) & (S.q < 0.05)]


def panel_spread(ax, site, title, sub, label_dy):
    for m, tag, name in MODELS:
        R = confirmed(m, tag, site)
        t = R.groupby("turn").spread.mean()
        ax.plot(t.index, t.values, color=style.INK, lw=2.0, ms=3.6, mec="white", mew=0.7,
                markevery=({"Q9": 0, "L31": 1, "G4": 2}[m], 3), zorder=3, **MSTYLE[m])
        nb = R.groupby("turn").null_p95.mean()
        ax.plot(nb.index, nb.values, color=style.MUTED, lw=1.1, ls=MSTYLE[m]["ls"], zorder=1)
        ax.annotate(name, xy=(t.index[-1], t.values[-1]), xytext=(-3, label_dy[m]),
                    textcoords="offset points", color=style.INK, fontsize=8.4, ha="right", zorder=4)
    head(ax, title, sub)
    ax.set_xlabel("debate turn")
    ax.set_xticks([1, 3, 5, 7, 9, 11])
    ax.set_ylim(bottom=0)


def main():
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4))

    A = axes[0]
    for m, tag, name in MODELS:
        C = table(f"traj_domain_{m}_{tag}_dec_curves.csv")
        C = C[(C.half == "confirmation") & (C.concept == CONCEPT)]
        for d in ORDER:
            s = C[C.domain == d].sort_values("turn")
            if not len(s):
                continue
            A.plot(s.turn, s.drift, color=style.DOMAIN_COLOR[d], lw=2.0, ms=3.4, mec="white", mew=0.6,
                   markevery=({"Q9": 0, "L31": 1, "G4": 2}[m], 3), zorder=3, **MSTYLE[m])
    A.axhline(0, color=style.INK, lw=0.9, ls=(0, (4, 3)), zorder=2)
    head(A, f"One concept, all three models: '{CONCEPT}'",
         "colour is the domain; solid Qwen, dotted Llama, dashed Gemma")
    A.set_xlabel("debate turn"); A.set_ylabel("debate $-$ thinking alone  (z)")
    A.set_xticks([1, 3, 5, 7, 9, 11])
    for d in ORDER:
        A.plot([], [], color=style.DOMAIN_COLOR[d], lw=2.0, label=SHORT[d])
    A.legend(frameon=False, loc="center", bbox_to_anchor=(0.5, 0.40), ncol=3, handlelength=1.0,
             columnspacing=0.8, handletextpad=0.4, borderaxespad=0.0, fontsize=7.0)

    panel_spread(axes[1], "read", "While reading the peers", "Qwen converges, Gemma diverges, Llama neither",
                 {"Q9": 11, "L31": -16, "G4": 9})
    panel_spread(axes[2], "dec", "While deciding", "only Qwen moves",
                 {"Q9": 9, "L31": -15, "G4": -15})
    axes[1].set_ylabel("spread across domains  (z)")
    axes[2].text(0.97, 0.05, "grey: shuffled-domain null (95th pct)", transform=axes[2].transAxes,
                 ha="right", fontsize=7.6, color=style.SLATE)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=2.2)
    for d in (FIGS, FIGS / "paper"):
        for ext in ("pdf", "png"):
            fig.savefig(d / f"V3_traj_domain.{ext}", dpi=220, bbox_inches="tight", facecolor="white")
    (FIGS / "paper" / "V3_traj_domain.sources.txt").write_text("\n".join(sorted(set(SOURCES))) + "\n")
    print("[done] V3_traj_domain.pdf (three models, shared 99 concepts)")
    print("       read: " + ", ".join(sorted(set(SOURCES))))


if __name__ == "__main__":
    main()
