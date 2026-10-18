"""What a group of agents does as the disagreement drags on, turn by turn.

Every panel is debate against the no-communication condition run on the same questions for the same number of turns, so
anything that comes merely from thinking longer is visible as the second line rather than hidden.

  a  how often an agent changes its answer
  b  how often the group ends the turn agreeing
  c  how often the group agrees on an answer that is wrong

The literature stops at two to four turns and reports that accuracy is flat after that, so flat accuracy here would be a
replication, not a finding. What these panels can show that a three-turn run cannot is whether agreement keeps hardening
after the answers have stopped moving.

Three models since 2026-09-20, when Llama's and then Gemma's twelve-turn runs landed. Claim 5 became a two-model claim over twelve
turns and this figure still showed one, which is the third time in this campaign that a claim gained a model and
its figure did not. Colour is the condition and line style is the model, so colour keeps meaning one thing.

Usage: python figures/fig_turns.py
"""
from __future__ import annotations

import sys

import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
import style  # noqa: E402
from style import plt  # noqa: E402
from common import TABLES  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "outputs/figures/paper"
S_, M_ = style.SLATE, style.MUTED
DEB, SOLO = style.TERRACOTTA, style.TEAL

# (model, dataset tag, display name, line style, marker)
MODELS = [("Q9", "release_600_t12", "Qwen", "-", "o"),
          ("L31", "release_600_t12", "Llama", (0, (4, 2)), "s"),
          ("G4", "release_600_t12", "Gemma", (0, (1, 1.4)), "^")]

PANELS = [
    ("switch_rate", "a  Agents that change their answer", "share of agents", True),
    ("unanimity", "b  Groups that agree", "share of groups", True),
    ("wrong_unanimity", "c  Groups agreeing on a wrong answer", "share of groups", True),
]


def main():
    style.apply_paper()
    T = {m: pd.read_csv(TABLES / f"lit_turns_{m}_{tg}.csv") for m, tg, _, _, _ in MODELS}
    rmax = max(int(t["round"].max()) for t in T.values())
    # Two by two rather than a strip of three: the fourth panel is what the first three are evidence for, and four
    # across one text width leaves no room for axis labels.
    fig, axes = plt.subplots(2, 2, figsize=(style.TEXTWIDTH, 3.35), constrained_layout=True)
    ax = axes.ravel()
    for i, (metric, title, ylab, pct) in enumerate(PANELS):
        for m, tg, name, ls, mk in MODELS:
            Tm = T[m]
            g = Tm[(Tm.metric == metric) & (Tm.get("scope", "contested") == "contested")] if "scope" in Tm.columns \
                else Tm[Tm.metric == metric]
            for cond, col in (("debate", DEB), ("reflect", SOLO)):
                d = g[g.cond == cond].sort_values("round")
                if not len(d):
                    continue
                k = 100 if pct else 1
                ax[i].plot(d["round"], k * d.est, color=col, marker=mk, ms=2.6, lw=1.2, ls=ls,
                           markevery=({"Q9": 0, "L31": 1, "G4": 2}[m], 3))
                # bands only for the first model: four overlapping ribbons is a wash, and the second model's
                # lines are the comparison rather than a separate estimate to read off
                if m == "Q9":
                    ax[i].fill_between(d["round"], k * d.lower, k * d.upper, color=col, alpha=0.14, lw=0)
        ax[i].set_xlabel("turn")
        ax[i].set_ylabel(ylab + (" (%)" if pct else ""))
        ax[i].set_title(title, loc="left")
        ax[i].set_xlim(0.5, rmax + 0.5)

    A = ax[3]
    for m, tg, name, ls, mk in MODELS:
        ip = TABLES / f"lit_independence_{m}_{tg}.csv"
        if not ip.exists():
            continue
        I = pd.read_csv(ip)
        I = I[I.scope == "split_at_start"] if "scope" in I.columns else I
        for cond, col in (("debate", DEB), ("reflect", SOLO)):
            d = I[(I.cond == cond) & (I["round"] > 0)].sort_values("round")
            if not len(d):
                continue
            A.plot(d["round"], d.n_eff, color=col, marker=mk, ms=2.6, lw=1.2, ls=ls,
                   markevery=({"Q9": 0, "L31": 1, "G4": 2}[m], 3))
            if m == "Q9":
                lo = d[["n_eff_lo", "n_eff_hi"]].min(axis=1)
                hi = d[["n_eff_lo", "n_eff_hi"]].max(axis=1)
                A.fill_between(d["round"], lo, hi, color=col, alpha=0.14, lw=0)
    A.axhline(3, color=style.MUTED, lw=0.7, ls=(0, (2, 2)))
    A.text(rmax + 0.35, 3.0, "three independent agents", fontsize=5.8, color=style.MUTED, va="bottom", ha="right")
    A.axhline(1, color=style.INK, lw=0.7)
    A.text(rmax + 0.35, 1.02, "one agent", fontsize=5.8, color=style.INK, va="bottom", ha="right")
    A.set_ylim(0.6, 3.62)
    # Qwen's silent arm at turn 1 has an upper bound of 5.3, far outside any scale that leaves the rest legible.
    # Clipping is fine; clipping without saying so is not.
    A.text(0.98, 0.985, "turn-1 interval for thinking alone reaches 5.3", transform=A.transAxes,
           fontsize=5.2, color=M_, ha="right", va="top")
    A.set_xlim(0.5, rmax + 0.5)
    A.set_xlabel("turn")
    A.set_ylabel("independent agents")
    A.set_title("d  Independent opinions left in the group", loc="left")

    from matplotlib.lines import Line2D
    ax[0].legend(handles=[Line2D([], [], color=DEB, lw=1.3, label="debate"),
                          Line2D([], [], color=SOLO, lw=1.3, label="thinking alone")],
                 loc="upper right", fontsize=6.0, frameon=False, labelspacing=0.22, handletextpad=0.4,
                 handlelength=1.4)
    ax[1].legend(handles=[Line2D([], [], color=S_, lw=1.3, ls=ls, marker=mk, ms=2.6, label=name)
                          for _, _, name, ls, mk in MODELS],
                 loc="lower right", fontsize=5.6, frameon=False, labelspacing=0.18, handletextpad=0.4,
                 handlelength=2.0, bbox_to_anchor=(1.0, -0.04))
    style.finish_paper(fig, ax, f"{OUT}/fig_turns_three")
    print(f"wrote {OUT}/fig_turns_three.png  ({rmax} turns, {len(MODELS)} models)")


if __name__ == "__main__":
    main()
