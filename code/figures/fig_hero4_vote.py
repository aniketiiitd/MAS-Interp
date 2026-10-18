"""Hero figure 4: debate buys a little accuracy and costs a lot of false certainty.

Visual form: a 2 x 2 grid of slopegraphs, rows = models, columns = the two ledger entries. Left column: the accuracy
of a plain majority vote over the agents' initial answers against the majority after three rounds of debate. Right
column: how often the group ends unanimous on a WRONG answer, after three rounds without communication against three
rounds of debate (a vote has no notion of unanimity, so the no-communication run is the matched baseline). One line
per task family plus the pooled line; every endpoint carries its number, and the pooled line carries its difference.
The right panels also state the conditional number: when the initial majority was wrong, how often debate ends
unanimous on it.

Reads the FULL-benchmark vote tables at the fully-parsed scope, which is what the claim quotes. It read the
half-benchmark tables at every scope until 2026-09-19, so the figure showed the pre-correction +3.0 point vote
gain while the claim beside it said +1.2: the gain shrank when unparseable answers stopped being scored as wrong,
and the vote arm, built from the worst-parsing round, had been penalised most.
Usage: python figures/fig_hero4_vote.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/figures"))
import style  # noqa: E402
from style import plt  # noqa: E402

TABLES = ROOT / "outputs/tables"
OUT = ROOT / "outputs/figures/paper"
MNAME = {"Q9": "Qwen3.5-9B", "L31": "Llama-3.1-8B"}
DOMS = [("all", "all items", style.INK, "o", 1.5),
        ("knowledge_reasoning", "knowledge", style.TEAL, "s", 1.0),
        ("safety_refusal", "safety", style.TERRACOTTA, "^", 1.0),
        ("values_moral_judgment", "values", style.PLUM, "D", 1.0)]


# Gemma ran the benchmark in one pass; the other two were pooled from halves and carry the _full2400 name.
VOTE = {"Q9": "lit_debate_Q9_full2400_vote.csv", "L31": "lit_debate_L31_full2400_vote.csv",
        "G4": "lit_debate_G4_core_vote.csv"}


def get(V, dom, metric):
    r = V[(V.domain == dom) & (V.metric == metric)].iloc[0]
    return 100 * r.est, 100 * r.lower, 100 * r.upper


def load_vote(m):
    """The fully-parsed scope: both arms measured on the questions where every agent actually answered."""
    V = pd.read_csv(TABLES / VOTE[m])
    if "scope" in V.columns:
        V = V[V.scope == "fully_parsed"]
        if not len(V):
            raise SystemExit(f"[fig_hero4_vote] {VOTE[m]} has no fully_parsed rows")
    return V


def spread(ys, gap):
    ys = list(ys)
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    s = [ys[i] for i in order]
    for i in range(1, len(s)):
        s[i] = max(s[i], s[i - 1] + gap)
    for i in range(len(s) - 2, -1, -1):
        s[i] = min(s[i], s[i + 1] - gap)
    out = [0.0] * len(ys)
    for k, i in enumerate(order):
        out[i] = s[k]
    return out


def panel(ax, V, m0, m1, delta_metric, ylo, yhi, gap, xt, first_col, note=None):
    left, right = [], []
    for dom, lab, c, mk, lw in DOMS:
        e0, l0, h0 = get(V, dom, m0)
        e1, l1, h1 = get(V, dom, m1)
        ax.plot([0, 1], [e0, e1], color=c, lw=lw, zorder=3)
        for x, (e, lo, hi) in ((0, (e0, l0, h0)), (1, (e1, l1, h1))):
            ax.plot([x, x], [lo, hi], color=c, lw=0.6, zorder=2, alpha=0.8)
        ax.plot([0, 1], [e0, e1], marker=mk, ms=3.6, color=c, ls="", zorder=4, markeredgecolor="white",
                markeredgewidth=0.5)
        left.append((e0, f"{e0:.1f}", c, dom == "all"))
        if dom == "all":
            d, dl, dh = get(V, "all", delta_metric)
            right.append((e1, f"{e1:.1f}  {lab}  ({d:+.1f} pts)", c, True))
        else:
            right.append((e1, f"{e1:.1f}  {lab}", c, False))
    gap = (yhi - ylo) / 11.5  # label pitch: about 0.1 in on a 1.15 in panel, whatever the data range
    for x, L, ha in ((-0.06, left, "right"), (1.06, right, "left")):
        ys = spread([l[0] for l in L], gap)
        for (y0, t, c, bold), y in zip(L, ys):
            ax.text(x, y, t, ha=ha, va="center", fontsize=6.2, color=c, fontweight="bold" if bold else "normal")
    if note:
        # This line carries the sharpest number on the panel, so it is set in ink on a gold band rather than in grey:
        # at 5.8pt slate it read as faded background text.
        ax.text(0.5, yhi, note, ha="center", va="top", fontsize=6.6, color=style.INK, linespacing=1.2, zorder=6)
    ax.set_xlim(-0.5, 2.35)
    ax.set_ylim(ylo, yhi)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(xt, fontsize=6.4)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", length=0)


def main():
    style.apply_paper()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(style.TEXTWIDTH, 3.15), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.0, 1.0]))
    for ri, m in enumerate(("Q9", "L31")):
        V = load_vote(m)
        aL, aR = axes[ri]
        # accuracy ledger
        vals = [get(V, d, k)[0] for d, *_ in DOMS for k in ("acc_vote", "acc_debate_maj")]
        span = max(vals) - min(vals)
        gap = max(2.2, span / 9)
        panel(aL, V, "acc_vote", "acc_debate_maj", "debate - vote", min(vals) - 0.5 * span - gap, max(vals) + 0.55 * span + gap,
              gap, ["vote of the\ninitial answers", "majority after\n3 rounds of debate"], True)
        # false-certainty ledger
        vals = [get(V, d, k)[0] for d, *_ in DOMS for k in ("wrong_unan_reflect", "wrong_unan_debate")]
        span = max(vals) - min(vals)
        gap = max(1.6, span / 9)
        w0 = get(V, "all", "wrong_unan_reflect | initial majority wrong")[0]
        w1 = get(V, "all", "wrong_unan_debate | initial majority wrong")[0]
        note = f"when the initial majority was wrong:\n{w0:.0f}% → {w1:.0f}% end unanimous on it"
        panel(aR, V, "wrong_unan_reflect", "wrong_unan_debate", "wrong_unan debate - reflect",
              min(vals) - 0.35 * span - gap, max(vals) + 0.9 * span + 2.4 * gap, gap,
              ["3 rounds,\nno communication", "3 rounds\nof debate"], False, note=note)
        aL.set_title(f"{'ab'[ri]}  {MNAME[m]}: final majority correct (%)", loc="left")
        aR.set_title(f"{'cd'[ri]}  {MNAME[m]}: ends unanimous on a wrong answer (%)", loc="left")
    style.finish_paper(fig, axes, str(OUT / "fig_hero4_vote"))
    print("wrote", OUT / "fig_hero4_vote.png")


if __name__ == "__main__":
    main()
