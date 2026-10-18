"""Paper figures as horizontal strips at the ICLR text width (style.apply_paper). Reads the analysis tables; no recomputation
except cluster-bootstrap CIs for per-group means.
Usage: python figures/paper_figs.py Q9 [source commit vote audience carrier map]
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

style.apply_paper()
OUT = ROOT / "outputs/figures/paper"
OUT.mkdir(parents=True, exist_ok=True)
CORE = {r["id"]: r for r in load_core()}
MNAME = {"Q9": "Qwen3.5-9B", "L31": "Llama-3.1-8B", "G4": "Gemma-4-E4B"}
NL = {"Q9": 32, "L31": 32, "G4": 42}
DOMS = [("knowledge_reasoning", "Knowledge"), ("safety_refusal", "Safety"), ("values_moral_judgment", "Values")]
T_, R_, P_, S_, M_ = style.TERRACOTTA, style.TEAL, style.PLUM, style.SLATE, style.MUTED
FAMS = [("knowledge_reasoning", ["verifiable_knowledge", "disciplinary_reasoning", "misconception_resistance", "calibrated_abstention"]),
        ("safety_refusal", ["direct_harmful_request", "unsafe_request", "contextual_safety_boundary", "over_refusal", "strategic_deception"]),
        ("values_moral_judgment", ["clear_moral_norms", "ethical_framework", "moral_tradeoffs", "public_policy", "social_influence"])]
FLAB = {"verifiable_knowledge": "verifiable knowledge", "disciplinary_reasoning": "disciplinary reasoning",
        "misconception_resistance": "misconceptions", "calibrated_abstention": "abstention", "direct_harmful_request": "harmful requests",
        "unsafe_request": "unsafe contrasts", "contextual_safety_boundary": "safety boundaries", "over_refusal": "over-refusal",
        "strategic_deception": "deception", "clear_moral_norms": "clear moral norms", "ethical_framework": "ethical frameworks",
        "moral_tradeoffs": "moral trade-offs", "public_policy": "public policy", "social_influence": "persuasion"}


def ci(x, c, s=1.0):
    x = np.asarray(x, float)
    if np.isfinite(x).sum() < 5:
        return np.nan, np.nan, np.nan
    e, lo, hi, *_ = fast_cluster_mean_ci(x, np.asarray(c))
    return s * e, s * lo, s * hi


def hdot(ax, y, e, lo, hi, c, m="o", filled=True, label=None, s=14, lw=1.0):
    ax.plot([lo, hi], [y, y], color=c, lw=lw, solid_capstyle="round", zorder=2)
    ax.scatter([e], [y], s=s, marker=m, facecolor=c if filled else "white", edgecolor=c, lw=0.9, zorder=3, label=label)


def tidy_y(ax, labels, legend_lane=0.0):
    """legend_lane > 0 adds that many empty row-heights below the last category so a legend never sits on the data."""
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels[::-1])
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6 - legend_lane, len(labels) - 0.4)


# ---------------------------------------------------------------------------------------------------------------------------
def fig_source(m):
    C = pd.read_csv(TABLES / f"lit_source_{m}_cells.csv").set_index("source")
    E = pd.read_csv(TABLES / f"lit_source_{m}_effects.csv").set_index("effect")
    RA = pd.read_csv(TABLES / f"lit_source_{m}_readout_alignment.csv")
    fig, ax = style.strip(3, 1.78, [0.88, 0.92, 1.18])
    recs = [("good", "strong record"), ("unst", "unknown record"), ("bad", "poor record")]
    for si, (s, lab, c, mk) in enumerate((("prof", "professor", P_, "o"), ("ord", "ordinary member", R_, "s"))):
        for ri, (r, _) in enumerate(recs):
            row = C.loc[f"{s}_{r}"]
            hdot(ax[0], 2 - ri + (0.14 if si == 0 else -0.14), row.est, row.lower, row.upper, c, mk, label=lab if ri == 0 else None)
    ax[0].axvline(0, color=S_, lw=0.6)
    tidy_y(ax[0], [l for _, l in recs])
    ax[0].set_xlabel("shift toward the message vs. no label (logits)")
    ax[0].set_title("a  Who said it", loc="left")
    ax[0].legend(loc="lower right", fontsize=5.6, labelspacing=0.3)
    R = RA[RA.layer >= 9].copy()
    R["depth"] = R.layer / (NL[m] - 1)
    b = E.loc["status share (behaviour)"]
    ax[1].axhspan(b.lower, b.upper, color=T_, alpha=0.2, lw=0)
    ax[1].axhline(b.est, color=T_, lw=0.9, ls="--", label="behaviour")
    ax[1].plot(R.depth, R.norm_ratio_status_over_record / (1 + R.norm_ratio_status_over_record), color=P_, marker="o", ms=2.8,
               label="encoded (code size)")
    ax[1].plot(R.depth, R.readout_share_status, color=R_, marker="s", ms=2.8, label="used by the decision")
    ax[1].set_ylim(0, 0.95)
    ax[1].set_xlim(0.26, 1.02)
    ax[1].set_xlabel("relative depth")
    ax[1].set_ylabel("status share (status : record)")
    ax[1].set_title("b  Encoded vs used", loc="left")
    ax[1].legend(loc="upper right", fontsize=5.5, labelspacing=0.3)
    # c: the magnitude control. Every arm is the raw shift in log p(advocated) - log p(prior) from the interchange at
    # the last layer, so the natural and magnitude-matched swaps sit on one scale. (The depth-of-integration curves that
    # used to occupy this panel are not a claim we make -- that result is published.)
    MG = pd.read_csv(TABLES / f"lit_source_{m}_magnitude.csv")
    MG["arm"] = MG.arm.str.replace("random direction, record magnitude", "random direction", regex=False)
    bars = [("record swap, as written", R_, "o"), ("record, shrunk to\nstatus magnitude", R_, "s"),
            ("status, amplified to\nrecord magnitude", P_, "D"), ("status swap, as written", P_, "o"),
            ("random direction", M_, "v")]
    for k, (lab, c, mk) in enumerate(bars):
        key = lab.replace("\n", " ")
        r = MG[MG.arm == key]
        if not len(r):
            continue
        r = r.iloc[0]
        hdot(ax[2], len(bars) - 1 - k, r.est, r.lower, r.upper, c, mk, mk != "v", s=16)
    ax[2].axvline(0, color=S_, lw=0.6)
    ax[2].set_yticks(range(len(bars)))
    ax[2].set_yticklabels([lab for lab, _, _ in bars][::-1])
    ax[2].grid(axis="y", visible=False)
    ax[2].set_ylim(-0.6, len(bars) - 0.4)
    ax[2].set_xlabel("shift produced by the swap (logits)")
    ax[2].set_title("c  Not a matter of size", loc="left")
    style.finish_paper(fig, ax, OUT / f"fig_source_{m}")


def fig_commit(m):
    T = pd.read_csv(TABLES / f"lit_commit_{m}_curves.csv")
    F = pd.read_csv(TABLES / f"lit_commit_{m}_family_fraction0.csv")
    fig, ax = style.strip(3, 1.8, [1.10, 0.80, 0.80])
    for g, c, mk, ls, lab in (("changed answer in debate", T_, "o", "-", "changed, debate"),
                              ("changed answer without communication", R_, "s", "-", "changed, no communication"),
                              ("kept answer in debate", T_, "o", ":", "kept, debate"),
                              ("kept answer without communication", R_, "s", ":", "kept, no communication")):
        d = T[(T.group == g) & (T.domain == "all") & (T.metric == "argmax_is_final")].sort_values("fraction")
        ax[0].plot(100 * d.fraction, 100 * d.est, color=c, ls=ls, marker=mk, ms=2.8, label=lab)
        ax[0].fill_between(100 * d.fraction, 100 * d.lower, 100 * d.upper, color=c, alpha=0.15, lw=0)
    ax[0].set_ylim(0, 104)
    ax[0].set_xlabel("own reasoning written so far (%)")
    ax[0].set_ylabel("forced answer = final answer (%)")
    ax[0].set_title("a  When is the answer chosen?", loc="left")
    ax[0].legend(loc="lower left", bbox_to_anchor=(0.2, 0.0), fontsize=5.1, handlelength=1.6, markerscale=0.8, labelspacing=0.25,
                 borderaxespad=0.3)
    rows = [("all", "all")] + DOMS
    for ri, (d, lab) in enumerate(rows):
        for k, (g, c, mk) in enumerate((("changed answer in debate", T_, "o"), ("changed answer without communication", R_, "s"))):
            r = T[(T.group == g) & (T.domain == d) & (T.metric == "argmax_is_final") & (T.fraction == 0.0)].iloc[0]
            hdot(ax[1], len(rows) - 1 - ri + (0.15 if k == 0 else -0.15), 100 * r.est, 100 * r.lower, 100 * r.upper, c, mk)
            ax[1].text(99, len(rows) - 1 - ri + (0.15 if k == 0 else -0.15), f"n={int(r.n)}", fontsize=4.6, va="center", ha="right",
                       color=M_)
    tidy_y(ax[1], [l for _, l in rows])
    ax[1].set_xlim(0, 100)
    ax[1].set_xlabel("decided before writing (%)")
    ax[1].set_title("b  By answer regime", loc="left")
    fams = []  # family detail lives in the MAD Bench map; this slot now carries the controls
    _unused_fams = [f for f in F.family.unique() if F[(F.family == f)].est.notna().sum() == 2]
    fams = sorted(fams, key=lambda f: F[(F.family == f) & (F.cond == "debate")].est.iloc[0])
    for fi, f in enumerate(fams):
        for k, (cond, c, mk) in enumerate((("debate", T_, "o"), ("reflect", R_, "s"))):
            r = F[(F.family == f) & (F.cond == cond)].iloc[0]
            hdot(ax[2], fi + (0.15 if k == 0 else -0.15), 100 * r.est, 100 * r.lower, 100 * r.upper, c, mk)

    # d: the two controls. Matched-length neutral filler tests whether the fraction-0 read-out is about occupying
    # tokens; another item's reasoning tests whether the agent's OWN reasoning is what carries the answer.
    fp = TABLES / f"lit_filler_{m}_lit1200_summary.csv"
    if fp.exists():
        S = pd.read_csv(fp)
        arms = [("frac0", "nothing yet"), ("filler", "neutral filler"), ("cross_item", "other item's")]
        for k, (cond, c, mk) in enumerate((("debate", T_, "o"), ("reflect", R_, "s"))):
            for ai, (arm, _) in enumerate(arms):
                r = S[(S.cond == cond) & (S.group == "changed") & (S.arm == arm)]
                if not len(r):
                    continue
                r = r.iloc[0]
                hdot(ax[2], len(arms) - 1 - ai + (0.15 if k == 0 else -0.15), 100 * r.est, 100 * r.lower, 100 * r.upper, c, mk)
        ax[2].set_yticks(range(len(arms)))
        ax[2].set_yticklabels([l for _, l in arms][::-1])
        ax[2].grid(axis="y", visible=False)
        ax[2].set_ylim(-0.6, len(arms) - 0.4)
        ax[2].set_xlim(0, 100)
        ax[2].set_xlabel("gives its final answer (%)")
        ax[2].set_title("c  What comes before the answer", loc="left")
    style.finish_paper(fig, ax, OUT / f"fig_decided_before_reasoning_{m}")


def fig_vote(m):
    I = pd.read_csv(TABLES / f"lit_debate_{m}_lit1200_items.csv")
    T = I[I.target.notna() & (I.kind != "numeric")].copy()
    for c in ("vote", "debate_maj", "reflect_maj"):
        T[f"acc_{c}"] = (T[c] == T.target).astype(float)
    fig, ax = style.strip(2, 1.55, [1.1, 1.0])
    for di, (dom, lab) in enumerate(DOMS):
        g = T[T.domain == dom]
        for k, (c, clab, col, mk, fill) in enumerate((("vote", "majority vote (initial)", S_, "o", False),
                                                      ("reflect_maj", "3 rounds, no communication", R_, "s", True),
                                                      ("debate_maj", "3 rounds of debate", T_, "o", True))):
            e, lo, hi = ci(g[f"acc_{c}"], g.cluster, 100)
            hdot(ax[0], 2 - di + 0.22 - 0.22 * k, e, lo, hi, col, mk, fill, label=clab if di == 0 else None)
    tidy_y(ax[0], [l for _, l in DOMS])
    ax[0].set_xlabel("final majority correct (%)")
    ax[0].set_title("a  Accuracy", loc="left")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="outside lower center", ncols=3, fontsize=6)
    W = T[T.vote.notna() & (T.vote != T.target)].copy()
    for c in ("debate", "reflect"):
        W[f"wu_{c}"] = (W[f"{c}_unan"].astype(bool) & (W[f"{c}_maj"] != W.target)).astype(float)
    groups = [("all", "all regimes", W)] + [(d, l, W[W.domain == d]) for d, l in DOMS]
    for gi, (_, lab, g) in enumerate(groups):
        for k, (c, col, mk) in enumerate((("debate", T_, "o"), ("reflect", R_, "s"))):
            e, lo, hi = ci(g[f"wu_{c}"], g.cluster, 100)
            hdot(ax[1], len(groups) - 1 - gi + (0.15 if k == 0 else -0.15), e, lo, hi, col, mk)
        ax[1].text(101, len(groups) - 1 - gi, f"n={len(g)}", fontsize=5.5, va="center", color=S_)
    tidy_y(ax[1], [l for _, l, _ in groups])
    ax[1].set_xlim(0, 100)
    ax[1].set_xlabel("ends unanimous on the wrong answer (%)")
    ax[1].set_title("b  When the initial majority was wrong", loc="left")
    style.finish_paper(fig, ax, OUT / f"fig_debate_vs_vote_{m}")


def fig_audience(m):
    Q = pd.read_csv(TABLES / f"lit_debate_{m}_lit1200_requery.csv")
    U = Q[Q.unanimous.astype(bool)]
    fig, ax = style.strip(2, 1.65, [1.15, 1.0])
    groups = [("all", "all regimes")] + DOMS
    combos = [("debate", "private", T_, "o", True, "debate, private"), ("debate", "public", T_, "s", False, "debate, public"),
              ("reflect", "private", R_, "o", True, "no comm., private"), ("reflect", "public", R_, "s", False, "no comm., public")]
    for gi, (d, _) in enumerate(groups):
        g0 = U if d == "all" else U[U.domain == d]
        for k, (c, q, col, mk, fill, lab) in enumerate(combos):
            g = g0[g0.cond == c]
            e, lo, hi = ci(g[f"dev_{q}"], g.cluster, 100)
            hdot(ax[0], len(groups) - 1 - gi + 0.27 - 0.18 * k, e, lo, hi, col, mk, fill, label=lab if gi == 0 else None, s=11)
    tidy_y(ax[0], [l for _, l in groups])
    ax[0].set_xlabel("answer differs from own last public answer (%)")
    ax[0].set_title("a  Asked privately vs publicly", loc="left")
    ax[0].legend(loc="upper right", fontsize=5.4, labelspacing=0.3)
    for gi, (d, _) in enumerate(groups):
        g = U if d == "all" else U[U.domain == d]
        A = g.groupby(["item_id", "cluster", "cond"])[["dev_private", "dev_public"]].mean().unstack("cond").dropna()
        cl = A.index.get_level_values("cluster")
        for k, (q, col, mk) in enumerate((("private", P_, "o"), ("public", S_, "s"))):
            e, lo, hi = ci(A[(f"dev_{q}", "debate")] - A[(f"dev_{q}", "reflect")], cl, 100)
            hdot(ax[1], len(groups) - 1 - gi + (0.15 if k == 0 else -0.15), e, lo, hi, col, mk, k == 0,
                 label=f"{q} answer" if gi == 0 else None)
    ax[1].axvline(0, color=S_, lw=0.6)
    tidy_y(ax[1], [l for _, l in groups])
    ax[1].set_xlabel("debate − no communication (points)")
    ax[1].set_title("b  Effect of having debated", loc="left")
    ax[1].legend(loc="upper left", fontsize=5.8)
    style.finish_paper(fig, ax, OUT / f"fig_audience_{m}")


def fig_carrier(m):
    S = pd.read_csv(TABLES / f"lit_carrier_{m}_summary.csv")
    E = pd.read_csv(TABLES / f"lit_carrier_{m}_episodes.csv")
    fig, ax = style.strip(2, 1.6, [1.0, 1.15])
    groups = [("all", "all regimes")] + DOMS
    for gi, (d, lab) in enumerate(groups):
        for k, (metric, col, mk, fill, ml) in enumerate((("full", S_, "o", True, "full responses"),
                                                         ("reasoning_only", R_, "s", True, "reasoning, answer line removed"),
                                                         ("answer_only", T_, "D", True, "bare answers only"))):
            r = S[(S.scope == d) & (S.metric == metric)].iloc[0]
            hdot(ax[0], len(groups) - 1 - gi + 0.2 - 0.2 * k, r.est, r.lower, r.upper, col, mk, fill, label=ml if gi == 0 else None)
    ax[0].axvline(0, color=S_, lw=0.6)
    tidy_y(ax[0], [l for _, l in groups])
    ax[0].set_xlabel("pull toward the peers' answer vs. no peers (logits)")
    ax[0].set_title("a  What in a peer's message persuades", loc="left")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="outside lower center", ncols=3, fontsize=6)
    rows = []
    for dom, fams in FAMS:
        for f in fams:
            g = E[E.family == f]
            if g.cluster.nunique() < 4:
                continue
            share = g.answer_only.sum() / g.full.sum() if g.full.sum() > 0 else np.nan
            e, lo, hi = ci(g.answer_only, g.cluster)
            ef, lof, hif = ci(g.full, g.cluster)
            rows.append((dom, f, e, lo, hi, ef, lof, hif, len(g)))
    # panel a already spends the three categorical hues on message content, so the marks here stay ink and the
    # regime is carried by the coloured row labels and the separators -- one hue never means two things in one figure
    for ri, (dom, f, e, lo, hi, ef, lof, hif, n) in enumerate(rows):
        y = len(rows) - 1 - ri
        hdot(ax[1], y + 0.14, ef, lof, hif, S_, "o", True, s=11)
        hdot(ax[1], y - 0.14, e, lo, hi, S_, "D", False, s=11)
    ax[1].axvline(0, color=S_, lw=0.6)
    tidy_y(ax[1], [FLAB[f] for _, f, *_ in rows])
    for lab, (dom, *_ ) in zip(ax[1].get_yticklabels(), rows[::-1]):
        lab.set_color(style.DOMAIN_COLOR[dom])
    doms = [r[0] for r in rows]
    for k in range(1, len(doms)):
        if doms[k] != doms[k - 1]:
            ax[1].axhline(len(rows) - k - 0.5, color=style.GRID, lw=0.8)
    from matplotlib.lines import Line2D
    ax[1].legend(handles=[Line2D([], [], marker="o", ls="", color=S_, ms=3.5, label="full responses"),
                          Line2D([], [], marker="D", ls="", color=S_, markerfacecolor="white", ms=3.2, label="bare answers only")],
                 loc="lower right", fontsize=5.8, frameon=False, labelspacing=0.22, borderpad=0.1, handletextpad=0.35)
    ax[1].set_xlabel("pull toward the peers' answer vs. no peers (logits)")
    ax[1].set_title("b  By task family", loc="left")
    style.finish_paper(fig, ax, OUT / f"fig_what_persuades_{m}")


def fig_role(m):
    """Identical text, different attribution: who the words belong to, and who is reading them."""
    R = pd.read_csv(TABLES / f"lit_frames_{m}_lit1200_role_r1.csv")
    O = pd.read_csv(TABLES / f"lit_frames_{m}_lit1200_own.csv")
    fig, ax = style.strip(3, 1.75, [1.0, 1.0, 1.0])
    # a: adoption when outnumbered, by reader
    rows = [("adopt participant", "participant", T_, "o"), ("adopt observer", "outside reader", R_, "s"),
            ("adopt paraphrase", "participant, reworded", M_, "D")]
    groups = [("all", "all regimes")] + DOMS
    for gi, (d, _) in enumerate(groups):
        for k, (metric, lab, c, mk) in enumerate(rows):
            r = R[(R.domain == d) & (R.metric == metric)]
            if not len(r):
                continue
            r = r.iloc[0]
            hdot(ax[0], len(groups) - 1 - gi + 0.2 - 0.2 * k, 100 * r.est, 100 * r.lower, 100 * r.upper, c, mk,
                 label=lab if gi == 0 else None)
    tidy_y(ax[0], [l for _, l in groups], legend_lane=1.15)
    ax[0].set_xlabel("adopts the peers' answer (%)")
    ax[0].set_title("a  Outnumbered: who is reading", loc="left")
    ax[0].legend(loc="lower left", fontsize=5.8, frameon=False, labelspacing=0.22, borderpad=0.1, handletextpad=0.35)
    # b: the same on the margin scale, role effect vs wording effect
    for gi, (d, _) in enumerate(groups):
        for k, (metric, lab, c, mk) in enumerate((("margin participant - observer", "participant − reader", P_, "o"),
                                                  ("margin participant - paraphrase", "participant − reworded", M_, "D"))):
            r = R[(R.domain == d) & (R.metric == metric)]
            if not len(r):
                continue
            r = r.iloc[0]
            hdot(ax[1], len(groups) - 1 - gi + (0.15 if k == 0 else -0.15), r.est, r.lower, r.upper, c, mk,
                 label=lab if gi == 0 else None)
    ax[1].axvline(0, color=S_, lw=0.6)
    tidy_y(ax[1], [l for _, l in groups], legend_lane=1.15)
    ax[1].set_xlabel("difference in pull (logits)")
    ax[1].set_title("b  Role vs wording", loc="left")
    ax[1].legend(loc="lower left", fontsize=5.8, frameon=False, labelspacing=0.22, borderpad=0.1, handletextpad=0.35)
    # c: ownership of the words
    for gi, (d, _) in enumerate(groups):
        for k, (metric, lab, c, mk) in enumerate((("delta_own", "label swap", style.TERRACOTTA, "o"),
                                                  ("position_last_minus_first", "position control", M_, "s"))):
            r = O[(O.domain == d) & (O.metric == metric)]
            if not len(r):
                continue
            r = r.iloc[0]
            hdot(ax[2], len(groups) - 1 - gi + (0.15 if k == 0 else -0.15), r.est, r.lower, r.upper, c, mk,
                 label=lab if gi == 0 else None)
    ax[2].axvline(0, color=S_, lw=0.6)
    tidy_y(ax[2], [l for _, l in groups], legend_lane=1.15)
    ax[2].set_xlabel("extra pull when labelled own (logits)")
    ax[2].set_title("c  Whose words they are", loc="left")
    ax[2].legend(loc="lower left", fontsize=5.8, frameon=False, labelspacing=0.22, borderpad=0.1, handletextpad=0.35)
    style.finish_paper(fig, ax, OUT / f"fig_role_{m}")


def fig_map(m):
    """Heatmap: rows = effects, columns = task families; cell = estimate (row-scaled colour), bold = 95% CI excludes 0."""
    M = pd.read_csv(TABLES / f"madbench_map_{m}.csv")
    Cf = pd.read_csv(TABLES / f"lit_commit_{m}_family_fraction0.csv")
    Ec = pd.read_csv(TABLES / f"lit_carrier_{m}_episodes.csv")
    rows = [("unan_gain", "unanimity gain (debate − no comm.)", "pts"), ("acc_gain", "accuracy gain (debate − vote)", "pts"),
            ("lockin", "wrong-majority lock-in (debate − no comm.)", "pts"), ("priv_minus_pub", "private − public answer change", "pts"),
            ("debate_x_priv", "private change (debate − no comm.)", "pts"), ("decided", "debate changes decided before writing", "%"),
            ("answer_pull", "pull of a bare answer", "logits")]
    cols = [(d, f) for d, fs in FAMS for f in fs]
    V = np.full((len(rows), len(cols)), np.nan)
    SIG = np.zeros_like(V, bool)
    for ci_, (d, f) in enumerate(cols):
        for ri, (key, _, _) in enumerate(rows):
            if key == "decided":
                r = Cf[(Cf.family == f) & (Cf.cond == "debate")]
                if len(r) and np.isfinite(r.est.iloc[0]) and r.n.iloc[0] >= 8:
                    V[ri, ci_], SIG[ri, ci_] = 100 * r.est.iloc[0], True
            elif key == "answer_pull":
                g = Ec[Ec.family == f]
                if g.cluster.nunique() >= 4:
                    e, lo, hi = ci(g.answer_only, g.cluster)
                    V[ri, ci_], SIG[ri, ci_] = e, lo > 0 or hi < 0
            else:
                r = M[(M.family == f) & (M.metric == key)]
                if len(r) and np.isfinite(r.est.iloc[0]):
                    V[ri, ci_], SIG[ri, ci_] = r.est.iloc[0], (r.lower.iloc[0] > 0 or r.upper.iloc[0] < 0)
    fig, ax = plt.subplots(figsize=(style.TEXTWIDTH, 2.05), constrained_layout=True)
    S = V / np.nanmax(np.abs(V), axis=1, keepdims=True)
    ax.imshow(np.ma.masked_invalid(S), cmap=style.DIVERGING, vmin=-1, vmax=1, aspect="auto")
    for ri in range(len(rows)):
        for ci_ in range(len(cols)):
            v = V[ri, ci_]
            if not np.isfinite(v):
                # undefined, not zero: the family produced no episode of the kind the row measures (usually because the
                # agents were already unanimous at round 0), so the cell is marked rather than left to read as a gap
                ax.text(ci_, ri, "\u2013", ha="center", va="center", fontsize=6, color=M_)
                continue
            txt = f"{v:.0f}" if rows[ri][2] in ("pts", "%") else f"{v:.1f}"
            txt = txt.replace("-0.0", "0.0") if txt in ("-0.0", "-0") else txt.replace("-0", "0") if txt == "-0" else txt
            ax.text(ci_, ri, txt, ha="center", va="center", fontsize=5.4, fontweight="bold" if SIG[ri, ci_] else "normal",
                    color=style.INK if SIG[ri, ci_] else S_)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{lab} ({u})" for _, lab, u in rows], fontsize=6)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([FLAB[f] for _, f in cols], rotation=35, ha="right", fontsize=6)
    for lab, c in zip(ax.get_xticklabels(), cols):
        lab.set_color(style.DOMAIN_COLOR[c[0]])
    for x in (3.5, 8.5):
        ax.axvline(x, color="white", lw=1.6)
    for (dom, fams), x0 in zip(FAMS, (0, 4, 9)):
        ax.text(x0 + (len(fams) - 1) / 2, -0.75, {"knowledge_reasoning": "Knowledge", "safety_refusal": "Safety",
                                                   "values_moral_judgment": "Values"}[dom], ha="center", va="bottom", fontsize=6.6,
                color=style.DOMAIN_COLOR[dom])
    ax.text(len(cols) - 0.5, len(rows) - 0.32, "\u2013  no episode of this kind (agents already unanimous)",
            ha="right", va="top", fontsize=5.4, color=M_)
    ax.set_ylim(len(rows) + 0.25, -1.0)
    ax.grid(False)
    ax.set_facecolor("white")
    ax.tick_params(length=0, pad=1.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    style.finish_paper(fig, [ax], OUT / f"fig_madbench_map_{m}")


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "Q9"
    which = sys.argv[2:] or ["source", "commit", "vote", "audience", "carrier", "role", "map"]
    for w in which:
        try:
            globals()[f"fig_{w}"](m)
            print("ok", w)
        except FileNotFoundError as e:
            print("missing data for", w, e)
