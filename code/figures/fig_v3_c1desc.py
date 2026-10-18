"""V3-C1'': who said it = stated reliability, read out late (analysis/attribution.py desc + analysis/desc_depth.py tables).
Usage: python figures/fig_v3_c1desc.py Q9 cpt_desc_release_600
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


def main(model, dname):
    W = pd.DataFrame(read_jsonl(ACTS / model / dname / "worlds.jsonl"))
    core = {r["id"]: r for r in load_core()}
    W["cluster"] = W.item_id.map(lambda i: core[i]["meta"].get("analysis_group") if core[i]["stratum"] == "persuasion" else core[i]["source_item_id"])
    key = ["item_id", "adv", "qual", "prior"]
    W = W.join(W[W.prov == "none"].set_index(key).m_adv.rename("m_none"), on=key)
    W["P"] = W.m_adv - W.m_none
    dis = W[(W.adv != W.prior) & (W.prov != "none")]
    per = dis.groupby(["item_id", "cluster", "prov"]).P.mean().reset_index()
    T = pd.read_csv(TABLES / f"v3_desc_depth_{model}_{dname}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.0, 1.15]))
    ax = axes[0]
    recs = [("good", "strong track record"), ("unst", "unknown track record"), ("bad", "poor track record")]
    for si, (s, lab, c, m) in enumerate((("prof", "tenured professor", style.PLUM, "o"), ("ord", "ordinary member of the public", style.TEAL, "s"))):
        for ri, (r, rlab) in enumerate(recs):
            g = per[per.prov == f"{s}_{r}"]
            e, lo, hi, p, n = fast_cluster_mean_ci(g.P.to_numpy(), g.cluster.to_numpy())
            y = 2 - ri + (0.13 if si == 0 else -0.13)
            ax.plot([lo, hi], [y, y], color=c, lw=1.3, solid_capstyle="round")
            ax.scatter([e], [y], s=32, color=c, marker=m, zorder=3, label=lab if ri == 0 else None)
    for prov, lab, y in (("expert", "original 'expert'", 3.1), ("anon", "original 'anonymous'", 3.1)):
        pass
    ax.axvline(0, color=style.SLATE, lw=0.8)
    ax.set_yticks(range(3))
    ax.set_yticklabels([r for _, r in recs][::-1])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Adoption shift vs. no source label (logits)")
    ax.set_title("a  Same message, 2 statuses x 3 records", loc="left")
    ax.legend(loc="upper left", fontsize=7.5)
    ax = axes[1]
    S = T[(T.metric == "share") & (T.representation != "m_adv")].sort_values("depth")
    b = T[(T.metric == "share") & (T.representation == "m_adv")].iloc[0]
    ax.axhspan(b.lower, b.upper, color=style.TERRACOTTA, alpha=0.18, lw=0)
    ax.axhline(b.est, color=style.TERRACOTTA, lw=1.2)
    ax.text(0.02, b.est - 0.03, "behaviour (adoption)", color=style.TERRACOTTA, fontsize=8, ha="left", va="top")
    ax.plot(S.depth, S.est, color=style.INK, lw=1.2, zorder=2)
    ax.vlines(S.depth, S.lower, S.upper, color=style.INK, lw=1.2)
    ax.scatter(S.depth, S.est, s=26, color=style.INK, zorder=3)
    ax.axvspan(15 / 31, 21 / 31, color=style.GOLD, alpha=0.22, lw=0)
    ax.text(18 / 31, 0.93, "credential integration\nat decision token (patching)", ha="center", va="top", fontsize=7.5, color=style.SLATE)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Relative depth (layer / last layer)")
    ax.set_ylabel("Status share of source effect")
    ax.set_title("b  Status is represented early, reliability late", loc="left")
    style.finish(fig, axes, ROOT / "outputs/figures" / f"V3_C1_status_reliability_{model}_{dname}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
