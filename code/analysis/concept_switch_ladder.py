"""Does a concept level read before an agent writes predict that this agent is about to change its answer?

The concept-behaviour scan reports family-adjusted AUROC. That is the estimator that produced a retraction earlier in
this campaign: a direction can separate switchers from non-switchers across contexts because some questions produce more
switching, without saying anything about which agent switches. So every concept that clears the shuffled-label band in
that scan is re-tested here at the tightest adjustment the design allows, plus the decisive multi-agent test.

Levels, in order of increasing tightness:
  pooled           no adjustment
  family           within task family        <- what the scan reports
  item             within item
  item_round       within item and round     <- the tightest the design allows
  switcher_test    in groups where exactly one of the three agents switched, is that agent the most extreme of the three?

Out: outputs/tables/concept_switch_ladder_<tag>_<model>_<dataset>.csv
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import *  # noqa

CORE = {r["id"]: r for r in load_core()}
RAW_LIT = ROOT / "outputs/raw_lit"


def auroc(y, s):
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def grouped_auroc(y, s, g):
    """AUROC pooled over groups, weighted by the number of comparable pairs each group contributes."""
    num = den = 0.0
    for key in np.unique(g):
        m = g == key
        if m.sum() < 2:
            continue
        a = auroc(y[m], s[m])
        if np.isnan(a):
            continue
        w = y[m].sum() * (~y[m]).sum()
        num += a * w
        den += w
    return num / den if den else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="lit1200")
    ap.add_argument("--tag", default="emotion_stories")
    ap.add_argument("--top", type=int, default=12, help="how many of the scan's strongest concept-layer cells to re-test")
    a = ap.parse_args()

    cd = ACTS / a.model / a.tag
    concepts = json.load(open(cd / "meta.json"))["concepts"]
    V = np.load(cd / "vectors.npz")
    VN = np.load(cd / "null_vectors.npz")
    NPERM = 40   # shuffled-label refits re-tested per concept; the whole bank is larger than this test needs

    rd = ACTS / a.model / f"lit_replay_{a.dataset}"
    rmeta = json.load(open(rd / "meta.json"))
    grid, si = rmeta["grid"], rmeta["sites"].index("reading")
    I = pd.DataFrame(read_jsonl(rd / "index.jsonl"))
    A = np.load(rd / "acts.npy", mmap_mode="r")

    ans = {}
    for f in (RAW_LIT / a.model / a.dataset).glob("*.jsonl"):
        for x in read_jsonl(f):
            ans[(x["item_id"], x["cond"], x["agent"], x["round"], x.get("query"))] = x["answer"]
    rows = []
    for r in I.itertuples():
        if isinstance(r.query, str) or r.cond not in ("debate", "reflect") or r.round == 0:
            continue
        prev = ans.get((r.item_id, "r0" if r.round == 1 else r.cond, r.agent, r.round - 1, None))
        cur = ans.get((r.item_id, r.cond, r.agent, r.round, None))
        if prev is None or cur is None:
            continue
        rows.append(dict(row=r.row, cond=r.cond, item_id=r.item_id, agent=r.agent, round=r.round,
                         switch=float(prev != cur), family=family_of(CORE[r.item_id])))
    D = pd.DataFrame(rows)
    print(f"[ladder] {a.tag} {a.model} {a.dataset}: {len(D)} responses, {int(D.switch.sum())} answer changes", flush=True)

    scan = pd.read_csv(TABLES / f"lit_concept_behaviour_{a.tag}_{a.model}.csv")
    scan = scan[scan.cond == "debate"].copy()
    scan["dev"] = (scan.auroc - 0.5).abs()
    cells = scan.sort_values("dev", ascending=False).head(a.top)[["layer", "concept", "auroc"]]

    sub = D[D.cond == "debate"].reset_index(drop=True)
    out = []
    for L, c, scan_auroc in cells.itertuples(index=False):
        L = int(L)
        gi = grid.index(L)
        j = concepts.index(c)
        H = np.asarray(A[sub.row.to_numpy(), si, gi], np.float32)
        H = (H - H.mean(0)) / (H.std(0) + 1e-6)
        s = H @ V[f"L{L}"][j]
        y = sub.switch.to_numpy(bool)
        ir = (sub.item_id.astype(str) + "|" + sub["round"].astype(str)).to_numpy()

        # The decisive multi-agent test. Within one item and round the three agents saw the same question and the same
        # peers, so "which of them switched" is the only thing left to explain. Groups with exactly one switcher are the
        # informative ones; a direction that identifies the switcher ranks it first more often than one time in three.
        # Rank by the signed score in whichever direction the item+round AUROC says predicts switching, not by
        # magnitude: for a concept like self-confident the switcher is the LOWEST of the three, and an absolute value
        # would pick whichever agent is furthest from the pooled mean instead.
        sign = 1.0 if grouped_auroc(y, s, ir) >= 0.5 else -1.0
        hit = tot = 0
        for _, g in sub.assign(score=sign * s).groupby(["item_id", "round"]):
            if len(g) != 3 or g.switch.sum() != 1:
                continue
            tot += 1
            hit += int(g.score.idxmax() == g.index[g.switch == 1][0])
        # One-sided exact binomial: how often chance alone puts the switcher first at least this many times.
        p = float(stats.binomtest(hit, tot, 1 / 3, alternative="greater").pvalue) if tot else np.nan

        # The control that decides whether the concept's identity matters. The permuted-label directions were refit on
        # the same stories with the concept labels shuffled inside each topic, so they live in the same subspace and
        # carry any generic "this agent's state is unusual" factor. If they identify the switcher at the same rate, the
        # result is about state atypicality, not about being puzzled.
        nr = []
        for b in range(min(NPERM, VN[f"L{L}"].shape[0])):
            sn = H @ VN[f"L{L}"][b, j]
            sgn = 1.0 if grouped_auroc(y, sn, ir) >= 0.5 else -1.0
            h2 = t2 = 0
            for _, g in sub.assign(score=sgn * sn).groupby(["item_id", "round"]):
                if len(g) != 3 or g.switch.sum() != 1:
                    continue
                t2 += 1
                h2 += int(g.score.idxmax() == g.index[g.switch == 1][0])
            nr.append(h2 / t2 if t2 else np.nan)
        null_rate = float(np.nanmean(nr))
        null_p95 = float(np.nanquantile(nr, 0.95))

        out.append(dict(layer=L, concept=c, scan_family_auroc=float(scan_auroc),
                        null_switcher_rate=null_rate, null_switcher_p95=null_p95,
                        beats_null=bool(hit / tot > null_p95) if tot else False,
                        pooled=auroc(y, s), family=grouped_auroc(y, s, sub.family.to_numpy()),
                        item=grouped_auroc(y, s, sub.item_id.to_numpy()), item_round=grouped_auroc(y, s, ir),
                        switcher_groups=tot, switcher_hits=hit,
                        switcher_rate=hit / tot if tot else np.nan, switcher_p=p, direction=sign))
        o = out[-1]
        print(f"  L{L:2d} {c:16s} family {o['family']:.3f} -> item {o['item']:.3f} -> item+round {o['item_round']:.3f}"
              f"   switcher {100 * o['switcher_rate']:.1f}%  vs shuffled-label {100 * null_rate:.1f}% "
              f"(95th {100 * null_p95:.1f}%)  {'BEATS' if o['beats_null'] else 'no'}", flush=True)

    T = pd.DataFrame(out)
    T.to_csv(TABLES / f"concept_switch_ladder_{a.tag}_{a.model}_{a.dataset}.csv", index=False)
    print(f"\n[done] outputs/tables/concept_switch_ladder_{a.tag}_{a.model}_{a.dataset}.csv")
    print(f"median family-adjusted {T.family.median():.3f}, median within item+round {T.item_round.median():.3f}; "
          f"{int(T.beats_null.sum())} of {len(T)} concepts identify the switcher above the shuffled-label band "
          f"(mean shuffled-label rate {100 * T.null_switcher_rate.mean():.1f}%, chance 33.3%)")


if __name__ == "__main__":
    main()
