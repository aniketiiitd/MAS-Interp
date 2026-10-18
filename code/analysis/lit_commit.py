"""Decided before reasoning? and what carries debate influence? (interp/commit_lit.py outputs)

commit  : P(written final answer) when the agent is made to answer after reading only the round message (fraction 0) and after
          growing fractions of its own reasoning. decided_at_read = argmax at fraction 0 equals the answer it finally wrote.
          Compared for answer changes in debate vs answer changes without communication, and for unchanged answers; by domain/family.
carrier : shift toward the disagreeing peers' answer Y vs the agent's own previous answer X (m = lp(Y) - lp(X)) relative to the
          no-peers message: full, answer_only (peers' bare answers), reasoning without the answer line. Shares by domain/family.
Tables: outputs/tables/lit_commit_<M>_*.csv, lit_carrier_<M>_*.csv
"""
from __future__ import annotations

import argparse
import json
import collections
import sys

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import *  # noqa
from stats import fast_cluster_mean_ci

CORE = {r["id"]: r for r in load_core()}
RAW_LIT = ROOT / "outputs/raw_lit"
cluster_of = lambda i: CORE[i]["meta"].get("analysis_group") if CORE[i]["stratum"] == "persuasion" else CORE[i]["source_item_id"]


def ci(x, c):
    x = np.asarray(x, float)
    if np.isfinite(x).sum() < 5:
        return dict(est=np.nan, lower=np.nan, upper=np.nan, n_clusters=0, n=int(np.isfinite(x).sum()))
    e, lo, hi, p, n = fast_cluster_mean_ci(x, np.asarray(c))
    return dict(est=e, lower=lo, upper=hi, n_clusters=n, n=int(np.isfinite(x).sum()))


def pnorm(lp, L):
    v = np.array(list(lp.values()))
    return float(np.exp(lp[L] - np.logaddexp.reduce(v))) if L in lp else np.nan


def run_commit(m, ds):
    # Same pooling rule as the carrier arm: rows.jsonl is record-keyed, so halves union safely, and a pooled run
    # writes its own files rather than overwriting the half-benchmark tables the current claim quotes.
    parts = [x for x in ds.split(",") if x]
    tag = m if len(parts) == 1 else f"{m}_full2400"
    D = pd.concat([pd.DataFrame(read_jsonl(ACTS / m / f"lit_commit_{p2}" / "rows.jsonl")) for p2 in parts],
                  ignore_index=True)
    if len(parts) > 1:
        print(f"[commit] pooled {len(parts)} halves -> {D.item_id.nunique()} distinct items", flush=True)
    D["p_final"] = [pnorm(lp, a) for lp, a in zip(D.lp_letters, D.answer)]
    D["p_prev"] = [pnorm(lp, a) for lp, a in zip(D.lp_letters, D.prev_answer)]
    D["argmax_is_final"] = [float(max(lp, key=lp.get) == a) for lp, a in zip(D.lp_letters, D.answer)]
    D["cluster"] = D.item_id.map(cluster_of)
    D["domain"] = D.item_id.map(lambda i: CORE[i]["primary_domain"])
    D["family"] = D.item_id.map(lambda i: family_of(CORE[i]))
    D["group"] = np.where(D.changed, np.where(D.cond == "debate", "changed answer in debate", "changed answer without communication"),
                          np.where(D.cond == "debate", "kept answer in debate", "kept answer without communication"))
    rows = []
    for (grp, f), g in D.groupby(["group", "frame"]):
        for scope, gg in [("all", g)] + list(g.groupby("domain")):
            for col in ("p_final", "argmax_is_final", "p_prev"):
                rows.append(dict(group=grp, fraction=f, domain=scope, metric=col, **ci(gg[col], gg.cluster)))
    T = pd.DataFrame(rows)
    T.to_csv(TABLES / f"lit_commit_{tag}_curves.csv", index=False)
    # paired contrast at fraction 0: debate changes vs no-communication changes (independent groups; cluster bootstrap on union)
    z = D[(D.frame == 0.0) & D.changed]
    fam = []
    for f, g in z.groupby("family"):
        for cond, gc in g.groupby("cond"):
            fam.append(dict(family=f, cond=cond, **ci(gc.argmax_is_final, gc.cluster)))
    pd.DataFrame(fam).to_csv(TABLES / f"lit_commit_{tag}_family_fraction0.csv", index=False)
    print(T[(T.domain == "all") & (T.metric == "argmax_is_final")].pivot_table(index="fraction", columns="group", values="est").round(3).to_string())
    print(T[(T.frame if False else True) & (T.fraction == 0.0) & (T.metric == "argmax_is_final")].pivot_table(index="domain", columns="group", values="est").round(3).to_string())
    balanced_check(m, ds, D)


def balanced_check(m, ds, D):
    """Re-estimate the fraction-0 result on items that parsed completely in BOTH conditions.

    The debate arm and the no-communication arm can lose episodes at different rates -- Llama-3.1-8B parses about five
    points worse alone than in debate, because the peers' formatted answers act as a template the solo condition lacks.
    That selects episodes asymmetrically inside the very contrast this result rests on. Restricting to items where every
    agent parsed at every round in both conditions removes the asymmetry; if the estimate moves, the full-sample figure
    is not usable for that model.
    """
    d = RAW_LIT / m / ds
    if not d.exists():
        return
    R = json.load(open(d / "meta.json"))["rounds"]
    seen, bad, n_files = set(), set(), 0
    files = [d / "r0.jsonl"] + [d / f"{c}_r{k}.jsonl" for c in ("debate", "reflect") for k in range(1, R + 1)]
    for f in files:
        if not f.exists():
            continue
        n_files += 1
        for x in read_jsonl(f):
            seen.add(x["item_id"])            # every item must be registered, not only the failing ones,
            if x.get("answer") is None:       # or the "parsed everywhere" set comes out empty
                bad.add(x["item_id"])
    if n_files == 0:
        return
    keep = seen - bad
    z = D[(D.frame == 0.0) & D.changed]
    rows = []
    for grp, g in z.groupby("group"):
        gb = g[g.item_id.isin(keep)]
        rows.append(dict(group=grp, subset="full", **ci(g.argmax_is_final, g.cluster)))
        rows.append(dict(group=grp, subset="balanced", **ci(gb.argmax_is_final, gb.cluster)))
    B = pd.DataFrame(rows)
    B.to_csv(TABLES / f"lit_commit_{m}_balanced.csv", index=False)
    n_items = len(seen)
    print(f"\n[balanced] {len(keep)}/{n_items} items parsed completely in both conditions "
          f"({100*len(keep)/max(1,n_items):.1f}%)")
    print(B.round(3).to_string(index=False))
    for grp in B.group.unique():
        a = B[(B.group == grp) & (B["subset"] == "full")].iloc[0]
        b = B[(B.group == grp) & (B["subset"] == "balanced")].iloc[0]
        if not (np.isfinite(a.est) and np.isfinite(b.est)):
            continue
        moved = abs(a.est - b.est) > 0.05 or not (b.lower <= a.est <= b.upper)
        print(f"  {grp:40s} full {100*a.est:5.1f}%  balanced {100*b.est:5.1f}%  "
              f"{'*** MOVED -- full-sample figure not usable' if moved else 'stable'}")


def run_carrier(m, ds):
    # ds may name several halves, comma-separated. The output filename keeps the single-dataset form untouched so
    # queued jobs behave exactly as before, and a pooled run writes its own "full2400" files instead of overwriting
    # the half-benchmark tables the current numbers come from.
    parts = [x for x in ds.split(",") if x]
    tag = m if len(parts) == 1 else f"{m}_full2400"
    D = pd.concat([pd.DataFrame(read_jsonl(ACTS / m / f"lit_carrier_{p}" / "rows.jsonl")) for p in parts],
                  ignore_index=True)
    if len(parts) > 1:
        print(f"[carrier] pooled {len(parts)} halves -> {D.item_id.nunique()} distinct items", flush=True)
    out = []
    for (i, g), e in D.groupby(["item_id", "agent"]):
        fr = {r.frame: r for r in e.itertuples()}
        X = fr["full"].own
        peers = [a for a in fr["full"].peers.values() if a not in (None, X)]
        if not peers:
            continue
        Y = collections.Counter(peers).most_common(1)[0][0]
        both = sum(a == Y for a in fr["full"].peers.values()) == 2
        mm = lambda r: r.lp_letters[Y] - r.lp_letters[X]
        base = mm(fr["no_peers"])
        out.append(dict(item_id=i, agent=g, cluster=cluster_of(i), domain=CORE[i]["primary_domain"], family=family_of(CORE[i]),
                        outnumbered=both, full=mm(fr["full"]) - base, answer_only=mm(fr["answer_only"]) - base,
                        reasoning_only=mm(fr["reasoning_no_answer_line"]) - base))
    E = pd.DataFrame(out)
    E.to_csv(TABLES / f"lit_carrier_{tag}_episodes.csv", index=False)
    rows = []
    for scope, g in [("all", E)] + list(E.groupby("domain")) + list(E.groupby("family")):
        for col in ("full", "answer_only", "reasoning_only"):
            rows.append(dict(scope=scope, metric=col, **ci(g[col], g.cluster)))
        rows.append(dict(scope=scope, metric="full - answer_only", **ci(g.full - g.answer_only, g.cluster)))
        rows.append(dict(scope=scope, metric="full - reasoning_only", **ci(g.full - g.reasoning_only, g.cluster)))
    T = pd.DataFrame(rows)
    T.to_csv(TABLES / f"lit_carrier_{tag}_summary.csv", index=False)
    print(T[T.scope.isin(["all", "knowledge_reasoning", "safety_refusal", "values_moral_judgment"])].round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["commit", "carrier"])
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="lit1200")
    a = ap.parse_args()
    (run_commit if a.mode == "commit" else run_carrier)(a.model, a.dataset)
