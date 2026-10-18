"""When a debate ends unanimous on a wrong answer, where did that answer come from?

Contribution five says a debated group is worth about one effective voter, so a vote over it cannot correct much.
That predicts something specific about the errors: most confident error should be the group *failing to dislodge* a
wrong answer it already held, rather than the group inventing a new one. This measures that split.

For every item where the three agents did not all agree before anyone spoke, and where they end the debate
unanimous, we ask whether the answer they settled on was the one the plurality already held. Two outcomes:

  ratified   the initial plurality answer won. Debate did not correct an error it inherited.
  overturned a different answer won. Debate moved the group off its better-supported starting point.

Both are reported for wrong and for right endings, because the interesting quantity is whether the split differs:
if debate overturns the plurality just as often when it ends up right, then overturning is not itself the problem.

The matched no-communication arm is measured the same way, since agents thinking alone also converge sometimes and
the question is what debating adds.

Intervals are item-clustered bootstrap, the same unit used everywhere else in this campaign.

Out: outputs/tables/lit_convergence_source_<model>_<dataset>.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import *  # noqa

RAW_LIT = ROOT / "outputs/raw_lit"


def load_round(path):
    d = defaultdict(dict)
    for x in read_jsonl(path):
        d[x["item_id"]][x["agent"]] = x
    return d


def boot(flags, clusters, B, rng):
    """Item-clustered bootstrap of a proportion."""
    if not len(flags):
        return float("nan"), float("nan"), float("nan")
    flags = np.asarray(flags, float)
    cl = np.asarray(clusters)
    uniq = np.unique(cl)
    groups = {u: np.where(cl == u)[0] for u in uniq}
    est = flags.mean()
    draws = np.empty(B)
    for b in range(B):
        ix = np.concatenate([groups[u] for u in rng.choice(uniq, len(uniq))])
        draws[b] = flags[ix].mean()
    return float(est), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="lit1200")
    ap.add_argument("--B", type=int, default=4000)
    a = ap.parse_args()
    rng = np.random.default_rng(240091)

    core = {r["id"]: r for r in load_core()}
    unit = lambda i: core[i]["meta"].get("analysis_group") if core[i]["stratum"] == "persuasion" else core[i]["source_item_id"]

    parts = [p for p in a.dataset.split(",") if p]
    rows = []
    for cond, last in (("debate", "debate_r{}"), ("reflect", "reflect_r{}")):
        r0, rN = defaultdict(dict), defaultdict(dict)
        rounds = None
        for p in parts:
            d = RAW_LIT / a.model / p
            meta = json.load(open(d / "meta.json"))
            if rounds is None:
                rounds = meta["rounds"]
            elif rounds != meta["rounds"]:
                raise SystemExit(f"[convergence] {p} ran {meta['rounds']} rounds, expected {rounds}; refusing to pool")
            for it, v in load_round(d / "r0.jsonl").items():
                r0[it].update(v)
            for it, v in load_round(d / (last.format(rounds) + ".jsonl")).items():
                rN[it].update(v)

        recs = []
        for it, ags in rN.items():
            if len(ags) < 3 or it not in r0 or len(r0[it]) < 3:
                continue
            a0 = [str(r0[it][g].get("answer")) for g in sorted(r0[it])]
            aN = [str(ags[g].get("answer")) for g in sorted(ags)]
            if "None" in a0 or "None" in aN:
                continue
            if len(set(a0)) == 1 or len(set(aN)) != 1:      # must start split and end unanimous
                continue
            top, n = Counter(a0).most_common(1)[0]
            if n < 2:                                        # no plurality to ratify
                continue
            correct = ags[sorted(ags)[0]].get("correct")
            if correct is None:
                continue
            recs.append(dict(item_id=it, cluster=unit(it), ratified=int(aN[0] == top), correct=bool(correct)))
        D = pd.DataFrame(recs)
        if not len(D):
            continue
        for scope, sub in (("wrong", D[~D.correct]), ("right", D[D.correct]), ("all", D)):
            if not len(sub):
                continue
            est, lo, hi = boot(sub.ratified.to_numpy(), sub.cluster.to_numpy(), a.B, rng)
            rows.append(dict(model=a.model, dataset=a.dataset, arm=cond, ending=scope, n_items=len(sub),
                             n_clusters=sub.cluster.nunique(), ratified=est, lower=lo, upper=hi,
                             overturned=1 - est))
    T = pd.DataFrame(rows)
    tag = f"{a.model}_" + (parts[0] if len(parts) == 1 else "full2400")
    T.to_csv(TABLES / f"lit_convergence_source_{tag}.csv", index=False)
    print(T.round(3).to_string(index=False))
    print(f"\n[done] outputs/tables/lit_convergence_source_{tag}.csv")


if __name__ == "__main__":
    main()
