"""Do agents write more as a debate drags on, while the answer stops moving?

Claim 5 says a debating group stops changing its answer early and then keeps generating: between turns 6 and 11
the debating arm moves half a point while producing eight million tokens. That is an accounting of tokens, not of
what happens to the responses themselves. This measures the responses.

It exists because of a qualitative mistake. Reading three twelve-turn transcripts suggested that agents stop
citing their peers as the debate wears on. Measured over all 1,800 responses per turn, the citation rate is flat
(21.7% at turn 1, 20.3% at turn 11), so the impression was a sampling artifact -- the third time this campaign a
story from a handful of transcripts failed against its own base rate. What the measurement DID find is the
opposite arm of the same question, and it is better: the responses keep growing.

Paired by agent and item, bootstrapped over item clusters because three agents share each question.

Out: outputs/tables/response_length_<M>_<dataset>.csv
"""
from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import *  # noqa

CITE = re.compile(r"\b(other agents?|the (?:recent )?opinions?|consensus|fellow agents?|peers?|"
                  r"agent \d|as (?:noted|argued|pointed out) by|previous (?:agent )?opinions?)\b", re.I)


def load(base, cond, rnd):
    return {(x["item_id"], x.get("agent")): (x.get("text") or "")
            for x in read_jsonl(f"{base}/{cond}_r{rnd}.jsonl")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="release_600_t12")
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=11)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()

    base = ROOT / "outputs/raw_lit" / a.model / a.dataset
    rng = np.random.default_rng(0)
    rows = []
    for cond in ("debate", "reflect"):
        A, B = load(base, cond, a.first), load(base, cond, a.last)
        ks = [k for k in A if k in B and len(A[k]) > 100 and len(B[k]) > 100]
        items = np.array([k[0] for k in ks])
        d = np.array([len(B[k]) - len(A[k]) for k in ks], float)
        uniq = np.unique(items)
        by = {i: d[items == i] for i in uniq}
        boot = [np.concatenate([by[i] for i in rng.choice(uniq, len(uniq), replace=True)]).mean()
                for _ in range(a.boot)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        cite_first = float(np.mean([1.0 if CITE.search(A[k]) else 0.0 for k in ks]))
        cite_last = float(np.mean([1.0 if CITE.search(B[k]) else 0.0 for k in ks]))
        rows.append(dict(model=a.model, cond=cond, first=a.first, last=a.last, n_pairs=len(ks),
                         n_items=len(uniq), mean_first=float(np.mean([len(A[k]) for k in ks])),
                         mean_last=float(np.mean([len(B[k]) for k in ks])), delta=float(d.mean()),
                         lower=float(lo), upper=float(hi), frac_longer=float((d > 0).mean()),
                         cite_rate_first=cite_first, cite_rate_last=cite_last))
        r = rows[-1]
        print(f"  {cond:8s} {r['n_pairs']:5d} pairs / {r['n_items']} items: "
              f"turn {a.last} minus turn {a.first} = {r['delta']:+7.1f} chars [{lo:+.1f}, {hi:+.1f}], "
              f"{100 * r['frac_longer']:.1f}% longer | cites peers {100 * cite_first:.1f}% -> "
              f"{100 * cite_last:.1f}%", flush=True)

    # The portable quantity is the CONTRAST between arms, not either arm's direction: Qwen's solo agents shrink
    # and Llama's grow, but debate outgrows solo by a wide margin in both. Bootstrapped over the same item
    # clusters so the two arms resample together.
    A1, B1 = load(base, "debate", a.first), load(base, "debate", a.last)
    A0, B0 = load(base, "reflect", a.first), load(base, "reflect", a.last)
    kd = [k for k in A1 if k in B1 and len(A1[k]) > 100 and len(B1[k]) > 100]
    kr = [k for k in A0 if k in B0 and len(A0[k]) > 100 and len(B0[k]) > 100]
    dd = pd.DataFrame({"item": [k[0] for k in kd], "v": [len(B1[k]) - len(A1[k]) for k in kd]})
    rr = pd.DataFrame({"item": [k[0] for k in kr], "v": [len(B0[k]) - len(A0[k]) for k in kr]})
    uniq = np.array(sorted(set(dd.item) | set(rr.item)))
    gd = {i: g.v.to_numpy() for i, g in dd.groupby("item")}
    gr = {i: g.v.to_numpy() for i, g in rr.groupby("item")}
    rng2 = np.random.default_rng(1)
    bs = []
    for _ in range(a.boot):
        s_ = rng2.choice(uniq, len(uniq), replace=True)
        d_ = np.concatenate([gd[i] for i in s_ if i in gd])
        r_ = np.concatenate([gr[i] for i in s_ if i in gr])
        bs.append(d_.mean() - r_.mean())
    lo2, hi2 = np.percentile(bs, [2.5, 97.5])
    contrast = float(dd.v.mean() - rr.v.mean())
    rows.append(dict(model=a.model, cond="debate_minus_solo", first=a.first, last=a.last,
                     n_pairs=len(kd) + len(kr), n_items=len(uniq), mean_first=float("nan"),
                     mean_last=float("nan"), delta=contrast, lower=float(lo2), upper=float(hi2),
                     frac_longer=float("nan"), cite_rate_first=float("nan"), cite_rate_last=float("nan")))
    print(f"  {'contrast':8s} debate minus solo growth = {contrast:+7.1f} chars [{lo2:+.1f}, {hi2:+.1f}]",
          flush=True)

    T = pd.DataFrame(rows)
    # The turn range MUST be in the filename. Without it, running a 1-to-7 window to preview a second model
    # silently overwrote the 1-to-11 table the claim quotes, and the audit went red on numbers nobody had
    # touched. A table whose name does not encode the parameters that produced it is a table that can be
    # clobbered by a curiosity.
    out = TABLES / f"response_length_{a.model}_{a.dataset}_t{a.first}to{a.last}.csv"
    T.to_csv(out, index=False)
    print(f"[done] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
