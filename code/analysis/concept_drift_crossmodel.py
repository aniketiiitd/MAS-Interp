"""Do the moral concepts that move most in one model move the same way in another?

Claim 7 says they do not: "the two moral concepts that move most in Qwen over twelve turns both flip sign in
Llama." That sentence was the fourth found in this campaign with no table behind it, after the shuffled-label
AUROC, the prompt lengths, and Gemma's concept counts. Its numbers lived only in the ledger.

Worse than missing, the sentence was compressed in a way that misleads. The two concepts were SELECTED on Qwen's
twelve-turn run, but the cross-model comparison was run on the THREE-ROUND runs both models have, because Llama
has no twelve-turn replay. Written as it was, a reader takes the Llama result to be over twelve turns too.

So: the concepts are fixed in advance (no re-selection, which is the whole point of a cross-model check), and
both models are measured on the same three-round data at the decision site. Drift is the debate arm minus the
matched no-communication arm, z-scored per layer on the round-0 states, bootstrapped over item clusters.

Out: outputs/tables/concept_drift_crossmodel_<tag>.csv
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import *  # noqa


def drift(model, tag, concepts, site, first, last, boot, rng):
    cd = ACTS / model / tag
    meta = json.load(open(cd / "meta.json"))
    V = np.load(cd / "vectors.npz")
    names = meta["concepts"]
    idx = [names.index(c) for c in concepts]

    rd = ACTS / model / "lit_replay_lit1200"
    rmeta = json.load(open(rd / "meta.json"))
    grid = rmeta["grid"]
    si = rmeta["sites"].index(site)
    I = pd.DataFrame(read_jsonl(rd / "index.jsonl"))
    A = np.load(rd / "acts.npy", mmap_mode="r")

    rows = []
    for L in grid:
        if f"L{L}" not in V.files:
            continue
        gi = grid.index(L)
        Vr = V[f"L{L}"][idx]
        Vr = Vr / (np.linalg.norm(Vr, axis=1, keepdims=True) + 1e-8)
        # z-scale on the round-0 states, which neither arm has acted on yet
        b = I.index[(I.cond == "r0")]
        Pb = np.asarray(A[b, si, gi, :], np.float32) @ Vr.T
        mu, sd = Pb.mean(0), Pb.std(0) + 1e-8

        def proj(cond, rnd):
            k = I.index[(I.cond == cond) & (I["round"] == rnd)]
            P = (np.asarray(A[k, si, gi, :], np.float32) @ Vr.T - mu) / sd
            return P, I.loc[k, "item_id"].to_numpy()

        for ci, c in enumerate(concepts):
            D1, it1 = proj("debate", first)
            D3, it3 = proj("debate", last)
            R1, _ = proj("reflect", first)
            R3, _ = proj("reflect", last)
            n = min(len(D1), len(D3), len(R1), len(R3))
            d = (D3[:n, ci] - D1[:n, ci]) - (R3[:n, ci] - R1[:n, ci])
            items = it1[:n]
            uniq = np.unique(items)
            by = {i: d[items == i] for i in uniq}
            bs = [np.concatenate([by[i] for i in rng.choice(uniq, len(uniq), replace=True)]).mean()
                  for _ in range(boot)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append(dict(model=model, layer=L, concept=c, site=site, first=first, last=last,
                             drift=float(d.mean()), lower=float(lo), upper=float(hi),
                             n=int(n), n_items=int(len(uniq))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Q9,L31")
    ap.add_argument("--tag", default="moral_stories")
    ap.add_argument("--concepts", default="self-righteous,conflicted-duty")
    ap.add_argument("--sites", default="reading,decision",
                    help="both by default: the sign reversal is present at one and absent at the other, which is "
                         "the finding")
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=3)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()

    concepts = [c.strip() for c in a.concepts.split(",") if c.strip()]
    rng = np.random.default_rng(0)
    rows = []
    for site in [x.strip() for x in a.sites.split(",") if x.strip()]:
        for m in [x.strip() for x in a.models.split(",") if x.strip()]:
            rows += drift(m, a.tag, concepts, site, a.first, a.last, a.boot, rng)
    T = pd.DataFrame(rows)

    # one number per model and concept: the layer where the effect is largest in the FIRST model, used for both,
    # because picking each model's own best layer would be selecting on the outcome we are comparing
    m0 = T.model.iloc[0]
    T["is_reference_layer"] = False
    for site in T.site.unique():
        ref = T[(T.model == m0) & (T.site == site)]
        for c in concepts:
            rc = ref[ref.concept == c]
            L = int(rc.loc[rc.drift.abs().idxmax()].layer)
            T.loc[(T.site == site) & (T.concept == c) & (T.layer == L), "is_reference_layer"] = True
    # how often the two models disagree about the sign, layer by layer: the stability the claim now reports
    dis = []
    for site in T.site.unique():
        for c in concepts:
            q = T[(T.model == m0) & (T.site == site) & (T.concept == c)].set_index("layer").drift
            o = T[(T.model != m0) & (T.site == site) & (T.concept == c)].set_index("layer").drift
            n = sum((q[L] > 0) != (o[L] > 0) for L in q.index)
            dis.append(dict(site=site, concept=c, layers=len(q), layers_sign_differs=int(n)))
    D = pd.DataFrame(dis)
    out = TABLES / f"concept_drift_crossmodel_{a.tag}.csv"
    T.to_csv(out, index=False)
    D.to_csv(TABLES / f"concept_drift_crossmodel_{a.tag}_signs.csv", index=False)
    for site in T.site.unique():
        print(f"  --- {site} site")
        for c in concepts:
            r0 = T[(T.site == site) & (T.concept == c) & T.is_reference_layer]
            L = int(r0.layer.iloc[0])
            print(f"  {c} (layer {L}, chosen on {m0}):")
            for m in T.model.unique():
                r = r0[r0.model == m].iloc[0]
                print(f"      {m:4s} {r.drift:+.3f} [{r.lower:+.3f}, {r.upper:+.3f}]  n={r.n} items={r.n_items}")
    print(D.to_string(index=False))
    print(f"[done] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
