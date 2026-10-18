"""Transplanting the reading-time state: does it move the answer, and does the reasoning still have no peer to refer to?
(interp/prewrite_lit.py outputs)

Per arm and layer, over episodes where a peer disagreed with the agent:
  p_target    probability of the disagreeing peers' answer at the forced answer token
  d_target    that probability minus the unpatched no-communication arm on the same episode (the causal effect)
  wrote_target / wrote_own  the answer the model actually writes, when it finishes inside the token budget
  mentions    share of responses whose reasoning refers to other agents at all
Item-cluster bootstrap CIs throughout (persuasion items cluster on analysis_group, otherwise on source_item_id). Split by domain and
by whether the agent really did switch. A qualitative dump pairs the unpatched and transplanted responses for ten episodes.
Tables: outputs/tables/lit_prewrite_<M>_{arms,layers,domain,switch,quotes}.csv
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import *  # noqa
from stats import fast_cluster_mean_ci

CORE = {r["id"]: r for r in load_core()}
cluster_of = lambda i: CORE[i]["meta"].get("analysis_group") if CORE[i]["stratum"] == "persuasion" else CORE[i]["source_item_id"]


def ci(x, c):
    x = np.asarray(x, float)
    if np.isfinite(x).sum() < 5:
        return dict(est=np.nan, lower=np.nan, upper=np.nan, p=np.nan, n_clusters=0, n=int(np.isfinite(x).sum()))
    e, lo, hi, p, n = fast_cluster_mean_ci(x, np.asarray(c))
    return dict(est=e, lower=lo, upper=hi, p=p, n_clusters=n, n=int(np.isfinite(x).sum()))


def pnorm(lp, L):
    v = np.array(list(lp.values()))
    return float(np.exp(lp[L] - np.logaddexp.reduce(v))) if L in lp else np.nan


def load(m, ds):
    D = pd.DataFrame(read_jsonl(ACTS / m / f"lit_prewrite_{ds}" / "rows.jsonl"))
    D["p_target"] = [pnorm(lp, t) for lp, t in zip(D.lp_letters, D.target)]
    D["p_own"] = [pnorm(lp, t) for lp, t in zip(D.lp_letters, D.own_prev)]
    D["cluster"] = D.item_id.map(cluster_of)
    D["episode"] = D.item_id + "|" + D.agent.astype(str) + "|" + D["round"].astype(str)
    base = D[D.arm == "none"].set_index("episode")
    for col in ("p_target", "p_own"):
        D["d_" + col] = D[col].to_numpy() - D.episode.map(base[col]).to_numpy()
    D["wrote_target"] = D.is_target.map(lambda x: np.nan if x is None else float(x))
    D["wrote_own"] = D.is_own_prev.map(lambda x: np.nan if x is None else float(x))
    D["mentions"] = D.mentions_peers.astype(float)
    return D


MEASURES = ["p_target", "d_p_target", "p_own", "wrote_target", "wrote_own", "mentions", "complete"]


def table(D, by):
    rows = []
    for key, g in D.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        for msr in MEASURES:
            rows.append(dict(zip(by if isinstance(by, list) else [by], key), measure=msr,
                             **ci(g[msr].astype(float), g.cluster)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="lit1200")
    a = ap.parse_args()
    D = load(a.model, a.dataset)
    n_ep = D.episode.nunique()
    print(f"[prewrite] {a.model}: {n_ep} episodes, {len(D)} rows, "
          f"{D[D.arm=='none'].changed.mean():.0%} of episodes are real switches", flush=True)

    # per arm x layer, and collapsed per arm kind
    T_arm = table(D, ["kind", "layer"])
    T_arm.to_csv(TABLES / f"lit_prewrite_{a.model}_arms.csv", index=False)

    # headline: the transplant against its equal-magnitude control, at each layer
    piv = T_arm[T_arm.measure == "d_p_target"].pivot_table(index="layer", columns="kind", values="est")
    print("\nshift toward the peers' answer vs the no-communication arm (d p_target):")
    print(piv.round(3).to_string())

    # the best layer for the plain transplant, then everything else at that layer
    cand = T_arm[(T_arm.measure == "d_p_target") & (T_arm.kind == "debate")].dropna(subset=["est"])
    best = int(cand.loc[cand.est.idxmax(), "layer"]) if len(cand) else None
    print(f"\nlargest plain-transplant effect at layer {best}")
    B = D[(D.layer == best) | D.arm.isin(["none", "debate_ref"])]
    T_best = table(B, ["kind"])
    T_best["layer"] = best
    T_best.to_csv(TABLES / f"lit_prewrite_{a.model}_layers.csv", index=False)
    for msr in ("d_p_target", "wrote_target", "mentions"):
        s = T_best[T_best.measure == msr].set_index("kind")
        print(f"\n{msr} at layer {best}:")
        print("  " + "  ".join(f"{k} {r.est:+.3f} [{r.lower:+.3f},{r.upper:+.3f}]" for k, r in s.iterrows() if np.isfinite(r.est)))

    # by domain and by whether the agent really switched
    table(B, ["kind", "domain"]).to_csv(TABLES / f"lit_prewrite_{a.model}_domain.csv", index=False)
    table(B, ["kind", "changed"]).to_csv(TABLES / f"lit_prewrite_{a.model}_switch.csv", index=False)

    # qualitative: the unpatched response next to the transplanted one, for the ten clearest episodes
    q = D[(D.kind == "debate") & (D.layer == best)].copy()
    q = q.sort_values("d_p_target", ascending=False).head(10)
    none = D[D.arm == "none"].set_index("episode")
    ref = D[D.arm == "debate_ref"].set_index("episode")
    out = []
    for _, r in q.iterrows():
        out.append(dict(item_id=r.item_id, agent=r.agent, round=r["round"], domain=r.domain, family=r.family,
                        own_prev=r.own_prev, peers_answer=r.target, wrote_in_debate=r.written,
                        d_p_target=r.d_p_target, p_target_none=none.loc[r.episode, "p_target"], p_target_patched=r.p_target,
                        mentions_none=none.loc[r.episode, "mentions_peers"], mentions_patched=r.mentions_peers,
                        text_none=none.loc[r.episode, "text"][:1200], text_patched=r.text[:1200],
                        text_debate=ref.loc[r.episode, "text"][:1200]))
    pd.DataFrame(out).to_csv(TABLES / f"lit_prewrite_{a.model}_quotes.csv", index=False)
    print(f"\n[done] tables in {TABLES}/lit_prewrite_{a.model}_*.csv")


if __name__ == "__main__":
    main()
