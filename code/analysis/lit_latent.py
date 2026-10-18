"""How much of a peer's influence survives as a state, how many directions carry it, and whether it carries the truth.

Reads ACTS/<M>/lit_latent_<dataset>/rows.jsonl (interp/latent_lit.py).

Quantities (all ratios of means over episodes, 95% item-cluster bootstrap; persuasion items clustered by claim):
  E_text   m(W_textj) - m(W_bare): what the peer's written response is worth, over its bare answer
  E_none   m(W_bare)  - m(W_none): what the bare answer alone is worth
  T(arm)   [m(arm) - m(W_bare in the same batch)] / E_text, the share of the written response reproduced by the state
  k_50     smallest k whose lower CI reaches half of the whole-state transfer at that layer
  truth    the same shifts measured on the margin toward the item's evaluation target, split by whether the peer was right

Tables: outputs/tables/lit_latent_<M>_{transfer,rank,controls,truth,summary}.csv|json
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

CORE = {r["id"]: r for r in load_core()}
cluster_of = lambda i: CORE[i]["meta"].get("analysis_group") if CORE[i]["stratum"] == "persuasion" else CORE[i]["source_item_id"]
B = 2000
SEED = 240091


def ratio_ci(num, den, clusters, B=B, seed=SEED):
    """Ratio of means with a cluster bootstrap; returns (est, lo, hi, n)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    ok = np.isfinite(num) & np.isfinite(den)
    num, den, cl = num[ok], den[ok], np.asarray(clusters)[ok]
    if len(num) < 5 or abs(den.sum()) < 1e-9:
        return np.nan, np.nan, np.nan, len(num)
    codes, uniq = pd.factorize(pd.Series(cl))
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=B).astype(np.float32)[:, codes]
    bs = (W * num).sum(1) / (W * den).sum(1)
    est = num.sum() / den.sum()
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return float(est), float(lo), float(hi), len(num)


def mean_ci(x, clusters, B=B, seed=SEED):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, cl = x[ok], np.asarray(clusters)[ok]
    if len(x) < 5:
        return np.nan, np.nan, np.nan, len(x)
    codes, uniq = pd.factorize(pd.Series(cl))
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=B).astype(np.float32)[:, codes]
    bs = (W * x).sum(1) / W.sum(1)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi), len(x)


def load(model, dataset, tag=""):
    d = ACTS / model / f"lit_latent_{dataset}{tag}"
    D = pd.DataFrame(read_jsonl(d / "rows.jsonl"))
    meta = json.load(open(d / "meta.json"))
    D["ep"] = D.item_id + "|" + D["round"].astype(str) + "|" + D.receiver.astype(str) + "|" + D.sender_agent.astype(str)
    D["cluster"] = D.item_id.map(cluster_of)
    D["m"] = [lp[s] - lp[o] if s in lp and o in lp else np.nan for lp, s, o in zip(D.lp_letters, D.sender, D.own)]

    def m_truth(lp, t):
        if not isinstance(t, str) or t not in lp or len(lp) < 2:
            return np.nan
        return lp[t] - max(v for k, v in lp.items() if k != t)
    D["m_truth"] = [m_truth(lp, t) for lp, t in zip(D.lp_letters, D.target)]
    return D, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="lit1200")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    D, meta = load(a.model, a.dataset, a.tag)
    tag = f"{a.model}_{a.dataset}"
    base = D[D.arm.isin(["W_none", "W_bare", "W_textj", "W_text"])].pivot_table(index="ep", columns="arm", values="m", aggfunc="first")
    base_t = D[D.arm.isin(["W_bare", "W_textj"])].pivot_table(index="ep", columns="arm", values="m_truth", aggfunc="first")
    info = D.groupby("ep").agg(cluster=("cluster", "first"), domain=("domain", "first"), family=("family", "first"),
                               sender_correct=("sender_correct", "first"), target=("target", "first"))
    base = base.join(info)
    base["E_text"] = base.W_textj - base.W_bare
    base["E_bare"] = base.W_bare - base.W_none
    base["E_text_truth"] = base_t.W_textj - base_t.W_bare
    chunk_base = D[D.arm == "W_bare_in_chunk"].groupby(["ep", "chunk"]).m.first()
    chunk_base_t = D[D.arm == "W_bare_in_chunk"].groupby(["ep", "chunk"]).m_truth.first()
    text_chunk = D[D.arm == "W_textj_in_chunk"].groupby("ep").m.first()

    rows = []
    for name, x in (("text channel (written response over bare answer)", base.E_text),
                    ("bare answer over no peer", base.E_bare)):
        e, lo, hi, n = mean_ci(x, base.cluster)
        rows.append(dict(quantity=name, est=e, lower=lo, upper=hi, n=n))
    pd.DataFrame(rows).to_csv(TABLES / f"lit_latent_{tag}_effects.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())

    P = D[~D.arm.isin(["W_none", "W_bare", "W_textj", "W_text", "W_bare_in_chunk", "W_textj_in_chunk"])].copy()
    P = P.join(base[["E_text", "cluster", "domain", "sender_correct", "E_text_truth"]], on="ep", rsuffix="_ep")
    P["m_base"] = [chunk_base.get((e, c), np.nan) for e, c in zip(P.ep, P.chunk)]
    P["mt_base"] = [chunk_base_t.get((e, c), np.nan) for e, c in zip(P.ep, P.chunk)]
    anc = P.arm.str.startswith("ablate")
    P.loc[anc, "m_base"] = [text_chunk.get(e, np.nan) for e in P.ep[anc]]
    P["shift"] = P.m - P.m_base
    P["shift_truth"] = P.m_truth - P.mt_base

    def tab(g):
        e, lo, hi, n = ratio_ci(g["shift"], g.E_text, g.cluster)
        s, slo, shi, _ = mean_ci(g["shift"], g.cluster)
        return dict(transfer=e, lower=lo, upper=hi, shift=s, shift_lower=slo, shift_upper=shi, n=n)

    out = [dict(arm=arm, layer=L, k=k, basis=bs, **tab(g)) for (arm, L, k, bs), g in
           P.groupby(["arm", "layer", P.k.fillna(-1), P.basis.fillna("-")], dropna=False)]
    T = pd.DataFrame(out).sort_values(["arm", "layer", "k"])
    T.to_csv(TABLES / f"lit_latent_{tag}_transfer.csv", index=False)
    full = T[T.arm == "full_state"]
    print("\nwhole-state transfer by layer:")
    print(full[["layer", "transfer", "lower", "upper", "shift", "n"]].round(3).to_string(index=False))
    print("\ncontrols:")
    print(T[T.arm.isin(["rotated", "random_vector", "other_position", "same_letter_other_item", "deranged_sender"])]
          [["arm", "layer", "transfer", "lower", "upper"]].round(3).to_string(index=False))

    # rank curves and k_50
    k50 = []
    for (L, bs), g in T[T.arm.isin(["rank", "rank_random_subspace", "rank_state_variance"])].groupby(["layer", "basis"]):
        ref = full[full.layer == L].transfer
        if not len(ref):
            continue
        half = 0.5 * float(ref.iloc[0])
        g = g.sort_values("k")
        hit = g[g.lower >= half]
        k50.append(dict(layer=L, basis=bs, full_transfer=float(ref.iloc[0]), half=half,
                        k_50=int(hit.k.iloc[0]) if len(hit) else None,
                        max_k_transfer=float(g.transfer.iloc[-1]), max_k=int(g.k.iloc[-1])))
    K = pd.DataFrame(k50)
    K.to_csv(TABLES / f"lit_latent_{tag}_rank.csv", index=False)
    print("\nrank needed for half the whole-state transfer:")
    print(K.round(3).to_string(index=False))

    # influence vs truth, for the whole-state arm at each layer and for the text channel
    tr = []
    for correct in (True, False):
        m = base.sender_correct == correct
        e, lo, hi, n = mean_ci(base.E_text[m], base.cluster[m])
        et, lot, hit_, _ = mean_ci(base.E_text_truth[m], base.cluster[m])
        tr.append(dict(channel="text", sender_correct=correct, toward_sender=e, ts_lower=lo, ts_upper=hi,
                       toward_target=et, tt_lower=lot, tt_upper=hit_, n=n))
        for L, g in P[(P.arm == "full_state") & (P.sender_correct == correct)].groupby("layer"):
            e, lo, hi, n = mean_ci(g["shift"], g.cluster)
            et, lot, hit_, _ = mean_ci(g["shift_truth"], g.cluster)
            tr.append(dict(channel=f"state L{int(L)}", sender_correct=correct, toward_sender=e, ts_lower=lo, ts_upper=hi,
                           toward_target=et, tt_lower=lot, tt_upper=hit_, n=n))
    TR = pd.DataFrame(tr)
    TR.to_csv(TABLES / f"lit_latent_{tag}_truth.csv", index=False)
    print("\ninfluence vs truth (margin shift in logits):")
    print(TR.round(3).to_string(index=False))

    dom = [dict(domain=d, **tab(g)) for (d,), g in P[P.arm == "full_state"].groupby(["domain"])]
    pd.DataFrame(dom).to_csv(TABLES / f"lit_latent_{tag}_by_domain.csv", index=False)
    best = full.sort_values("transfer", ascending=False).head(1)
    summary = dict(model=a.model, n_episodes=int(P.ep.nunique()), n_fit=meta.get("n_fit"),
                   text_channel=float(base.E_text.mean()), bare_channel=float(base.E_bare.mean()),
                   best_layer=None if not len(best) else int(best.layer.iloc[0]),
                   best_transfer=None if not len(best) else float(best.transfer.iloc[0]),
                   k_50=K.to_dict("records"))
    json.dump(summary, open(TABLES / f"lit_latent_{tag}_summary.json", "w"), indent=1, default=float)
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "k_50"}, indent=1, default=float))


if __name__ == "__main__":
    main()
