"""Does a decision state carry across models, and does it need the same number of directions as within one model?

Reads ACTS/<receiver>/lit_latentx_<sender>to<receiver>_<dataset>/{rows.jsonl, map.json} (interp/latent_cross.py).

Quantities (ratios of means over episodes, 95% item-cluster bootstrap; persuasion items clustered by claim):
  E_text     m(W_textj) - m(W_bare): what the peer's written response is worth over its bare answer
  T(arm)     [m(arm) - m(W_bare in the same batch)] / E_text, the share of the written response the transplant reproduces
  ceiling    T(within_model): the receiver's own state on the same text, the same measurement without a space change
  k_50       smallest k whose lower CI reaches half the whole-state transfer, cross-model and within-model
  fidelity   transfer in terciles of the per-episode mapping cosine: does a better-carried state move the decision more?
  truth      the same shifts measured on the margin toward the item's evaluation target, split by whether the sender was right

Tables: outputs/tables/lit_latentx_<S>to<R>_{effects,transfer,rank,map,fidelity,truth,by_domain}.csv and _summary.json
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
WORLDS = ["W_none", "W_bare", "W_textj", "W_text"]


def _boot(cl, n, seed=SEED):
    codes, uniq = pd.factorize(pd.Series(cl))
    rng = np.random.default_rng(seed)
    return rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=B).astype(np.float32)[:, codes]


def ratio_ci(num, den, clusters):
    num, den = np.asarray(num, float), np.asarray(den, float)
    ok = np.isfinite(num) & np.isfinite(den)
    num, den, cl = num[ok], den[ok], np.asarray(clusters)[ok]
    if len(num) < 5 or abs(den.sum()) < 1e-9:
        return np.nan, np.nan, np.nan, len(num)
    W = _boot(cl, len(num))
    bs = (W * num).sum(1) / (W * den).sum(1)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return float(num.sum() / den.sum()), float(lo), float(hi), len(num)


def mean_ci(x, clusters):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, cl = x[ok], np.asarray(clusters)[ok]
    if len(x) < 5:
        return np.nan, np.nan, np.nan, len(x)
    W = _boot(cl, len(x))
    bs = (W * x).sum(1) / W.sum(1)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi), len(x)


def load(sender, receiver, dataset, tag=""):
    d = ACTS / receiver / f"lit_latentx_{sender}to{receiver}_{dataset}{tag}"
    D = pd.DataFrame(read_jsonl(d / "rows.jsonl"))
    meta = json.load(open(d / "meta.json"))
    mp = json.load(open(d / "map.json"))
    D["ep"] = D.item_id + "|" + D["round"].astype(str) + "|" + D.receiver.astype(str) + "|" + D.sender_agent.astype(str)
    D["cluster"] = D.item_id.map(cluster_of)
    D["m"] = [lp.get(s, np.nan) - lp.get(o, np.nan) if lp else np.nan for lp, s, o in zip(D.lp_letters, D.sender, D.own)]

    def m_truth(lp, t):
        if not lp or not isinstance(t, str) or t not in lp or len(lp) < 2:
            return np.nan
        return lp[t] - max(v for k, v in lp.items() if k != t)
    D["m_truth"] = [m_truth(lp, t) for lp, t in zip(D.lp_letters, D.target)]
    if "depth" not in D:
        D["depth"] = np.nan
    return D, meta, mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sender", required=True)
    ap.add_argument("--receiver", required=True)
    ap.add_argument("--dataset", default="lit1200")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    D, meta, mp = load(a.sender, a.receiver, a.dataset, a.tag)
    pair = f"{a.sender}to{a.receiver}"
    f_ctrl = meta["f_ctrl"]

    base = D[D.arm.isin(WORLDS)].pivot_table(index="ep", columns="arm", values="m", aggfunc="first")
    base_t = D[D.arm.isin(["W_bare", "W_textj"])].pivot_table(index="ep", columns="arm", values="m_truth", aggfunc="first")
    info = D.groupby("ep").agg(cluster=("cluster", "first"), domain=("domain", "first"), family=("family", "first"),
                               sender_correct=("sender_correct", "first"), target=("target", "first"))
    base = base.join(info)
    base["E_text"] = base.W_textj - base.W_bare
    base["E_bare"] = base.W_bare - base.W_none
    base["E_text_truth"] = base_t.W_textj - base_t.W_bare
    rows = [dict(quantity=q, est=e, lower=lo, upper=hi, n=n) for q, (e, lo, hi, n) in
            (("text channel (written response over bare answer)", mean_ci(base.E_text, base.cluster)),
             ("bare answer over no peer", mean_ci(base.E_bare, base.cluster)))]
    pd.DataFrame(rows).to_csv(TABLES / f"lit_latentx_{pair}_effects.csv", index=False)
    print(f"[{pair}] {len(base)} episodes, receiver transcripts {meta['transcripts']}, "
          f"d {meta['d_sender']} -> {meta['d_receiver']}")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # per-episode mapping fidelity
    Q = D[D.arm.str.startswith("map_quality")]
    fid = Q[(Q.depth == f_ctrl) & (Q.arm == "map_quality_ridge")].set_index("ep")[["cos", "norm_ratio"]]
    fid.columns = ["map_cos", "map_norm"]

    chunk_base = D[D.arm == "W_bare_in_chunk"].groupby(["ep", "chunk"]).m.first()
    chunk_base_t = D[D.arm == "W_bare_in_chunk"].groupby(["ep", "chunk"]).m_truth.first()
    P = D[~D.arm.isin(WORLDS + ["W_bare_in_chunk"]) & ~D.arm.str.startswith("map_quality")].copy()
    P = P.join(base[["E_text", "cluster", "domain", "sender_correct", "E_text_truth"]], on="ep", rsuffix="_ep")
    P["m_base"] = [chunk_base.get((e, c), np.nan) for e, c in zip(P.ep, P.chunk)]
    P["mt_base"] = [chunk_base_t.get((e, c), np.nan) for e, c in zip(P.ep, P.chunk)]
    P["shift"] = P.m - P.m_base
    P["shift_truth"] = P.m_truth - P.mt_base

    def tab(g):
        e, lo, hi, n = ratio_ci(g["shift"], g.E_text, g.cluster)
        s, slo, shi, _ = mean_ci(g["shift"], g.cluster)
        return dict(transfer=e, lower=lo, upper=hi, shift=s, shift_lower=slo, shift_upper=shi, n=n)

    T = pd.DataFrame([dict(arm=arm, depth=f, layer=L, k=k, basis=bs, **tab(g)) for (arm, f, L, k, bs), g in
                      P.groupby(["arm", P.depth.fillna(-1), "layer", P.k.fillna(-1), P.basis.fillna("-")], dropna=False)])
    T = T.sort_values(["arm", "depth", "k"])
    T.to_csv(TABLES / f"lit_latentx_{pair}_transfer.csv", index=False)
    whole = T[T.arm.isin(["mapped_ridge", "mapped_proc", "within_model"])]
    print("\nwhole-state transfer by relative depth:")
    print(whole.pivot_table(index="depth", columns="arm", values="transfer").round(3).to_string())
    print("\ncontrols at the primary depth:")
    ctrl = T[(T.depth == f_ctrl) & T.arm.isin(["mapped_ridge", "mapped_proc", "within_model", "mapped_shuffled_fit",
                                               "mapped_mean_sender", "mapped_same_letter_other_item", "mapped_deranged",
                                               "raw_unmapped", "random_vector", "rotated", "other_position"])]
    print(ctrl[["arm", "transfer", "lower", "upper", "shift", "n"]].round(3).to_string(index=False))

    # rank: k_50 cross-model vs within-model
    k50 = []
    for (f, arm, bs), g in T[T.arm.isin(["rank_cross", "rank_within"])].groupby(["depth", "arm", "basis"]):
        ref = T[(T.arm == ("within_model" if arm == "rank_within" else "mapped_ridge")) & (T.depth == f)].transfer
        if not len(ref) or not np.isfinite(ref.iloc[0]):
            continue
        half = 0.5 * float(ref.iloc[0])
        g = g.sort_values("k")
        hit = g[g.lower >= half]
        k50.append(dict(depth=f, arm=arm, basis=bs, whole_state=float(ref.iloc[0]), half=half,
                        k_50=int(hit.k.iloc[0]) if len(hit) else None, max_k=int(g.k.iloc[-1]),
                        transfer_at_max_k=float(g.transfer.iloc[-1])))
    K = pd.DataFrame(k50)
    K.to_csv(TABLES / f"lit_latentx_{pair}_rank.csv", index=False)
    print("\ndirections needed for half the whole-state transfer:")
    print(K.round(3).to_string(index=False))

    # mapping quality: held-out cosine and retrieval, and the per-episode cosine on the evaluation half
    MQ = pd.DataFrame(mp["quality"])
    for kind in ("ridge", "proc"):
        g = Q[Q.arm == f"map_quality_{kind}"]
        for f, gg in g.groupby("depth"):
            e, lo, hi, n = mean_ci(gg["cos"], gg.ep.map(base.cluster))
            MQ = pd.concat([MQ, pd.DataFrame([dict(map=kind, f_sender=f, f_recv=f, split="eval_episodes", cos=e,
                                                   cos_lo=lo, cos_hi=hi, norm_ratio=float(gg.norm_ratio.mean()), n=n)])])
    MQ.to_csv(TABLES / f"lit_latentx_{pair}_map.csv", index=False)
    print("\nmapping quality at equal relative depth (held-out items):")
    print(MQ[(MQ.f_sender == MQ.f_recv) & (MQ.split == "heldout")][["map", "f_sender", "lam_rel", "cos", "top1", "mrr", "n"]]
          .round(3).to_string(index=False))

    # does a better-carried state move the decision more?
    F = P[(P.arm == "mapped_ridge") & (P.depth == f_ctrl)].drop(columns=["cos", "norm_ratio"], errors="ignore").join(fid, on="ep")
    frows = []
    if len(F) >= 15 and F.map_cos.notna().any():
        F["tercile"] = pd.qcut(F.map_cos, 3, labels=["low", "mid", "high"], duplicates="drop")
        for t, g in F.groupby("tercile", observed=True):
            frows.append(dict(tercile=str(t), cos_mean=float(g.map_cos.mean()), **tab(g)))
        r = F[["map_cos", "shift"]].dropna()
        frows.append(dict(tercile="spearman(cos, shift)", cos_mean=np.nan,
                          transfer=float(r.map_cos.rank().corr(r["shift"].rank())), lower=np.nan, upper=np.nan,
                          shift=np.nan, shift_lower=np.nan, shift_upper=np.nan, n=len(r)))
    FD = pd.DataFrame(frows)
    FD.to_csv(TABLES / f"lit_latentx_{pair}_fidelity.csv", index=False)
    if len(FD):
        print("\ntransfer by mapping fidelity (per-episode cosine):")
        print(FD[["tercile", "cos_mean", "transfer", "lower", "upper", "n"]].round(3).to_string(index=False))

    # influence vs truth
    tr = []
    for correct in (True, False):
        m = base.sender_correct == correct
        e, lo, hi, n = mean_ci(base.E_text[m], base.cluster[m])
        et, lot, hit_, _ = mean_ci(base.E_text_truth[m], base.cluster[m])
        tr.append(dict(channel="text", sender_correct=correct, toward_sender=e, ts_lower=lo, ts_upper=hi,
                       toward_target=et, tt_lower=lot, tt_upper=hit_, n=n))
        for arm in ("mapped_ridge", "within_model"):
            g = P[(P.arm == arm) & (P.depth == f_ctrl) & (P.sender_correct == correct)]
            if not len(g):
                continue
            e, lo, hi, n = mean_ci(g["shift"], g.cluster)
            et, lot, hit_, _ = mean_ci(g["shift_truth"], g.cluster)
            tr.append(dict(channel=arm, sender_correct=correct, toward_sender=e, ts_lower=lo, ts_upper=hi,
                           toward_target=et, tt_lower=lot, tt_upper=hit_, n=n))
    TR = pd.DataFrame(tr)
    TR.to_csv(TABLES / f"lit_latentx_{pair}_truth.csv", index=False)
    print("\ninfluence vs truth (margin shift in logits):")
    print(TR.round(3).to_string(index=False))

    dom = [dict(domain=d, arm=arm, **tab(g)) for (d, arm), g in
           P[P.arm.isin(["mapped_ridge", "within_model"]) & (P.depth == f_ctrl)].groupby(["domain", "arm"])]
    pd.DataFrame(dom).to_csv(TABLES / f"lit_latentx_{pair}_by_domain.csv", index=False)

    getT = lambda arm, f: (lambda g: None if not len(g) else float(g.transfer.iloc[0]))(T[(T.arm == arm) & (T.depth == f)])
    best = whole[whole.arm == "mapped_ridge"].sort_values("transfer", ascending=False).head(1)
    summary = dict(sender=a.sender, receiver=a.receiver, transcripts=meta["transcripts"], d_sender=meta["d_sender"],
                   d_receiver=meta["d_receiver"], n_episodes=int(P.ep.nunique()), n_map=meta["n_map"],
                   text_channel=float(base.E_text.mean()), primary_depth=f_ctrl,
                   mapped_ridge=getT("mapped_ridge", f_ctrl), mapped_proc=getT("mapped_proc", f_ctrl),
                   within_model=getT("within_model", f_ctrl), shuffled_map=getT("mapped_shuffled_fit", f_ctrl),
                   raw_unmapped=getT("raw_unmapped", f_ctrl), random_vector=getT("random_vector", f_ctrl),
                   best_depth=None if not len(best) else float(best.depth.iloc[0]),
                   best_transfer=None if not len(best) else float(best.transfer.iloc[0]),
                   k_50=K.to_dict("records"))
    json.dump(summary, open(TABLES / f"lit_latentx_{pair}_summary.json", "w"), indent=1, default=float)
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "k_50"}, indent=1, default=float))


if __name__ == "__main__":
    main()
