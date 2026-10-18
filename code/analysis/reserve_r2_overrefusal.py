"""Reserve R2 (V3 plan): over-caution dissociation, Q9 debates (release_600 = DISCOVERY; core gate frozen separately).

Question: when an agent publicly refuses a SAFE request (XSTest safe prompt, target assist) or abstains on an ANSWERABLE request
(CoCoNot, target answer), does its internal safety/answerability state still look like the "safe/answerable" state?
Slices (states pooled over agents and public turns 0..11; debate, pool and the shared turn 0; clusters = source_item_id):
  refusal:    a = refuse on over_refusal items (safe)    b = refuse on unsafe_request items (XSTest unsafe contrasts)
              c = assist on over_refusal items           (direct_harmful_request refusals reported as a second b)
  abstention: a = abstain on answerable items            b = abstain on should-abstain items      c = answer on answerable items
Read-outs at the decision token (prefilled "VERDICT:"): frozen refusal_boundary probe (L31; fit at prior_end of CPT receiver
worlds, so this is a position transfer, checked below) and frozen abstention probe (L16, answer position); DoM primary, logistic
reported; the unprinted caution margin m = lp(refuse|abstain) - lp(assist|answer).
Estimands (cluster bootstrap):
  position index R = (mean_a - mean_c) / (mean_b - mean_c): 0 = the over-cautious state reads like a correct helpful state,
  1 = like a genuine refusal of an unsafe request; fraction of a classified "internally safe" by the midpoint of b and c.
  reader control: the same read-outs from the minimal-observer and paraphrase readers on the identical debate text (turns 1..11).
  behaviour: private re-query (final turn) answer switches to assist/answer, a vs b.
  prediction: among over-cautious final-turn states (a), does the probe / margin predict private reversal; transcript baselines.
Core power: counts only from outputs/raw/Q9/core transcripts outside release_600.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
from common import *  # noqa
from hollow import Boot, auroc_table, oof_text_model, probe_scores

SEED = 240091
M_, DS, T_ = "Q9", "release_600", 12
SLICES = {
    "refusal": dict(fam_a="over_refusal", fam_b="unsafe_request", caution="refuse", helpful="assist", probe="refusal_boundary"),
    "abstention": dict(fam_a="calibrated_abstention", fam_b="calibrated_abstention", caution="abstain", helpful="answer", probe="abstention"),
}
CORE = {r["id"]: r for r in load_core()}


def opt_key(i, letter):
    if not isinstance(letter, str):
        return None
    for L, _, k in item_view(CORE[i])["options"] or []:
        if L == letter:
            return k
    return None


def key_letter(i, key):
    for L, _, k in item_view(CORE[i])["options"] or []:
        if k == key:
            return L
    return None


def caution_margin(lp, i, caution, helpful):
    if not isinstance(lp, dict):
        return np.nan
    a, b = key_letter(i, caution), key_letter(i, helpful)
    if a not in lp or b not in lp:
        return np.nan
    return lp[a] - lp[b]


def wmean(W, x, m):
    return (W[:, m] * x[m]).sum(1) / np.maximum(W[:, m].sum(1), 1e-9)


def load_states():
    fams = {"over_refusal", "unsafe_request", "direct_harmful_request", "calibrated_abstention"}
    idx = pd.DataFrame(read_jsonl(ACTS / M_ / f"replay_{DS}_{DS}_T{T_}" / "index.jsonl"))
    idx["family"] = idx.item_id.map(lambda i: family_of(CORE[i]))
    idx = idx[idx.family.isin(fams)].copy()
    fz = json.load(open(ACTS / M_ / "probes_frozen_cpt_core.json"))
    readers = {"participant": (f"replay_{DS}_{DS}_T{T_}", idx)}
    for nm, run in (("obsmin", f"reader_obsmin_{DS}_T{T_}"), ("para", f"reader_para_{DS}_T{T_}")):
        o = pd.DataFrame(read_jsonl(ACTS / M_ / run / "index.jsonl"))
        o["family"] = o.item_id.map(lambda i: family_of(CORE[i]))
        o = o[o.family.isin(fams)].copy()
        o["cond"] = "debate"
        readers[nm] = (run, o)
    out = []
    for nm, (run, df) in readers.items():
        df = df.copy()
        for pn in ("refusal_boundary", "abstention"):
            lg, dm = probe_scores(ACTS / M_ / run / "acts.npy", fz[pn])
            df[f"{pn}_dom"] = dm[df.row.to_numpy()]
            df[f"{pn}_log"] = lg[df.row.to_numpy()]
        df["reader"] = nm
        out.append(df)
    S = pd.concat(out, ignore_index=True)
    # public verdicts (the participant's own text at that turn) for every reader row
    pub, txt, conf = {}, {}, {}
    for f in ["t0.jsonl"] + [f"{c}_t{t}.jsonl" for c in ("debate", "pool") for t in range(1, T_)]:
        for x in read_jsonl(RAW / M_ / DS / f):
            if x["agent"] < 3:
                c = "t0" if x["turn"] == 0 else x["cond"]
                pub[(x["item_id"], c, x["agent"], x["turn"])] = x["verdict"]
                txt[(x["item_id"], c, x["agent"], x["turn"])] = x["text"]
                conf[(x["item_id"], c, x["agent"], x["turn"])] = x["confidence"]
    k = list(zip(S.item_id, S.cond, S.agent, S.turn))
    S["verdict"] = [pub.get(x) for x in k]
    S["text"] = [txt.get(x, "") for x in k]
    S["confidence"] = [conf.get(x) for x in k]
    S["vkey"] = [opt_key(i, v) for i, v in zip(S.item_id, S.verdict)]
    S["tkey"] = [opt_key(i, item_view(CORE[i])["target"]) for i in S.item_id]
    S["cluster"] = S.item_id.map(lambda i: CORE[i]["source_item_id"])
    S["xstest_type"] = S.item_id.map(lambda i: CORE[i]["meta"].get("xstest_type"))
    return S


def slice_groups(S, sl):
    c = SLICES[sl]
    if sl == "refusal":
        a = (S.family == c["fam_a"]) & (S.vkey == "refuse")
        b = (S.family == c["fam_b"]) & (S.vkey == "refuse")
        cc = (S.family == c["fam_a"]) & (S.vkey == "assist")
        b2 = (S.family == "direct_harmful_request") & (S.vkey == "refuse")
    else:
        fa = S.family == c["fam_a"]
        a = fa & (S.tkey == "answer") & (S.vkey == "abstain")
        b = fa & (S.tkey == "abstain") & (S.vkey == "abstain")
        cc = fa & (S.tkey == "answer") & (S.vkey == "answer")
        b2 = pd.Series(False, index=S.index)
    return a.to_numpy(), b.to_numpy(), cc.to_numpy(), b2.to_numpy()


def position_stats(D, sl, measure, B=2000):
    """R index, group means and 'internally safe' classification fractions on one reader/condition subset."""
    a, b, c, b2 = slice_groups(D, sl)
    keep = (a | b | c) & np.isfinite(D[measure].to_numpy(float))
    d = D[keep].reset_index(drop=True)
    a, b, c, _ = slice_groups(d, sl)
    x = d[measure].to_numpy(float)
    if a.sum() < 3 or b.sum() < 3 or c.sum() < 3:
        return None
    bt = Boot(d.cluster.to_numpy(), B=B)
    W = np.vstack([np.ones((1, len(d)), np.float32), bt.W])
    ma, mb, mc = wmean(W, x, a), wmean(W, x, b), wmean(W, x, c)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = (ma - mc) / (mb - mc)
    thr = (mb + mc) / 2
    sign = np.sign(mb - mc)  # side of b = "unsafe-like"
    safe_like = lambda g: (W[:, g] * ((x[g][None] - thr[:, None]) * sign[:, None] < 0)).sum(1) / np.maximum(W[:, g].sum(1), 1e-9)
    fa, fb, fc = safe_like(a), safe_like(b), safe_like(c)
    q = lambda v: [float(v[0]), float(np.nanquantile(v[1:], .025)), float(np.nanquantile(v[1:], .975))]
    return dict(n_a=int(a.sum()), n_b=int(b.sum()), n_c=int(c.sum()), clusters_a=int(d[a].cluster.nunique()),
                clusters_b=int(d[b].cluster.nunique()), clusters_c=int(d[c].cluster.nunique()),
                mean_a=q(ma), mean_b=q(mb), mean_c=q(mc), a_minus_c=q(ma - mc), b_minus_a=q(mb - ma), R=q(R),
                frac_a_safe_like=q(fa), frac_b_safe_like=q(fb), frac_c_safe_like=q(fc))


def core_power():
    rel = set(json.load(open(ROOT / "data/splits/splits.json"))["release_600"])
    out = {}
    for sl in SLICES:
        cnt = dict(a_states=0, b_states=0, c_states=0, a_items=set(), a_final_debate=0, a_final_pool=0)
        for f in ["t0.jsonl"] + [f"{c}_t{t}.jsonl" for c in ("debate", "pool") for t in range(1, 4)]:
            for x in read_jsonl(RAW / M_ / "core" / f):
                i = x["item_id"]
                if i in rel or x["agent"] >= 3:
                    continue
                fam = family_of(CORE[i])
                vk, tk = opt_key(i, x["verdict"]), opt_key(i, item_view(CORE[i])["target"])
                if sl == "refusal":
                    isa = fam == "over_refusal" and vk == "refuse"
                    isb = fam == "unsafe_request" and vk == "refuse"
                    isc = fam == "over_refusal" and vk == "assist"
                else:
                    ca = fam == "calibrated_abstention"
                    isa, isb, isc = ca and tk == "answer" and vk == "abstain", ca and tk == "abstain" and vk == "abstain", ca and tk == "answer" and vk == "answer"
                cnt["a_states"] += isa
                cnt["b_states"] += isb
                cnt["c_states"] += isc
                if isa:
                    cnt["a_items"].add(i)
                    if x["turn"] == 3:
                        cnt[f"a_final_{x['cond']}"] += 1
        cnt["a_items"] = len(cnt["a_items"])
        out[sl] = cnt
    return out


def main():
    import os
    import pickle
    cache = ROOT / "outputs/cache/r2_states.pkl"
    if os.path.exists(cache):
        S = pickle.load(open(cache, "rb"))
    else:
        S = load_states()
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        pickle.dump(S, open(cache, "wb"))
    S["one"] = "all"  # constant stratum: family-adjusted AUROC is undefined when the label is a family property
    summ = dict(split_role="discovery", model=M_, dataset=DS)
    rows = []
    # probe validity on this population (participant, all states): item-level gold label
    P = S[S.reader == "participant"]
    for pn, fam_pos, fam_neg in (("refusal_boundary", {"unsafe_request", "direct_harmful_request"}, {"over_refusal"}),
                                 ("abstention", None, None)):
        if pn == "refusal_boundary":
            d = P[P.family.isin(fam_pos | fam_neg)].copy()
            d["y"] = d.family.isin(fam_pos)
        else:
            d = P[P.family == "calibrated_abstention"].copy()
            d["y"] = d.tkey == "abstain"
        for kind in ("dom", "log"):
            A, _ = auroc_table(d.assign(s=d[f"{pn}_{kind}"]), ["s"], "y", "cluster", fam_col="one")
            r = A[A.scale == "pooled"].iloc[0]
            summ[f"validity_{pn}_{kind}"] = [r.auroc, r.lower, r.upper, int(r.n), int(r.n_clusters)]
    # position index per slice x reader x condition x measure
    S["caution_margin"] = [caution_margin(lp, i, SLICES["refusal" if f in ("over_refusal", "unsafe_request", "direct_harmful_request") else "abstention"]["caution"],
                                          SLICES["refusal" if f in ("over_refusal", "unsafe_request", "direct_harmful_request") else "abstention"]["helpful"])
                           for lp, i, f in zip(S.lp_letters, S.item_id, S.family)]
    pos = {}
    for sl, c in SLICES.items():
        for meas in (f"{c['probe']}_dom", f"{c['probe']}_log", "caution_margin"):
            for reader in ("participant", "obsmin", "para"):
                for cond in (("all", "debate", "pool", "t0") if reader == "participant" else ("debate",)):
                    D = S[(S.reader == reader) & ((S.cond == cond) if cond != "all" else True)]
                    if reader == "participant" and cond == "debate_turns1":
                        continue
                    r = position_stats(D, sl, meas)
                    if r:
                        pos[f"{sl}|{meas}|{reader}|{cond}"] = r
                        rows.append(dict(slice=sl, measure=meas, reader=reader, cond=cond, **{k: (v[0] if isinstance(v, list) else v) for k, v in r.items()},
                                         R_lo=r["R"][1], R_hi=r["R"][2], a_minus_c_lo=r["a_minus_c"][1], a_minus_c_hi=r["a_minus_c"][2],
                                         b_minus_a_lo=r["b_minus_a"][1], b_minus_a_hi=r["b_minus_a"][2]))
            # participant restricted to the observer's rows (debate turns 1..11) for a like-for-like reader comparison
            D = S[(S.reader == "participant") & (S.cond == "debate") & (S.turn >= 1)]
            r = position_stats(D, sl, meas)
            if r:
                pos[f"{sl}|{meas}|participant|debate_t1plus"] = r
                rows.append(dict(slice=sl, measure=meas, reader="participant", cond="debate_t1plus", **{k: (v[0] if isinstance(v, list) else v) for k, v in r.items()},
                                 R_lo=r["R"][1], R_hi=r["R"][2], a_minus_c_lo=r["a_minus_c"][1], a_minus_c_hi=r["a_minus_c"][2],
                                 b_minus_a_lo=r["b_minus_a"][1], b_minus_a_hi=r["b_minus_a"][2]))
    Tpos = pd.DataFrame(rows)
    # participant - observer on the over-cautious states themselves (paired by item/agent/turn)
    pair = {}
    for sl, c in SLICES.items():
        for meas in (f"{c['probe']}_dom", "caution_margin"):
            Pp = S[(S.reader == "participant") & (S.cond == "debate") & (S.turn >= 1)]
            J = Pp.merge(S[S.reader == "obsmin"][["item_id", "agent", "turn", meas]], on=["item_id", "agent", "turn"], suffixes=("", "_obs"))
            J = J.merge(S[S.reader == "para"][["item_id", "agent", "turn", meas]].rename(columns={meas: f"{meas}_para"}), on=["item_id", "agent", "turn"])
            a, b, cc, _ = slice_groups(J, sl)
            for gname, g in (("a", a), ("b", b), ("c", cc)):
                d = J[g]
                if len(d) < 3:
                    continue
                bt = Boot(d.cluster.to_numpy(), B=2000)
                for lab, v in (("participant_minus_observer", d[meas] - d[f"{meas}_obs"]), ("participant_minus_para", d[meas] - d[f"{meas}_para"])):
                    x = v.to_numpy(float)
                    bs = (bt.W * x).sum(1) / bt.W.sum(1)
                    pair[f"{sl}|{meas}|{gname}|{lab}"] = [float(x.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975)), len(d), int(d.cluster.nunique())]
    # behaviour: over-caution rates by condition/turn; private re-query of the final public turn
    priv = {}
    for x in read_jsonl(RAW / M_ / DS / "private.jsonl"):
        priv[(x["item_id"], x["cond"].replace("private_", ""), x["agent"])] = x["verdict"]
    Pf = S[(S.reader == "participant") & (S.turn == T_ - 1)].copy()
    Pf["priv_key"] = [opt_key(i, priv.get((i, c, ag))) for i, c, ag in zip(Pf.item_id, Pf.cond, Pf.agent)]
    beh = {}
    for sl, c in SLICES.items():
        for cond in ("debate", "pool"):
            d = Pf[Pf.cond == cond]
            a, b, cc, _ = slice_groups(d, sl)
            for gname, g in (("a", a), ("b", b), ("c", cc)):
                dd = d[g & d.priv_key.notna()]
                if len(dd) == 0:
                    continue
                flip = (dd.priv_key == c["helpful"]).to_numpy(float) if gname != "c" else (dd.priv_key == c["caution"]).to_numpy(float)
                bt = Boot(dd.cluster.to_numpy(), B=2000)
                bs = (bt.W * flip).sum(1) / bt.W.sum(1)
                beh[f"{sl}|{cond}|{gname}|private_flip"] = [float(flip.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975)), len(dd), int(dd.cluster.nunique())]
        # over-caution rate by condition and turn (safe/answerable items)
        c0 = SLICES[sl]
        fam = c0["fam_a"]
        d = S[(S.reader == "participant") & (S.family == fam) & (S.tkey == c0["helpful"])]
        beh[f"{sl}|overcaution_rate_by_cond_turn"] = d.assign(o=(d.vkey == c0["caution"]).astype(float)).groupby(["cond", "turn"]).o.mean().round(3)
        beh[f"{sl}|overcaution_rate_by_cond_turn"] = {f"{k[0]}|{k[1]}": float(v) for k, v in beh[f"{sl}|overcaution_rate_by_cond_turn"].items()}
        # paired debate - pool over-caution at the final turn, item level
        dd = d[d.turn.isin([T_ - 1])].assign(o=lambda x: (x.vkey == c0["caution"]).astype(float)).groupby(["item_id", "cluster", "cond"]).o.mean().unstack("cond").dropna()
        if len(dd):
            bt = Boot(dd.index.get_level_values("cluster").to_numpy(), B=2000)
            x = (dd.debate - dd.pool).to_numpy()
            bs = (bt.W * x).sum(1) / bt.W.sum(1)
            beh[f"{sl}|final_overcaution_debate_minus_pool"] = [float(x.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975)), len(dd)]
    # prediction: among over-cautious states at the final public turn (debate + pool), private reversal to the helpful option
    pred = {}
    for sl, c in SLICES.items():
        d = Pf.copy()
        a, _, _, _ = slice_groups(d, sl)
        d = d[a & d.priv_key.notna()].reset_index(drop=True)
        d["reverse"] = d.priv_key == c["helpful"]
        d["neg_probe"] = -d[f"{c['probe']}_dom"]
        d["neg_margin"] = -d.caution_margin
        d["neg_conf"] = -d.confidence.astype(float)
        y = d.reverse.to_numpy(bool)
        pred[f"{sl}_counts"] = dict(n=len(d), n_reverse=int(y.sum()), clusters=int(d.cluster.nunique()))
        if y.sum() >= 5 and (~y).sum() >= 5:
            g = d.cluster.to_numpy()
            d["tfidf"] = oof_text_model(d.text.tolist(), y, g, kind="tfidf") if d.cluster.nunique() >= 5 else np.nan
            A, Ad = auroc_table(d, [p for p in ("neg_probe", "neg_margin", "neg_conf", "tfidf") if d[p].notna().all()], "reverse", "cluster",
                                fam_col="one", ref_pairs=[("neg_probe", "tfidf"), ("neg_margin", "tfidf"), ("neg_probe", "neg_conf")])
            pred[sl] = A[A.scale == "pooled"][["predictor", "auroc", "lower", "upper", "n", "n_pos", "n_clusters"]].to_dict("records")
            pred[f"{sl}_diffs"] = Ad[Ad.scale == "pooled"][["contrast", "diff", "lower", "upper"]].to_dict("records")
    # transcript baseline for the dissociation itself: can the agent's own message tell safe-refused from unsafe-refused?
    txtb = {}
    for sl, c in SLICES.items():
        d = S[(S.reader == "participant") & (S.cond == "debate") & (S.turn >= 1)].copy()
        a, b, _, _ = slice_groups(d, sl)
        d = d[a | b].reset_index(drop=True)
        a, b, _, _ = slice_groups(d, sl)
        d["is_a"] = a
        d["neg_probe"] = -d[f"{c['probe']}_dom"]
        d["neg_margin"] = -d.caution_margin
        obs = S[S.reader == "obsmin"][["item_id", "agent", "turn", f"{c['probe']}_dom", "caution_margin"]]
        d = d.merge(obs, on=["item_id", "agent", "turn"], suffixes=("", "_obs"), how="left")
        d["neg_probe_obs"] = -d[f"{c['probe']}_dom_obs"]
        d["neg_margin_obs"] = -d["caution_margin_obs"]
        y = d.is_a.to_numpy(bool)
        if y.sum() >= 5 and (~y).sum() >= 5:
            d["tfidf"] = oof_text_model(d.text.tolist(), y, d.cluster.to_numpy(), kind="tfidf")
            preds = ["neg_probe", "neg_margin", "neg_probe_obs", "neg_margin_obs", "tfidf"]
            A, Ad = auroc_table(d, preds, "is_a", "cluster", fam_col="one",
                                ref_pairs=[("neg_probe", "tfidf"), ("neg_probe", "neg_probe_obs"), ("neg_margin", "neg_margin_obs")])
            txtb[sl] = A[A.scale == "pooled"][["predictor", "auroc", "lower", "upper", "n", "n_pos", "n_clusters"]].to_dict("records")
            txtb[f"{sl}_diffs"] = Ad[Ad.scale == "pooled"][["contrast", "diff", "lower", "upper"]].to_dict("records")
    summ.update(position=pos, reader_pairs=pair, behaviour=beh, prediction=pred, dissociation_vs_text=txtb, core_power=core_power())
    tag = f"{M_}_{DS}_T{T_}"
    Tpos.to_csv(TABLES / f"r2_position_{tag}.csv", index=False)
    S.drop(columns=["text", "lp_letters"]).to_csv(TABLES / f"r2_states_{tag}.csv", index=False)
    json.dump(summ, open(TABLES / f"r2_summary_{tag}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    pd.set_option("display.width", 250)
    print(json.dumps({k: summ[k] for k in summ if k.startswith("validity")}, indent=1))
    show = Tpos[["slice", "measure", "reader", "cond", "n_a", "n_b", "n_c", "clusters_a", "mean_a", "mean_b", "mean_c", "R", "R_lo", "R_hi",
                 "frac_a_safe_like", "frac_b_safe_like", "frac_c_safe_like"]]
    print(show.round(3).to_string())
    print(json.dumps(dict(reader_pairs=pair, behaviour=beh, prediction=pred, dissociation_vs_text=txtb, core_power=summ["core_power"]), indent=1, default=float))


if __name__ == "__main__":
    main()
