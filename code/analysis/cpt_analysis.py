"""CPT estimands from world scores.

Endpoint Y = adoption margin m_adv = log p(advocated option) - log p(other focal option) at the receiver's decision.
Per item (and template), with the message disagreeing with the receiver's prior unless stated:
  Q   evidence-quality effect      mean_P [Y(strong,P) - Y(weak,P)]                P in {expert, anon}
  P   provenance effect            mean_Q [Y(Q,expert) - Y(Q,anon)]
  IQP quality x provenance         [Y(strong,exp)-Y(strong,anon)] - [Y(weak,exp)-Y(weak,anon)]
  C   correctness effect (targets) mean_P [Y(correct adv,P) - Y(incorrect adv,P)]   (strong evidence)
  ICP correctness x provenance     [Y(C+,exp)-Y(C+,anon)] - [Y(C-,exp)-Y(C-,anon)]
  Pexp_none / Panon_none           label ablation: Y(Q,expert)-Y(Q,none), Y(Q,anon)-Y(Q,none)
  S   social relation              Y_agree(...) - Y_disagree(...) on m_adv averaged over Q,P
  Sneu                             neutral-message baseline: m_prior under a no-content message (expert vs anon)
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
from stats import bh_fdr, fast_cluster_mean_ci

CONTRASTS = ["Q", "P", "IQP", "C", "ICP", "Pexp_none", "Panon_none", "S", "P_agree", "IQP_agree", "P_neutral"]


def load_worlds(model, run):
    p = ACTS / model / run / "worlds.jsonl"
    df = pd.DataFrame(read_jsonl(p))
    core = {r["id"]: r for r in load_core()}
    df["cluster"] = df.item_id.map(lambda i: core[i]["meta"].get("analysis_group") if core[i]["stratum"] == "persuasion" else core[i]["source_item_id"])
    return df


def item_contrasts(df, endpoint="m_adv"):
    """One row per item x template x advocated option with all contrasts."""
    out = []
    key = ["item_id", "template"]
    for (iid, tpl), g in df.groupby(key, sort=False):
        o1, o2 = g.o1.iloc[0], g.o2.iloc[0]
        has_t = bool(g.has_target.iloc[0])
        cell = {}
        for r in g.itertuples(index=False):
            cell[(r.adv, r.qual, r.prov, r.prior)] = getattr(r, endpoint) if r.adv is not None else None
        neutral = {(r.prov, r.prior): r.m_prior for r in g.itertuples(index=False) if r.qual == "neutral"}
        base = dict(item_id=iid, template=tpl, family=g.family.iloc[0], domain=g.domain.iloc[0], stratum=g.stratum.iloc[0],
                    cluster=g.cluster.iloc[0], has_target=has_t, author=g.author.iloc[0])
        for adv in (o1, o2):
            dis = o2 if adv == o1 else o1
            Y = lambda q, p, prior=dis: cell.get((adv, q, p, prior))
            try:
                row = dict(base, adv=adv, adv_correct=(adv == o1) if has_t else None,
                           Q=np.mean([Y("strong", p) - Y("weak", p) for p in ("expert", "anon")]),
                           P=np.mean([Y(q, "expert") - Y(q, "anon") for q in ("strong", "weak")]),
                           IQP=(Y("strong", "expert") - Y("strong", "anon")) - (Y("weak", "expert") - Y("weak", "anon")),
                           Pexp_none=np.mean([Y(q, "expert") - Y(q, "none") for q in ("strong", "weak")]),
                           Panon_none=np.mean([Y(q, "anon") - Y(q, "none") for q in ("strong", "weak")]),
                           S=np.mean([Y(q, p, adv) - Y(q, p) for q in ("strong", "weak") for p in ("expert", "anon")]),
                           P_agree=np.mean([Y(q, "expert", adv) - Y(q, "anon", adv) for q in ("strong", "weak")]),
                           IQP_agree=(Y("strong", "expert", adv) - Y("strong", "anon", adv)) - (Y("weak", "expert", adv) - Y("weak", "anon", adv)),
                           Y_dis_strong_exp=Y("strong", "expert"), Y_dis_strong_anon=Y("strong", "anon"),
                           Y_dis_weak_exp=Y("weak", "expert"), Y_dis_weak_anon=Y("weak", "anon"),
                           Y_dis_strong_none=Y("strong", "none"), Y_dis_weak_none=Y("weak", "none"))
            except TypeError:
                continue
            # neutral message: margin toward the receiver's prior (retention) under expert vs anon header
            if ("expert", dis) in neutral and ("anon", dis) in neutral:
                row["P_neutral"] = neutral[("expert", dis)] - neutral[("anon", dis)]
            out.append(row)
    ic = pd.DataFrame(out)
    # correctness x provenance on items with a target (strong evidence, disagreement): pair the two advocacy directions
    rows = []
    for (iid, tpl), g in ic[ic.has_target].groupby(["item_id", "template"]):
        if len(g) != 2:
            continue
        cp = g[g.adv_correct == True].iloc[0]
        cm = g[g.adv_correct == False].iloc[0]
        C = np.mean([cp.Y_dis_strong_exp - cm.Y_dis_strong_exp, cp.Y_dis_strong_anon - cm.Y_dis_strong_anon])
        ICP = (cp.Y_dis_strong_exp - cp.Y_dis_strong_anon) - (cm.Y_dis_strong_exp - cm.Y_dis_strong_anon)
        rows.append(dict(item_id=iid, template=tpl, C=C, ICP=ICP,
                         P_correct=cp.Y_dis_strong_exp - cp.Y_dis_strong_anon, P_incorrect=cm.Y_dis_strong_exp - cm.Y_dis_strong_anon))
    cc = pd.DataFrame(rows)
    return ic, cc


def summarize(ic, cc, model, run):
    # item-level: average over advocacy directions (and templates) so each item contributes one value
    agg = ic.groupby(["item_id", "family", "domain", "cluster"], as_index=False)[
        ["Q", "P", "IQP", "Pexp_none", "Panon_none", "S", "P_agree", "IQP_agree", "P_neutral"]].mean()
    if len(cc):
        agg = agg.merge(cc.groupby("item_id", as_index=False)[["C", "ICP", "P_correct", "P_incorrect"]].mean(), on="item_id", how="left")
    rows = []
    levels = [("ALL", "all", agg)] + [(d, "domain", g) for d, g in agg.groupby("domain")] + [(f, "family", g) for f, g in agg.groupby("family")]
    for name, level, g in levels:
        for c in [c for c in CONTRASTS + ["P_correct", "P_incorrect"] if c in g.columns]:
            vals = g[c].to_numpy(float)
            est, lo, hi, p, n = fast_cluster_mean_ci(vals, g.cluster.to_numpy())
            if not n:
                continue
            sd = np.nanstd(vals, ddof=1) if np.isfinite(vals).sum() > 2 else np.nan
            rows.append(dict(model=model, run=run, level=level, group=name, contrast=c, estimate=est, lower=lo, upper=hi,
                             p_value=p, n_items=int(np.isfinite(vals).sum()), n_clusters=n, dz=est / sd if sd else np.nan))
    tab = pd.DataFrame(rows)
    tab["q_value"] = np.nan
    fam = tab.level == "family"
    for c, g in tab[fam].groupby("contrast"):
        tab.loc[g.index, "q_value"] = bh_fdr(g.p_value)
    # domain-balanced pooled estimate (mean of the three domain means)
    bal = tab[tab.level == "domain"].groupby("contrast").estimate.mean().rename("domain_balanced")
    tab = tab.merge(bal, left_on="contrast", right_index=True, how="left")
    # sign consistency across families
    signs = {}
    for c, g in tab[fam].groupby("contrast"):
        allsign = np.sign(tab[(tab.level == "all") & (tab.contrast == c)].estimate.iloc[0])
        signs[c] = dict(same_sign=int((np.sign(g.estimate) == allsign).sum()), n_families=len(g),
                        ci_excl0_same=int(((np.sign(g.estimate) == allsign) & ((g.lower > 0) | (g.upper < 0))).sum()))
    return agg, tab, signs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", default="cpt_core")
    ap.add_argument("--endpoint", default="m_adv")
    a = ap.parse_args()
    df = load_worlds(a.model, a.run)
    rej_strict = set(json.load(open(ROOT / "data/mad_interp/rejected_items_strict.json"))) if (ROOT / "data/mad_interp/rejected_items_strict.json").exists() else set()
    ic, cc = item_contrasts(df, a.endpoint)
    agg, tab, signs = summarize(ic, cc, a.model, a.run)
    tag = f"{a.model}_{a.run}_{a.endpoint}"
    ic.to_csv(TABLES / f"cpt_{tag}_itemcontrasts.csv", index=False)
    agg.to_csv(TABLES / f"cpt_{tag}_items.csv", index=False)
    tab.to_csv(TABLES / f"cpt_{tag}_summary.csv", index=False)
    # robustness: strict stimulus validation subset
    if rej_strict:
        ic2 = ic[~ic.item_id.isin(rej_strict)]
        cc2 = cc[~cc.item_id.isin(rej_strict)] if len(cc) else cc
        _, tab2, signs2 = summarize(ic2, cc2, a.model, a.run + "_strict")
        tab2.to_csv(TABLES / f"cpt_{tag}_summary_strict.csv", index=False)
    # pre-registered robustness: length-matched half (|strong - weak| word gap <= median)
    lm_p = ROOT / "data/mad_interp/length_matched_items.json"
    if lm_p.exists():
        lm = set(json.load(open(lm_p))["items"])
        ic3 = ic[ic.item_id.isin(lm)]
        cc3 = cc[cc.item_id.isin(lm)] if len(cc) else cc
        _, tab3, _ = summarize(ic3, cc3, a.model, a.run + "_lengthmatched")
        tab3.to_csv(TABLES / f"cpt_{tag}_summary_lengthmatched.csv", index=False)
    json.dump(signs, open(TABLES / f"cpt_{tag}_signs.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    print(tab[tab.level != "family"][["level", "group", "contrast", "estimate", "lower", "upper", "dz", "n_items"]].round(3).to_string())
    print(json.dumps(signs, indent=0))


def adoption_table(model, run="cpt_core"):
    """Frozen table for F4: P(receiver switches to the advocated option | disagreement) by family x correctness x provenance x
    quality, with item-cluster bootstrap CIs."""
    W = load_worlds(model, run)
    W = W[W.qual.isin(["strong", "weak"]) & (W.agree == False)].copy()  # noqa: E712
    W["adopt"] = (W.argmax_label == W.adv).astype(float)
    W["advocacy"] = np.where(~W.has_target, "no_target", np.where(W.adv == W.o1, "correct", "wrong"))
    rows = []
    for level, keycols in (("all", []), ("domain", ["domain"]), ("family", ["family"])):
        for keys, g in (W.groupby(keycols + ["advocacy", "prov", "qual"]) if keycols else W.groupby(["advocacy", "prov", "qual"])):
            keys = keys if isinstance(keys, tuple) else (keys,)
            per = g.groupby(["item_id", "cluster"]).adopt.mean().reset_index()
            est, lo, hi, p, n = fast_cluster_mean_ci(per.adopt.to_numpy(), per.cluster.to_numpy())
            d = dict(level=level, group=keys[0] if keycols else "ALL", advocacy=keys[-3], prov=keys[-2], qual=keys[-1],
                     p_adopt=est, lower=lo, upper=hi, n_items=n)
            rows.append(d)
    T = pd.DataFrame(rows)
    T.to_csv(TABLES / f"cpt_{model}_{run}_adoption.csv", index=False)
    return T


def state_matched(model, run="cpt_core"):
    """Receiver-state controls for CPT interactions. P(q) = Y(q, expert) - Y(q, anon) under disagreement.
    Baselines: 'anon' = Y(q, anon) (shares noise with P) and 'none' = Y(q, no label) (independent measurement).
    Outputs: state curve (P by independent-baseline bin x evidence quality, cluster CIs) and adjusted interactions
    (P ~ factor + bs(baseline, 5); item-cluster robust SEs) for I_QP (all items) and I_CP (target items, strong evidence)."""
    import statsmodels.formula.api as smf
    ic = pd.read_csv(TABLES / f"cpt_{model}_{run}_m_adv_itemcontrasts.csv")
    long = []
    for r in ic.itertuples():
        for q in ("strong", "weak"):
            long.append(dict(item_id=r.item_id, cluster=r.cluster, domain=r.domain, family=r.family, has_target=r.has_target,
                             adv_correct=r.adv_correct, qual=q, P=getattr(r, f"Y_dis_{q}_exp") - getattr(r, f"Y_dis_{q}_anon"),
                             base_anon=getattr(r, f"Y_dis_{q}_anon"), base_none=getattr(r, f"Y_dis_{q}_none")))
    L = pd.DataFrame(long).dropna(subset=["P", "base_anon", "base_none"])
    edges = [-np.inf, -8, -6, -4, -2, 0, 2, 4, 6, np.inf]
    L["bin"] = pd.cut(L.base_none, edges)
    curve = []
    for (b, q), g in L.groupby(["bin", "qual"], observed=True):
        est, lo, hi, p, n = fast_cluster_mean_ci(g.P.to_numpy(), g.cluster.to_numpy())
        mid = float(np.clip((b.left + b.right) / 2 if np.isfinite(b.left) and np.isfinite(b.right) else (b.right - 1 if np.isfinite(b.right) else b.left + 1), -9, 7))
        curve.append(dict(bin=str(b), bin_mid=mid, qual=q, P=est, lower=lo, upper=hi, n=n))
    pd.DataFrame(curve).to_csv(TABLES / f"cpt_{model}_{run}_state_curve.csv", index=False)

    def fit(df, formula, term):
        try:
            m = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.cluster)[0]})
            ci = m.conf_int().loc[term]
            return float(m.params[term]), float(ci[0]), float(ci[1])
        except Exception:  # noqa: BLE001
            return np.nan, np.nan, np.nan

    rows = []
    T = L[L.has_target & (L.qual == "strong")].copy()
    T["correct"] = T.adv_correct.astype(bool)
    levels = [("ALL", "all", None)] + [(d, "domain", "domain") for d in sorted(L.domain.unique())] + \
             [(f, "family", "family") for f in sorted(L.family.unique())]
    for name, level, col in levels:
        LL = L if col is None else L[L[col] == name]
        TT = T if col is None else T[T[col] == name]
        for adj in ("raw", "anon", "none"):
            spline = "" if adj == "raw" else f" + bs(base_{adj}, df=5)"
            if LL.cluster.nunique() >= 20:
                e, lo, hi = fit(LL, 'P ~ C(qual, Treatment("weak"))' + spline, 'C(qual, Treatment("weak"))[T.strong]')
                rows.append(dict(interaction="I_QP", level=level, group=name, adjustment=adj, estimate=e, lower=lo, upper=hi, n_items=LL.item_id.nunique()))
            if len(TT) and TT.cluster.nunique() >= 20 and TT.correct.nunique() == 2:
                e, lo, hi = fit(TT, "P ~ C(correct)" + spline, "C(correct)[T.True]")
                rows.append(dict(interaction="I_CP", level=level, group=name, adjustment=adj, estimate=e, lower=lo, upper=hi, n_items=TT.item_id.nunique()))
    R = pd.DataFrame(rows)
    # robust = same sign under both adjustments and both CIs exclude zero
    piv = R[R.adjustment != "raw"].pivot_table(index=["interaction", "level", "group"], columns="adjustment", values=["estimate", "lower", "upper"])
    ok = ((np.sign(piv[("estimate", "anon")]) == np.sign(piv[("estimate", "none")])) &
          ((piv[("lower", "anon")] > 0) | (piv[("upper", "anon")] < 0)) & ((piv[("lower", "none")] > 0) | (piv[("upper", "none")] < 0)))
    R = R.merge(ok.rename("robust_both_adjustments").reset_index(), on=["interaction", "level", "group"], how="left")
    R.to_csv(TABLES / f"cpt_{model}_{run}_state_adjusted.csv", index=False)
    return R
