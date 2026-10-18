"""Gate evaluation for the V3 attribution-counterfactual jobs (PLAN_V2.md, V3 PLAN).

  own  <M> <dataset>   V3-C2 whose words      (ACTS/<M>/frames_own_<dataset>[_notrelease_600])
  aud  <M> <dataset>   V3-C4(i) who will hear (ACTS/<M>/frames_audience_<dataset>[_notrelease_600])
  say  <M>             V3-C1(c) say vs know   (ACTS/<M>/cpt_say_release_600 + cpt_core HF worlds/acts for the probe)
  desc <M>             V3-C1(a) status vs reliability (ACTS/<M>/cpt_desc_release_600)
All CIs: item-cluster bootstrap (analysis_group for persuasion); paired contrasts are computed within episode/item first.
Outputs: outputs/tables/v3_<kind>_<M>_<dataset>*.csv and a JSON summary with gate verdicts.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import *  # noqa
from stats import bh_fdr, fast_cluster_mean_ci

CORE = {r["id"]: r for r in load_core()}


def cluster_of(i):
    r = CORE[i]
    return r["meta"].get("analysis_group") if r["stratum"] == "persuasion" else r["source_item_id"]


def ci(values, clusters):
    e, lo, hi, p, n = fast_cluster_mean_ci(np.asarray(values, float), np.asarray(clusters))
    return dict(est=e, lower=lo, upper=hi, p=p, n_clusters=n, n=int(np.isfinite(np.asarray(values, float)).sum()))


def excl0(c):
    return bool(c["lower"] > 0 or c["upper"] < 0)


def frames_dir(M, family, dataset):
    base = ACTS / M
    for name in (f"frames_{family}_{dataset}_notrelease_600", f"frames_{family}_{dataset}"):
        if (base / name / "index.jsonl").exists():
            return base / name
    raise FileNotFoundError(f"frames_{family}_{dataset} for {M}")


def probe_scores(M, acts_path, rows, site=0):
    fz = json.load(open(ACTS / M / "probes_frozen_cpt_core.json"))
    A = np.load(acts_path, mmap_mode="r")
    out = {}
    for name in ("provenance", "social_disagreement", "revision_readiness", "content_correctness"):
        if name not in fz:
            continue
        f = fz[name]
        X = np.asarray(A[rows, site, f["grid_index"]], dtype=np.float32)
        Z = (X - np.asarray(f["mu"], np.float32)) / np.asarray(f["sd"], np.float32)
        out[name] = Z @ np.asarray(f["w_dom"], np.float32)
    return out


def auroc(y, s):
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


# ---------------------------------------------------------------------------------------------------------------------------
def run_own(M, dataset):
    d = frames_dir(M, "own", dataset)
    I = pd.DataFrame(read_jsonl(d / "index.jsonl"))
    sc = probe_scores(M, d / "acts.npy", I.row.to_numpy())
    for k, v in sc.items():
        I[f"probe_{k}"] = v
    I["episode"] = I.item_id + "|" + I.turn.astype(str) + "|" + I.agent.astype(str)
    recs = []
    for ep, g in I.groupby("episode", sort=False):
        fr = {r.frame: r for r in g.itertuples()}
        o = fr["orig"]
        X = o.own
        peers = o.peers
        n_x_peers = sum(v == X for v in peers.values())
        config = "outnumbered" if n_x_peers == 0 and len(set(peers.values())) == 1 else ("majority" if n_x_peers == 1 else "split3")
        for name, r in fr.items():
            if not name.startswith("swap:"):
                continue
            j = name.split(":")[1]
            Y = peers[j]
            m = lambda row: row.lp_letters[X] - row.lp_letters[Y]
            rec = dict(item_id=o.item_id, turn=o.turn, agent=o.agent, peer=j, config=config, X=X, Y=Y, name_mention=o.name_mention,
                       m_orig=m(o), m_swap=m(r), d_own=m(o) - m(r), d_order=m(o) - m(fr["order"]), d_names=m(o) - m(fr["names"]),
                       p_mass=float(np.exp(np.logaddexp.reduce(list(o.lp_letters.values())))))
            for k in sc:
                rec[f"dprobe_{k}"] = getattr(o, f"probe_{k}") - getattr(r, f"probe_{k}")
                rec[f"dprobe_order_{k}"] = getattr(o, f"probe_{k}") - getattr(fr["order"], f"probe_{k}")
            recs.append(rec)
    E = pd.DataFrame(recs)
    E["cluster"] = E.item_id.map(cluster_of)
    E["family"] = E.item_id.map(lambda i: family_of(CORE[i]))
    E["domain"] = E.item_id.map(lambda i: CORE[i]["primary_domain"])
    tag = f"{M}_{dataset}"
    E.to_csv(TABLES / f"v3_own_episodes_{tag}.csv", index=False)
    rows = []
    def add(scope, g):
        for col in ["d_own", "d_order", "d_names", "m_orig", "m_swap"] + [c for c in E.columns if c.startswith("dprobe_")]:
            c = ci(g[col], g.cluster)
            rows.append(dict(scope=scope, metric=col, **c))
    add("ALL", E)
    add("no_name_mention", E[~E.name_mention])
    for k, g in E.groupby("config"):
        add(f"config={k}", g)
    for k, g in E.groupby("domain"):
        add(f"domain={k}", g)
    for k, g in E.groupby("family"):
        add(f"family={k}", g)
    for k, g in E.groupby(pd.cut(E.turn, [0, 1, 3, 6, 11])):
        add(f"turn={k}", g)
    S = pd.DataFrame(rows)
    S.to_csv(TABLES / f"v3_own_summary_{tag}.csv", index=False)
    a = S[(S.scope == "ALL")].set_index("metric")
    fam = S[S.scope.str.startswith("family=") & (S.metric == "d_own")]
    ctrl_ratio = {k: float(abs(a.loc[k, "est"]) / abs(a.loc["d_own", "est"])) for k in ("d_order", "d_names")}
    nm = S[(S.scope == "no_name_mention") & (S.metric == "d_own")].iloc[0]
    gate = dict(d_own=a.loc["d_own"].to_dict(), families_positive=int((fam.est > 0).sum()), n_families=len(fam),
                control_ratio=ctrl_ratio, no_name_mention=nm.to_dict(),
                PASS_effect=bool(a.loc["d_own", "est"] >= 0.5 and a.loc["d_own", "lower"] > 0),
                PASS_families=bool((fam.est > 0).sum() >= 10), PASS_controls=bool(max(ctrl_ratio.values()) < 0.2),
                PASS_no_name=bool(nm["lower"] > 0))
    for k in sc:
        gate[f"probe_{k}"] = a.loc[f"dprobe_{k}"].to_dict()
    json.dump(gate, open(TABLES / f"v3_own_gate_{tag}.json", "w"), indent=1, default=float)
    print(json.dumps(gate, indent=1, default=float))
    print(S[S.metric == "d_own"].round(3).to_string())


# ---------------------------------------------------------------------------------------------------------------------------
def run_aud(M, dataset, T0=0.7):
    d = frames_dir(M, "audience", dataset)
    I = pd.DataFrame(read_jsonl(d / "index.jsonl"))
    I = I[I.public.notna()].copy()
    labs = lambda r: r.lp_letters
    I["argmax"] = I.lp_letters.map(lambda x: max(x, key=x.get))
    I["switch"] = (I.argmax != I.public).astype(float)
    I["m_pub"] = [lp[p] - max(v for k, v in lp.items() if k != p) if len(lp) > 1 else np.nan for lp, p in zip(I.lp_letters, I.public)]
    def p_T(lp, p):
        v = np.array(list(lp.values())) / T0
        return float(np.exp(lp[p] / T0 - np.logaddexp.reduce(v)))
    I["switch_T07"] = [1 - p_T(lp, p) for lp, p in zip(I.lp_letters, I.public)]
    I["cluster"] = I.item_id.map(cluster_of)
    I["domain"] = I.item_id.map(lambda i: CORE[i]["primary_domain"])
    I["has_target"] = I.item_id.map(lambda i: item_view(CORE[i])["target"] is not None)
    tag = f"{M}_{dataset}"
    W = I.pivot_table(index=["item_id", "cond", "agent", "cluster", "domain", "unanimous", "public"], columns="frame",
                      values=["switch", "m_pub", "switch_T07"], aggfunc="first")
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    W = W.reset_index()
    W.to_csv(TABLES / f"v3_aud_episodes_{tag}.csv", index=False)
    rows = []
    def add(scope, g):
        for col in [c for c in W.columns if c.startswith(("switch_", "m_pub_"))]:
            rows.append(dict(scope=scope, metric=col, **ci(g[col], g.cluster)))
        for a_, b_ in (("priv_ctx", "pub_ctx"), ("priv_self", "pub_self"), ("priv_ctx", "priv_self")):
            rows.append(dict(scope=scope, metric=f"switch_{a_}-{b_}", **ci(g[f"switch_{a_}"] - g[f"switch_{b_}"], g.cluster)))
            rows.append(dict(scope=scope, metric=f"m_pub_{b_}-{a_}", **ci(g[f"m_pub_{b_}"] - g[f"m_pub_{a_}"], g.cluster)))
    for cond in ("debate", "pool"):
        for un in (True, None):
            g = W[(W.cond == cond) & ((W.unanimous == True) if un else True)]
            add(f"{cond}|{'unanimous' if un else 'all'}", g)
            for dom, gd in g.groupby("domain"):
                add(f"{cond}|{'unanimous' if un else 'all'}|{dom}", gd)
    # debate - pool gap per frame, paired at item level (agents averaged), unanimous-in-both items and all items
    gap_rows = []
    for un in (True, False):
        g = W if not un else W[W.unanimous == True]
        Ag = g.groupby(["item_id", "cluster", "cond"])[[c for c in W.columns if c.startswith("switch_") or c.startswith("m_pub_")]].mean()
        for col in Ag.columns:
            P = Ag[col].unstack("cond").dropna()
            if {"debate", "pool"} <= set(P.columns):
                gap_rows.append(dict(scope="unanimous_both" if un else "all", metric=col,
                                     **ci(P.debate - P.pool, P.index.get_level_values("cluster"))))
    S = pd.concat([pd.DataFrame(rows), pd.DataFrame(gap_rows).assign(scope=lambda x: "debate-pool|" + x.scope)])
    S.to_csv(TABLES / f"v3_aud_summary_{tag}.csv", index=False)
    s = S.set_index(["scope", "metric"])
    gate = dict(
        priv_minus_pub_ctx_debate_unan=s.loc[("debate|unanimous", "switch_priv_ctx-pub_ctx")].to_dict(),
        priv_minus_pub_self_debate_unan=s.loc[("debate|unanimous", "switch_priv_self-pub_self")].to_dict(),
        gap_priv_ctx=s.loc[("debate-pool|unanimous_both", "switch_priv_ctx")].to_dict(),
        gap_priv_self=s.loc[("debate-pool|unanimous_both", "switch_priv_self")].to_dict(),
        gap_pub_ctx=s.loc[("debate-pool|unanimous_both", "switch_pub_ctx")].to_dict(),
    )
    gate["PASS_wording"] = excl0(gate["priv_minus_pub_ctx_debate_unan"]) and gate["priv_minus_pub_ctx_debate_unan"]["est"] > 0
    gate["PASS_transcript_free_gap"] = excl0(gate["gap_priv_self"]) and gate["gap_priv_self"]["est"] > 0
    gen = RAW / M / dataset / "private.jsonl"
    if gen.exists():
        P = pd.DataFrame(read_jsonl(gen))
        P["cond"] = P.cond.str.replace("private_", "")
        J = I[I.frame == "priv_ctx"].merge(P[["item_id", "cond", "agent", "verdict"]], on=["item_id", "cond", "agent"])
        J = J[J.verdict.notna()]
        gate["consistency_priv_ctx_argmax_vs_generated"] = float((J.argmax == J.verdict).mean())
    json.dump(gate, open(TABLES / f"v3_aud_gate_{tag}.json", "w"), indent=1, default=float)
    print(json.dumps(gate, indent=1, default=float))
    print(S[S.scope.isin(["debate|unanimous", "pool|unanimous"]) | S.scope.str.startswith("debate-pool")].round(4).to_string())


# ---------------------------------------------------------------------------------------------------------------------------
def run_say(M, split="release_600"):
    S = pd.DataFrame(read_jsonl(ACTS / M / f"cpt_say_{split}" / "worlds.jsonl"))
    S = S[S.adv.notna()].copy()
    S["adopt"] = (S.verdict == S.adv).astype(float)
    S.loc[S.verdict.isna(), "adopt"] = np.nan
    S["first_argmax"] = S.first_lp_letters.map(lambda x: max(x, key=x.get))
    H = pd.DataFrame(read_jsonl(ACTS / M / "cpt_core" / "worlds.jsonl"))
    key = ["item_id", "template", "adv", "qual", "prov", "prior"]
    H = H[H.adv.notna()]
    J = S.merge(H[key + ["argmax_label", "world_idx"]], on=key, how="left")
    consistency = dict(gen_vs_hf=float((J.verdict == J.argmax_label).mean()), first_token_vs_hf=float((J.first_argmax == J.argmax_label).mean()),
                       n_matched=int(J.argmax_label.notna().sum()))
    J["cluster"] = J.item_id.map(cluster_of)
    J["yes_src"] = (J.p_yes_src > 0.5).astype(float)
    J["yes_arg"] = (J.p_yes_arg > 0.5).astype(float)
    dis = J[J.adv != J.prior]
    # credential-caused vs content-caused (same adv, qual, prior; expert vs anon)
    idx = dis.set_index(["item_id", "adv", "qual", "prior", "prov"])
    cred, cont, ev = [], [], []
    for (i, adv, qual, prior), g in dis.groupby(["item_id", "adv", "qual", "prior"]):
        w = g.set_index("prov")
        if not {"expert", "anon"} <= set(w.index) or w.adopt.isna().any():
            continue
        e, a_ = w.loc["expert"], w.loc["anon"]
        if e.adopt == 1 and a_.adopt == 0:
            cred.append(dict(item_id=i, qual=qual, p_yes_src=e.p_yes_src, yes_src=e.yes_src, p_yes_arg=e.p_yes_arg, su=e.source_use,
                             row_expert=e.world_idx, row_anon=a_.world_idx))
        elif e.adopt == 1 and a_.adopt == 1:
            cont.append(dict(item_id=i, qual=qual, p_yes_src=e.p_yes_src, yes_src=e.yes_src, p_yes_src_anon=a_.p_yes_src,
                             yes_src_anon=a_.yes_src, su=e.source_use))
    for (i, adv, prov, prior), g in dis.groupby(["item_id", "adv", "prov", "prior"]):
        w = g.set_index("qual")
        if not {"strong", "weak"} <= set(w.index) or w.adopt.isna().any():
            continue
        s_, wk = w.loc["strong"], w.loc["weak"]
        if s_.adopt == 1 and wk.adopt == 0:
            ev.append(dict(item_id=i, prov=prov, p_yes_arg=s_.p_yes_arg, yes_arg=s_.yes_arg, p_yes_src=s_.p_yes_src, su=s_.source_use))
    C, N, V = pd.DataFrame(cred), pd.DataFrame(cont), pd.DataFrame(ev)
    for X in (C, N, V):
        if len(X):
            X["cluster"] = X.item_id.map(cluster_of)
    out = dict(consistency=consistency, n_credential_caused=len(C), n_content_caused=len(N), n_evidence_caused=len(V))
    if len(C) and len(V):
        out["a_cred_binary"] = ci(C.yes_src, C.cluster)
        out["a_cred_p"] = ci(C.p_yes_src, C.cluster)
        out["a_ev_binary"] = ci(V.yes_arg, V.cluster)
        out["a_ev_p"] = ci(V.p_yes_arg, V.cluster)
        out["ack_src_content_caused_expert"] = ci(N.yes_src, N.cluster) if len(N) else None
        out["false_ack_src_content_caused_anon"] = ci(N.yes_src_anon, N.cluster) if len(N) else None
        out["field_src_or_both_cred"] = ci(C.su.isin(["source", "both"]).astype(float), C.cluster)
        # asymmetry with a joint item-cluster bootstrap over the union of clusters
        cl = sorted(set(C.cluster) | set(V.cluster))
        cix = {c: k for k, c in enumerate(cl)}
        def sums(X, col):
            s = np.zeros(len(cl)); n = np.zeros(len(cl))
            np.add.at(s, X.cluster.map(cix).to_numpy(), X[col].to_numpy(float)); np.add.at(n, X.cluster.map(cix).to_numpy(), 1)
            return s, n
        sv, nv = sums(V, "yes_arg"); sc_, nc = sums(C, "yes_src")
        rng = np.random.default_rng(240091)
        B = rng.integers(0, len(cl), (4000, len(cl)))
        boots = sv[B].sum(1) / np.maximum(nv[B].sum(1), 1) - sc_[B].sum(1) / np.maximum(nc[B].sum(1), 1)
        est = sv.sum() / nv.sum() - sc_.sum() / nc.sum()
        out["asymmetry_ev_minus_cred"] = dict(est=float(est), lower=float(np.quantile(boots, .025)), upper=float(np.quantile(boots, .975)))
        # frozen provenance probe on the credential-caused expert/anon world pairs (HF cpt_core activations)
        fz = json.load(open(ACTS / M / "probes_frozen_cpt_core.json"))["provenance"]
        A = np.load(ACTS / M / "cpt_core" / "acts.npy", mmap_mode="r")
        rows = np.concatenate([C.row_expert.to_numpy(int), C.row_anon.to_numpy(int)])
        order = np.argsort(rows)
        X = np.asarray(A[rows[order], 0, fz["grid_index"]], np.float32)
        s = np.empty(len(rows)); s[order] = ((X - np.asarray(fz["mu"], np.float32)) / np.asarray(fz["sd"], np.float32)) @ np.asarray(fz["w_dom"], np.float32)
        y = np.r_[np.ones(len(C)), np.zeros(len(C))]
        out["probe_auroc_credential_caused"] = float(auroc(y, s))
        g1 = out["a_cred_binary"]; asy = out["asymmetry_ev_minus_cred"]
        out["PASS_a_cred"] = bool(g1["est"] <= 0.50 and g1["upper"] < 0.60)
        out["PASS_asymmetry"] = bool(asy["est"] >= 0.25 and asy["lower"] > 0)
        out["PASS_probe"] = bool(out["probe_auroc_credential_caused"] >= 0.85)
        out["PASS_consistency"] = bool(consistency["first_token_vs_hf"] >= 0.97)
        out["PASS"] = out["PASS_a_cred"] and out["PASS_asymmetry"] and out["PASS_probe"] and out["PASS_consistency"]
        # by domain
        dom = []
        for X_, col, nm in ((C, "yes_src", "a_cred"), (V, "yes_arg", "a_ev")):
            X_ = X_.assign(domain=X_.item_id.map(lambda i: CORE[i]["primary_domain"]))
            for k, g in X_.groupby("domain"):
                dom.append(dict(metric=nm, domain=k, **ci(g[col], g.cluster)))
        pd.DataFrame(dom).to_csv(TABLES / f"v3_say_domain_{M}.csv", index=False)
    C.to_csv(TABLES / f"v3_say_credential_caused_{M}.csv", index=False)
    V.to_csv(TABLES / f"v3_say_evidence_caused_{M}.csv", index=False)
    json.dump(out, open(TABLES / f"v3_say_gate_{M}.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


# ---------------------------------------------------------------------------------------------------------------------------
def run_desc(M, split="release_600"):
    d = ACTS / M / f"cpt_desc_{split}"
    W = pd.DataFrame(read_jsonl(d / "worlds.jsonl"))
    fz = json.load(open(ACTS / M / "probes_frozen_cpt_core.json"))["provenance"]
    A = np.load(d / "acts.npy", mmap_mode="r")
    X = np.asarray(A[:, 0, fz["grid_index"]], np.float32)
    W["probe_prov"] = ((X - np.asarray(fz["mu"], np.float32)) / np.asarray(fz["sd"], np.float32)) @ np.asarray(fz["w_dom"], np.float32)
    W["cluster"] = W.item_id.map(cluster_of)
    key = ["item_id", "adv", "qual", "prior"]
    base = W[W.prov == "none"].set_index(key)[["m_adv", "probe_prov"]]
    W = W.join(base, on=key, rsuffix="_none")
    W["P"] = W.m_adv - W.m_adv_none
    W["dprobe"] = W.probe_prov - W.probe_prov_none
    per = W[W.prov != "none"].groupby(["item_id", "cluster", "prov", "agree"])[["P", "dprobe"]].mean().reset_index()
    rows = []
    for (prov, agree), g in per.groupby(["prov", "agree"]):
        for col in ("P", "dprobe"):
            rows.append(dict(prov=prov, agree=agree, metric=col, **ci(g[col], g.cluster)))
    piv = per[per.agree == False].pivot_table(index=["item_id", "cluster"], columns="prov", values="P")
    pivp = per[per.agree == False].pivot_table(index=["item_id", "cluster"], columns="prov", values="dprobe")
    cl = piv.index.get_level_values("cluster")
    contrasts = {
        "status|bad (prof_bad-ord_bad)": piv.prof_bad - piv.ord_bad,
        "status|good (prof_good-ord_good)": piv.prof_good - piv.ord_good,
        "status|unknown (prof_unst-ord_unst)": piv.prof_unst - piv.ord_unst,
        "record|prof (prof_good-prof_bad)": piv.prof_good - piv.prof_bad,
        "record|ord (ord_good-ord_bad)": piv.ord_good - piv.ord_bad,
        "record_vs_unknown|ord (ord_good-ord_unst)": piv.ord_good - piv.ord_unst,
        "expert-anon": piv.expert - piv.anon,
        "status x record interaction": (piv.prof_good - piv.prof_bad) - (piv.ord_good - piv.ord_bad),
    }
    crow = []
    for k, v in contrasts.items():
        crow.append(dict(contrast=k, scale="m_adv", **ci(v, cl)))
    for k, v in {"status|bad": pivp.prof_bad - pivp.ord_bad, "record|ord": pivp.ord_good - pivp.ord_bad,
                 "record|prof": pivp.prof_good - pivp.prof_bad, "expert-anon": pivp.expert - pivp.anon}.items():
        crow.append(dict(contrast=k, scale="provenance_probe", **ci(v, pivp.index.get_level_values("cluster"))))
    Ctab = pd.DataFrame(crow)
    pd.DataFrame(rows).to_csv(TABLES / f"v3_desc_cells_{M}.csv", index=False)
    Ctab.to_csv(TABLES / f"v3_desc_contrasts_{M}.csv", index=False)
    s = Ctab.set_index(["contrast", "scale"]).loc[("status|bad (prof_bad-ord_bad)", "m_adv")]
    gate = dict(status_given_bad_record=s.to_dict(), PASS_authority=bool(s["lower"] > 0),
                length_check=W.groupby("prov").src_len.mean().to_dict())
    json.dump(gate, open(TABLES / f"v3_desc_gate_{M}.json", "w"), indent=1, default=float)
    print(json.dumps(gate, indent=1, default=float))
    print(Ctab.round(3).to_string())
    print(pd.DataFrame(rows).query("metric=='P'").round(3).to_string())


if __name__ == "__main__":
    kind, M = sys.argv[1], sys.argv[2]
    ds = sys.argv[3] if len(sys.argv) > 3 else "release_600"
    dict(own=lambda: run_own(M, ds), aud=lambda: run_aud(M, ds), say=lambda: run_say(M), desc=lambda: run_desc(M))[kind]()
