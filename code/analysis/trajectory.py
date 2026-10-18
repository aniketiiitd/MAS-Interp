"""C2 concept-atlas trajectory analyses over 12-turn debates (pre-registered in PLAN_V2.md).

Input: tables/concept_traj_<M>_<replay>.parquet (z-scored projections at decision 'dec_' and reading 'read_' sites),
       raw debate/pool turn files (verdicts), concept_validation_<M>.csv (only validated concepts are analysed).
Split: discovery / confirmation halves of release_600 by hash of the source unit. Selection happens on discovery only;
       confirmation re-tests the selected concepts with the discovery sign (one-sided direction, two-sided CI reported).

(a) drift      debate - pool concept z, averaged over agents, per turn and averaged over turns 1..T-1
(b) lead       revise_at_t ~ z_{t-1} + dissent_config_{t-1} + own_conf_{t-1} + agreement_{t-1} + turn  (GEE logit, cluster)
               and revision-locked averages (relative turn -3..+2) vs agents facing dissent who did not revise
(c) synchrony  within-debate cross-agent correlation of concept trajectories vs shuffled-debate null; debate vs pool
(d) moral      moral-foundation trajectories in values & safety domains vs pool; association with stance revision
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import *  # noqa
from stats import bh_fdr, fast_cluster_mean_ci

warnings.filterwarnings("ignore")
TOPK = 10


def load(model, replay, dataset, turns):
    P = pd.read_parquet(TABLES / f"concept_traj_{model}_{replay}.parquet")
    V = pd.read_csv(TABLES / f"concept_validation_{model}.csv")
    valid = list(V[V.passes].concept)
    core = {r["id"]: r for r in load_core()}
    unit = lambda i: core[i]["meta"].get("analysis_group") if core[i]["stratum"] == "persuasion" else core[i]["source_item_id"]
    P["cluster"] = P.item_id.map(unit)
    P["half"] = P.cluster.map(lambda u: "discovery" if h64("disc240091", u) % 2 == 0 else "confirmation")
    P["domain"] = P.item_id.map(lambda i: core[i]["primary_domain"])
    P["family"] = P.item_id.map(lambda i: family_of(core[i]))
    # verdict trajectories to define revisions and dissent configuration (debate)
    deb = P[P.cond.isin(["t0", "debate"])].sort_values(["item_id", "agent", "turn"]).copy()
    deb["prev_verdict"] = deb.groupby(["item_id", "agent"]).verdict.shift(1)
    deb["revised"] = (deb.verdict != deb.prev_verdict) & deb.prev_verdict.notna() & deb.verdict.notna()
    vt = deb.pivot_table(index=["item_id", "turn"], columns="agent", values="verdict", aggfunc="first")
    conf = []
    for (iid, t), r in vt.iterrows():
        vs = [r.get(a) for a in range(3)]
        for a in range(3):
            others = [vs[b] for b in range(3) if b != a]
            dis = [o for o in others if o is not None and o == o and o != vs[a]]
            cfg = 0 if not dis else (1 if len(dis) == 1 else (2 if dis[0] == dis[1] else 3))
            agree = max([vs.count(v) for v in set(vs)]) / 3 if all(isinstance(v, str) for v in vs) else np.nan
            conf.append(dict(item_id=iid, turn=t, agent=a, cfg=cfg, agreement=agree))
    C = pd.DataFrame(conf)
    deb = deb.merge(C, on=["item_id", "turn", "agent"], how="left")
    return P, deb, valid


def drift(P, valid, site, half):
    rows = []
    H = P[P.half == half]
    for c in valid:
        col = f"{site}_{c}"
        g = H[H.cond.isin(["debate", "pool"]) & (H.turn >= 1)].groupby(["item_id", "cluster", "cond"])[col].mean().unstack("cond").dropna()
        d = (g["debate"] - g["pool"]).to_numpy()
        est, lo, hi, p, n = fast_cluster_mean_ci(d, g.index.get_level_values("cluster").to_numpy())
        rows.append(dict(site=site, half=half, concept=c, drift=est, lower=lo, upper=hi, p=p, n=n))
    D = pd.DataFrame(rows, columns=["site", "half", "concept", "drift", "lower", "upper", "p", "n"])
    D["q"] = bh_fdr(D.p) if len(D) else []
    return D


def lead(deb, valid, site, half):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    X = deb[(deb.half == half)].sort_values(["item_id", "agent", "turn"]).copy()
    for c in valid:
        X[f"lag_{c}"] = X.groupby(["item_id", "agent"])[f"{site}_{c}"].shift(1)
    X["lag_cfg"] = X.groupby(["item_id", "agent"]).cfg.shift(1)
    X["lag_agree"] = X.groupby(["item_id", "agent"]).agreement.shift(1)
    X["lag_conf"] = X.groupby(["item_id", "agent"]).confidence.shift(1)
    X["lag_general"] = X.groupby(["item_id", "agent"])[f"{site}_GENERAL"].shift(1)
    X = X[(X.turn >= 2) & X.lag_cfg.notna() & (X.lag_cfg > 0)]  # agents that faced dissent at t-1
    X["y"] = X.revised.astype(int)
    rows = []
    for c in valid:
        d = X[[f"lag_{c}", "y", "lag_cfg", "lag_agree", "lag_conf", "lag_general", "turn", "cluster"]].dropna().rename(columns={f"lag_{c}": "z"})
        if d.y.nunique() < 2 or len(d) < 200:
            continue
        try:
            m = smf.gee("y ~ z + lag_general + C(lag_cfg) + lag_agree + lag_conf + turn", groups="cluster", data=d,
                        family=sm.families.Binomial()).fit()
            ci = m.conf_int().loc["z"]
            rows.append(dict(site=site, half=half, concept=c, coef=m.params["z"], lower=ci[0], upper=ci[1], p=m.pvalues["z"], n=len(d)))
        except Exception:  # noqa: BLE001
            continue
    L = pd.DataFrame(rows)
    if len(L):
        L["q"] = bh_fdr(L.p)
    return L


def locked(deb, concepts, site, half, window=(-3, 2)):
    X = deb[deb.half == half]
    first = X[X.revised & (X.turn >= 2)].groupby(["item_id", "agent"]).turn.min().rename("rev_turn").reset_index()
    X = X.merge(first, on=["item_id", "agent"], how="left")
    rows = []
    # matched controls: agents that faced dissent at some turn t-1 >= 1 and never revised; aligned at that t
    ctrl = X[X.rev_turn.isna()].sort_values(["item_id", "agent", "turn"]).copy()
    ctrl["lag_cfg"] = ctrl.groupby(["item_id", "agent"]).cfg.shift(1)
    anchor = ctrl[(ctrl.turn >= 2) & (ctrl.lag_cfg > 0)].groupby(["item_id", "agent"]).turn.min().rename("rev_turn").reset_index()
    C = X[X.rev_turn.isna()].drop(columns=["rev_turn"]).merge(anchor, on=["item_id", "agent"], how="inner")
    for label, G in (("revisers", X[X.rev_turn.notna()]), ("dissent_non_revisers", C)):
        G = G.assign(rel=G.turn - G.rev_turn)
        G = G[(G.rel >= window[0]) & (G.rel <= window[1])]
        for c in concepts:
            m = G.groupby("rel")[f"{site}_{c}"].agg(["mean", "sem", "count"]).reset_index()
            for r in m.itertuples():
                rows.append(dict(site=site, half=half, group=label, concept=c, rel_turn=r.rel, mean=r.mean, sem=r.sem, n=r.count))
    return pd.DataFrame(rows)


def synchrony(P, concepts, site, half, n_null=50, seed=240091):
    rng = np.random.default_rng(seed)
    H = P[(P.half == half) & (P.turn >= 1)]
    rows = []
    for cond in ("debate", "pool"):
        G = H[H.cond == cond]
        for c in concepts:
            piv = G.pivot_table(index=["item_id", "agent"], columns="turn", values=f"{site}_{c}")
            items = piv.index.get_level_values(0).unique()
            vals = {}
            for iid in items:
                try:
                    m = piv.loc[iid].to_numpy()
                except KeyError:
                    continue
                if m.shape[0] == 3 and np.isfinite(m).all():
                    vals[iid] = m - m.mean(1, keepdims=True)
            keys = list(vals)
            if len(keys) < 30:
                continue

            def corr(a, b):
                den = np.linalg.norm(a) * np.linalg.norm(b)
                return float(a @ b / den) if den > 0 else np.nan

            within = np.nanmean([np.nanmean([corr(v[i], v[j]) for i, j in ((0, 1), (0, 2), (1, 2))]) for v in vals.values()])
            # null for the AGGREGATE: mean over len(keys) random cross-item triples, repeated n_null times
            null = []
            for _ in range(n_null):
                trip = rng.integers(0, len(keys), size=(len(keys), 3))
                null.append(np.nanmean([np.nanmean([corr(vals[keys[a]][0], vals[keys[b]][1]), corr(vals[keys[a]][0], vals[keys[c2]][2]),
                                                    corr(vals[keys[b]][1], vals[keys[c2]][2])]) for a, b, c2 in trip if len({a, b, c2}) == 3]))
            rows.append(dict(site=site, half=half, cond=cond, concept=c, within=within, null_mean=float(np.mean(null)),
                             null_95=float(np.quantile(null, 0.95)), n_debates=len(keys)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--replay", default="replay_release_600_release_600_T12")
    ap.add_argument("--dataset", default="release_600")
    ap.add_argument("--turns", type=int, default=12)
    a = ap.parse_args()
    P, deb, valid = load(a.model, a.replay, a.dataset, a.turns)
    tax = json.load(open(ROOT / "data/concepts/taxonomy.json"))
    tag = f"{a.model}_T{a.turns}"
    out = {}
    for site in ("dec", "read"):
        Dd = drift(P, valid, site, "discovery")
        sig = Dd[Dd.q < 0.05]
        sel = sig.loc[sig.drift.abs().sort_values(ascending=False).index].head(TOPK)
        Dc = drift(P, list(sel.concept), site, "confirmation") if len(sel) else pd.DataFrame(columns=["concept", "drift", "lower", "upper"])
        Dc = Dc.merge(sel[["concept", "drift"]].rename(columns={"drift": "disc_drift"}), on="concept")
        Dc["replicates"] = (np.sign(Dc.drift) == np.sign(Dc.disc_drift)) & ((Dc.lower > 0) | (Dc.upper < 0))
        Dd.to_csv(TABLES / f"traj_drift_discovery_{tag}_{site}.csv", index=False)
        Dc.to_csv(TABLES / f"traj_drift_confirmation_{tag}_{site}.csv", index=False)
        Ld = lead(deb, valid, site, "discovery")
        Lc = pd.DataFrame(columns=["concept", "coef", "lower", "upper", "p", "n", "disc_coef", "replicates"])
        Lc.to_csv(TABLES / f"traj_lead_confirmation_{tag}_{site}.csv", index=False)
        if len(Ld):
            lsig = Ld[Ld.q < 0.05]
            lsel = lsig.loc[lsig.coef.abs().sort_values(ascending=False).index].head(TOPK)
            Lc = lead(deb, list(lsel.concept), site, "confirmation") if len(lsel) else pd.DataFrame()
            if len(Lc):
                Lc = Lc.merge(lsel[["concept", "coef"]].rename(columns={"coef": "disc_coef"}), on="concept")
                Lc["replicates"] = (np.sign(Lc.coef) == np.sign(Lc.disc_coef)) & ((Lc.lower > 0) | (Lc.upper < 0))
            Ld.to_csv(TABLES / f"traj_lead_discovery_{tag}_{site}.csv", index=False)
            Lc.to_csv(TABLES / f"traj_lead_confirmation_{tag}_{site}.csv", index=False)
            focus = list(dict.fromkeys(list(lsel.concept[:6]) + [c for c in tax["prereg_debate"] if c in valid]))
            locked(deb, focus, site, "confirmation").to_csv(TABLES / f"traj_locked_{tag}_{site}.csv", index=False)
        syn_concepts = list(dict.fromkeys([c for c in tax["prereg_debate"] if c in valid] + list(sel.concept)))
        S = synchrony(P, syn_concepts, site, "confirmation")
        S.to_csv(TABLES / f"traj_synchrony_{tag}_{site}.csv", index=False)
        out[site] = dict(n_valid=len(valid), drift_disc_sig=int((Dd.q < 0.05).sum()), drift_conf_replicated=int(Dc.replicates.sum()) if len(Dc) else 0,
                         drift_selected=len(sel), lead_disc_sig=int((Ld.q < 0.05).sum()) if len(Ld) else 0,
                         lead_conf_replicated=int(Lc.replicates.sum()) if len(Ld) and len(Lc) else 0)
    # moral drift by domain (decision site), confirmation half, per turn
    moral = [c for c in tax["taxonomy"]["moral"] if c in valid]
    rows = []
    H = P[(P.half == "confirmation") & P.cond.isin(["debate", "pool"])]
    for dom, G in H.groupby("domain"):
        for c in moral:
            for t, g in G.groupby("turn"):
                w = g.groupby(["item_id", "cluster", "cond"])[f"dec_{c}"].mean().unstack("cond").dropna()
                if len(w) < 20:
                    continue
                est, lo, hi, p, n = fast_cluster_mean_ci((w["debate"] - w["pool"]).to_numpy(), w.index.get_level_values("cluster").to_numpy())
                rows.append(dict(domain=dom, concept=c, turn=t, drift=est, lower=lo, upper=hi, n=n))
    pd.DataFrame(rows).to_csv(TABLES / f"traj_moral_{tag}.csv", index=False)
    json.dump(out, open(TABLES / f"traj_summary_{tag}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
