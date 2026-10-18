"""V3-C1 CPU robustness (PLAN_V2.md V3-C1 (b) and (d), frozen 2026-09-17 07:00) and the lock-in vs turn-0 majority baseline.

Sections (run all by default, or pick with --parts):
  iqp     Complementarity I_QP = [Y(strong,exp)-Y(strong,anon)] - [Y(weak,exp)-Y(weak,anon)] under disagreement, estimated
          raw, with the PRE-treatment baseline (neutral-message x anonymous x same-prior world: the receiver's margin toward the
          advocated option before any argument is read), and with the two post-treatment baselines of cpt_analysis.state_matched
          (anon-world margin; no-label-world margin). Also: I_QP by pre-treatment baseline bin (legitimate conditioning), the
          probability-scale interaction split by content correctness, all per domain and on the length-matched half.
          Frozen rule: I_QP is claimed only if raw, pre-, anon- and none-adjusted estimates agree in sign in all three domains.
  state   Within-item state dependence: each item x evidence level has two advocacy directions with different independent
          baselines b (no-label world). Differencing the two directions removes every item-level quantity; the hinge model
          dP = b1*db + b2*d max(b+3, 0) (+ intercept) tests rise-then-fall around the pooled peak (-3, fixed from the pooled
          curve before this analysis). Rise-then-fall iff b1 > 0 and b1 + b2 < 0, both with cluster-bootstrap CIs excluding 0.
          Also: label mass p(o1)+p(o2) per baseline bin (bins with mean mass < 0.99 flagged), the curve on the probability and
          adoption (argmax) scales, and domain/family composition per bin.
  gain    N1 credibility gain on the internal reading of the evidence. Per layer x position, a difference-of-means evidence-
          quality direction (strong - weak, non-neutral worlds) and a provenance direction (expert - anon) are fit on probe_pool
          items; the quality direction is orthogonalised to the provenance direction. On release_600 items (disagreement worlds):
          halo = E[q|expert] - E[q|anon]; gain = (q_strong - q_weak | expert) - (q_strong - q_weak | anon), per item, item-cluster
          bootstrap, in units of the SD of q over release_600 worlds. Positions: answer (decision token; gate), arg_mid and msg_end
          (evidence tokens; negative controls). Control: 20 norm-matched random unit directions per layer x position.
          Frozen gate (b): gain > 0 with CI excl. 0 at >= 2 of L16/19/22 (answer) AND CI incl. 0 at L6 (answer) AND CI incl. 0
          at the evidence tokens (arg_mid and msg_end at L16/19/22; committed before computing). Q4 same sign pending.
  lockin  Reviewer 1.5 E1 on 12-turn release_600 transcripts: turn-0 majority vote vs turn-11 debate / pool.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import *  # noqa
from stats import fast_cluster_mean_ci

DOMAINS = ["knowledge_reasoning", "values_moral_judgment", "safety_refusal"]
HINGE = -3.0
BINS = [-np.inf, -8, -6, -4, -2, 0, 2, 4, 6, np.inf]
SEED = 240091


def unit_of(core):
    return lambda i: core[i]["meta"].get("analysis_group") if core[i]["stratum"] == "persuasion" else core[i]["source_item_id"]


def load_worlds(model, run="cpt_core"):
    W = pd.DataFrame(read_jsonl(ACTS / model / run / "worlds.jsonl"))
    core = {r["id"]: r for r in load_core()}
    W["cluster"] = W.item_id.map(unit_of(core))
    W["row"] = np.arange(len(W))
    return W


def ci_row(vals, clusters, **kw):
    est, lo, hi, p, n = fast_cluster_mean_ci(np.asarray(vals, float), np.asarray(clusters))
    return dict(kw, estimate=est, lower=lo, upper=hi, p=p, n_clusters=n, n=int(np.isfinite(np.asarray(vals, float)).sum()))


def excl0(lo, hi):
    return bool((lo > 0) or (hi < 0))


# ------------------------------------------------------------------------------------------------------------------ iqp
def long_table(W):
    """One row per item x advocated option x evidence level, disagreement worlds."""
    rows = []
    for iid, g in W.groupby("item_id", sort=False):
        o1, o2 = g.o1.iloc[0], g.o2.iloc[0]
        cell = {(r.adv, r.qual, r.prov, r.prior): r for r in g.itertuples(index=False) if r.qual != "neutral"}
        neu = {(r.prov, r.prior): r for r in g.itertuples(index=False) if r.qual == "neutral"}
        for adv in (o1, o2):
            dis = o2 if adv == o1 else o1
            if ("anon", dis) not in neu:
                continue
            base_pre = -neu[("anon", dis)].m_prior  # lp(adv) - lp(dis) under the no-argument message, before any evidence
            for q in ("strong", "weak"):
                c = {p: cell.get((adv, q, p, dis)) for p in ("expert", "anon", "none")}
                if any(v is None for v in c.values()):
                    continue
                rows.append(dict(item_id=iid, cluster=g.cluster.iloc[0], domain=g.domain.iloc[0], family=g.family.iloc[0],
                                 has_target=bool(g.has_target.iloc[0]), adv=adv, direction="o1" if adv == o1 else "o2",
                                 advocacy=("no_target" if not g.has_target.iloc[0] else ("correct" if adv == o1 else "wrong")),
                                 qual=q, P=c["expert"].m_adv - c["anon"].m_adv, Pp=c["expert"].p_adv - c["anon"].p_adv,
                                 Pa=float(c["expert"].argmax_label == adv) - float(c["anon"].argmax_label == adv),
                                 base_anon=c["anon"].m_adv, base_none=c["none"].m_adv, base_pre=base_pre,
                                 mass_exp=c["expert"].p_label_mass, mass_anon=c["anon"].p_label_mass,
                                 mass_none=c["none"].p_label_mass))
    return pd.DataFrame(rows)


def ols_cluster(df, formula, term):
    import statsmodels.formula.api as smf
    try:
        m = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.cluster)[0]})
        ci = m.conf_int().loc[term]
        return float(m.params[term]), float(ci[0]), float(ci[1])
    except Exception:  # noqa: BLE001
        return np.nan, np.nan, np.nan


def part_iqp(model, W, L, lm_items):
    term = 'C(qual, Treatment("weak"))[T.strong]'
    rows, cond_rows, prob_rows = [], [], []
    sets = [("full", L), ("length_matched", L[L.item_id.isin(lm_items)])]
    for sname, S in sets:
        for gname, G in [("ALL", S)] + [(d, S[S.domain == d]) for d in DOMAINS]:
            for adj in ("raw", "pre", "anon", "none"):
                spline = "" if adj == "raw" else f" + bs(base_{adj}, df=5)"
                e, lo, hi = ols_cluster(G, 'P ~ C(qual, Treatment("weak"))' + spline, term)
                rows.append(dict(set=sname, group=gname, adjustment=adj, estimate=e, lower=lo, upper=hi,
                                 n_items=G.item_id.nunique(), excl0=excl0(lo, hi)))
            # within item x direction interaction, conditioned on the pre-treatment baseline bin
            piv = G.pivot_table(index=["item_id", "cluster", "adv", "base_pre", "advocacy"], columns="qual",
                                values=["P", "Pp"], aggfunc="first").dropna().reset_index()
            piv.columns = ["_".join(c).strip("_") for c in piv.columns]
            piv["IQP"] = piv.P_strong - piv.P_weak
            piv["IQPp"] = piv.Pp_strong - piv.Pp_weak
            piv["bin"] = pd.cut(piv.base_pre, BINS)
            for b, g in piv.groupby("bin", observed=True):
                cond_rows.append(ci_row(g.IQP, g.cluster, set=sname, group=gname, bin=str(b), bin_mid=bin_mid(b)))
            for adv_cls, g in [("all", piv)] + [(a, piv[piv.advocacy == a]) for a in ("correct", "wrong", "no_target")]:
                if len(g) < 20:
                    continue
                prob_rows.append(ci_row(g.IQPp, g.cluster, set=sname, group=gname, advocacy=adv_cls, scale="probability"))
                prob_rows.append(ci_row(g.IQP, g.cluster, set=sname, group=gname, advocacy=adv_cls, scale="logit"))
    R, C, Pr = pd.DataFrame(rows), pd.DataFrame(cond_rows), pd.DataFrame(prob_rows)
    verdict = {}
    for sname in ("full", "length_matched"):
        dom = R[(R.set == sname) & R.group.isin(DOMAINS)]
        per = {d: sorted(set(np.sign(dom[dom.group == d].estimate))) for d in DOMAINS}
        agree = {d: len(s) == 1 for d, s in per.items()}
        verdict[sname] = dict(sign_agreement_by_domain=agree, signs={d: [int(x) for x in s] for d, s in per.items()},
                              PASS=all(agree.values()))
    R.to_csv(TABLES / f"c1r_{model}_iqp_adjustments.csv", index=False)
    C.to_csv(TABLES / f"c1r_{model}_iqp_by_prebaseline.csv", index=False)
    Pr.to_csv(TABLES / f"c1r_{model}_iqp_scales.csv", index=False)
    return dict(rule=verdict, table=R.round(4).to_dict("records"))


def bin_mid(b):
    if np.isfinite(b.left) and np.isfinite(b.right):
        return float((b.left + b.right) / 2)
    return float(b.right - 1) if np.isfinite(b.right) else float(b.left + 1)


# ---------------------------------------------------------------------------------------------------------------- state
def cluster_boot_ols(X, y, clusters, B=2000, seed=SEED):
    """OLS coefficients with an item-cluster bootstrap (resampling per-cluster X'X and X'y sums)."""
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    X, y, cl = X[ok], y[ok], np.asarray(clusters)[ok]
    codes, uniq = pd.factorize(cl)
    k = X.shape[1]
    XtX = np.zeros((len(uniq), k, k))
    Xty = np.zeros((len(uniq), k))
    np.add.at(XtX, codes, X[:, :, None] * X[:, None, :])
    np.add.at(Xty, codes, X * y[:, None])
    beta = np.linalg.solve(XtX.sum(0), Xty.sum(0))
    rng = np.random.default_rng(seed)
    bs = np.empty((B, k))
    for b in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        bs[b] = np.linalg.lstsq(XtX[pick].sum(0), Xty[pick].sum(0), rcond=None)[0]
    return beta, bs, len(uniq), int(ok.sum())


def part_state(model, W, L):
    out = {}
    # ---- within-item differences between the two advocacy directions (same item, same evidence level)
    piv = L.pivot_table(index=["item_id", "cluster", "domain", "family", "has_target", "qual"], columns="direction",
                        values=["P", "base_none", "base_pre"], aggfunc="first").dropna().reset_index()
    piv.columns = ["_".join(c).strip("_") for c in piv.columns]
    hinge = lambda b: np.maximum(b - HINGE, 0.0)
    fits = []
    for base in ("base_none", "base_pre"):
        b1, b2 = piv[f"{base}_o1"].to_numpy(), piv[f"{base}_o2"].to_numpy()
        dP = (piv.P_o1 - piv.P_o2).to_numpy()
        for sub, m in (("all", np.ones(len(piv), bool)), ("no_target", ~piv.has_target.to_numpy(bool)),
                       ("target", piv.has_target.to_numpy(bool))):
            X = np.column_stack([np.ones(m.sum()), (b1 - b2)[m], (hinge(b1) - hinge(b2))[m]])
            beta, bs, ncl, n = cluster_boot_ols(X, dP[m], piv.cluster.to_numpy()[m])
            post = bs[:, 1] + bs[:, 2]
            q = lambda v: np.quantile(v, [0.025, 0.975])
            Xq = np.column_stack([np.ones(m.sum()), (b1 - b2)[m], (b1 ** 2 - b2 ** 2)[m]])
            gq, gbs, _, _ = cluster_boot_ols(Xq, dP[m], piv.cluster.to_numpy()[m])
            vert = -gbs[:, 1] / (2 * gbs[:, 2])
            f = dict(baseline=base, subset=sub, n_pairs=n, n_clusters=ncl,
                     slope_below=float(beta[1]), slope_below_ci=[float(x) for x in q(bs[:, 1])],
                     slope_above=float(beta[1] + beta[2]), slope_above_ci=[float(x) for x in q(post)],
                     intercept=float(beta[0]), quad=float(gq[2]), quad_ci=[float(x) for x in q(gbs[:, 2])],
                     vertex=float(-gq[1] / (2 * gq[2])), vertex_ci=[float(x) for x in np.nanquantile(vert, [0.025, 0.975])])
            f["rise_then_fall"] = bool(f["slope_below_ci"][0] > 0 and f["slope_above_ci"][1] < 0)
            fits.append(f)
    F = pd.DataFrame(fits)
    F.to_csv(TABLES / f"c1r_{model}_state_within_item.csv", index=False)
    prim = F[(F.baseline == "base_none") & (F.subset == "all")].iloc[0]
    out["within_item"] = F.to_dict("records")
    out["gate_rise_then_fall_primary"] = bool(prim.rise_then_fall)
    # ---- per-bin curve (pooled, independent baseline) on logit, probability and adoption scales + label mass + composition
    L = L.copy()
    L["bin"] = pd.cut(L.base_none, BINS)
    L["mass_min"] = L[["mass_exp", "mass_anon"]].min(1)
    rows, comp = [], []
    for (b, q), g in L.groupby(["bin", "qual"], observed=True):
        base = dict(bin=str(b), bin_mid=bin_mid(b), qual=q, n_rows=len(g), mass_mean=g[["mass_exp", "mass_anon"]].mean(1).mean(),
                    mass_p05=float(np.quantile(g.mass_min, 0.05)), frac_mass_lt_099=float((g.mass_min < 0.99).mean()))
        for sc, col in (("logit", "P"), ("probability", "Pp"), ("adoption", "Pa")):
            r = ci_row(g[col], g.cluster)
            rows.append(dict(base, scale=sc, estimate=r["estimate"], lower=r["lower"], upper=r["upper"], n_clusters=r["n_clusters"]))
    B = pd.DataFrame(rows)
    B["flag_mass_lt_099"] = B.mass_mean < 0.99
    B.to_csv(TABLES / f"c1r_{model}_state_bins.csv", index=False)
    for b, g in L.groupby("bin", observed=True):
        n = len(g)
        for d in DOMAINS:
            comp.append(dict(bin=str(b), bin_mid=bin_mid(b), level="domain", group=d, share=(g.domain == d).mean(), n_rows=n))
        for f, c in g.family.value_counts().items():
            comp.append(dict(bin=str(b), bin_mid=bin_mid(b), level="family", group=f, share=c / n, n_rows=n))
    pd.DataFrame(comp).to_csv(TABLES / f"c1r_{model}_state_composition.csv", index=False)
    out["bins_flagged_mass_lt_099"] = sorted(set(B[B.flag_mass_lt_099].bin))
    return out


# ----------------------------------------------------------------------------------------------------------------- gain
POS = {"answer": 0, "msg_end": 1, "arg_mid": 3}


def row_ranges(idx):
    idx = np.sort(np.asarray(idx))
    br = np.where(np.diff(idx) != 1)[0]
    starts = np.r_[idx[0], idx[br + 1]]
    ends = np.r_[idx[br] + 1, idx[-1] + 1]
    return list(zip(starts, ends))


def auroc(pos, neg):
    x = np.r_[pos, neg]
    r = pd.Series(x).rank().to_numpy()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def part_gain(model, W, n_random=20, chunk=1500):
    meta = json.load(open(ACTS / model / "cpt_core" / "meta.json"))
    grid = meta["grid"]
    A = np.load(ACTS / model / "cpt_core" / "acts.npy", mmap_mode="r")
    splits = json.load(open(ROOT / "data/splits/splits.json"))
    pool, rel = set(splits["probe_pool"]), set(splits["release_600"])
    pnames = list(POS)
    pidx = [POS[p] for p in pnames]
    d = A.shape[-1]
    NW = W[W.qual != "neutral"]
    # ---- pass 1: difference-of-means directions on probe_pool items
    fitrows = NW[NW.item_id.isin(pool)]
    groups = {"strong": (fitrows.qual == "strong"), "weak": (fitrows.qual == "weak"),
              "expert": (fitrows.prov == "expert"), "anon": (fitrows.prov == "anon")}
    lab = {g: set(fitrows.row[m]) for g, m in groups.items()}
    sums = {g: np.zeros((len(pnames), len(grid), d)) for g in groups}
    cnts = {g: 0 for g in groups}
    t = time.time()
    fit_idx = np.sort(fitrows.row.to_numpy())
    for s, e in row_ranges(fit_idx):
        for c0 in range(s, e, chunk):
            c1 = min(e, c0 + chunk)
            X = np.asarray(A[c0:c1][:, pidx], dtype=np.float32)
            ids = np.arange(c0, c1)
            for g in groups:
                m = np.fromiter((i in lab[g] for i in ids), bool, len(ids))
                if m.any():
                    sums[g] += X[m].sum(0, dtype=np.float64)
                    cnts[g] += int(m.sum())
    print(f"[gain] pass 1 {time.time()-t:.0f}s counts {cnts}", flush=True)
    mean = {g: sums[g] / cnts[g] for g in groups}
    U_q = mean["strong"] - mean["weak"]
    U_p = mean["expert"] - mean["anon"]
    U_p = U_p / np.linalg.norm(U_p, axis=-1, keepdims=True)
    U_q_raw = U_q / np.linalg.norm(U_q, axis=-1, keepdims=True)
    U_q = U_q - (U_q * U_p).sum(-1, keepdims=True) * U_p
    U_q = U_q / np.linalg.norm(U_q, axis=-1, keepdims=True)
    rng = np.random.default_rng(SEED)
    R = rng.standard_normal((n_random, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    # ---- pass 2: projections of release_600 worlds
    ev = W[W.item_id.isin(rel)]
    ev_idx = np.sort(ev.row.to_numpy())
    proj = {k: np.zeros((len(W), len(pnames), len(grid)), np.float32) for k in ("q", "q_raw", "p")}
    projR = np.zeros((len(W), len(pnames), len(grid), n_random), np.float32)
    t = time.time()
    for s, e in row_ranges(ev_idx):
        for c0 in range(s, e, chunk):
            c1 = min(e, c0 + chunk)
            X = np.asarray(A[c0:c1][:, pidx], dtype=np.float32)
            proj["q"][c0:c1] = np.einsum("npld,pld->npl", X, U_q.astype(np.float32))
            proj["q_raw"][c0:c1] = np.einsum("npld,pld->npl", X, U_q_raw.astype(np.float32))
            proj["p"][c0:c1] = np.einsum("npld,pld->npl", X, U_p.astype(np.float32))
            projR[c0:c1] = X @ R.T.astype(np.float32)
    print(f"[gain] pass 2 {time.time()-t:.0f}s", flush=True)
    ev = ev[ev.qual != "neutral"]
    rows, rand_rows, auc_rows = [], [], []
    dis = ev[ev.adv != ev.prior]
    for world_set, E in (("disagreement", dis), ("all_nonneutral", ev)):
        for pi, pn in enumerate(pnames):
            for li, layer in enumerate(grid):
                for kind in ("q", "q_raw"):
                    v = proj[kind][E.row.to_numpy(), pi, li]
                    sd = float(np.std(v))
                    rows.extend(halo_gain(E, v / sd, dict(world_set=world_set, position=pn, layer=layer, direction=kind, sd=sd)))
                if world_set == "disagreement":
                    vR = projR[E.row.to_numpy(), pi, li]
                    g_obs = [r for r in rows if r["world_set"] == world_set and r["position"] == pn and r["layer"] == layer
                             and r["direction"] == "q" and r["quantity"] == "gain"][0]["estimate"]
                    gr = []
                    for k in range(n_random):
                        v = vR[:, k]
                        gr.append(point_gain(E, v / np.std(v)))
                    gr = np.abs(np.array(gr))
                    rand_rows.append(dict(position=pn, layer=layer, gain_obs_sd=g_obs, rand_abs_mean=float(gr.mean()),
                                          rand_abs_p95=float(np.quantile(gr, 0.95)), ratio_rand_p95_to_obs=float(np.quantile(gr, 0.95) / max(abs(g_obs), 1e-9))))
                    for kind in ("q", "q_raw", "p"):
                        v = proj[kind][E.row.to_numpy(), pi, li]
                        auc_rows.append(dict(position=pn, layer=layer, direction=kind,
                                             prov_auroc=auroc(v[(E.prov == "expert").to_numpy()], v[(E.prov == "anon").to_numpy()]),
                                             quality_auroc=auroc(v[(E.qual == "strong").to_numpy()], v[(E.qual == "weak").to_numpy()])))
    G, Rr, Au = pd.DataFrame(rows), pd.DataFrame(rand_rows), pd.DataFrame(auc_rows)
    G.to_csv(TABLES / f"c1r_{model}_gain.csv", index=False)
    Rr.to_csv(TABLES / f"c1r_{model}_gain_random.csv", index=False)
    Au.to_csv(TABLES / f"c1r_{model}_gain_auroc.csv", index=False)
    # ---- frozen gate (b), Q9 part
    sel = G[(G.world_set == "disagreement") & (G.direction == "q") & (G.quantity == "gain")].set_index(["position", "layer"])
    gl = [16, 19, 22]
    sig_answer = [bool(sel.loc[("answer", l)].lower > 0) for l in gl]
    l6_incl = not excl0(sel.loc[("answer", 6)].lower, sel.loc[("answer", 6)].upper)
    ev_incl = {f"{p}_L{l}": not excl0(sel.loc[(p, l)].lower, sel.loc[(p, l)].upper) for p in ("arg_mid", "msg_end") for l in gl}
    gate = dict(answer_gain_pos_sig={f"L{l}": s for l, s in zip(gl, sig_answer)}, n_sig=int(sum(sig_answer)),
                L6_answer_ci_includes_0=l6_incl, evidence_tokens_ci_includes_0=ev_incl,
                PASS_Q9=bool(sum(sig_answer) >= 2 and l6_incl and all(ev_incl.values())), Q4_same_sign="pending (CPT_score_Q4_core)")
    return gate


def item_cells(E, v):
    df = pd.DataFrame(dict(item_id=E.item_id.to_numpy(), cluster=E.cluster.to_numpy(), adv=E.adv.to_numpy(),
                           prior=E.prior.to_numpy(), qual=E.qual.to_numpy(), prov=E.prov.to_numpy(), v=v))
    df = df[df.prov != "none"]
    return df.pivot_table(index=["item_id", "cluster", "adv", "prior"], columns=["qual", "prov"], values="v", aggfunc="mean")


def point_gain(E, v):
    c = item_cells(E, v)
    g = (c[("strong", "expert")] - c[("weak", "expert")]) - (c[("strong", "anon")] - c[("weak", "anon")])
    return float(g.groupby(level="item_id").mean().mean())


def halo_gain(E, v, base):
    c = item_cells(E, v)
    halo = ((c[("strong", "expert")] + c[("weak", "expert")]) - (c[("strong", "anon")] + c[("weak", "anon")])) / 2
    gain = (c[("strong", "expert")] - c[("weak", "expert")]) - (c[("strong", "anon")] - c[("weak", "anon")])
    qeff = ((c[("strong", "expert")] + c[("strong", "anon")]) - (c[("weak", "expert")] + c[("weak", "anon")])) / 2
    out = []
    for name, s in (("halo", halo), ("gain", gain), ("quality", qeff)):
        per = s.groupby(level=["item_id", "cluster"]).mean().reset_index()
        out.append(ci_row(per[0], per.cluster, quantity=name, **base))
    return out


# --------------------------------------------------------------------------------------------------------------- lockin
def part_lockin(model, dataset="release_600", turns=12):
    core = {r["id"]: r for r in load_core()}
    unit = unit_of(core)
    d = RAW / model / dataset
    last = turns - 1
    V = {}
    for f, cond in (("t0", "t0"), (f"debate_t{last}", "debate"), (f"pool_t{last}", "pool")):
        for x in read_jsonl(d / f"{f}.jsonl"):
            if x["agent"] < 3:
                V.setdefault(x["item_id"], {}).setdefault(cond, {})[x["agent"]] = x
    rows = []
    for iid, dd in V.items():
        v = item_view(core[iid])
        if any(len(dd.get(c, {})) < 3 for c in ("t0", "debate", "pool")):
            continue
        rec = dict(item_id=iid, cluster=unit(iid), domain=core[iid]["primary_domain"], family=family_of(core[iid]),
                   has_target=v["target"] is not None and v["kind"] != "deception")

        def summ(c):
            vs = [dd[c][a]["verdict"] for a in range(3)]
            if any(not isinstance(x, str) for x in vs):
                return None
            cnt = pd.Series(vs).value_counts()
            maj = cnt.index[0] if (len(cnt) == 1 or cnt.iloc[0] > cnt.iloc[1]) else None
            ok = None
            if rec["has_target"] and maj is not None:
                ok = bool(numeric_equal(maj, v["target"])) if v["kind"] == "numeric" else maj == v["target"]
            acc = np.nanmean([float(x) if isinstance(x, bool) else np.nan for x in (dd[c][a]["correct"] for a in range(3))]) \
                if rec["has_target"] else np.nan
            return dict(maj=maj, maj_ok=ok, unan=len(cnt) == 1, acc=acc)
        s = {c: summ(c) for c in ("t0", "debate", "pool")}
        if any(x is None for x in s.values()):
            continue
        rec.update(t0_unan=s["t0"]["unan"], t0_has_maj=s["t0"]["maj"] is not None)
        for c in ("t0", "debate", "pool"):
            rec[f"{c}_acc"] = s[c]["acc"]
            rec[f"{c}_unan"] = float(s[c]["unan"])
            if rec["has_target"]:
                rec[f"{c}_maj_acc"] = float(s[c]["maj_ok"] is True)  # no majority counts as not correct
                rec[f"{c}_maj_wrong"] = float(s[c]["maj_ok"] is False)
                rec[f"{c}_wrong_unan"] = float(s[c]["unan"] and s[c]["maj_ok"] is False)
            if c != "t0":
                rec[f"{c}_keeps_t0_maj"] = (float(s[c]["maj"] == s["t0"]["maj"]) if s["t0"]["maj"] is not None else np.nan)
                rec[f"{c}_unan_on_t0_maj"] = (float(s[c]["unan"] and s[c]["maj"] == s["t0"]["maj"]) if s["t0"]["maj"] is not None else np.nan)
        rows.append(rec)
    D = pd.DataFrame(rows)
    for c in ("t0_unan", "t0_has_maj", "has_target"):
        D[c] = D[c].astype(bool)
    D.to_csv(TABLES / f"c1r_{model}_lockin_items.csv", index=False)
    est = []
    for gname, G in [("ALL", D)] + [(dm, D[D.domain == dm]) for dm in DOMAINS]:
        T = G[G.has_target]
        for name, col, S in (("t0_mean_agent_acc", "t0_acc", T), ("t0_majority_acc", "t0_maj_acc", T),
                             ("t0_majority_wrong", "t0_maj_wrong", T), ("t0_wrong_unanimous", "t0_wrong_unan", T),
                             ("debate_t11_mean_agent_acc", "debate_acc", T), ("debate_t11_majority_acc", "debate_maj_acc", T),
                             ("debate_t11_wrong_unanimous", "debate_wrong_unan", T), ("pool_t11_mean_agent_acc", "pool_acc", T),
                             ("pool_t11_majority_acc", "pool_maj_acc", T), ("pool_t11_wrong_unanimous", "pool_wrong_unan", T)):
            est.append(ci_row(S[col], S.cluster, group=gname, estimand=name))
        for name, a, b in (("debate_maj_acc_minus_t0_maj_acc", "debate_maj_acc", "t0_maj_acc"),
                           ("pool_maj_acc_minus_t0_maj_acc", "pool_maj_acc", "t0_maj_acc"),
                           ("debate_wrong_unan_minus_t0_maj_wrong", "debate_wrong_unan", "t0_maj_wrong"),
                           ("pool_wrong_unan_minus_t0_maj_wrong", "pool_wrong_unan", "t0_maj_wrong"),
                           ("debate_minus_pool_maj_acc", "debate_maj_acc", "pool_maj_acc"),
                           ("debate_minus_pool_wrong_unan", "debate_wrong_unan", "pool_wrong_unan")):
            est.append(ci_row(T[a] - T[b], T.cluster, group=gname, estimand=name))
        # among t0-majority-wrong / t0-majority-right target items: where does the debate end?
        for lab_, m in (("t0_maj_wrong", T.t0_maj_wrong == 1), ("t0_maj_right", T.t0_maj_acc == 1)):
            S = T[m]
            for c in ("debate", "pool"):
                est.append(ci_row(S[f"{c}_maj_acc"], S.cluster, group=gname, estimand=f"{c}_ends_majority_correct | {lab_}"))
                est.append(ci_row(S[f"{c}_wrong_unan"], S.cluster, group=gname, estimand=f"{c}_ends_wrong_unanimous | {lab_}"))
        split = G[(~G.t0_unan) & G.t0_has_maj]
        for c in ("debate", "pool"):
            est.append(ci_row(split[f"{c}_keeps_t0_maj"], split.cluster, group=gname, estimand=f"{c}_t11_majority_equals_t0_majority | t0_split"))
            est.append(ci_row(split[f"{c}_unan_on_t0_maj"], split.cluster, group=gname, estimand=f"{c}_t11_unanimous_on_t0_majority | t0_split"))
        est.append(ci_row(split.debate_keeps_t0_maj - split.pool_keeps_t0_maj, split.cluster, group=gname,
                          estimand="debate_minus_pool_keeps_t0_majority | t0_split"))
    E = pd.DataFrame(est)
    E.to_csv(TABLES / f"c1r_{model}_lockin.csv", index=False)
    a = E[E.group == "ALL"].set_index("estimand")
    restate = (not excl0(a.loc["debate_maj_acc_minus_t0_maj_acc"].lower, a.loc["debate_maj_acc_minus_t0_maj_acc"].upper)) and \
              (not excl0(a.loc["debate_wrong_unan_minus_t0_maj_wrong"].lower, a.loc["debate_wrong_unan_minus_t0_maj_wrong"].upper))
    return dict(majority_vote_restatement=bool(restate), n_items=len(D), n_target=int(D.has_target.sum()),
                n_t0_split_with_majority=int(((~D.t0_unan) & D.t0_has_maj).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--parts", default="iqp,state,gain,lockin")
    a = ap.parse_args()
    parts = a.parts.split(",")
    summ_p = TABLES / f"c1r_{a.model}_summary.json"
    summary = json.load(open(summ_p)) if summ_p.exists() else {}
    if {"iqp", "state", "gain"} & set(parts):
        W = load_worlds(a.model)
    if {"iqp", "state"} & set(parts):
        L = long_table(W)
        L.to_csv(TABLES / f"c1r_{a.model}_long.csv", index=False)
        lm = set(json.load(open(ROOT / "data/mad_interp/length_matched_items.json"))["items"])
    if "iqp" in parts:
        summary["iqp"] = part_iqp(a.model, W, L, lm)
        print(json.dumps(summary["iqp"]["rule"], indent=1), flush=True)
    if "state" in parts:
        summary["state"] = part_state(a.model, W, L)
        print(pd.DataFrame(summary["state"]["within_item"]).round(3).to_string(), flush=True)
    if "gain" in parts:
        summary["gain_gate"] = part_gain(a.model, W)
        print(json.dumps(summary["gain_gate"], indent=1), flush=True)
    if "lockin" in parts:
        summary["lockin"] = part_lockin(a.model)
        print(json.dumps(summary["lockin"], indent=1), flush=True)
    json.dump(summary, open(summ_p, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
