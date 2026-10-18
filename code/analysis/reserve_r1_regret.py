"""Reserve R1 (V3 plan): post-capitulation regret asymmetry, Q9 debates (release_600 = DISCOVERY; core gate frozen separately).

Episode: debate agent-turn t whose agent was outnumbered at t-1 (both peers share M != own X) and gives in at t (public verdict M),
on items with a target, split by whether the abandoned answer X was right (right-minority capitulation: moves to a wrong M) or wrong
(wrong-minority capitulation: moves to the right M). States are read at the capitulation turn t (the prompt that produced the
switch) and at t+1 (the agent now reads its own capitulated message and its peers).
Readers of the byte-identical text (interp/observer_replay.py): participant (replay), minimal observer (obsmin), paraphrased
participant (para, wording null). Measures per reader:
  mX      = lp(X) - lp(M), the unprinted pull toward the abandoned answer (t: reluctance of the switch; t+1: residual pull)
  probe   = frozen content_correctness DoM read-out (CPT worlds, L19, decision token; never trained on debates)
  concept = validated concept z (layer 16) at the decision and reading sites; participant - reader deltas residualised on the
            GENERAL-axis delta (inner_life.residualise, slope fit on all release_600 agent-turns).
Primary estimand (regret asymmetry): DiD = [participant - observer](right-minority) - [participant - observer](wrong-minority)
of mX at t+1; paraphrase version as the wording null. Paired cluster bootstrap (inner_life.Clusters: one resample of item clusters
for every arm), BH-FDR over validated concepts, 200 random unit directions at layer 16 as the concept-specificity null.
Behaviour: reversal back to X at any later public turn; private re-query answer == X (final unanimous only).
Prediction: family-adjusted and pooled AUROC (hollow.auroc_table) of the t+1 regret signal for later reversal, vs transcript
baselines (TF-IDF on the t+1 message; public history: confidence at t and t+1, prior number of verdict changes, turn).
Core power: counts only, from outputs/raw/Q9/core transcripts (no core activations or outcomes are read).
"""
from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
from common import *  # noqa
from stats import bh_fdr
from inner_life import Clusters, KEY, PREREG, SITES, build, paired_diff, paired_mean, reader_delta, residualise
from hollow import auroc_table, oof_text_model, probe_scores

SEED = 240091
B = 2000


def lp_margin(lp, a, b):
    if not isinstance(lp, dict) or a not in lp or b not in lp:
        return np.nan
    return lp[a] - lp[b]


def ci_row(name, est, lo, hi, n=None, **kw):
    return dict(quantity=name, estimate=float(est), lower=float(lo), upper=float(hi), n=n, **kw)


def core_power(model="Q9"):
    """Eligible R1 episodes in the 4-turn core debates outside release_600 (counts only)."""
    core = {r["id"]: r for r in load_core()}
    rel = set(json.load(open(ROOT / "data/splits/splits.json"))["release_600"])
    recs = {}
    for f in ["t0.jsonl"] + [f"debate_t{t}.jsonl" for t in range(1, 4)]:
        for x in read_jsonl(RAW / model / "core" / f):
            if x["agent"] < 3 and x["item_id"] not in rel:
                recs[(x["item_id"], x["turn"], x["agent"])] = x
    cnt = dict(right_t=0, wrong_t=0, right_t1=0, wrong_t1=0)
    clusters = dict(right_t1=set(), wrong_t1=set())
    for (iid, t, ag), x in recs.items():
        if t < 1:
            continue
        v = item_view(core[iid])
        if v["kind"] != "choice" or v["target"] is None or family_of(core[iid]) == "strategic_deception":
            continue
        prev = [recs.get((iid, t - 1, b)) for b in range(3)]
        if any(p is None or p["verdict"] is None for p in prev):
            continue
        others = [prev[b]["verdict"] for b in range(3) if b != ag]
        X = prev[ag]["verdict"]
        if not (others[0] == others[1] != X) or x["verdict"] != others[0]:
            continue
        side = "right" if X == v["target"] else "wrong"
        cnt[f"{side}_t"] += 1
        if (iid, t + 1, ag) in recs:
            cnt[f"{side}_t1"] += 1
            u = core[iid]["meta"].get("analysis_group") if core[iid]["stratum"] == "persuasion" else core[iid]["source_item_id"]
            clusters[f"{side}_t1"].add(u)
    cnt.update({f"clusters_{k}": len(v) for k, v in clusters.items()})
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--n_random", type=int, default=200)
    a = ap.parse_args()
    M_ = a.model
    ds, T_ = "release_600", 12
    args = SimpleNamespace(model=M_, replay=f"replay_{ds}_{ds}_T{T_}", exclude_split=None, observer=f"reader_obsmin_{ds}_T{T_}",
                           null=f"reader_para_{ds}_T{T_}", extra=[])
    base, concepts, valid, readers = build(args)
    n = len(base)
    core = {r["id"]: r for r in load_core()}
    # ---------------------------------------------------------------- rows of every reader
    ridx = pd.DataFrame(read_jsonl(ACTS / M_ / args.replay / "index.jsonl")).drop(columns=["lp_letters"])
    prow = base[KEY].merge(ridx[ridx.cond == "debate"][KEY + ["row"]], on=KEY, how="left").row.to_numpy(float)
    fz = json.load(open(ACTS / M_ / "probes_frozen_cpt_core.json"))["content_correctness"]
    _, pr_dom = probe_scores(ACTS / M_ / args.replay / "acts.npy", fz)
    probe = {"participant": np.where(np.isfinite(prow), pr_dom[np.nan_to_num(prow).astype(int)], np.nan)}
    lps = {"participant": base.lp_letters.to_numpy()}
    deltas = {}
    for nm in ("obsmin", "para"):
        run = readers[nm]["run"]
        oidx = pd.DataFrame(read_jsonl(ACTS / M_ / run / "index.jsonl")).drop(columns=["lp_letters"])
        orow = base[KEY].merge(oidx[KEY + ["row"]], on=KEY, how="left").row.to_numpy(float)
        _, o_dom = probe_scores(ACTS / M_ / run / "acts.npy", fz)
        probe[nm] = np.where(np.isfinite(orow), o_dom[np.nan_to_num(orow).astype(int)], np.nan)
        D, aligned = reader_delta(base, readers[nm]["df"], concepts)
        lps[nm] = aligned.lp_letters.to_numpy()
        deltas[nm] = residualise(D, base, concepts, "general", np.ones(n, bool))
    # ---------------------------------------------------------------- episodes
    lookup = {k: i for i, k in enumerate(zip(base.item_id, base.agent, base.turn))}
    pc = base.prev_correct.astype(str)
    # deception items are excluded: their replay prefill is "RESPONSE:", so letter margins and the decision-token site differ
    ep_mask = ((base.role == "outnumbered") & (base.move == "gives_in") & pc.isin(["True", "False"])
               & (base.family != "strategic_deception")).to_numpy()
    E = base[ep_mask][KEY + ["X", "M", "cluster", "family", "domain", "confidence"]].copy()
    E["right"] = (pc[ep_mask] == "True").to_numpy()
    E["i_t"] = np.flatnonzero(ep_mask)
    E["i_t1"] = [lookup.get((i, ag, t + 1), -1) for i, ag, t in zip(E.item_id, E.agent, E.turn)]
    E = E.reset_index(drop=True)
    has1 = (E.i_t1 >= 0).to_numpy()
    print(f"[stage] capitulations: right-minority {int(E.right.sum())}, wrong-minority {int((~E.right).sum())}; with t+1: "
          f"{int((E.right & has1).sum())} / {int((~E.right & has1).sum())}", flush=True)
    for r in ("participant", "obsmin", "para"):
        for tag, col in (("t", "i_t"), ("t1", "i_t1")):
            idx = E[col].to_numpy()
            E[f"mX_{r}_{tag}"] = [lp_margin(lps[r][i], x, m) if i >= 0 else np.nan for i, x, m in zip(idx, E.X, E.M)]
            E[f"probe_{r}_{tag}"] = [probe[r][i] if i >= 0 else np.nan for i in idx]
    for tag in ("t", "t1"):
        for r in ("obsmin", "para"):
            E[f"dmX_{r}_{tag}"] = E[f"mX_participant_{tag}"] - E[f"mX_{r}_{tag}"]
            E[f"dprobe_{r}_{tag}"] = E[f"probe_participant_{tag}"] - E[f"probe_{r}_{tag}"]
    # behaviour: later reversal, private re-query
    V = {k: v for k, v in zip(zip(base.item_id, base.agent, base.turn), base.verdict)}
    C = {k: v for k, v in zip(zip(base.item_id, base.agent, base.turn), base.confidence)}
    E["reverts_t1"] = [V.get((i, ag, t + 1)) == x if (i, ag, t + 1) in V else np.nan for i, ag, t, x in zip(E.item_id, E.agent, E.turn, E.X)]
    E["reverts_later"] = [any(V.get((i, ag, s)) == x for s in range(t + 1, T_)) if t + 1 < T_ else np.nan
                          for i, ag, t, x in zip(E.item_id, E.agent, E.turn, E.X)]
    E["conf_t1"] = [C.get((i, ag, t + 1), np.nan) for i, ag, t in zip(E.item_id, E.agent, E.turn)]
    E["n_changes_before"] = [sum(V.get((i, ag, s)) != V.get((i, ag, s - 1)) for s in range(1, t + 1)) for i, ag, t in zip(E.item_id, E.agent, E.turn)]
    priv = {(x["item_id"], x["agent"]): x["verdict"] for x in read_jsonl(RAW / M_ / ds / "private.jsonl") if x["cond"] == "private_debate"}
    fin = base[base.turn == T_ - 1]
    un = fin.groupby("item_id").verdict.agg(lambda s: bool(s.notna().all() and s.nunique() == 1 and len(s) == 3))
    E["final_unanimous"] = E.item_id.map(un).fillna(False).astype(bool)
    E["final_verdict"] = [V.get((i, ag, T_ - 1)) for i, ag in zip(E.item_id, E.agent)]
    E["private_to_X"] = [(priv.get((i, ag)) == x) if u and priv.get((i, ag)) is not None else np.nan
                         for i, ag, x, u in zip(E.item_id, E.agent, E.X, E.final_unanimous)]
    tx = {}
    for t in range(1, T_):
        for x in read_jsonl(RAW / M_ / ds / f"debate_t{t}.jsonl"):
            if x["agent"] < 3:
                tx[(x["item_id"], x["agent"], t)] = x["text"]
    E["text_t1"] = [tx.get((i, ag, t + 1), "") for i, ag, t in zip(E.item_id, E.agent, E.turn)]
    # ---------------------------------------------------------------- estimands (paired cluster bootstrap)
    cl = Clusters(E.cluster.to_numpy(), B)
    R_, W_ = E.right.to_numpy(), ~E.right.to_numpy()
    rows = []
    scal = [f"mX_{r}_{t}" for t in ("t", "t1") for r in ("participant", "obsmin", "para")] + \
           [f"dmX_{r}_{t}" for t in ("t", "t1") for r in ("obsmin", "para")] + \
           [f"probe_{r}_{t}" for t in ("t", "t1") for r in ("participant", "obsmin", "para")] + \
           [f"dprobe_{r}_{t}" for t in ("t", "t1") for r in ("obsmin", "para")] + ["reverts_t1", "reverts_later", "private_to_X", "conf_t1"]
    for q in scal:
        y = E[q].astype(float).to_numpy()
        ok = np.isfinite(y)
        Y = np.nan_to_num(y)[:, None]
        e, lo, hi, se, p, bt = paired_diff(cl, Y, R_ & ok, W_ & ok)
        er, lr, hr, _ = paired_mean(cl, Y, R_ & ok)
        ew, lw, hw, _ = paired_mean(cl, Y, W_ & ok)
        rows.append(dict(quantity=q, right=er[0], right_lo=lr[0], right_hi=hr[0], wrong=ew[0], wrong_lo=lw[0], wrong_hi=hw[0],
                         right_minus_wrong=e[0], lower=lo[0], upper=hi[0], p=p[0], n_right=int((R_ & ok).sum()), n_wrong=int((W_ & ok).sum())))
    S = pd.DataFrame(rows)
    # |DiD observer| - |DiD para| for the primary estimand
    extra = {}
    for tag in ("t", "t1"):
        yo = E[f"dmX_obsmin_{tag}"].to_numpy(float)
        yp = E[f"dmX_para_{tag}"].to_numpy(float)
        ok = np.isfinite(yo) & np.isfinite(yp)
        Y = np.column_stack([np.nan_to_num(yo), np.nan_to_num(yp), np.nan_to_num(yo - yp)])
        e, lo, hi, se, p, bt = paired_diff(cl, Y, R_ & ok, W_ & ok)
        q = np.nanquantile(np.abs(bt[:, 0]) - np.abs(bt[:, 1]), [0.025, 0.975])
        extra[f"regret_DiD_{tag}"] = dict(observer=[e[0], lo[0], hi[0]], para=[e[1], lo[1], hi[1]], observer_minus_para=[e[2], lo[2], hi[2]],
                                          absobs_minus_abspara=[abs(e[0]) - abs(e[1]), q[0], q[1]], n_right=int((R_ & ok).sum()),
                                          n_wrong=int((W_ & ok).sum()), n_clusters=cl.nc)
    # ---------------------------------------------------------------- concept screen (DiD right - wrong of participant - reader deltas)
    Ct = []
    for tag, col in (("t", "i_t"), ("t1", "i_t1")):
        idx = E[col].to_numpy()
        m1 = idx >= 0
        for s in SITES:
            res = {}
            for nm in ("obsmin", "para"):
                Yc = np.full((len(E), len(concepts)), np.nan)
                Yc[m1] = deltas[nm][s][idx[m1]]
                ok = np.isfinite(Yc).all(1)
                res[nm] = paired_diff(cl, np.nan_to_num(Yc), R_ & ok, W_ & ok)
            eo, loo, hio, seo, po, bo = res["obsmin"]
            ep, lop, hip, sep_, pp, bp = res["para"]
            T = pd.DataFrame(dict(concept=concepts, time=tag, site=s, did=eo, lo=loo, hi=hio, p=po, para_did=ep, para_lo=lop, para_hi=hip))
            vm = T.concept.isin(valid).to_numpy()
            T["q"] = np.nan
            T.loc[vm, "q"] = bh_fdr(T.loc[vm, "p"].to_numpy())
            T["para_p95"] = float(np.nanquantile(np.abs(ep[vm]), 0.95))
            T["sig_fdr"] = T.q < 0.05
            T["exceeds_para"] = T.did.abs() > T.para_p95
            T["prereg"] = T.concept.isin(PREREG)
            T["validated"] = vm
            Ct.append(T)
    Ct = pd.concat(Ct)
    # random-direction null at layer 16 for the concept screen (t+1 and t, decision + reading), projected only on episode rows
    rnd = {}
    if a.n_random:
        cm = json.load(open(ACTS / M_ / "concepts" / "directions_meta.json"))
        U = np.load(ACTS / M_ / "concepts" / "directions.npy")
        gdir = U.mean(0)
        gdir /= np.linalg.norm(gdir)
        rng = np.random.default_rng(SEED)
        Rd = rng.standard_normal((a.n_random, U.shape[1])).astype(np.float32)
        Rd /= np.linalg.norm(Rd, axis=1, keepdims=True)
        Rg = np.vstack([Rd, gdir[None]]).astype(np.float32)
        rep = ACTS / M_ / args.replay
        obs = ACTS / M_ / readers["obsmin"]["run"]
        RA, OA = np.load(rep / "acts.npy", mmap_mode="r"), np.load(obs / "acts.npy", mmap_mode="r")
        oidx = pd.DataFrame(read_jsonl(obs / "index.jsonl")).drop(columns=["lp_letters"])
        orow_all = base[KEY].merge(oidx[KEY + ["row"]], on=KEY, how="left").row.to_numpy(float)
        solo = np.sort(ridx[ridx.turn == 0].row.to_numpy())
        gi_r = json.load(open(rep / "meta.json"))["grid"].index(cm["layer"])
        gi_o = json.load(open(obs / "meta.json"))["grid"].index(cm["layer"])
        # GENERAL residualisation slope for random directions: fit on a hashed sample of 3000 agent-turns (all release_600)
        samp = np.sort(np.random.default_rng(SEED + 1).choice(np.flatnonzero(np.isfinite(prow) & np.isfinite(orow_all)), 3000, replace=False))
        for si, s in enumerate(SITES):
            XR = RA[solo, si, gi_r].astype(np.float32)
            mu, sd = XR.mean(0), XR.std(0) + 1e-3
            SR = ((XR - mu) / sd) @ Rg.T
            smu, ssd = SR.mean(0), SR.std(0) + 1e-6

            def proj(A, rows_, gi):
                rows_ = np.asarray(rows_, int)
                o = np.argsort(rows_)
                Z = np.empty((len(rows_), Rg.shape[0]), np.float32)
                Z[o] = (((A[rows_[o], si, gi].astype(np.float32) - mu) / sd) @ Rg.T - smu) / ssd
                return Z

            Ds = proj(RA, prow[samp], gi_r) - proj(OA, orow_all[samp], gi_o)
            X = np.column_stack([np.ones(len(samp)), Ds[:, -1]])
            beta, *_ = np.linalg.lstsq(X, Ds[:, :-1], rcond=None)
            for tag, col in (("t", "i_t"), ("t1", "i_t1")):
                idx = E[col].to_numpy()
                m1 = (idx >= 0)
                m1 &= np.isfinite(np.where(m1, prow[np.maximum(idx, 0)], np.nan)) & np.isfinite(np.where(m1, orow_all[np.maximum(idx, 0)], np.nan))
                Yr = np.full((len(E), a.n_random), np.nan)
                ii = idx[m1]
                Dr = proj(RA, prow[ii], gi_r) - proj(OA, orow_all[ii], gi_o)
                Yr[m1] = Dr[:, :-1] - Dr[:, -1:] * beta[1][None]
                ok = np.isfinite(Yr).all(1)
                er, lr, hr, ser, prr, _ = paired_diff(cl, np.nan_to_num(Yr), R_ & ok, W_ & ok)
                qr = bh_fdr(prr)
                rnd[f"{tag}_{s}"] = dict(abs_did_p95=float(np.nanquantile(np.abs(er), 0.95)), n_sig_fdr=int((qr < 0.05).sum()),
                                         n_random=a.n_random)
        Ct["random_abs_p95"] = [rnd.get(f"{t}_{s}", {}).get("abs_did_p95", np.nan) for t, s in zip(Ct.time, Ct.site)]
        Ct["exceeds_random"] = Ct.did.abs() > Ct.random_abs_p95
    # ---------------------------------------------------------------- prediction of later reversal from the t+1 signal
    P = E[has1].copy().reset_index(drop=True)
    P["reverts"] = P.reverts_later.astype(float)
    P = P[P.reverts.notna()].reset_index(drop=True)
    P["regret_obs"] = P.dmX_obsmin_t1
    P["pull_participant"] = P.mX_participant_t1
    P["neg_conf_t1"] = -P.conf_t1
    P["probe_d"] = -P.dprobe_obsmin_t1
    y = P.reverts.astype(bool).to_numpy()
    groups = P.cluster.to_numpy()
    P["tfidf"] = oof_text_model(P.text_t1.tolist(), y, groups, kind="tfidf") if y.sum() >= 10 else np.nan
    Xh = np.column_stack([P.conf_t1.fillna(P.conf_t1.median()), P.confidence.fillna(P.confidence.median()), P.n_changes_before, P.turn,
                          P.right.astype(float)])
    P["public_hist"] = oof_text_model(None, y, groups, kind="hist", X=Xh) if y.sum() >= 10 else np.nan
    preds = ["regret_obs", "pull_participant", "neg_conf_t1", "probe_d", "tfidf", "public_hist"]
    pred_ok = [p_ for p_ in preds if P[p_].notna().any()]
    A_, Ad = auroc_table(P, pred_ok, "reverts", "cluster",
                         ref_pairs=[("regret_obs", "tfidf"), ("regret_obs", "public_hist"), ("pull_participant", "public_hist"),
                                    ("pull_participant", "tfidf")])
    # ---------------------------------------------------------------- outputs
    tag = f"{M_}_{ds}_T{T_}"
    E.drop(columns=["text_t1"]).to_csv(TABLES / f"r1_episodes_{tag}.csv", index=False)
    S.to_csv(TABLES / f"r1_contrasts_{tag}.csv", index=False)
    Ct.to_csv(TABLES / f"r1_concepts_{tag}.csv", index=False)
    A_.to_csv(TABLES / f"r1_auroc_reversal_{tag}.csv", index=False)
    Ad.to_csv(TABLES / f"r1_auroc_reversal_diffs_{tag}.csv", index=False)
    summ = dict(tag=tag, split_role="discovery", n_right=int(E.right.sum()), n_wrong=int((~E.right).sum()),
                n_right_t1=int((E.right & has1).sum()), n_wrong_t1=int((~E.right & has1).sum()), n_clusters=cl.nc,
                families_right=E[E.right].family.value_counts().to_dict(), families_wrong=E[~E.right].family.value_counts().to_dict(),
                primary=extra, random_null=rnd, n_reversal_episodes=len(P), n_reversals=int(y.sum()),
                concepts_sig=Ct[Ct.validated & Ct.sig_fdr].assign(a=lambda x: x.did.abs()).sort_values("a", ascending=False)
                [["concept", "time", "site", "did", "lo", "hi", "q", "exceeds_para", "exceeds_random"]].head(30).to_dict("records"),
                prereg_t1_dec=Ct[(Ct.time == "t1") & (Ct.site == "dec") & Ct.prereg][["concept", "did", "lo", "hi", "q"]].to_dict("records"),
                core_power=core_power(M_))
    json.dump(summ, open(TABLES / f"r1_summary_{tag}.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 250)
    print(S.round(3).to_string())
    print(json.dumps({k: summ[k] for k in ("n_right", "n_wrong", "n_right_t1", "n_wrong_t1", "n_clusters", "primary", "random_null",
                                           "n_reversal_episodes", "n_reversals", "core_power")}, indent=1, default=float))
    print(pd.DataFrame(summ["concepts_sig"]).round(3).to_string())
    print(A_.round(3).to_string())
    print(Ad.round(3).to_string())


if __name__ == "__main__":
    main()
