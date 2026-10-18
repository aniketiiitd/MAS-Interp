"""V3-C3 "Who is reading": participant vs observer on the byte-identical transcript, plus scenario contrasts (Q9 debates).

Readers (same task, same own previous message, same peer messages; interp/observer_replay.py):
  participant  the real debate prompt (replay)                          primary comparison
  obsmin       minimal observer edit (--observer, primary control)       delta_c = participant z - observer z
  para         participant role, meaning-preserving paraphrase (--null)  wording null delta^para_c = participant z - para z
  stakes       superseded v1 observer (--extra, robustness only)
Concept z-scores come from emotion_vectors.py (participant turn-0 normalisation for every reader, layer 16).

Roles from PUBLIC verdicts at t-1: outnumbered (both peers share an answer M != own X), majority (agent agrees with exactly one
peer; M = the dissenter's answer), unanimous, split. Moves of outnumbered agents at t: gives_in (verdict M) / holds_out (verdict X).

(i)  Behavioural role effect on outnumbered episodes: B = P(participant argmax = M) - P(observer argmax = M), both from the
     prefilled-VERDICT letter distributions of forward passes on the same text; margin scale B_m = mean[(lp_M - lp_X)_P - (..)_O].
     B_para the same with the paraphrase reader. Paired cluster bootstrap (one resample of item clusters for all arms), so
     |B| - |B_para| has a proper CI. Rate scale is primary (first definition in the frozen gate).
(ii) Role fingerprint DiD_c = mean_outnumbered(delta_c) - mean_majority(delta_c); primary estimand residualises every concept delta
     on the GENERAL-axis delta (slope fit on the discovery half, applied everywhere); sensitivity variants: raw; GENERAL +
     nuisance (participant prompt length, turn, agent slot); GENERAL + decision (participant-minus-reader verdict margin toward M). Discovery half: all validated concepts, normal-approximation bootstrap
     p (continuous, so BH-FDR is not limited by bootstrap resolution), BH-FDR; confirmation half: same sign + 95% CI excluding 0.
     Nulls: paraphrase DiD (95th percentile of |DiD^para| over validated concepts), 200 random unit directions at layer 16 run
     through the identical pipeline (count of replicating directions, and the 95th percentile of |full-sample DiD| over random
     directions as a concept-specificity bar), and a role-permutation null that relabels the outnumbered agent uniformly among
     the three agents of each 2-1 item-turn (identity included, so the permuted statistic has mean 0).
Older scenario contrasts kept on the same machinery: B gives_in vs holds_out, C the same at t-1, D hidden dissent (private switch
after a unanimous final turn; missing private verdicts dropped and counted), E domain split, F minority whose own t-1 answer was right vs wrong.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import erfc

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import *  # noqa
from stats import bh_fdr

TOPK = 12
PREREG = ["dissent", "conflicted", "moral_uncertainty", "deferential", "trust_in_source", "belief_updating",
          "conformity_to_majority", "stubbornness"]
SITES = ("dec", "read")
SEED = 240091
KEY = ["item_id", "agent", "turn"]


# ----------------------------------------------------------------------------------------------------------------- bootstrap
class Clusters:
    """Shared cluster index + multinomial bootstrap weights: every arm and contrast of one sample is resampled identically."""

    def __init__(self, cluster_of_row, B, seed=SEED):
        self.codes, self.names = pd.factorize(pd.Series(cluster_of_row))
        self.nc = len(self.names)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, self.nc, (B, self.nc))
        self.W = np.zeros((B, self.nc), np.float64)
        for b in range(B):
            self.W[b] = np.bincount(idx[b], minlength=self.nc)

    def sums(self, Y, mask):
        S = np.zeros((self.nc, Y.shape[1]), np.float64)
        np.add.at(S, self.codes[mask], Y[mask])
        C = np.bincount(self.codes[mask], minlength=self.nc).astype(np.float64)
        return S, C


def paired_diff(cl, Y, ma, mb):
    """mean_a - mean_b per column, paired cluster bootstrap. Returns est, lo, hi, se, p (normal approx), boot (B x k)."""
    SA, CA = cl.sums(Y, ma)
    SB, CB = cl.sums(Y, mb)
    k = Y.shape[1]
    if CA.sum() == 0 or CB.sum() == 0:
        nan = np.full(k, np.nan)
        return nan, nan, nan, nan, nan, np.full((cl.W.shape[0], k), np.nan)
    est = SA.sum(0) / CA.sum() - SB.sum(0) / CB.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = (cl.W @ SA) / (cl.W @ CA)[:, None] - (cl.W @ SB) / (cl.W @ CB)[:, None]
    lo, hi = np.nanquantile(boot, [0.025, 0.975], axis=0)
    se = np.nanstd(boot, axis=0)
    p = erfc(np.abs(est / np.maximum(se, 1e-12)) / np.sqrt(2))
    return est, lo, hi, se, p, boot


def paired_mean(cl, Y, m):
    S, C = cl.sums(Y, m)
    est = S.sum(0) / C.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = (cl.W @ S) / (cl.W @ C)[:, None]
    lo, hi = np.nanquantile(boot, [0.025, 0.975], axis=0)
    return est, lo, hi, boot


# ----------------------------------------------------------------------------------------------------------------- data
def cluster_of(core, i):
    return core[i]["meta"].get("analysis_group") if core[i]["stratum"] == "persuasion" else core[i]["source_item_id"]


def roles_from_verdicts(P):
    V_ = P.pivot_table(index=["item_id", "turn"], columns="agent", values="verdict", aggfunc="first").reindex(columns=[0, 1, 2])
    vals = {k: r for k, r in zip(V_.index, V_.to_numpy())}
    rows = []
    for (iid, t), now in vals.items():
        prev = vals.get((iid, t - 1))
        if prev is None or any(not isinstance(x, str) for x in prev):
            continue
        for a in range(3):
            others = [prev[b] for b in range(3) if b != a]
            M = move = None
            X = prev[a]
            if others[0] == others[1] and others[0] != prev[a]:
                role, M = "outnumbered", others[0]
                move = "gives_in" if now[a] == M else ("holds_out" if now[a] == X else "other")
            elif (others[0] == prev[a]) != (others[1] == prev[a]):
                role = "majority"
                M = others[0] if others[0] != prev[a] else others[1]
            elif others[0] == others[1] == prev[a]:
                role = "unanimous"
            else:
                role = "split"
            rows.append(dict(item_id=iid, turn=t, agent=a, role=role, move=move, M=M, X=X))
    return pd.DataFrame(rows)


def build(a):
    core = {r["id"]: r for r in load_core()}
    V = pd.read_csv(TABLES / f"concept_validation_{a.model}.csv")
    valid = list(V[V.passes].concept)
    concepts = valid + [c for c in PREREG if c not in valid]  # conflicted is pre-registered but not story-validated
    zc = [f"{s}_{c}" for s in SITES for c in concepts + ["GENERAL"]]
    P = pd.read_parquet(TABLES / f"concept_traj_{a.model}_{a.replay}.parquet")
    P = P[P.cond.isin(["t0", "debate"])].copy()
    if a.exclude_split:
        drop = set(json.load(open(ROOT / "data/splits/splits.json"))[a.exclude_split])
        P = P[~P.item_id.isin(drop)]
    pidx = pd.DataFrame(read_jsonl(ACTS / a.model / a.replay / "index.jsonl"))
    pidx = pidx[pidx.cond.isin(["t0", "debate"])][KEY + ["lp_letters"]]
    P = P.merge(pidx, on=KEY, how="left")
    S = roles_from_verdicts(P)
    base = P[P.turn >= 1][KEY + ["verdict", "correct", "confidence", "n_tok", "lp_letters"] + zc].merge(S, on=KEY, how="left")
    base["cluster"] = base.item_id.map(lambda i: cluster_of(core, i))
    base["half"] = base.cluster.map(lambda u: "discovery" if h64("disc240091", u) % 2 == 0 else "confirmation")
    base["domain"] = base.item_id.map(lambda i: core[i]["primary_domain"])
    base["family"] = base.item_id.map(lambda i: family_of(core[i]))
    corr = {(i, ag, t): c for i, ag, t, c in zip(P.item_id, P.agent, P.turn, P.correct)}
    base["prev_correct"] = [corr.get((i, ag, t - 1)) for i, ag, t in zip(base.item_id, base.agent, base.turn)]
    base = base.sort_values(KEY).reset_index(drop=True)
    readers = {}
    for name, run in [("obsmin", a.observer), ("para", a.null)] + [tuple(r.split(":")) for r in a.extra]:
        f = TABLES / f"concept_obs_{a.model}_{run}.parquet"
        if not run or not f.exists():
            print(f"[warn] reader {name}: {f.name} missing", flush=True)
            continue
        O = pd.read_parquet(f)
        oi = pd.DataFrame(read_jsonl(ACTS / a.model / run / "index.jsonl"))[KEY + ["lp_letters"]]
        readers[name] = dict(run=run, df=O[KEY + zc].merge(oi, on=KEY, how="left"))
    return base, concepts, valid, readers


def reader_delta(base, O, concepts):
    """participant z - reader z per (site, concept) aligned to base rows (NaN where the reader row is missing) + aligned reader rows."""
    cols = [f"{s}_{c}" for s in SITES for c in concepts + ["GENERAL"]]
    M = base[KEY].merge(O[KEY + cols + ["lp_letters"]], on=KEY, how="left")
    D = base[cols].to_numpy(np.float64) - M[cols].to_numpy(np.float64)
    return pd.DataFrame(D, columns=cols), M


def residualise(D, base, concepts, variant, fit_mask, dmargin=None):
    """{site: n x k} concept deltas after the adjustment (fit on fit_mask rows, applied to all rows).
    variant 'decision' adds the participant-minus-reader verdict margin toward M (lp_M - lp_X) as a covariate, so a concept
    contrast cannot be carried by the upcoming verdict itself (rows without M get NaN and drop out of the role contrasts)."""
    out = {}
    for s in SITES:
        Y = D[[f"{s}_{c}" for c in concepts]].to_numpy(np.float64)
        if variant == "raw":
            out[s] = Y
            continue
        Xs = [D[f"{s}_GENERAL"].to_numpy(np.float64)]
        if variant == "nuisance":
            nt = base.n_tok.to_numpy(np.float64)
            Xs.append((nt - nt[fit_mask].mean()) / (nt[fit_mask].std() + 1e-9))
            Xs += [(base.turn == t).to_numpy(np.float64) for t in sorted(base.turn.unique())[1:]]
            Xs += [(base.agent == ag).to_numpy(np.float64) for ag in (1, 2)]
        if variant == "decision":
            Xs.append(np.asarray(dmargin, np.float64))
        X = np.column_stack([np.ones(len(Y))] + Xs)
        ok = fit_mask & np.isfinite(Y).all(1) & np.isfinite(X).all(1)
        beta, *_ = np.linalg.lstsq(X[ok], Y[ok], rcond=None)
        out[s] = Y - X[:, 1:] @ beta[1:]
    return out


def split_by_half(base, arr):
    return {h: arr[(base.half == h).to_numpy()] for h in ("discovery", "confirmation")}


def screen(Yh, clh, masks_h, names, sel_pool):
    """Discovery (BH-FDR over sel_pool) -> confirmation for one site."""
    res = {}
    for h in ("discovery", "confirmation"):
        ma, mb = masks_h[h]
        ok = np.isfinite(Yh[h]).all(1)
        res[h] = paired_diff(clh[h], np.nan_to_num(Yh[h]), ma & ok, mb & ok)
    ed, ld, hd, _, pd_, _ = res["discovery"]
    ec, lc, hc, _, pc, _ = res["confirmation"]
    T = pd.DataFrame(dict(concept=names, disc_est=ed, disc_lo=ld, disc_hi=hd, disc_p=pd_, conf_est=ec, conf_lo=lc, conf_hi=hc, conf_p=pc))
    pool = T.concept.isin(sel_pool).to_numpy()
    T["disc_q"] = np.nan
    T.loc[pool, "disc_q"] = bh_fdr(T.loc[pool, "disc_p"].to_numpy())
    T["disc_sig"] = T.disc_q < 0.05
    T["conf_same_sign_excl0"] = (np.sign(T.conf_est) == np.sign(T.disc_est)) & ((T.conf_lo > 0) | (T.conf_hi < 0))
    T["replicates_uncapped"] = T.disc_sig & T.conf_same_sign_excl0
    top = T[T.disc_sig].assign(_a=lambda x: x.disc_est.abs()).sort_values("_a", ascending=False).head(TOPK).concept
    T["selected_top"] = T.concept.isin(top)
    T["replicates"] = T.selected_top & T.conf_same_sign_excl0
    return T


def excl0(r):
    return bool(r is not None and (r["lower"] > 0 or r["upper"] < 0))


# ----------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="release_600")
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--replay", default=None)
    ap.add_argument("--observer", default=None, help="primary observer reader run (default reader_obsmin_<ds>_T<T>)")
    ap.add_argument("--null", default=None, help="participant paraphrase reader run (default reader_para_<ds>_T<T>)")
    ap.add_argument("--extra", default="stakes:observer_release_600_T12", help="name:run,... robustness readers (release_600 only)")
    ap.add_argument("--exclude_split", default=None)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--n_random", type=int, default=200)
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    ds, T_ = a.dataset, a.turns
    a.replay = a.replay or f"replay_{ds}_{'all' if ds == 'core' else ds}_T{T_}"
    a.observer = a.observer or f"reader_obsmin_{ds}_T{T_}"
    a.null = a.null if a.null is not None else f"reader_para_{ds}_T{T_}"
    a.extra = [x for x in a.extra.split(",") if x] if ds == "release_600" else []
    tag = f"{a.model}_{ds}_T{T_}{a.tag}"
    t_start = time.time()
    base, concepts, valid, readers = build(a)
    assert "obsmin" in readers, "primary observer reader missing"
    n = len(base)
    print(f"[stage] {n} participant agent-turns; readers {list(readers)}; {len(valid)} validated concepts", flush=True)
    base[KEY + ["role", "move", "M", "X", "half", "domain", "family"]].to_csv(TABLES / f"inner_roles_{tag}.csv", index=False)
    summary = dict(tag=tag, readers={k: v["run"] for k, v in readers.items()}, n_agent_turns=n,
                   role_counts=base.role.value_counts().to_dict(), move_counts=base.move.value_counts().to_dict())
    halves = {h: (base.half == h).to_numpy() for h in ("discovery", "confirmation")}
    cl_all = Clusters(base.cluster.to_numpy(), a.B)
    cl_h = {h: Clusters(base.cluster.to_numpy()[m], a.B) for h, m in halves.items()}
    summary["clusters"] = dict(all=cl_all.nc, discovery=cl_h["discovery"].nc, confirmation=cl_h["confirmation"].nc)
    out_mask = (base.role == "outnumbered").to_numpy()
    maj_mask = (base.role == "majority").to_numpy()
    deltas, aligned = {}, {}
    for nm, r in readers.items():
        deltas[nm], aligned[nm] = reader_delta(base, r["df"], concepts)
    summary["missing_reader_rows"] = {nm: int(np.isnan(D_["dec_GENERAL"]).sum()) for nm, D_ in deltas.items()}

    # ------------------------------------------------------------------ (i) behavioural role effect
    def lp_stats(lp_col):
        adopt, marg = np.full(n, np.nan), np.full(n, np.nan)
        for i, (lp, M, X) in enumerate(zip(lp_col, base.M, base.X)):
            if isinstance(lp, dict) and M in lp and X in lp:
                adopt[i] = float(max(lp, key=lp.get) == M)
                marg[i] = lp[M] - lp[X]
        return adopt, marg

    stats = {"participant": lp_stats(base.lp_letters)}
    for nm in readers:
        stats[nm] = lp_stats(aligned[nm].lp_letters)
    sampled = np.where(base.M.notna(), (base.verdict == base.M).astype(float), np.nan)
    rnames = list(stats)
    beh_rows, boots = [], {}
    for role, rmask in (("outnumbered", out_mask), ("majority", maj_mask)):
        for scope, dmask in [("ALL", np.ones(n, bool))] + [(d, (base.domain == d).to_numpy()) for d in DOMAIN_LABEL]:
            common = rmask & dmask & np.all([np.isfinite(stats[r][0]) for r in rnames], axis=0)
            if common.sum() < 10:
                continue
            names = [f"adopt_{r}" for r in rnames] + [f"margin_{r}" for r in rnames] + ["adopt_sampled_participant"]
            Y = np.column_stack([stats[r][0] for r in rnames] + [stats[r][1] for r in rnames] + [sampled])
            est, lo, hi, boot = paired_mean(cl_all, np.nan_to_num(Y), common)
            col = {nm: j for j, nm in enumerate(names)}
            for nm, j in col.items():
                beh_rows.append(dict(role=role, scope=scope, quantity=nm, estimate=est[j], lower=lo[j], upper=hi[j], n=int(common.sum())))
            for r in rnames[1:]:
                for scale in ("adopt", "margin"):
                    d = boot[:, col[f"{scale}_participant"]] - boot[:, col[f"{scale}_{r}"]]
                    e = est[col[f"{scale}_participant"]] - est[col[f"{scale}_{r}"]]
                    q = np.nanquantile(d, [0.025, 0.975])
                    beh_rows.append(dict(role=role, scope=scope, quantity=f"B_{scale}_participant_minus_{r}", estimate=e,
                                         lower=q[0], upper=q[1], n=int(common.sum())))
                    boots[(role, scope, scale, r)] = (e, d)
            if "obsmin" in rnames and "para" in rnames:
                for scale in ("adopt", "margin"):
                    eB, dB = boots[(role, scope, scale, "obsmin")]
                    eP, dP = boots[(role, scope, scale, "para")]
                    q = np.nanquantile(np.abs(dB) - np.abs(dP), [0.025, 0.975])
                    beh_rows.append(dict(role=role, scope=scope, quantity=f"absB_minus_absBpara_{scale}", estimate=abs(eB) - abs(eP),
                                         lower=q[0], upper=q[1], n=int(common.sum())))
    Beh = pd.DataFrame(beh_rows)
    Beh.to_csv(TABLES / f"inner_behaviour_{tag}.csv", index=False)
    g = {(r.role, r.scope, r.quantity): r._asdict() for r in Beh.itertuples()}
    get = lambda q: g.get(("outnumbered", "ALL", q))  # noqa: E731
    rB, rd = get("B_adopt_participant_minus_obsmin"), get("absB_minus_absBpara_adopt")
    summary["gate_i_this_dataset"] = dict(
        B_adopt=rB and [rB["estimate"], rB["lower"], rB["upper"]],
        B_margin=get("B_margin_participant_minus_obsmin") and [get("B_margin_participant_minus_obsmin")[k] for k in ("estimate", "lower", "upper")],
        B_para_adopt=get("B_adopt_participant_minus_para") and [get("B_adopt_participant_minus_para")[k] for k in ("estimate", "lower", "upper")],
        absB_minus_absBpara_adopt=rd and [rd["estimate"], rd["lower"], rd["upper"]],
        absB_minus_absBpara_margin=get("absB_minus_absBpara_margin") and [get("absB_minus_absBpara_margin")[k] for k in ("estimate", "lower", "upper")],
        B_ci_excl0=excl0(rB), exceeds_para=bool(rd is not None and rd["lower"] > 0),
        PASS_this_dataset=bool(excl0(rB) and rd is not None and rd["lower"] > 0))
    print(f"[stage] behaviour done ({time.time()-t_start:.0f}s)", flush=True)

    # ------------------------------------------------------------------ (ii) role fingerprint
    fit_mask = halves["discovery"]
    variants = ("general", "raw", "nuisance", "decision")
    dm_o = stats["participant"][1] - stats["obsmin"][1]
    Y = {v: residualise(deltas["obsmin"], base, concepts, v, fit_mask, dm_o) for v in variants}
    Yp = ({v: residualise(deltas["para"], base, concepts, v, fit_mask, stats["participant"][1] - stats["para"][1]) for v in variants}
          if "para" in deltas else None)
    Yx = {nm: residualise(deltas[nm], base, concepts, "general", fit_mask) for nm in deltas if nm not in ("obsmin", "para")}
    masks = {h: (out_mask[m], maj_mask[m]) for h, m in halves.items()}
    grp_of = pd.read_csv(TABLES / f"concept_validation_{a.model}.csv").set_index("concept").group
    fp = {}
    for v in variants:
        for s in SITES:
            Tm = screen(split_by_half(base, Y[v][s]), cl_h, masks, concepts, valid)
            ok = np.isfinite(Y[v][s]).all(1)
            e, lo, hi, _, _, bootm = paired_diff(cl_all, np.nan_to_num(Y[v][s]), out_mask & ok, maj_mask & ok)
            Tm["full_est"], Tm["full_lo"], Tm["full_hi"] = e, lo, hi
            if Yp is not None:
                Tp = screen(split_by_half(base, Yp[v][s]), cl_h, masks, concepts, valid)
                okp = np.isfinite(Yp[v][s]).all(1)
                ep, lop, hip, _, _, bootp = paired_diff(cl_all, np.nan_to_num(Yp[v][s]), out_mask & okp, maj_mask & okp)
                Tm["para_conf_est"], Tm["para_conf_lo"], Tm["para_conf_hi"] = Tp.conf_est, Tp.conf_lo, Tp.conf_hi
                Tm["para_full_est"], Tm["para_full_lo"], Tm["para_full_hi"] = ep, lop, hip
                vm = Tm.concept.isin(valid).to_numpy()
                p95_conf = float(np.nanquantile(np.abs(Tp.conf_est.to_numpy()[vm]), 0.95))
                p95_full = float(np.nanquantile(np.abs(ep[vm]), 0.95))
                Tm["para_null_p95_conf"], Tm["para_null_p95_full"] = p95_conf, p95_full
                Tm["exceeds_para_null"] = Tm.conf_est.abs() > p95_conf
                q = np.nanquantile(np.abs(bootm) - np.abs(bootp), [0.025, 0.975], axis=0)
                Tm["absDiD_minus_absDiDpara_lo"], Tm["absDiD_minus_absDiDpara_hi"] = q[0], q[1]
            for nm, Yx_ in Yx.items():
                okx = np.isfinite(Yx_[s]).all(1)
                ex, lox, hix, *_ = paired_diff(cl_all, np.nan_to_num(Yx_[s]), out_mask & okx, maj_mask & okx)
                Tm[f"{nm}_full_est"], Tm[f"{nm}_full_lo"], Tm[f"{nm}_full_hi"] = ex, lox, hix
            Tm["variant"], Tm["site"] = v, s
            Tm["validated"], Tm["prereg"], Tm["group"] = Tm.concept.isin(valid), Tm.concept.isin(PREREG), Tm.concept.map(grp_of)
            fp[(v, s)] = Tm
    pd.concat(fp.values()).to_csv(TABLES / f"inner_fingerprint_{tag}.csv", index=False)
    print(f"[stage] fingerprint done ({time.time()-t_start:.0f}s)", flush=True)

    # ------------------------------------------------------------------ nulls
    rng = np.random.default_rng(SEED)
    nulls = {}
    if a.n_random:
        cm = json.load(open(ACTS / a.model / "concepts" / "directions_meta.json"))
        U = np.load(ACTS / a.model / "concepts" / "directions.npy")
        gdir = U.mean(0)
        gdir /= np.linalg.norm(gdir)
        R = rng.standard_normal((a.n_random, U.shape[1])).astype(np.float32)
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        Rg = np.vstack([R, gdir[None]]).astype(np.float32)
        rep, obs = ACTS / a.model / a.replay, ACTS / a.model / readers["obsmin"]["run"]
        ridx = pd.DataFrame(read_jsonl(rep / "index.jsonl")).drop(columns=["lp_letters"])
        oidx = pd.DataFrame(read_jsonl(obs / "index.jsonl")).drop(columns=["lp_letters"])
        prow = base[KEY].merge(ridx[ridx.cond == "debate"][KEY + ["row"]], on=KEY, how="left").row.to_numpy(float)
        orow = base[KEY].merge(oidx[KEY + ["row"]], on=KEY, how="left").row.to_numpy(float)
        okrow = np.isfinite(prow) & np.isfinite(orow)
        prow, orow = np.where(okrow, prow, 0).astype(np.int64), np.where(okrow, orow, 0).astype(np.int64)
        solo = np.sort(ridx[ridx.turn == 0].row.to_numpy())
        RA, OA = np.load(rep / "acts.npy", mmap_mode="r"), np.load(obs / "acts.npy", mmap_mode="r")
        gi_r = json.load(open(rep / "meta.json"))["grid"].index(cm["layer"])
        gi_o = json.load(open(obs / "meta.json"))["grid"].index(cm["layer"])

        def project(A, rows, si, gi, mu, sd, smu, ssd):
            Z = np.empty((n, Rg.shape[0]), np.float32)
            order = np.argsort(rows, kind="stable")
            for c0 in range(0, n, 3000):
                sl = order[c0:c0 + 3000]
                Z[sl] = (((A[rows[sl], si, gi].astype(np.float32) - mu) / sd) @ Rg.T - smu) / ssd
            return Z

        rc = {}
        for si, s in enumerate(SITES):
            XR = RA[solo, si, gi_r].astype(np.float32)
            mu, sd = XR.mean(0), XR.std(0) + 1e-3
            SR = ((XR - mu) / sd) @ Rg.T
            smu, ssd = SR.mean(0), SR.std(0) + 1e-6
            del XR
            Dr = (project(RA, prow, si, gi_r, mu, sd, smu, ssd) - project(OA, orow, si, gi_o, mu, sd, smu, ssd)).astype(np.float64)
            Dr[~okrow] = np.nan
            X = np.column_stack([np.ones(n), Dr[:, -1]])
            fm = fit_mask & okrow
            beta, *_ = np.linalg.lstsq(X[fm], Dr[fm, :-1], rcond=None)
            Yr = Dr[:, :-1] - X[:, 1:] @ beta[1:]
            names = [f"rand{j}" for j in range(a.n_random)]
            Tr = screen(split_by_half(base, Yr), cl_h, masks, names, names)
            Tr.to_csv(TABLES / f"inner_randomdirs_{s}_{tag}.csv", index=False)
            flags = Tr.replicates_uncapped.to_numpy()
            k = len([c for c in valid])
            draws = np.array([flags[rng.choice(len(flags), size=k, replace=len(flags) < k)].sum() for _ in range(5000)])
            okr = np.isfinite(Yr).all(1)
            er = paired_diff(cl_all, np.nan_to_num(Yr), out_mask & okr, maj_mask & okr)[0]
            p95r = float(np.nanquantile(np.abs(er), 0.95))
            fp[("general", s)]["random_full_abs_p95"] = p95r
            fp[("general", s)]["exceeds_random_full"] = fp[("general", s)].full_est.abs() > p95r
            obs_rep = int(fp[("general", s)].loc[lambda x: x.validated, "replicates_uncapped"].sum())
            rc[s] = dict(random_disc_sig=int(Tr.disc_sig.sum()), random_replicating=int(flags.sum()), n_random=a.n_random,
                         random_full_abs_p95=p95r, validated_exceeding_random_full=int(fp[("general", s)].loc[lambda x: x.validated, "exceeds_random_full"].sum()),
                         null_count_p95=float(np.quantile(draws, 0.95)), observed_replicating_validated=obs_rep,
                         observed_disc_sig_validated=int(fp[("general", s)].loc[lambda x: x.validated, "disc_sig"].sum()),
                         exceeds=bool(obs_rep > np.quantile(draws, 0.95)))
        nulls["random_directions"] = rc
        print(f"[stage] random-direction null done ({time.time()-t_start:.0f}s)", flush=True)
    if a.n_perm:
        sub = base[base.role.isin(["outnumbered", "majority"])]
        tri = [g_.sort_values("agent") for _, g_ in sub.groupby(["item_id", "turn"]) if len(g_) == 3 and (g_.role == "outnumbered").sum() == 1]
        tri_rows = np.array([g_.index.to_numpy() for g_ in tri])
        true_out = np.array([int(np.where(g_.role.to_numpy() == "outnumbered")[0][0]) for g_ in tri])
        B_small = 400
        clh_s = {h: Clusters(base.cluster.to_numpy()[m], B_small, seed=SEED + 7) for h, m in halves.items()}
        cla_s = Clusters(base.cluster.to_numpy(), B_small, seed=SEED + 7)

        def tri_masks(po):
            mo, mm = np.zeros(n, bool), np.zeros(n, bool)
            mo[tri_rows[np.arange(len(tri_rows)), po]] = True
            for j in range(3):
                mm[tri_rows[po != j, j]] = True
            return mo, mm

        pc = {s: [] for s in SITES}
        pf = {s: [] for s in SITES}
        for _ in range(a.n_perm):
            mo, mm = tri_masks((true_out + rng.integers(0, 3, len(tri_rows))) % 3)  # uniform relabelling (identity included): E[stat]=0 under H0
            pm = {h: (mo[m], mm[m]) for h, m in halves.items()}
            for s in SITES:
                Tq = screen(split_by_half(base, Y["general"][s]), clh_s, pm, concepts, valid)
                pc[s].append(int(Tq[Tq.concept.isin(valid)].replicates_uncapped.sum()))
                ok = np.isfinite(Y["general"][s]).all(1)
                pf[s].append(paired_diff(cla_s, np.nan_to_num(Y["general"][s]), mo & ok, mm & ok)[0])
        mo, mm = tri_masks(true_out)
        pinfo = {}
        for s in SITES:
            ok = np.isfinite(Y["general"][s]).all(1)
            e_true = paired_diff(cla_s, np.nan_to_num(Y["general"][s]), mo & ok, mm & ok)[0]
            PF = np.array(pf[s])
            fp[("general", s)]["triples_est"] = e_true
            fp[("general", s)]["perm_p_triples"] = ((np.abs(PF) >= np.abs(e_true)[None]).sum(0) + 1) / (len(PF) + 1)
            obs_rep = int(fp[("general", s)].loc[lambda x: x.validated, "replicates_uncapped"].sum())
            pinfo[s] = dict(n_triples=len(tri_rows), perm_count_mean=float(np.mean(pc[s])), perm_count_p95=float(np.quantile(pc[s], 0.95)),
                            observed_replicating_validated=obs_rep, exceeds=bool(obs_rep > np.quantile(pc[s], 0.95)))
        nulls["role_permutation"] = pinfo
        pd.concat(fp.values()).to_csv(TABLES / f"inner_fingerprint_{tag}.csv", index=False)
        print(f"[stage] permutation null done ({time.time()-t_start:.0f}s)", flush=True)
    summary["nulls"] = nulls

    # ------------------------------------------------------------------ gate (ii), this dataset's component
    gate2 = {}
    for s in SITES:
        Tm = fp[("general", s)].set_index("concept")
        pre = Tm.loc[PREREG]
        good = pre.conf_same_sign_excl0 & (pre.exceeds_para_null if "exceeds_para_null" in pre else False)
        rn, pn = nulls.get("random_directions", {}).get(s, {}), nulls.get("role_permutation", {}).get(s, {})
        gate2[s] = dict(
            prereg_passing=list(pre[good].index), n_prereg_passing=int(good.sum()),
            count_exceeds_random=rn.get("exceeds"), count_exceeds_permutation=pn.get("exceeds"),
            PASS_this_dataset=bool(good.sum() >= 3 and rn.get("exceeds") and pn.get("exceeds")),
            replicating_topk=list(Tm[Tm.replicates].index),
            prereg={c: dict(disc=[r.disc_est, r.disc_lo, r.disc_hi], conf=[r.conf_est, r.conf_lo, r.conf_hi],
                            para_conf=[r.get("para_conf_est"), r.get("para_conf_lo"), r.get("para_conf_hi")],
                            exceeds_para_null=bool(r.get("exceeds_para_null", False)), validated=bool(r.validated))
                    for c, r in pre.iterrows()})
        for v in ("raw", "nuisance", "decision"):
            Tv = fp[(v, s)].set_index("concept").loc[PREREG]
            okv = Tv.conf_same_sign_excl0 & (Tv.exceeds_para_null if "exceeds_para_null" in Tv else False)
            gate2[s][f"sensitivity_{v}_prereg_passing"] = list(Tv[okv].index)
    summary["gate_ii_this_dataset"] = gate2

    # ------------------------------------------------------------------ scenario contrasts on the GENERAL-residualised deltas
    scen = {}
    lookup = {k: i for i, k in enumerate(zip(base.item_id, base.agent, base.turn))}
    prev_idx = np.array([lookup.get((i, ag, t - 1), -1) for i, ag, t in zip(base.item_id, base.agent, base.turn)])
    gi_, ho_ = (base.move == "gives_in").to_numpy(), (base.move == "holds_out").to_numpy()
    for label, shift in (("B_givesin_vs_holdsout", 0), ("C_givesin_vs_holdsout_before", -1)):
        for s in SITES:
            Ys = Y["general"][s]
            if shift:
                Ys = np.where((prev_idx >= 0)[:, None], Ys[np.maximum(prev_idx, 0)], np.nan)
            Ts = screen(split_by_half(base, Ys), cl_h, {h: (gi_[m], ho_[m]) for h, m in halves.items()}, concepts, valid)
            Ts.to_csv(TABLES / f"inner_{label}_{s}_{tag}.csv", index=False)
            scen[f"{label}_{s}"] = dict(n_a=int(gi_.sum()), n_b=int(ho_.sum()), disc_sig=int(Ts.disc_sig.sum()),
                                        replicated=list(Ts[Ts.replicates].concept),
                                        prereg_conf={c: [r.conf_est, r.conf_lo, r.conf_hi] for c, r in Ts.set_index("concept").loc[PREREG].iterrows()})
    pr = RAW / a.model / ds / "private.jsonl"
    if pr.exists():
        priv = {(x["item_id"], x["agent"]): x["verdict"] for x in read_jsonl(pr) if x["cond"] == "private_debate"}
        fin = (base.turn == T_ - 1).to_numpy()
        un = base[fin].groupby("item_id").verdict.agg(lambda s_: bool(s_.notna().all() and s_.nunique() == 1 and len(s_) == 3))
        in_un = fin & base.item_id.map(un).fillna(False).astype(bool).to_numpy()
        pv = np.array([priv.get((i, ag)) for i, ag in zip(base.item_id, base.agent)], dtype=object)
        has = np.array([isinstance(x, str) for x in pv])
        switch = in_un & has & (pv != base.verdict.to_numpy())
        stay = in_un & has & (pv == base.verdict.to_numpy())
        scen["D_hidden_dissent_counts"] = dict(unanimous_final_agents=int(in_un.sum()), missing_private=int((in_un & ~has).sum()),
                                               switch=int(switch.sum()), stay=int(stay.sum()))
        for s in SITES:
            Ts = screen(split_by_half(base, Y["general"][s]), cl_h, {h: (switch[m], stay[m]) for h, m in halves.items()}, concepts, valid)
            Ts.to_csv(TABLES / f"inner_D_hidden_dissent_{s}_{tag}.csv", index=False)
            scen[f"D_hidden_dissent_{s}"] = dict(disc_sig=int(Ts.disc_sig.sum()), replicated=list(Ts[Ts.replicates].concept))
    EF = []
    for s in SITES:
        cs = list(dict.fromkeys(PREREG + gate2[s]["replicating_topk"]))
        ci = [concepts.index(c) for c in cs]
        ok = np.isfinite(Y["general"][s]).all(1)
        Ys = np.nan_to_num(Y["general"][s][:, ci])
        for dom in DOMAIN_LABEL:
            dm = (base.domain == dom).to_numpy()
            e, lo, hi, *_ = paired_diff(cl_all, Ys, out_mask & dm & ok, maj_mask & dm & ok)
            EF += [dict(contrast=f"E_{dom}", site=s, concept=c, estimate=e[j], lower=lo[j], upper=hi[j]) for j, c in enumerate(cs)]
        right = out_mask & (base.prev_correct.astype(str) == "True").to_numpy()  # own t-1 answer (the held minority view) correct
        wrong = out_mask & (base.prev_correct.astype(str) == "False").to_numpy()
        e, lo, hi, *_ = paired_diff(cl_all, Ys, right & ok, wrong & ok)
        EF += [dict(contrast="F_right_vs_wrong_minority", site=s, concept=c, estimate=e[j], lower=lo[j], upper=hi[j]) for j, c in enumerate(cs)]
    pd.DataFrame(EF).to_csv(TABLES / f"inner_EF_{tag}.csv", index=False)
    summary["scenarios"] = scen
    summary["runtime_min"] = round((time.time() - t_start) / 60, 1)
    json.dump(summary, open(TABLES / f"inner_summary_{tag}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: summary[k] for k in ("gate_i_this_dataset", "nulls", "role_counts", "missing_reader_rows")}, indent=1,
                     default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
