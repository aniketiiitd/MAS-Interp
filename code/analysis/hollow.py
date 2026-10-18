"""V3-C4 (ii) hollow consensus and (iii) reader invariance (PLAN_V2.md, V3 plan frozen 2026-09-17 07:00).

SAY measures per public agent-turn: stated CONFIDENCE; consensus-citation flag on EVIDENCE.
ENCODE measures from the participant replay forward pass on the same prompt (prefilled "VERDICT:"): p(verdict) renormalised over
the option letters and margin = lp(verdict) - max lp(other letters), for the verdict the agent actually wrote; frozen
content_correctness probe (CPT worlds, answer position, L19; positive = the advocated answer is the target) read at the decision
site. The probe never saw a debate.
Outcomes: (1) private switch among agents whose final public verdicts are unanimous (private verdict != own final public verdict;
also != greedy public verdict); missing private verdicts are dropped and the drop rate is reported. (2) wrong unanimous consensus
(choice items with a target; item level).
Statistics: family-adjusted AUROC = within-family pairs pooled over the 15 families; item-cluster bootstrap (analysis_group for
persuasion) with a single weight draw shared by every predictor, so differences between predictors are paired. Transcript
baselines are out-of-fold (GroupKFold by cluster): TF-IDF logistic on final messages; a public-history logistic.
Reader invariance: the same encode predictors from an observer forward pass on the identical transcript (--observer), participant AUROC
minus observer AUROC with a paired CI.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
from common import *  # noqa

B = 2000
SEED = 240091
CITE = re.compile(r"consensus|unanimous|all agents|other agents|everyone|agree", re.I)


# ---------------------------------------------------------------------------------------------------------------------------
def letter_stats(lp, v):
    if not isinstance(lp, dict) or not lp or not isinstance(v, str) or v not in lp:
        return np.nan, np.nan, None
    L = list(lp)
    x = np.array([lp[k] for k in L], dtype=np.float64)
    p = np.exp(x - np.logaddexp.reduce(x))
    i = L.index(v)
    others = np.delete(x, i)
    return float(p[i]), float(x[i] - others.max()) if len(others) else np.nan, L[int(x.argmax())]


def argmax_letter(lp):
    if not isinstance(lp, dict) or not lp:
        return None
    return max(lp, key=lp.get)


def probe_scores(acts_path, probe, site=0):
    A = np.load(acts_path, mmap_mode="r")
    gi = probe["grid_index"]
    mu, sd = np.array(probe["mu"], np.float32), np.array(probe["sd"], np.float32)
    wl, wd = np.array(probe["w_log"], np.float32), np.array(probe["w_dom"], np.float32)
    out_l, out_d = np.empty(A.shape[0], np.float32), np.empty(A.shape[0], np.float32)
    for i in range(0, A.shape[0], 2048):
        Z = (A[i:i + 2048, site, gi].astype(np.float32) - mu) / sd
        out_l[i:i + 2048] = Z @ wl + probe["b_log"]
        out_d[i:i + 2048] = Z @ wd
    return out_l, out_d


# ---------------------------------------------------------------------------------------------------------------------------
class Boot:
    """Shared cluster-bootstrap weights for one analysis table (rows -> clusters)."""

    def __init__(self, clusters, B=B, seed=SEED):
        codes, uniq = pd.factorize(pd.Series(clusters))
        self.codes = codes
        self.nc = len(uniq)
        rng = np.random.default_rng(seed)
        counts = rng.multinomial(self.nc, np.full(self.nc, 1.0 / self.nc), size=B).astype(np.float32)
        self.W = counts[:, codes]  # B x n

    def ci(self, boots, est):
        b = boots[np.isfinite(boots)]
        lo, hi = np.quantile(b, [0.025, 0.975])
        p = 2 * min((b <= 0).mean(), (b >= 0).mean())
        return float(est), float(lo), float(hi), float(min(1.0, max(p, 1.0 / len(b))))


def _auc_parts(score, y, W):
    """Weighted concordance numerator and denominator for one family. W: (k, n)."""
    order = np.argsort(score, kind="mergesort")
    s, yy, Ws = score[order], y[order].astype(bool), W[:, order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    posw = np.add.reduceat(Ws * yy, starts, axis=1)
    negw = np.add.reduceat(Ws * (~yy), starts, axis=1)
    below = np.cumsum(negw, axis=1) - negw
    num = (posw * (below + 0.5 * negw)).sum(1)
    den = posw.sum(1) * negw.sum(1)
    return num, den


def auc_weighted(score, y, fam, W):
    """Family-adjusted (fam given) or pooled (fam None) AUROC for each weight row of W (k, n)."""
    score, y = np.asarray(score, float), np.asarray(y, bool)
    num = np.zeros(W.shape[0])
    den = np.zeros(W.shape[0])
    groups = [np.arange(len(y))] if fam is None else [np.flatnonzero(fam == f) for f in np.unique(fam)]
    for idx in groups:
        if y[idx].all() or (~y[idx]).all():
            continue
        n_, d_ = _auc_parts(score[idx], y[idx], W[:, idx])
        num += n_
        den += d_
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def auroc_table(df, preds, outcome, boot_on, fam_col="family", ref_pairs=(), item_col="item_id"):
    """AUROC at three levels of adjustment, with a paired bootstrap, on rows where every predictor is present.

    The third level exists because of a retraction: a claim about which agent behaves a certain way was reported with
    family-adjusted AUROC, and collapsed from 0.711 to 0.521 once agents facing the same item were compared with each
    other. Family adjustment removes task composition but leaves item difficulty, which is enough to carry an effect on
    its own. Reporting pooled, family- and item-adjusted side by side makes that visible instead of hidden: a claim whose
    number falls away as the adjustment tightens is an artefact of what is being compared.
    """
    d = df.dropna(subset=list(preds) + [outcome]).reset_index(drop=True)
    y = d[outcome].astype(bool).to_numpy()
    fam = d[fam_col].to_numpy()
    item = d[item_col].to_numpy() if item_col in d else None
    # An item-level outcome has one row per item, so "within item" has nothing to compare and every group AUROC is
    # undefined -- which empties the bootstrap array and makes np.quantile raise. Only adjust within item when items
    # actually repeat.
    if item is not None and len(np.unique(item)) == len(item):
        item = None
    bt = Boot(d[boot_on].to_numpy())
    one = np.ones((1, len(d)), np.float32)
    rows, boots = [], {}
    for name in preds:
        s = d[name].to_numpy(float)
        levels = [("family_adjusted", fam), ("pooled", None)]
        if item is not None:
            levels.append(("item_adjusted", item))
        for adj, f in levels:
            est = auc_weighted(s, y, f, one)[0]
            bs = auc_weighted(s, y, f, bt.W)
            boots[(name, adj)] = (est, bs)
            e, lo, hi, _ = bt.ci(bs, est)
            rows.append(dict(outcome=outcome, predictor=name, scale=adj, auroc=e, lower=lo, upper=hi, n=len(d), n_pos=int(y.sum()),
                             n_clusters=bt.nc))
    diffs = []
    for a, b in ref_pairs:
        for adj in ("family_adjusted", "pooled", "item_adjusted"):
            if (a, adj) in boots and (b, adj) in boots:
                ea, ba = boots[(a, adj)]
                eb, bb = boots[(b, adj)]
                e, lo, hi, p = bt.ci(ba - bb, ea - eb)
                diffs.append(dict(outcome=outcome, contrast=f"{a} - {b}", scale=adj, diff=e, lower=lo, upper=hi, p=p, n=len(d)))
    return pd.DataFrame(rows), pd.DataFrame(diffs)


def oof_text_model(texts, y, groups, kind="tfidf", X=None, seed=SEED):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    y = np.asarray(y, bool)
    out = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in gkf.split(np.zeros(len(y)), y, groups):
        if y[tr].all() or (~y[tr]).all():
            continue
        if kind == "tfidf":
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000)
            Xtr, Xte = vec.fit_transform([texts[i] for i in tr]), vec.transform([texts[i] for i in te])
        else:
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        clf.fit(Xtr, y[tr])
        out[te] = clf.predict_proba(Xte)[:, 1]
    return out


def fam_std_diff(df, col, group_col, boot_on, fam_col="family"):
    """Family-standardised difference mean(group) - mean(rest) with ATT weights (defector counts per family); cluster bootstrap."""
    d = df.dropna(subset=[col]).reset_index(drop=True)
    g = d[group_col].astype(bool).to_numpy()
    x = d[col].to_numpy(float)
    fam = d[fam_col].to_numpy()
    bt = Boot(d[boot_on].to_numpy())
    Wall = np.vstack([np.ones((1, len(d)), np.float32), bt.W])

    def stat(W):
        num = np.zeros(W.shape[0])
        den = np.zeros(W.shape[0])
        for f in np.unique(fam):
            idx = fam == f
            gi, ri = idx & g, idx & ~g
            if not gi.any() or not ri.any():
                continue
            wg, wr = W[:, gi], W[:, ri]
            sg, sr = wg.sum(1), wr.sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                diff = (wg * x[gi]).sum(1) / sg - (wr * x[ri]).sum(1) / sr
            ok = (sg > 0) & (sr > 0)
            num += np.where(ok, sg * diff, 0)
            den += np.where(ok, sg, 0)
        return num / den

    raw_all = stat(Wall)
    e, lo, hi, p = bt.ci(raw_all[1:], raw_all[0])

    def raw(W):
        return (W[:, g] * x[g]).sum(1) / W[:, g].sum(1) - (W[:, ~g] * x[~g]).sum(1) / W[:, ~g].sum(1)

    r_all = raw(Wall)
    re_, rlo, rhi, rp = bt.ci(r_all[1:], r_all[0])
    return dict(fam_std_diff=e, lower=lo, upper=hi, p=p, raw_diff=re_, raw_lower=rlo, raw_upper=rhi, n_group=int(g.sum()),
                n_rest=int((~g).sum()), n_clusters=bt.nc)


def group_mean_ci(df, col, boot_on):
    d = df.dropna(subset=[col])
    if not len(d):
        return np.nan, np.nan, np.nan, 0
    bt = Boot(d[boot_on].to_numpy(), B=1000)
    x = d[col].to_numpy(float)
    bs = (bt.W * x).sum(1) / bt.W.sum(1)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi), len(d)


# ---------------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="release_600")
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--replay", default=None, help="participant replay dir under ACTS/<model> (default replay_<ds>_<ds>_T<turns>)")
    ap.add_argument("--observer", default=None, help="observer replay dir under ACTS/<model> (reader invariance)")
    ap.add_argument("--private", default="generated", choices=["generated", "aud"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--no_figures", action="store_true")
    ap.add_argument("--exclude_split", default=None, help="drop items of this split (core confirmation: release_600)")
    a = ap.parse_args()
    M, ds, T = a.model, a.dataset, a.turns
    last = T - 1
    tag = f"{M}_{ds}_T{T}{a.tag}"
    replay = ACTS / M / (a.replay or f"replay_{ds}_{ds}_T{T}")
    core = {r["id"]: r for r in load_core()}
    unit = lambda i: core[i]["meta"].get("analysis_group") if core[i]["stratum"] == "persuasion" else core[i]["source_item_id"]
    views = {}

    def view(i):
        if i not in views:
            views[i] = item_view(core[i])
        return views[i]

    # ---- public transcripts
    d = RAW / M / ds
    recs = []
    for f in ["t0"] + [f"{c}_t{t}" for c in ("debate", "pool") for t in range(1, T)]:
        for x in read_jsonl(d / f"{f}.jsonl"):
            if x["agent"] < 3:
                recs.append(dict(item_id=x["item_id"], cond=x["cond"], agent=x["agent"], turn=x["turn"], verdict=x["verdict"],
                                 confidence=x["confidence"], evidence=x.get("evidence") or "", text=x["text"]))
    P = pd.DataFrame(recs)
    if a.exclude_split:
        ex = set(json.load(open(ROOT / "data/splits/splits.json"))[a.exclude_split])
        P = P[~P.item_id.isin(ex)].reset_index(drop=True)
    # ---- replay (participant encode)
    idx = pd.DataFrame(read_jsonl(replay / "index.jsonl"))
    probe = json.load(open(ACTS / M / "probes_frozen_cpt_core.json"))["content_correctness"]
    rmeta = json.load(open(replay / "meta.json"))
    assert rmeta["grid"][probe["grid_index"]] == probe["layer"], "probe layer not at the same grid index in the replay"
    assert rmeta["sites"][0] == "decision"
    s_log, s_dom = probe_scores(replay / "acts.npy", probe)
    idx["probe_log"] = s_log[idx.row.to_numpy()]
    idx["probe_dom"] = s_dom[idx.row.to_numpy()]
    P = P.merge(idx[["item_id", "cond", "agent", "turn", "lp_letters", "probe_log", "probe_dom"]], on=["item_id", "cond", "agent", "turn"],
                how="left", validate="one_to_one")
    st = [letter_stats(lp, v) for lp, v in zip(P.lp_letters, P.verdict)]
    P["p_verdict"] = [s[0] for s in st]
    P["margin"] = [s[1] for s in st]
    P["greedy"] = P.lp_letters.map(argmax_letter)
    P["cite"] = P.evidence.map(lambda e: float(bool(CITE.search(e))))
    P["family"] = P.item_id.map(lambda i: family_of(core[i]))
    P["domain"] = P.item_id.map(lambda i: core[i]["primary_domain"])
    P["cluster"] = P.item_id.map(unit)
    P["kind"] = P.item_id.map(lambda i: view(i)["kind"])
    P["target"] = P.item_id.map(lambda i: view(i)["target"])
    # ---- observer encode (optional)
    have_obs = False
    if a.observer and (ACTS / M / a.observer / "index.jsonl").exists():
        oidx = pd.DataFrame(read_jsonl(ACTS / M / a.observer / "index.jsonl"))
        ometa = json.load(open(ACTS / M / a.observer / "meta.json"))
        assert ometa["grid"][probe["grid_index"]] == probe["layer"]
        ol, od = probe_scores(ACTS / M / a.observer / "acts.npy", probe)
        oidx["obs_probe_log"], oidx["obs_probe_dom"] = ol[oidx.row.to_numpy()], od[oidx.row.to_numpy()]
        oidx = oidx.rename(columns={"lp_letters": "obs_lp"})
        oidx["cond"] = "debate"
        P = P.merge(oidx[["item_id", "cond", "agent", "turn", "obs_lp", "obs_probe_log", "obs_probe_dom"]],
                    on=["item_id", "cond", "agent", "turn"], how="left", validate="one_to_one")
        so = [letter_stats(lp, v) for lp, v in zip(P.obs_lp, P.verdict)]
        P["obs_p_verdict"] = [s[0] for s in so]
        P["obs_margin"] = [s[1] for s in so]
        have_obs = True
    # ---- private verdicts
    if a.private == "generated":
        pr = pd.DataFrame(read_jsonl(d / "private.jsonl"))
        pr["cond"] = pr.cond.str.replace("private_", "", regex=False)
        priv = pr.set_index(["item_id", "cond", "agent"]).verdict.to_dict()
    else:
        adir = next(p_ for p_ in (ACTS / M / f"frames_audience_{ds}_notrelease_600", ACTS / M / f"frames_audience_{ds}") if p_.exists())
        ai = pd.DataFrame(read_jsonl(adir / "index.jsonl"))
        ai = ai[ai.frame == "priv_ctx"]
        priv = {(i, c, g): argmax_letter(lp) for i, c, g, lp in zip(ai.item_id, ai.cond, ai.agent, ai.lp_letters)}
    summary = dict(model=M, dataset=ds, turns=T, replay=replay.name, observer=a.observer if have_obs else None, private_source=a.private,
                   split_role="discovery" if ds == "release_600" else "confirmation")

    # ---- per-condition agent table at the final turn
    tables, diffs, traj_rows, calib_rows, fs_rows = [], [], [], [], []
    per_cond = {}
    for cond in ("debate", "pool"):
        F = P[(P.cond == cond) & (P.turn == last)].copy()
        cnt = F.groupby("item_id").verdict.agg(lambda s: s.notna().sum() == 3 and s.nunique() == 1)
        F["pub_unanimous"] = F.item_id.map(cnt)
        U = F[F.pub_unanimous].copy()
        U["private"] = [priv.get((i, cond, g)) for i, g in zip(U.item_id, U.agent)]
        n_before = len(U)
        miss = U.private.isna()
        summary[f"{cond}_unanimous_agents"] = int(n_before)
        summary[f"{cond}_private_missing_rate"] = float(miss.mean())
        U = U[~miss].copy()
        U["switch"] = (U.private != U.verdict).astype(float)
        U["switch_greedy"] = np.where(U.greedy.notna(), (U.private != U.greedy).astype(float), np.nan)
        summary[f"{cond}_switch_rate"] = float(U.switch.mean())
        summary[f"{cond}_n_switch"] = int(U.switch.sum())
        # direction on target items
        tg = U[U.target.notna() & (U.kind == "choice") & (U.switch == 1)]
        summary[f"{cond}_switch_wrong_to_right"] = int(((tg.verdict != tg.target) & (tg.private == tg.target)).sum())
        summary[f"{cond}_switch_right_to_wrong"] = int(((tg.verdict == tg.target) & (tg.private != tg.target)).sum())
        # oriented predictors
        U["neg_margin"], U["neg_p"], U["neg_conf"] = -U.margin, -U.p_verdict, -pd.to_numeric(U.confidence, errors="coerce")
        U["neg_probe_log"], U["neg_probe_dom"] = -U.probe_log, -U.probe_dom
        # public history features (transcript-only)
        H = P[(P.cond.isin([cond, "t0"]))].sort_values("turn")
        hist = H.groupby(["item_id", "agent"]).agg(n_changes=("verdict", lambda s: int((s.dropna() != s.dropna().shift()).iloc[1:].sum())),
                                                   min_conf=("confidence", lambda s: pd.to_numeric(s, errors="coerce").min()),
                                                   cite_rate=("cite", "mean"))
        U = U.join(hist, on=["item_id", "agent"])
        itemagree = P[P.cond == cond].groupby(["item_id", "turn"]).verdict.agg(lambda s: s.notna().sum() == 3 and s.nunique() == 1)
        first_un = {}
        for (i, t), v in itemagree.items():
            if v and i not in first_un:
                first_un[i] = t
        U["first_unanimous_turn"] = U.item_id.map(lambda i: first_un.get(i, T))
        per_cond[cond] = U
        # AUROCs (choice/deception items with letters; common rows)
        U["tfidf"] = oof_text_model(list(U.text), U.switch.to_numpy(bool), U.cluster.to_numpy())
        Xh = U[["confidence", "min_conf", "n_changes", "cite", "cite_rate", "first_unanimous_turn"]].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
        U["public_hist"] = oof_text_model(None, U.switch.to_numpy(bool), U.cluster.to_numpy(), kind="features", X=Xh)
        preds = ["neg_margin", "neg_p", "neg_probe_log", "neg_probe_dom", "neg_conf", "cite", "tfidf", "public_hist"]
        pairs = [("neg_margin", "neg_conf"), ("neg_margin", "tfidf"), ("neg_margin", "public_hist"), ("neg_probe_log", "tfidf"),
                 ("neg_probe_log", "neg_conf"), ("neg_probe_log", "public_hist"), ("neg_probe_dom", "tfidf"), ("neg_probe_dom", "neg_conf"),
                 ("neg_probe_dom", "public_hist")]
        if have_obs and cond == "debate":
            U["obs_neg_margin"], U["obs_neg_p"] = -U.obs_margin, -U.obs_p_verdict
            U["obs_neg_probe_log"], U["obs_neg_probe_dom"] = -U.obs_probe_log, -U.obs_probe_dom
            preds += ["obs_neg_margin", "obs_neg_p", "obs_neg_probe_log", "obs_neg_probe_dom"]
            pairs += [("neg_margin", "obs_neg_margin"), ("neg_p", "obs_neg_p"), ("neg_probe_log", "obs_neg_probe_log"),
                      ("neg_probe_dom", "obs_neg_probe_dom")]
        for outcome in ("switch", "switch_greedy"):
            t_, d_ = auroc_table(U.assign(**{outcome: U[outcome]}), preds, outcome, "cluster", ref_pairs=pairs)
            tables.append(t_.assign(cond=cond))
            diffs.append(d_.assign(cond=cond))
        # calibration: stated-confidence bins vs internal p bins
        conf = pd.to_numeric(U.confidence, errors="coerce")
        U["conf_bin"] = pd.cut(conf, [-1, 89, 99, 100], labels=["<=89", "90-99", "100"])
        U["p_bin"] = pd.cut(U.p_verdict, [-1, 0.9, 0.99, 0.999, 2], labels=["<0.9", "0.9-0.99", "0.99-0.999", ">=0.999"])
        for kind_, col in (("stated_confidence", "conf_bin"), ("internal_p", "p_bin")):
            for b, g in U.dropna(subset=["p_verdict"]).groupby(col, observed=False):
                e, lo, hi, n = group_mean_ci(g, "switch", "cluster")
                calib_rows.append(dict(cond=cond, measure=kind_, bin=str(b), switch_rate=e, lower=lo, upper=hi, n_agents=n,
                                       n_switch=int(g.switch.sum()), share=n / max(1, len(U.dropna(subset=["p_verdict"])))))
        # trajectories for eventual defectors vs stayers (same agents, all turns)
        keyset = U[["item_id", "agent", "switch", "verdict"]].rename(columns={"verdict": "final_verdict"})
        Tr = P[(P.cond.isin([cond, "t0"]))].merge(keyset, on=["item_id", "agent"])
        Tr["agrees_final"] = (Tr.verdict == Tr.final_verdict).astype(float)
        Tr["conf01"] = pd.to_numeric(Tr.confidence, errors="coerce") / 100
        Tr["doubt"] = 1 - Tr.p_verdict
        Tr = Tr[Tr.item_id.isin(U.dropna(subset=["p_verdict"]).item_id)]
        for t in range(T):
            G = Tr[Tr.turn == t]
            for col in ("agrees_final", "conf01", "p_verdict", "margin", "probe_log", "probe_dom", "cite") + (
                    ("obs_margin", "obs_probe_dom") if (have_obs and cond == "debate") else ()):
                for grp, gg in (("defectors", G[G.switch == 1]), ("stayers", G[G.switch == 0])):
                    e, lo, hi, n = group_mean_ci(gg, col, "cluster")
                    traj_rows.append(dict(cond=cond, turn=t, measure=col, group=grp, mean=e, lower=lo, upper=hi, n=n))
            if cond == "debate" or t in (1, last):
                for col in ("margin", "p_verdict", "probe_log", "probe_dom", "conf01", "agrees_final", "cite"):
                    r = fam_std_diff(G.assign(defector=G.switch == 1), col, "defector", "cluster")
                    fs_rows.append(dict(cond=cond, turn=t, measure=col, **r))
        # within-item mixed cases (>=1 defector and >=1 stayer in the same item) at turns 1 and final
        mixed = U.groupby("item_id").switch.agg(lambda s: 0 < s.sum() < len(s))
        mixed_items = set(mixed[mixed].index)
        summary[f"{cond}_mixed_items"] = len(mixed_items)
        for t in (1, last):
            G = Tr[(Tr.turn == t) & Tr.item_id.isin(mixed_items)]
            for col in ("margin", "p_verdict", "probe_log", "probe_dom"):
                w = G.groupby(["item_id", "switch"])[col].mean().unstack()
                if {0.0, 1.0} <= set(w.columns):
                    dd = (w[1.0] - w[0.0]).dropna()
                    if len(dd) >= 5:
                        from stats import fast_cluster_mean_ci
                        e, lo, hi, p, n = fast_cluster_mean_ci(dd.to_numpy(), dd.index.map(unit).to_numpy())
                        fs_rows.append(dict(cond=cond, turn=t, measure=f"{col}_within_item_mixed", fam_std_diff=e, lower=lo, upper=hi,
                                            p=p, n_group=len(dd), n_clusters=n))
    Tab = pd.concat(tables, ignore_index=True)
    Dif = pd.concat(diffs, ignore_index=True)
    Traj, Cal, FS = pd.DataFrame(traj_rows), pd.DataFrame(calib_rows), pd.DataFrame(fs_rows)

    # ---- participant vs observer on identical text, final turn (debate)
    U = per_cond["debate"]
    if have_obs:
        from stats import fast_cluster_mean_ci
        V = U.dropna(subset=["margin", "obs_margin"]).copy()
        V["obs_greedy"] = V.obs_lp.map(argmax_letter)
        V["obs_disagree"] = (V.obs_greedy != V.verdict).astype(float)
        V["m_diff"] = V.margin - V.obs_margin
        for lab, g in (("all", V), ("defectors", V[V.switch == 1]), ("stayers", V[V.switch == 0])):
            e, lo, hi, _, n = fast_cluster_mean_ci(g.m_diff.to_numpy(), g.cluster.to_numpy())
            summary[f"obs_final_margin_participant_minus_observer_{lab}"] = [e, lo, hi, int(len(g))]
            e, lo, hi, _, n = fast_cluster_mean_ci(g.obs_disagree.to_numpy(), g.cluster.to_numpy())
            summary[f"obs_final_observer_argmax_differs_{lab}"] = [e, lo, hi, int(len(g))]
            summary[f"obs_final_mean_margin_participant_{lab}"] = float(g.margin.mean())
            summary[f"obs_final_mean_margin_observer_{lab}"] = float(g.obs_margin.mean())
        for t_col, lab in (("m_diff", "final"),):
            r = fam_std_diff(V.assign(defector=V.switch == 1), t_col, "defector", "cluster")
            fs_rows.append(dict(cond="debate", turn=last, measure="participant_minus_observer_margin", **r))
        Tobs = P[(P.cond == "debate") & (P.turn >= 1)].merge(U[["item_id", "agent", "switch"]], on=["item_id", "agent"]).dropna(subset=["margin", "obs_margin"])
        Tobs["m_diff"] = Tobs.margin - Tobs.obs_margin
        Tobs["pd_diff"] = Tobs.probe_dom - Tobs.obs_probe_dom
        for t in range(1, T):
            G = Tobs[Tobs.turn == t].assign(defector=lambda x: x.switch == 1)
            for col, lab in (("m_diff", "participant_minus_observer_margin"), ("pd_diff", "participant_minus_observer_probe_dom")):
                if t != last or col != "m_diff":
                    fs_rows.append(dict(cond="debate", turn=t, measure=lab, **fam_std_diff(G, col, "defector", "cluster")))
                for grp, gg in (("defectors", G[G.defector]), ("stayers", G[~G.defector])):
                    e, lo, hi, n = group_mean_ci(gg, col, "cluster")
                    traj_rows.append(dict(cond="debate", turn=t, measure=lab, group=grp, mean=e, lower=lo, upper=hi, n=n))
        od, _ = auroc_table(V, ["obs_disagree"], "switch", "cluster")
        summary["obs_disagree_auroc_switch_family_adjusted"] = od[od.scale == "family_adjusted"][["auroc", "lower", "upper"]].iloc[0].tolist()
        sw = V[V.obs_disagree == 1]
        summary["switch_rate_when_observer_disagrees"] = [float(sw.switch.mean()) if len(sw) else None, int(len(sw))]
        summary["switch_rate_when_observer_agrees"] = [float(V[V.obs_disagree == 0].switch.mean()), int((V.obs_disagree == 0).sum())]
        summary["private_equals_observer_argmax_among_disagreeing_switchers"] = float((sw[sw.switch == 1].private == sw[sw.switch == 1].obs_greedy).mean()) if len(sw[sw.switch == 1]) else None
        # selection control: the participant's own verdict letter was sampled from the participant's distribution, which inflates the
        # participant margin for that letter relative to any other reader. Re-score both readers on the letter of the previous-turn
        # unanimous consensus (fixed by the text both readers read), for agents whose previous-turn group was unanimous.
        prevv = P[(P.cond == "debate") & (P.turn == last - 1)].groupby("item_id").verdict.agg(
            lambda s_: s_.iloc[0] if (s_.notna().sum() == 3 and s_.nunique() == 1) else None)
        V["prev_letter"] = V.item_id.map(prevv)
        W_ = V[V.prev_letter.notna()].copy()
        W_["m_part_prev"] = [letter_stats(lp, L)[1] for lp, L in zip(W_.lp_letters, W_.prev_letter)]
        W_["m_obs_prev"] = [letter_stats(lp, L)[1] for lp, L in zip(W_.obs_lp, W_.prev_letter)]
        W_["m_diff_prev"] = W_.m_part_prev - W_.m_obs_prev
        W_ = W_.dropna(subset=["m_diff_prev"])
        r = fam_std_diff(W_.assign(defector=W_.switch == 1), "m_diff_prev", "defector", "cluster")
        fs_rows.append(dict(cond="debate", turn=last, measure="participant_minus_observer_margin_prevconsensus_letter", **r))
        # scale control: is the defector gap just "less saturated participants move more under any re-framing"? Standardise within
        # family x quintile of the participant margin itself.
        V["mbin"] = pd.qcut(V.margin.rank(method="first"), 5, labels=False)
        V["fam_mbin"] = V.family + "|" + V.mbin.astype(str)
        r = fam_std_diff(V.assign(defector=V.switch == 1), "m_diff", "defector", "cluster", fam_col="fam_mbin")
        fs_rows.append(dict(cond="debate", turn=last, measure="participant_minus_observer_margin_family_x_marginquintile", **r))
        r = fam_std_diff(V.assign(defector=V.switch == 1), "m_diff", "defector", "cluster", fam_col="mbin")
        fs_rows.append(dict(cond="debate", turn=last, measure="participant_minus_observer_margin_marginquintile_only", **r))
        for lab, g in (("defectors", W_[W_.switch == 1]), ("stayers", W_[W_.switch == 0])):
            e, lo, hi, _, n = fast_cluster_mean_ci(g.m_diff_prev.to_numpy(), g.cluster.to_numpy())
            summary[f"obs_final_margin_participant_minus_observer_prevconsensus_letter_{lab}"] = [e, lo, hi, int(len(g))]
        W_["neg_m_part_prev"], W_["neg_m_obs_prev"] = -W_.m_part_prev, -W_.m_obs_prev
        t_, d_ = auroc_table(W_, ["neg_m_part_prev", "neg_m_obs_prev"], "switch", "cluster", ref_pairs=[("neg_m_part_prev", "neg_m_obs_prev")])
        tables.append(t_.assign(cond="debate_prevconsensus_letter"))
        diffs.append(d_.assign(cond="debate_prevconsensus_letter"))
        Tab, Dif = pd.concat(tables, ignore_index=True), pd.concat(diffs, ignore_index=True)
        Traj, FS = pd.DataFrame(traj_rows), pd.DataFrame(fs_rows)
    # ---- AUROC by turn (debate, private switch): does the unprinted state predict defection before the words converge?
    keyset = U[["item_id", "agent", "switch", "cluster", "family"]]
    byturn = []
    for t in range(T):
        G = P[(P.cond.isin(["debate", "t0"])) & (P.turn == t)].drop(columns=["cluster", "family"]).merge(keyset, on=["item_id", "agent"])
        G["neg_margin"], G["neg_p"], G["neg_probe_log"], G["neg_probe_dom"] = -G.margin, -G.p_verdict, -G.probe_log, -G.probe_dom
        G["neg_conf"] = -pd.to_numeric(G.confidence, errors="coerce")
        t_, _ = auroc_table(G, ["neg_margin", "neg_p", "neg_probe_log", "neg_probe_dom", "neg_conf", "cite"], "switch", "cluster")
        byturn.append(t_.assign(turn=t))
        if have_obs and t >= 1:
            G["obs_neg_margin"], G["obs_neg_probe_dom"] = -G.obs_margin, -G.obs_probe_dom
            t_, _ = auroc_table(G, ["obs_neg_margin", "obs_neg_probe_dom"], "switch", "cluster")
            byturn.append(t_.assign(turn=t))
    ByTurn = pd.concat(byturn, ignore_index=True)

    # ---- wrong unanimous consensus (item level, debate and pool, choice items with a target)
    wrong_tabs, wrong_diffs = [], []
    for cond in ("debate", "pool"):
        F = P[(P.cond == cond) & (P.turn == last) & (P.kind == "choice") & P.target.notna()].copy()
        g = F.groupby("item_id")
        I = pd.DataFrame(dict(unanimous=g.verdict.agg(lambda s: s.notna().sum() == 3 and s.nunique() == 1),
                              consensus=g.verdict.agg(lambda s: s.iloc[0]), target=g.target.first(),
                              mean_margin=g.margin.mean(), min_margin=g.margin.min(), mean_p=g.p_verdict.mean(),
                              mean_probe_log=g.probe_log.mean(), mean_probe_dom=g.probe_dom.mean(),
                              mean_conf=g.confidence.agg(lambda s: pd.to_numeric(s, errors="coerce").mean()),
                              cite_rate=g.cite.mean(), text=g.text.agg(lambda s: "\n\n".join(s)),
                              family=g.family.first(), cluster=g.cluster.first()))
        if have_obs and cond == "debate":
            I["obs_mean_margin"] = g.obs_margin.mean()
            I["obs_mean_p"] = g.obs_p_verdict.mean()
            I["obs_mean_probe_log"] = g.obs_probe_log.mean()
            I["obs_mean_probe_dom"] = g.obs_probe_dom.mean()
        I = I[I.unanimous].copy()
        I["wrong"] = (I.consensus != I.target).astype(float)
        summary[f"{cond}_unanimous_target_items"] = int(len(I))
        summary[f"{cond}_wrong_unanimous_items"] = int(I.wrong.sum())
        for c in ("mean_margin", "min_margin", "mean_p", "mean_probe_log", "mean_probe_dom", "mean_conf"):
            I["neg_" + c] = -I[c]
        I["tfidf"] = oof_text_model(list(I.text), I.wrong.to_numpy(bool), I.cluster.to_numpy())
        preds = ["neg_mean_margin", "neg_min_margin", "neg_mean_p", "neg_mean_probe_log", "neg_mean_probe_dom", "neg_mean_conf",
                 "cite_rate", "tfidf"]
        pairs = [("neg_mean_probe_log", "tfidf"), ("neg_mean_margin", "tfidf"), ("neg_mean_probe_log", "neg_mean_conf"),
                 ("neg_mean_margin", "neg_mean_conf"), ("neg_mean_probe_dom", "tfidf"), ("neg_mean_probe_dom", "neg_mean_conf")]
        if have_obs and cond == "debate":
            I["obs_neg_mean_margin"], I["obs_neg_mean_probe_log"] = -I.obs_mean_margin, -I.obs_mean_probe_log
            I["obs_neg_mean_probe_dom"] = -I.obs_mean_probe_dom
            preds += ["obs_neg_mean_margin", "obs_neg_mean_probe_log", "obs_neg_mean_probe_dom"]
            pairs += [("neg_mean_margin", "obs_neg_mean_margin"), ("neg_mean_probe_log", "obs_neg_mean_probe_log"),
                      ("neg_mean_probe_dom", "obs_neg_mean_probe_dom")]
        t_, d_ = auroc_table(I, preds, "wrong", "cluster", ref_pairs=pairs)
        wrong_tabs.append(t_.assign(cond=cond))
        wrong_diffs.append(d_.assign(cond=cond))
    Wt, Wd = pd.concat(wrong_tabs, ignore_index=True), pd.concat(wrong_diffs, ignore_index=True)

    # ---- debate vs pool: does debate destroy the information in the words but not in the state? (shared item-cluster bootstrap)
    xc = []
    need = ["neg_margin", "neg_conf", "neg_probe_dom", "neg_probe_log", "switch"]
    C = pd.concat([per_cond[c].dropna(subset=need).assign(cond=c) for c in ("debate", "pool")], ignore_index=True)
    btc = Boot(C.cluster.to_numpy())
    one = np.ones((1, len(C)), np.float32)
    for pred in ("neg_margin", "neg_conf", "neg_probe_dom", "neg_probe_log"):
        vals = {}
        for c in ("debate", "pool"):
            m = (C.cond == c).to_numpy()
            s_, y_, f_ = C.loc[m, pred].to_numpy(float), C.loc[m, "switch"].to_numpy(bool), C.loc[m, "family"].to_numpy()
            vals[c] = (auc_weighted(s_, y_, f_, one[:, m])[0], auc_weighted(s_, y_, f_, btc.W[:, m]))
        e, lo, hi, p = btc.ci(vals["debate"][1] - vals["pool"][1], vals["debate"][0] - vals["pool"][0])
        xc.append(dict(predictor=pred, auroc_debate=vals["debate"][0], auroc_pool=vals["pool"][0], diff_debate_minus_pool=e, lower=lo,
                       upper=hi, p=p))
    XC = pd.DataFrame(xc)
    # internal-minus-stated advantage, debate vs pool (difference in differences)
    dd_ = {}
    for c in ("debate", "pool"):
        m = (C.cond == c).to_numpy()
        y_, f_ = C.loc[m, "switch"].to_numpy(bool), C.loc[m, "family"].to_numpy()
        a1 = auc_weighted(C.loc[m, "neg_margin"].to_numpy(float), y_, f_, np.vstack([one[:, m], btc.W[:, m]]))
        a2 = auc_weighted(C.loc[m, "neg_conf"].to_numpy(float), y_, f_, np.vstack([one[:, m], btc.W[:, m]]))
        dd_[c] = a1 - a2
    e, lo, hi, p = btc.ci(dd_["debate"][1:] - dd_["pool"][1:], dd_["debate"][0] - dd_["pool"][0])
    XC = pd.concat([XC, pd.DataFrame([dict(predictor="(neg_margin - neg_conf) DiD", diff_debate_minus_pool=e, lower=lo, upper=hi, p=p)])],
                   ignore_index=True)

    # ---- say saturation facts
    D = per_cond["debate"]
    summary["debate_final_unanimous_conf100_share"] = float((pd.to_numeric(D.confidence, errors="coerce") == 100).mean())
    summary["debate_final_unanimous_cite_share"] = float(D.cite.mean())
    summary["debate_cite_defectors"] = float(D[D.switch == 1].cite.mean())
    summary["debate_cite_stayers"] = float(D[D.switch == 0].cite.mean())
    summary["debate_final_p_verdict_ge_0999_share"] = float((D.p_verdict >= 0.999).mean())

    # ---- write
    out = TABLES
    Tab.to_csv(out / f"hollow_{tag}_auroc_switch.csv", index=False)
    Dif.to_csv(out / f"hollow_{tag}_auroc_switch_diffs.csv", index=False)
    Wt.to_csv(out / f"hollow_{tag}_auroc_wrong.csv", index=False)
    Wd.to_csv(out / f"hollow_{tag}_auroc_wrong_diffs.csv", index=False)
    Traj.to_csv(out / f"hollow_{tag}_trajectories.csv", index=False)
    Cal.to_csv(out / f"hollow_{tag}_calibration.csv", index=False)
    FS.to_csv(out / f"hollow_{tag}_defector_minus_stayer.csv", index=False)
    ByTurn.to_csv(out / f"hollow_{tag}_auroc_by_turn.csv", index=False)
    XC.to_csv(out / f"hollow_{tag}_debate_vs_pool.csv", index=False)
    json.dump(summary, open(out / f"hollow_{tag}_summary.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    print(json.dumps(summary, indent=1))
    print(Tab[Tab.scale == "family_adjusted"].round(3).to_string())
    print(Dif.round(3).to_string())
    print(Wt.round(3).to_string())
    print(Wd.round(3).to_string())
    print(FS[FS.turn.isin([0, 1, last])].round(3).to_string())
    print(Cal.round(3).to_string())
    print(XC.round(3).to_string())
    print(ByTurn[ByTurn.scale == "family_adjusted"].pivot_table(index="turn", columns="predictor", values="auroc").round(3).to_string())
    if not a.no_figures:
        figures(tag, Traj, Cal, ByTurn, FS, last)


# ---------------------------------------------------------------------------------------------------------------------------
def figures(tag, Traj, Cal, ByTurn, FS, last):
    import style
    from matplotlib.lines import Line2D
    from style import plt
    style.apply()
    DEF, STAY = style.TERRACOTTA, style.TEAL
    Tj = Traj[Traj.cond == "debate"]
    Fd = FS[FS.cond == "debate"]

    def band(ax, measure, grp, color, ls="-"):
        g = Tj[(Tj.measure == measure) & (Tj.group == grp)].sort_values("turn")
        ax.fill_between(g.turn, g.lower, g.upper, color=color, alpha=0.16, lw=0)
        ax.plot(g.turn, g["mean"], color=color, ls=ls, lw=1.8, marker="o", ms=3.0)
        return g

    def gap_note(ax, measure, fmt="{:+.2f}", ypos=0.06):
        parts = []
        for t in (1, last):
            r = Fd[(Fd.turn == t) & (Fd.measure == measure)]
            if len(r):
                r = r.iloc[0]
                parts.append(f"t={t}: {fmt.format(r.fam_std_diff)} [{fmt.format(r.lower)}, {fmt.format(r.upper)}]")
        ax.text(0.995, ypos, "family-standardised gap (defectors - stayers)  " + "   ".join(parts), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.6, color=style.SLATE)

    # Figure 1: three aligned strips over turns (debate)
    has_obs = (Tj.measure == "participant_minus_observer_margin").any()
    fig, axes = plt.subplots(4 if has_obs else 3, 1, figsize=(7.2, 7.6 if has_obs else 6.0), sharex=True, constrained_layout=True)
    ax = axes[0]
    for grp, col in (("defectors", DEF), ("stayers", STAY)):
        band(ax, "agrees_final", grp, col, "-")
        band(ax, "conf01", grp, col, (0, (1.0, 1.4)))
    ax.set_ylim(0.55, 1.03)
    ax.set_ylabel("Public (0-1)")
    ax.set_title("What they say: verdict = final consensus (solid), stated CONFIDENCE / 100 (dotted)", loc="left")
    gap_note(ax, "conf01", "{:+.3f}")
    ax = axes[1]
    for grp, col in (("defectors", DEF), ("stayers", STAY)):
        band(ax, "margin", grp, col)
        if (Tj.measure == "obs_margin").any():
            g = Tj[(Tj.measure == "obs_margin") & (Tj.group == grp)].sort_values("turn")
            ax.plot(g.turn, g["mean"], color=col, lw=1.2, ls=(0, (3, 2)), alpha=0.9)
    if (Tj.measure == "obs_margin").any():
        ax.text(0.16, 0.22, "dashed: an observer reading the identical text (same letters)", transform=ax.transAxes, fontsize=7.8,
                color=style.SLATE, va="bottom")
    ax.set_ylabel("Internal margin\n(logits)")
    ax.set_title("What the forward pass encodes, never printed: lp(own verdict) - lp(best alternative)", loc="left")
    gap_note(ax, "margin")
    ax = axes[2]
    for grp, col in (("defectors", DEF), ("stayers", STAY)):
        band(ax, "probe_dom", grp, col)
    ax.set_ylabel("Correctness probe\n(DoM score, a.u.)")
    ax.set_title("Frozen content-correctness probe (L19; fit on CPT worlds, never on debates)", loc="left")
    gap_note(ax, "probe_dom", "{:+.0f}")
    if has_obs:
        ax = axes[3]
        for grp, col in (("defectors", DEF), ("stayers", STAY)):
            band(ax, "participant_minus_observer_margin", grp, col)
        ax.axhline(0, color=style.MUTED, lw=0.7, ls=(0, (2, 2)))
        ax.set_ylabel("Participant - observer\nmargin (logits)")
        ax.set_title("Same transcript, different reader: participant minus observer margin (minimal framing edit)", loc="left")
        gap_note(ax, "participant_minus_observer_margin", ypos=0.30)
    ax = axes[-1]
    ax.set_xlabel("Debate turn")
    ax.set_xticks(range(0, last + 1))
    n_def = int(Tj[(Tj.measure == "margin") & (Tj.group == "defectors") & (Tj.turn == last)].n.iloc[0])
    n_st = int(Tj[(Tj.measure == "margin") & (Tj.group == "stayers") & (Tj.turn == last)].n.iloc[0])
    handles = [Line2D([], [], color=DEF, lw=1.8, marker="o", ms=3.0, label=f"eventual private defectors (n={n_def})"),
               Line2D([], [], color=STAY, lw=1.8, marker="o", ms=3.0, label=f"stayers (n={n_st})")]
    axes[0].legend(handles=handles, loc="center left", bbox_to_anchor=(0.0, 0.42), ncol=1)
    style.finish(fig, axes, str(FIGS / f"V3_C4_trajectory_{tag}"))

    # Figure 2: calibration (stated vs internal) + predictive information by turn
    fig = plt.figure(figsize=(7.2, 2.9), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.72, 1.12, 1.3])
    axes = [fig.add_subplot(gs[0, 0])]
    axes.append(fig.add_subplot(gs[0, 1], sharey=axes[0]))
    axes.append(fig.add_subplot(gs[0, 2]))
    for ax, (measure, title, xl, col) in zip(axes[:2], (("stated_confidence", "Stated", "stated CONFIDENCE", style.SLATE),
                                                        ("internal_p", "Internal p(own verdict)", "internal", style.PLUM))):
        g = Cal[(Cal.cond == "debate") & (Cal.measure == measure)].reset_index(drop=True)
        g = g[g.n_agents >= 10].reset_index(drop=True)
        x = np.arange(len(g))
        ax.bar(x, g.switch_rate, width=0.64, color=col, edgecolor="white", linewidth=1.0)
        ax.errorbar(x, g.switch_rate, yerr=[g.switch_rate - g.lower, g.upper - g.switch_rate], fmt="none", ecolor=style.INK, lw=0.8,
                    capsize=2.2)
        for xi, r in g.iterrows():
            ax.text(xi, r.upper + 0.008, f"{r.share*100:.0f}%\n{r.n_switch}/{r.n_agents}", ha="center", va="bottom", fontsize=7.2,
                    color=style.SLATE, linespacing=1.1)
        short = {"90-99": "90-99", "100": "100", "<0.9": "<.9", "0.9-0.99": ".9-.99", "0.99-0.999": ".99-.999", ">=0.999": "\u2265.999"}
        ax.set_xticks(x, [short.get(b_, b_) for b_ in g.bin], fontsize=8)
        ax.set_xlabel(f"{xl} bin", fontsize=8.5)
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Private switch rate\n(label: % of agents, switches/n)")
    cd = Cal[(Cal.cond == "debate") & (Cal.n_agents >= 10)]
    axes[0].set_ylim(0, float(cd.upper.max()) + 0.07)
    plt.setp(axes[1].get_yticklabels(), visible=False)
    ax = axes[2]
    ends = {}
    for pred, col, lab in (("neg_margin", style.PLUM, "internal margin"), ("neg_probe_dom", style.TERRACOTTA, "correctness probe"),
                           ("neg_conf", style.SLATE, "stated confidence")):
        g = ByTurn[(ByTurn.predictor == pred) & (ByTurn.scale == "family_adjusted")].sort_values("turn")
        ax.fill_between(g.turn, g.lower, g.upper, color=col, alpha=0.08, lw=0)
        ax.plot(g.turn, g.auroc, color=col, lw=1.7, marker="o", ms=2.8)
        ends[lab] = [g.turn.iloc[-1], g.auroc.iloc[-1], col]
    ys = sorted(ends.items(), key=lambda kv: kv[1][1])
    for k in range(1, len(ys)):  # keep end labels >= 0.022 apart
        if ys[k][1][1] - ys[k - 1][1][1] < 0.022:
            ys[k][1][1] = ys[k - 1][1][1] + 0.022
    if (ByTurn.predictor == "obs_neg_margin").any():
        g = ByTurn[(ByTurn.predictor == "obs_neg_margin") & (ByTurn.scale == "family_adjusted")].sort_values("turn")
        ax.fill_between(g.turn, g.lower, g.upper, color=style.PLUM, alpha=0.06, lw=0)
        ax.plot(g.turn, g.auroc, color=style.PLUM, lw=1.3, ls=(0, (3, 2)), marker="o", ms=2.2, mfc="white")
        ys.append(("observer margin (same text)", [g.turn.iloc[-1], g.auroc.iloc[-1], style.PLUM]))
        ys = sorted(ys, key=lambda kv: kv[1][1])
        for k in range(1, len(ys)):
            if ys[k][1][1] - ys[k - 1][1][1] < 0.022:
                ys[k][1][1] = ys[k - 1][1][1] + 0.022
        ax.set_xlim(-0.4, last + 5.6)
    for lab, (x_, y_, col) in ys:
        ax.text(x_ + 0.35, y_, lab, color=style.INK, fontsize=7.8, va="center")
    ax.axhline(0.5, color=style.MUTED, lw=0.8, ls=(0, (2, 2)))
    ax.set_xlim(-0.4, last + 4.2)
    ax.set_xticks([0, 3, 6, 9, last])
    ax.set_xlabel("turn at which the predictor is read")
    ax.set_ylabel("AUROC for later private switch\n(family-adjusted)")
    ax.set_title("Predictive information by turn", loc="left", fontsize=9.5)
    style.finish(fig, axes, str(FIGS / f"V3_C4_calibration_{tag}"))


if __name__ == "__main__":
    main()
