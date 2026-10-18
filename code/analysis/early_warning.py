"""C4 latent early warning of false consensus over 12-turn debates.

Unit: release_600 item with a target; y = debate majority at the final turn is wrong (ties count as not-wrong-majority and are
excluded). Features from turns <= k only (k = 0..6):
  public      agreement and mean confidence per turn, #verdict changes, unanimity at k
  transcript  TF-IDF of all public messages at turns <= k (fit inside training folds)
  activation  mean decision-site state of the 3 agents at turn k, PCA-64 inside folds
  concept     validated concept z (decision + reading sites) at turn k: agent mean and agent spread; plus change since turn 0
  cpt         item CPT sensitivities Q, P, IQP, S (controlled worlds; no debate information)
Evaluation: leave-one-family-out within model; cross-model transfer (fit on source model, score target model) for the
public and concept sets using concept names shared by both models' validated atlases. Metrics: AUROC, AURC, accuracy at 80%
coverage; paired cluster bootstrap of AURC gain vs public-only.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import *  # noqa
from monitor import risk_coverage

warnings.filterwarnings("ignore")


def build(model, k, turns=12, replay="replay_release_600_release_600_T12", dataset="release_600"):
    core = {r["id"]: r for r in load_core()}
    P = pd.read_parquet(TABLES / f"concept_traj_{model}_{replay}.parquet")
    V = pd.read_csv(TABLES / f"concept_validation_{model}.csv")
    valid = list(V[V.passes].concept)
    last = turns - 1
    deb = P[P.cond.isin(["t0", "debate"])]
    y = {}
    for iid, g in deb[deb.turn == last].groupby("item_id"):
        v = item_view(core[iid])
        if v["target"] is None or v["kind"] == "deception":
            continue
        vs = [x for x in g.verdict if isinstance(x, str)]
        if len(vs) < 3:
            continue
        counts = pd.Series(vs).value_counts()
        if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
            continue
        maj = counts.index[0]
        ok = numeric_equal(maj, v["target"]) if v["kind"] == "numeric" else maj == v["target"]
        y[iid] = int(not ok)
    items = sorted(y)
    D = pd.DataFrame(index=items)
    D["y"] = [y[i] for i in items]
    D["family"] = [family_of(core[i]) for i in items]
    D["cluster"] = [core[i]["source_item_id"] for i in items]
    early = deb[(deb.turn <= k) & deb.item_id.isin(items)]
    for t in range(k + 1):
        g = early[early.turn == t].groupby("item_id")
        D[f"pub_agree_t{t}"] = g.verdict.agg(lambda s: s.value_counts().iloc[0] / 3 if s.notna().any() else np.nan)
        D[f"pub_conf_t{t}"] = g.confidence.mean()
    ch = early.sort_values("turn").groupby(["item_id", "agent"]).verdict.apply(lambda s: (s != s.shift()).iloc[1:].sum())
    D["pub_changes"] = ch.groupby("item_id").sum().reindex(items).fillna(0)
    at_k = early[early.turn == k].groupby("item_id")
    at_0 = early[early.turn == 0].groupby("item_id")
    for c in valid:
        for site in ("dec", "read"):
            col = f"{site}_{c}"
            D[f"con_{col}_mean"] = at_k[col].mean()
            D[f"con_{col}_spread"] = at_k[col].std()
            D[f"con_{col}_delta"] = at_k[col].mean() - at_0[col].mean()
    cpt_p = TABLES / f"cpt_{model}_cpt_core_m_adv_items.csv"
    if cpt_p.exists():
        C = pd.read_csv(cpt_p).set_index("item_id")
        for c in ("Q", "P", "IQP", "S"):
            D[f"cpt_{c}"] = C[c].reindex(items)
    texts = {}
    for t in range(k + 1):
        f = RAW / model / dataset / ("t0.jsonl" if t == 0 else f"debate_t{t}.jsonl")
        for x in read_jsonl(f):
            if x["agent"] < 3 and x["item_id"] in y:
                texts.setdefault(x["item_id"], []).append(x["text"])
    D["transcript"] = [" \n ".join(texts.get(i, [])) for i in items]
    rep = ACTS / model / replay
    meta = json.load(open(rep / "meta.json"))
    A = np.load(rep / "acts.npy", mmap_mode="r")
    gi = int(np.argmin([abs(g / max(meta["grid"]) - 0.6) for g in meta["grid"]]))
    rows = early[early.turn == k].groupby("item_id").row.apply(list)
    X_act = np.stack([A[sorted(rows[i]), 0, gi].astype(np.float32).mean(0) for i in items])
    return D, X_act


def fit_score(kind, D, X_act, tr, te):
    y = D.y.to_numpy()
    pub = [c for c in D.columns if c.startswith("pub_")]
    con = [c for c in D.columns if c.startswith("con_")]
    cpt = [c for c in D.columns if c.startswith("cpt_")]
    if kind == "transcript":
        m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000), LogisticRegression(C=1.0, max_iter=3000))
        m.fit(D.transcript.iloc[tr], y[tr])
        return m.predict_proba(D.transcript.iloc[te])[:, 1]
    if kind == "activation":
        m = make_pipeline(StandardScaler(), PCA(64, random_state=0), LogisticRegression(C=0.1, max_iter=3000))
        m.fit(X_act[tr], y[tr])
        return m.predict_proba(X_act[te])[:, 1]
    prereg = json.load(open(ROOT / "data/concepts/taxonomy.json"))["prereg_debate"]
    con_pre = [c for c in con if any(c.startswith(f"con_{site}_{p}_") for p in prereg for site in ("dec", "read"))]
    cols = {"public": pub, "concept": con, "concept+public": pub + con, "cpt": cpt, "all_latent+public": pub + con + cpt,
            "concept_prereg+public": pub + con_pre}[kind]
    if not cols:
        return np.full(len(te), np.nan)
    X = D[cols].astype(float)
    X = X.fillna(X.iloc[tr].median()).to_numpy()
    C = 0.05 if len(cols) > 50 else 0.5
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, penalty="l2", max_iter=3000))
    m.fit(X[tr], y[tr])
    return m.predict_proba(X[te])[:, 1]


def evaluate(model, horizons, turns=12, replay="replay_release_600_release_600_T12", dataset="release_600"):
    kinds = ["public", "transcript", "activation", "concept", "concept+public", "concept_prereg+public", "cpt", "all_latent+public"]
    rows = []
    rng = np.random.default_rng(240091)
    for k in horizons:
        D, X_act = build(model, k, turns, replay, dataset)
        preds = {kd: np.full(len(D), np.nan) for kd in kinds}
        fams = D.family.to_numpy()
        for f in np.unique(fams):
            te = np.where(fams == f)[0]
            tr = np.where(fams != f)[0]
            if D.y.iloc[tr].nunique() < 2:
                continue
            for kd in kinds:
                preds[kd][te] = fit_score(kd, D, X_act, tr, te)
        ok = np.isfinite(preds["public"])
        yv = D.y.to_numpy()[ok]
        cl = D.cluster.to_numpy()[ok]
        uc = np.unique(cl)
        idx_of = {c: np.where(cl == c)[0] for c in uc}
        for kd in kinds:
            p = preds[kd][ok]
            if not np.isfinite(p).all():
                continue
            _, _, aurc, a80 = risk_coverage(p, 1 - yv)
            gains = []
            for _ in range(500):
                ii = np.concatenate([idx_of[c] for c in rng.choice(uc, len(uc))])
                gains.append(risk_coverage(preds["public"][ok][ii], 1 - yv[ii])[2] - risk_coverage(p[ii], 1 - yv[ii])[2])
            rows.append(dict(model=model, horizon=k, monitor=kd, auroc=roc_auc_score(yv, p) if len(set(yv)) > 1 else np.nan,
                             aurc=aurc, acc_at_80=a80, base_rate_wrong=float(yv.mean()), n=int(ok.sum()),
                             aurc_gain_vs_public=float(np.mean(gains)), gain_lower=float(np.quantile(gains, 0.025)),
                             gain_upper=float(np.quantile(gains, 0.975))))
        print(f"[horizon {k}] done", flush=True)
    return pd.DataFrame(rows)


def cross_model(src, tgt, k):
    Ds, _ = build(src, k)
    Dt, _ = build(tgt, k)
    out = []
    for kind, prefix in (("public", "pub_"), ("concept", "con_"), ("concept+public", None)):
        cols = [c for c in Ds.columns if (c.startswith("pub_") or c.startswith("con_")) and c in Dt.columns] if prefix is None else \
            [c for c in Ds.columns if c.startswith(prefix) and c in Dt.columns]
        if not cols:
            continue
        Xs = Ds[cols].astype(float).fillna(Ds[cols].median()).to_numpy()
        Xt = Dt[cols].astype(float).fillna(Ds[cols].median()).to_numpy()
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05 if len(cols) > 50 else 0.5, max_iter=3000))
        m.fit(Xs, Ds.y)
        p = m.predict_proba(Xt)[:, 1]
        _, _, aurc, a80 = risk_coverage(p, 1 - Dt.y.to_numpy())
        out.append(dict(source=src, target=tgt, horizon=k, monitor=kind, n_features=len(cols), aurc=aurc, acc_at_80=a80,
                        auroc=roc_auc_score(Dt.y, p) if Dt.y.nunique() > 1 else np.nan, n=len(Dt)))
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--horizons", default="0,1,2,3,4,6")
    ap.add_argument("--transfer_to", default="")
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--replay", default="replay_release_600_release_600_T12")
    ap.add_argument("--dataset", default="release_600")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    R = evaluate(a.model, [int(x) for x in a.horizons.split(",")], a.turns, a.replay, a.dataset)
    R.to_csv(TABLES / f"early_warning_{a.model}{a.tag}.csv", index=False)
    pd.set_option("display.width", 220)
    print(R.round(3).to_string())
    for tgt in [t for t in a.transfer_to.split(",") if t]:
        X = pd.concat([cross_model(a.model, tgt, k) for k in (1, 3)])
        X.to_csv(TABLES / f"early_warning_transfer_{a.model}_to_{tgt}.csv", index=False)
        print(X.round(3).to_string())
