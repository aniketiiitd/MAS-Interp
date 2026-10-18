"""E22 held-out consensus monitor + E23 leave-one-family-out transfer audit.

Unit: release_600 item with a scorable target (majority verdict at the final debate turn). y = final majority wrong/unsafe.
Feature sets (all computed without the final label):
  public      final/turn-0 agreement, unanimity, mean confidence, #revisions, turn-0 accuracy proxy (none: labels unused)
  transcript  TF-IDF of the final-turn public messages (fit inside training folds)
  activation  final-turn decision-boundary states (mean of 3 agents), PCA-64 fit inside training folds
  cpt         item CPT sensitivities (Q, P, IQP, S from controlled worlds) + frozen probe scores on final debate states
  cpt+public  union
Evaluation: leave-one-family-out; predictions pooled over held-out families; risk-coverage curves (accept lowest risk first),
AURC, accuracy at 80% coverage; cluster bootstrap CIs on AURC differences vs public-only.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import *  # noqa

POS = ["answer", "msg_end", "src_end", "arg_mid", "prior_end"]


def risk_coverage(risk, correct):
    order = np.argsort(risk)
    c = np.asarray(correct, float)[order]
    cov = np.arange(1, len(c) + 1) / len(c)
    acc = np.cumsum(c) / np.arange(1, len(c) + 1)
    aurc = float(np.mean(1 - acc))
    k80 = int(np.ceil(0.8 * len(c))) - 1
    return cov, acc, aurc, float(acc[k80])


def build(model, dataset, replay):
    ep = pd.read_csv(TABLES / f"{model}_{dataset}_turns.csv")
    items = pd.read_csv(TABLES / f"{model}_{dataset}_items.csv")
    rev = pd.read_csv(TABLES / f"{model}_{dataset}_revisions.csv")
    splits = json.load(open(ROOT / "data/splits/splits.json"))
    keep = set(splits["release_600"])
    last = ep.turn.max()
    deb = ep[(ep.cond == "debate") & ep.item_id.isin(keep)]
    fin = deb[deb.turn == last].set_index("item_id")
    t0 = deb[deb.turn == 0].set_index("item_id")
    df = pd.DataFrame(index=fin.index)
    df["y_wrong"] = fin.majority_correct.map({True: 0, False: 1, "True": 0, "False": 1})
    df = df[df.y_wrong.notna()]
    df["family"] = fin.family
    df["domain"] = fin.domain
    df["cluster"] = fin.cluster
    df["pub_agree_final"] = fin.agreement
    df["pub_unanimous"] = fin.unanimous.astype(float)
    df["pub_conf_final"] = fin.mean_conf
    df["pub_agree_t0"] = t0.agreement
    df["pub_conf_t0"] = t0.mean_conf
    r = rev[(rev.cond == "debate")].groupby("item_id").changed.sum()
    df["pub_revisions"] = r.reindex(df.index).fillna(0)
    # transcript
    msgs = {}
    for x in read_jsonl(RAW / model / dataset / f"debate_t{last}.jsonl"):
        msgs.setdefault(x["item_id"], []).append(x["text"])
    df["transcript"] = [" \n ".join(msgs.get(i, [])) for i in df.index]
    # activations: final-turn debate states
    rd = ACTS / model / replay
    idx = pd.DataFrame(read_jsonl(rd / "index.jsonl"))
    meta = json.load(open(rd / "meta.json"))
    A = np.load(rd / "acts.npy", mmap_mode="r")
    gi = int(np.argmin([abs(g / max(meta["grid"]) - 0.6) for g in meta["grid"]]))
    sel = idx[(idx.cond == "debate") & (idx.turn == last)]
    A = A[:, 0] if A.ndim == 4 else A
    act = {iid: A[g.row.to_numpy(), gi].astype(np.float32).mean(0) for iid, g in sel.groupby("item_id")}
    df = df[df.index.isin(act.keys())]
    X_act = np.stack([act[i] for i in df.index])
    # CPT item sensitivities
    cpt_items = pd.read_csv(TABLES / f"cpt_{model}_cpt_core_m_adv_items.csv").set_index("item_id")
    for c in ("Q", "P", "IQP", "S", "C", "ICP"):
        df[f"cpt_{c}"] = cpt_items[c].reindex(df.index) if c in cpt_items else np.nan
    # frozen probe scores on final debate states (probe site layer; position=answer boundary proxy)
    fz = json.load(open(ACTS / model / "probes_frozen_cpt_core.json"))
    for name in ("provenance", "social_disagreement", "evidence_quality", "revision_readiness"):
        if name not in fz:
            continue
        f = fz[name]
        gj = meta["grid"].index(f["layer"]) if f["layer"] in meta["grid"] else gi
        w = np.array(f["w_dom"], np.float32) / np.array(f["sd"], np.float32)
        sc = {iid: float((A[g.row.to_numpy(), gj].astype(np.float32) @ w).mean()) for iid, g in sel.groupby("item_id")}
        df[f"cpt_probe_{name}"] = [sc[i] for i in df.index]
    return df, X_act


def fit_predict(kind, tr, te, df, X_act):
    y = df.y_wrong.to_numpy().astype(int)
    pub = [c for c in df.columns if c.startswith("pub_")]
    cpt = [c for c in df.columns if c.startswith("cpt_")]
    if kind == "transcript":
        m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000), LogisticRegression(C=1.0, max_iter=2000))
        m.fit(df.transcript.iloc[tr], y[tr])
        return m.predict_proba(df.transcript.iloc[te])[:, 1]
    if kind == "activation":
        m = make_pipeline(StandardScaler(), PCA(64, random_state=0), LogisticRegression(C=0.1, max_iter=2000))
        m.fit(X_act[tr], y[tr])
        return m.predict_proba(X_act[te])[:, 1]
    cols = {"public": pub, "cpt": cpt, "cpt+public": pub + cpt}[kind]
    X = df[cols].fillna(df[cols].iloc[tr].median()).to_numpy(float)
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    m.fit(X[tr], y[tr])
    return m.predict_proba(X[te])[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="core")
    ap.add_argument("--replay", default="replay_core_release_600")
    a = ap.parse_args()
    df, X_act = build(a.model, a.dataset, a.replay)
    kinds = ["public", "transcript", "activation", "cpt", "cpt+public"]
    preds = {k: np.full(len(df), np.nan) for k in kinds}
    fams = df.family.to_numpy()
    for f in np.unique(fams):
        te = np.where(fams == f)[0]
        tr = np.where(fams != f)[0]
        if df.y_wrong.iloc[tr].nunique() < 2:
            continue
        for k in kinds:
            preds[k][te] = fit_predict(k, tr, te, df, X_act)
    ok = np.isfinite(preds["public"])
    correct = 1 - df.y_wrong.to_numpy()[ok]
    rows, curves = [], []
    rng = np.random.default_rng(240091)
    clusters = df.cluster.to_numpy()[ok]
    uc = np.unique(clusters)
    for k in kinds:
        cov, acc, aurc, a80 = risk_coverage(preds[k][ok], correct)
        boots = []
        for _ in range(500):
            pick = rng.choice(uc, len(uc))
            ii = np.concatenate([np.where(clusters == c)[0] for c in pick])
            _, _, au_k, _ = risk_coverage(preds[k][ok][ii], correct[ii])
            _, _, au_p, _ = risk_coverage(preds["public"][ok][ii], correct[ii])
            boots.append(au_p - au_k)
        rows.append(dict(monitor=k, aurc=aurc, acc_at_80=a80, base_acc=float(correct.mean()), aurc_gain_vs_public=float(np.mean(boots)),
                         gain_lower=float(np.quantile(boots, 0.025)), gain_upper=float(np.quantile(boots, 0.975)), n=int(ok.sum())))
        curves += [dict(monitor=k, coverage=c, accuracy=v) for c, v in zip(cov[::max(1, len(cov) // 200)], acc[::max(1, len(cov) // 200)])]
    R = pd.DataFrame(rows)
    R.to_csv(TABLES / f"monitor_{a.model}.csv", index=False)
    pd.DataFrame(curves).to_csv(TABLES / f"monitor_curves_{a.model}.csv", index=False)
    # per held-out family AURC (E23 transfer audit)
    fam_rows = []
    for f in np.unique(fams[ok]):
        m = fams[ok] == f
        if correct[m].min() == correct[m].max():
            continue
        for k in kinds:
            _, _, au, a80 = risk_coverage(preds[k][ok][m], correct[m])
            fam_rows.append(dict(family=f, monitor=k, aurc=au, acc_at_80=a80, n=int(m.sum()), base_acc=float(correct[m].mean())))
    pd.DataFrame(fam_rows).to_csv(TABLES / f"monitor_by_family_{a.model}.csv", index=False)
    print(R.round(4).to_string())


if __name__ == "__main__":
    main()
