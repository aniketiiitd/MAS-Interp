"""Hero 4 internal side on literature-format debates: what the agents write vs the unprinted decision margin.

Per debate response (replay_lit): margin = lp(written answer) - max lp(other letters) at the answer token; say-features from the
text: hedging-word rate, answer changes so far, citation of agreement. Outcomes among publicly unanimous final groups: private
deviation (generated private re-query != final answer); item-level wrong unanimous consensus.
Predictors are compared with the strongest transcript model (best of TF-IDF on the final response and a public-history logistic),
family-adjusted AUROC, item-cluster bootstrap; the same margin read by an outside observer (frames_lit observer) tests whether the
signal is specific to the participant.
Tables: outputs/tables/lit_hollow_<M>_<dataset>_*.csv / .json
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
from common import *  # noqa
from hollow import auroc_table, fam_std_diff, oof_text_model

CORE = {r["id"]: r for r in load_core()}
cluster_of = lambda i: CORE[i]["meta"].get("analysis_group") if CORE[i]["stratum"] == "persuasion" else CORE[i]["source_item_id"]
RAW_LIT = ROOT / "outputs/raw_lit"
HEDGE = re.compile(r"\b(might|may|possibly|perhaps|likely|unlikely|uncertain|unclear|not sure|arguably|could|however|although|but)\b", re.I)
AGREE = re.compile(r"\b(agree|consensus|other agents|all agents|both agents|unanimous)\b", re.I)


def margin(lp, F):
    if not isinstance(lp, dict) or F not in lp or len(lp) < 2:
        return np.nan
    return lp[F] - max(v for k, v in lp.items() if k != F)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="lit1200")
    a = ap.parse_args()
    tag = f"{a.model}_{a.dataset}"
    # Pooling the halves is NOT supported here. Each half has its own acts.npy with row indices starting at zero, so
    # concatenating the index frames without renumbering would make half B's rows read half A's activations: a silent
    # data-mixing bug that produces plausible numbers. Refuse it until the renumbering is implemented and tested.
    if "," in a.dataset:
        raise SystemExit("[hollow] pooling halves is not implemented: row indices are per-half and would silently "
                         "cross-reference the wrong activation array. Run each half separately.")
    d = RAW_LIT / a.model / a.dataset
    R = json.load(open(d / "meta.json"))["rounds"]
    X = pd.DataFrame(read_jsonl(ACTS / a.model / f"lit_replay_{a.dataset}" / "index.jsonl"))
    texts = {}
    for f in d.glob("*.jsonl"):
        for x in read_jsonl(f):
            texts[(x["item_id"], x["cond"], x["agent"], x["round"], x.get("query"))] = x["text"]
    # the replay index is arrow-backed, so an absent `query` is pd.NA while the transcript keys use None; without
    # this normalisation every round-0..3 row silently fails to join and the transcript baseline is fitted on blanks
    nq = lambda q: None if q is None or q is pd.NA or (isinstance(q, float) and np.isnan(q)) else q
    X["text"] = [texts.get((i, c, g, k, nq(q))) for i, c, g, k, q in zip(X.item_id, X.cond, X.agent, X["round"], X["query"])]
    X["margin"] = [margin(lp, ans) for lp, ans in zip(X.lp_letters, X.answer)]
    # a missing transcript arrives as NaN, not None, and `t or ""` keeps NaN because NaN is truthy -- coerce by type
    astext = lambda t: t if isinstance(t, str) else ""
    n_missing = int((~X.text.map(lambda t: isinstance(t, str))).sum())
    print(f"[hollow] {len(X)} replay rows, {n_missing} without a matching transcript ({100*n_missing/max(1,len(X)):.2f}%)", flush=True)
    X["has_text"] = X.text.map(lambda t: isinstance(t, str))
    X["hedge_rate"] = X.text.map(lambda t: len(HEDGE.findall(astext(t))) / max(1, len(astext(t).split())) * 100)
    X["agree_cite"] = X.text.map(lambda t: float(bool(AGREE.search(astext(t)))))
    X["n_words"] = X.text.map(lambda t: len(astext(t).split()))
    X["cluster"] = X.item_id.map(cluster_of)
    X["family"] = X.item_id.map(lambda i: family_of(CORE[i]))
    X["domain"] = X.item_id.map(lambda i: CORE[i]["primary_domain"])
    deb = X[(X.cond.isin(["r0", "debate"])) & X["query"].isna()]
    hist = deb.pivot_table(index=["item_id", "agent"], columns="round", values="answer", aggfunc="first")
    changes = hist.apply(lambda r: sum(r.iloc[k] != r.iloc[k - 1] for k in range(1, len(r)) if pd.notna(r.iloc[k]) and pd.notna(r.iloc[k - 1])), axis=1)
    fin = deb[deb["round"] == R].copy()
    fin["n_changes"] = [changes.get((i, g), np.nan) for i, g in zip(fin.item_id, fin.agent)]
    un = fin.groupby("item_id").answer.agg(lambda s: s.nunique() == 1 and s.notna().all() and len(s) == 3)
    fin["unanimous"] = fin.item_id.map(un)
    priv = {(x["item_id"], x["agent"]): x["answer"] for x in read_jsonl(d / "requery.jsonl") if x["cond"] == "debate" and x["query"] == "private"}
    fin["private"] = [priv.get((i, g)) for i, g in zip(fin.item_id, fin.agent)]
    fr0 = ACTS / a.model / f"lit_frames_{a.dataset}" / "index.jsonl"
    reread = []
    if not fr0.exists():
        print(f"[hollow] WARNING: {fr0} is missing, so the fresh-read predictors (the written-token vs re-read "
              f"comparison) are NOT computed. This table is incomplete -- run the frames job first.", flush=True)
    if fr0.exists():
        F0 = pd.DataFrame(read_jsonl(fr0))
        F0 = F0[F0["round"] == int(F0["round"].max())]  # final-round family; "observer" also exists at round 1
        # Three re-reads of byte-identical text, differing only in who the prompt says is reading. `part` and
        # `part_para` are the wording controls: if they predict as well as `observer`, the asymmetry is about
        # re-reading a finished transcript, not about role.
        for frame, nm in (("observer", "neg_margin_observer"), ("part", "neg_margin_part"), ("part_para", "neg_margin_part_para")):
            mp = {(r.item_id, r.agent): r.lp_letters for r in F0[F0.frame == frame].itertuples()}
            if not mp:
                continue
            fin[nm] = [-margin(mp.get((i, g)), ans) for i, g, ans in zip(fin.item_id, fin.agent, fin.answer)]
            reread.append(nm)
        got = {f for f in ("observer", "part", "part_para") if f in set(F0.frame)}
        missing = {"observer", "part", "part_para"} - got
        if missing:
            print(f"[hollow] WARNING: frames arms missing from the index: {sorted(missing)} -- the role/wording "
                  f"controls on the re-read comparison will be absent.", flush=True)
        print(f"[hollow] fresh-read predictors built: {reread}", flush=True)
    U = fin[fin.unanimous & fin.private.notna() & fin.margin.notna()].copy().reset_index(drop=True)
    U["defect"] = (U.private != U.answer).astype(float)
    U["neg_margin"] = -U.margin
    U["neg_hedge"] = U.hedge_rate
    preds = ["neg_margin", "neg_hedge", "tfidf", "public_hist"] + list(reread)
    U["tfidf"] = oof_text_model(U.text.fillna("").tolist(), U.defect.to_numpy(), U.cluster.to_numpy())
    H = U[["n_changes", "hedge_rate", "agree_cite", "n_words"]].fillna(0).to_numpy(float)
    U["public_hist"] = oof_text_model(None, U.defect.to_numpy(), U.cluster.to_numpy(), kind="features", X=H)
    pairs = [("neg_margin", "tfidf"), ("neg_margin", "public_hist"), ("neg_margin", "neg_hedge")]
    for nm in ("neg_margin_observer", "neg_margin_part", "neg_margin_part_para"):
        if nm in U:
            pairs.append(("neg_margin", nm))
    if "neg_margin_observer" in U and "neg_margin_part_para" in U:
        pairs.append(("neg_margin_part_para", "neg_margin_observer"))  # the wording-controlled role contrast
    if "neg_margin_observer" in U and "neg_margin_part" in U:
        pairs.append(("neg_margin_part", "neg_margin_observer"))
    # the fresh re-read against the best transcript models: does the unprinted state beat what the agents wrote?
    for nm in ("neg_margin_part", "neg_margin_observer"):
        if nm in U:
            pairs += [(nm, "tfidf"), (nm, "public_hist"), (nm, "neg_hedge")]
    A1, D1 = auroc_table(U, preds, "defect", "cluster", ref_pairs=pairs)
    A1.to_csv(TABLES / f"lit_hollow_{tag}_auroc_defect.csv", index=False)
    D1.to_csv(TABLES / f"lit_hollow_{tag}_auroc_defect_diffs.csv", index=False)
    gaps = {c: fam_std_diff(U, c, "defect", "cluster") for c in ("margin", "hedge_rate", "agree_cite", "n_changes")}
    # wrong unanimous consensus (item level)
    items = fin[fin.unanimous].groupby("item_id").agg(answer=("answer", "first"), margin=("margin", "mean"), hedge_rate=("hedge_rate", "mean"),
                                                      text=("text", lambda s: " ".join(s.fillna(""))), cluster=("cluster", "first"),
                                                      family=("family", "first"), n_changes=("n_changes", "mean"),
                                                      agree_cite=("agree_cite", "mean"), n_words=("n_words", "mean"),
                                                      **{nm: (nm, "mean") for nm in reread}).reset_index()
    items["target"] = items.item_id.map(lambda i: item_view(CORE[i])["target"])
    items = items[items.target.notna() & items.margin.notna()].reset_index(drop=True)
    items["wrong"] = (items.answer != items.target).astype(float)
    items["neg_margin"] = -items.margin
    items["neg_hedge"] = items.hedge_rate
    items["tfidf"] = oof_text_model(items.text.tolist(), items.wrong.to_numpy(), items.cluster.to_numpy())
    items["public_hist"] = oof_text_model(None, items.wrong.to_numpy(), items.cluster.to_numpy(), kind="features",
                                          X=items[["n_changes", "hedge_rate", "agree_cite", "n_words"]].fillna(0).to_numpy(float))
    ip = ["neg_margin", "neg_hedge", "tfidf", "public_hist"] + [nm for nm in reread if items[nm].notna().any()]
    rp = [("neg_margin", "tfidf"), ("neg_margin", "public_hist")] + [("neg_margin", nm) for nm in reread]
    if "neg_margin_part" in ip and "neg_margin_observer" in ip:
        rp += [("neg_margin_part", "neg_margin_observer"), ("neg_margin_part", "tfidf")]
    A2, D2 = auroc_table(items, ip, "wrong", "cluster", ref_pairs=rp)
    A2.to_csv(TABLES / f"lit_hollow_{tag}_auroc_wrong.csv", index=False)
    D2.to_csv(TABLES / f"lit_hollow_{tag}_auroc_wrong_diffs.csv", index=False)
    summ = dict(n_unanimous_agents=len(U), n_defect=int(U.defect.sum()), n_items_target=len(items), n_wrong=int(items.wrong.sum()),
                defector_minus_stayer=gaps)
    json.dump(summ, open(TABLES / f"lit_hollow_{tag}_summary.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 250)
    print(json.dumps(summ, indent=1, default=float))
    print(A1[A1.scale == "family_adjusted"].round(3).to_string())
    print(D1[D1.scale == "family_adjusted"].round(3).to_string())
    print(A2[A2.scale == "family_adjusted"].round(3).to_string())
    print(D2[D2.scale == "family_adjusted"].round(3).to_string())


if __name__ == "__main__":
    main()
