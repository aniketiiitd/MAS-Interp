"""V3-C3(iv)/(v): participation-inflated commitment on identical text, with the paraphrase reader as the wording null.

Per agent-turn: margin toward the agent's own public verdict at that turn from three forward passes on the same transcript:
participant (replay), observer (reader_obsmin), paraphrased participant (reader_para). dO = mP - mO, dQ = mP - mQ.
Final-turn publicly unanimous agents with a private verdict: switch = private != public.
Gates (PLAN_V2.md A1/A2): (a) mean dO - dQ > 0; (b)/(v) family-standardised defectors - stayers of dO - dQ > 0.
Usage: python analysis/commit_gap.py --model Q9 --dataset release_600 --turns 12
       python analysis/commit_gap.py --model Q9 --dataset core --turns 4 --private aud
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
from stats import fast_cluster_mean_ci

CORE = {r["id"]: r for r in load_core()}
cluster_of = lambda i: CORE[i]["meta"].get("analysis_group") if CORE[i]["stratum"] == "persuasion" else CORE[i]["source_item_id"]


def margin(lp, v):
    if not isinstance(lp, dict) or v not in lp or len(lp) < 2:
        return np.nan
    return lp[v] - max(x for k, x in lp.items() if k != v)


def fam_gap(df, col):
    num = den = 0.0
    for _, g in df.groupby("family"):
        s = g.switch.to_numpy(bool)
        if s.sum() == 0 or (~s).sum() == 0:
            continue
        num += s.sum() * (g[col].to_numpy()[s].mean() - g[col].to_numpy()[~s].mean())
        den += s.sum()
    return num / den if den else np.nan


def boot_fam_gap(df, col, B=1000, seed=240091):
    by = {c: g for c, g in df.groupby("cluster")}
    keys = list(by)
    rng = np.random.default_rng(seed)
    est = fam_gap(df, col)
    bs = [fam_gap(pd.concat([by[keys[i]] for i in rng.integers(0, len(keys), len(keys))]), col) for _ in range(B)]
    return dict(est=float(est), lower=float(np.nanquantile(bs, .025)), upper=float(np.nanquantile(bs, .975)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="release_600")
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--private", default="generated", choices=["generated", "aud"])
    a = ap.parse_args()
    ds, T = a.dataset, a.turns
    rep = f"replay_{ds}_{ds}_T{T}" if ds != "core" else f"replay_core_all_T{T}"
    load = lambda d: pd.DataFrame(read_jsonl(ACTS / a.model / d / "index.jsonl"))
    P = load(rep)
    P = P[P.cond == "debate"][["item_id", "agent", "turn", "lp_letters"]]
    O = load(f"reader_obsmin_{ds}_T{T}")[["item_id", "agent", "turn", "lp_letters"]]
    Q = load(f"reader_para_{ds}_T{T}")[["item_id", "agent", "turn", "lp_letters"]]
    M = P.merge(O, on=["item_id", "agent", "turn"], suffixes=("_P", "_O")).merge(Q.rename(columns={"lp_letters": "lp_letters_Q"}),
                                                                              on=["item_id", "agent", "turn"])
    pub = {}
    for t in range(1, T):
        for x in read_jsonl(RAW / a.model / ds / f"debate_t{t}.jsonl"):
            if x["agent"] < 3:
                pub[(x["item_id"], x["agent"], t)] = x["verdict"]
    M["v"] = [pub.get(k) for k in zip(M.item_id, M.agent, M.turn)]
    M = M[M.v.notna()].copy()
    for s in "POQ":
        M["m" + s] = [margin(lp, v) for lp, v in zip(M["lp_letters_" + s], M.v)]
    M = M.dropna(subset=["mP", "mO", "mQ"]).drop(columns=["lp_letters_P", "lp_letters_O", "lp_letters_Q"])
    M["dO"], M["dQ"] = M.mP - M.mO, M.mP - M.mQ
    M["dOQ"] = M.dO - M.dQ
    M["cluster"] = M.item_id.map(cluster_of)
    M["family"] = M.item_id.map(lambda i: family_of(CORE[i]))
    M["domain"] = M.item_id.map(lambda i: CORE[i]["primary_domain"])
    fin = M[M.turn == T - 1].copy()
    un = fin.groupby("item_id").v.agg(lambda s: s.nunique() == 1 and len(s) == 3)
    fin = fin[fin.item_id.map(un)]
    if a.private == "generated":
        pr = {(x["item_id"], x["agent"]): x["verdict"] for x in read_jsonl(RAW / a.model / ds / "private.jsonl") if x["cond"] == "private_debate"}
    else:
        d = next(p for p in (ACTS / a.model / f"frames_audience_{ds}_notrelease_600", ACTS / a.model / f"frames_audience_{ds}") if p.exists())
        pr = {(x["item_id"], x["agent"]): max(x["lp_letters"], key=x["lp_letters"].get)
              for x in read_jsonl(d / "index.jsonl") if x["frame"] == "priv_ctx" and x["cond"] == "debate"}
    fin["priv"] = [pr.get(k) for k in zip(fin.item_id, fin.agent)]
    n_missing = int(fin.priv.isna().sum())
    fin = fin[fin.priv.notna()].copy()
    fin["switch"] = fin.priv != fin.v
    ci = lambda col, df=fin: dict(zip(("est", "lower", "upper", "p", "n_clusters"), map(float, fast_cluster_mean_ci(df[col].to_numpy(), df.cluster.to_numpy()))))
    out = dict(model=a.model, dataset=ds, turns=T, private=a.private, n_final_unanimous=len(fin), n_switch=int(fin.switch.sum()),
               n_missing_private=n_missing,
               mean_dO=ci("dO"), mean_dQ=ci("dQ"), mean_dO_minus_dQ=ci("dOQ"),
               defstay_dO=boot_fam_gap(fin, "dO"), defstay_dQ=boot_fam_gap(fin, "dQ"), defstay_dOQ=boot_fam_gap(fin, "dOQ"),
               defstay_mP=boot_fam_gap(fin, "mP"), defstay_mO=boot_fam_gap(fin, "mO"), defstay_mQ=boot_fam_gap(fin, "mQ"))
    out["PASS_a_mean_beyond_para"] = bool(out["mean_dO_minus_dQ"]["lower"] > 0)
    out["PASS_v_defector_gap_beyond_para"] = bool(out["defstay_dOQ"]["lower"] > 0)
    dom = {}
    for k, g in fin.groupby("domain"):
        if g.switch.sum() >= 5 and (~g.switch).sum() >= 5:
            dom[k] = dict(n_switch=int(g.switch.sum()), raw_defstay_dOQ=float(g[g.switch].dOQ.mean() - g[~g.switch].dOQ.mean()))
    out["by_domain"] = dom
    lab = fin.set_index(["item_id", "agent"]).switch
    M2 = M.join(lab, on=["item_id", "agent"], how="inner")
    out["by_turn_defstay"] = {int(t): dict(dO=fam_gap(g, "dO"), dQ=fam_gap(g, "dQ")) for t, g in M2.groupby("turn")}
    tag = f"{a.model}_{ds}_T{T}"
    M.to_csv(TABLES / f"v3_commit_readers_{tag}.csv", index=False)
    fin.to_csv(TABLES / f"v3_commit_final_{tag}.csv", index=False)
    json.dump(out, open(TABLES / f"v3_commit_gate_{tag}.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
