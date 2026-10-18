"""Behavioural analyses of literature-format debates (outputs/raw_lit/<M>/<dataset>).

Hero 5 (debate vs vote on MAD Bench): per item, majority of the independent initial answers (vote), final debate majority and final
no-communication (reflect) majority; accuracy on items with a target; unanimity; wrong-and-unanimous; amplification of the initial
majority (outcome given that the initial majority was right / wrong / absent); by primary domain.
Hero 4 (who will hear): after the final round, private vs public re-query answer differs from the agent's own final answer;
debate vs reflect; publicly unanimous groups; by domain.
Tables: outputs/tables/lit_debate_<M>_<dataset>_*.csv
"""
from __future__ import annotations

import argparse
import collections
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
RAW_LIT = ROOT / "outputs/raw_lit"


def majority(answers):
    a = [x for x in answers if x is not None]
    if not a:
        return None
    c = collections.Counter(a).most_common()
    return c[0][0] if len(c) == 1 or c[0][1] > c[1][1] else None


def ci(x, c):
    e, lo, hi, p, n = fast_cluster_mean_ci(np.asarray(x, float), np.asarray(c))
    return dict(est=e, lower=lo, upper=hi, n_clusters=n, n=int(np.isfinite(np.asarray(x, float)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="lit1200")
    a = ap.parse_args()
    # The benchmark was run in two halves, so a full-benchmark number means reading both. Comma-separate them and the
    # halves are pooled; item ids are unique across them, so the records simply union.
    parts = [p for p in a.dataset.split(",") if p]
    dirs = [RAW_LIT / a.model / p for p in parts]
    missing = [str(x) for x in dirs if not x.exists()]
    if missing:
        raise SystemExit(f"[lit_debate] missing run directories: {missing}")
    metas = [json.load(open(x / "meta.json")) for x in dirs]
    rr = {m["rounds"] for m in metas}
    if len(rr) != 1:
        raise SystemExit(f"[lit_debate] the halves ran different round counts {rr}; refusing to pool them")
    R = rr.pop()
    recs = collections.defaultdict(dict)
    seen = set()
    for x_dir in dirs:
        for f in x_dir.glob("*.jsonl"):
            for x in read_jsonl(f):
                k = (x["item_id"], x["agent"])
                recs[(x["cond"], x["round"], x.get("query"))][k] = x
                seen.add(x["item_id"])
    if len(parts) > 1:
        print(f"[lit_debate] pooled {len(parts)} halves -> {len(seen)} distinct items", flush=True)
    items = sorted({i for (i, g) in recs[("r0", 0, None)]})
    tag = f"{a.model}_" + ("full2400" if len(parts) > 1 else a.dataset)
    rows = []
    for i in items:
        v = item_view(CORE[i])
        get = lambda c, k, q=None: [recs[(c, k, q)].get((i, g), {}).get("answer") for g in range(3)]
        r0 = get("r0", 0)
        row = dict(item_id=i, cluster=cluster_of(i), domain=CORE[i]["primary_domain"], family=family_of(CORE[i]), target=v["target"],
                   kind=v["kind"], vote=majority(r0), r0_unan=len(set(r0)) == 1 and None not in r0,
                   r0_parsed=sum(x is not None for x in r0) / 3)
        for c in ("debate", "reflect"):
            fin = get(c, R)
            row[f"{c}_maj"] = majority(fin)
            row[f"{c}_unan"] = len(set(fin)) == 1 and None not in fin
            row[f"{c}_parsed"] = sum(x is not None for x in fin) / 3
        rows.append(row)
    I = pd.DataFrame(rows)
    # Both arms must be measured on the same items or the comparison between them is partly a comparison of parse
    # rates. An unparsed answer was being scored as that arm getting it wrong, and the arms do not fail equally: the
    # vote is built from round-0 answers, which parse worst. Emit both scopes so a claim has to say which it quotes.
    I["fully_parsed"] = (I.debate_parsed == 1.0) & (I.reflect_parsed == 1.0) & I.r0_parsed.eq(1.0)
    T_all = I[I.target.notna() & (I.kind != "numeric")].copy()
    for c in ("vote", "debate_maj", "reflect_maj"):
        T_all[f"acc_{c}"] = (T_all[c] == T_all.target).astype(float)
    for c in ("debate", "reflect"):
        T_all[f"wrong_unan_{c}"] = (T_all[f"{c}_unan"] & (T_all[f"{c}_maj"] != T_all.target)).astype(float)
    T_all["vote_state"] = np.where(T_all.vote.isna(), "no majority", np.where(T_all.vote == T_all.target, "initial majority right", "initial majority wrong"))
    out = []
    for scope, T in (("all_items", T_all), ("fully_parsed", T_all[T_all.fully_parsed])):
      for dom, g in [("all", T)] + list(T.groupby("domain")):
          for m in ("acc_vote", "acc_debate_maj", "acc_reflect_maj", "wrong_unan_debate", "wrong_unan_reflect"):
              out.append(dict(scope=scope, domain=dom, metric=m, **ci(g[m], g.cluster)))
          for m, (x, y) in {"debate - vote": ("acc_debate_maj", "acc_vote"), "reflect - vote": ("acc_reflect_maj", "acc_vote"),
                            "debate - reflect": ("acc_debate_maj", "acc_reflect_maj"),
                            "wrong_unan debate - reflect": ("wrong_unan_debate", "wrong_unan_reflect")}.items():
              out.append(dict(scope=scope, domain=dom, metric=m, **ci(g[x] - g[y], g.cluster)))
          for st, gs in g.groupby("vote_state"):
              for m in ("acc_debate_maj", "acc_reflect_maj", "wrong_unan_debate", "wrong_unan_reflect"):
                  out.append(dict(scope=scope, domain=dom, metric=f"{m} | {st}", **ci(gs[m], gs.cluster)))
    # amplification on items without a target: final majority = initial majority
    NT = I[I.vote.notna()].copy()
    for c in ("debate", "reflect"):
        NT[f"keeps_{c}"] = (NT[f"{c}_maj"] == NT.vote).astype(float)
    for dom, g in [("all", NT)] + list(NT.groupby("domain")):
        split = g[~g.r0_unan]
        for c in ("debate", "reflect"):
            out.append(dict(scope="all_items", domain=dom, metric=f"split items: final majority = initial majority ({c})", **ci(split[f"keeps_{c}"], split.cluster)))
        out.append(dict(scope="all_items", domain=dom, metric="split items: keeps initial majority, debate - reflect", **ci(split.keeps_debate - split.keeps_reflect, split.cluster)))
    O = pd.DataFrame(out)
    O.to_csv(TABLES / f"lit_debate_{tag}_vote.csv", index=False)
    I.to_csv(TABLES / f"lit_debate_{tag}_items.csv", index=False)
    print(O[O.domain == "all"].round(3).to_string())
    # ---- audience re-query
    if ("debate", R + 1, "private") in recs:
        rr = []
        for c in ("debate", "reflect"):
            for (i, g), x in recs[(c, R + 1, "private")].items():
                pub_final = recs[(c, R, None)].get((i, g), {}).get("answer")
                y = recs[(c, R + 1, "public")].get((i, g), {})
                fin = [recs[(c, R, None)].get((i, h), {}).get("answer") for h in range(3)]
                rr.append(dict(item_id=i, agent=g, cond=c, cluster=cluster_of(i), domain=CORE[i]["primary_domain"],
                               final=pub_final, private=x["answer"], public=y.get("answer"),
                               unanimous=len(set(fin)) == 1 and None not in fin))
        Q = pd.DataFrame(rr)
        Q = Q[Q.final.notna() & Q.private.notna() & Q.public.notna()].copy()
        Q["dev_private"] = (Q.private != Q.final).astype(float)
        Q["dev_public"] = (Q.public != Q.final).astype(float)
        Q.to_csv(TABLES / f"lit_debate_{tag}_requery.csv", index=False)
        out = []
        for scope, g0 in (("unanimous", Q[Q.unanimous]), ("all", Q)):
            for dom, g in [("all", g0)] + list(g0.groupby("domain")):
                for c in ("debate", "reflect"):
                    gc = g[g.cond == c]
                    for m in ("dev_private", "dev_public"):
                        out.append(dict(scope=scope, domain=dom, cond=c, metric=m, **ci(gc[m], gc.cluster)))
                    out.append(dict(scope=scope, domain=dom, cond=c, metric="private - public", **ci(gc.dev_private - gc.dev_public, gc.cluster)))
                A = g.groupby(["item_id", "cluster", "cond"])[["dev_private", "dev_public"]].mean().unstack("cond").dropna()
                if len(A):
                    acl = A.index.get_level_values("cluster")
                    for m in ("dev_private", "dev_public"):
                        out.append(dict(scope=scope, domain=dom, cond="debate - reflect", metric=m, **ci(A[(m, "debate")] - A[(m, "reflect")], acl)))
                    dd = (A[("dev_private", "debate")] - A[("dev_private", "reflect")]) - (A[("dev_public", "debate")] - A[("dev_public", "reflect")])
                    out.append(dict(scope=scope, domain=dom, cond="debate - reflect", metric="private - public", **ci(dd, acl)))
        AQ = pd.DataFrame(out)
        AQ.to_csv(TABLES / f"lit_debate_{tag}_audience.csv", index=False)
        print(AQ[(AQ.domain == "all")].round(3).to_string())


if __name__ == "__main__":
    main()
