"""Long-horizon behavior of 12-turn debates (C2 context, B2 candidate).

Per turn and condition: accuracy (target items), agreement, unanimity, revision rate, wrong-unanimity rate.
Consensus stability: among items unanimous at turn t (t>=2), the fraction that break unanimity later.
Private re-query (after the final turn, transcript retained): agent-level switch rate relative to its own final public
verdict and item-level public/private majority mismatch, split by final public unanimity. Item-cluster bootstrap CIs.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
from common import *  # noqa
from stats import fast_cluster_mean_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="release_600")
    ap.add_argument("--turns", type=int, default=12)
    a = ap.parse_args()
    core = {r["id"]: r for r in load_core()}
    d = RAW / a.model / a.dataset
    recs = []
    for f in ["t0"] + [f"{c}_t{t}" for c in ("debate", "pool") for t in range(1, a.turns)]:
        for x in read_jsonl(d / f"{f}.jsonl"):
            if x["agent"] < 3:
                x["file"] = f
                recs.append(x)
    df = pd.DataFrame(recs)
    t0 = df[df.file == "t0"]
    rows = []
    for cond in ("debate", "pool"):
        cur = pd.concat([t0.assign(cond=cond), df[df.cond == cond]]).sort_values(["item_id", "agent", "turn"])
        cur["prev"] = cur.groupby(["item_id", "agent"]).verdict.shift(1)
        for (iid, t), g in cur.groupby(["item_id", "turn"]):
            vs = [v for v in g.verdict if isinstance(v, str)]
            if len(vs) < 3:
                continue
            cnt = pd.Series(vs).value_counts()
            v = item_view(core[iid])
            maj = cnt.index[0] if (len(cnt) == 1 or cnt.iloc[0] > cnt.iloc[1]) else None
            maj_ok = None
            if maj is not None and v["target"] is not None and v["kind"] != "deception":
                maj_ok = numeric_equal(maj, v["target"]) if v["kind"] == "numeric" else maj == v["target"]
            rev = g[g.prev.notna()]
            rows.append(dict(item_id=iid, cluster=core[iid]["source_item_id"], domain=core[iid]["primary_domain"], cond=cond, turn=t,
                             agreement=cnt.iloc[0] / 3, unanimous=len(cnt) == 1,
                             acc=pd.to_numeric(g.correct, errors="coerce").mean(),
                             revision=(rev.verdict != rev.prev).mean() if len(rev) else np.nan,
                             wrong_unanimous=(len(cnt) == 1 and maj_ok is False)))
    E = pd.DataFrame(rows)
    E.to_csv(TABLES / f"long_{a.model}_{a.dataset}_turns.csv", index=False)
    summ = []
    for (cond, t), g in E.groupby(["cond", "turn"]):
        for c in ("agreement", "unanimous", "acc", "revision", "wrong_unanimous"):
            est, lo, hi, p, n = fast_cluster_mean_ci(g[c].astype(float).to_numpy(), g.cluster.to_numpy())
            summ.append(dict(cond=cond, turn=t, endpoint=c, estimate=est, lower=lo, upper=hi, n=n))
    S = pd.DataFrame(summ)
    S.to_csv(TABLES / f"long_{a.model}_{a.dataset}_summary.csv", index=False)
    # consensus stability: unanimous at t, not unanimous at some later turn
    stab = []
    for cond, G in E.groupby("cond"):
        P = G.pivot_table(index=["item_id", "cluster"], columns="turn", values="unanimous", aggfunc="first")
        for t in range(2, a.turns - 1):
            un = P[P[t] == True]  # noqa: E712
            if len(un) < 10:
                continue
            broke = (un[[c for c in P.columns if c > t]] == False).any(axis=1).astype(float)  # noqa: E712
            est, lo, hi, p, n = fast_cluster_mean_ci(broke.to_numpy(), un.index.get_level_values("cluster").to_numpy())
            stab.append(dict(cond=cond, from_turn=t, p_break_later=est, lower=lo, upper=hi, n=n))
    pd.DataFrame(stab).to_csv(TABLES / f"long_{a.model}_{a.dataset}_stability.csv", index=False)
    # private re-query
    pr_p = d / "private.jsonl"
    out = {}
    if pr_p.exists():
        pr = pd.DataFrame(read_jsonl(pr_p))
        last = a.turns - 1
        for cond in ("debate", "pool"):
            pub = df[(df.cond == cond) & (df.turn == last)].set_index(["item_id", "agent"]).verdict
            pv = pr[pr.cond == f"private_{cond}"].set_index(["item_id", "agent"]).verdict
            J = pd.DataFrame(dict(pub=pub, priv=pv)).dropna().reset_index()
            J["switch"] = (J.pub != J.priv).astype(float)
            un = J.groupby("item_id").pub.agg(lambda s: s.nunique() == 1).rename("pub_unanimous")
            J = J.join(un, on="item_id")
            J["cluster"] = J.item_id.map(lambda i: core[i]["source_item_id"])
            for label, g in (("all", J), ("public_unanimous", J[J.pub_unanimous]), ("public_split", J[~J.pub_unanimous])):
                est, lo, hi, p, n = fast_cluster_mean_ci(g.switch.to_numpy(), g.cluster.to_numpy())
                out[f"{cond}_{label}"] = dict(private_switch=est, lower=lo, upper=hi, n_items=n)
        # matched comparisons (B2): items publicly unanimous in BOTH conditions, and additionally with the same public answer
        import re as _re
        pubv = {c: df[(df.cond == c) & (df.turn == last)].set_index(["item_id", "agent"]).verdict for c in ("debate", "pool")}
        prv = {c: pr[pr.cond == f"private_{c}"].set_index(["item_id", "agent"]).verdict for c in ("debate", "pool")}
        items = sorted(set(df.item_id))
        unan = {c: {i for i in items if pubv[c].get((i, 0)) is not None and len({pubv[c].get((i, k)) for k in range(3)}) == 1} for c in ("debate", "pool")}
        both = sorted(unan["debate"] & unan["pool"])
        same = [i for i in both if pubv["debate"][(i, 0)] == pubv["pool"][(i, 0)]]
        for label, subset in (("unanimous_both", both), ("unanimous_both_same_answer", same)):
            diffs = []
            for i in subset:
                sw = {c: np.mean([float(prv[c].get((i, k)) != pubv[c][(i, k)]) for k in range(3)]) for c in ("debate", "pool")}
                diffs.append((i, sw["debate"], sw["pool"]))
            if diffs:
                arr = np.array([[x[1], x[2]] for x in diffs])
                cl = np.array([core[x[0]]["source_item_id"] for x in diffs])
                for j, c in enumerate(("debate", "pool")):
                    est, lo, hi, p, n = fast_cluster_mean_ci(arr[:, j], cl)
                    out[f"{label}_{c}"] = dict(private_switch=est, lower=lo, upper=hi, n_items=n)
                est, lo, hi, p, n = fast_cluster_mean_ci(arr[:, 0] - arr[:, 1], cl)
                out[f"{label}_paired_diff"] = dict(estimate=est, lower=lo, upper=hi, p=p, n_items=n)
        pat = _re.compile(r"\b(unanimous|consensus|all agents|all three agents|other agents|every agent|agents (?:agree|concur))", _re.I)
        for c in ("debate", "pool"):
            ev = df[(df.cond == c) & (df.turn == last)].evidence.fillna("")
            out[f"consensus_rhetoric_{c}_final"] = float(ev.map(lambda e: bool(pat.search(e))).mean())
        out["consensus_rhetoric_turn0"] = float(t0.evidence.fillna("").map(lambda e: bool(pat.search(e))).mean())
    import json
    json.dump(out, open(TABLES / f"long_{a.model}_{a.dataset}_private.json", "w"), indent=1)
    pd.set_option("display.width", 200)
    key = S[S.turn.isin([0, 1, 3, 6, 11]) & S.endpoint.isin(["acc", "agreement", "unanimous", "revision", "wrong_unanimous"])]
    print(key.pivot_table(index=["endpoint", "turn"], columns="cond", values="estimate").round(3).to_string())
    print(pd.DataFrame(stab).round(3).to_string())
    print(json.dumps(out, indent=1))

    import style
    from style import plt
    style.apply()
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3), sharex=True, constrained_layout=True)
    for ax, (ep, lab) in zip(axes, [("acc", "Accuracy"), ("agreement", "Agreement"), ("revision", "Revision rate"), ("wrong_unanimous", "Wrong & unanimous")]):
        for cond, col, ls in (("debate", style.TERRACOTTA, "-"), ("pool", style.SLATE, (0, (1.2, 1.4)))):
            g = S[(S.cond == cond) & (S.endpoint == ep)].sort_values("turn")
            ax.fill_between(g.turn, g.lower, g.upper, color=col, alpha=0.15, lw=0)
            ax.plot(g.turn, g.estimate, color=col, ls=ls, lw=1.6)
        ax.set_title(lab, fontsize=9.5)
        ax.set_xlabel("Turn")
    handles = [plt.Line2D([], [], color=style.TERRACOTTA, lw=1.6, label="debate"),
               plt.Line2D([], [], color=style.SLATE, lw=1.6, ls=(0, (1.2, 1.4)), label="independent pool")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=8.5)
    style.finish(fig, axes, FIGS / f"S_long_horizon_{a.model}")


if __name__ == "__main__":
    main()
