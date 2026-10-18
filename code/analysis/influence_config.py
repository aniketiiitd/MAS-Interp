"""Natural-debate influence structure (E2, correlational): for every agent transition t-1 -> t in rounds 1..T-1, the
configuration of peers at t-1 (one dissenter / united dissenting pair / split dissenting pair) x whether the dissenting
answer was correct, and whether the agent adopts the dissenting answer at t. The independent pool, where the agent never
sees its peers, gives the no-exposure baseline for the same configuration (same turn-0 answers, same items).
Exposure effect = P(adopt | debate) - P(adopt | pool). Item-cluster bootstrap CIs."""
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


def transitions(model, dataset, turns):
    core = {r["id"]: r for r in load_core()}
    by = {}
    for f in ["t0"] + [f"{c}_t{t}" for c in ("debate", "pool") for t in range(1, turns)]:
        for x in read_jsonl(RAW / model / dataset / f"{f}.jsonl"):
            if x["agent"] < 3:
                by[(x["item_id"], x["cond"] if x["cond"] != "t0" else "t0", x["turn"], x["agent"])] = x
    rows = []
    items = sorted({k[0] for k in by})
    for i in items:
        r = core[i]
        cl = r["meta"].get("analysis_group") if r["stratum"] == "persuasion" else r["source_item_id"]
        for cond in ("debate", "pool"):
            for t in range(1, turns):
                prev_key = lambda a: (i, "t0", 0, a) if t == 1 else (i, cond, t - 1, a)
                prev = [by.get(prev_key(a)) for a in range(3)]
                # the configuration is always defined by the DEBATE trajectory's previous turn, so debate and pool
                # compare the same peer answers; for t>1 the pool's own trajectory differs, so use t=1 for the baseline
                if cond == "pool" and t > 1:
                    continue
                if any(p is None or p["verdict"] is None for p in prev):
                    continue
                for a in range(3):
                    nx = by.get((i, cond, t, a))
                    if nx is None or nx["verdict"] is None:
                        continue
                    me = prev[a]
                    dis = [prev[b] for b in range(3) if b != a and prev[b]["verdict"] != me["verdict"]]
                    if not dis:
                        continue
                    if len(dis) == 1:
                        cfg = "one dissenter"
                    elif dis[0]["verdict"] == dis[1]["verdict"]:
                        cfg = "united pair dissents"
                    else:
                        cfg = "split pair dissents"
                    alt_correct = dis[0]["correct"] if cfg != "split pair dissents" else None
                    rows.append(dict(item_id=i, cluster=cl, family=family_of(r), domain=r["primary_domain"], cond=cond, turn=t,
                                     cfg=cfg, alt_correct=alt_correct, me_correct=me["correct"],
                                     adopt=nx["verdict"] in {d["verdict"] for d in dis}))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="core")
    ap.add_argument("--turns", type=int, default=4)
    a = ap.parse_args()
    D = transitions(a.model, a.dataset, a.turns)
    D.to_csv(TABLES / f"influence_transitions_{a.model}_{a.dataset}.csv", index=False)
    r1 = D[D.turn == 1]
    out = []
    r1 = r1.assign(altc=r1.alt_correct.map(lambda x: "None" if x is None or x != x else str(bool(x))))
    for (cfg, altc), g in r1.groupby(["cfg", "altc"]):
        wide = g.pivot_table(index=["item_id", "cluster"], columns="cond", values="adopt", aggfunc="mean").reset_index()
        for cond in ("debate", "pool"):
            est, lo, hi, p, n = fast_cluster_mean_ci(wide[cond].to_numpy(float), wide.cluster.to_numpy())
            out.append(dict(cfg=cfg, alt_correct=altc, quantity=f"P(adopt|{cond})", estimate=est, lower=lo, upper=hi, n_items=n))
        dd = (wide["debate"] - wide["pool"]).to_numpy(float)
        est, lo, hi, p, n = fast_cluster_mean_ci(dd, wide.cluster.to_numpy())
        out.append(dict(cfg=cfg, alt_correct=altc, quantity="exposure effect (debate-pool)", estimate=est, lower=lo, upper=hi, n_items=n))
    # interaction: exposure effect under united pair for wrong vs right alternatives
    T = pd.DataFrame(out)
    T.to_csv(TABLES / f"influence_config_{a.model}_{a.dataset}.csv", index=False)
    pd.set_option("display.width", 200)
    print(T.round(3).to_string())

    import style
    from style import plt
    style.apply()
    E = T[T.quantity == "exposure effect (debate-pool)"]
    fig, ax = plt.subplots(figsize=(7.2, 2.4), constrained_layout=True)
    cfgs = ["one dissenter", "united pair dissents", "split pair dissents"]
    y = 0
    ticks, labels = [], []
    for cfg in cfgs:
        for altc, c, lab in (("True", style.TEAL, "dissenting answer correct"), ("False", style.TERRACOTTA, "dissenting answer wrong"),
                             ("None", style.SLATE, "dissenters split" if cfg == "split pair dissents" else "no ground truth (normative/behavioral)")):
            s = E[(E.cfg == cfg) & (E.alt_correct == altc)]
            if not len(s):
                continue
            s = s.iloc[0]
            ax.plot([s.lower, s.upper], [y, y], color=c, lw=1.5)
            ax.scatter([s.estimate], [y], color=c, s=26, zorder=3)
            ticks.append(y)
            labels.append(f"{cfg}: {lab}")
            y -= 1
        y -= 0.5
    ax.axvline(0, color=style.SLATE, lw=0.8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Exposure effect on adopting the dissenting answer: P(debate) − P(independent pool), round 1")
    ax.grid(axis="y", visible=False)
    style.finish(fig, ax, FIGS / f"S_influence_config_{a.model}_{a.dataset}")


if __name__ == "__main__":
    main()
