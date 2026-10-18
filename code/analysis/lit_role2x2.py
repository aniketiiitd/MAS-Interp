"""Is 'participant vs outside reader' a role effect or a wording effect? (round-1 outnumbered episodes)

The main frames run gives two participant wordings (`part`, `part_para`) and one observer wording (`observer`);
the `--extra` run adds a second observer wording (`observer_persona`, a named reviewer) and an unlabelled control
(`neutral_label`), all on the same episodes and the same byte-identical peer text. That completes a 2x2 in which the
between-role contrast can finally be read against the within-role wording spread:

    participant : part, part_para          observer : observer, observer_persona

Reported per episode (both peers agree on M != the agent's own X):
    adopt  = the forced-answer read-out is M        margin = lp[M] - lp[X]
    role contrast     = mean(observer arms) - mean(participant arms)
    wording contrasts = part - part_para,  observer - observer_persona
    excess            = |role contrast| - max(|wording contrasts|)     <- the quantity the claim lives or dies on

`excess` is recomputed inside every bootstrap replicate, so its percentile interval is the interval of the plug-in
statistic. A role effect is only reportable if excess > 0 with a CI excluding 0; otherwise the between-role gap is
inside the range that rewording alone produces and the claim is dropped rather than caveated.

Outputs: outputs/tables/lit_role2x2_<M>_<dataset>_{arms,contrasts}.csv and _episodes.csv
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import *  # noqa

CORE = {r["id"]: r for r in load_core()}
cluster_of = lambda i: CORE[i]["meta"].get("analysis_group") if CORE[i]["stratum"] == "persuasion" else CORE[i]["source_item_id"]

PART = ["part", "part_para"]
OBS = ["observer", "observer_persona"]
# The role arms are confounded with ownership labelling: `part`/`part_para` name the agent's own previous answer
# ("Your own latest answer was: ..."), while `observer`/`observer_persona`/`neutral_label` present all three responses
# as an unlabelled pool. `neutral_label` is the arm that separates the two accounts -- it is a PARTICIPANT prompt
# ("revise your recent opinion") with no ownership label, so role and ownership predict opposite groupings for it.
OWN_LABELLED = ["part", "part_para"]
OWN_UNLABELLED = ["observer", "observer_persona", "neutral_label"]
B = 4000


def load_round1(d):
    I = pd.DataFrame(read_jsonl(d / "index.jsonl"))
    return {(r.item_id, r.agent): {} for r in I.itertuples()} and \
        {k: {r.frame: r for r in g.itertuples()} for k, g in I[I["round"] == 1].groupby(["item_id", "agent"])}


def boot(df, stats, seed=99131):
    """Cluster bootstrap over item clusters; `stats` maps a name to a function of a resampled frame."""
    by = {c: g for c, g in df.groupby("cluster")}
    keys = list(by)
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in stats}
    for _ in range(B):
        d = pd.concat([by[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        for k, f in stats.items():
            draws[k].append(f(d))
    out = {}
    for k, f in stats.items():
        v = np.asarray(draws[k], float)
        out[k] = dict(est=float(f(df)), lower=float(np.nanpercentile(v, 2.5)), upper=float(np.nanpercentile(v, 97.5)),
                      n_clusters=len(keys), n=len(df))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="lit1200")
    a = ap.parse_args()
    base = ACTS / a.model
    main_ep = load_round1(base / f"lit_frames_{a.dataset}")
    extra_ep = load_round1(base / f"lit_frames_extra_{a.dataset}")
    tag = f"{a.model}_{a.dataset}"

    rows, ref_gap = [], []
    for key, fr in main_ep.items():
        ex = extra_ep.get(key)
        if ex is None or "observer_persona" not in ex:
            continue
        p = fr["part"]
        pa = p.prev_answers
        i, g = key
        peers = [pa[h] for h in range(3) if h != g]
        if peers[0] is None or peers[0] != peers[1] or pa[g] == peers[0]:
            continue  # same outnumbered definition as analysis/lit_frames.py
        M_, X = peers[0], pa[g]
        mm = lambda r: r.lp_letters[M_] - r.lp_letters[X] if M_ in r.lp_letters and X in r.lp_letters else np.nan
        am = lambda r: float(max(r.lp_letters, key=r.lp_letters.get) == M_)
        src = {"part": p, "part_para": fr["part_para"], "observer": fr["observer"],
               "observer_persona": ex["observer_persona"], "neutral_label": ex.get("neutral_label")}
        if src["neutral_label"] is None:
            continue
        rec = dict(item_id=i, agent=g, cluster=cluster_of(i), domain=CORE[i]["primary_domain"], family=family_of(CORE[i]))
        for nm, r in src.items():
            rec[f"adopt_{nm}"], rec[f"m_{nm}"] = am(r), mm(r)
        rows.append(rec)
        # the `part` arm is re-run inside the extra batch: a disagreement means the two runs are not comparable
        ref_gap.append(abs(am(p) - am(ex["part"])))
    D = pd.DataFrame(rows)
    if not len(D):
        print("[role2x2] no matched episodes -- is the extra frames run finished?")
        return
    agree = 1 - float(np.mean(ref_gap))
    print(f"[role2x2] {len(D)} matched outnumbered episodes, {D.cluster.nunique()} clusters; "
          f"`part` reproduces across the two runs on {100*agree:.1f}% of episodes")
    D.to_csv(TABLES / f"lit_role2x2_{tag}_episodes.csv", index=False)

    arms, contrasts = [], []
    for scale, pre in (("adopt", "adopt_"), ("margin", "m_")):
        role = lambda d, p=pre: d[[p + x for x in OBS]].mean(axis=1).mean() - d[[p + x for x in PART]].mean(axis=1).mean()
        w_p = lambda d, p=pre: d[p + PART[0]].mean() - d[p + PART[1]].mean()
        w_o = lambda d, p=pre: d[p + OBS[0]].mean() - d[p + OBS[1]].mean()
        mean_of = lambda d, arms, p_=pre: d[[p_ + x for x in arms]].mean(axis=1).mean()
        spread = lambda d, arms, p_=pre: (max(d[p_ + x].mean() for x in arms) - min(d[p_ + x].mean() for x in arms)) if len(arms) > 1 else 0.0
        own = lambda d: mean_of(d, OWN_UNLABELLED) - mean_of(d, OWN_LABELLED)
        s_lab = lambda d: spread(d, OWN_LABELLED)
        s_unlab = lambda d: spread(d, OWN_UNLABELLED)
        stats = {"role contrast (observer - participant)": role, "wording within participant": w_p,
                 "wording within observer": w_o,
                 "excess (|role| - max|wording|)": lambda d, r=role, a_=w_p, b_=w_o: abs(r(d)) - max(abs(a_(d)), abs(b_(d))),
                 "ownership contrast (unlabelled - labelled)": own,
                 "wording spread within labelled": s_lab,
                 "wording spread within unlabelled": s_unlab,
                 "excess ownership (|own| - max spread)": lambda d, o=own, a_=s_lab, b_=s_unlab: abs(o(d)) - max(a_(d), b_(d))}
        for dom, g in [("all", D)] + list(D.groupby("domain")):
            res = boot(g, stats, seed=hash(("r2", scale, dom)) % (2**31))
            for nm, v in res.items():
                contrasts.append(dict(scale=scale, domain=dom, metric=nm, **v))
            for arm in PART + OBS + ["neutral_label"]:
                col = pre + arm
                res1 = boot(g, {arm: (lambda d, c=col: d[c].mean())}, seed=hash(("a", scale, dom, arm)) % (2**31))
                arms.append(dict(scale=scale, domain=dom, arm=arm, **res1[arm]))
    A, C = pd.DataFrame(arms), pd.DataFrame(contrasts)
    A.to_csv(TABLES / f"lit_role2x2_{tag}_arms.csv", index=False)
    C.to_csv(TABLES / f"lit_role2x2_{tag}_contrasts.csv", index=False)
    print(A[A.domain == "all"].round(3).to_string(index=False))
    print(C[C.domain == "all"].round(3).to_string(index=False))
    for scale in ("adopt", "margin"):
        for pre, lab in (("excess (", "role"), ("excess ownership", "ownership")):
            g = C[(C.scale == scale) & (C.domain == "all") & (C.metric.str.startswith(pre))]
            if not len(g):
                continue
            e = g.iloc[0]
            verdict = "exceeds wording" if e.lower > 0 else "NOT separable from wording -- claim drops"
            print(f"[verdict/{scale}/{lab}] excess {e.est:+.3f} [{e.lower:+.3f}, {e.upper:+.3f}] -> {verdict}")


if __name__ == "__main__":
    main()
