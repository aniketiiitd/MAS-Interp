"""Heroes 2-4 on literature-format attribution frames (interp/frames_lit.py) + generated re-queries (runner/debate_lit.py).

Hero 2 (whose words): X = agent's own previous answer, Y = peer j's answer (disagreement). m = lp(X) - lp(Y).
  delta_own = mean over positions of [m(own-labelled slot holds X) - m(own-labelled slot holds Y)]
            = ((part - swap:j) + (part_first - swap_first:j)) / 2 ; position effect = ((part - part_first) + (swap - swap_first)) / 2.
Hero 3 (who reads): commitment to the agent's final public answer F: margin m_F = lp(F) - max other letter.
  dO = m_F(part) - m_F(observer); dQ = m_F(part) - m_F(part_para) (wording control).
  Private deviation from the generated private re-query (debate condition). Defectors vs stayers among publicly unanimous groups:
  family-standardised difference in dO, dQ and dO - dQ.
Hero 4 (who will hear, internal): direct read-out argmax != F for priv_ctx / pub_ctx / priv_self / pub_self.
Tables: outputs/tables/lit_frames_<M>_<dataset>_*.csv / .json
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


def ci(x, c):
    e, lo, hi, p, n = fast_cluster_mean_ci(np.asarray(x, float), np.asarray(c))
    return dict(est=e, lower=lo, upper=hi, p=p, n_clusters=n, n=int(np.isfinite(np.asarray(x, float)).sum()))


def margin(lp, F):
    if not isinstance(lp, dict) or F not in lp or len(lp) < 2:
        return np.nan
    return lp[F] - max(v for k, v in lp.items() if k != F)


def fam_gap(df, col, B=1000, seed=240091):
    """Defector-minus-stayer difference in `col`, pooled over the families holding both classes and weighted by the
    number of defectors. Rows where `col` is undefined (the read-out did not contain the agent's own final letter)
    are dropped first: leaving them in makes the point estimate NaN while np.nanquantile silently reports a CI from
    only those resamples that happened to miss them."""
    d0 = df[np.isfinite(pd.to_numeric(df[col], errors="coerce").to_numpy(float))].copy()

    def est(d):
        num = den = 0.0
        for _, g in d.groupby("family"):
            s = g.defect.to_numpy(bool)
            if s.sum() == 0 or (~s).sum() == 0:
                continue
            num += s.sum() * (g[col].to_numpy()[s].mean() - g[col].to_numpy()[~s].mean())
            den += s.sum()
        return num / den if den else np.nan
    by = {c: g for c, g in d0.groupby("cluster")}
    keys = list(by)
    base = dict(n=len(d0), n_defect=int(d0.defect.sum()), n_dropped=int(len(df) - len(d0)))
    if not keys or d0.defect.sum() < 10:
        return dict(est=np.nan, lower=np.nan, upper=np.nan, **base)
    rng = np.random.default_rng(seed)
    bs = [est(pd.concat([by[keys[i]] for i in rng.integers(0, len(keys), len(keys))])) for _ in range(B)]
    return dict(est=float(est(d0)), lower=float(np.nanquantile(bs, .025)), upper=float(np.nanquantile(bs, .975)), **base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="lit1200")
    ap.add_argument("--transcripts", default=None)
    a = ap.parse_args()
    tm = a.transcripts or a.model
    # index.jsonl holds computed per-row values, not offsets into an activation array, so comma-separated halves
    # union safely. A single dataset name behaves exactly as before and keeps its own filenames.
    parts = [x for x in a.dataset.split(",") if x]
    suff = "_" + tm + "text" if a.transcripts else ""
    I = pd.concat([pd.DataFrame(read_jsonl(ACTS / a.model / f"lit_frames_{p2}{suff}" / "index.jsonl"))
                   for p2 in parts], ignore_index=True)
    tag = f"{a.model}_" + (parts[0] if len(parts) == 1 else "full2400")
    if len(parts) > 1:
        print(f"[frames] pooled {len(parts)} halves -> {I.item_id.nunique()} distinct items", flush=True)
    I["cluster"] = I.item_id.map(cluster_of)
    I["domain"] = I.item_id.map(lambda i: CORE[i]["primary_domain"])
    I["family"] = I.item_id.map(lambda i: family_of(CORE[i]))
    ep_all = {k: {r.frame: r for r in g.itertuples()} for k, g in I.groupby(["item_id", "agent", "round"])}
    K = int(I["round"].max())
    ep1 = {(i, g): fr for (i, g, k), fr in ep_all.items() if k == 1}
    ep = {(i, g): fr for (i, g, k), fr in ep_all.items() if k == K}
    # ---- hero 2: ownership
    own_rows = []
    for (i, g), fr in ep1.items():
        p = fr["part"]
        X = p.prev_answers[g]
        for name in fr:
            if not name.startswith("swap:"):
                continue
            j = int(name.split(":")[1])
            Y = p.prev_answers[j]
            m = lambda r: r.lp_letters[X] - r.lp_letters[Y] if X in r.lp_letters and Y in r.lp_letters else np.nan
            last = m(p) - m(fr[name])
            first = m(fr["part_first"]) - m(fr[f"swap_first:{j}"])
            pos = ((m(p) - m(fr["part_first"])) + (m(fr[name]) - m(fr[f"swap_first:{j}"]))) / 2
            own_rows.append(dict(item_id=i, agent=g, peer=j, cluster=p.cluster, domain=p.domain, family=p.family,
                                 delta_own=(last + first) / 2, delta_own_last=last, delta_own_first=first, position_last_minus_first=pos))
    O = pd.DataFrame(own_rows)
    out = []
    if len(O):
        for dom, g in [("all", O)] + list(O.groupby("domain")):
            for m_ in ("delta_own", "delta_own_last", "delta_own_first", "position_last_minus_first"):
                out.append(dict(domain=dom, metric=m_, **ci(g[m_], g.cluster)))
        fam = [dict(family=f, **ci(g.delta_own, g.cluster)) for f, g in O.groupby("family")]
        pd.DataFrame(fam).to_csv(TABLES / f"lit_frames_{tag}_own_family.csv", index=False)
        O.to_csv(TABLES / f"lit_frames_{tag}_own_pairs.csv", index=False)
    OWN = pd.DataFrame(out)
    OWN.to_csv(TABLES / f"lit_frames_{tag}_own.csv", index=False)
    print(OWN.round(3).to_string())
    # ---- hero 3 at the first debate round: outnumbered readers (both peers agree on an answer different from the agent's own)
    rows1 = []
    for (i, g), fr in ep1.items():
        p = fr["part"]
        pa = p.prev_answers
        peers = [pa[h] for h in range(3) if h != g]
        if peers[0] is None or peers[0] != peers[1] or pa[g] == peers[0]:
            continue
        M_, X = peers[0], pa[g]
        mm = lambda r: r.lp_letters[M_] - r.lp_letters[X] if M_ in r.lp_letters and X in r.lp_letters else np.nan
        am = lambda r: float(max(r.lp_letters, key=r.lp_letters.get) == M_)
        rows1.append(dict(item_id=i, agent=g, cluster=p.cluster, domain=p.domain, family=p.family,
                          m_part=mm(p), m_observer=mm(fr["observer"]), m_para=mm(fr["part_para"]),
                          adopt_part=am(p), adopt_observer=am(fr["observer"]), adopt_para=am(fr["part_para"])))
    O1 = pd.DataFrame(rows1)
    role = []
    if len(O1):
        O1.to_csv(TABLES / f"lit_frames_{tag}_outnumbered_r1.csv", index=False)
        for dom, g in [("all", O1)] + list(O1.groupby("domain")):
            for m_, x in {"adopt participant": g.adopt_part, "adopt observer": g.adopt_observer, "adopt paraphrase": g.adopt_para,
                          "adopt participant - observer": g.adopt_part - g.adopt_observer,
                          "adopt participant - paraphrase": g.adopt_part - g.adopt_para,
                          "margin participant - observer": g.m_part - g.m_observer,
                          "margin participant - paraphrase": g.m_part - g.m_para}.items():
                role.append(dict(domain=dom, metric=m_, **ci(x, g.cluster)))
    pd.DataFrame(role).to_csv(TABLES / f"lit_frames_{tag}_role_r1.csv", index=False)
    print(pd.DataFrame(role).round(3).to_string() if role else "no outnumbered episodes")
    # ---- hero 3/4: readers and audience
    # round count is read from the first half; refuse to pool halves that ran different counts
    metas = [json.load(open(RAW_LIT / tm / p2 / "meta.json")) for p2 in parts]
    rr = {mm["rounds"] for mm in metas}
    if len(rr) != 1:
        raise SystemExit(f"[frames] halves ran different round counts {rr}; refusing to pool")
    R = rr.pop()
    priv = {(x["item_id"], x["agent"]): x["answer"]
            for p2 in parts for x in read_jsonl(RAW_LIT / a.model / p2 / "requery.jsonl")
            if x["cond"] == "debate" and x["query"] == "private"}
    rows = []
    for (i, g), fr in ep.items():
        p = fr["part"]
        F = p.final_answer
        if F is None:
            continue
        rec = dict(item_id=i, agent=g, cluster=p.cluster, domain=p.domain, family=p.family, F=F, private=priv.get((i, g)))
        for name in ("part", "observer", "part_para", "priv_ctx", "pub_ctx", "priv_self", "pub_self"):
            if name in fr:
                rec[f"m_{name}"] = margin(fr[name].lp_letters, F)
                rec[f"dev_{name}"] = float(max(fr[name].lp_letters, key=fr[name].lp_letters.get) != F)
        rows.append(rec)
    Rd = pd.DataFrame(rows)
    finals = Rd.groupby("item_id").F.agg(lambda s: s.nunique() == 1 and len(s) == 3)
    Rd["unanimous"] = Rd.item_id.map(finals)
    Rd["dO"] = Rd.m_part - Rd.m_observer
    Rd["dQ"] = Rd.m_part - Rd.m_part_para
    Rd["dOQ"] = Rd.dO - Rd.dQ
    Rd.to_csv(TABLES / f"lit_frames_{tag}_readers.csv", index=False)
    U = Rd[Rd.unanimous & Rd.private.notna()].copy()
    U["defect"] = U.private != U.F
    res = dict(n_unanimous_agents=len(U), n_defect=int(U.defect.sum()),
               mean_dO=ci(U.dO, U.cluster), mean_dQ=ci(U.dQ, U.cluster), mean_dOQ=ci(U.dOQ, U.cluster))
    if U.defect.sum() >= 10:
        for col in ("dO", "dQ", "dOQ", "m_part", "m_observer", "m_part_para"):
            res[f"defector_minus_stayer_{col}"] = fam_gap(U, col)
    aud = {}
    for dom, g in [("all", U)] + list(U.groupby("domain")):
        aud[dom] = {m_: ci(g[m_], g.cluster) for m_ in ("dev_priv_ctx", "dev_pub_ctx", "dev_priv_self", "dev_pub_self") if m_ in g}
        if "dev_priv_ctx" in g:
            aud[dom]["priv_ctx - pub_ctx"] = ci(g.dev_priv_ctx - g.dev_pub_ctx, g.cluster)
            aud[dom]["priv_self - pub_self"] = ci(g.dev_priv_self - g.dev_pub_self, g.cluster)
    res["audience_direct_readout"] = aud
    json.dump(res, open(TABLES / f"lit_frames_{tag}_readers.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "audience_direct_readout"}, indent=1, default=float))


if __name__ == "__main__":
    main()
