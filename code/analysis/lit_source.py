"""Hero 1 (who said it) on literature-format source worlds (interp/cpt_lit.py score + swap).

Tables (outputs/tables/lit_source_<M>_*):
  cells     adoption shift vs no label for the 2 x 3 status x record descriptors (disagreement worlds), item-cluster CIs
  effects   status effect per record level, record effect per status level, status share, professor-with-poor-record vs no label
  depth     status share along the per-layer credibility axis (decision token): axis = mean(prof_good) - mean(ord_bad), fit on half
            of the items and evaluated on the other half (and vice versa); CIs by item-cluster bootstrap
  swap      interchange recovery by layer: (patched - recipient) / (donor - recipient) for status and record contrasts, patching
            the full decision-token state or only its component along the record axis, the status axis, or a random axis
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
DESCS = ["prof_good", "prof_unst", "prof_bad", "ord_good", "ord_unst", "ord_bad"]


def ci(x, c):
    e, lo, hi, p, n = fast_cluster_mean_ci(np.asarray(x, float), np.asarray(c))
    return dict(est=e, lower=lo, upper=hi, n_clusters=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dir", default="cptlit_core_n600")
    ap.add_argument("--swap_dir", default="cptlit_swap_core")
    ap.add_argument("--B", type=int, default=500)
    a = ap.parse_args()
    d = ACTS / a.model / a.dir
    W = pd.DataFrame(read_jsonl(d / "worlds.jsonl"))
    meta = json.load(open(d / "meta.json"))
    grid = meta["grid"]
    W["cluster"] = W.item_id.map(cluster_of)
    key = ["item_id", "adv", "qual", "prior"]
    W = W.join(W[W.source == "none"].set_index(key).m_adv.rename("m_none"), on=key)
    W["P"] = W.m_adv - W.m_none
    dis = W[(W.adv != W.prior) & (W.source != "none")]
    per = dis.groupby(["item_id", "cluster", "source"]).P.mean().unstack("source")
    cl = per.index.get_level_values("cluster")
    tag = f"{a.model}"
    rows = [dict(source=s, **ci(per[s], cl)) for s in DESCS]
    pd.DataFrame(rows).to_csv(TABLES / f"lit_source_{tag}_cells.csv", index=False)
    eff = {
        "status | good record": per.prof_good - per.ord_good, "status | unknown record": per.prof_unst - per.ord_unst,
        "status | poor record": per.prof_bad - per.ord_bad, "record | professor": per.prof_good - per.prof_bad,
        "record | ordinary": per.ord_good - per.ord_bad, "professor with poor record vs no label": per.prof_bad,
    }
    E = pd.DataFrame([dict(effect=k, **ci(v, cl)) for k, v in eff.items()])
    st = ((per.prof_good - per.ord_good) + (per.prof_bad - per.ord_bad)) / 2
    rc = ((per.prof_good - per.prof_bad) + (per.ord_good - per.ord_bad)) / 2
    rng = np.random.default_rng(240091)
    uc = np.unique(cl)
    grp = {c: np.where(cl == c)[0] for c in uc}
    boots = []
    for _ in range(a.B):
        ix = np.concatenate([grp[c] for c in rng.choice(uc, len(uc))])
        boots.append(st.iloc[ix].mean() / (st.iloc[ix].mean() + rc.iloc[ix].mean()))
    share = st.mean() / (st.mean() + rc.mean())
    E = pd.concat([E, pd.DataFrame([dict(effect="status share (behaviour)", est=share, lower=np.quantile(boots, .025),
                                         upper=np.quantile(boots, .975), n_clusters=len(uc))])])
    E.to_csv(TABLES / f"lit_source_{tag}_effects.csv", index=False)
    print(E.round(3).to_string())
    # ---- depth: status share along a cross-fitted credibility axis
    A = np.load(d / "acts.npy", mmap_mode="r")
    items = np.array(sorted(W.item_id.unique()))
    half = {i: h64("litsplit", i) % 2 for i in items}
    W["half"] = W.item_id.map(half)
    dis_idx = W[(W.adv != W.prior) & (W.source != "none")]
    proj = np.zeros((len(W), len(grid)), np.float32)
    for gi, li in enumerate(grid):
        X = np.asarray(A[:, 0, gi], np.float32)
        for h in (0, 1):
            fit = W[(W.half == h)]
            u = X[fit.index[fit.source == "prof_good"]].mean(0) - X[fit.index[fit.source == "ord_bad"]].mean(0)
            u /= np.linalg.norm(u)
            ev = W.index[W.half != h]
            proj[ev, gi] = X[ev] @ u
    rows = []
    for gi, li in enumerate(grid):
        W["_p"] = proj[:, gi]
        pp = W.loc[dis_idx.index].groupby(["item_id", "cluster", "source"])._p.mean().unstack("source")
        pcl = pp.index.get_level_values("cluster")
        s_ = ((pp.prof_good - pp.ord_good) + (pp.prof_bad - pp.ord_bad)) / 2
        r_ = ((pp.prof_good - pp.prof_bad) + (pp.ord_good - pp.ord_bad)) / 2
        pg = {c: np.where(pcl == c)[0] for c in np.unique(pcl)}
        bs = []
        for _ in range(a.B):
            ix = np.concatenate([pg[c] for c in rng.choice(list(pg), len(pg))])
            bs.append(s_.iloc[ix].mean() / (s_.iloc[ix].mean() + r_.iloc[ix].mean()))
        rows.append(dict(layer=li, depth=li / ({"Q9": 32, "Q4": 32, "L31": 32, "G4": 42}.get(a.model, max(grid) + 1) - 1),
                         share=s_.mean() / (s_.mean() + r_.mean()), lower=np.quantile(bs, .025), upper=np.quantile(bs, .975),
                         status=s_.mean(), record=r_.mean()))
    D = pd.DataFrame(rows)
    D.to_csv(TABLES / f"lit_source_{tag}_depth.csv", index=False)
    print(D.round(3).to_string())
    # ---- interchange interventions
    sp = ACTS / a.model / a.swap_dir / "rows.jsonl"
    # exists() is not the question. A swap run that processed no items leaves a zero-byte file, which passed this
    # check and then died on an empty frame with an AttributeError naming a column rather than the cause.
    if sp.exists() and sp.stat().st_size == 0:
        print(f"\n[interchange] SKIPPED: {sp} is empty, so the swap run produced no rows. The effects and depth "
              f"tables above are complete and unaffected; only the interchange section is missing.", flush=True)
    elif sp.exists():
        S = pd.DataFrame(read_jsonl(sp))
        S["cluster"] = S.item_id.map(cluster_of)
        base = S[S.kind.isin(["donor", "recipient"])].pivot_table(index=["item_id", "adv", "contrast"], columns="kind", values="m_adv")
        P = S[~S.kind.isin(["donor", "recipient"])].join(base, on=["item_id", "adv", "contrast"])
        P["gap"] = P.donor - P.recipient
        rows = []
        for (c, k, L), g in P.groupby(["contrast", "kind", "layer"]):
            g = g[g.gap.abs() > 0.25]  # contrasts with a measurable donor-recipient difference
            num = (g.m_adv - g.recipient).to_numpy()
            den = g.gap.to_numpy()
            gc = g.cluster.to_numpy()
            ucs = np.unique(gc)
            gi_ = {u: np.where(gc == u)[0] for u in ucs}
            bs = []
            for _ in range(a.B):
                ix = np.concatenate([gi_[u] for u in rng.choice(ucs, len(ucs))])
                bs.append(num[ix].sum() / den[ix].sum())
            rows.append(dict(contrast=c, kind=k, layer=int(L), recovery=num.sum() / den.sum(), lower=np.quantile(bs, .025),
                             upper=np.quantile(bs, .975), n=len(g), n_clusters=len(ucs)))
        SW = pd.DataFrame(rows)
        SW.to_csv(TABLES / f"lit_source_{tag}_swap.csv", index=False)
        print(SW.pivot_table(index="layer", columns=["contrast", "kind"], values="recovery").round(2).to_string())
        # ---- the magnitude control, on one common scale.
        # `recovery` is normalised per contrast, so it cannot answer whether status moves behaviour less than record
        # only because its residual-stream difference is smaller. Here every arm is reported as the raw shift in
        # log p(advocated) - log p(prior), so the natural and magnitude-matched swaps are directly comparable.
        L = int(P.layer.max())
        Q = P[P.layer == L].copy()
        Q["eff"] = Q.m_adv - Q.recipient
        W = Q.pivot_table(index=["item_id", "adv"], columns=["contrast", "kind"], values="eff")
        W = W.dropna(subset=[("record", "full"), ("status", "full"), ("record", "norm_matched"), ("status", "norm_matched")])
        cl_ = np.array([cluster_of(i) for i, _ in W.index])
        arms = [("record", "full", "record swap, as written"), ("record", "norm_matched", "record, shrunk to status magnitude"),
                ("status", "norm_matched", "status, amplified to record magnitude"), ("status", "full", "status swap, as written"),
                ("record", "proj_rand", "random direction, record magnitude")]
        ucs = np.unique(cl_)
        gi2 = {u: np.where(cl_ == u)[0] for u in ucs}
        draws = [np.concatenate([gi2[u] for u in rng.choice(ucs, len(ucs))]) for _ in range(a.B)]
        mrows = []
        for c, k, lab in arms:
            if (c, k) not in W:
                continue
            v = W[(c, k)].to_numpy(float)
            bs = [v[ix].mean() for ix in draws]
            mrows.append(dict(arm=lab, contrast=c, kind=k, est=v.mean(), lower=np.quantile(bs, .025),
                              upper=np.quantile(bs, .975), n=len(v), n_clusters=len(ucs)))
        rf, sn = W[("record", "full")].to_numpy(float), W[("status", "norm_matched")].to_numpy(float)
        rn = W[("record", "norm_matched")].to_numpy(float)
        sf_ = W[("status", "full")].to_numpy(float)
        # Both directions respond sublinearly to rescaling, so a single normalised "recovery" number is misleading.
        # Report the record:status advantage measured at BOTH magnitudes -- the conclusion must not depend on which
        # one the two contrasts are matched at -- plus how each direction actually responds to being scaled.
        for lab, f in (("status at record magnitude / record swap", lambda ix: sn[ix].mean() / rf[ix].mean()),
                       ("record at status magnitude / status at record magnitude", lambda ix: rn[ix].mean() / sn[ix].mean()),
                       ("record:status advantage at the record magnitude", lambda ix: rf[ix].mean() / sn[ix].mean()),
                       ("record:status advantage at the status magnitude", lambda ix: rn[ix].mean() / sf_[ix].mean()),
                       ("status effect ratio when scaled up to record magnitude", lambda ix: sn[ix].mean() / sf_[ix].mean()),
                       ("record effect ratio when scaled down to status magnitude", lambda ix: rn[ix].mean() / rf[ix].mean())):
            bs = [f(ix) for ix in draws]
            mrows.append(dict(arm=lab, contrast="ratio", kind="ratio", est=f(np.arange(len(rf))),
                              lower=np.quantile(bs, .025), upper=np.quantile(bs, .975), n=len(rf), n_clusters=len(ucs)))
        M = pd.DataFrame(mrows)
        # A ratio whose denominator is not distinguishable from zero is not a number worth printing: in L31 the
        # as-written status swap is +0.011 [-0.020, 0.044], and ratios built on it came out as 42.6 [-515, 270].
        # Blank those cells rather than let a meaningless interval reach a table or a figure.
        den_ok = {"status at record magnitude / record swap": ("record", "full"),
                  "record at status magnitude / status at record magnitude": ("status", "norm_matched"),
                  "record:status advantage at the record magnitude": ("status", "norm_matched"),
                  "record:status advantage at the status magnitude": ("status", "full"),
                  "status effect ratio when scaled up to record magnitude": ("status", "full"),
                  "record effect ratio when scaled down to status magnitude": ("record", "full")}
        lut = {(r.contrast, r.kind): r for _, r in M[M.kind != "ratio"].iterrows()}
        for i, r in M.iterrows():
            key = den_ok.get(r.arm)
            if key is None or key not in lut:
                continue
            d = lut[key]
            if d.lower <= 0 <= d.upper:       # denominator straddles zero -> ratio undefined
                M.loc[i, ["est", "lower", "upper"]] = np.nan
                M.loc[i, "arm"] = r.arm + "  [undefined: denominator not distinguishable from 0]"
        M.to_csv(TABLES / f"lit_source_{tag}_magnitude.csv", index=False)
        print("\n[magnitude control, layer %d, raw logits]" % L)
        print(M.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
