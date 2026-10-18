"""E6 probe bank fit on controlled CPT receiver states (never on debate test states).

Train = probe_pool items (disjoint units from release_600); test = release_600. Site (layer, position) is selected by
grouped 5-fold CV on train only, then frozen. For each probe we report CV AUROC, held-out test AUROC for a
difference-of-means (DoM) direction and an L2 logistic probe, a within-item label-permutation null, selectivity against
the other probes' labels, and cross-domain transfer. Frozen directions are saved for PCIF projection.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa

POS = ["answer", "msg_end", "src_end", "arg_mid", "prior_end"]


def probe_specs(W, core):
    # Validity rule (added after the first probe returned AUROC 1.000 at the source-label token): a probe may only read
    # positions whose token sequence up to that position is identical across the contrasted conditions' shared suffix, i.e.
    # the decision token (same prompt tail in every world). Message-internal positions differ in token identity and are
    # trivially separable. social_disagreement keeps msg_end because agree/disagree worlds share the same message tokens.
    gt = {r["id"]: r["ground_truth"] for r in core}
    strat = W.stratum
    q = W.qual.isin(["strong", "weak"])
    pe = W.prov.isin(["expert", "anon"])
    specs = {
        "provenance": dict(mask=pe, y=(W.prov == "expert"), pos=["answer"]),
        "evidence_quality": dict(mask=q, y=(W.qual == "strong"), pos=["answer"]),
        "social_disagreement": dict(mask=q, y=(W.agree == False), pos=["msg_end", "answer"]),  # noqa: E712
        "content_correctness": dict(mask=q & (W.qual == "strong") & W.has_target, y=(W.adv == W.o1), pos=["answer"]),
        "binding_xor": dict(mask=q & pe, y=((W.qual == "strong") ^ (W.prov == "expert")), pos=["answer"]),
        "revision_readiness": dict(mask=q & (W.agree == False), y=(W.argmax_label == W.adv), pos=["msg_end", "prior_end"]),  # noqa: E712
        "refusal_boundary": dict(mask=strat.isin(["refusal_calibration", "attack"]),
                                 y=W.item_id.map(lambda i: gt[i] == "refuse"), pos=["prior_end", "answer"]),
        "abstention": dict(mask=strat == "abstention", y=W.item_id.map(lambda i: gt[i] == "abstain"), pos=["prior_end", "answer"]),
    }
    return specs


def load_slice(acts, rows, pos_i, layer_i, chunk=4096):
    out = np.empty((len(rows), acts.shape[-1]), dtype=np.float32)
    rows = np.asarray(rows)
    order = np.argsort(rows)
    srt = rows[order]
    for a in range(0, len(srt), chunk):
        idx = srt[a:a + chunk]
        out[order[a:a + chunk]] = acts[idx, pos_i, layer_i].astype(np.float32)
    return out


def standardize(Xtr, Xte=None):
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-3
    return (Xtr - mu) / sd, (None if Xte is None else (Xte - mu) / sd), mu, sd


def fit_logistic(X, y, l2=1e-2, iters=60):
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y.astype(np.float32))
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt) + l2 * (w @ w) / 2
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy(), float(b.detach())


def auroc(y, s):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return np.nan
    return roc_auc_score(y, s)


def within_item_perm_null(y, s, groups, n=200, seed=240091):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, bool)
    g = pd.Series(np.arange(len(y))).groupby(np.asarray(groups)).indices
    vals = []
    for _ in range(n):
        yp = y.copy()
        for idx in g.values():
            yp[idx] = y[rng.permutation(idx)]
        if yp.all() or (~yp).all():
            continue
        vals.append(roc_auc_score(yp, s))
    return float(np.quantile(vals, 0.95)) if vals else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", default="cpt_core")
    ap.add_argument("--max_train", type=int, default=16000)
    args = ap.parse_args()
    d = ACTS / args.model / args.run
    meta = json.load(open(d / "meta.json"))
    acts = np.load(d / "acts.npy", mmap_mode="r")
    W = pd.DataFrame(read_jsonl(d / "worlds.jsonl"))
    core = load_core()
    cl = {r["id"]: (r["meta"].get("analysis_group") if r["stratum"] == "persuasion" else r["source_item_id"]) for r in core}
    W["cluster"] = W.item_id.map(cl)
    splits = json.load(open(ROOT / "data/splits/splits.json"))
    tr_items, te_items = set(splits["probe_pool"]), set(splits["release_600"])
    specs = probe_specs(W, core)
    rng = np.random.default_rng(240091)
    results, frozen = [], {}
    grid = meta["grid"]
    t0 = time.time()
    for name, sp in specs.items():
        m = sp["mask"].fillna(False).astype(bool).to_numpy()
        y_all = sp["y"].fillna(False).astype(bool).to_numpy()
        tr = np.where(m & W.item_id.isin(tr_items).to_numpy())[0]
        te = np.where(m & W.item_id.isin(te_items).to_numpy())[0]
        if len(tr) < 200 or len(te) < 50:
            print(f"[skip] {name}: n_train={len(tr)} n_test={len(te)}", flush=True)
            continue
        sel = tr if len(tr) <= args.max_train else rng.choice(tr, args.max_train, replace=False)
        # ---- site selection by grouped CV on train ----
        clusters = W.cluster.to_numpy()[sel]
        uniq = np.unique(clusters)
        fold_of = {c: i % 5 for i, c in enumerate(rng.permutation(uniq))}
        folds = np.array([fold_of[c] for c in clusters])
        best = None
        for pos in sp["pos"]:
            pi = POS.index(pos)
            for gi, layer in enumerate(grid):
                X = load_slice(acts, sel, pi, gi)
                if not np.isfinite(X).all() or np.abs(X).sum() == 0:
                    continue
                y = y_all[sel]
                cv = []
                for f in range(5):
                    a, b = folds != f, folds == f
                    Xa, Xb, _, _ = standardize(X[a], X[b])
                    w = Xa[y[a]].mean(0) - Xa[~y[a]].mean(0)
                    cv.append(auroc(y[b], Xb @ w))
                score = np.nanmean(cv)
                results.append(dict(probe=name, stage="cv_dom", position=pos, layer=layer, auroc=score, n=len(sel)))
                if best is None or score > best[0]:
                    best = (score, pos, gi, layer)
        _, pos, gi, layer = best
        pi = POS.index(pos)
        Xtr = load_slice(acts, sel, pi, gi)
        Xte = load_slice(acts, te, pi, gi)
        ytr, yte = y_all[sel], y_all[te]
        Ztr, Zte, mu, sd = standardize(Xtr, Xte)
        w_dom = Ztr[ytr].mean(0) - Ztr[~ytr].mean(0)
        w_log, b_log = fit_logistic(Ztr, ytr)
        s_dom, s_log = Zte @ w_dom, Zte @ w_log + b_log
        groups_te = W.item_id.to_numpy()[te]
        rec = dict(probe=name, stage="test", position=pos, layer=layer, cv_auroc=best[0],
                   test_auroc_dom=auroc(yte, s_dom), test_auroc_logistic=auroc(yte, s_log),
                   perm95_within_item=within_item_perm_null(yte, s_dom, groups_te, n=100),
                   n_train=len(sel), n_test=len(te), pos_rate_test=float(yte.mean()))
        # random-direction baseline (same site)
        rd = rng.standard_normal(Ztr.shape[1]).astype(np.float32)
        rec["random_dir_auroc"] = auroc(yte, Zte @ rd)
        # prior_end control: state before the message is read (item identity / difficulty only)
        if pos != "prior_end":
            Xc = load_slice(acts, sel, POS.index("prior_end"), gi)
            Xct = load_slice(acts, te, POS.index("prior_end"), gi)
            Zc, Zct, _, _ = standardize(Xc, Xct)
            wc = Zc[ytr].mean(0) - Zc[~ytr].mean(0)
            rec["pre_message_control_auroc"] = auroc(yte, Zct @ wc)
        # cross-domain transfer: fit on one training domain, test on the other two test domains
        doms_tr = W.domain.to_numpy()[sel]
        doms_te = W.domain.to_numpy()[te]
        for dom in DOMAIN_LABEL:
            a = doms_tr == dom
            b = doms_te != dom
            if a.sum() > 100 and b.sum() > 50 and ytr[a].any() and (~ytr[a]).any():
                wd = Ztr[a][ytr[a]].mean(0) - Ztr[a][~ytr[a]].mean(0)
                rec[f"transfer_from_{dom}"] = auroc(yte[b], Zte[b] @ wd)
        results.append(rec)
        frozen[name] = dict(position=pos, layer=int(layer), grid_index=int(gi), mu=mu.tolist(), sd=sd.tolist(),
                            w_dom=w_dom.tolist(), w_log=w_log.tolist(), b_log=b_log)
        print(f"[probe] {name}: site={pos}@L{layer} cv={best[0]:.3f} test DoM={rec['test_auroc_dom']:.3f} "
              f"logit={rec['test_auroc_logistic']:.3f} perm95={rec['perm95_within_item']:.3f} ({(time.time()-t0)/60:.1f} min)", flush=True)
    # ---- selectivity: each frozen probe scored against every other probe's label on the test split ----
    sel_rows = []
    for name, fz in frozen.items():
        pi, gi = POS.index(fz["position"]), fz["grid_index"]
        for other, sp in specs.items():
            m = sp["mask"].fillna(False).astype(bool).to_numpy() & W.item_id.isin(te_items).to_numpy()
            te = np.where(m)[0]
            if len(te) < 50:
                continue
            if len(te) > 20000:
                te = rng.choice(te, 20000, replace=False)
            X = load_slice(acts, te, pi, gi)
            Z = (X - np.array(fz["mu"], dtype=np.float32)) / np.array(fz["sd"], dtype=np.float32)
            sel_rows.append(dict(probe=name, label=other, auroc=auroc(sp["y"].fillna(False).astype(bool).to_numpy()[te], Z @ np.array(fz["w_dom"], dtype=np.float32))))
    out = TABLES / f"probes_{args.model}_{args.run}"
    pd.DataFrame(results).to_csv(f"{out}.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(f"{out}_selectivity.csv", index=False)
    json.dump(frozen, open(ACTS / args.model / f"probes_frozen_{args.run}.json", "w"))
    print(f"[done] probes {args.model}: {len(frozen)} probes in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
