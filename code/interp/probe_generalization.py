"""Abstraction tests for CPT probes (decision-token site): leave-one-wrapper-template-out and cross-author transfer.
A probe that only copies the descriptor's wording forward should fail on a template whose expert/anonymous descriptions use
different words; an abstract provenance representation should transfer."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "interp"))
from common import *  # noqa
from probes import POS, load_slice, standardize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--run", default="cpt_core")
    ap.add_argument("--max_rows", type=int, default=24000)
    a = ap.parse_args()
    d = ACTS / a.model / a.run
    fz = json.load(open(ACTS / a.model / f"probes_frozen_{a.run}.json"))
    meta = json.load(open(d / "meta.json"))
    acts = np.load(d / "acts.npy", mmap_mode="r")
    W = pd.DataFrame(read_jsonl(d / "worlds.jsonl"))
    rng = np.random.default_rng(240091)
    specs = {
        "provenance": (W.prov.isin(["expert", "anon"]), W.prov == "expert"),
        "evidence_quality": (W.qual.isin(["strong", "weak"]), W.qual == "strong"),
        "social_disagreement": (W.qual.isin(["strong", "weak"]), W.agree == False),  # noqa: E712
    }
    rows = []
    for name, (mask, y) in specs.items():
        if name not in fz:
            continue
        f = fz[name]
        pi, gi = POS.index(f["position"]), f["grid_index"]
        idx = np.where(mask.fillna(False).to_numpy())[0]
        if len(idx) > a.max_rows:
            idx = np.sort(rng.choice(idx, a.max_rows, replace=False))
        X = load_slice(acts, idx, pi, gi)
        yy = y.fillna(False).to_numpy()[idx]
        tpl = W.template.to_numpy()[idx]
        auth = W.author.to_numpy()[idx]
        for held in sorted(set(tpl)):
            tr, te = tpl != held, tpl == held
            Ztr, Zte, _, _ = standardize(X[tr], X[te])
            w = Ztr[yy[tr]].mean(0) - Ztr[~yy[tr]].mean(0)
            rows.append(dict(probe=name, test="heldout_template", held=int(held), auroc=roc_auc_score(yy[te], Zte @ w), n_test=int(te.sum())))
        for src in sorted(set(auth)):
            tr, te = auth == src, auth != src
            Ztr, Zte, _, _ = standardize(X[tr], X[te])
            w = Ztr[yy[tr]].mean(0) - Ztr[~yy[tr]].mean(0)
            rows.append(dict(probe=name, test="cross_author", held=str(src), auroc=roc_auc_score(yy[te], Zte @ w), n_test=int(te.sum())))
        print(name, [round(r["auroc"], 3) for r in rows if r["probe"] == name], flush=True)
    pd.DataFrame(rows).to_csv(TABLES / f"probe_generalization_{a.model}.csv", index=False)


if __name__ == "__main__":
    main()
