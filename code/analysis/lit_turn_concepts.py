"""How emotion and moral states move across the turns of a debate, against the matched no-communication condition.

Built for the 12-turn run. For every turn, every agent's state at the point just before it writes is projected onto the
emotion and moral concept directions, and the same quantity is computed in the no-communication condition on the same
items. What is plotted is always the difference between the two, so anything that comes merely from the context getting
longer cancels.

Significance comes from the permuted-label refits saved beside the real directions: the same stories, pooling and
within-topic centring with the concept labels shuffled inside each topic. A concept counts as moving only if its
debate-minus-solo difference exceeds what those permuted directions produce at the same turn and depth.

Outputs: outputs/tables/lit_turn_concepts_<tag>_<M>_<dataset>.csv
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

CORE = {r["id"]: r for r in load_core()}
RAW_LIT = ROOT / "outputs/raw_lit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="release_600_t12")
    ap.add_argument("--tag", default="emotion_stories")
    ap.add_argument("--layers", default="", help="comma list; default every layer in the replay grid")
    a = ap.parse_args()
    cd = ACTS / a.model / a.tag
    meta = json.load(open(cd / "meta.json"))
    concepts = meta["concepts"]
    V, VN = np.load(cd / "vectors.npz"), np.load(cd / "null_vectors.npz")

    rd = ACTS / a.model / f"lit_replay_{a.dataset}"
    rmeta = json.load(open(rd / "meta.json"))
    grid = rmeta["grid"]
    si = rmeta["sites"].index("reading")
    layers = [int(x) for x in a.layers.split(",") if x] or grid
    I = pd.DataFrame(read_jsonl(rd / "index.jsonl"))
    A = np.load(rd / "acts.npy", mmap_mode="r")

    D = I[(~I["query"].map(lambda q: isinstance(q, str))) & (I["round"] > 0) & (I.cond.isin(["debate", "reflect"]))].copy()
    D["family"] = D.item_id.map(lambda i: family_of(CORE[i]))
    D["domain"] = D.item_id.map(lambda i: CORE[i]["primary_domain"])
    R = int(D["round"].max())
    print(f"[turn-concepts] {a.tag} {a.model} {a.dataset}: {len(D)} states, {R} turns, layers {layers}", flush=True)

    rows = []
    for L in layers:
        if f"L{L}" not in V.files:
            continue
        gi = grid.index(L)
        Vr = V[f"L{L}"]
        Vn = VN[f"L{L}"]
        for k in range(1, R + 1):
            for scope, col in (("all", None), ("domain", "domain")):
                groups = [("all", D[D["round"] == k])] if col is None else list(D[D["round"] == k].groupby(col))
                for gname, g in groups:
                    di = g.index[g.cond == "debate"]
                    ri = g.index[g.cond == "reflect"]
                    if len(di) < 30 or len(ri) < 30:
                        continue
                    Hd = np.asarray(A[D.loc[di, "row"].to_numpy(), si, gi], np.float32)
                    Hr = np.asarray(A[D.loc[ri, "row"].to_numpy(), si, gi], np.float32)
                    mu = np.concatenate([Hd, Hr]).mean(0)
                    sd = np.concatenate([Hd, Hr]).std(0) + 1e-6
                    Hd, Hr = (Hd - mu) / sd, (Hr - mu) / sd
                    diff = (Hd @ Vr.T).mean(0) - (Hr @ Vr.T).mean(0)
                    nd = np.stack([(Hd @ Vn[b].T).mean(0) - (Hr @ Vn[b].T).mean(0) for b in range(Vn.shape[0])])
                    band = np.nanquantile(np.abs(nd), 0.95)
                    for j, c in enumerate(concepts):
                        rows.append(dict(layer=L, round=k, scope=scope, group=gname, concept=c,
                                         debate_minus_solo=float(diff[j]), null_band95=float(band),
                                         moves=bool(abs(diff[j]) > band), n_debate=len(di), n_solo=len(ri)))
        print(f"  layer {L:2d} done", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(TABLES / f"lit_turn_concepts_{a.tag}_{a.model}_{a.dataset}.csv", index=False)
    s = T[T.scope == "all"].groupby("round").moves.mean() * 100
    print("\nshare of concepts moving beyond the shuffled-label band, by turn:")
    print("  " + "  ".join(f"t{k}:{v:.0f}%" for k, v in s.items()))
    print(f"[done] outputs/tables/lit_turn_concepts_{a.tag}_{a.model}_{a.dataset}.csv")


if __name__ == "__main__":
    main()
