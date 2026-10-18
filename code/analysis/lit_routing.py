"""Where a peer's influence reaches the decision, across three architectures (interp/routing_lit.py outputs).

attn : m = lp(the disagreeing peer's answer) - lp(the agent's own previous answer) at the decision token. All mask arms and the
       `none` baseline sit in one batch and go down one attention kernel, so differences between them are exact; `none_ref`,
       `filler` and `nopeer` are ordinary forwards and are only compared with each other.
         removed(arm)   = m(none) - m(arm)
         total          = m(none_ref) - m(filler)   text-level deletion of the peer's message at matched token length
         share(arm)     = removed(arm) / total      ratio of cluster-bootstrapped means
       Reported per layer, cumulatively over layers, for every attention layer at once, and for the controls
       (own-response span of matched length, peer keys blocked for non-decision queries, peer keys blocked for every query).
span : recovery = (patched - recipient) / (donor - recipient) for a residual-stream interchange over the source descriptor, and
       on Gemma-4 also through the per-layer embedding channel.
Tables: outputs/tables/lit_routing_{layers,summary,span,models}*.csv
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
RNG = 240091


def ci(x, c):
    x = np.asarray(x, float)
    if np.isfinite(x).sum() < 5:
        return dict(est=np.nan, lower=np.nan, upper=np.nan, n_clusters=0, n=int(np.isfinite(x).sum()))
    e, lo, hi, p, n = fast_cluster_mean_ci(x, np.asarray(c))
    return dict(est=e, lower=lo, upper=hi, p=p, n_clusters=n, n=int(np.isfinite(x).sum()))


def ratio_ci(num, den, clusters, B=4000, alpha=0.05):
    """Cluster bootstrap for mean(num)/mean(den): the share of the text-level effect that the intervention removes."""
    d = pd.DataFrame(dict(n=num, d=den, c=clusters)).dropna()
    if len(d) < 5 or abs(d.d.mean()) < 1e-9:
        return dict(est=np.nan, lower=np.nan, upper=np.nan, n_clusters=0)
    g = d.groupby("c")[["n", "d"]].sum()
    N, D = g.n.to_numpy(), g.d.to_numpy()
    rng = np.random.default_rng(RNG)
    idx = rng.integers(0, len(N), size=(B, len(N)))
    bd = D[idx].sum(1)
    boots = np.where(np.abs(bd) > 1e-9, N[idx].sum(1) / np.where(np.abs(bd) > 1e-9, bd, 1), np.nan)
    lo, hi = np.nanquantile(boots, [alpha / 2, 1 - alpha / 2])
    return dict(est=float(N.sum() / D.sum()), lower=float(lo), upper=float(hi), n_clusters=len(N))


def load_attn(m, ds):
    # ds may be a comma-separated list of halves; they pool, since item ids are disjoint across them.
    parts = [x for x in ds.split(",") if x]
    dirs = [ACTS / m / f"routing_attn_{p}" for p in parts]
    have = [d for d in dirs if (d / "rows.jsonl").exists()]
    if not have:
        return None, None
    if len(have) < len(dirs):
        print(f"[routing] only {len(have)} of {len(dirs)} halves present; using what exists", flush=True)
    R = pd.concat([pd.DataFrame(read_jsonl(d / "rows.jsonl")) for d in have], ignore_index=True)
    R["m"] = [r["lp"].get(r["adv"], np.nan) - r["lp"].get(r["own"], np.nan) for _, r in R.iterrows()]
    W = R.pivot_table(index=["item_id", "agent"], columns="arm", values="m")
    # Merge the halves' metadata rather than taking one: they differ in n_episodes and skipped, and quoting one half's
    # episode count for a pooled result would understate the sample the claim rests on.
    metas = [json.load(open(x / "meta.json")) for x in have]
    meta = dict(metas[0])
    for k in ("n_episodes", "skipped"):
        if all(k in mm for mm in metas):
            meta[k] = sum(mm[k] for mm in metas)
    meta["dataset"] = ",".join(mm.get("dataset", "?") for mm in metas)
    key = R.drop_duplicates(["item_id", "agent"]).set_index(["item_id", "agent"])
    W["cluster"] = [cluster_of(i) for i, _ in W.index]
    W["domain"] = [CORE[i]["primary_domain"] for i, _ in W.index]
    W["family"] = [family_of(CORE[i]) for i, _ in W.index]
    for c in ("n_tok", "n_peer_tok", "n_own_tok", "dec_minus_peer_end", "peer_in_window", "n_dis"):
        if c in key:
            W[c] = key[c].reindex(W.index).to_numpy()
    W["total"] = W["none_ref"] - W["filler"]
    W["total_nopeer"] = W["none_ref"] - W["nopeer"]
    return W, meta


def attn_tables(m, ds, W, meta):
    # A pooled run writes its own files; the half-benchmark tables must survive, since claims still quote them until
    # every half has been pooled.
    # The tag has to say which data the table came from. "," meant pooled halves and anything else meant the
    # default half, so a run on `core` would have written lit_routing_G4_summary.csv, indistinguishable from a
    # half-benchmark table and not the name the audit looks for.
    tagm = m if ds == "lit1200" else (f"{m}_full2400" if "," in ds else f"{m}_{ds}")
    grid, attn_layers, nl = meta["grid"], meta["attn_layers"], meta["n_layers"]
    arms = [c for c in W.columns if isinstance(c, str) and (c.startswith(("peer_", "own_", "qctrl")))]
    rows = []
    for arm in arms:
        rem = W["none"] - W[arm]
        kind = ("peer_layer" if arm.startswith("peer_L") else "peer_cumulative" if arm.startswith("peer_cum") else
                "own_layer" if arm.startswith("own_L") else arm)
        layer = int("".join(ch for ch in arm if ch.isdigit())) if any(ch.isdigit() for ch in arm) else np.nan
        for scope, g in [("all", W)] + list(W.groupby("domain")):
            r = ci(g["none"] - g[arm], g.cluster)
            s = ratio_ci(g["none"] - g[arm], g["total"], g.cluster)
            rows.append(dict(model=m, arm=arm, kind=kind, layer=layer,
                             depth=(layer / (nl - 1)) if layer == layer else np.nan, domain=scope,
                             removed=r["est"], removed_lo=r["lower"], removed_hi=r["upper"], n=r["n"],
                             share=s["est"], share_lo=s["lower"], share_hi=s["upper"]))
    T = pd.DataFrame(rows)
    T.to_csv(TABLES / f"lit_routing_{tagm}_layers.csv", index=False)
    srows = []
    for name, col in [("text: peer message replaced by filler", "total"), ("text: no peers at all", "total_nopeer")]:
        r = ci(W[col], W.cluster)
        srows.append(dict(model=m, arm=name, removed=r["est"], removed_lo=r["lower"], removed_hi=r["upper"], share=1.0 if col == "total" else np.nan,
                          share_lo=np.nan, share_hi=np.nan, n=r["n"]))
    for arm, name in [("peer_all", "peer keys blocked at every attention layer (decision query)"),
                      ("peer_allq_all", "peer keys blocked for every later query at every attention layer"),
                      ("own_all", "control: own-response keys of matched length"),
                      ("qctrl_all", "control: peer keys blocked for the prefill tokens, not the decision token")]:
        if arm not in W:
            continue
        r, s = ci(W["none"] - W[arm], W.cluster), ratio_ci(W["none"] - W[arm], W["total"], W.cluster)
        srows.append(dict(model=m, arm=name, removed=r["est"], removed_lo=r["lower"], removed_hi=r["upper"],
                          share=s["est"], share_lo=s["lower"], share_hi=s["upper"], n=r["n"]))
    S = pd.DataFrame(srows)
    S["n_attention_layers"], S["n_layers"] = len(attn_layers), nl
    S["kernel_offset"] = float((W["none"] - W["none_ref"]).abs().mean())
    S.to_csv(TABLES / f"lit_routing_{tagm}_summary.csv", index=False)
    frows = []
    for fam, g in W.groupby("family"):
        if len(g) < 20:
            continue
        s = ratio_ci(g["none"] - g["peer_allq_all"], g["total"], g.cluster)
        frows.append(dict(model=m, family=fam, n=len(g), share=s["est"], share_lo=s["lower"], share_hi=s["upper"],
                          total=float(g["total"].mean())))
    # With ~100-400 episodes spread over 15 families no family clears the 20-episode floor, so this is routinely empty.
    # An empty frame writes a headerless file that raises EmptyDataError in any reader, so give it columns and say so.
    F = pd.DataFrame(frows, columns=["model", "family", "n", "share", "share_lo", "share_hi", "total"])
    F.to_csv(TABLES / f"lit_routing_{tagm}_family.csv", index=False)
    if not len(F):
        print(f"[{m}] no task family reaches 20 episodes, so the family breakdown is empty; "
              f"the domain breakdown in the layers table is the usable split", flush=True)
    print(f"\n[{m}] {len(W)} episodes, {len(attn_layers)}/{nl} layers take an attention mask; "
          f"kernel offset |none - none_ref| = {S.kernel_offset.iloc[0]:.4f}")
    print(S[["arm", "removed", "removed_lo", "removed_hi", "share", "share_lo", "share_hi"]].round(3).to_string(index=False))
    return T, S


def span_tables(m, split):
    d = ACTS / m / f"routing_span_{split}"
    if not (d / "rows.jsonl").exists():
        return None
    R = pd.DataFrame(read_jsonl(d / "rows.jsonl"))
    P = R.pivot_table(index=["item_id", "adv"], columns="arm", values="m_adv")
    P["cluster"] = [cluster_of(i) for i, _ in P.index]
    don, rec = P["donor"], P["recipient"]
    rows = []
    for arm in [c for c in P.columns if isinstance(c, str) and c not in ("donor", "recipient", "cluster")]:
        L = int(arm.split("_L")[1].split("_")[0]) if "_L" in arm else int(arm.split("ge")[1])
        kind = "residual only" if arm.startswith("resid_L") and "ple" not in arm else (
            "per-layer embeddings only" if arm.startswith("ple_ge") else "residual + per-layer embeddings")
        s = ratio_ci(P[arm] - rec, don - rec, P.cluster)
        rows.append(dict(model=m, arm=arm, kind=kind, layer=L, recovery=s["est"], lower=s["lower"], upper=s["upper"]))
    T = pd.DataFrame(rows).sort_values(["kind", "layer"])
    g = ci(don - rec, P.cluster)
    T["donor_minus_recipient"] = g["est"]
    T.to_csv(TABLES / f"lit_routing_span_{m}.csv", index=False)
    print(f"\n[{m}] span interchange, donor - recipient = {g['est']:.3f} [{g['lower']:.3f}, {g['upper']:.3f}], n={g['n']}")
    print(T.pivot_table(index="layer", columns="kind", values="recovery").round(3).to_string())
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Q9,L31,G4")
    ap.add_argument("--dataset", default="lit1200")
    ap.add_argument("--split", default="core")
    a = ap.parse_args()
    # One --dataset applied to every model cannot express this campaign: Qwen and Llama ran the benchmark in two
    # halves that must be pooled, Gemma ran it whole as `core`. A ";"-separated spec aligns to --models, and a
    # single spec still applies to all of them, so existing invocations are unchanged.
    models = a.models.split(",")
    specs = a.dataset.split(";")
    if len(specs) == 1:
        specs = specs * len(models)
    if len(specs) != len(models):
        raise SystemExit(f"[lit_routing] --dataset has {len(specs)} ';'-separated specs for {len(models)} models; "
                         f"give one per model or a single spec for all")
    allT, allS, allSp = [], [], []
    for m, ds_m in zip(models, specs):
        W, meta = load_attn(m, ds_m)
        if W is not None:
            T, S = attn_tables(m, ds_m, W, meta)
            allT.append(T)
            allS.append(S)
        sp = span_tables(m, a.split)
        if sp is not None:
            allSp.append(sp)
    if allT:
        # These two are cross-model aggregates rebuilt from scratch each run, so a run over a subset of the models
        # silently narrows them. Refuse rather than overwrite three models' worth of rows with one.
        prev = TABLES / "lit_routing_models.csv"
        if prev.exists():
            had = set(pd.read_csv(prev).model.unique())
            now = set(pd.concat(allS).model.unique())
            if had - now:
                raise SystemExit(f"[lit_routing] refusing to rebuild the cross-model tables from {sorted(now)}: "
                                 f"they currently hold {sorted(had)} and {sorted(had - now)} would be dropped. "
                                 f"Pass every model, with a per-model --dataset spec if they differ.")
        pd.concat(allT).to_csv(TABLES / "lit_routing_layers.csv", index=False)
        pd.concat(allS).to_csv(TABLES / "lit_routing_models.csv", index=False)
    if allSp:
        pd.concat(allSp).to_csv(TABLES / "lit_routing_span.csv", index=False)
    print("\n[done] lit_routing")


if __name__ == "__main__":
    main()
