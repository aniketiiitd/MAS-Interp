"""E18/E19 functional emotion directions (Anthropic-style protocol, adapted).

extract: read every emotion story and its matched neutral rewrite (plus gradients and lexical sentences) through the HF
         model; mean residual over story tokens on a normalized layer grid.
fit:     per concept, u = normalize(mean_train(h_story - h_neutral)); stories 0-2 & 4-6 train, 3 & 7 test (one agent-centered
         and one social held out). Layer chosen by held-out pairwise accuracy averaged over concepts (nested: selection
         on train-fold CV). Validation: held-out pair AUROC, specificity vs other concepts, gradient slope, lexical-probe
         similarity, BH-FDR over 171 concepts.
debate:  project E7 replay states (decision boundary) onto frozen directions -> z-scores per (item, cond, agent, turn).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats as sst
from sklearn.metrics import roc_auc_score

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "interp"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import *  # noqa
from stats import bh_fdr

E = ROOT / "data/concepts"


def cmd_extract(args):
    import torch
    from cpt import Receiver
    from run_cpt import layer_grid
    R = Receiver(args.model)
    grid = layer_grid(R.n_layers)
    stories = read_jsonl(E / "stories.jsonl")
    grads = read_jsonl(E / "gradients.jsonl")
    lex = read_jsonl(E / "lexical.jsonl")
    texts = []
    for s in stories:
        texts.append(("story", s["concept"], s["story_idx"], s["story"]))
        texts.append(("neutral", s["concept"], s["story_idx"], s["neutral"]))
        if s.get("counterfactual"):
            texts.append(("counterfactual", s["concept"], s["story_idx"], s["counterfactual"]))
    for g in grads:
        texts.append(("gradient", g["concept"], g["event_idx"] * 10 + g["level"], g["story"]))
    for i, l in enumerate(lex):
        texts.append(("lexical", l["concept"], i, l["sentence"]))
    for i in range(40):
        texts.append(("lexical_neutral", "none", i, f"{['Maya','Daniel','Priya','Tomas'][i % 4]} walked to the {['store','station','office','park'][i // 4 % 4]} {['in the morning','after lunch','at noon','later'][i // 16 % 4]}."))
    out = np.zeros((len(texts), len(grid), R.model.config.get_text_config().hidden_size), dtype=np.float32)
    t0 = time.time()
    # B was fixed at 32, tuned on a 32-layer model. The parallel emotion_stories.py pipeline was already cut to
    # --batch 8 for Gemma, which has 42 layers and per-layer embeddings, so leaving this one hardcoded would have
    # run the same model at four times that batch. Scale it with depth, as the replay token budget now does.
    B = getattr(args, "batch", 0) or max(4, int(32 * 32 / max(R.n_layers, 1)))
    print(f"[extract] {len(texts)} texts, batch {B} ({R.n_layers} layers)", flush=True)
    for b0 in range(0, len(texts), B):
        chunk = texts[b0:b0 + B]
        batch, spans = [], []
        for kind, c, idx, tx in chunk:
            msgs = [{"role": "user", "content": "Read the following passage.\n\n" + tx}]
            prefix = R.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            enc = R.tok(prefix, add_special_tokens=False, return_offsets_mapping=True)
            a = prefix.index(tx)
            b = a + len(tx)
            idxs = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if e > a and s < b]
            batch.append(enc["input_ids"])
            spans.append(idxs)
        store = {}
        for li in grid:
            R.add_capture(li, spans, store)
        R.forward(batch)
        R.clear()
        for bi in range(len(chunk)):
            for gi, li in enumerate(grid):
                out[b0 + bi, gi] = store[li][bi].mean(0).numpy()
        if (b0 // B) % 20 == 0:
            print(f"[progress] {b0}/{len(texts)} {(time.time()-t0)/60:.1f} min", flush=True)
    d = ACTS / args.model / "concepts"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "story_acts.npy", out.astype(np.float16))
    write_jsonl(d / "story_index.jsonl", [dict(kind=k, concept=c, idx=i) for k, c, i, _ in texts])
    json.dump(dict(grid=grid, sha=R.sha), open(d / "meta.json", "w"))
    print(f"[done] emotion extract {args.model}: {len(texts)} passages in {(time.time()-t0)/60:.1f} min", flush=True)


def cmd_fit(args):
    d = ACTS / args.model / "concepts"
    A = np.load(d / "story_acts.npy").astype(np.float32)
    idx = pd.DataFrame(read_jsonl(d / "story_index.jsonl"))
    meta = json.load(open(d / "meta.json"))
    tax = json.load(open(E / "taxonomy.json"))
    stories = pd.DataFrame(read_jsonl(E / "stories.jsonl"))
    leak = {(r.concept, r.story_idx): r.lexical_leak for r in stories.itertuples()}
    concepts = [c for g in tax["taxonomy"].values() for c in g]
    grid = meta["grid"]
    has_cf = set(idx[idx.kind == "counterfactual"].concept)
    contrast_kind = lambda c: "counterfactual" if c in has_cf else "neutral"  # minimal pairs for behavioral/moral concepts
    test_idx = {3, 7}
    rows = []
    best_layer, best_score = None, -1
    # layer selection: mean over concepts of train-only leave-one-story-out pair accuracy
    layer_scores = {}
    for gi, L in enumerate(grid):
        H = A[:, gi]
        mu, sd = H.mean(0), H.std(0) + 1e-3
        Z = (H - mu) / sd
        accs = []
        for c in concepts:
            s = idx[(idx.concept == c) & (idx.kind == "story")]
            n = idx[(idx.concept == c) & (idx.kind == contrast_kind(c))]
            tr = [i for i in s.idx if i not in test_idx]
            diffs = {i: Z[s[s.idx == i].index[0]] - Z[n[n.idx == i].index[0]] for i in tr}
            for hold in tr:
                u = np.mean([diffs[i] for i in tr if i != hold], 0)
                accs.append(float(diffs[hold] @ u > 0))
        layer_scores[L] = np.mean(accs)
    best_layer = max(layer_scores, key=layer_scores.get)
    gi = grid.index(best_layer)
    H = A[:, gi]
    mu, sd = H.mean(0), H.std(0) + 1e-3
    Z = (H - mu) / sd
    U = {}
    for c in concepts:
        s = idx[(idx.concept == c) & (idx.kind == "story")]
        n = idx[(idx.concept == c) & (idx.kind == contrast_kind(c))]
        tr = [i for i in s.idx if i not in test_idx and not leak.get((c, i), False)] or [i for i in s.idx if i not in test_idx]
        u = np.mean([Z[s[s.idx == i].index[0]] - Z[n[n.idx == i].index[0]] for i in tr], 0)
        U[c] = u / (np.linalg.norm(u) + 1e-8)
    Umat = np.stack([U[c] for c in concepts])
    lex_neu = Z[idx[idx.kind == "lexical_neutral"].index].mean(0)
    for c in concepts:
        s = idx[(idx.concept == c) & (idx.kind == "story")]
        n = idx[(idx.concept == c) & (idx.kind == contrast_kind(c))]
        te_s = [Z[s[s.idx == i].index[0]] @ U[c] for i in test_idx]
        te_n = [Z[n[n.idx == i].index[0]] @ U[c] for i in test_idx]
        # specificity: held-out stories of this concept vs held-out stories of all other concepts, on this direction
        other = idx[(idx.kind == "story") & (idx.concept != c) & (idx.idx.isin(test_idx))].index
        so = Z[other] @ U[c]
        spec_auc = roc_auc_score(np.r_[np.ones(len(te_s)), np.zeros(len(so))], np.r_[te_s, so])
        pair_auc = roc_auc_score(np.r_[np.ones(len(te_s)), np.zeros(len(te_n))], np.r_[te_s, te_n])
        # rank of own direction among all directions for the held-out stories
        own_rank = np.mean([(Z[s[s.idx == i].index[0]] @ Umat.T >= Z[s[s.idx == i].index[0]] @ U[c]).sum() for i in test_idx])
        lx = idx[(idx.kind == "lexical") & (idx.concept == c)].index
        lex_dir = Z[lx].mean(0) - lex_neu
        lex_cos = float(lex_dir @ U[c] / (np.linalg.norm(lex_dir) + 1e-8))
        g = idx[(idx.kind == "gradient") & (idx.concept == c)]
        slope, slope_p = np.nan, np.nan
        if len(g):
            lv = (g.idx % 10).to_numpy()
            sc = Z[g.index] @ U[c]
            res = sst.linregress(lv, sc)
            slope, slope_p = res.slope, res.pvalue
        # permutation p for pair separation: sign-flip of held-out differences on random directions
        rng = np.random.default_rng(h64("emo", c) % 2**32)
        rnd = rng.standard_normal((200, Z.shape[1])).astype(np.float32)
        rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
        gap = np.mean(te_s) - np.mean(te_n)
        null = [np.mean([Z[s[s.idx == i].index[0]] @ r for i in test_idx]) - np.mean([Z[n[n.idx == i].index[0]] @ r for i in test_idx]) for r in rnd]
        p = (np.sum(np.array(null) >= gap) + 1) / (len(null) + 1)
        rows.append(dict(concept=c, group=next(gname for gname, cs in tax["taxonomy"].items() if c in cs), layer=best_layer,
                         contrast=contrast_kind(c),
                         heldout_gap=gap, pair_auroc=pair_auc, specificity_auroc=spec_auc, own_rank=own_rank,
                         lexical_cos=lex_cos, gradient_slope=slope, gradient_p=slope_p, perm_p=p,
                         leak_rate=float(np.mean([leak.get((c, i), False) for i in range(8)]))))
    V = pd.DataFrame(rows)
    V["q_value"] = bh_fdr(V.perm_p)
    V["passes_v1"] = (V.q_value < 0.05) & (V.specificity_auroc > 0.75)
    V.loc[V.concept.isin(tax["prereg_debate"]), "passes_v1"] &= (V.gradient_slope > 0)
    # v2 (stricter, added before any debate/trajectory analysis): leave-one-story-out pair accuracy over all 8 minimal pairs
    loo = {}
    for c in concepts:
        s_ = idx[(idx.concept == c) & (idx.kind == "story")]
        n_ = idx[(idx.concept == c) & (idx.kind == contrast_kind(c))]
        diffs = {i: Z[s_[s_.idx == i].index[0]] - Z[n_[n_.idx == i].index[0]] for i in s_.idx if (n_.idx == i).any()}
        acc = [float(diffs[i] @ np.mean([diffs[j] for j in diffs if j != i], 0) > 0) for i in diffs]
        loo[c] = float(np.mean(acc)) if acc else np.nan
    V["loo_pair_acc"] = V.concept.map(loo)
    V["passes"] = V.passes_v1 & (V.loo_pair_acc >= 0.75)
    V["kind"] = V.group.map(lambda g: g if g in ("moral", "social_epistemic") else "emotion")
    V.to_csv(TABLES / f"concept_validation_{args.model}.csv", index=False)
    np.save(d / "directions.npy", Umat)
    json.dump(dict(concepts=concepts, layer=best_layer, grid_index=gi, mu=mu.tolist(), sd=sd.tolist(), layer_scores=layer_scores),
              open(d / "directions_meta.json", "w"))
    print(f"[done] emotion fit {args.model}: layer {best_layer}, pass {int(V.passes.sum())}/{len(V)}; "
          f"median specificity {V.specificity_auroc.median():.3f}, median own rank {V.own_rank.median():.1f}", flush=True)


def cmd_debate(args):
    """Project replay states (decision + reading sites) onto frozen concept directions; z-scored against turn-0 states."""
    d = ACTS / args.model / "concepts"
    U = np.load(d / "directions.npy")
    m = json.load(open(d / "directions_meta.json"))
    rep = ACTS / args.model / args.replay
    rmeta = json.load(open(rep / "meta.json"))
    gi = rmeta["grid"].index(m["layer"])
    idx = pd.DataFrame(read_jsonl(rep / "index.jsonl"))
    A = np.load(rep / "acts.npy", mmap_mode="r")
    solo_rows = idx[idx.turn == 0].row.to_numpy()
    out = idx.drop(columns=["lp_letters"]).copy()
    g = U.mean(0)
    g = g / np.linalg.norm(g)  # shared "story vs flat rewrite" component across all concepts (general affect/style axis)
    Ug = np.vstack([U, g[None]])
    for site, sname in enumerate(["dec", "read"]):
        X = A[:, site, gi].astype(np.float32)
        mu, sd = X[solo_rows].mean(0), X[solo_rows].std(0) + 1e-3
        S = ((X - mu) / sd) @ Ug.T
        S = (S - S[solo_rows].mean(0)) / (S[solo_rows].std(0) + 1e-6)
        for j, c in enumerate(m["concepts"] + ["GENERAL"]):
            out[f"{sname}_{c}"] = S[:, j].astype(np.float32)
    out.to_parquet(TABLES / f"concept_traj_{args.model}_{args.replay}.parquet")
    print(f"[done] concept trajectories {args.model} {args.replay}: {len(out)} states x {len(m['concepts'])} concepts x 2 sites", flush=True)


def cmd_observer(args):
    """Project observer-replay states with the PARTICIPANT replay's turn-0 normalization so participant − observer is on one scale."""
    d = ACTS / args.model / "concepts"
    U = np.load(d / "directions.npy")
    m = json.load(open(d / "directions_meta.json"))
    g = U.mean(0)
    g = g / np.linalg.norm(g)
    Ug = np.vstack([U, g[None]])
    rep = ACTS / args.model / args.replay
    obs = ACTS / args.model / args.observer
    gi_r = json.load(open(rep / "meta.json"))["grid"].index(m["layer"])
    gi_o = json.load(open(obs / "meta.json"))["grid"].index(m["layer"])
    ridx = pd.DataFrame(read_jsonl(rep / "index.jsonl"))
    oidx = pd.DataFrame(read_jsonl(obs / "index.jsonl"))
    RA = np.load(rep / "acts.npy", mmap_mode="r")
    OA = np.load(obs / "acts.npy", mmap_mode="r")
    solo = ridx[ridx.turn == 0].row.to_numpy()
    out = oidx.drop(columns=["lp_letters"]).copy()
    for site, sname in enumerate(["dec", "read"]):
        XR = RA[solo, site, gi_r].astype(np.float32)
        mu, sd = XR.mean(0), XR.std(0) + 1e-3
        SR = ((XR - mu) / sd) @ Ug.T
        smu, ssd = SR.mean(0), SR.std(0) + 1e-6
        XO = OA[:, site, gi_o].astype(np.float32)
        SO = (((XO - mu) / sd) @ Ug.T - smu) / ssd
        for j, c in enumerate(m["concepts"] + ["GENERAL"]):
            out[f"{sname}_{c}"] = SO[:, j].astype(np.float32)
    out.to_parquet(TABLES / f"concept_obs_{args.model}_{args.observer}.parquet")
    print(f"[done] observer concept projection {args.model}: {len(out)} states", flush=True)


CPT_EXPECT = {  # pre-registered expected sign of (manipulated - control) on the receiver's decision-boundary state
    "expert_vs_anon": ["trust_in_source", "deference_to_expertise", "deferential", "confident", "reassured"],
    "disagree_vs_agree": ["doubtful", "uncertain", "pressured", "conflicted", "dissent", "moral_uncertainty"],
    "strong_vs_weak": ["belief_updating", "confident"],
}


def cmd_validate_cpt(args):
    """Construct validity: CPT manipulations must move the pre-registered concepts in the expected direction."""
    d = ACTS / args.model / "concepts"
    U = np.load(d / "directions.npy")
    m = json.load(open(d / "directions_meta.json"))
    run = ACTS / args.model / "cpt_core"
    meta = json.load(open(run / "meta.json"))
    gi = meta["grid"].index(m["layer"])
    W = pd.DataFrame(read_jsonl(run / "worlds.jsonl"))
    A = np.load(run / "acts.npy", mmap_mode="r")
    splits = json.load(open(ROOT / "data/splits/splits.json"))
    keep = W.item_id.isin(set(splits["release_600"])).to_numpy()
    rows = np.where(keep & W.qual.isin(["strong", "weak"]).to_numpy())[0]
    X = A[np.sort(rows), 0, gi].astype(np.float32)  # position 0 = answer boundary
    Wr = W.iloc[np.sort(rows)].reset_index(drop=True)
    mu, sd = X.mean(0), X.std(0) + 1e-3
    S = ((X - mu) / sd) @ U.T
    S = (S - S.mean(0)) / (S.std(0) + 1e-6)
    out = []
    contrasts = {
        "expert_vs_anon": (Wr.prov == "expert", Wr.prov == "anon"),
        "disagree_vs_agree": (Wr.agree == False, Wr.agree == True),  # noqa: E712
        "strong_vs_weak": (Wr.qual == "strong", Wr.qual == "weak"),
    }
    for name, (a, b) in contrasts.items():
        # within-item paired difference, then mean over items with item bootstrap
        df = pd.DataFrame(S, columns=m["concepts"])
        df["item"] = Wr.item_id
        da = df[a.to_numpy()].groupby("item").mean()
        db = df[b.to_numpy()].groupby("item").mean()
        diff = (da - db).dropna()
        rng = np.random.default_rng(240091)
        boot = np.stack([diff.iloc[rng.integers(0, len(diff), len(diff))].mean().to_numpy() for _ in range(1000)])
        for j, c in enumerate(m["concepts"]):
            est = diff[c].mean()
            lo, hi = np.quantile(boot[:, j], [0.025, 0.975])
            out.append(dict(contrast=name, concept=c, estimate=est, lower=lo, upper=hi, prereg=c in CPT_EXPECT.get(name, []),
                            p=2 * min((boot[:, j] <= 0).mean(), (boot[:, j] >= 0).mean())))
    O = pd.DataFrame(out)
    O["q"] = bh_fdr(O.p)
    O.to_csv(TABLES / f"concept_cpt_validity_{args.model}.csv", index=False)
    print(O[O.prereg].round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "fit", "debate", "validate_cpt", "observer"])
    ap.add_argument("--observer", default="observer_release_600_T12")
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--batch", type=int, default=0, help="texts per forward; 0 scales from model depth")
    ap.add_argument("--replay", default="replay_core_release_600")
    a = ap.parse_args()
    {"extract": cmd_extract, "fit": cmd_fit, "debate": cmd_debate, "validate_cpt": cmd_validate_cpt, "observer": cmd_observer}[a.cmd](a)
