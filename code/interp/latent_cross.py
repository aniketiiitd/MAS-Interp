"""Does one model's committed decision state move *another* model's decision, and does it need the same number of directions?

interp/latent_lit.py writes a peer's decision state into a receiver that is a copy of the same model. Here sender and receiver are
different models (different tokenizer, depth and width), so the state has to be carried across spaces first.

Construction. Episodes come from the receiver model's own debates (item, round, receiver g, disagreeing peer j), exactly as in the
within-model design: the receiver reads the round message with every peer replaced by its bare "{final answer: (X)}", and we write a
state at the "(" of peer j's bare answer. What we write is the *sender model's* state, formed by letting the sender read the same
conversation peer j read and teacher-forcing peer j's response up to its own "(" - the sender's commitment state on that text.

Mapping. On a disjoint half of the items, the same texts give matched pairs (sender state, receiver state at the same token), and we
fit two maps per relative depth: ridge regression (unconstrained, lambda chosen on a held-out fifth) and orthogonal Procrustes with
a scale (structure-preserving, and defined for unequal widths). Held-out mapping quality is reported as cosine and as retrieval
accuracy of the matched state among the other held-out states, for every (sender depth, receiver depth) pair.

Arms, all measured on byte-identical text inside the same batch: the mapped state (ridge and Procrustes), the receiver's own state on
the same text (the within-model ceiling), rank-k writes h <- h + P_k(s_hat - h) with k = 1..256 in bases fitted on the fitting half
(principal subspace of the mapped state differences, and the effect-aligned subspace), and controls: a map fitted on shuffled pairs,
a mapped state from another item whose sender gave the same letter, a mapped state from a random other episode, the mapped *mean*
sender state (item-independent), the raw unmapped state (truncated or zero-padded), a norm-matched random vector, and a random
rotation of the mapped difference.

Endpoint and transfer follow latent_lit: m = lp(peer's answer) - lp(receiver's previous answer), T = [m(arm) - m(W_bare)] / [m(W_textj)
- m(W_bare)]. Every episode also records the margin toward the item's evaluation target and whether the sender's answer was correct.

Only the residual stream is written. With a Gemma-4 E-series receiver the per-layer embeddings keep re-injecting the token's own
identity below the write, so the write competes with it; the within-model arm is measured the same way, so the ratio between them
is unaffected.

Output: ACTS/<receiver>/lit_latentx_<sender>to<receiver>_<dataset>/{rows.jsonl, map.json, meta.json}
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time

import numpy as np
import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "interp"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runner"))
from common import *  # noqa
from cpt import Receiver
from prompts_lit import N_AGENTS, answer_span, messages

RAW_LIT = ROOT / "outputs/raw_lit"
PREFILL = "{final answer: ("
PEERS_HEAD = "These are the recent opinions from other agents: "
PEER_PREFIX = "One of the agents' response: "
BARE_PREFIX = "{final answer: ("
INSTR_MARK = "Use these opinions carefully"
# relative depths, so the same design runs on 32-layer (Qwen, Llama) and 42-layer (Gemma) models and pairs them by depth share
F_FULL = (0.29, 0.39, 0.52, 0.61, 0.81)
F_PROC = (0.39, 0.52, 0.61)
F_RANK = (0.52, 0.39)  # primary first: full k sweep and the effect basis run only at the primary depth
F_CTRL = 0.52
KS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
KS2 = (1, 4, 16, 64, 256)
LAM_GRID = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3)  # relative to mean(diag(X^T X))
CHUNK = 12


def bare_answer(a):
    return f"{{final answer: ({a})}}"


def tok_at(offsets, char_idx):
    for i, (s, e) in enumerate(offsets):
        if s <= char_idx < e:
            return i
    return None


def peer_spans(content, rendered):
    """Char offset of each peer's rendered response inside a round message built by prompts_lit.debate_prompt."""
    cur, out = len(PEERS_HEAD), []
    for o in rendered:
        piece = PEER_PREFIX + o.strip()
        s = content.index(piece, max(cur - 1, 0))
        out.append(s + len(PEER_PREFIX))
        cur = s + len(piece) + 1
    return out


def chat(tok, msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def ctx_for(v, recs, i, agent, r):
    """The prompt the agent saw when it produced its round-r response (model-independent; rendered per model later)."""
    if r == 0:
        return messages(v, 0, None, None, "debate")
    pc = "r0" if r - 1 == 0 else "debate"
    own = recs.get((i, pc, agent, r - 1, None))
    others = [recs.get((i, pc, h, r - 1, None)) for h in range(N_AGENTS) if h != agent]
    if own is None or any(x is None for x in others):
        return None
    return messages(v, r, own["text"], [x["text"] for x in others], "debate")


def spec_for(v, recs, i, k, g, j, pc):
    """Everything about one (item, round, receiver, peer) episode that does not depend on a tokenizer."""
    prev = {h: recs[(i, pc, h, k - 1, None)] for h in range(N_AGENTS)}
    a = {h: prev[h]["answer"] for h in range(N_AGENTS)}
    t = {h: prev[h]["text"] for h in range(N_AGENTS)}
    others_idx = [h for h in range(N_AGENTS) if h != g]
    jj = others_idx.index(j)
    r_bare = [bare_answer(a[h]) for h in others_idx]
    r_textj = [t[h] if h == j else bare_answer(a[h]) for h in others_idx]
    sp = answer_span(t[j].strip(), v)
    if sp is None or t[j].strip()[sp[1] - 1] != "(":
        return None
    msgs_s = ctx_for(v, recs, i, j, k - 1)
    if msgs_s is None:
        return None
    return dict(item_id=i, round=k, receiver=g, sender_agent=j, jj=jj, r_bare=r_bare,
                w_none=messages(v, k, t[g], None, "reflect"), w_bare=messages(v, k, t[g], r_bare, "debate"),
                w_textj=messages(v, k, t[g], r_textj, "debate"),
                w_text=messages(v, k, t[g], [t[h] for h in others_idx], "debate"),
                sender_msgs=msgs_s, sender_prefix=t[j].strip()[: sp[1]],
                own=a[g], sender=a[j], labels=v["labels"], target=v["target"])


def sender_ids(tok, s, max_tokens):
    ids = tok(chat(tok, s["sender_msgs"]) + s["sender_prefix"], add_special_tokens=False)["input_ids"]
    return None if len(ids) > max_tokens else ids


def receiver_build(tok, s, max_tokens):
    """World token ids for the receiver plus the patch position (the "(" of peer j's bare answer in W_bare)."""
    worlds = {}
    for name in ("none", "bare", "textj", "text"):
        prefix = chat(tok, s[f"w_{name}"])
        enc = tok(prefix + PREFILL, add_special_tokens=False, return_offsets_mapping=True)
        worlds[name] = (enc["input_ids"], enc["offset_mapping"], prefix)
    if max(len(w[0]) for w in worlds.values()) > max_tokens:
        return None
    ids_b, off_b, pre_b = worlds["bare"]
    content = s["w_bare"][-1]["content"]
    base = pre_b.rindex(content)
    starts = peer_spans(content, s["r_bare"])
    p_patch = tok_at(off_b, base + starts[s["jj"]] + len(BARE_PREFIX) - 1)
    p_nonslot = tok_at(off_b, base + content.index(INSTR_MARK) + 12)
    if p_patch is None or p_nonslot is None:
        return None
    return dict(worlds={n: w[0] for n, w in worlds.items()}, p_patch=p_patch, p_nonslot=p_nonslot)


def capture(R, ids_list, positions, layers):
    """One forward over a right-padded batch; returns {layer: [state per row]} (float32 cpu) and the log-probs."""
    store = {}
    for L in layers:
        R.add_capture(L, positions, store)
    lp = R.forward(ids_list)
    R.clear()
    return {L: list(store[L]) for L in layers}, lp


def margin_grads(R, ids, pos, layers, la, lb):
    """d margin / d h_L[pos] for every layer in `layers` (one forward + backward, float32 cpu)."""
    leaves, handles = {}, []

    def mk(L):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            z = torch.zeros_like(h, requires_grad=True)
            leaves[L] = z
            h2 = h + z
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        return hook

    for L in layers:
        handles.append(R.layers[L].register_forward_hook(mk(L)))
    inp = torch.tensor([ids], device=R.device)
    try:
        with torch.enable_grad():
            out = R.model.model(input_ids=inp, use_cache=False)
            hs = out.last_hidden_state[0, -1]
            logits = torch.nn.functional.linear(hs.float(), R.W32)
            if R.softcap:
                logits = torch.tanh(logits / R.softcap) * R.softcap
            lp = torch.log_softmax(logits, -1)
            ia = torch.tensor(R._letter_ids[la], device=R.device)
            ib = torch.tensor(R._letter_ids[lb], device=R.device)
            m = torch.logsumexp(lp[ia], 0) - torch.logsumexp(lp[ib], 0)
            m.backward()
    finally:
        for h in handles:
            h.remove()
    return {L: leaves[L].grad[0, pos].detach().float().cpu().clone() for L in layers}


def orthobasis(M, k):
    U, S, V = torch.linalg.svd(M.to(torch.float32), full_matrices=False)
    return V[:k].contiguous()


def effect_basis(D, G, k):
    """Top-k eigenvectors of the symmetrised sum_i (delta_i g_i^T + g_i delta_i^T)/2: the subspace that maximises the
    first-order predicted change of the margin."""
    S = (D.T @ G + G.T @ D) / (2 * len(D))
    w, Vec = torch.linalg.eigh(S.to(torch.float32))
    order = torch.argsort(w.abs(), descending=True)[:k]
    return Vec[:, order].T.contiguous()


def project(basis, k, delta):
    B = basis[:k]
    return B.T @ (B @ delta)


# ---------------------------------------------------------------------------
# maps between the two residual spaces
# ---------------------------------------------------------------------------
def ridge_map(X, Y, lam_rel):
    """Least squares with ridge on centred states: y_hat = (x - mx) W + my."""
    mx, my = X.mean(0, keepdim=True), Y.mean(0, keepdim=True)
    Xc, Yc = X - mx, Y - my
    A = Xc.T @ Xc
    lam = lam_rel * float(torch.diagonal(A).mean())
    A = A + lam * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    W = torch.linalg.solve(A, Xc.T @ Yc)
    return dict(kind="ridge", mx=mx, my=my, W=W, lam_rel=lam_rel)


def proc_map(X, Y):
    """Orthogonal Procrustes with a scale: y_hat = s (x - mx) Q + my, Q^T Q = I on the smaller side."""
    mx, my = X.mean(0, keepdim=True), Y.mean(0, keepdim=True)
    Xc, Yc = X - mx, Y - my
    U, S, Vh = torch.linalg.svd(Xc.T @ Yc, full_matrices=False)
    Q = U @ Vh
    scale = float(S.sum() / (Xc * Xc).sum().clamp_min(1e-9))
    return dict(kind="proc", mx=mx, my=my, W=Q * scale, lam_rel=None)


def apply_map(M, x):
    return (x - M["mx"]) @ M["W"] + M["my"]


def map_quality(M, X, Y):
    """Held-out cosine (centred and raw) and retrieval of the matched state among the other held-out states."""
    P = apply_map(M, X)
    Pc, Yc = P - M["my"], Y - M["my"]
    cos_c = torch.nn.functional.cosine_similarity(Pc, Yc, dim=1)
    cos_r = torch.nn.functional.cosine_similarity(P, Y, dim=1)
    A = torch.nn.functional.normalize(Pc, dim=1) @ torch.nn.functional.normalize(Yc, dim=1).T
    rank = (A > A.diagonal().unsqueeze(1)).sum(1)
    return dict(cos=float(cos_c.mean()), cos_raw=float(cos_r.mean()), top1=float((rank == 0).float().mean()),
                mrr=float((1.0 / (rank + 1).float()).mean()), n=int(len(X)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sender", required=True)
    ap.add_argument("--receiver", required=True)
    ap.add_argument("--dataset", default="lit1200")
    ap.add_argument("--transcripts", default=None, help="model whose debates supply the episodes (default: the receiver)")
    ap.add_argument("--n_map", type=int, default=500, help="matched texts used to fit the maps (fitting half of the items)")
    ap.add_argument("--n_basis", type=int, default=150, help="fitting episodes used for the rank bases and the gradients")
    ap.add_argument("--n_eval", type=int, default=300)
    ap.add_argument("--max_tokens", type=int, default=3200)
    ap.add_argument("--budget_min", type=float, default=55.0, help="stop the evaluation loop after this many minutes")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--smoke", action="store_true", help="one depth and two ranks: a code-coverage run, not a measurement")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.smoke:
        globals().update(F_FULL=(0.52,), F_PROC=(0.52,), F_RANK=(0.52,), F_CTRL=0.52, KS=(1, 16), KS2=(1,))
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    t0 = time.time()

    # ---- episodes from the transcript model's debates
    tm = a.transcripts or a.receiver
    d = RAW_LIT / tm / a.dataset
    rounds = json.load(open(d / "meta.json"))["rounds"]
    recs = {}
    for f in d.glob("*.jsonl"):
        for x in read_jsonl(f):
            recs[(x["item_id"], x["cond"], x["agent"], x["round"], x.get("query"))] = x
    core = {r["id"]: r for r in load_core()}
    views = {i: item_view(core[i]) for i in sorted({k_[0] for k_ in recs})}
    views = {i: v for i, v in views.items() if v["kind"] != "numeric" and v["labels"]}
    eps = []
    for i, v in views.items():
        for k in range(1, rounds + 1):
            pc = "r0" if k == 1 else "debate"
            if any((i, pc, h, k - 1, None) not in recs for h in range(N_AGENTS)):
                continue
            ans = {h: recs[(i, pc, h, k - 1, None)]["answer"] for h in range(N_AGENTS)}
            for g in range(N_AGENTS):
                if ans[g] is None:
                    continue
                for j in range(N_AGENTS):
                    if j != g and ans[j] is not None and ans[j] != ans[g]:
                        eps.append((i, k, g, j, pc))
    eps.sort(key=lambda e: h64("latentxep", e[0], e[1], e[2], e[3]))
    half = lambda i: h64("latentxsplit", i) % 2
    specs_fit, specs_ev = [], []
    for e in eps:
        s = spec_for(views[e[0]], recs, e[0], e[1], e[2], e[3], e[4])
        if s is None:
            continue
        (specs_fit if half(e[0]) == 0 else specs_ev).append(s)
        if len(specs_fit) >= a.n_map and len(specs_ev) >= a.n_eval:
            break
    specs_fit, specs_ev = specs_fit[: a.n_map], specs_ev[: a.n_eval]
    print(f"[stage] {len(eps)} disagreement episodes; {len(specs_fit)} for the maps, {len(specs_ev)} for evaluation", flush=True)
    # FULL_CPTSWAP_G4 exited rc=0 in 0.0 minutes having selected no items, wrote a zero-byte rows.jsonl, and its
    # wait-gated consumer then crashed four frames deep inside pandas. The same shape is reachable here: if the
    # transcripts yield no disagreement episodes, every loop below is empty and the run "succeeds". Fail where the
    # cause is still legible.
    if not specs_fit or not specs_ev:
        raise SystemExit(f"[latent_cross] nothing to work with: {len(eps)} disagreement episodes in "
                         f"raw_lit/{tm}/{a.dataset} gave {len(specs_fit)} fitting and {len(specs_ev)} evaluation "
                         f"specs. Check that the transcripts exist and that the agents actually disagreed.")

    # ---- pass 1: the sender model's commitment states on those texts
    S = Receiver(a.sender, dtype=dtype, device=a.device)
    fS = {f: int(round(f * (S.n_layers - 1))) for f in F_FULL}
    print(f"[stage] sender {a.sender}: {S.n_layers} layers, depths {sorted(fS.values())}", flush=True)
    send = {}
    for nm, specs in (("fit", specs_fit), ("eval", specs_ev)):
        batch, keys = [], []
        for si, s in enumerate(specs):
            ids = sender_ids(S.tok, s, a.max_tokens)
            if ids is not None:
                batch.append(ids)
                keys.append((nm, si))
            if len(batch) == 4 or (si == len(specs) - 1 and batch):
                st, _ = capture(S, batch, [len(x) - 1 for x in batch], sorted(fS.values()))
                for r, key in enumerate(keys):
                    send[key] = {f: st[L][r] for f, L in fS.items()}
                batch, keys = [], []
        print(f"[stage] sender states for {sum(1 for k_ in send if k_[0] == nm)}/{len(specs)} {nm} texts "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    d_send = send[next(iter(send))][F_FULL[0]].shape[0]
    del S.model, S
    gc.collect()
    torch.cuda.empty_cache()

    # ---- pass 2: the receiver
    R = Receiver(a.receiver, dtype=dtype, device=a.device)
    fR = {f: int(round(f * (R.n_layers - 1))) for f in F_FULL}
    LR = sorted(fR.values())
    print(f"[stage] receiver {a.receiver}: {R.n_layers} layers, depths {LR} ({(time.time()-t0)/60:.1f} min)", flush=True)
    dev = R.device

    recv_fit, ok_fit = {}, []
    batch, keys = [], []
    for si, s in enumerate(specs_fit):
        if ("fit", si) not in send:
            continue
        ids = sender_ids(R.tok, s, a.max_tokens)
        if ids is not None:
            batch.append(ids)
            keys.append(si)
        if len(batch) == 4 or (si == len(specs_fit) - 1 and batch):
            st, _ = capture(R, batch, [len(x) - 1 for x in batch], LR)
            for r, si_ in enumerate(keys):
                recv_fit[si_] = {f: st[L][r] for f, L in fR.items()}
                ok_fit.append(si_)
            batch, keys = [], []
    ok_fit = sorted(ok_fit)
    print(f"[stage] {len(ok_fit)} matched (sender, receiver) pairs ({(time.time()-t0)/60:.1f} min)", flush=True)

    # ---- fit the maps on the matched pairs, holding out a fifth for lambda and for the reported quality
    tr = [i for i in ok_fit if h64("latentxmap", specs_fit[i]["item_id"]) % 5 != 0]
    te = [i for i in ok_fit if h64("latentxmap", specs_fit[i]["item_id"]) % 5 == 0]
    X = {f: torch.stack([send[("fit", i)][f] for i in ok_fit]).to(dev) for f in F_FULL}
    Y = {f: torch.stack([recv_fit[i][f] for i in ok_fit]).to(dev) for f in F_FULL}
    itr = torch.tensor([ok_fit.index(i) for i in tr], device=dev)
    ite = torch.tensor([ok_fit.index(i) for i in te], device=dev)
    quality, maps = [], {}
    lam_best = {}
    for f in F_FULL:
        best = None
        for lam in LAM_GRID:
            M = ridge_map(X[f][itr], Y[f][itr], lam)
            q = map_quality(M, X[f][ite], Y[f][ite])
            quality.append(dict(map="ridge", f_sender=f, f_recv=f, lam_rel=lam, split="heldout", **q))
            if best is None or q["cos"] > best[1]["cos"]:
                best = (lam, q)
        lam_best[f] = best[0]
        maps[("ridge", f)] = ridge_map(X[f], Y[f], best[0])
        Mp = proc_map(X[f][itr], Y[f][itr])
        quality.append(dict(map="proc", f_sender=f, f_recv=f, lam_rel=None, split="heldout", **map_quality(Mp, X[f][ite], Y[f][ite])))
        maps[("proc", f)] = proc_map(X[f], Y[f])
        print(f"[map] f={f:.2f} ridge lam={best[0]:g} cos={best[1]['cos']:.3f} top1={best[1]['top1']:.3f} | "
              f"proc cos={quality[-1]['cos']:.3f} top1={quality[-1]['top1']:.3f}", flush=True)
    # cross-depth grid: is the best match at equal relative depth?
    for fs in F_FULL:
        for fr in F_FULL:
            if fs == fr:
                continue
            M = ridge_map(X[fs][itr], Y[fr][itr], lam_best[fr])
            quality.append(dict(map="ridge", f_sender=fs, f_recv=fr, lam_rel=lam_best[fr], split="heldout",
                                **map_quality(M, X[fs][ite], Y[fr][ite])))
            del M
    # a map fitted on shuffled pairs: the same linear machinery with the correspondence destroyed
    perm = torch.randperm(len(ok_fit), generator=torch.Generator().manual_seed(11), device="cpu").to(dev)
    maps[("shuffled", F_CTRL)] = ridge_map(X[F_CTRL], Y[F_CTRL][perm], lam_best[F_CTRL])
    quality.append(dict(map="shuffled", f_sender=F_CTRL, f_recv=F_CTRL, lam_rel=lam_best[F_CTRL], split="heldout",
                        **map_quality(maps[("shuffled", F_CTRL)], X[F_CTRL][ite], Y[F_CTRL][ite])))
    mean_send = {f: X[f].mean(0) for f in F_FULL}
    del X, Y
    torch.cuda.empty_cache()

    # ---- rank bases and donor pool, on the fitting half
    LRANK = [fR[f] for f in F_RANK]
    D_cross = {f: [] for f in F_RANK}
    D_within = {f: [] for f in F_RANK}
    G_fit = {f: [] for f in F_RANK}
    pool = []
    n_b = 0
    for si in ok_fit:
        if n_b >= a.n_basis:
            break
        s = specs_fit[si]
        rb = receiver_build(R.tok, s, a.max_tokens)
        if rb is None:
            continue
        st, _ = capture(R, [rb["worlds"]["bare"]], [rb["p_patch"]], LR)
        hb = {f: st[fR[f]][0] for f in F_FULL}
        mp = {f: apply_map(maps[("ridge", f)], send[("fit", si)][f].to(dev).unsqueeze(0))[0].cpu() for f in F_FULL}
        gr = margin_grads(R, rb["worlds"]["bare"], rb["p_patch"], LRANK, s["sender"], s["own"])
        for f in F_RANK:
            D_cross[f].append(mp[f] - hb[f])
            D_within[f].append(recv_fit[si][f] - hb[f])
            G_fit[f].append(gr[fR[f]])
        pool.append((s["sender"], s["item_id"], mp))
        n_b += 1
        if n_b % 25 == 0:
            print(f"[progress] bases {n_b}/{a.n_basis} ({(time.time()-t0)/60:.1f} min)", flush=True)
    kmax = max(KS)
    bases = {}
    for f in F_RANK:
        Dc = torch.stack(D_cross[f]).to(dev)
        Dw = torch.stack(D_within[f]).to(dev)
        G = torch.stack(G_fit[f]).to(dev)
        bases[("cross_pca", f)] = orthobasis(Dc, kmax).cpu()
        bases[("within_pca", f)] = orthobasis(Dw, kmax).cpu()
        if f == F_RANK[0]:
            bases[("cross_effect", f)] = effect_basis(Dc, G, kmax).cpu()
        del Dc, Dw, G
    torch.cuda.empty_cache()
    gen = torch.Generator(device="cpu").manual_seed(240091)
    d_recv = pool[0][2][F_CTRL].shape[0]
    rot = torch.linalg.qr(torch.randn(d_recv, d_recv, generator=gen))[0].contiguous()
    print(f"[stage] bases on {n_b} episodes, d_sender={d_send}, d_receiver={d_recv} ({(time.time()-t0)/60:.1f} min)", flush=True)

    # ---- evaluation half
    out, n_ev = [], 0
    for si, s in enumerate(specs_ev):
        if (time.time() - t0) / 60 > a.budget_min:
            print(f"[stage] budget reached after {n_ev} episodes", flush=True)
            break
        if ("eval", si) not in send:
            continue
        rb = receiver_build(R.tok, s, a.max_tokens)
        if rb is None:
            continue
        ids_sr = sender_ids(R.tok, s, a.max_tokens)
        if ids_sr is None:
            continue
        i = s["item_id"]
        meta = dict(item_id=i, round=s["round"], receiver=s["receiver"], sender_agent=s["sender_agent"], own=s["own"],
                    sender=s["sender"], target=s["target"], domain=core[i]["primary_domain"], family=family_of(core[i]),
                    sender_correct=(None if s["target"] is None else s["sender"] == s["target"]))
        names = ["none", "bare", "textj", "text"]
        _, lp0 = capture(R, [rb["worlds"][n] for n in names], [len(rb["worlds"][n]) - 1 for n in names], [])
        for r_, n_ in enumerate(names):
            out.append(dict(meta, arm=f"W_{n_}", layer=None, k=None, basis=None, lp_letters=R.letter_lp(lp0[r_], s["labels"])))
        st, _ = capture(R, [rb["worlds"]["bare"], ids_sr], [rb["p_patch"], len(ids_sr) - 1], LR)
        h_bare = {f: st[fR[f]][0] for f in F_FULL}
        s_own = {f: st[fR[f]][1] for f in F_FULL}  # the receiver's own state on the sender's text: the within-model ceiling
        s_send = {f: send[("eval", si)][f].to(dev).unsqueeze(0) for f in F_FULL}
        mapped = {(kind, f): apply_map(maps[(kind, f)], s_send[f])[0].cpu() for kind, f in maps}
        for f in F_FULL:
            for kind in ("ridge",) + (("proc",) if f in F_PROC else ()):
                p, y = mapped[(kind, f)] - maps[(kind, f)]["my"][0].cpu(), s_own[f] - maps[(kind, f)]["my"][0].cpu()
                out.append(dict(meta, arm=f"map_quality_{kind}", layer=fR[f], depth=f, k=None, basis=None, lp_letters={},
                                cos=float(torch.nn.functional.cosine_similarity(p, y, dim=0)),
                                norm_ratio=float(mapped[(kind, f)].norm() / s_own[f].norm().clamp_min(1e-6))))
        same = [p for p in pool if p[0] == s["sender"] and p[1] != i]
        donor_same = same[h64("same", i, s["round"], s["receiver"]) % len(same)][2] if same else None
        donor_any = pool[h64("any", i, s["round"], s["receiver"]) % len(pool)][2] if pool else None
        rows = []  # (arm, f, k, basis, position, value)
        for f in F_FULL:
            rows.append(("mapped_ridge", f, None, None, rb["p_patch"], mapped[("ridge", f)]))
            rows.append(("within_model", f, None, None, rb["p_patch"], s_own[f]))
            if f in F_PROC:
                rows.append(("mapped_proc", f, None, None, rb["p_patch"], mapped[("proc", f)]))
        for fi, f in enumerate(F_RANK):
            ks = KS if fi == 0 else KS2
            dc, dw = mapped[("ridge", f)] - h_bare[f], s_own[f] - h_bare[f]
            for kk in ks:
                rows.append(("rank_cross", f, kk, "pca", rb["p_patch"], h_bare[f] + project(bases[("cross_pca", f)], kk, dc)))
                rows.append(("rank_within", f, kk, "pca", rb["p_patch"], h_bare[f] + project(bases[("within_pca", f)], kk, dw)))
            if fi == 0:
                for kk in KS:
                    rows.append(("rank_cross", f, kk, "effect", rb["p_patch"],
                                 h_bare[f] + project(bases[("cross_effect", f)], kk, dc)))
        f = F_CTRL
        dc = mapped[("ridge", f)] - h_bare[f]
        raw = s_send[f][0].cpu()
        raw = raw[:d_recv] if d_send >= d_recv else torch.cat([raw, torch.zeros(d_recv - d_send)])
        u = torch.randn(d_recv, generator=gen)
        rows += [("mapped_shuffled_fit", f, None, None, rb["p_patch"], mapped[("shuffled", f)]),
                 ("mapped_mean_sender", f, None, None, rb["p_patch"],
                  apply_map(maps[("ridge", f)], mean_send[f].unsqueeze(0))[0].cpu()),
                 ("raw_unmapped", f, None, None, rb["p_patch"], raw),
                 ("random_vector", f, None, None, rb["p_patch"], h_bare[f] + u / u.norm() * dc.norm()),
                 ("rotated", f, None, None, rb["p_patch"], h_bare[f] + rot @ dc),
                 ("other_position", f, None, None, rb["p_nonslot"], mapped[("ridge", f)])]
        if donor_same is not None:
            rows.append(("mapped_same_letter_other_item", f, None, None, rb["p_patch"], donor_same[f]))
        if donor_any is not None:
            rows.append(("mapped_deranged", f, None, None, rb["p_patch"], donor_any[f]))
        for c0 in range(0, len(rows), CHUNK - 1):
            chunk = rows[c0: c0 + CHUNK - 1]
            batch = [rb["worlds"]["bare"]] * (len(chunk) + 1)
            by_layer = {}
            for ri, (arm, f_, kk, basis, pos, val) in enumerate(chunk, start=1):
                by_layer.setdefault(fR[f_], []).append((ri, torch.tensor([pos], device=dev), val.unsqueeze(0).to(dev)))
            for L, patches in by_layer.items():
                R.add_patch(L, patches)
            lp = R.forward(batch)
            R.clear()
            out.append(dict(meta, arm="W_bare_in_chunk", layer=None, k=None, basis=None, chunk=c0,
                            lp_letters=R.letter_lp(lp[0], s["labels"])))
            for ri, (arm, f_, kk, basis, pos, val) in enumerate(chunk, start=1):
                out.append(dict(meta, arm=arm, layer=fR[f_], depth=f_, k=kk, basis=basis, chunk=c0,
                                lp_letters=R.letter_lp(lp[ri], s["labels"])))
        n_ev += 1
        if n_ev % 10 == 0:
            el = time.time() - t0
            print(f"[progress] eval {n_ev}/{len(specs_ev)} episodes, {el/60:.1f} min elapsed", flush=True)

    od = ACTS / a.receiver / f"lit_latentx_{a.sender}to{a.receiver}_{a.dataset}{a.tag}"
    od.mkdir(parents=True, exist_ok=True)
    if not out:
        raise SystemExit(f"[latent_cross] no rows produced; refusing to write an empty rows.jsonl into {od}. "
                         f"An empty file would satisfy a downstream wait-gate and be read as a finished result.")
    write_jsonl(od / "rows.jsonl", out)
    json.dump(dict(quality=quality, lam_best={str(k): v for k, v in lam_best.items()}), open(od / "map.json", "w"), indent=1)
    json.dump(dict(sender=a.sender, receiver=a.receiver, transcripts=tm, dataset=a.dataset, d_sender=d_send, d_receiver=d_recv,
                   f_full=list(F_FULL), f_proc=list(F_PROC), f_rank=list(F_RANK), f_ctrl=F_CTRL, layers_recv=LR,
                   layers_send=sorted(fS.values()), ks=list(KS), ks2=list(KS2), n_map=len(ok_fit), n_basis=n_b, n_eval=n_ev,
                   prefill=PREFILL), open(od / "meta.json", "w"), indent=1)
    print(f"[done] latent cross {a.sender}->{a.receiver}: {n_ev} episodes, {len(out)} rows in {(time.time()-t0)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
