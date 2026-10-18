"""The latent channel between two debating agents: how many directions carry a peer's influence, and what they carry.

All three debaters are copies of one model, so a peer's internal state can be written directly into the receiver's forward pass
in place of that peer's words. For every receiver/peer pair in which the peer disagrees, we build four worlds of the same round
message (frames conventions, decision read after the prefilled "{final answer: ("):

  W_none   the no-communication message (the receiver sees only its own previous answer)
  W_bare   every peer's response replaced by its bare "{final answer: (X)}" (speaker-free, answer asserted)
  W_textj  the studied peer's response as written, the other peer bare
  W_text   both peers as written

and intervene inside W_bare at the "(" token of the studied peer's bare answer, writing that peer's own decision-state (captured
by teacher-forcing its response up to its "(" token). Endpoint m = lp(peer's answer) - lp(receiver's previous answer); transfer
T = [m(patched) - m(W_bare)] / [m(W_textj) - m(W_bare)], so 1.0 means the state reproduces the whole text channel.

Rank: the state is written only inside a k-dimensional subspace, h <- h + P_k (s - h), with k swept from 1 to 256 and two bases
fitted on a disjoint half of the items: the principal subspace of the state differences, and an effect-aligned subspace (the
top eigenvectors of the symmetrised sum of d(margin)/dh outer state difference, a first-order stand-in for a learned rotation).
k_50 (the rank reaching half the full-state transfer) is the reported quantity.

Controls, all measured on byte-identical text in the same batch shape: unpatched W_bare inside every chunk; a random rotation of
the state difference (same energy, scrambled code); a norm-matched random vector; the same state written at a token in the closing
instruction instead of the answer slot; a state from a different item whose peer gave the same answer letter; a state from a
randomly reassigned episode; random k-dimensional subspaces; the state's own high-variance subspace; and a necessity arm that
removes the subspace component from W_textj.

Influence vs truth: every world also records the margin toward the item's evaluation target, and senders are split into those
whose answer is correct and those whose answer is wrong.

Output: ACTS/<M>/lit_latent_<dataset>/{rows.jsonl, meta.json}
"""
from __future__ import annotations

import argparse
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
# layer positions as a share of depth, so the same design runs on 32-layer (Qwen, Llama) and 42-layer (Gemma) models
F_FULL = (0.19, 0.29, 0.39, 0.52, 0.61, 0.71, 0.81, 0.90)
F_RANK = (0.39, 0.52, 0.71)
F_CTRL = (0.52, 0.71)
F_NEED = 0.52
KS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
KS_CTRL = (4, 16, 64)
CHUNK = 12  # default batch rows per forward; override with --chunk (the fla gated-delta kernel
# allocates B*T*HV*BT per chunked call, so long contexts need a smaller batch or it OOMs on a 46 GB card)


def bare_answer(a):
    return f"{{final answer: ({a})}}"


def tok_at(offsets, char_idx):
    """Index of the token whose span contains char_idx."""
    for i, (s, e) in enumerate(offsets):
        if s <= char_idx < e:
            return i
    return None


def peer_spans(content, rendered):
    """Char span of each peer's rendered response inside a round message built by prompts_lit.debate_prompt."""
    cur, out = len(PEERS_HEAD), []
    for o in rendered:
        piece = PEER_PREFIX + o.strip()
        s = content.index(piece, max(cur - 1, 0))
        out.append(s + len(PEER_PREFIX))
        cur = s + len(piece) + 1
    return out


def encode(tok, msgs, prefill=PREFILL):
    prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    text = prefix + prefill
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return text, enc["input_ids"], enc["offset_mapping"], prefix


def margin(lp_letters, a, b):
    return lp_letters[a] - lp_letters[b] if a in lp_letters and b in lp_letters else float("nan")


def ctx_for(v, recs, i, agent, r):
    """The prompt the agent saw when it produced its round-r response."""
    if r == 0:
        return messages(v, 0, None, None, "debate")
    pc = "r0" if r - 1 == 0 else "debate"
    own = recs.get((i, pc, agent, r - 1, None))
    others = [recs.get((i, pc, h, r - 1, None)) for h in range(N_AGENTS) if h != agent]
    if own is None or any(x is None for x in others):
        return None
    return messages(v, r, own["text"], [x["text"] for x in others], "debate")


def build_episode(tok, v, recs, i, k, g, j, prev_cond):
    """All worlds, positions and the sender context for one (item, round, receiver, peer) episode."""
    prev = {h: recs[(i, prev_cond, h, k - 1, None)] for h in range(N_AGENTS)}
    a = {h: prev[h]["answer"] for h in range(N_AGENTS)}
    t = {h: prev[h]["text"] for h in range(N_AGENTS)}
    others_idx = [h for h in range(N_AGENTS) if h != g]
    jj = others_idx.index(j)
    r_bare = [bare_answer(a[h]) for h in others_idx]
    r_textj = [t[h] if h == j else bare_answer(a[h]) for h in others_idx]
    r_text = [t[h] for h in others_idx]
    worlds = {}
    for name, msgs in (("none", messages(v, k, t[g], None, "reflect")), ("bare", messages(v, k, t[g], r_bare, "debate")),
                       ("textj", messages(v, k, t[g], r_textj, "debate")), ("text", messages(v, k, t[g], r_text, "debate"))):
        worlds[name] = encode(tok, msgs)
    # patch position: the "(" of peer j's bare answer in W_bare
    txt_b, ids_b, off_b, pre_b = worlds["bare"]
    content_b = messages(v, k, t[g], r_bare, "debate")[-1]["content"]
    base_b = pre_b.rindex(content_b)
    starts_b = peer_spans(content_b, r_bare)
    c_paren = base_b + starts_b[jj] + len(BARE_PREFIX) - 1
    p_patch = tok_at(off_b, c_paren)
    c_instr = base_b + content_b.index(INSTR_MARK) + 12
    p_nonslot = tok_at(off_b, c_instr)
    # the same peer's "(" inside its full response, for the necessity arm
    txt_t, ids_t, off_t, pre_t = worlds["textj"]
    content_t = messages(v, k, t[g], r_textj, "debate")[-1]["content"]
    base_t = pre_t.rindex(content_t)
    starts_t = peer_spans(content_t, r_textj)
    sp = answer_span(t[j].strip(), v)
    p_text = tok_at(off_t, base_t + starts_t[jj] + sp[1] - 1) if sp else None
    # sender context: its own prompt plus its response truncated just before the answer letter
    msgs_s = ctx_for(v, recs, i, j, k - 1)
    if msgs_s is None or sp is None or t[j].strip()[sp[1] - 1] != "(":
        return None
    pre_s = tok.apply_chat_template(msgs_s, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids_s = tok(pre_s + t[j].strip()[: sp[1]], add_special_tokens=False)["input_ids"]
    if p_patch is None or p_nonslot is None or p_text is None:
        return None
    return dict(worlds={n: w[1] for n, w in worlds.items()}, p_patch=p_patch, p_nonslot=p_nonslot, p_text=p_text,
                ids_sender=ids_s, own=a[g], sender=a[j], labels=v["labels"], target=v["target"])


def capture(R, ids_list, positions, layers):
    """One forward; returns {layer: [state per row]} (float32 cpu) and the letter log-probs."""
    store = {}
    for L in layers:
        R.add_capture(L, positions, store)
    lp = R.forward(ids_list)
    R.clear()
    return {L: [x for x in store[L]] for L in layers}, lp


def _ckpt_safe(R):
    """True when switching the model to train mode cannot change the forward numerically (no dropout anywhere)."""
    cfg = R.model.config
    cfg = getattr(cfg, "text_config", cfg)
    bad = {k: v for k, v in vars(cfg).items() if "drop" in k.lower() and isinstance(v, (int, float)) and v}
    if bad:
        print(f"[latent] not enabling train-mode checkpointing: non-zero dropout {bad}", flush=True)
        return False
    return True


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
    # The backward needs gradients only at `layers`, but a plain forward stores activations for all 32 blocks, and the
    # gated-delta kernel's saved intermediates make that ~25 GB at a few thousand tokens -- enough to OOM a 46 GB card
    # on top of the weights. Checkpointing recomputes them during backward instead; the gradients are unchanged.
    ckpt = False
    if _ckpt_safe(R):
        try:
            R.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            # transformers gates checkpointing on `self.training` (modeling_layers.py: "if self.gradient_checkpointing
            # and self.training"), so in eval mode enabling it is a silent no-op. Switching to train mode for the
            # backward is what actually engages it; _ckpt_safe has verified the model has no dropout, and RMSNorm has
            # no train/eval behaviour, so the numbers are unchanged.
            R.model.train()
            ckpt = True
        except Exception as e:
            print(f"[latent] gradient checkpointing unavailable ({e}); the backward may OOM", flush=True)
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
        if ckpt:
            try:
                R.model.gradient_checkpointing_disable()
            finally:
                R.model.eval()
    g = {L: leaves[L].grad[0, pos].detach().float().cpu().clone() for L in layers}
    leaves.clear()
    del out, inp
    torch.cuda.empty_cache()
    return g


def orthobasis(M, k):
    """Top-k orthonormal rows spanning the principal subspace of the rows of M."""
    U, S, V = torch.linalg.svd(M.to(torch.float32), full_matrices=False)
    return V[:k].contiguous()


def effect_basis(D, G, k):
    """Top-k eigenvectors of the symmetrised sum_i (delta_i g_i^T + g_i delta_i^T)/2: the subspace that maximises the
    first-order predicted change of the margin, a closed-form stand-in for a learned rotation."""
    S = (D.T @ G + G.T @ D) / (2 * len(D))
    w, Vec = torch.linalg.eigh(S.to(torch.float32))
    order = torch.argsort(w.abs(), descending=True)[:k]
    return Vec[:, order].T.contiguous()


def project(basis, k, delta):
    B = basis[:k]
    return B.T @ (B @ delta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="lit1200")
    ap.add_argument("--transcripts", default=None)
    ap.add_argument("--n_fit", type=int, default=300)
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--max_tokens", type=int, default=4000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--chunk", type=int, default=CHUNK, help="rows per forward batch; lower it if the kernel OOMs")
    a = ap.parse_args()
    R = Receiver(a.model)
    d_model = R.model.config.get_text_config().hidden_size
    at = lambda f: int(round(f * (R.n_layers - 1)))
    LAYERS_FULL = tuple(sorted({at(f) for f in F_FULL}))
    LAYERS_RANK = tuple(sorted({at(f) for f in F_RANK}))
    LAYERS_CTRL = tuple(sorted({at(f) for f in F_CTRL}))
    L_NEED = at(F_NEED)
    print(f"[stage] layers: full {LAYERS_FULL}, rank {LAYERS_RANK}, controls {LAYERS_CTRL}, necessity {L_NEED}", flush=True)
    tm = a.transcripts or a.model
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
                    if j == g or ans[j] is None or ans[j] == ans[g]:
                        continue
                    eps.append((i, k, g, j, pc))
    rng_split = lambda i: h64("latentsplit", i) % 2
    eps.sort(key=lambda e: h64("latentep", e[0], e[1], e[2], e[3]))
    fit = [e for e in eps if rng_split(e[0]) == 0][: a.n_fit]
    ev = [e for e in eps if rng_split(e[0]) == 1][: a.n_eval]
    print(f"[stage] {len(eps)} disagreement episodes; fitting on {len(fit)}, evaluating on {len(ev)}", flush=True)
    gen = torch.Generator(device="cpu").manual_seed(240091)
    out, t0 = [], time.time()

    def prep(e):
        i, k, g, j, pc = e
        b = build_episode(R.tok, views[i], recs, i, k, g, j, pc)
        if b is None or max(len(x) for x in b["worlds"].values()) > a.max_tokens or len(b["ids_sender"]) > a.max_tokens:
            return None
        return b

    # ---- pass 1: fitting half -> state differences and margin gradients
    D_fit = {L: [] for L in LAYERS_RANK}
    G_fit = {L: [] for L in LAYERS_RANK}
    H_fit = {L: [] for L in LAYERS_RANK}
    pool = []  # (sender answer letter, {layer: state}) donors for the sham arms
    n_fit_used = 0
    for e in fit:
        b = prep(e)
        if b is None:
            continue
        st_b, _ = capture(R, [b["worlds"]["bare"]], [b["p_patch"]], LAYERS_FULL)
        st_s, _ = capture(R, [b["ids_sender"]], [len(b["ids_sender"]) - 1], LAYERS_FULL)
        gr = margin_grads(R, b["worlds"]["bare"], b["p_patch"], LAYERS_RANK, b["sender"], b["own"])
        for L in LAYERS_RANK:
            D_fit[L].append(st_s[L][0] - st_b[L][0])
            G_fit[L].append(gr[L])
            H_fit[L].append(st_b[L][0])
        pool.append((b["sender"], {L: st_s[L][0] for L in LAYERS_FULL}))
        n_fit_used += 1
        if n_fit_used % 50 == 0:
            print(f"[progress] fit {n_fit_used}/{len(fit)} ({(time.time()-t0)/60:.1f} min)", flush=True)
    kmax = max(KS)
    bases = {}
    for L in LAYERS_RANK:
        D = torch.stack(D_fit[L]).to(R.device)
        G = torch.stack(G_fit[L]).to(R.device)
        H = torch.stack(H_fit[L]).to(R.device)
        bases[("pca", L)] = orthobasis(D, kmax).cpu()
        bases[("effect", L)] = effect_basis(D, G, kmax).cpu()
        bases[("statepc", L)] = orthobasis(H - H.mean(0, keepdim=True), kmax).cpu()
        del D, G, H
    rot = {}
    for L in LAYERS_CTRL:
        Q, _ = torch.linalg.qr(torch.randn(d_model, d_model, generator=gen))
        rot[L] = Q.contiguous()
    haar = {k: torch.linalg.qr(torch.randn(d_model, k, generator=gen))[0].T.contiguous() for k in KS_CTRL}
    print(f"[stage] bases fitted on {n_fit_used} episodes in {(time.time()-t0)/60:.1f} min", flush=True)

    # ---- pass 2: evaluation half
    n_ev = 0
    for e in ev:
        i, k, g, j, pc = e
        b = prep(e)
        if b is None:
            continue
        v = views[i]
        meta = dict(item_id=i, round=k, receiver=g, sender_agent=j, own=b["own"], sender=b["sender"], target=v["target"],
                    domain=core[i]["primary_domain"], family=family_of(core[i]),
                    sender_correct=(None if v["target"] is None else b["sender"] == v["target"]))
        names = ["none", "bare", "textj", "text"]
        st_b, lp0 = capture(R, [b["worlds"][n] for n in names], [len(b["worlds"][n]) - 1 for n in names], LAYERS_FULL)
        for r_, n_ in enumerate(names):
            ll = R.letter_lp(lp0[r_], b["labels"])
            out.append(dict(meta, arm=f"W_{n_}", layer=None, k=None, basis=None, lp_letters=ll))
        st_bp, _ = capture(R, [b["worlds"]["bare"]], [b["p_patch"]], LAYERS_FULL)
        st_s, _ = capture(R, [b["ids_sender"]], [len(b["ids_sender"]) - 1], LAYERS_FULL)
        h_bare = {L: st_bp[L][0] for L in LAYERS_FULL}
        s_st = {L: st_s[L][0] for L in LAYERS_FULL}
        same = [p for p in pool if p[0] == b["sender"]]
        donor_same = same[h64("same", i, k, g, j) % len(same)][1] if same else None
        donor_any = pool[h64("any", i, k, g, j) % len(pool)][1] if pool else None
        rows = []  # (arm, layer, k, basis, position, value)
        for L in LAYERS_FULL:
            rows.append(("full_state", L, None, None, b["p_patch"], s_st[L]))
        for L in LAYERS_RANK:
            delta = s_st[L] - h_bare[L]
            for basis in ("pca", "effect"):
                B = bases[(basis, L)]
                for kk in KS:
                    rows.append(("rank", L, kk, basis, b["p_patch"], h_bare[L] + project(B, kk, delta)))
        for L in LAYERS_CTRL:
            delta = s_st[L] - h_bare[L]
            rows.append(("rotated", L, None, None, b["p_patch"], h_bare[L] + rot[L] @ delta))
            u = torch.randn(d_model, generator=gen)
            rows.append(("random_vector", L, None, None, b["p_patch"], h_bare[L] + u / u.norm() * delta.norm()))
            rows.append(("other_position", L, None, None, b["p_nonslot"], s_st[L]))
            if donor_same is not None:
                rows.append(("same_letter_other_item", L, None, None, b["p_patch"], donor_same[L]))
            if donor_any is not None:
                rows.append(("deranged_sender", L, None, None, b["p_patch"], donor_any[L]))
            for kk in KS_CTRL:
                rows.append(("rank_random_subspace", L, kk, "haar", b["p_patch"], h_bare[L] + project(haar[kk], kk, delta)))
        for kk in KS_CTRL:
            L = L_NEED
            delta = s_st[L] - h_bare[L]
            rows.append(("rank_state_variance", L, kk, "statepc", b["p_patch"], h_bare[L] + project(bases[("statepc", L)], kk, delta)))
        for c0 in range(0, len(rows), a.chunk - 1):
            chunk = rows[c0: c0 + a.chunk - 1]
            batch = [b["worlds"]["bare"]] * (len(chunk) + 1)
            by_layer = {}
            for ri, (arm, L, kk, basis, pos, val) in enumerate(chunk, start=1):
                by_layer.setdefault(L, []).append((ri, torch.tensor([pos], device=R.device), val.unsqueeze(0).to(R.device)))
            for L, patches in by_layer.items():
                R.add_patch(L, patches)
            lp = R.forward(batch)
            R.clear()
            out.append(dict(meta, arm="W_bare_in_chunk", layer=None, k=None, basis=None, chunk=c0,
                            lp_letters=R.letter_lp(lp[0], b["labels"])))
            for ri, (arm, L, kk, basis, pos, val) in enumerate(chunk, start=1):
                out.append(dict(meta, arm=arm, layer=L, k=kk, basis=basis, chunk=c0, lp_letters=R.letter_lp(lp[ri], b["labels"])))
        # necessity: remove the subspace component from the written message
        st_t, _ = capture(R, [b["worlds"]["textj"]], [b["p_text"]], (L_NEED,))
        h_t = st_t[L_NEED][0]
        diff = h_t - h_bare[L_NEED]
        nrows = [("ablate_full", None), ("ablate_rank", 16), ("ablate_rank", 64)]
        batch = [b["worlds"]["textj"]] * (len(nrows) + 1)
        patches = []
        for ri, (arm, kk) in enumerate(nrows, start=1):
            val = h_bare[L_NEED] if kk is None else h_t - project(bases[("pca", L_NEED)], kk, diff)
            patches.append((ri, torch.tensor([b["p_text"]], device=R.device), val.unsqueeze(0).to(R.device)))
        R.add_patch(L_NEED, patches)
        lp = R.forward(batch)
        R.clear()
        out.append(dict(meta, arm="W_textj_in_chunk", layer=None, k=None, basis=None, lp_letters=R.letter_lp(lp[0], b["labels"])))
        for ri, (arm, kk) in enumerate(nrows, start=1):
            out.append(dict(meta, arm=arm, layer=L_NEED, k=kk, basis="pca", lp_letters=R.letter_lp(lp[ri], b["labels"])))
        n_ev += 1
        if n_ev % 20 == 0:
            el = time.time() - t0
            print(f"[progress] eval {n_ev}/{len(ev)} episodes, {el/60:.1f} min elapsed, eta {el/n_ev*(len(ev)-n_ev)/60:.1f} min", flush=True)
    od = ACTS / a.model / f"lit_latent_{a.dataset}{a.tag}"
    od.mkdir(parents=True, exist_ok=True)
    write_jsonl(od / "rows.jsonl", out)
    json.dump(dict(model=a.model, transcripts=tm, dataset=a.dataset, layers_full=list(LAYERS_FULL), layers_rank=list(LAYERS_RANK),
                   layers_ctrl=list(LAYERS_CTRL), ks=list(KS), ks_ctrl=list(KS_CTRL), n_fit=n_fit_used, n_eval=n_ev,
                   layer_necessity=L_NEED, d_model=d_model, prefill=PREFILL), open(od / "meta.json", "w"), indent=1)
    print(f"[done] lit latent {a.model}: {n_ev} evaluation episodes, {len(out)} rows in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
