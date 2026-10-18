"""Where does a peer's influence reach the decision, and which architecture carries it?

Two arms, both run per model.

--mode attn  Direct-edge attention knockout. In a first debate round with a disagreeing peer, the attention weights from the
             decision token (the "(" of the prefilled "{final answer: (") to the key positions of the disagreeing peer's message
             are set to zero, at one attention layer, cumulatively up to a layer, or at every attention layer. The three models
             differ in how many layers can carry that edge at all: Llama-3.1-8B has 32 full-attention layers, Qwen3.5-9B has 8
             (the other 24 are Gated-DeltaNet linear-attention blocks that no attention mask can touch), Gemma-4-E4B has 7 plus
             35 sliding-window layers whose 512-token window usually does not reach back to the peer's message.
             Controls in the same batch:
               own_*        the same number of key positions taken from the end of the agent's own previous response
               qctrl_all    the peer's keys are blocked for the prefill tokens before the decision token, not for the decision
                            token itself (does blocking peer keys at all hurt, wherever the query sits?)
               peer_allq_*  the peer's keys are blocked for every query position after the peer's message: the strongest thing
                            attention masking can do, i.e. attention cannot read the peer any more
               filler       the peer's text replaced by neutral filler of matched token length (a text-level deletion)
               nopeer       the no-communication message (our usual no-influence endpoint)
               none         no mask edits, checked against an unmasked forward of the same rows (numerics check)

--mode span  Interchange at a text span rather than at the decision token, on the source-attribution worlds of cpt_lit: the
             residual states over the 20-token source descriptor of a donor world (strong track record) are written into a
             recipient world (poor track record) at one layer. On Gemma-4 the same span is also patched through the per-layer
             embedding channel, which re-injects token identity at every layer and which a residual-stream patch leaves
             untouched: resid@L, ple>=L and resid@L+ple>L. Recovery = (patched - recipient) / (donor - recipient).

Outputs ACTS/<M>/routing_attn_<dataset>/rows.jsonl and ACTS/<M>/routing_span_<split>/rows.jsonl.
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
from cpt import Receiver, load_stimuli
from cpt_lit import DESC, build, items_for, layer_grid
from prompts_lit import N_AGENTS, debate_prompt, messages

RAW_LIT = ROOT / "outputs/raw_lit"
PREFILL = "{final answer: ("
FILLER_UNIT = "Let me work through this carefully. "
HEAD = "These are the recent opinions from other agents: "
OWN_LEAD = " This was your most recent opinion: "


# ---------------------------------------------------------------------------
# spans
# ---------------------------------------------------------------------------
def content_spans(own, others):
    """Char spans of each peer's response and of the agent's own previous response inside debate_prompt()."""
    parts = [f"One of the agents' response: {o.strip()}" for o in others]
    peer, off = [], len(HEAD)
    for p, o in zip(parts, others):
        s = off + len(p) - len(o.strip())
        peer.append((s, s + len(o.strip())))
        off += len(p) + 1
    own_s = len(HEAD) + len(" ".join(parts)) + len(OWN_LEAD)
    return peer, (own_s, own_s + len(own.strip()))


def tokspan(offs, a, b):
    idx = [i for i, (s, e) in enumerate(offs) if e > a and s < b]
    return (idx[0], idx[-1] + 1) if idx else None


# ---------------------------------------------------------------------------
# attention-mask router
# ---------------------------------------------------------------------------
class Router:
    """Per-layer edits to the additive attention mask of a batch of equal-length rows.

    One working mask per attention type is built once per forward; a pre-hook on each attention module hands it over after
    applying that layer's edits and a forward hook restores the edited slices, so the tensor is reused across layers.
    """

    def __init__(self, R):
        self.R = R
        tc = R.model.config.get_text_config()
        lt = getattr(tc, "layer_types", None)
        self.window = getattr(tc, "sliding_window", None)
        self.info = []
        for i, l in enumerate(R.layers):
            sa = getattr(l, "self_attn", None)
            if sa is None:  # Qwen3.5 linear-attention blocks have .linear_attn and take no attention mask
                continue
            at = getattr(l, "attention_type", None) or (lt[i] if lt else "full_attention")
            self.info.append((i, sa, str(at)))
        self.attn_layers = [i for i, _, _ in self.info]
        self.types = {i: at for i, _, at in self.info}
        self.base, self.work, self.edits, self.applied, self.hooks = {}, {}, {}, [], []

    def _mask(self, T, at, dtype, device):
        ar = torch.arange(T, device=device)
        d = ar[:, None] - ar[None, :]
        ok = d >= 0
        if at.startswith("sliding") and self.window:
            ok &= d < self.window
        M = torch.zeros((T, T), dtype=dtype, device=device)
        M.masked_fill_(~ok, torch.finfo(dtype).min)
        return M

    def prepare(self, B, T, edits, dtype=torch.bfloat16, device="cuda"):
        """edits: {layer_idx: [(row, q_index_tensor, k0, k1), ...]}. Every attention layer is routed through the built mask,
        edited or not, so that all rows of a batch go down the same kernel path and differ only by the edits."""
        self.edits = {l: edits.get(l, []) for l in self.attn_layers}
        self.base, self.work = {}, {}
        for at in sorted(set(self.types.values())):
            m = self._mask(T, at, dtype, device)
            self.base[at] = m
            self.work[at] = m[None, None].expand(B, 1, T, T).contiguous()
        self.clear()
        neg = {at: torch.finfo(dtype).min for at in self.base}
        for li, sa, at in self.info:
            def pre(mod, args, kwargs, li=li, at=at):
                W = self.work[at]
                for r, q, k0, k1 in self.edits[li]:
                    W[r, 0, q, k0:k1] = neg[at]
                kwargs = dict(kwargs)
                kwargs["attention_mask"] = W
                return args, kwargs

            def post(mod, inp, out, li=li, at=at):
                W, Bs = self.work[at], self.base[at]
                for r, q, k0, k1 in self.edits[li]:
                    W[r, 0, q, k0:k1] = Bs[q, k0:k1]
                return out

            self.hooks.append(sa.register_forward_pre_hook(pre, with_kwargs=True))
            self.hooks.append(sa.register_forward_hook(post))

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ---------------------------------------------------------------------------
# mode attn
# ---------------------------------------------------------------------------
def load_raw(tm, dataset):
    d = RAW_LIT / tm / dataset
    recs = {}
    for f in d.glob("*.jsonl"):
        for x in read_jsonl(f):
            recs[(x["item_id"], x["cond"], x["agent"], x["round"], x.get("query"))] = x
    return recs


def episodes(a, recs, views, tok):
    """First-round debate episodes in which one or more peers disagree and all disagreeing peers name the same option."""
    out = []
    for i, v in views.items():
        prev = [recs.get((i, "r0", g, 0, None)) for g in range(N_AGENTS)]
        if any(p is None or p["answer"] is None for p in prev):
            continue
        texts, ans = [p["text"] for p in prev], [p["answer"] for p in prev]
        for g in range(N_AGENTS):
            oi = [h for h in range(N_AGENTS) if h != g]
            dis = [h for h in oi if ans[h] != ans[g]]
            if not dis or len({ans[h] for h in dis}) != 1:
                continue
            out.append(dict(item_id=i, agent=g, own=ans[g], adv=ans[dis[0]], n_dis=len(dis),
                            _others=[texts[h] for h in oi], _own_text=texts[g],
                            _dis_local=[oi.index(h) for h in dis]))
    out.sort(key=lambda e: h64("routing", e["item_id"], e["agent"]))
    return out[: a.n] if a.n else out


def render(tok, v, own_text, others, cond="debate"):
    msgs = messages(v, 1, own_text, others, cond)
    prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(prefix + PREFILL, add_special_tokens=False, return_offsets_mapping=True)
    return prefix, enc["input_ids"], enc["offset_mapping"]


def retrieval_check(R, rt, dt):
    """Positive control for the mask itself: an answer that lives only inside the blocked span must stop being readable."""
    pre = "The password for the vault is banana."
    txt = pre + " The weather today is mild and the meeting starts at four. Question: what is the password for the vault? Answer: the password is"
    enc = R.tok(txt, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    sp = tokspan(offs, pre.index("banana"), pre.index("banana") + len("banana"))
    T = len(ids)
    tid = R.tok(" banana", add_special_tokens=False)["input_ids"][0]
    q = torch.arange(sp[1], T, device=R.device)
    rt.prepare(2, T, {l: [(1, q, sp[0], sp[1])] for l in rt.attn_layers}, dtype=dt, device=R.device)
    lp = R.forward([ids] * 2)
    rt.clear()
    return dict(readable=float(lp[0, tid]), blocked=float(lp[1, tid]))


def cmd_attn(a):
    dt = torch.float32 if a.device == "cpu" else torch.bfloat16
    R = Receiver(a.model, dtype=dt, device=a.device)
    rt = Router(R)
    print(f"[stage] {a.model}: {len(rt.attn_layers)}/{R.n_layers} layers take an attention mask "
          f"({sorted(set(rt.types.values()))}, window={rt.window}); ple={R.has_ple}", flush=True)
    grid = [rt.attn_layers[j] for j in sorted({int(round(f)) for f in np.linspace(0, len(rt.attn_layers) - 1, min(12, len(rt.attn_layers)))})]
    tok = R.tok
    tm = a.transcripts or a.model
    recs = load_raw(tm, a.dataset)
    core = {r["id"]: r for r in load_core()}
    views = {i: item_view(core[i]) for i in sorted({k[0] for k in recs})}
    views = {i: v for i, v in views.items() if v["kind"] != "numeric"}
    eps = episodes(a, recs, views, tok)
    rc = retrieval_check(R, rt, dt)
    print(f"[stage] {len(eps)} episodes with a disagreeing peer; grid {grid}; mask positive control: "
          f"log p(answer inside the span) {rc['readable']:.2f} -> {rc['blocked']:.2f} when the span is blocked", flush=True)
    nfill = len(tok(FILLER_UNIT, add_special_tokens=False)["input_ids"])
    out = ACTS / a.model / f"routing_attn_{a.dataset}"
    out.mkdir(parents=True, exist_ok=True)
    rows_out, t0, skipped = [], time.time(), 0
    for ei, e in enumerate(eps):
        v = views[e["item_id"]]
        prefix, ids, offs = render(tok, v, e["_own_text"], e["_others"])
        if len(ids) > a.maxlen:
            skipped += 1
            continue
        content = debate_prompt(v, e["_own_text"], e["_others"])
        c0 = prefix.rfind(content)
        if c0 < 0:
            skipped += 1
            continue
        pspan_c, ospan_c = content_spans(e["_own_text"], e["_others"])
        peer_k = [tokspan(offs, c0 + s, c0 + t) for j, (s, t) in enumerate(pspan_c) if j in e["_dis_local"]]
        own_k = tokspan(offs, c0 + ospan_c[0], c0 + ospan_c[1])
        if any(x is None for x in peer_k) or own_k is None:
            skipped += 1
            continue
        T = len(ids)
        dec = T - 1
        n_prefill = len(tok(prefix + PREFILL, add_special_tokens=False)["input_ids"]) - len(tok(prefix, add_special_tokens=False)["input_ids"])
        n_peer = sum(b - s for s, b in peer_k)
        own_use = [(max(own_k[0], own_k[1] - n_peer), own_k[1])]
        peer_end = max(b for _, b in peer_k)
        dev = R.device
        q_dec = torch.tensor([dec], device=dev)
        q_pre = torch.arange(max(T - n_prefill, 0), dec, device=dev)
        q_all = torch.arange(peer_end, T, device=dev)

        def ed(layers, qsel, spans):
            return {l: [(None, qsel, s, b) for s, b in spans] for l in layers}

        arms = [("none", {})]
        for l in grid:
            arms.append((f"peer_L{l}", ed([l], q_dec, peer_k)))
        for l in grid:
            arms.append((f"peer_cum{l}", ed([x for x in rt.attn_layers if x <= l], q_dec, peer_k)))
        arms.append(("peer_all", ed(rt.attn_layers, q_dec, peer_k)))
        for l in grid:
            arms.append((f"own_L{l}", ed([l], q_dec, own_use)))
        arms.append(("own_all", ed(rt.attn_layers, q_dec, own_use)))
        arms.append(("qctrl_all", ed(rt.attn_layers, q_pre, peer_k)))
        arms.append(("peer_allq_all", ed(rt.attn_layers, q_all, peer_k)))

        key = dict(item_id=e["item_id"], agent=e["agent"], own=e["own"], adv=e["adv"], n_dis=e["n_dis"],
                   domain=v.get("domain") or family_of(core[e["item_id"]]), family=family_of(core[e["item_id"]]),
                   n_tok=T, n_peer_tok=n_peer, n_own_tok=own_use[0][1] - own_use[0][0], peer_end=peer_end,
                   dec_minus_peer_end=dec - peer_end, dec_minus_own_end=dec - own_k[1],
                   peer_in_window=(None if not rt.window else float(sum(max(0, min(b, T) - max(s, dec - rt.window + 1))
                                                                        for s, b in peer_k) / max(n_peer, 1))))
        labels = v["labels"]
        for c0_ in range(0, len(arms), a.batch):
            chunk = arms[c0_: c0_ + a.batch]
            edits = {}
            for r, (_, e_) in enumerate(chunk):
                for l, lst in e_.items():
                    edits.setdefault(l, []).extend([(r, q, s, b) for (_, q, s, b) in lst])
            rt.prepare(len(chunk), T, edits, dtype=dt, device=dev)
            lp = R.forward([ids] * len(chunk))
            rt.clear()
            for r, (name, _) in enumerate(chunk):
                rows_out.append(dict(key, arm=name, lp=R.letter_lp(lp[r], labels)))

        # text-level endpoints (different token lengths, no mask edits)
        fill = [FILLER_UNIT * max(1, round(len(tok(o, add_special_tokens=False)["input_ids"]) / nfill)) for o in e["_others"]]
        _, ids_f, _ = render(tok, v, e["_own_text"], fill)
        _, ids_n, _ = render(tok, v, e["_own_text"], None, cond="reflect")
        lp = R.forward([ids, ids_f[: a.maxlen], ids_n[: a.maxlen]])
        for r, name in enumerate(["none_ref", "filler", "nopeer"]):
            rows_out.append(dict(key, arm=name, lp=R.letter_lp(lp[r], labels)))
        done = ei + 1
        if done % 25 == 0:
            el = time.time() - t0
            print(f"[progress] {done}/{len(eps)} episodes, eta {el/done*(len(eps)-done)/60:.1f} min", flush=True)
        if a.max_min and (time.time() - t0) / 60 > a.max_min:
            print(f"[stop] budget of {a.max_min} min reached after {done} episodes", flush=True)
            break
    write_jsonl(out / "rows.jsonl", rows_out)
    json.dump(dict(model=a.model, sha=R.sha, dataset=a.dataset, transcripts=tm, attn_layers=rt.attn_layers, n_layers=R.n_layers,
                   grid=grid, layer_types=rt.types, window=rt.window, skipped=skipped, prefill=PREFILL,
                   n_episodes=done - skipped, retrieval_check=rc),
              open(out / "meta.json", "w"), indent=1)
    print(f"[done] routing attn {a.model}: {done-skipped} episodes ({skipped} skipped) in {(time.time()-t0)/60:.1f} min", flush=True)


# ---------------------------------------------------------------------------
# mode span: residual vs per-layer-embedding interchange over a text span
# ---------------------------------------------------------------------------
def cmd_span(a):
    R = Receiver(a.model, dtype=torch.float32 if a.device == "cpu" else torch.bfloat16, device=a.device)
    grid = layer_grid(R.n_layers)
    stim = load_stimuli()
    items = [x for x in items_for(a.split, None, 0, seed_key="routing") if x[0]["id"] in stim][: a.n]
    print(f"[stage] span interchange on {len(items)} items; ple={R.has_ple}; grid {grid}", flush=True)
    out = ACTS / a.model / f"routing_span_{a.split}"
    out.mkdir(parents=True, exist_ok=True)
    rows_out, t0 = [], time.time()
    for ii, (r, cv, s) in enumerate(items):
        for adv in (cv["o1"], cv["o2"]):
            prior = cv["o2"] if adv == cv["o1"] else cv["o1"]
            base = dict(adv=adv, qual="strong", prior=prior)
            dids, dpos, labels = build(R.tok, r, cv, s, dict(base, source="ord_good"))
            rids, rpos, _ = build(R.tok, r, cv, s, dict(base, source="ord_bad"))
            if len(dids) != len(rids) or dpos["src_span"] != rpos["src_span"]:
                continue
            sp = dpos["src_span"]
            arms = [("donor", None, None), ("recipient", None, None)]
            for L in grid:
                arms.append((f"resid_L{L}", L, None))
                if R.has_ple:
                    arms.append((f"ple_ge{L}", None, L))
                    arms.append((f"resid_L{L}_ple_gt{L}", L, L + 1))
            ids = [dids] + [rids] * (len(arms) - 1)
            pos = torch.arange(sp[0], sp[1], device=R.device)
            for li in range(R.n_layers):
                rp = [i for i, (_, rl, _) in enumerate(arms) if rl == li]
                pl = [i for i, (_, _, pg) in enumerate(arms) if pg is not None and pg <= li]
                if rp:
                    def hook(mod, inp, outp, rp=rp):
                        h = outp[0] if isinstance(outp, tuple) else outp
                        for i in rp:
                            h[i, pos] = h[0, pos]
                        return outp
                    R.hooks.append(R.layers[li].register_forward_hook(hook))
                if pl:
                    def pre(mod, args, pl=pl):
                        ple = args[1]
                        for i in pl:
                            ple[i, pos] = ple[0, pos]
                        return None
                    R.hooks.append(R.layers[li].register_forward_pre_hook(pre))
            lp = R.forward(ids)
            R.clear()
            for i, (name, rl, pg) in enumerate(arms):
                ll = R.letter_lp(lp[i], labels)
                rows_out.append(dict(item_id=r["id"], adv=adv, prior=prior, arm=name, resid_layer=rl, ple_from=pg,
                                     src_len=sp[1] - sp[0], m_adv=ll[adv] - ll[prior], domain=cv["domain"], family=cv["family"]))
        if (ii + 1) % 25 == 0:
            el = time.time() - t0
            print(f"[progress] {ii+1}/{len(items)} items, eta {el/(ii+1)*(len(items)-ii-1)/60:.1f} min", flush=True)
        if a.max_min and (time.time() - t0) / 60 > a.max_min:
            print(f"[stop] budget of {a.max_min} min reached after {ii+1} items", flush=True)
            break
    write_jsonl(out / "rows.jsonl", rows_out)
    json.dump(dict(model=a.model, sha=R.sha, grid=grid, has_ple=R.has_ple, n_items=len(items), descriptors=DESC),
              open(out / "meta.json", "w"), indent=1)
    print(f"[done] routing span {a.model}: {len(items)} items in {(time.time()-t0)/60:.1f} min", flush=True)


# ---------------------------------------------------------------------------
def cmd_selftest(a):
    """The mask the router builds must reproduce an unmasked forward exactly, and a blocked span must change the read-out."""
    R = Receiver(a.model, dtype=torch.float32 if a.device == "cpu" else torch.bfloat16, device=a.device)
    rt = Router(R)
    print(f"[selftest] {a.model}: attention layers {rt.attn_layers} of {R.n_layers}; types {sorted(set(rt.types.values()))}; "
          f"window {rt.window}", flush=True)
    ids = R.tok("The capital of France is Paris. The capital of Japan is Tokyo. Question: what is the capital of Japan? Answer:",
                add_special_tokens=False)["input_ids"]
    B, T = 3, len(ids)
    dt = torch.float32 if a.device == "cpu" else torch.bfloat16
    # 1. the mask the router builds must be the mask the model itself builds (checked against the eager implementation)
    R.model.set_attn_implementation("eager")
    got = {}

    def mk(i):
        def f(m, ar, kw):
            got[i] = kw["attention_mask"].detach().clone()
        return f
    hs = [l.self_attn.register_forward_pre_hook(mk(i), with_kwargs=True) for i, l in enumerate(R.layers) if hasattr(l, "self_attn")]
    R.forward([ids] * B)
    for h in hs:
        h.remove()
    R.model.set_attn_implementation("sdpa")
    ok = all(torch.equal(m[0, 0], rt._mask(T, rt.types[i], m.dtype, m.device)) for i, m in got.items())
    print(f"[selftest] built mask == model's own mask at all {len(got)} attention layers: {'PASS' if ok else 'FAIL'}", flush=True)
    # 2. routing the mask explicitly changes the attention kernel; every row of a batch must take the same path
    lp0 = R.forward([ids] * B)
    rt.prepare(B, T, {}, dtype=dt, device=R.device)
    lp1 = R.forward([ids] * B)
    rt.clear()
    top = lp0[0].topk(20).indices
    print(f"[selftest] kernel offset of the routed path vs the default one (top-20 tokens): "
          f"{(lp0[0, top] - lp1[0, top]).abs().max().item():.3e} — baselines are taken inside the routed batch", flush=True)
    q = torch.tensor([T - 1], device=R.device)
    ed = {l: [(1, q, 0, T - 3)] for l in rt.attn_layers}
    rt.prepare(B, T, ed, dtype=dt, device=R.device)
    lp2 = R.forward([ids] * B)
    rt.clear()
    print(f"[selftest] in-batch unedited rows vs no edits: {(lp1[0]-lp2[0]).abs().max().item():.3e} / "
          f"{(lp1[2]-lp2[2]).abs().max().item():.3e}; edited row moved {(lp1[1]-lp2[1]).abs().max().item():.3f}", flush=True)
    # 3. a span blocked for every downstream query must become unreadable: the answer lives only inside it
    pre = "The password for the vault is banana."
    txt = pre + " The weather today is mild and the meeting starts at four. Question: what is the password for the vault? Answer: the password is"
    enc = R.tok(txt, add_special_tokens=False, return_offsets_mapping=True)
    ids2, offs = enc["input_ids"], enc["offset_mapping"]
    sp = tokspan(offs, pre.index("banana"), pre.index("banana") + len("banana"))
    T2 = len(ids2)
    tid = R.tok(" banana", add_special_tokens=False)["input_ids"][0]
    qa = torch.arange(sp[1], T2, device=R.device)
    rt.prepare(2, T2, {l: [(1, qa, sp[0], sp[1])] for l in rt.attn_layers}, dtype=dt, device=R.device)
    lp3 = R.forward([ids2] * 2)
    rt.clear()
    print(f"[selftest] log p(' banana') with the span readable {lp3[0, tid]:.2f} -> blocked for all later queries "
          f"{lp3[1, tid]:.2f}  ({'PASS' if lp3[1, tid] < lp3[0, tid] - 2 else 'CHECK'})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["attn", "span", "selftest"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="lit1200")
    ap.add_argument("--transcripts", default=None)
    ap.add_argument("--split", default="core")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--maxlen", type=int, default=3500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_min", type=float, default=0, help="stop and write what is finished after this many minutes")
    a = ap.parse_args()
    dict(attn=cmd_attn, span=cmd_span, selftest=cmd_selftest)[a.mode](a)
