"""Does the reading-time state cause the switch, and the reasoning that follows it? (literature-format debates, HF generation)

For an agent at round k we build two contexts that end at the same generation boundary:
  D = the real debate round message (the peers' responses are visible)
  N = the no-communication message for the same agent and round (its own previous response only)
The reading site is the assistant-turn header -- the last SPAN tokens, byte-identical in both contexts, at which the model has read
the round message and written nothing. D's residual state there is captured on a layer grid and written into a generation from N at
the same tokens and one layer, during the prefill only; the model then writes its response normally. If the decision is already in
that state, the model produces the peers' answer out of a context that never showed it the peers -- with reasoning that has no peer
to refer to.

Arms (one batch per episode, 2 + 5 x |LAYERS| rows):
  none            N, unpatched                                     what the agent does without the peers
  debate@L        N's reading state replaced by D's at layer L      the interchange intervention
  debate_x3@L     the same difference applied three times over      dose-response
  randdiff@L      a random difference of the same magnitude         equal-magnitude control for debate@L
  randdiff_x3@L   three times that random difference                equal-magnitude control for debate_x3@L
  cross@L         D's reading state from another item, same family  donor-specificity control
  debate_ref      D, unpatched                                      the real behaviour being explained

Two read-outs per row: the answer the model actually writes (parsed), and -- defined even when the response runs past the token
budget -- the letter distribution when its own reasoning is cut at the final answer and the answer is forced ("... {final answer: (").
Output: ACTS/<M>/lit_prewrite_<dataset>/rows.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time

import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "interp"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runner"))
from common import *  # noqa
from cpt import Receiver
from prompts_lit import N_AGENTS, answer_span, messages, parse_answer

RAW_LIT = ROOT / "outputs/raw_lit"
LAYERS = (9, 12, 16, 19, 22)
SPAN = 4          # tokens of the assistant-turn header that are transplanted
PREFILL = "{final answer: ("
# does the written reasoning refer to anyone else at all?
PEER_RE = re.compile(r"(\bagents?\b|\bthe other\b|\bothers\b|\bpeers?\b|\bconsensus\b|their (?:answer|response|reasoning|argument|"
                     r"point|view|position)|they (?:argue|argued|claim|claimed|point|pointed|suggest|suggested|state|stated))", re.I)


def load(tm, dataset):
    d = RAW_LIT / tm / dataset
    recs = {}
    for f in d.glob("*.jsonl"):
        for x in read_jsonl(f):
            recs[(x["item_id"], x["cond"], x["agent"], x["round"], x.get("query"))] = x
    return recs, json.load(open(d / "meta.json"))["rounds"]


def pick_episodes(recs, views, core, n_conform, n_keep):
    """Conformity cases: the answer changed in debate onto an answer a peer gave. Keepers: same setting, answer unchanged."""
    conf, keep = [], []
    for (i, cond, g, k, q), x in recs.items():
        if q is not None or cond != "debate" or k == 0 or i not in views or x["answer"] is None:
            continue
        pc = "r0" if k == 1 else "debate"
        prev = recs.get((i, pc, g, k - 1, None))
        peers = [recs.get((i, pc, h, k - 1, None)) for h in range(N_AGENTS) if h != g]
        if prev is None or prev["answer"] is None or any(p is None or p["answer"] is None for p in peers):
            continue
        own_prev, peer_ans = prev["answer"], [p["answer"] for p in peers]
        dis = [a for a in peer_ans if a != own_prev]
        if not dis:  # no disagreeing peer -> nothing to be moved toward
            continue
        rec = dict(item_id=i, agent=g, round=k, own_prev=own_prev, written=x["answer"],
                   target=collections.Counter(dis).most_common(1)[0][0], changed=x["answer"] != own_prev,
                   family=family_of(core[i]), domain=core[i]["primary_domain"])
        if rec["changed"]:
            if x["answer"] in peer_ans:  # moved onto a peer's answer
                conf.append(rec)
        else:
            keep.append(rec)
    key = lambda r: h64("prewrite", r["item_id"], r["agent"], r["round"])
    conf, keep = sorted(conf, key=key), sorted(keep, key=key)
    print(f"[stage] available: {len(conf)} conformity, {len(keep)} keepers", flush=True)
    return conf[:n_conform] + keep[:n_keep]


def build_contexts(R, views, recs, ep):
    i, g, k = ep["item_id"], ep["agent"], ep["round"]
    v = views[i]
    pc = "r0" if k == 1 else "debate"
    own = recs[(i, pc, g, k - 1, None)]["text"]
    others = [recs[(i, pc, h, k - 1, None)]["text"] for h in range(N_AGENTS) if h != g]
    enc = lambda msgs: R.tok(R.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False),
                             add_special_tokens=False)["input_ids"][-7000:]
    return enc(messages(v, k, own, others, "debate")), enc(messages(v, k, own, None, "reflect"))


def make_hook(rows_vals):
    """rows_vals: (row, start column, value [span, d]). Prefill only -- generated steps arrive with h.shape[1] == 1."""
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] > 1:
            for r, c, v in rows_vals:
                h[r, c:c + v.shape[0]] = v.to(h.dtype).to(h.device)
        return out
    return hook


def pad_batch(ids_list, pad):
    """Left-padded, so every row's last real token is the final column."""
    n, T = len(ids_list), max(len(x) for x in ids_list)
    inp = torch.full((n, T), pad, dtype=torch.long)
    att = torch.zeros((n, T), dtype=torch.long)
    for r, x in enumerate(ids_list):
        inp[r, T - len(x):] = torch.tensor(x)
        att[r, T - len(x):] = 1
    return inp, att, T


def hooks_for(R, rows, ids_list, T, span):
    """One hook per patched layer; the reading site of a row ends span tokens after the end of its own prompt."""
    by_layer = {}
    for r, (_, _, patch, n_prompt) in enumerate(rows):
        if patch is not None:
            by_layer.setdefault(patch[0], []).append((r, T - len(ids_list[r]) + n_prompt - span, patch[1]))
    return [R.layers[L].register_forward_hook(make_hook(rv)) for L, rv in by_layer.items()]


@torch.inference_mode()
def generate(R, rows, max_new, pad, span):
    ids_list = [r[1] for r in rows]
    inp, att, T = pad_batch(ids_list, pad)
    handles = hooks_for(R, rows, ids_list, T, span)
    try:
        out = R.model.generate(input_ids=inp.to(R.device), attention_mask=att.to(R.device), max_new_tokens=max_new,
                               do_sample=False, pad_token_id=pad)
    finally:
        for h in handles:
            h.remove()
    gen = out[:, T:].cpu()
    return [R.tok.decode(g, skip_special_tokens=True) for g in gen], [int((g != pad).sum()) for g in gen]


@torch.inference_mode()
def forced_lp(R, rows, ids_list, pad, span, chunk=8):
    """Log-probs at the forced answer token, in the same patched world (the patch sits inside the prompt, not at the end)."""
    outs = []
    for s in range(0, len(rows), chunk):
        rw, ids = rows[s:s + chunk], ids_list[s:s + chunk]
        inp, att, T = pad_batch(ids, pad)
        handles = hooks_for(R, rw, ids, T, span)
        try:
            out = R.model.model(input_ids=inp.to(R.device), attention_mask=att.to(R.device), use_cache=False)
        finally:
            for h in handles:
                h.remove()
        logits = torch.nn.functional.linear(out.last_hidden_state[:, -1].float(), R.W32)
        if R.softcap:
            logits = torch.tanh(logits / R.softcap) * R.softcap
        outs.append(torch.log_softmax(logits.float(), -1).cpu())
        del out, logits
    return torch.cat(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="lit1200")
    ap.add_argument("--transcripts", default=None)
    ap.add_argument("--n_conform", type=int, default=140)
    ap.add_argument("--n_keep", type=int, default=60)
    ap.add_argument("--max_new", type=int, default=320)
    ap.add_argument("--span", type=int, default=SPAN)
    ap.add_argument("--layers", default=",".join(map(str, LAYERS)))
    ap.add_argument("--chunk", type=int, default=8, help="rows per forward in the forced read-out pass")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    R = Receiver(a.model, dtype=torch.float32 if a.device == "cpu" else torch.bfloat16, device=a.device)
    layers = [int(x) for x in a.layers.split(",") if int(x) < R.n_layers]
    tm = a.transcripts or a.model
    recs, _ = load(tm, a.dataset)
    core = {r["id"]: r for r in load_core()}
    views = {i: item_view(core[i]) for i in sorted({k[0] for k in recs})}
    views = {i: v for i, v in views.items() if v["kind"] != "numeric"}
    eps = pick_episodes(recs, views, core, a.n_conform, a.n_keep)
    pad = R.tok.pad_token_id if R.tok.pad_token_id is not None else R.tok.eos_token_id
    prefill_ids = R.tok("\n\n" + PREFILL, add_special_tokens=False)["input_ids"]
    t0 = time.time()

    # ---- phase 1: the reading-site state of both contexts, on the layer grid
    ctx, S = {}, {}
    for ei, ep in enumerate(eps):
        d_ids, n_ids = build_contexts(R, views, recs, ep)
        if d_ids[-a.span:] != n_ids[-a.span:]:  # transplant only where the tokens themselves match
            continue
        ctx[ei] = (d_ids, n_ids)
        got = {}
        for tag, ids in (("D", d_ids), ("N", n_ids)):
            store = {}
            for L in layers:
                R.add_capture(L, [list(range(len(ids) - a.span, len(ids)))], store)
            R.forward([ids])
            R.clear()
            got[tag] = {L: store[L][0] for L in layers}  # [span, d]
        S[ei] = got
    print(f"[stage] {len(S)} episodes, layers {layers}, span {a.span}, {a.max_new} new tokens "
          f"(states captured in {(time.time()-t0)/60:.1f} min)", flush=True)

    # ---- donor for the cross-item control: another episode of the same family, a different item
    by_fam = collections.defaultdict(list)
    for ei in S:
        by_fam[eps[ei]["family"]].append(ei)
    donor = {}
    for ei in S:
        ep = eps[ei]
        pool = [j for j in by_fam[ep["family"]] if eps[j]["item_id"] != ep["item_id"]]
        donor[ei] = pool[h64("donor", ep["item_id"], ep["agent"], ep["round"]) % len(pool)] if pool else None

    gen_ = torch.Generator().manual_seed(240091)
    out_rows, done = [], 0
    for ei in sorted(S):
        ep, (d_ids, n_ids) = eps[ei], ctx[ei]
        v_ = views[ep["item_id"]]
        # (name, ids, patch, n_prompt); patch = (layer, value written at the reading site)
        rows, kinds = [("none", n_ids, None, len(n_ids))], [("none", None, 0.0)]
        for L in layers:
            hD, hN = S[ei]["D"][L], S[ei]["N"][L]
            delta = hD - hN
            r = torch.randn(delta.shape, generator=gen_)
            r = r / r.norm() * delta.norm()  # equal-magnitude random difference
            arms = [("debate", hD, 1.0), ("debate_x3", hN + 3 * delta, 3.0),
                    ("randdiff", hN + r, 1.0), ("randdiff_x3", hN + 3 * r, 3.0)]
            if donor[ei] is not None:
                arms.append(("cross", S[donor[ei]]["D"][L], 1.0))
            for kind, val, al in arms:
                rows.append((f"{kind}@{L}", n_ids, (L, val), len(n_ids)))
                kinds.append((kind, L, al))
        rows.append(("debate_ref", d_ids, None, len(d_ids)))
        kinds.append(("debate_ref", None, 0.0))

        texts, ntok = generate(R, rows, a.max_new, pad, a.span)
        # forced read-out: the model's own reasoning, cut at its final answer, then the answer forced
        fids, bodies = [], []
        for (name, ids, patch, npf), text in zip(rows, texts):
            sp = answer_span(text, v_)
            body = (text[: sp[0]] if sp else text).rstrip()
            bodies.append(body)
            fids.append(ids + R.tok(body, add_special_tokens=False)["input_ids"] + prefill_ids)
        lp = forced_lp(R, rows, fids, pad, a.span, a.chunk)

        for bi, ((name, ids, patch, npf), (kind, L, al), text, nt) in enumerate(zip(rows, kinds, texts, ntok)):
            ans = parse_answer(text, v_)
            out_rows.append(dict(ep, arm=name, kind=kind, layer=L, alpha=al, answer=ans, complete=ans is not None,
                                 is_target=(None if ans is None else ans == ep["target"]),
                                 is_own_prev=(None if ans is None else ans == ep["own_prev"]),
                                 lp_letters=R.letter_lp(lp[bi], v_["labels"]),
                                 mentions_peers=bool(PEER_RE.search(bodies[bi])), n_gen=nt, n_prompt=npf,
                                 donor_item=(eps[donor[ei]]["item_id"] if kind == "cross" else None), text=text))
        done += 1
        if done % 10 == 0:
            el = time.time() - t0
            print(f"[progress] {done}/{len(S)} episodes, eta {el/done*(len(S)-done)/60:.1f} min", flush=True)

    out = ACTS / a.model / f"lit_prewrite_{a.dataset}{'_' + tm + 'text' if a.transcripts else ''}"
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "rows.jsonl", out_rows)
    json.dump(dict(model=a.model, transcripts=tm, layers=layers, span=a.span, max_new=a.max_new, prefill=PREFILL,
                   n_episodes=len(S), n_conform=sum(eps[ei]["changed"] for ei in S), peer_regex=PEER_RE.pattern),
              open(out / "meta.json", "w"), indent=1)
    print(f"[done] lit prewrite {a.model}: {len(S)} episodes, {len(out_rows)} rows in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
