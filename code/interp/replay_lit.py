"""Teacher-forced replay of literature-format debates (outputs/raw_lit): internal state at the moment each agent commits to its answer.

For every generated response (rounds 0..R, debate and reflect, plus the private/public re-queries) the exact chat context and the
generated text are re-encoded. Read-outs:
  decision site  : the token just before the final answer letter inside "(X)" -> next-token distribution over option letters
                   (the unprinted margin of the answer the agent wrote) and residuals on the layer grid;
  reading site   : last prompt token before the response (state after reading the other agents, before any reasoning).
Responses without a parseable "(X)" are recorded with decision fields empty.
Output: ACTS/<M>/lit_replay_<dataset>/{index.jsonl, acts.npy [rows, 2 sites, layers, d], meta.json}
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "interp"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runner"))
from common import *  # noqa
from cpt import Receiver
from prompts_lit import N_AGENTS, PRIVATE_Q, PUBLIC_Q, answer_instruction, answer_span, messages
from debate_vllm import load_items

RAW_LIT = ROOT / "outputs/raw_lit"
MAXLEN = 7000


def layer_grid(n):
    return sorted({int(round(f * (n - 1))) for f in np.linspace(0.1, 1.0, 10)})


def contexts(a, views):
    d = RAW_LIT / a.model / a.dataset
    meta = json.load(open(d / "meta.json"))
    R_ = meta["rounds"]
    recs = {}
    for f in d.glob("*.jsonl"):
        for x in read_jsonl(f):
            recs[(x["item_id"], x["cond"], x["agent"], x["round"], x.get("query"))] = x
    out = []
    for i, v in views.items():
        for cond in ("debate", "reflect"):
            for g in range(N_AGENTS):
                for k in range(0, R_ + 1):
                    x = recs.get((i, "r0" if k == 0 else cond, g, k, None))
                    if x is None or (k == 0 and cond == "reflect"):
                        continue
                    if k == 0:
                        msgs = messages(v, 0, None, None, "debate")
                    else:
                        own = recs[(i, "r0" if k == 1 else cond, g, k - 1, None)]["text"]
                        others = [recs[(i, "r0" if k == 1 else cond, h, k - 1, None)]["text"] for h in range(N_AGENTS) if h != g]
                        msgs = messages(v, k, own, others, cond)
                    out.append((dict(item_id=i, cond="r0" if k == 0 else cond, agent=g, round=k, query=None, answer=x["answer"]), msgs, x["text"]))
                for q, qt in (("private", PRIVATE_Q), ("public", PUBLIC_Q)):
                    x = recs.get((i, cond, g, R_ + 1, q))
                    if x is None:
                        continue
                    own = recs[(i, "r0" if R_ == 1 else cond, g, R_ - 1, None)]["text"]
                    others = [recs[(i, "r0" if R_ == 1 else cond, h, R_ - 1, None)]["text"] for h in range(N_AGENTS) if h != g]
                    msgs = messages(v, R_, own, others, cond) + [{"role": "assistant", "content": recs[(i, cond, g, R_, None)]["text"]},
                                                                {"role": "user", "content": qt + " " + answer_instruction(v)}]
                    out.append((dict(item_id=i, cond=cond, agent=g, round=R_ + 1, query=q, answer=x["answer"]), msgs, x["text"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="core")
    ap.add_argument("--transcripts", default=None, help="model whose transcripts are replayed (cross-reader); default --model")
    ap.add_argument("--rows", type=int, default=16, help="upper bound on rows per batch")
    ap.add_argument("--max_tokens", type=int, default=0,
                    help="token budget per batch (rows x padded length); the real memory constraint. "
                         "0 scales it from the model's depth, which is what sets the activation cost per token")
    ap.add_argument("--max_l2", type=int, default=0,
                    help="budget on rows x padded_length^2, the attention workspace; 0 scales it from model depth")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    R = Receiver(a.model)
    grid = layer_grid(R.n_layers)
    # 32768 was tuned on a 32-layer model and carried to Llama unchanged, which was fine because Llama is also 32.
    # Gemma has 42 layers and per-layer embeddings, so the same token budget buys about a third more activation
    # memory and the allocator began reporting OOM-and-retry again. The budget is a memory quantity, so it scales
    # with depth instead of being a constant that happened to work twice.
    if not a.max_tokens:
        a.max_tokens = int(32768 * 32 / max(R.n_layers, 1))
        print(f"[batching] token budget {a.max_tokens} scaled from {R.n_layers} layers "
              f"(32768 at the 32-layer reference)", flush=True)
    # A token budget caps rows x L, which is right for the captured activations: those are linear in tokens. It is
    # the wrong invariant for the attention workspace, which goes as rows x L^2 = (token budget) x L and therefore
    # GROWS with sequence length even while the token budget is held constant. Rows are length-sorted, so the run
    # walks from short to long and the allocator is squeezed harder the further it gets: on Gemma the OOM-and-retry
    # interval fell monotonically from 232s to 147s across the run. Across this data rows x L^2 spans 0.17e6 to
    # 147e6, an 860-fold range the token budget does not constrain at all. Bound it directly.
    if not a.max_l2:
        a.max_l2 = int(60e6 * 32 / max(R.n_layers, 1))
        print(f"[batching] attention budget rows*L^2 <= {a.max_l2:.3g}, scaled from {R.n_layers} layers", flush=True)
    tm = a.transcripts or a.model
    core = {r["id"]: r for r in load_core()}
    ids_in = {x["item_id"] for x in read_jsonl(RAW_LIT / tm / a.dataset / "r0.jsonl")}
    views = {i: item_view(core[i]) for i in sorted(ids_in)}
    views = {i: v for i, v in views.items() if v["kind"] != "numeric"}
    if a.limit:
        views = dict(list(views.items())[: a.limit])
    ctx = contexts(argparse.Namespace(model=tm, dataset=a.dataset), views)
    print(f"[stage] {len(ctx)} responses to replay", flush=True)
    enc_rows = []
    for key, msgs, text in ctx:
        v = views[key["item_id"]]
        prefix = R.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        span = answer_span(text, v)
        full = prefix + (text[: span[1]] if span else text)  # up to (not including) the final answer letter
        ids = R.tok(full, add_special_tokens=False)["input_ids"]
        n_prefix = len(R.tok(prefix, add_special_tokens=False)["input_ids"])
        if len(ids) > MAXLEN:  # keep the tail; shift the reading site by the number of dropped tokens
            drop = len(ids) - MAXLEN
            ids = ids[drop:]
            n_prefix -= drop
        read_pos = min(max(n_prefix - 1, 0), len(ids) - 1)
        enc_rows.append((key, ids, [len(ids) - 1, read_pos], span is not None))
    order = sorted(range(len(enc_rows)), key=lambda j: len(enc_rows[j][1]))
    out = ACTS / a.model / f"lit_replay_{a.dataset}{'_' + tm + 'text' if a.transcripts else ''}"
    out.mkdir(parents=True, exist_ok=True)
    d = R.model.config.get_text_config().hidden_size
    acts = np.lib.format.open_memmap(out / "acts.npy", mode="w+", dtype=np.float16, shape=(len(enc_rows), 2, len(grid), d))
    index = [None] * len(enc_rows)
    # Pack batches to a token budget rather than a fixed row count. Rows are length-sorted, so a fixed count makes the
    # final batches the longest sequences at full width: memory per batch is rows x padded length, which grew about
    # fivefold from the first batch to the last and was what drove the allocator into repeated OOM-and-retry near the
    # end of the twelve-turn run. Sorted order means the newest row is the longest, so it sets the padded width.
    batches, cur = [], []
    for j in order:
        L = len(enc_rows[j][1])
        if cur and ((len(cur) + 1) * L > a.max_tokens or (len(cur) + 1) * L * L > a.max_l2
                    or len(cur) + 1 > a.rows):
            batches.append(cur)
            cur = []
        cur.append(j)
    if cur:
        batches.append(cur)
    widths = [len(enc_rows[c[-1]][1]) for c in batches]
    print(f"[batching] {len(order)} rows in {len(batches)} batches; rows per batch "
          f"{min(len(c) for c in batches)}-{max(len(c) for c in batches)}, padded width {min(widths)}-{max(widths)}, "
          f"peak tokens {max(len(c) * len(enc_rows[c[-1]][1]) for c in batches)}", flush=True)

    t0 = time.time()
    done = 0
    for bn, chunk in enumerate(batches):
        store = {}
        for li in grid:
            R.add_capture(li, [enc_rows[j][2] for j in chunk], store)
        lp = R.forward([enc_rows[j][1] for j in chunk])
        R.clear()
        for bi, j in enumerate(chunk):
            key, ids, pos, has = enc_rows[j]
            v = views[key["item_id"]]
            rec = dict(key, row=j, n_tok=len(ids), has_answer=has)
            if has:
                rec["lp_letters"] = R.letter_lp(lp[bi], v["labels"])
            index[j] = rec
            for gi, li in enumerate(grid):
                acts[j, :, gi] = store[li][bi].numpy().astype(np.float16)
        done += len(chunk)
        if (bn + 1) % 100 == 0:
            el = time.time() - t0
            print(f"[progress] {done}/{len(order)} responses, eta {el / done * (len(order) - done) / 60:.1f} min",
                  flush=True)
    acts.flush()
    write_jsonl(out / "index.jsonl", index)
    json.dump(dict(model=a.model, transcripts=tm, sha=R.sha, grid=grid, sites=["decision", "reading"]), open(out / "meta.json", "w"), indent=1)
    print(f"[done] lit replay {a.model} on {tm} {a.dataset}: {len(index)} rows in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
