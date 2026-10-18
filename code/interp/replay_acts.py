"""Replay of finished debates (C2/C4): re-run every (cond, agent, turn) prompt through the HF model with the reply
prefilled up to the decision field and capture residual states on a normalized layer grid at two sites:
  0 = decision boundary (last prefilled token), 1 = reading site (last token of the messages the agent just read:
      peers' messages in debate, its own previous response in pool, the task at turn 0),
plus option-letter log-probs (a pre-decode readout of the public decision). Fixed-shape batches of 12 rows."""
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
from prompts import build_messages
from run_cpt import layer_grid

MARKERS = ["\n\nConsidering these responses", "\n\nReconsider the task", "\n\nGive your response"]
ROWS = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="release_600")
    ap.add_argument("--turns", type=int, default=4)
    args = ap.parse_args()
    R = Receiver(args.model)
    grid = layer_grid(R.n_layers)
    splits = json.load(open(ROOT / "data/splits/splits.json"))
    keep = set(splits[args.split]) if args.split != "all" else None
    d = RAW / args.model / args.dataset
    rows = {r["id"]: r for r in load_core()}
    texts = {}
    for f in ["t0.jsonl"] + [f"{c}_t{t}.jsonl" for c in ("debate", "pool") for t in range(1, args.turns)]:
        for x in read_jsonl(d / f):
            if x["agent"] < 3 and (keep is None or x["item_id"] in keep):
                texts[(x["item_id"], x["cond"], x["agent"], x["turn"])] = x
    items = sorted({k[0] for k in texts})
    out_dir = ACTS / args.model / f"replay_{args.dataset}_{args.split}_T{args.turns}"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_item = 3 + 2 * 3 * (args.turns - 1)
    hdim = R.model.config.get_text_config().hidden_size
    acts = np.lib.format.open_memmap(out_dir / "acts.npy", mode="w+", dtype=np.float16,
                                     shape=(len(items) * per_item, 2, len(grid), hdim))
    index = []
    t0 = time.time()
    n = 0
    for ii, iid in enumerate(items):
        v = item_view(rows[iid])
        prefill = "RESPONSE:" if v["kind"] == "deception" else "VERDICT:"
        prompts, keys = [], []
        for cond in ("debate", "pool"):
            hist = {0: {a: texts[(iid, "t0", a, 0)]["text"] for a in range(3)}}
            for t in range(1, args.turns):
                hist[t] = {a: texts[(iid, cond, a, t)]["text"] for a in range(3)}
            for t in range(args.turns):
                if t == 0 and cond == "pool":
                    continue
                for a in range(3):
                    msgs = build_messages(v, a, t, cond, hist)
                    prefix = R.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    enc = R.tok(prefix + prefill, add_special_tokens=False, return_offsets_mapping=True)
                    cut = min([prefix.find(mk) for mk in MARKERS if prefix.find(mk) >= 0])
                    read_pos = max(i for i, (s, e) in enumerate(enc["offset_mapping"]) if e <= cut)
                    src = texts[(iid, "t0" if t == 0 else cond, a, t)]
                    prompts.append((enc["input_ids"], [len(enc["input_ids"]) - 1, read_pos]))
                    keys.append(dict(item_id=iid, cond=("t0" if t == 0 else cond), agent=a, turn=t, verdict=src["verdict"],
                                     correct=src["correct"], confidence=src["confidence"], n_tok=len(enc["input_ids"])))
        labels = v["labels"] or []
        for b0 in range(0, len(prompts), ROWS):
            chunk = prompts[b0:b0 + ROWS]
            pad = ROWS - len(chunk)
            batch = chunk + [chunk[-1]] * pad
            store = {}
            for li in grid:
                R.add_capture(li, [p for _, p in batch], store)
            lp = R.forward([ids for ids, _ in batch])
            R.clear()
            for bi in range(len(chunk)):
                k = dict(keys[b0 + bi])
                k["lp_letters"] = R.letter_lp(lp[bi], labels) if labels else None
                k["row"] = n
                for gi, li in enumerate(grid):
                    acts[n, :, gi] = store[li][bi].numpy().astype(np.float16)
                index.append(k)
                n += 1
        if (ii + 1) % 25 == 0:
            el = time.time() - t0
            print(f"[progress] {ii+1}/{len(items)} items, eta {el/(ii+1)*(len(items)-ii-1)/60:.1f} min", flush=True)
    acts.flush()
    write_jsonl(out_dir / "index.jsonl", index)
    json.dump(dict(model=args.model, sha=R.sha, grid=grid, n_rows=n, split=args.split, turns=args.turns,
                   sites=["decision", "reading"]), open(out_dir / "meta.json", "w"), indent=1)
    print(f"[done] replay {args.model} {args.dataset} T={args.turns}: {n} states in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
