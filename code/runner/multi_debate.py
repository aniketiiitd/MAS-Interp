"""E15 heterogeneous three-agent debate with all 6 model->slot permutations (hero_48).

Run as steps: `--triad Q9,G4,Q27 --turn t --model m` generates turn t for every (item, permutation, slot) occupied by m,
reading earlier turns from disk. A driver loops turns x models (one vLLM load per step)."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runner"))
from common import *  # noqa
from prompts import build_messages
from debate_vllm import GEN, load_items, seed_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triad", required=True)
    ap.add_argument("--turn", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="hero_48")
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    triad = args.triad.split(",")
    perms = list(itertools.permutations(triad))
    out_dir = RAW / "multi" / "_".join(triad) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"t{args.turn}_{args.model}.jsonl"
    if fp.exists():
        print(f"[skip] {fp.name}", flush=True)
        return
    rows = load_items(args.dataset)
    views = {r["id"]: item_view(r) for r in rows}
    hist = {}
    for t in range(args.turn):
        for m in triad:
            for x in read_jsonl(out_dir / f"t{t}_{m}.jsonl"):
                hist.setdefault((x["item_id"], x["perm"]), {}).setdefault(t, {})[x["agent"]] = x["text"]
    info = MODELS[args.model]
    path, sha = snapshot(args.model)
    kw = {}
    if mm_limits(args.model):
        kw["limit_mm_per_prompt"] = mm_limits(args.model)
    llm = LLM(model=path, tokenizer=path, dtype="auto", gpu_memory_utilization=0.9, max_model_len=4096,
              enable_prefix_caching=True, seed=0, max_num_seqs=max_seqs(args.model), **kw)
    jobs = []
    for iid, v in views.items():
        for pi, perm in enumerate(perms):
            for slot, m in enumerate(perm):
                if m != args.model:
                    continue
                msgs = build_messages(v, slot, args.turn, "debate", hist.get((iid, pi), {}), info["family"] == "mistral")
                sp = SamplingParams(**GEN, max_tokens=512 if v["kind"] == "deception" else 180,
                                    seed=seed_of("multi", args.triad, iid, pi, slot, args.turn))
                jobs.append((dict(item_id=iid, perm=pi, perm_models=list(perm), agent=slot, model=m, turn=args.turn), msgs, sp))
    t = time.time()
    outs = llm.chat([j[1] for j in jobs], [j[2] for j in jobs], use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
    recs = []
    for (k, _, _), o in zip(jobs, outs):
        v = views[k["item_id"]]
        p = parse_message(o.outputs[0].text, v)
        recs.append(dict(k, text=o.outputs[0].text, **p, correct=is_correct(p["verdict"], v), model_sha=sha))
    write_jsonl(fp, recs)
    print(f"[done] multi {args.triad} t{args.turn} {args.model}: {len(recs)} in {time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
