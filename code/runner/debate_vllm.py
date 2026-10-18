"""Solo / independent-pool / three-agent debate runner (vLLM). Resumable per round.

Output: outputs/raw/<MODEL>/<dataset>/{t0,debate_t{r},pool_t{r},greedy,behavior}.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runner"))
from common import *  # noqa
from prompts import BEHAVIOR_FOLLOWUP, PRIVATE_REQUERY, build_messages

GEN = dict(temperature=0.7, top_p=0.95)
BEHAVIOR_STRATA = {"attack", "refusal_calibration", "refusal_boundary", "abstention"}


def load_items(name):
    splits = json.load(open(ROOT / "data/splits/splits.json"))
    if name == "core":
        rows = load_core()
    elif name in ("hero_48", "sprint_240", "release_600"):
        keep = set(splits[name])
        rows = [r for r in load_core() if r["id"] in keep]
    elif name.startswith("strat"):  # stratified_600 etc: release_600
        keep = set(splits["release_600"])
        rows = [r for r in load_core() if r["id"] in keep]
    elif name == "smoke30":
        import collections
        seen = collections.Counter(); rows = []
        for r in load_core():
            if seen[family_of(r)] < 2 and r["id"] in set(splits["release_600"]):
                seen[family_of(r)] += 1; rows.append(r)
    elif name == "lit1200b":
        # The other half of the benchmark: every core item not in lit1200. Running this alongside lit1200 covers all
        # 2,400 items without regenerating the half already done.
        return [r for r in load_core() if r["id"] not in {x["id"] for x in load_items("lit1200")}]
    elif name == "lit1200":  # release_600 + 600 hash-selected core items outside release_600
        rel = set(splits["release_600"])
        core = load_core()
        extra = sorted([r for r in core if r["id"] not in rel], key=lambda r: h64("lit1200", r["id"]))[:600]
        rows = [r for r in core if r["id"] in rel] + extra
    elif name == "original":
        rows = read_jsonl(ROOT / "data/original_v2view.jsonl")
    else:
        raise ValueError(name)
    return rows


def seed_of(*parts):
    return h64("seed", *parts) % (2**31 - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--solo_k", type=int, default=5)
    ap.add_argument("--conds", default="debate,pool")
    ap.add_argument("--gpu_util", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=6144)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_behavior", action="store_true")
    ap.add_argument("--max_num_seqs", type=int, default=512)
    ap.add_argument("--private", action="store_true", help="E8 private re-query after the final round")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    info = MODELS[args.model]
    path, sha = snapshot(args.model)
    out_dir = RAW / args.model / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_items(args.dataset)
    if args.limit:
        rows = rows[: args.limit]
    views = {r["id"]: item_view(r) for r in rows}
    merge_system = info["family"] in ("mistral",)
    kw = {}
    if mm_limits(args.model):
        kw["limit_mm_per_prompt"] = mm_limits(args.model)
    t_load = time.time()
    llm = LLM(model=path, tokenizer=path, dtype="auto", gpu_memory_utilization=args.gpu_util,
              max_model_len=min(args.max_model_len, info.get("max_len", args.max_model_len)), enable_prefix_caching=True, seed=0, max_num_seqs=min(args.max_num_seqs, max_seqs(args.model, args.max_num_seqs)), **kw)
    load_s = time.time() - t_load
    tmpl_kwargs = {"enable_thinking": False}
    meta = dict(model=args.model, repo=info["repo"], model_sha=sha, precision=info["precision"], dataset=args.dataset,
                gen=GEN, turns=args.turns, solo_k=args.solo_k, load_s=load_s, thinking=False)
    json.dump(meta, open(out_dir / "meta.json", "w"), indent=1)

    def max_tokens(v):
        return 512 if v["kind"] == "deception" else 180

    def run(tag, jobs):
        """jobs: list of (key dict, messages, SamplingParams). Writes tag.jsonl; returns {key tuple: text}."""
        fp = out_dir / f"{tag}.jsonl"
        if fp.exists():
            done = read_jsonl(fp)
            if len(done) == len(jobs):
                print(f"[skip] {tag} ({len(done)})", flush=True)
                return done
        t = time.time()
        outs = llm.chat([j[1] for j in jobs], [j[2] for j in jobs], use_tqdm=False,
                        chat_template_kwargs=tmpl_kwargs)
        recs = []
        for (key, msgs, sp), o in zip(jobs, outs):
            v = views[key["item_id"]]
            text = o.outputs[0].text
            p = parse_message(text, v)
            rec = dict(model=args.model, model_sha=sha, **key, text=text, **p,
                       correct=is_correct(p["verdict"], v), finish=o.outputs[0].finish_reason,
                       n_prompt=len(o.prompt_token_ids), n_gen=len(o.outputs[0].token_ids), seed=sp.seed)
            recs.append(rec)
        tmp = fp.with_suffix(".tmp")
        write_jsonl(tmp, recs)
        os.replace(tmp, fp)
        dt = time.time() - t
        ntok = sum(r["n_gen"] for r in recs)
        valid = sum(r["parser_valid"] for r in recs) / max(1, len(recs))
        print(f"[done] {tag}: {len(recs)} gens, {ntok} toks, {dt:.0f}s ({ntok/dt:.0f} tok/s), parser_valid={valid:.3f}", flush=True)
        return recs

    ids = list(views)
    # ---- turn 0: shared across conditions; agents 0..solo_k-1 (debate uses 0..2) ----
    jobs = []
    for iid in ids:
        v = views[iid]
        for a in range(args.solo_k):
            # agents 0-2 are the debaters; extra solo samples (a>=3) reuse Agent 1's prompt with new seeds
            msgs = build_messages(v, a if a < 3 else 0, 0, "debate", None, merge_system)
            sp = SamplingParams(**GEN, max_tokens=max_tokens(v), seed=seed_of(args.model, iid, a, 0, "t0"))
            jobs.append((dict(item_id=iid, cond="t0", agent=a, turn=0), msgs, sp))
    t0 = run("t0", jobs)
    greedy_jobs = [(dict(item_id=iid, cond="greedy", agent=0, turn=0), build_messages(views[iid], 0, 0, "debate", None, merge_system),
                    SamplingParams(temperature=0.0, max_tokens=max_tokens(views[iid]), seed=0)) for iid in ids]
    run("greedy", greedy_jobs)

    hist = {c: {iid: {0: {}} for iid in ids} for c in args.conds.split(",")}
    for r in t0:
        if r["agent"] < 3:
            for c in hist:
                hist[c][r["item_id"]][0][r["agent"]] = r["text"]
    for turn in range(1, args.turns):
        for c in hist:
            jobs = []
            for iid in ids:
                v = views[iid]
                for a in range(3):
                    msgs = build_messages(v, a, turn, c, hist[c][iid], merge_system)
                    sp = SamplingParams(**GEN, max_tokens=max_tokens(v), seed=seed_of(args.model, iid, a, turn, c))
                    jobs.append((dict(item_id=iid, cond=c, agent=a, turn=turn), msgs, sp))
            recs = run(f"{c}_t{turn}", jobs)
            for rr in recs:
                hist[c][rr["item_id"]].setdefault(turn, {})[rr["agent"]] = rr["text"]

    # ---- behavioral reply for refusal/abstention items: T0 and final turn, all 3 agents ----
    if not args.no_behavior:
        jobs = []
        last = args.turns - 1
        for iid in ids:
            v = views[iid]
            if v["stratum"] not in BEHAVIOR_STRATA:
                continue
            for c in hist:
                for turn in (0, last):
                    if turn == 0 and c != list(hist)[0]:
                        continue
                    for a in range(3):
                        msgs = build_messages(v, a, turn, c, hist[c][iid], merge_system)
                        msgs = msgs + [{"role": "assistant", "content": hist[c][iid][turn][a]},
                                       {"role": "user", "content": BEHAVIOR_FOLLOWUP}]
                        sp = SamplingParams(temperature=0.0, max_tokens=160, seed=0)
                        jobs.append((dict(item_id=iid, cond=("t0" if turn == 0 else c), agent=a, turn=turn), msgs, sp))
        if jobs:
            run("behavior", jobs)
    # ---- E8: private re-query with transcript retained (public/private gap, old-answer retention) ----
    if args.private:
        jobs = []
        last = args.turns - 1
        for iid in ids:
            v = views[iid]
            for c in hist:
                for a in range(3):
                    msgs = build_messages(v, a, last, c, hist[c][iid], merge_system)
                    msgs = msgs + [{"role": "assistant", "content": hist[c][iid][last][a]},
                                   {"role": "user", "content": PRIVATE_REQUERY}]
                    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens(v), seed=0)
                    jobs.append((dict(item_id=iid, cond=f"private_{c}", agent=a, turn=last + 1), msgs, sp))
        run("private", jobs)
    print("[complete]", args.model, args.dataset, flush=True)


if __name__ == "__main__":
    main()
