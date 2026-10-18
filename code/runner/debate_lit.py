"""Literature-standard multi-agent debate runner (vLLM). See runner/prompts_lit.py for the protocol.

Conditions: debate (agents see each other's most recent responses) and reflect (no-communication control, same rounds).
Round 0 is shared. Re-queries are greedy. After the last round: private and public re-queries in both conditions (same context, one extra user turn).
Output: outputs/raw_lit/<MODEL>/<dataset>/{r0,debate_r{k},reflect_r{k},requery}.jsonl (resumable per stage).
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
from prompts_lit import N_AGENTS, PRIVATE_Q, PUBLIC_Q, answer_instruction, messages, parse_answer
from debate_vllm import load_items, seed_of

RAW_LIT = ROOT / "outputs/raw_lit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--conds", default="debate,reflect")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_tokens", type=int, default=2048)  # Choi et al. use 512; Qwen3.5 truncates 28-50% of responses at 512
    ap.add_argument("--gpu_util", type=float, default=0.92)
    ap.add_argument("--max_model_len", type=int, default=12288)
    ap.add_argument("--max_num_seqs", type=int, default=512)
    ap.add_argument("--seeded", action="store_true", help="per-request seeds (reproducible, slower sampling)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_requery", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams

    info = MODELS[a.model]
    path, sha = snapshot(a.model)
    out = RAW_LIT / a.model / (a.dataset + a.tag)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_items(a.dataset)
    if a.limit:
        rows = rows[: a.limit]
    views = {r["id"]: item_view(r) for r in rows}
    kw = {}
    if mm_limits(a.model):
        kw["limit_mm_per_prompt"] = mm_limits(a.model)
    t_load = time.time()
    llm = LLM(model=path, tokenizer=path, dtype="auto", gpu_memory_utilization=a.gpu_util,
              max_model_len=min(a.max_model_len, info.get("max_len", a.max_model_len)), enable_prefix_caching=True, seed=0,
              max_num_seqs=min(a.max_num_seqs, max_seqs(a.model, a.max_num_seqs)), **kw)
    merge = info["family"] in ("mistral",)
    json.dump(dict(model=a.model, sha=sha, dataset=a.dataset, rounds=a.rounds, agents=N_AGENTS, temperature=a.temperature,
                   top_p=a.top_p, max_tokens=a.max_tokens, seeded=a.seeded, load_s=time.time() - t_load,
                   protocol=("prompt format and decoding from Choi et al. 2025, Debate or Vote (arXiv 2508.17536), App. B.1 and A.1: "
                             "temperature 1.0, top-p 0.9; max_tokens raised from their 512 because 28-50%% of Qwen3.5 responses "
                             "truncated at 512. 3 agents following Du et al. 2023 (arXiv 2305.14325), whose own main setting is "
                             "3 agents x 2 debate rounds; we use 3 debate rounds, inside the T in {2,3,5} that Choi et al. "
                             "themselves run (they use N=5 agents; we do not). Round convention: an initial independent answer "
                             "(r0) followed by R debate rounds, matching Choi et al.'s y_{i,0} then t=1..T, so R=3 means 4 "
                             "generations per agent. Memory is one step: each round shows only the previous round's responses, "
                             "as in Choi et al., not an accumulating transcript.")),
              open(out / "meta.json", "w"), indent=1)

    def fix(msgs):
        if merge and msgs and msgs[0]["role"] == "system":
            msgs = [{"role": "user", "content": msgs[0]["content"] + "\n\n" + msgs[1]["content"]}] + msgs[2:]
        return msgs

    def sp(*key, temperature=None, max_tokens=None):
        kw_ = dict(temperature=a.temperature if temperature is None else temperature, top_p=a.top_p if (temperature or a.temperature) > 0 else 1.0,
                   max_tokens=max_tokens or a.max_tokens)
        if a.seeded:
            kw_["seed"] = seed_of(a.model, *key)
        return SamplingParams(**kw_)

    def run(tag, jobs):
        fp = out / f"{tag}.jsonl"
        if fp.exists():
            done = read_jsonl(fp)
            if len(done) == len(jobs):
                print(f"[skip] {tag}", flush=True)
                return done
        t = time.time()
        # Stage start is logged as well as stage end. The generation call is one blocking request for the whole stage,
        # so a long stage is otherwise indistinguishable from a hang: the requery pass on 1,200 items ran 50 minutes
        # without writing a line, and telling those two apart took a check of the engine subprocess's CPU time.
        print(f"[start] {tag}: {len(jobs)} generations", flush=True)
        outs = llm.chat([fix(j[1]) for j in jobs], [j[2] for j in jobs], use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
        recs = []
        for (key, msgs, _), o in zip(jobs, outs):
            v = views[key["item_id"]]
            text = o.outputs[0].text
            ans = parse_answer(text, v)
            recs.append(dict(model=a.model, **key, text=text, answer=ans, correct=is_correct(ans, v) if ans is not None else None,
                             finish=o.outputs[0].finish_reason, n_prompt=len(o.prompt_token_ids), n_gen=len(o.outputs[0].token_ids)))
        tmp = fp.with_suffix(".tmp")
        write_jsonl(tmp, recs)
        os.replace(tmp, fp)
        dt = time.time() - t
        ntok = sum(r["n_gen"] for r in recs)
        print(f"[done] {tag}: {len(recs)} gens, {ntok} toks, {dt:.0f}s ({ntok/dt:.0f} tok/s), parsed={sum(r['answer'] is not None for r in recs)/len(recs):.3f}, "
              f"truncated={sum(r['finish']=='length' for r in recs)/len(recs):.3f}", flush=True)
        return recs

    ids = list(views)
    jobs = [(dict(item_id=i, cond="r0", agent=g, round=0), messages(views[i], 0, None, None, "debate"), sp(i, g, 0, "r0"))
            for i in ids for g in range(N_AGENTS)]
    r0 = run("r0", jobs)
    conds = a.conds.split(",")
    hist = {c: {i: {0: {}} for i in ids} for c in conds}
    for r in r0:
        for c in conds:
            hist[c][r["item_id"]][0][r["agent"]] = r["text"]
    for k in range(1, a.rounds + 1):
        for c in conds:
            jobs = []
            for i in ids:
                for g in range(N_AGENTS):
                    own = hist[c][i][k - 1][g]
                    others = [hist[c][i][k - 1][h] for h in range(N_AGENTS) if h != g]
                    jobs.append((dict(item_id=i, cond=c, agent=g, round=k), messages(views[i], k, own, others, c), sp(i, g, k, c)))
            for r in run(f"{c}_r{k}", jobs):
                hist[c][r["item_id"]].setdefault(k, {})[r["agent"]] = r["text"]
    if not a.no_requery:
        jobs = []
        for c in conds:
            for i in ids:
                v = views[i]
                for g in range(N_AGENTS):
                    own = hist[c][i][a.rounds - 1][g]
                    others = [hist[c][i][a.rounds - 1][h] for h in range(N_AGENTS) if h != g]
                    base = messages(v, a.rounds, own, others, c) + [{"role": "assistant", "content": hist[c][i][a.rounds][g]}]
                    for q, qt in (("private", PRIVATE_Q), ("public", PUBLIC_Q)):
                        jobs.append((dict(item_id=i, cond=c, agent=g, round=a.rounds + 1, query=q),
                                     base + [{"role": "user", "content": qt + " " + answer_instruction(v)}],
                                     sp(i, g, c, q, temperature=0.0, max_tokens=1024)))
        run("requery", jobs)
    print("[complete]", a.model, a.dataset, flush=True)


if __name__ == "__main__":
    main()
