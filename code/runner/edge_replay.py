"""E14 directed one-edge replay A <- B (vLLM, receiver-side scoring).

For every choice item in release_600 where the greedy turn-0 verdicts of receiver A and sender B disagree, A sees its own
turn-0 response and B's turn-0 message in the standard round-2 debate format, and we read A's option-letter distribution
after "VERDICT:". Message variants: full | verdict_only (EVIDENCE: NONE) | full + expert header | full + anonymous header.
influence   = P(A adopts B's verdict | full)
content     = adopt(full) - adopt(verdict_only)
provenance  = adopt(expert) - adopt(anon)
Self edges (A <- A's own second turn-0 sample) give the within-model baseline.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runner"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "interp"))
from common import *  # noqa
from cpt import letter_token_ids
from prompts import AGENT_NAMES, system_prompt, t0_user

WR = json.load(open(ROOT / "configs/wrappers.json"))[0]


def receiver_user(v, own, other, header):
    return (f"Task:\n{v['task']}\n\nRound 2. Your previous response:\n{own.strip()}\n\n"
            f"Responses from the other agents in the previous round:\n\n{header}\n{other.strip()}\n\n"
            "Considering these responses, give your updated response in the required format.")


def strip_evidence(msg):
    out = []
    for line in msg.splitlines():
        if re.match(r"^\s*\**\s*EVIDENCE", line, re.I):
            out.append("EVIDENCE: NONE")
        elif re.match(r"^\s*\**\s*(SOURCE_USE|CONFIDENCE|VERDICT)", line, re.I):
            out.append(line)
    return "\n".join(out)


def run_dir(model, dataset):
    d = RAW / model / dataset
    return d if (d / "greedy.jsonl").exists() else RAW / model / "core"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--receiver", required=True)
    ap.add_argument("--senders", required=True)
    ap.add_argument("--dataset", default="release_600")
    ap.add_argument("--gpu_util", type=float, default=0.9)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    A = args.receiver
    info = MODELS[A]
    path, sha = snapshot(A)
    tok = AutoTokenizer.from_pretrained(path)
    rows = {r["id"]: r for r in load_core()}
    keep = set(json.load(open(ROOT / "data/splits/splits.json"))["release_600"])
    own = {x["item_id"]: x for x in read_jsonl(run_dir(A, args.dataset) / "greedy.jsonl") if x["item_id"] in keep}
    own2 = {x["item_id"]: x for x in read_jsonl(run_dir(A, args.dataset) / "t0.jsonl") if x["agent"] == 1 and x["item_id"] in keep}
    merge = info["family"] == "mistral"
    prompts, meta = [], []
    for B in args.senders.split(","):
        src = own2 if B == A else {x["item_id"]: x for x in read_jsonl(run_dir(B, args.dataset) / "greedy.jsonl") if x["item_id"] in keep}
        for iid, xa in own.items():
            v = item_view(rows[iid])
            xb = src.get(iid)
            if v["kind"] != "choice" or xb is None or not xa["verdict"] or not xb["verdict"] or xa["verdict"] == xb["verdict"]:
                continue
            variants = {
                "full": (f"[{AGENT_NAMES[1]}]", xb["text"]),
                "verdict_only": (f"[{AGENT_NAMES[1]}]", strip_evidence(xb["text"])),
                "expert": (f"[{AGENT_NAMES[1]} — {WR['expert']}]", xb["text"]),
                "anon": (f"[{AGENT_NAMES[1]} — {WR['anon']}]", xb["text"]),
            }
            for var, (hdr, msg) in variants.items():
                sysm = system_prompt(v, 0)
                user = receiver_user(v, xa["text"], msg, hdr)
                msgs = [{"role": "user", "content": sysm + "\n\n" + user}] if merge else [{"role": "system", "content": sysm}, {"role": "user", "content": user}]
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False) + "VERDICT:"
                prompts.append(tok(text, add_special_tokens=False)["input_ids"])
                meta.append(dict(receiver=A, sender=B, item_id=iid, family=v["family"], domain=v["domain"], variant=var,
                                 own_verdict=xa["verdict"], sender_verdict=xb["verdict"], target=v["target"], labels=v["labels"]))
    print(f"[stage] {A}: {len(prompts)} edge prompts", flush=True)
    kw = {}
    if mm_limits(A):
        kw["limit_mm_per_prompt"] = mm_limits(A)
    llm = LLM(model=path, tokenizer=path, dtype="auto", gpu_memory_utilization=args.gpu_util, max_model_len=4096,
              enable_prefix_caching=True, seed=0, max_logprobs=50, max_num_seqs=max_seqs(A), **kw)
    lids = letter_token_ids(tok)
    t = time.time()
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], SamplingParams(temperature=0.0, max_tokens=1, logprobs=50), use_tqdm=False)
    recs = []
    for m, o in zip(meta, outs):
        lp = o.outputs[0].logprobs[0]
        floor = min(x.logprob for x in lp.values())
        letters = {}
        for L in m["labels"]:
            vals = [lp[i].logprob for i in lids[L] if i in lp]
            letters[L] = math.log(sum(math.exp(x) for x in vals)) if vals else floor
        z = math.log(sum(math.exp(x) for x in letters.values()))
        P = {L: math.exp(x - z) for L, x in letters.items()}
        recs.append(dict(m, p_adopt=P[m["sender_verdict"]], p_keep=P[m["own_verdict"]],
                         adopt_margin=letters[m["sender_verdict"]] - letters[m["own_verdict"]],
                         adopt_argmax=max(P, key=P.get) == m["sender_verdict"], receiver_sha=sha))
    d = ACTS / "edges"
    d.mkdir(parents=True, exist_ok=True)
    write_jsonl(d / f"edges_{A}.jsonl", recs)
    print(f"[done] edge replay receiver {A}: {len(recs)} in {(time.time()-t)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
