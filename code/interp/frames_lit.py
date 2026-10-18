"""Attribution frames on literature-format debates (outputs/raw_lit): identical words, different attribution. HF direct read-out.

Two episode families, rows of one episode in one fixed-shape batch (8 rows):
first debate round (only agents with a disagreeing peer): part, part_first, part_para, observer, swap:j, swap_first:j;
final debate round (every agent): part, part_para, observer, priv_ctx, pub_ctx, priv_self, pub_self. Rows, each followed by the prefilled assistant text "{final answer: (":
  part         participant context exactly as in the debate (Choi et al. round message)
  part_para    same, with a meaning-preserving paraphrase of the round instruction (wording control)
  observer     identical three responses, all presented as agents' responses (none labelled as the reader's own)
  part_first   same words with the agent's own most recent opinion placed before the other agents' responses (position control)
  swap:j       (disagreement only) peer j's response in the "your most recent opinion" slot and the agent's own response in slot j
  swap_first:j the swap with the own-opinion slot placed first
  priv_ctx / pub_ctx    after the agent's final response: private / public re-query (as generated in debate_lit)
  priv_self / pub_self  transcript-free: initial question, the agent's final response, private / public re-query
Decision-token residuals are saved on the layer grid for every row.
Output: ACTS/<M>/lit_frames_<dataset>/{index.jsonl, acts.npy [rows, layers, d], meta.json}
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
from prompts_lit import N_AGENTS, PRIVATE_Q, PUBLIC_Q, answer_instruction, initial_prompt, messages, question_text
from debate_vllm import load_items

RAW_LIT = ROOT / "outputs/raw_lit"
PREFILL = "{final answer: ("


def para_prompt(v, own, others):
    s = "Here are the latest answers from the other agents: " + " ".join(f"Answer from one agent: {o.strip()}" for o in others)
    s += (f" Your own latest answer was: {own.strip()} Treat these answers as extra advice when updating your answer, and give your "
          "final answer to the question: " + question_text(v) + " " + answer_instruction(v))
    return s


def observer_prompt(v, responses):
    s = "These are the recent opinions from agents: " + " ".join(f"One of the agents' response: {o.strip()}" for o in responses)
    s += " Use these opinions carefully as additional advice to give your final answer to the question: " + question_text(v) + " " + answer_instruction(v)
    return s


def own_first_prompt(v, own, others):
    """Same words as the debate prompt with the agent's own most recent opinion placed before the other agents' responses."""
    s = f"This was your most recent opinion: {own.strip()} These are the recent opinions from other agents: "
    s += " ".join(f"One of the agents' response: {o.strip()}" for o in others)
    s += (" Use these opinions carefully as additional advice to revise your recent opinion to give your final answer to the question: "
          + question_text(v) + " " + answer_instruction(v))
    return s


def neutral_label_prompt(v, own, others):
    """Ownership control: the agent's own previous response is present but carries no ownership label."""
    s = "These are the recent opinions from agents: " + " ".join(f"One of the agents' response: {o.strip()}" for o in others + [own])
    s += (" Use these opinions carefully as additional advice to revise your recent opinion to give your final answer to the question: "
          + question_text(v) + " " + answer_instruction(v))
    return s


def persona_observer_prompt(v, responses):
    """Reader control: a named reviewer role, to separate 'not a participant' from 'a different persona'."""
    s = ("You are Agent D, a reviewer who has just joined. These are the recent opinions from the three agents: "
         + " ".join(f"One of the agents' response: {o.strip()}" for o in responses))
    s += " Use these opinions carefully as additional advice to give your final answer to the question: " + question_text(v) + " " + answer_instruction(v)
    return s


PUBLIC_ANON_Q = ("The discussion is now over. Your final answer is public: it will be shown to all the other agents and used for the "
                 "group decision, but no one will know which agent gave it. What do you yourself believe is the correct answer?")


def layer_grid(n):
    return sorted({int(round(f * (n - 1))) for f in np.linspace(0.1, 1.0, 10)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="core")
    ap.add_argument("--transcripts", default=None)
    ap.add_argument("--round", type=int, default=0, help="debate round to frame (default: last)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--extra", action="store_true", help="only the supplementary control arms (neutral label, reviewer persona, "
                                                          "public-but-unattributed)")
    a = ap.parse_args()
    R = Receiver(a.model)
    grid = layer_grid(R.n_layers)
    tm = a.transcripts or a.model
    d = RAW_LIT / tm / a.dataset
    meta = json.load(open(d / "meta.json"))
    K = a.round or meta["rounds"]
    recs = {}
    for f in d.glob("*.jsonl"):
        for x in read_jsonl(f):
            recs[(x["item_id"], x["cond"], x["agent"], x["round"], x.get("query"))] = x
    core = {r["id"]: r for r in load_core()}
    views = {i: item_view(core[i]) for i in sorted({k[0] for k in recs})}
    views = {i: v for i, v in views.items() if v["kind"] != "numeric"}
    if a.limit:
        views = dict(list(views.items())[: a.limit])
    episodes = []

    def ctx(i, k):
        pc = "r0" if k == 1 else "debate"
        prev = [recs.get((i, pc, g, k - 1, None)) for g in range(N_AGENTS)]
        cur = [recs.get((i, "debate", g, k, None)) for g in range(N_AGENTS)]
        return (None, None) if any(x is None for x in prev + cur) else (prev, cur)

    for i, v in views.items():
        sysm = [{"role": "system", "content": v["system_extra"].strip()}] if v.get("system_extra") else []
        # (1) first debate round, agents with at least one disagreeing peer: ownership swaps with position control + readers
        prev, cur = ctx(i, 1)
        if prev is not None:
            texts, answers = [x["text"] for x in prev], [x["answer"] for x in prev]
            for g in range(N_AGENTS):
                oi = [h for h in range(N_AGENTS) if h != g]
                dis = [j for j in oi if answers[j] is not None and answers[g] is not None and answers[j] != answers[g]]
                if not dis:
                    continue
                rows = [("part", messages(v, 1, texts[g], [texts[h] for h in oi], "debate")),
                        ("part_first", sysm + [{"role": "user", "content": own_first_prompt(v, texts[g], [texts[h] for h in oi])}]),
                        ("part_para", sysm + [{"role": "user", "content": para_prompt(v, texts[g], [texts[h] for h in oi])}]),
                        ("observer", sysm + [{"role": "user", "content": observer_prompt(v, [texts[h] for h in oi] + [texts[g]])}])]
                for j in dis:
                    swapped = [texts[g] if h == j else texts[h] for h in oi]
                    rows.append((f"swap:{j}", messages(v, 1, texts[j], swapped, "debate")))
                    rows.append((f"swap_first:{j}", sysm + [{"role": "user", "content": own_first_prompt(v, texts[j], swapped)}]))
                if a.extra:
                    # observer_persona belongs in THIS family too: the role claim is measured on round-1 outnumbered
                    # episodes, so a second observer wording is only useful if it is collected here. With part/part_para
                    # from the main run and observer/observer_persona, the role effect becomes a 2x2 whose between-role
                    # gap can be read against the within-role wording spread.
                    rows = [("part", messages(v, 1, texts[g], [texts[h] for h in oi], "debate")),
                            ("neutral_label", sysm + [{"role": "user", "content": neutral_label_prompt(v, texts[g], [texts[h] for h in oi])}]),
                            ("observer_persona", sysm + [{"role": "user", "content": persona_observer_prompt(v, [texts[h] for h in oi] + [texts[g]])}])]
                episodes.append((dict(item_id=i, agent=g, round=1, prev_answers=answers, final_answer=cur[g]["answer"]), rows))
        # (2) final debate round, every agent: readers + audience
        prev, cur = ctx(i, K)
        if prev is None:
            continue
        texts, answers = [x["text"] for x in prev], [x["answer"] for x in prev]
        for g in range(N_AGENTS):
            oi = [h for h in range(N_AGENTS) if h != g]
            rows = [("part", messages(v, K, texts[g], [texts[h] for h in oi], "debate")),
                    ("part_para", sysm + [{"role": "user", "content": para_prompt(v, texts[g], [texts[h] for h in oi])}]),
                    ("observer", sysm + [{"role": "user", "content": observer_prompt(v, [texts[h] for h in oi] + [texts[g]])}])]
            final = cur[g]["text"]
            base = messages(v, K, texts[g], [texts[h] for h in oi], "debate") + [{"role": "assistant", "content": final}]
            selfb = sysm + [{"role": "user", "content": initial_prompt(v)}, {"role": "assistant", "content": final}]
            if a.extra:
                rows = [("part", messages(v, K, texts[g], [texts[h] for h in oi], "debate")),
                        ("observer_persona", sysm + [{"role": "user", "content": persona_observer_prompt(v, [texts[h] for h in oi] + [texts[g]])}]),
                        ("pub_anon_ctx", base + [{"role": "user", "content": PUBLIC_ANON_Q + " " + answer_instruction(v)}])]
            else:
                for q, qt in (("priv", PRIVATE_Q), ("pub", PUBLIC_Q)):
                    rows.append((f"{q}_ctx", base + [{"role": "user", "content": qt + " " + answer_instruction(v)}]))
                    rows.append((f"{q}_self", selfb + [{"role": "user", "content": qt + " " + answer_instruction(v)}]))
            episodes.append((dict(item_id=i, agent=g, round=K, prev_answers=answers, final_answer=cur[g]["answer"]), rows))
    print(f"[stage] {len(episodes)} episodes, {sum(len(r) for _, r in episodes)} rows", flush=True)
    ROWS = 3 if a.extra else 8
    out = ACTS / a.model / f"lit_frames{'_extra' if a.extra else ''}_{a.dataset}{'_' + tm + 'text' if a.transcripts else ''}"
    out.mkdir(parents=True, exist_ok=True)
    N = sum(len(r) for _, r in episodes)
    dmod = R.model.config.get_text_config().hidden_size
    acts = np.lib.format.open_memmap(out / "acts.npy", mode="w+", dtype=np.float16, shape=(N, len(grid), dmod))
    index, n, t0 = [], 0, time.time()
    for ei, (key, rows) in enumerate(episodes):
        v = views[key["item_id"]]
        ids = []
        for name, msgs in rows:
            prefix = R.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            t = R.tok(prefix + PREFILL, add_special_tokens=False)["input_ids"]
            ids.append(t[-7000:])
        batch = ids + [ids[-1]] * (ROWS - len(ids))
        store = {}
        for li in grid:
            R.add_capture(li, [len(x) - 1 for x in batch], store)
        lp = R.forward(batch)
        R.clear()
        for bi, (name, _) in enumerate(rows):
            index.append(dict(key, frame=name, row=n, n_tok=len(ids[bi]), lp_letters=R.letter_lp(lp[bi], v["labels"])))
            for gi, li in enumerate(grid):
                acts[n, gi] = store[li][bi].numpy().astype(np.float16)
            n += 1
        if (ei + 1) % 200 == 0:
            el = time.time() - t0
            print(f"[progress] {ei+1}/{len(episodes)} episodes, eta {el/(ei+1)*(len(episodes)-ei-1)/60:.1f} min", flush=True)
    acts.flush()
    write_jsonl(out / "index.jsonl", index)
    json.dump(dict(model=a.model, transcripts=tm, sha=R.sha, grid=grid, round=K, prefill=PREFILL), open(out / "meta.json", "w"), indent=1)
    print(f"[done] lit frames {a.model}: {n} rows in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
