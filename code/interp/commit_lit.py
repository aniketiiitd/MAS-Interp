"""When is the answer decided, and what carries the influence? (literature-format debates, HF direct read-out)

--mode commit : for debate and no-communication responses in rounds 1..R (every answer change + a hash sample of unchanged answers),
                the written reasoning before the final answer is truncated at fractions {0, .1, .25, .5, .75, 1} and the model is made
                to answer immediately ("... {final answer: (") -> letter distribution at each fraction. fraction 0 = right after
                reading the round message, before writing anything. All fractions of one response share one forward batch.
--mode filler : the controls the truncation result needs. For the same responses as --mode commit, the reasoning is replaced by
                (a) neutral filler text of matched token length ("Let me work through this carefully." repeated), and
                (b) the same agent's reasoning from a different item of the same family (cross-item prefix), before the forced
                answer. If the peers' answer only appears when the agent's own reasoning is absent, these arms show it.
--mode carrier: first debate round, agents with a disagreeing peer. Rows (one batch): full (peers' responses as written),
                answer_only (each peer response replaced by its bare final answer), no_peers (the no-communication message), and
                answer_only_masked (peers' final answers hidden, i.e. the reasoning without the answer line).
Output: ACTS/<M>/lit_commit_<dataset>/rows.jsonl or ACTS/<M>/lit_carrier_<dataset>/rows.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "interp"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runner"))
from common import *  # noqa
from cpt import Receiver
from prompts_lit import N_AGENTS, answer_span, messages

CORE = {r["id"]: r for r in load_core()}

RAW_LIT = ROOT / "outputs/raw_lit"
FRACS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
PREFILL = "{final answer: ("
FINAL_RE = re.compile(r"\{?\s*final answer\s*[:：][^}\n]*\}?", re.I)


def load(tm, dataset):
    d = RAW_LIT / tm / dataset
    recs = {}
    for f in d.glob("*.jsonl"):
        for x in read_jsonl(f):
            recs[(x["item_id"], x["cond"], x["agent"], x["round"], x.get("query"))] = x
    return recs, json.load(open(d / "meta.json"))["rounds"]


def cut_words(s, frac):
    if frac >= 1.0:
        return s
    n = int(len(s) * frac)
    j = s.rfind(" ", 0, n)
    return s[: max(j, 0)]


def mode_commit(a, R, recs, views, tok):
    jobs = []
    for (i, cond, g, k, q), x in recs.items():
        if q or k == 0 or cond not in ("debate", "reflect") or i not in views or x["answer"] is None:
            continue
        prev = recs.get((i, "r0" if k == 1 else cond, g, k - 1, None))
        if prev is None or prev["answer"] is None:
            continue
        changed = prev["answer"] != x["answer"]
        if not changed and h64("commit_sample", i, cond, g, k) % 100 >= a.stay_pct:
            continue
        v = views[i]
        span = answer_span(x["text"], v)
        if span is None:
            continue
        own = recs[(i, "r0" if k == 1 else cond, g, k - 1, None)]["text"]
        others = [recs[(i, "r0" if k == 1 else cond, h, k - 1, None)]["text"] for h in range(N_AGENTS) if h != g]
        msgs = messages(v, k, own, others, cond)
        prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        reasoning = x["text"][: span[0]].rstrip()
        rows = []
        for f in FRACS:
            body = cut_words(reasoning, f)
            text = prefix + (body + "\n\n" if body else "") + PREFILL
            rows.append((f, tok(text, add_special_tokens=False)["input_ids"][-8000:]))
        jobs.append((dict(item_id=i, cond=cond, agent=g, round=k, prev_answer=prev["answer"], answer=x["answer"], changed=changed,
                          reasoning_chars=len(reasoning)), rows))
    return jobs


FILLER_UNIT = "Let me work through this carefully. "


def mode_filler(a, R, recs, views, tok):
    jobs = mode_commit(a, R, recs, views, tok)  # same episode selection
    unit_len = len(tok(FILLER_UNIT, add_special_tokens=False)["input_ids"])
    by_family = {}
    for key, _ in jobs:
        by_family.setdefault(family_of(CORE[key["item_id"]]), []).append(key)
    out = []
    for key, rows in jobs:
        i, cond, g, k = key["item_id"], key["cond"], key["agent"], key["round"]
        v = views[i]
        own = recs[(i, "r0" if k == 1 else cond, g, k - 1, None)]["text"]
        others = [recs[(i, "r0" if k == 1 else cond, h, k - 1, None)]["text"] for h in range(N_AGENTS) if h != g]
        prefix = tok.apply_chat_template(messages(v, k, own, others, cond), tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        n_reason = max(1, len(tok(recs[(i, cond, g, k, None)]["text"], add_special_tokens=False)["input_ids"]))
        filler = FILLER_UNIT * max(1, round(n_reason / unit_len))
        sib = [x for x in by_family[family_of(CORE[i])] if x["item_id"] != i]
        donor = sib[h64("donor", i, g, k) % len(sib)] if sib else None
        rows_out = [("filler", tok(prefix + filler.strip() + "\n\n" + PREFILL, add_special_tokens=False)["input_ids"][-8000:])]
        if donor is not None:
            dtext = recs[(donor["item_id"], donor["cond"], donor["agent"], donor["round"], None)]["text"]
            dspan = answer_span(dtext, views[donor["item_id"]])
            dreason = dtext[: dspan[0]].rstrip() if dspan else dtext
            rows_out.append(("cross_item", tok(prefix + dreason + "\n\n" + PREFILL, add_special_tokens=False)["input_ids"][-8000:]))
        out.append((dict(key, donor_item=donor["item_id"] if donor else None), rows_out))
    return out


def mode_carrier(a, R, recs, views, tok):
    jobs = []
    for i, v in views.items():
        prev = [recs.get((i, "r0", g, 0, None)) for g in range(N_AGENTS)]
        if any(p is None for p in prev):
            continue
        texts, answers = [p["text"] for p in prev], [p["answer"] for p in prev]
        for g in range(N_AGENTS):
            oi = [h for h in range(N_AGENTS) if h != g]
            if answers[g] is None or not any(answers[h] not in (None, answers[g]) for h in oi):
                continue
            bare = [f"{{final answer: ({answers[h]})}}" if answers[h] else texts[h] for h in oi]
            masked = [FINAL_RE.sub("", texts[h]).rstrip() for h in oi]
            frames = [("full", messages(v, 1, texts[g], [texts[h] for h in oi], "debate")),
                      ("answer_only", messages(v, 1, texts[g], bare, "debate")),
                      ("reasoning_no_answer_line", messages(v, 1, texts[g], masked, "debate")),
                      ("no_peers", messages(v, 1, texts[g], None, "reflect"))]
            rows = []
            for name, msgs in frames:
                prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                rows.append((name, tok(prefix + PREFILL, add_special_tokens=False)["input_ids"][-8000:]))
            jobs.append((dict(item_id=i, agent=g, own=answers[g], peers={str(h): answers[h] for h in oi}), rows))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="lit1200")
    ap.add_argument("--transcripts", default=None)
    ap.add_argument("--mode", required=True, choices=["commit", "carrier", "filler"])
    ap.add_argument("--stay_pct", type=int, default=6, help="percent of unchanged answers sampled (commit mode)")
    a = ap.parse_args()
    R_ = Receiver(a.model)
    tm = a.transcripts or a.model
    recs, R = load(tm, a.dataset)
    core = {r["id"]: r for r in load_core()}
    views = {i: item_view(core[i]) for i in sorted({k[0] for k in recs})}
    views = {i: v for i, v in views.items() if v["kind"] != "numeric"}
    jobs = dict(commit=mode_commit, carrier=mode_carrier, filler=mode_filler)[a.mode](a, R, recs, views, R_.tok)
    print(f"[stage] {len(jobs)} {a.mode} episodes", flush=True)
    out = ACTS / a.model / f"lit_{a.mode}_{a.dataset}{'_' + tm + 'text' if a.transcripts else ''}"
    out.mkdir(parents=True, exist_ok=True)
    recs_out, t0 = [], time.time()
    for ji, (key, rows) in enumerate(jobs):
        labels = views[key["item_id"]]["labels"]
        lp = R_.forward([ids for _, ids in rows])
        for bi, (name, ids) in enumerate(rows):
            recs_out.append(dict(key, frame=name, n_tok=len(ids), lp_letters=R_.letter_lp(lp[bi], labels)))
        if (ji + 1) % 200 == 0:
            el = time.time() - t0
            print(f"[progress] {ji+1}/{len(jobs)}, eta {el/(ji+1)*(len(jobs)-ji-1)/60:.1f} min", flush=True)
    write_jsonl(out / "rows.jsonl", recs_out)
    json.dump(dict(model=a.model, transcripts=tm, mode=a.mode, fracs=FRACS, prefill=PREFILL), open(out / "meta.json", "w"), indent=1)
    print(f"[done] lit {a.mode} {a.model}: {len(jobs)} episodes in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
