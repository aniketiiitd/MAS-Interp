"""Attribution counterfactuals on byte-identical debate text (HF forward, prefilled "VERDICT:"), V3-C2 and V3-C4(i).

--family own  (whose words): debate agent-turns t = 1..T-1 whose three t-1 public verdicts are not unanimous. One fixed-shape
    batch per episode (rows padded by repetition), so every contrast is within one forward pass:
      orig      the participant prompt exactly as generated
      swap:j    for each peer j whose t-1 verdict differs from the agent's: the agent's own t-1 text is shown under "[Agent j]"
                and peer j's text under "Your previous response" (the same three texts and the same headers)
      order     peer blocks in reversed order (labels travel with their texts)
      names     peer blocks keep their positions, the two peer names are exchanged
--family audience (who will hear): after the final turn, debate and pool conditions, every agent:
      priv_ctx  [system, final-round prompt, assistant(own final text), PRIVATE_REQUERY]   (= the E8 private re-query prompt)
      pub_ctx   same with PUBLIC_REQUERY (token-matched under the Qwen3.5 and Gemma-4 tokenizers)
      priv_self transcript-free: [system, turn-0 task prompt, assistant(own final text), PRIVATE_REQUERY]
      pub_self  transcript-free with PUBLIC_REQUERY
    Rows of one (item, cond, agent) episode always share a batch (3 episodes x 4 frames = 12 rows).
Saved: index.jsonl (one row per real prompt: keys, frame, lp_letters, n_tok, row) and acts.npy [rows, sites, layers, d] float16
with sites = (decision token, reading site = last token of the block before the closing instruction / query).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time

import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "interp"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runner"))
from common import *  # noqa
from cpt import Receiver
from prompts import AGENT_NAMES, PRIVATE_REQUERY, round_user, system_prompt, t0_user
from run_cpt import layer_grid

PUBLIC_REQUERY = ("The deliberation is now over. This final answer is public: it will be shown to all the other agents and used to "
                  "reach the group decision. Considering everything, what do you yourself believe is correct? Reply in the same "
                  "required format.")
NAME_RE = re.compile(r"Agent\s*[1-6]")


def own_frames(v, a, t, hist, verdicts):
    """Returns list of (frame, messages, meta) for one own-family episode."""
    sysm = system_prompt(v, a)
    prev = hist[t - 1]
    peers = [j for j in range(3) if j != a]

    def msgs(own_text, others):
        return [{"role": "system", "content": sysm}, {"role": "user", "content": round_user(v, a, t, "debate", own_text, others)}]

    out = [("orig", msgs(prev[a], [(j, prev[j]) for j in peers]), dict(peer=None))]
    for j in peers:
        if verdicts[j] != verdicts[a]:
            others = [(k, prev[a] if k == j else prev[k]) for k in peers]
            out.append((f"swap:{j}", msgs(prev[j], others), dict(peer=j)))
    out.append(("order", msgs(prev[a], [(j, prev[j]) for j in reversed(peers)]), dict(peer=None)))
    out.append(("names", msgs(prev[a], [(peers[1], prev[peers[0]]), (peers[0], prev[peers[1]])]), dict(peer=None)))
    return out


def audience_frames(v, a, cond, hist, last):
    sysm = system_prompt(v, a)
    own_final = hist[last][a]
    if last == 0:
        ctx_user = t0_user(v)
    else:
        others = [(j, hist[last - 1][j]) for j in range(3) if j != a]
        ctx_user = round_user(v, a, last, cond, hist[last - 1][a], others)
    out = []
    for frame, user0, q in (("priv_ctx", ctx_user, PRIVATE_REQUERY), ("pub_ctx", ctx_user, PUBLIC_REQUERY),
                            ("priv_self", t0_user(v), PRIVATE_REQUERY), ("pub_self", t0_user(v), PUBLIC_REQUERY)):
        out.append((frame, [{"role": "system", "content": sysm}, {"role": "user", "content": user0},
                            {"role": "assistant", "content": own_final}, {"role": "user", "content": q}], {}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="release_600")
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--family", required=True, choices=["own", "audience"])
    ap.add_argument("--items", default=None, help="restrict to a split name from splits.json (e.g. release_600) or 'not:release_600'")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--transcripts", default=None, help="model whose transcripts are read (default: --model); cross-reader runs")
    a = ap.parse_args()
    R = Receiver(a.model)
    grid = layer_grid(R.n_layers)
    d = RAW / (a.transcripts or a.model) / a.dataset
    core = {r["id"]: r for r in load_core()}
    recs = {}
    conds = ["debate"] if a.family == "own" else ["debate", "pool"]
    for c in conds:
        for f in ["t0.jsonl"] + [f"{c}_t{t}.jsonl" for t in range(1, a.turns)]:
            for x in read_jsonl(d / f):
                if x["agent"] < 3:
                    recs[(x["item_id"], c, x["turn"], x["agent"])] = x
    items = sorted({k[0] for k in recs})
    if a.items:
        splits = json.load(open(ROOT / "data/splits/splits.json"))
        neg = a.items.startswith("not:")
        S = set(splits[a.items[4:] if neg else a.items])
        items = [i for i in items if (i not in S) == neg]
    if a.limit:
        items = items[: a.limit]
    last = a.turns - 1
    # ---- build episodes ----
    episodes = []  # list of (keys, [(frame, ids, [dec_pos, read_pos], meta)])
    for iid in items:
        v = item_view(core[iid])
        if v["kind"] == "deception" or not v["labels"]:  # the letter endpoint needs a VERDICT-first format with option letters
            continue
        prefill = "VERDICT:"
        if a.family == "own":
            for t in range(1, a.turns):
                prev = [recs.get((iid, "debate", t - 1, b)) for b in range(3)]
                if any(p is None or p["verdict"] is None for p in prev):
                    continue
                verdicts = [p["verdict"] for p in prev]
                if len(set(verdicts)) == 1:
                    continue
                hist = {t - 1: {b: prev[b]["text"] for b in range(3)}}
                for ag in range(3):
                    fr = own_frames(v, ag, t, hist, verdicts)
                    if len(fr) <= 3:  # no peer disagrees with this agent
                        continue
                    episodes.append((dict(item_id=iid, cond="debate", turn=t, agent=ag, own=verdicts[ag],
                                          peers={str(j): verdicts[j] for j in range(3) if j != ag},
                                          name_mention=any(bool(NAME_RE.search(hist[t - 1][b])) for b in range(3))),
                                     fr, prefill, "\n\nConsidering these responses"))
        else:
            for c in ("debate", "pool"):
                if any((iid, c, tt, b) not in recs for tt in {max(last - 1, 0), last} for b in range(3)):
                    continue
                hist = {tt: {b: recs[(iid, c, tt, b)]["text"] for b in range(3)} for tt in {max(last - 1, 0), last}}
                pub = {b: recs[(iid, c, last, b)]["verdict"] for b in range(3)}
                for ag in range(3):
                    episodes.append((dict(item_id=iid, cond=c, turn=last + 1, agent=ag, public=pub[ag],
                                          unanimous=len(set(pub.values())) == 1 and None not in pub.values()),
                                     audience_frames(v, ag, c, hist, last), prefill, None))
    print(f"[stage] {len(episodes)} episodes ({a.family}, {a.dataset}, {len(items)} items)", flush=True)
    # ---- tokenise ----
    built = []
    for keys, frames, prefill, read_marker in episodes:
        rows = []
        for frame, msgs, meta in frames:
            prefix = R.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            enc = R.tok(prefix + prefill, add_special_tokens=False, return_offsets_mapping=True)
            if read_marker is not None:
                cut = prefix.rfind(read_marker)
            else:
                cut = prefix.rfind(msgs[-1]["content"])  # reading site = last token before the query text
            read_pos = max(i for i, (s, e) in enumerate(enc["offset_mapping"]) if e <= cut)
            rows.append((frame, enc["input_ids"], [len(enc["input_ids"]) - 1, read_pos], meta))
        built.append((keys, rows))
    per_batch = 5 if a.family == "own" else 12
    out_dir = ACTS / a.model / (f"frames_{a.family}_{a.dataset}{'_' + a.items.replace(':', '') if a.items else ''}"
                                f"{'_' + a.transcripts + 'text' if a.transcripts else ''}{'_limit' + str(a.limit) if a.limit else ''}")
    out_dir.mkdir(parents=True, exist_ok=True)
    N = sum(len(r) for _, r in built)
    acts = np.lib.format.open_memmap(out_dir / "acts.npy", mode="w+", dtype=np.float16,
                                     shape=(N, 2, len(grid), R.model.config.get_text_config().hidden_size))
    index, n, t0 = [], 0, time.time()
    # group episodes into fixed-shape batches: own = one episode per batch (<= 5 rows); audience = 3 episodes (12 rows)
    groups = [[e] for e in built] if a.family == "own" else [built[i:i + 3] for i in range(0, len(built), 3)]
    for gi_, grp in enumerate(groups):
        rows = [(keys, r) for keys, rs in grp for r in rs]
        batch = rows + [rows[-1]] * (per_batch - len(rows))
        store = {}
        for li in grid:
            R.add_capture(li, [r[2] for _, r in batch], store)
        lp = R.forward([r[1] for _, r in batch])
        R.clear()
        for bi, (keys, (frame, ids, pos, meta)) in enumerate(rows):
            labels = item_view(core[keys["item_id"]])["labels"] or []
            index.append(dict(keys, frame=frame, **meta, n_tok=len(ids), row=n,
                              lp_letters=R.letter_lp(lp[bi], labels) if labels else None))
            for gl, li in enumerate(grid):
                acts[n, :, gl] = store[li][bi].numpy().astype(np.float16)
            n += 1
        if (gi_ + 1) % 200 == 0:
            el = time.time() - t0
            print(f"[progress] {gi_+1}/{len(groups)} batches, eta {el/(gi_+1)*(len(groups)-gi_-1)/60:.1f} min", flush=True)
    acts.flush()
    write_jsonl(out_dir / "index.jsonl", index)
    json.dump(dict(model=a.model, sha=R.sha, grid=grid, n_rows=n, family=a.family, dataset=a.dataset, turns=a.turns,
                   sites=["decision", "reading"], public_requery=PUBLIC_REQUERY, private_requery=PRIVATE_REQUERY),
              open(out_dir / "meta.json", "w"), indent=1)
    print(f"[done] frames {a.family} {a.model} {a.dataset}: {n} rows in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
