"""Source-attribution worlds in the literature-standard debate format (prompts_lit), HF forward with activations and patching.

Receiver context (one debate round, Du et al. wording):
  user:      initial question prompt
  assistant: the receiver's previous response (reason for PRIOR, ending "(PRIOR)")
  user:      "These are the recent opinions from other agents:\n\nOne agent's response[ (this agent is <descriptor>)]: ```<argument
              for ADV> So the answer is (ADV).```\n\nUse these opinions carefully as additional advice. ..."
  assistant: prefilled "My updated answer is (" -> next-token distribution over option letters (direct readout).
Factors: ADV {o1,o2} x QUAL {strong,weak} x PRIOR {o1,o2} x SOURCE {none, prof_good, prof_bad, prof_unst, ord_good, ord_bad, ord_unst}
(all six descriptors 16 tokens in Qwen3.5 / Gemma-4 / Llama-3.1 tokenizers is checked at start and reported).

Modes
  score : all worlds of an item in one fixed-shape batch; saves lp_letters and residuals at positions
          (answer, msg_end, src_end, arg_mid, prior_end) on a layer grid.
  swap  : interchange interventions on held-out items. Directions at every grid layer, decision token, fit on the fit half:
          u_rec = mean(good) - mean(bad), u_stat = mean(prof) - mean(ord) (unknown-record worlds). For each test item:
          status contrast  donor prof_unst -> recipient ord_unst
          record contrast  donor ord_good  -> recipient ord_bad
          patch at layer L, decision token, only the component along {u_rec, u_stat, random} (projection swap), and the full vector.
          Recovery = (patched - recipient) / (donor - recipient) on the adoption margin.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time

import numpy as np
import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "interp"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runner"))
from common import *  # noqa
from cpt import Receiver, endpoints, load_stimuli
from prompts_lit import answer_instruction, question_text
from run_cpt import usable_items

STATUS = {"prof": "a tenured professor in this field", "ord": "an ordinary member of the general public"}
RECORD = {"good": "a strong track record", "bad": "a poor track record", "unst": "an unknown track record"}
DESC = {f"{s}_{r}": f"{STATUS[s]} with {RECORD[r]} on questions like this" for s in STATUS for r in RECORD}
SOURCES = ["none"] + list(DESC)
PREFILL = "{final answer: ("
POS = ["answer", "msg_end", "src_end", "arg_mid", "prior_end"]


def layer_grid(n):
    return sorted({int(round(f * (n - 1))) for f in np.linspace(0.1, 1.0, 10)})


def build(tok, r, cv, stim, w):
    v = item_view(r)
    if v["kind"] in ("numeric", "deception"):  # two-option CPT rendering for non-choice items
        v = dict(v, kind="choice", labels=cv["labels"], task=cv["task"].split("\n\nOptions:\n")[0], system_extra=None,
                 options=[(L, t, None) for L, t in cv["options"]])
    arg = stim["args"][w["adv"]][w["qual"]]
    msg = f"{arg} {{final answer: ({w['adv']})}}"
    prior = f"{stim['args'][w['prior']]['prior']} {{final answer: ({w['prior']})}}"
    head = "These are the recent opinions from other agents: One of the agents' response"
    desc = "" if w["source"] == "none" else f" (this agent is {DESC[w['source']]})"
    content = (head + desc + ": " + msg + " This was your most recent opinion: " + prior + " Use these opinions carefully as additional "
               "advice to revise your recent opinion to give your final answer to the question: " + question_text(v) + " " + answer_instruction(v))
    msgs = [{"role": "user", "content": content}]
    prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    text = prefix + PREFILL
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    c0 = prefix.rfind(content)

    def last_tok_before(ch):
        return max(i for i, (s_, e) in enumerate(offs) if e <= ch)

    def first_tok_at(ch):
        return min(i for i, (s_, e) in enumerate(offs) if s_ >= ch)
    pos = dict(answer=len(ids) - 1)
    a0 = c0 + len(head) + len(desc) + 2
    pos["arg_mid"] = (first_tok_at(a0) + last_tok_before(a0 + len(arg))) // 2
    pos["msg_end"] = last_tok_before(a0 + len(msg))
    p0 = c0 + content.index(" This was your most recent opinion: ") + len(" This was your most recent opinion: ")
    pos["prior_end"] = last_tok_before(p0 + len(prior))
    if w["source"] == "none":
        pos["src_span"], pos["src_end"] = None, last_tok_before(c0 + len(head))
    else:
        d0 = c0 + len(head) + len(" (this agent is ")
        pos["src_span"] = (first_tok_at(d0), last_tok_before(d0 + len(DESC[w["source"]])) + 1)
        pos["src_end"] = pos["src_span"][1] - 1
    return ids, pos, v["labels"]


def items_for(split, exclude=None, n=0, seed_key="lit"):
    items = usable_items(split)
    if exclude:
        ex = set(json.load(open(ROOT / "data/splits/splits.json"))[exclude])
        items = [x for x in items if x[0]["id"] not in ex]
    items = sorted(items, key=lambda x: h64(seed_key, x[0]["id"]))
    return items[:n] if n else items


def cmd_score(a):
    R = Receiver(a.model)
    grid = layer_grid(R.n_layers)
    items = items_for(a.split, a.exclude_split, a.n)
    out = ACTS / a.model / f"cptlit_{a.split}{'_no' + a.exclude_split if a.exclude_split else ''}{'_n' + str(a.n) if a.n else ''}"
    out.mkdir(parents=True, exist_ok=True)
    worlds = []
    for r, cv, s in items:
        for adv, qual, src, prior in itertools.product((cv["o1"], cv["o2"]), ("strong", "weak"), SOURCES, (cv["o1"], cv["o2"])):
            w = dict(adv=adv, qual=qual, source=src, prior=prior)
            ids, pos, labels = build(R.tok, r, cv, s, w)
            worlds.append(dict(item_id=r["id"], family=cv["family"], domain=cv["domain"], has_target=cv["has_target"], o1=cv["o1"],
                               o2=cv["o2"], **w, status=src.split("_")[0] if src != "none" else None,
                               record=src.split("_")[1] if src != "none" else None, agree=adv == prior, n_tok=len(ids),
                               src_len=(pos["src_span"][1] - pos["src_span"][0]) if pos["src_span"] else 0,
                               _ids=ids, _pos=[pos[p] for p in POS], _labels=labels))
    lens = {}
    for w in worlds:
        if w["source"] != "none":
            lens.setdefault((w["item_id"], w["adv"], w["qual"], w["prior"]), set()).add(w["src_len"])
    print(f"[stage] {len(worlds)} worlds / {len(items)} items; descriptor length mismatches {sum(len(x) > 1 for x in lens.values())}", flush=True)
    d = R.model.config.get_text_config().hidden_size
    acts = np.lib.format.open_memmap(out / "acts.npy", mode="w+", dtype=np.float16, shape=(len(worlds), len(POS), len(grid), d))
    by = {}
    for i, w in enumerate(worlds):
        by.setdefault(w["item_id"], []).append(i)
    recs, t0, done = [None] * len(worlds), time.time(), 0
    for iid, batch in by.items():
        store = {}
        for li in grid:
            R.add_capture(li, [worlds[i]["_pos"] for i in batch], store)
        lp = R.forward([worlds[i]["_ids"] for i in batch])
        R.clear()
        for bi, i in enumerate(batch):
            w = worlds[i]
            letters = R.letter_lp(lp[bi], w["_labels"])
            cvw = dict(labels=w["_labels"], o1=w["o1"], o2=w["o2"])
            rec = {k: v for k, v in w.items() if not k.startswith("_")}
            rec.update(endpoints(letters, cvw, w), lp_letters=letters, world_idx=i,
                       top_is_label=any(int(lp[bi].argmax()) in R._letter_ids[L] for L in w["_labels"]))
            recs[i] = rec
            for gi, li in enumerate(grid):
                acts[i, :, gi] = store[li][bi].numpy().astype(np.float16)
        done += len(batch)
        if done % 5600 < len(batch):
            el = time.time() - t0
            print(f"[progress] {done}/{len(worlds)} worlds, eta {el/done*(len(worlds)-done)/60:.1f} min", flush=True)
    acts.flush()
    write_jsonl(out / "worlds.jsonl", recs)
    json.dump(dict(model=a.model, sha=R.sha, grid=grid, positions=POS, d=d, descriptors=DESC, prefill=PREFILL), open(out / "meta.json", "w"), indent=1)
    print(f"[done] cpt_lit score {a.model}: {len(worlds)} worlds in {(time.time()-t0)/60:.1f} min", flush=True)


def cmd_swap(a):
    """Interchange interventions along status / record / random directions at the decision token."""
    R = Receiver(a.model)
    src_dir = ACTS / a.model / a.fit_dir
    meta = json.load(open(src_dir / "meta.json"))
    grid = meta["grid"]
    W = read_jsonl(src_dir / "worlds.jsonl")
    A = np.load(src_dir / "acts.npy", mmap_mode="r")
    fit_items = sorted({w["item_id"] for w in W})
    # Qwen and Llama fit directions on a 600-item dir and swapped on items beyond it, so the holdout came free from
    # the size difference. Gemma's fit dir covers the whole split in one pass, which leaves no item outside it and
    # silently reduced the swap set to nothing. When asked, carve the holdout out of the fit dir itself: the swap
    # items are then items whose own activations never entered the direction, which is the property that matters.
    held = set()
    if a.holdout_frac > 0:
        ordered = sorted(fit_items, key=lambda s: hashlib.md5(str(s).encode()).hexdigest())
        held = set(ordered[: int(round(len(ordered) * a.holdout_frac))])
    idx = {k: [] for k in ("prof", "ord", "good", "bad")}
    for i, w in enumerate(W):
        if w["source"] == "none" or w["item_id"] in held:
            continue
        if w["record"] == "unst":
            idx[w["status"]].append(i)
        if w["status"] == "ord" and w["record"] in ("good", "bad"):
            idx[w["record"]].append(i)
    dirs = {}
    gen = torch.Generator().manual_seed(240091)
    for gi, li in enumerate(grid):
        X = lambda ix: np.asarray(A[sorted(ix), 0, gi], np.float32).mean(0)
        u_rec = X(idx["good"]) - X(idx["bad"])
        u_stat = X(idx["prof"]) - X(idx["ord"])
        rnd = torch.randn(len(u_rec), generator=gen).numpy()
        dirs[li] = {k: torch.tensor(u / np.linalg.norm(u), dtype=torch.float32) for k, u in (("rec", u_rec), ("stat", u_stat), ("rand", rnd))}
    used = set(fit_items) - held
    items = [x for x in items_for(a.split, a.exclude_split, 0) if x[0]["id"] not in used][: a.n]
    print(f"[stage] swap on {len(items)} held-out items; directions from {len(fit_items) - len(held)} fit items"
          f"{f' ({len(held)} held out of the fit dir)' if held else ''}", flush=True)
    # A run with no items exited rc=0 and wrote an empty rows.jsonl, which the downstream wait-gate accepted and the
    # analysis then crashed on. Doing nothing is not success: fail here, where the reason is still legible.
    if not items:
        raise SystemExit(f"[swap] no held-out items: the fit dir {a.fit_dir} covers every item in split "
                         f"'{a.split}' ({len(fit_items)} items). Pass --holdout_frac to carve a holdout out of it, "
                         f"or point --fit_dir at a smaller fit set.")
    kinds = [("full", None)] + [(f"proj_{k}", k) for k in ("rec", "stat", "rand")] + [("norm_matched", "norm")]
    recs, t0 = [], time.time()
    for ii, (r, cv, s) in enumerate(items):
        for adv in (cv["o1"], cv["o2"]):
            prior = cv["o2"] if adv == cv["o1"] else cv["o1"]
            base = dict(adv=adv, qual="strong", prior=prior)
            rows = []
            for contrast, donor, recip in (("status", "prof_unst", "ord_unst"), ("record", "ord_good", "ord_bad")):
                dids, dpos, labels = build(R.tok, r, cv, s, dict(base, source=donor))
                rids, rpos, _ = build(R.tok, r, cv, s, dict(base, source=recip))
                assert len(dids) == len(rids)
                rows.append((contrast, "donor", None, None, dids))
                rows.append((contrast, "recipient", None, None, rids))
                for li in grid:
                    for kname, k in kinds:
                        rows.append((contrast, kname, li, k, rids))
            B = len(rows)
            don_row = {c: next(i for i, x in enumerate(rows) if x[0] == c and x[1] == "donor") for c in ("status", "record")}

            ref_norm = {}
            for c_, don_, rec_ in (("status", "prof_unst", "ord_unst"), ("record", "ord_good", "ord_bad")):
                ref_norm[c_] = None  # filled in the hook from the same batch

            def make_hook(li):
                def hook(mod, inp, outp):
                    h = outp[0] if isinstance(outp, tuple) else outp
                    for c_ in ("status", "record"):  # magnitude of each contrast at this layer, measured in this batch
                        dr_ = don_row[c_]
                        rr_ = next(i_ for i_, x in enumerate(rows) if x[0] == c_ and x[1] == "recipient")
                        ref_norm[c_] = torch.linalg.vector_norm(h[dr_, len(rows[dr_][4]) - 1].float() - h[rr_, len(rows[rr_][4]) - 1].float())
                    for ri, (c, kname, L, k, ids) in enumerate(rows):
                        if L != li:
                            continue
                        dr = don_row[c]
                        pd_, pr_ = len(rows[dr][4]) - 1, len(ids) - 1
                        hd, hr = h[dr, pd_].float(), h[ri, pr_].float()
                        if k is None:
                            new = hd
                        elif k == "norm":
                            # the same difference rescaled to the norm of the other contrast's difference: does the weaker cue move
                            # the decision when it is given equal magnitude?
                            other = "record" if c == "status" else "status"
                            sc = ref_norm[other] / (torch.linalg.vector_norm(hd - hr) + 1e-6)
                            new = hr + (hd - hr) * sc
                        else:
                            u = dirs[li][k].to(h.device)
                            new = hr + ((hd - hr) @ u) * u
                        h[ri, pr_] = new.to(h.dtype)
                    return outp
                return hook
            for li in grid:
                R.hooks.append(R.layers[li].register_forward_hook(make_hook(li)))
            lp = R.forward([x[4] for x in rows])
            R.clear()
            for ri, (c, kname, L, k, ids) in enumerate(rows):
                ll = R.letter_lp(lp[ri], labels)
                recs.append(dict(item_id=r["id"], adv=adv, prior=prior, contrast=c, kind=kname, layer=L, m_adv=ll[adv] - ll[prior],
                                 domain=cv["domain"], family=cv["family"]))
        if (ii + 1) % 25 == 0:
            el = time.time() - t0
            print(f"[progress] {ii+1}/{len(items)} items, eta {el/(ii+1)*(len(items)-ii-1)/60:.1f} min", flush=True)
    out = ACTS / a.model / f"cptlit_swap_{a.split}"
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "rows.jsonl", recs)
    json.dump(dict(model=a.model, grid=grid, fit_dir=a.fit_dir, n_items=len(items)), open(out / "meta.json", "w"), indent=1)
    print(f"[done] cpt_lit swap {a.model}: {len(items)} items in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["score", "swap"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="core")
    ap.add_argument("--exclude_split", default=None)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--fit_dir", default=None)
    ap.add_argument("--holdout_frac", type=float, default=0.0,
                    help="fraction of the fit dir's items to withhold from the directions and swap on "
                         "instead; needed when the fit dir already covers the whole split")
    a = ap.parse_args()
    dict(score=cmd_score, swap=cmd_swap)[a.mode](a)
