"""V3-C1(a): status vs stated reliability inside the source label (HF, same worlds/endpoint as CPT score).

Provenance levels (header descriptors; all six new descriptors are 16 tokens under the Qwen3.5 and Gemma-4 tokenizers and inside
each of the three header templates):
  status  in {prof = "a tenured professor in this field", ord = "an ordinary member of the general public"}
  record  in {good = "a strong track record", bad = "a poor track record", unst = "an unknown track record"}
  -> "<status> with <record> on questions like this"; plus the original expert / anon / none worlds of the same template.
Worlds per item: ADV{o1,o2} x QUAL{strong,weak} x PRIOR{o1,o2} x 9 provenance levels = 72, one forward batch per item.
Output: ACTS/<M>/cpt_desc_<split>/{worlds.jsonl, acts.npy [worlds, positions, layers, d], meta.json}; positions as cpt score.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time

import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "interp"))
from common import *  # noqa
import cpt
from cpt import Receiver, endpoints
from run_cpt import POS_NAMES, layer_grid, usable_items

STATUS = {"prof": "a tenured professor in this field", "ord": "an ordinary member of the general public"}
RECORD = {"good": "a strong track record", "bad": "a poor track record", "unst": "an unknown track record"}
DESC = {f"{s}_{r}": f"{STATUS[s]} with {RECORD[r]} on questions like this" for s in STATUS for r in RECORD}
PROVS = ["expert", "anon", "none"] + list(DESC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="release_600")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exclude_split", default=None, help="drop items of this split (e.g. release_600) for a disjoint confirmation set")
    ap.add_argument("--subsample", type=int, default=0, help="keep the N items with the smallest h64('desc_conf', id)")
    a = ap.parse_args()
    for W in cpt.WRAPPERS:  # render_world looks descriptors up by provenance key
        W.update(DESC)
    R = Receiver(a.model)
    grid = layer_grid(R.n_layers)
    items = usable_items(a.split)
    if a.exclude_split:
        ex = set(json.load(open(ROOT / "data/splits/splits.json"))[a.exclude_split])
        items = [x for x in items if x[0]["id"] not in ex]
    if a.subsample:
        items = sorted(items, key=lambda x: h64("desc_conf", x[0]["id"]))[: a.subsample]
    if a.limit:
        items = items[: a.limit]
    out_dir = ACTS / a.model / (f"cpt_desc_{a.split}{'_no' + a.exclude_split if a.exclude_split else ''}"
                                f"{'_n' + str(a.subsample) if a.subsample else ''}{'_limit' + str(a.limit) if a.limit else ''}")
    out_dir.mkdir(parents=True, exist_ok=True)
    worlds = []
    for r, cv, s in items:
        tpl = h64("tpl", r["id"]) % 3
        for adv, qual, prov, prior in itertools.product((cv["o1"], cv["o2"]), ("strong", "weak"), PROVS, (cv["o1"], cv["o2"])):
            w = dict(template=tpl, adv=adv, qual=qual, prov=prov, prior=prior)
            ids, pos = R.build(cv, s, w)
            if prov != "none":
                assert pos["src"][1] - pos["src"][0] > 0
            worlds.append(dict(item_id=r["id"], family=cv["family"], domain=cv["domain"], stratum=cv["stratum"],
                               has_target=cv["has_target"], o1=cv["o1"], o2=cv["o2"], author=s["author"], **w,
                               status=prov.split("_")[0] if prov in DESC else None,
                               record=prov.split("_")[1] if prov in DESC else None,
                               agree=adv == prior, n_tok=len(ids), src_len=(pos["src"][1] - pos["src"][0]) if pos["src"] else 0,
                               _ids=ids, _pos=[pos["answer"], pos["msg_end"], pos.get("src_end", -1), pos["arg_mid"], pos["prior_end"]],
                               _cv=cv))
    # token-length check: every descriptor world of an item/template must have the same length as its prof_good sibling
    by = {}
    for w in worlds:
        if w["prov"] in DESC:
            by.setdefault((w["item_id"], w["adv"], w["qual"], w["prior"]), set()).add(w["n_tok"])
    bad = sum(len(v) > 1 for v in by.values())
    print(f"[stage] {len(worlds)} worlds, {len(items)} items; descriptor length mismatches: {bad}/{len(by)}", flush=True)
    d = R.model.config.get_text_config().hidden_size
    acts = np.lib.format.open_memmap(out_dir / "acts.npy", mode="w+", dtype=np.float16, shape=(len(worlds), len(POS_NAMES), len(grid), d))
    by_item = {}
    for i, w in enumerate(worlds):
        by_item.setdefault(w["item_id"], []).append(i)
    recs = [None] * len(worlds)
    t0, done = time.time(), 0
    for iid, batch in by_item.items():
        store = {}
        for li in grid:
            R.add_capture(li, [worlds[i]["_pos"] for i in batch], store)
        lp = R.forward([worlds[i]["_ids"] for i in batch])
        R.clear()
        for bi, i in enumerate(batch):
            w = worlds[i]
            letters = R.letter_lp(lp[bi], w["_cv"]["labels"])
            top = int(lp[bi].argmax())
            rec = {k: v for k, v in w.items() if not k.startswith("_")}
            rec.update(endpoints(letters, w["_cv"], w), lp_letters=letters, world_idx=i,
                       top_is_label=any(top in R._letter_ids[L] for L in w["_cv"]["labels"]))
            recs[i] = rec
            for gi, li in enumerate(grid):
                acts[i, :, gi] = store[li][bi].numpy().astype(np.float16)
        done += len(batch)
        if done % 7200 < len(batch):
            el = time.time() - t0
            print(f"[progress] {done}/{len(worlds)} worlds, eta {el/done*(len(worlds)-done)/60:.1f} min", flush=True)
    acts.flush()
    write_jsonl(out_dir / "worlds.jsonl", recs)
    json.dump(dict(model=a.model, sha=R.sha, grid=grid, positions=POS_NAMES, d=d, split=a.split, descriptors=DESC, provs=PROVS),
              open(out_dir / "meta.json", "w"), indent=1)
    print(f"[done] cpt_desc {a.model} {len(worlds)} worlds in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
