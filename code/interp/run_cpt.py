"""CPT runs on HF models.

score: every item x world (one hashed wrapper template per item, or all templates for --all_templates),
       endpoint = option-letter log-probs after "VERDICT:"; captures residual states on a normalized layer grid
       at {answer, msg_end, src_end, arg_mid, prior_end}.
patch: provenance patching (expert <-> anon, same everything else) and content patching (strong <-> weak) on the
       hero items; one (layer, token group) site per forward row; norm-matched random-direction controls.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "interp"))
from common import *  # noqa
from cpt import Receiver, endpoints, load_stimuli, world_list

POS_NAMES = ["answer", "msg_end", "src_end", "arg_mid", "prior_end"]


def usable_items(split=None):
    splits = json.load(open(ROOT / "data/splits/splits.json"))
    stim = load_stimuli()
    rej = set(json.load(open(ROOT / "data/mad_interp/rejected_items.json"))) if (ROOT / "data/mad_interp/rejected_items.json").exists() else set()
    rows = load_core()
    keep = set(splits[split]) if split and split != "core" else None
    out = []
    for r in rows:
        if keep is not None and r["id"] not in keep:
            continue
        s = stim.get(r["id"])
        if s is None or r["id"] in rej:
            continue
        cv = cpt_view(r, s["numeric_distractor"])
        if cv is None:
            continue
        out.append((r, cv, s))
    return out


def layer_grid(n_layers, k=10):
    return sorted({int(round(f * (n_layers - 1))) for f in np.linspace(0.1, 1.0, k)})


def token_budget_batches(order, lengths, budget, max_rows=64):
    batch, tot = [], 0
    for i in order:
        L = lengths[i]
        if batch and (max(tot, L) * (len(batch) + 1) > budget or len(batch) >= max_rows):
            yield batch
            batch, tot = [], 0
        batch.append(i)
        tot = max(tot, L)
    if batch:
        yield batch


def cmd_score(args):
    R = Receiver(args.model)
    grid = layer_grid(R.n_layers)
    items = usable_items(args.split)
    if args.limit:
        items = items[: args.limit]
    out_dir = ACTS / args.model / f"cpt_{args.split}{'_alltpl' if args.all_templates else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    worlds = []
    for r, cv, s in items:
        tpls = [0, 1, 2] if args.all_templates else [h64("tpl", r["id"]) % 3]
        for w in world_list(cv, tpls):
            ids, pos = R.build(cv, s, w)
            worlds.append(dict(item_id=r["id"], family=cv["family"], domain=cv["domain"], stratum=cv["stratum"],
                               has_target=cv["has_target"], o1=cv["o1"], o2=cv["o2"], author=s["author"], **w,
                               agree=(w["adv"] == w["prior"]) if w["adv"] else None, n_tok=len(ids),
                               _ids=ids, _pos=[pos["answer"], pos["msg_end"], pos.get("src_end", -1), pos["arg_mid"], pos["prior_end"]],
                               _cv=cv))
    print(f"[stage] built {len(worlds)} worlds for {len(items)} items in {time.time()-t0:.0f}s", flush=True)
    d = R.model.config.get_text_config().hidden_size
    acts = np.lib.format.open_memmap(out_dir / "acts.npy", mode="w+", dtype=np.float16,
                                     shape=(len(worlds), len(POS_NAMES), len(grid), d))
    # one forward batch per item: every within-item contrast shares kernels/batch shape (bf16 numerics depend on shape)
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
        if done % 5000 < len(batch):
            el = time.time() - t0
            print(f"[progress] {done}/{len(worlds)} worlds {el:.0f}s eta {el/done*(len(worlds)-done)/60:.1f} min", flush=True)
    acts.flush()
    write_jsonl(out_dir / "worlds.jsonl", recs)
    json.dump(dict(model=args.model, sha=R.sha, grid=grid, positions=POS_NAMES, n_layers=R.n_layers, d=d,
                   split=args.split, all_templates=args.all_templates), open(out_dir / "meta.json", "w"), indent=1)
    print(f"[done] cpt score {args.model} {len(worlds)} worlds in {(time.time()-t0)/60:.1f} min", flush=True)


# ---------------------------------------------------------------------------
def groups_for(pos, n):
    """Token groups for patching (all aligned between donor and recipient for provenance swaps)."""
    return dict(src=list(range(*pos["src"])), arg=list(range(*pos["arg"])),
                post=list(range(pos["msg"][1], n - 1)), answer=[n - 1])


def cmd_patch(args):
    R = Receiver(args.model)
    layers = list(range(R.n_layers))
    rand_layers = layers[:: max(1, R.n_layers // 8)]
    items = usable_items(args.split)
    if args.limit:
        items = items[: args.limit]
    out_dir = ACTS / args.model / f"patch_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(240091)
    recs = []
    t0 = time.time()
    for ii, (r, cv, s) in enumerate(items):
        tpl = h64("tpl", r["id"]) % 3
        pairs = []
        for adv in (cv["o1"], cv["o2"]):
            prior = cv["o2"] if adv == cv["o1"] else cv["o1"]  # disagreement worlds: the message pushes a revision
            for qual in ("strong", "weak"):
                for donor_p, recip_p in (("expert", "anon"), ("anon", "expert")):
                    pairs.append(("prov", dict(template=tpl, adv=adv, qual=qual, prov=donor_p, prior=prior),
                                  dict(template=tpl, adv=adv, qual=qual, prov=recip_p, prior=prior)))
            for donor_q, recip_q in (("strong", "weak"), ("weak", "strong")):
                pairs.append(("qual", dict(template=tpl, adv=adv, qual=donor_q, prov="anon", prior=prior),
                              dict(template=tpl, adv=adv, qual=recip_q, prov="anon", prior=prior)))
        for kind, wd, wr in pairs:
            ids_d, pos_d = R.build(cv, s, wd)
            ids_r, pos_r = R.build(cv, s, wr)
            if kind == "prov" and (len(ids_d) != len(ids_r) or pos_d["src"] != pos_r["src"]):
                print(f"  skip misaligned {r['id']} {wd['prov']}", flush=True)
                continue
            gd, gr = groups_for(pos_d, len(ids_d)), groups_for(pos_r, len(ids_r))
            if kind == "qual":  # align post/answer groups by offset from the end; arg lengths differ
                n_post = min(len(gd["post"]), len(gr["post"]))
                gd = dict(post=gd["post"][-n_post:], answer=gd["answer"])
                gr = dict(post=gr["post"][-n_post:], answer=gr["answer"])
            # capture donor and recipient states from a batch with the same shape as the patched batches
            allpos_d = sorted({p for g in gd.values() for p in g})
            allpos_r = sorted({p for g in gr.values() for p in g})
            nrow = args.rows
            same_len = len(ids_d) == len(ids_r)
            store = {}
            ple_d = {}
            if same_len:
                for li in layers:
                    R.add_capture(li, [allpos_d, allpos_r] + [[0]] * (nrow - 2), store)
                    if R.has_ple:
                        R.add_ple_capture(li, [allpos_d] + [[0]] * (nrow - 1), ple_d)
                R.forward([ids_d, ids_r] + [ids_r] * (nrow - 2))
                R.clear()
                store_d = {li: [store[li][0]] for li in layers}
                store_r = {li: [store[li][1]] for li in layers}
            else:  # content swap: lengths differ, run donor batch separately with its own fixed shape
                for li in layers:
                    R.add_capture(li, [allpos_d] + [[0]] * (nrow - 1), store)
                R.forward([ids_d] * nrow)
                R.clear()
                store_d = {li: [store[li][0]] for li in layers}
                store = {}
                for li in layers:
                    R.add_capture(li, [allpos_r] + [[0]] * (nrow - 1), store)
                R.forward([ids_r] * nrow)
                R.clear()
                store_r = {li: [store[li][0]] for li in layers}
            idx_d = {p: k for k, p in enumerate(allpos_d)}
            idx_r = {p: k for k, p in enumerate(allpos_r)}
            sites = [(li, g, "patch") for li in layers for g in gd] + [(li, g, "random") for li in rand_layers for g in gd]
            per = nrow - 1  # row 0 of every chunk is the clean recipient
            for b0 in range(0, len(sites), per):
                chunk = sites[b0: b0 + per]
                per_layer, ple_rows = {}, {}
                for k, (li, g, mode) in enumerate(chunk):
                    row = k + 1
                    pr = torch.tensor(gr[g])
                    vd = store_d[li][0][[idx_d[p] for p in gd[g]]]
                    vr = store_r[li][0][[idx_r[p] for p in gr[g]]]
                    if mode == "patch":
                        per_layer.setdefault(li, []).append((row, pr, vd.to(R.device)))
                        if R.has_ple and ple_d and kind == "prov":
                            # token identity of the donor must also enter every later layer at the patched positions
                            for lj in range(li + 1, R.n_layers):
                                ple_rows.setdefault(lj, []).append((row, pr, ple_d[lj][0][[idx_d[p] for p in gd[g]]].to(R.device)))
                    else:
                        delta = vd - vr
                        rnd = torch.randn(delta.shape, generator=gen)
                        rnd = rnd / rnd.norm(dim=-1, keepdim=True) * delta.norm(dim=-1, keepdim=True)
                        per_layer.setdefault(li, []).append((row, pr, rnd.to(R.device), "add"))
                for li, rp in per_layer.items():
                    R.add_patch(li, rp)
                for lj, rp in ple_rows.items():
                    R.add_ple_patch(lj, rp)
                lp = R.forward([ids_r] * nrow)
                R.clear()
                ep_r = endpoints(R.letter_lp(lp[0], cv["labels"]), cv, wr)
                for k, (li, g, mode) in enumerate(chunk):
                    ep = endpoints(R.letter_lp(lp[k + 1], cv["labels"]), cv, wr)
                    recs.append(dict(item_id=r["id"], family=cv["family"], domain=cv["domain"], kind=kind,
                                     donor=wd, recipient=wr, layer=li, group=g, mode=mode,
                                     m12_patched=ep["m12"], madv_patched=ep["m_adv"], m12_recip=ep_r["m12"],
                                     madv_recip=ep_r["m_adv"], n_group=len(gr[g])))
            # clean donor margin under the identical batch shape (for recovery denominators)
            lp = R.forward([ids_d] * nrow)
            ep_d = endpoints(R.letter_lp(lp[0], cv["labels"]), cv, wd)
            recs.append(dict(item_id=r["id"], family=cv["family"], domain=cv["domain"], kind=kind, donor=wd, recipient=wr,
                             layer=-1, group="clean_donor", mode="clean", m12_patched=ep_d["m12"], madv_patched=ep_d["m_adv"],
                             m12_recip=ep_r["m12"], madv_recip=ep_r["m_adv"], n_group=0))
        if (ii + 1) % 4 == 0:
            el = time.time() - t0
            print(f"[progress] {ii+1}/{len(items)} items {el/60:.1f} min eta {el/(ii+1)*(len(items)-ii-1)/60:.1f} min", flush=True)
            write_jsonl(out_dir / "patch.jsonl", recs)
    write_jsonl(out_dir / "patch.jsonl", recs)
    json.dump(dict(model=args.model, sha=R.sha, n_layers=R.n_layers, rand_layers=rand_layers, split=args.split),
              open(out_dir / "meta.json", "w"), indent=1)
    print(f"[done] cpt patch {args.model}: {len(recs)} rows in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "patch"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="core")
    ap.add_argument("--all_templates", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--token_budget", type=int, default=24000)
    ap.add_argument("--rows", type=int, default=24)
    args = ap.parse_args()
    {"score": cmd_score, "patch": cmd_patch}[args.cmd](args)
