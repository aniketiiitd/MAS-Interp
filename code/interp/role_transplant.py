"""V3-C3(iii) causal role transplant: is the participant-vs-observer difference on identical text carried by a transferable state?

Episodes: debate agent-turns where the agent is outnumbered (both peers' t-1 verdicts agree and differ from its own).
Prompts: P = participant prompt as generated; O = minimal-edit observer prompt (observer_replay.observer_messages, framing "min").
The suffix from "Responses from the other agents" to the prefilled "VERDICT:" is byte-identical in P and O; episodes whose suffix
token ids differ are skipped. All rows of an episode share one forward pass, so donors are taken in-batch at the patched layer:
  P, O                                   unpatched
  P<-O  span  @L                         P's suffix residuals replaced by O's (same suffix offsets) at layer L
  P<-O  dec   @L                         only the decision token replaced
  P+rand span @L                         P's suffix plus a random direction with the per-position norm of (O - P)
  O<-P  span  @L                         reverse transplant
Endpoint: m_maj = lp(majority answer) - lp(own t-1 answer); recovery = (patched - P) / (O - P), aggregated as a ratio of means.
Output: ACTS/<M>/role_transplant_<dataset>[_items]/rows.jsonl
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runner"))
from common import *  # noqa
from cpt import Receiver
from observer_replay import observer_messages
from prompts import build_messages

LAYERS = (9, 12, 16, 19)
MARK = "Responses from the other agents"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="release_600")
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--items", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--transcripts", default=None)
    a = ap.parse_args()
    R = Receiver(a.model)
    layers = [l for l in LAYERS if l < R.n_layers] if R.n_layers >= 30 else sorted({int(round(f * (R.n_layers - 1))) for f in (0.29, 0.39, 0.52, 0.61)})
    core = {r["id"]: r for r in load_core()}
    d = RAW / (a.transcripts or a.model) / a.dataset
    recs = {}
    for f in ["t0.jsonl"] + [f"debate_t{t}.jsonl" for t in range(1, a.turns)]:
        for x in read_jsonl(d / f):
            if x["agent"] < 3:
                recs[(x["item_id"], x["turn"], x["agent"])] = x
    items = sorted({k[0] for k in recs})
    if a.items:
        S = set(json.load(open(ROOT / "data/splits/splits.json"))[a.items.replace("not:", "")])
        items = [i for i in items if (i in S) != a.items.startswith("not:")]
    eps = []
    for iid in items:
        v = item_view(core[iid])
        if v["kind"] == "deception" or not v["labels"]:
            continue
        for t in range(1, a.turns):
            prev = [recs.get((iid, t - 1, b)) for b in range(3)]
            if any(p is None or p["verdict"] is None for p in prev):
                continue
            for ag in range(3):
                others = [prev[b]["verdict"] for b in range(3) if b != ag]
                if others[0] == others[1] != prev[ag]["verdict"]:
                    eps.append((iid, t, ag, others[0], prev[ag]["verdict"]))
    if a.limit:
        eps = eps[: a.limit]
    print(f"[stage] {len(eps)} outnumbered episodes; layers {layers}", flush=True)
    gen = torch.Generator().manual_seed(240091)
    out, skipped, t0 = [], 0, time.time()
    for ei, (iid, t, ag, maj, own) in enumerate(eps):
        v = item_view(core[iid])
        hist = {t - 1: {b: recs[(iid, t - 1, b)]["text"] for b in range(3)}}
        enc = {}
        for role, msgs in (("P", build_messages(v, ag, t, "debate", hist)), ("O", observer_messages(v, ag, t, hist, "min"))):
            prefix = R.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            text = prefix + "VERDICT:"
            e = R.tok(text, add_special_tokens=False, return_offsets_mapping=True)
            c = text.rfind(MARK)
            start = min(i for i, (s, _) in enumerate(e["offset_mapping"]) if s >= c)
            enc[role] = (e["input_ids"], start)
        (idsP, sP), (idsO, sO) = enc["P"], enc["O"]
        if idsP[sP:] != idsO[sO:]:
            skipped += 1
            continue
        n_suf = len(idsP) - sP
        rows = [("P", None, None), ("O", None, None)]
        for L in layers:
            rows += [("P<-O", "span", L), ("P<-O", "dec", L), ("P+rand", "span", L), ("O<-P", "span", L)]
        batch = [idsO if r[0].startswith("O") else idsP for r in rows]
        base_of = lambda r: 1 if r[0].startswith("O") else 0
        spans = {0: torch.arange(sP, sP + n_suf), 1: torch.arange(sO, sO + n_suf)}

        def make_hook(L):
            def hook(mod, inp, outp):
                h = outp[0] if isinstance(outp, tuple) else outp
                hP, hO = h[0, spans[0]].clone(), h[1, spans[1]].clone()
                for ri, (kind, site, LL) in enumerate(rows):
                    if LL != L:
                        continue
                    if kind == "P<-O":
                        if site == "span":
                            h[ri, spans[0]] = hO
                        else:
                            h[ri, spans[0][-1]] = hO[-1]
                    elif kind == "O<-P":
                        h[ri, spans[1]] = hP
                    elif kind == "P+rand":
                        r = torch.randn(hP.shape, generator=gen).to(h.device, torch.float32)
                        r = r / r.norm(dim=-1, keepdim=True) * (hO.float() - hP.float()).norm(dim=-1, keepdim=True)
                        h[ri, spans[0]] = hP + r.to(h.dtype)
                return outp
            return hook
        for L in layers:
            R.hooks.append(R.layers[L].register_forward_hook(make_hook(L)))
        lp = R.forward(batch)
        R.clear()
        for ri, (kind, site, L) in enumerate(rows):
            ll = R.letter_lp(lp[ri], v["labels"])
            out.append(dict(item_id=iid, turn=t, agent=ag, majority=maj, own=own, kind=kind, site=site, layer=L,
                            m_maj=ll[maj] - ll[own], label_mass=float(sum(np.exp(x) for x in ll.values()))))
        if (ei + 1) % 100 == 0:
            el = time.time() - t0
            print(f"[progress] {ei+1}/{len(eps)} episodes, eta {el/(ei+1)*(len(eps)-ei-1)/60:.1f} min", flush=True)
    od = ACTS / a.model / f"role_transplant_{a.dataset}{'_' + a.items.replace(':', '') if a.items else ''}{'_limit' + str(a.limit) if a.limit else ''}"
    od.mkdir(parents=True, exist_ok=True)
    write_jsonl(od / "rows.jsonl", out)
    json.dump(dict(model=a.model, layers=layers, skipped_suffix_mismatch=skipped, n_episodes=len(eps) - skipped), open(od / "meta.json", "w"))
    print(f"[done] role transplant {a.model} {a.dataset}: {len(eps)-skipped} episodes ({skipped} skipped) in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
