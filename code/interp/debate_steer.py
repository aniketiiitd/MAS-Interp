"""C3 steering inside natural debates (confirmation half only; directions chosen on the discovery half).

Episode: agent a at turn t in a 12-turn Q9 debate whose two peers held the same, different answer at t-1 (united
dissent). The agent's real turn-t prompt is scored after a prefilled "VERDICT:"; endpoint m_adopt = log p(majority option)
- log p(agent's own t-1 option). Interventions add alpha * SD * u over the peer-message span and the decision tokens at the
concept layer, alpha in {-4,-2,2,4} (SD = std of the projection of debate decision states on u). Directions: replicated
discovery leaders (lead analysis), pre-registered stances, CPT probes; each paired with a norm-matched random direction
and one orthogonalized to it. Fixed batch shape: 25 rows per chunk, row 0 = unsteered baseline of the same episode.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "interp"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runner"))
from common import *  # noqa
from cpt import Receiver, endpoints
from prompts import build_messages

PREREG = ["deference_to_expertise", "conformity_to_majority", "independent_judgment", "pressure_from_numbers", "stubbornness",
          "belief_updating"]
ALPHAS = (-4, -2, 2, 4)
ROWS = 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--dataset", default="release_600")
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--max_episodes", type=int, default=420)
    ap.add_argument("--sd_replay", default=None, help="replay dir used to estimate projection SD (defaults to this dataset's replay)")
    ap.add_argument("--all_items", action="store_true", help="use both halves (valid only when no direction was selected on debate data)")
    ap.add_argument("--directions", default=None, help="comma list overriding the automatic choice")
    ap.add_argument("--tag", default="")
    ap.add_argument("--max_turn", type=int, default=3)
    a = ap.parse_args()
    R = Receiver(a.model)
    tag = f"{a.model}_T{a.turns}"
    cm = json.load(open(ACTS / a.model / "concepts" / "directions_meta.json"))
    U = np.load(ACTS / a.model / "concepts" / "directions.npy")
    concepts = cm["concepts"]
    layer = cm["layer"]
    V = pd.read_csv(TABLES / f"concept_validation_{a.model}.csv").set_index("concept")
    lead = pd.read_csv(TABLES / f"traj_lead_confirmation_{tag}_dec.csv") if (TABLES / f"traj_lead_confirmation_{tag}_dec.csv").exists() else pd.DataFrame()
    leaders = list(lead[lead.replicates].concept) if len(lead) else []
    chosen = list(dict.fromkeys(leaders[:6] + [c for c in PREREG if c in V.index and V.loc[c, "passes"]]))
    if a.directions:
        chosen = a.directions.split(",")
    dirs = {c: U[concepts.index(c)] for c in chosen}
    # CPT probes as additional directions (raw-space unit vectors at their own layer are not used; we re-express at this layer
    # only if the probe was fit at the same layer)
    fz = json.load(open(ACTS / a.model / "probes_frozen_cpt_core.json"))
    for name in ("social_disagreement", "provenance"):
        if name in fz and fz[name]["layer"] == layer:
            w = np.array(fz[name]["w_dom"], np.float32) / np.array(fz[name]["sd"], np.float32)
            dirs[f"probe_{name}"] = w / np.linalg.norm(w)
    # SD of projections on debate decision states at this layer (replay)
    rep = ACTS / a.model / (a.sd_replay or f"replay_{a.dataset}_release_600_T{a.turns}")
    rmeta = json.load(open(rep / "meta.json"))
    A = np.load(rep / "acts.npy", mmap_mode="r")
    gi = rmeta["grid"].index(layer)
    rng = np.random.default_rng(240091)
    sample = np.sort(rng.choice(A.shape[0], min(6000, A.shape[0]), replace=False))
    H = A[sample, 0, gi].astype(np.float32)
    units, steps = {}, {}
    for name, u in dirs.items():
        u = u / np.linalg.norm(u)
        r = rng.standard_normal(u.shape).astype(np.float32)
        r /= np.linalg.norm(r)
        o = r - (r @ u) * u
        o /= np.linalg.norm(o)
        sd = float((H @ u).std())
        for kind, vec in (("dir", u), ("random", r), ("orth", o)):
            units[(name, kind)] = torch.tensor(vec)
            steps[(name, kind)] = sd  # same magnitude for controls
    # episodes
    core = {r_["id"]: r_ for r_ in load_core()}
    unit = lambda i: core[i]["meta"].get("analysis_group") if core[i]["stratum"] == "persuasion" else core[i]["source_item_id"]
    d = RAW / a.model / a.dataset
    texts = {}
    for f in ["t0.jsonl"] + [f"debate_t{t}.jsonl" for t in range(1, a.turns)]:
        for x in read_jsonl(d / f):
            if x["agent"] < 3:
                texts[(x["item_id"], x["turn"], x["agent"])] = x
    eps = []
    for iid in sorted({k[0] for k in texts}):
        if not a.all_items and h64("disc240091", unit(iid)) % 2 == 0:  # discovery half excluded
            continue
        v = item_view(core[iid])
        if v["kind"] != "choice":
            continue
        for t in range(1, a.max_turn + 1):
            prev = [texts.get((iid, t - 1, b)) for b in range(3)]
            if any(p is None or p["verdict"] is None for p in prev):
                continue
            for ag in range(3):
                others = [prev[b]["verdict"] for b in range(3) if b != ag]
                if others[0] == others[1] and others[0] != prev[ag]["verdict"]:
                    maj = others[0]
                    kind = "normative" if v["target"] is None else ("wrong_majority" if maj != v["target"] else "right_majority")
                    eps.append(dict(item_id=iid, turn=t, agent=ag, majority=maj, own=prev[ag]["verdict"], episode=kind,
                                    family=v["family"], domain=v["domain"]))
    eps = sorted(eps, key=lambda e: h64("steer_ep", e["item_id"], e["turn"], e["agent"]))[: a.max_episodes]
    print(f"[stage] {len(eps)} episodes; directions: {list(dirs)}", flush=True)
    recs = []
    t0 = time.time()
    conds = [(name, kind, al) for name in dirs for kind in ("dir", "random", "orth") for al in ALPHAS]
    for ei, e in enumerate(eps):
        v = item_view(core[e["item_id"]])
        hist = {tt: {b: texts[(e["item_id"], tt, b)]["text"] for b in range(3)} for tt in range(e["turn"])}
        msgs = build_messages(v, e["agent"], e["turn"], "debate", hist)
        prefix = R.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = R.tok(prefix + "VERDICT:", add_special_tokens=False, return_offsets_mapping=True)
        ids = enc["input_ids"]
        start_char = prefix.find("Responses from the other agents")
        span = [i for i, (s, _) in enumerate(enc["offset_mapping"]) if s >= start_char]
        pos = torch.tensor(span)
        cv = dict(labels=v["labels"], o1=e["majority"], o2=e["own"])
        w = dict(adv=e["majority"], prior=e["own"])
        for c0 in range(0, len(conds), ROWS - 1):
            chunk = conds[c0: c0 + ROWS - 1]
            rows = [None] + chunk + [chunk[-1]] * (ROWS - 1 - len(chunk))
            patches = []
            for k, cnd in enumerate(rows):
                if cnd is None:
                    continue
                name, kind, al = cnd
                vec = (units[(name, kind)] * (al * steps[(name, kind)])).to(R.device)
                patches.append((k, pos, vec.expand(len(span), -1), "add"))
            R.add_patch(layer, patches)
            lp = R.forward([ids] * ROWS)
            R.clear()
            base = lp[0]
            eb = endpoints(R.letter_lp(base, v["labels"]), cv, w)
            for k, cnd in enumerate(rows[: len(chunk) + 1]):
                if cnd is None:
                    if c0 == 0:
                        recs.append(dict(e, direction="none", ctrl="base", alpha=0, m_adopt=eb["m_adv"], d_m_adopt=0.0, kl=0.0,
                                         label_mass=eb["p_label_mass"]))
                    continue
                ep = endpoints(R.letter_lp(lp[k], v["labels"]), cv, w)
                kl = float((base.exp() * (base - lp[k])).sum())
                recs.append(dict(e, direction=cnd[0], ctrl=cnd[1], alpha=cnd[2], m_adopt=ep["m_adv"], d_m_adopt=ep["m_adv"] - eb["m_adv"],
                                 kl=kl, label_mass=ep["p_label_mass"]))
        if (ei + 1) % 25 == 0:
            el = time.time() - t0
            print(f"[progress] {ei+1}/{len(eps)} episodes, eta {el/(ei+1)*(len(eps)-ei-1)/60:.1f} min", flush=True)
    out = ACTS / a.model / f"debate_steer{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "steer.jsonl", recs)
    json.dump(dict(layer=layer, directions=list(dirs), leaders=leaders, alphas=ALPHAS), open(out / "meta.json", "w"), indent=1)
    print(f"[done] debate steering: {len(recs)} rows in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
