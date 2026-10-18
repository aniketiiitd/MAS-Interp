"""E16 causal steering and E21 counterfactual recourse on the Q9 receiver (HF).

steer:    for frozen probes {provenance, evidence_quality, social_disagreement}, add alpha * SD(score) along the probe's
          raw-space direction at its selected (layer, token group), alpha in {-2,-1,0,1,2}; controls: norm-matched random
          direction and a random direction orthogonalized to the probe. Endpoints: m_adv, m12, output KL vs alpha=0,
          label mass. Items: sprint_240 minus hero_48 (hero was used to pick patch layers).
recourse: harmful-adoption worlds (target items; receiver prior correct; strong expert message for the incorrect option).
          Routes: (a) neutral message, (b) source label removed, (c) provenance state patch (anon source-span states at the
          onset layer chosen on hero patching), (d) provenance steering -2 SD, (e) evidence steering -2 SD.
          Side effects: the same intervention on the matched correct-advocacy world (C+) must not block correct uptake.
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
from common import *  # noqa
from cpt import Receiver, endpoints
from run_cpt import usable_items, groups_for, POS_NAMES

PROBES = ["provenance", "evidence_quality", "social_disagreement"]
CONCEPT_STEER = ["deferential", "trust_in_source", "dissent", "belief_updating", "conformity_to_majority"]
SITE_GROUP = {"src_end": "src", "arg_mid": "arg", "msg_end": "arg", "answer": "answer", "prior_end": "answer"}


def load_frozen(model, run="cpt_core"):
    fz = json.load(open(ACTS / model / f"probes_frozen_{run}.json"))
    meta = json.load(open(ACTS / model / run / "meta.json"))
    acts = np.load(ACTS / model / run / "acts.npy", mmap_mode="r")
    out = {}
    rng = np.random.default_rng(240091)
    for name in PROBES:
        f = fz[name]
        sd = np.array(f["sd"], dtype=np.float32)
        w_raw = np.array(f["w_dom"], dtype=np.float32) / sd
        # SD of the probe score at the site, estimated on 4000 random worlds
        rows = np.sort(rng.choice(acts.shape[0], 4000, replace=False))
        H = acts[rows, POS_NAMES.index(f["position"]), f["grid_index"]].astype(np.float32)
        s_sd = float((H @ w_raw).std())
        unit = w_raw / np.linalg.norm(w_raw)
        step = s_sd / np.linalg.norm(w_raw)  # shift along unit that moves the score by 1 SD
        r = rng.standard_normal(unit.shape).astype(np.float32)
        r /= np.linalg.norm(r)
        o = r - (r @ unit) * unit
        o /= np.linalg.norm(o)
        out[name] = dict(layer=f["layer"], group=SITE_GROUP[f["position"]], unit=torch.tensor(unit), rand=torch.tensor(r),
                         orth=torch.tensor(o), step=step)
    # construct-validated concept directions (pass story validation AND moved as predicted by CPT manipulations)
    cm = json.load(open(ACTS / model / "concepts" / "directions_meta.json"))
    U = np.load(ACTS / model / "concepts" / "directions.npy")
    V = pd.read_csv(TABLES / f"concept_validation_{model}.csv").set_index("concept")
    vd = pd.read_csv(TABLES / f"concept_cpt_validity_{model}.csv")
    confirmed = set(vd[vd.prereg & (vd.lower > 0)].concept)
    gi = meta["grid"].index(cm["layer"])
    rows = np.sort(rng.choice(acts.shape[0], 4000, replace=False))
    H = acts[rows, POS_NAMES.index("answer"), gi].astype(np.float32)
    for c in CONCEPT_STEER:
        if c not in V.index or not bool(V.loc[c, "passes"]) or c not in confirmed:
            print(f"  concept {c} not steered (validation={c in V.index and bool(V.loc[c, 'passes'])}, construct={c in confirmed})", flush=True)
            continue
        u = U[cm["concepts"].index(c)].astype(np.float32)
        u /= np.linalg.norm(u)
        r = rng.standard_normal(u.shape).astype(np.float32)
        r /= np.linalg.norm(r)
        o = r - (r @ u) * u
        o /= np.linalg.norm(o)
        out[f"concept_{c}"] = dict(layer=cm["layer"], group="post", unit=torch.tensor(u), rand=torch.tensor(r), orth=torch.tensor(o),
                                   step=float((H @ u).std()))
    return out


def kl(lp_a, lp_b):
    return float((lp_a.exp() * (lp_a - lp_b)).sum())


def cmd_steer(args):
    R = Receiver(args.model)
    P = load_frozen(args.model)
    splits = json.load(open(ROOT / "data/splits/splits.json"))
    keep = set(splits["sprint_240"]) - set(splits["hero_48"])
    items = [x for x in usable_items("core") if x[0]["id"] in keep][: args.n_items]
    recs = []
    t0 = time.time()
    for ii, (r, cv, s) in enumerate(items):
        tpl = h64("tpl", r["id"]) % 3
        worlds = []
        for adv in (cv["o1"], cv["o2"]):
            prior = cv["o2"] if adv == cv["o1"] else cv["o1"]
            for qual in ("strong", "weak"):
                for prov in ("expert", "anon"):
                    worlds.append(dict(template=tpl, adv=adv, qual=qual, prov=prov, prior=prior))
        built = [R.build(cv, s, w) for w in worlds]
        for name, pr in P.items():
            for w0 in range(0, len(built), 2):  # constant batch shape: 2 worlds x (1 baseline + 12 steered) = 26 rows
                rows = []
                for wi in (w0, w0 + 1):
                    ids, pos = built[wi]
                    g = groups_for(pos, len(ids))[pr["group"]]
                    rows.append((wi, g, "probe", 0))
                    for mode in ("probe", "random", "orth"):
                        for alpha in (-2, -1, 1, 2):
                            rows.append((wi, g, mode, alpha))
                per_layer = []
                for k, (wi, g, mode, alpha) in enumerate(rows):
                    if alpha == 0:
                        continue
                    vec = {"probe": pr["unit"], "random": pr["rand"], "orth": pr["orth"]}[mode] * (alpha * pr["step"])
                    per_layer.append((k, torch.tensor(g), vec.to(R.device).expand(len(g), -1), "add"))
                R.add_patch(pr["layer"], per_layer)
                lp = R.forward([built[wi][0] for wi, _, _, _ in rows])
                R.clear()
                base = {wi: k for k, (wi, g, mode, alpha) in enumerate(rows) if alpha == 0}
                for k, (wi, g, mode, alpha) in enumerate(rows):
                    w = worlds[wi]
                    ep = endpoints(R.letter_lp(lp[k], cv["labels"]), cv, w)
                    recs.append(dict(item_id=r["id"], family=cv["family"], domain=cv["domain"], has_target=cv["has_target"],
                                     probe=name, mode=mode, alpha=alpha, **{f"w_{kk}": vv for kk, vv in w.items()},
                                     m_adv=ep["m_adv"], m12=ep["m12"], label_mass=ep["p_label_mass"],
                                     kl=kl(lp[base[wi]], lp[k]), n_tokens_steered=len(g)))
        if (ii + 1) % 10 == 0:
            print(f"[progress] steer {ii+1}/{len(items)} {(time.time()-t0)/60:.1f} min", flush=True)
    d = ACTS / args.model / "steer"
    d.mkdir(parents=True, exist_ok=True)
    write_jsonl(d / "steer.jsonl", recs)
    print(f"[done] steer {args.model}: {len(recs)} rows in {(time.time()-t0)/60:.1f} min", flush=True)


def cmd_recourse(args):
    R = Receiver(args.model)
    P = load_frozen(args.model)
    patch = pd.DataFrame(read_jsonl(ACTS / args.model / "patch_hero_48" / "patch.jsonl"))
    pp = patch[(patch.kind == "prov") & (patch.mode == "patch") & (patch.group == "src")].copy()
    pp["rec"] = pp.madv_patched - pp.madv_recip
    onset_layer = int(pp.groupby("layer").rec.apply(lambda x: np.abs(x).mean()).idxmax())
    splits = json.load(open(ROOT / "data/splits/splits.json"))
    keep = set(splits["sprint_240"]) - set(splits["hero_48"])
    items = [x for x in usable_items("core") if x[0]["id"] in keep and x[1]["has_target"]
             and x[1]["domain"] in ("safety_refusal", "knowledge_reasoning")][: args.n_items]
    recs = []
    t0 = time.time()
    for ii, (r, cv, s) in enumerate(items):
        tpl = h64("tpl", r["id"]) % 3
        o1, o2 = cv["o1"], cv["o2"]
        for label, adv, prior in (("harmful", o2, o1), ("side_effect", o1, o2)):
            w_exp = dict(template=tpl, adv=adv, qual="strong", prov="expert", prior=prior)
            w_anon = dict(w_exp, prov="anon")
            w_none = dict(w_exp, prov="none")
            w_neu = dict(template=tpl, adv=None, qual="neutral", prov="expert", prior=prior)
            ids_e, pos_e = R.build(cv, s, w_exp)
            ids_a, pos_a = R.build(cv, s, w_anon)
            # capture anon source-span states at the onset layer (same batch shape as below)
            store = {}
            R.add_capture(onset_layer, [list(range(*pos_a["src"]))] + [[0]] * 7, store)
            R.forward([ids_a] + [ids_e] * 7)
            R.clear()
            anon_src = store[onset_layer][0].to(R.device)
            rows = [("before", ids_e, None), ("a_neutral_message", R.build(cv, s, w_neu)[0], None),
                    ("b_source_label_removed", R.build(cv, s, w_none)[0], None),
                    ("c_provenance_state_patch", ids_e, ("patch", onset_layer, list(range(*pos_e["src"])), anon_src)),
                    ("d_provenance_steer_-2", ids_e, ("add", P["provenance"]["layer"], groups_for(pos_e, len(ids_e))[P["provenance"]["group"]],
                                                     P["provenance"]["unit"] * (-2 * P["provenance"]["step"]))),
                    ("e_evidence_steer_-2", ids_e, ("add", P["evidence_quality"]["layer"], groups_for(pos_e, len(ids_e))[P["evidence_quality"]["group"]],
                                                   P["evidence_quality"]["unit"] * (-2 * P["evidence_quality"]["step"]))),
                    ("anon_reference", ids_a, None), ("pad", ids_e, None)]
            by_layer = {}
            for k, (_, _, iv) in enumerate(rows):
                if iv is None:
                    continue
                kind, layer, g, val = iv
                if kind == "patch":
                    by_layer.setdefault(layer, []).append((k, torch.tensor(g), val))
                else:
                    by_layer.setdefault(layer, []).append((k, torch.tensor(g), val.to(R.device).expand(len(g), -1), "add"))
            for layer, rp in by_layer.items():
                R.add_patch(layer, rp)
            lp = R.forward([x[1] for x in rows])
            R.clear()
            for k, (route, _, _) in enumerate(rows):
                if route == "pad":
                    continue
                w = w_neu if route == "a_neutral_message" else (w_none if route == "b_source_label_removed" else (w_anon if route == "anon_reference" else w_exp))
                ep = endpoints(R.letter_lp(lp[k], cv["labels"]), cv, dict(w, adv=w.get("adv") or adv))
                recs.append(dict(item_id=r["id"], family=cv["family"], domain=cv["domain"], scenario=label, route=route,
                                 m12=ep["m12"], p_correct=ep["p_o1"], label_mass=ep["p_label_mass"], kl_vs_before=kl(lp[0], lp[k]),
                                 onset_layer=onset_layer))
        if (ii + 1) % 10 == 0:
            print(f"[progress] recourse {ii+1}/{len(items)} {(time.time()-t0)/60:.1f} min", flush=True)
    d = ACTS / args.model / "recourse"
    d.mkdir(parents=True, exist_ok=True)
    write_jsonl(d / "recourse.jsonl", recs)
    print(f"[done] recourse {args.model}: {len(recs)} rows, onset layer {onset_layer}, {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["steer", "recourse"])
    ap.add_argument("--model", default="Q9")
    ap.add_argument("--n_items", type=int, default=120)
    a = ap.parse_args()
    {"steer": cmd_steer, "recourse": cmd_recourse}[a.cmd](a)
