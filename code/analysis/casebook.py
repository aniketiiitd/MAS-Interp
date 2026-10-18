"""E24 algorithmic casebook: 30 cards, one 'positive' and one 'failure' case per subdomain family, selected by rule + hash
from sprint_240 (never by hand). Positive = top quartile (within family) of |P| + |IQP| where the sign of P matches the
family estimate. Failure = the debate ended wrong/unsafe although turn-0 majority was right, or (no target) the largest
public/private mismatch; fallback: bottom quartile of CPT sensitivity. Ties broken by hash(item_id)."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa


def main(model="Q9"):
    splits = json.load(open(ROOT / "data/splits/splits.json"))
    sprint = set(splits["sprint_240"])
    core = {r["id"]: r for r in load_core()}
    cpt = pd.read_csv(TABLES / f"cpt_{model}_cpt_core_m_adv_items.csv").set_index("item_id")
    summ = pd.read_csv(TABLES / f"cpt_{model}_cpt_core_m_adv_summary.csv")
    fam_sign = summ[(summ.level == "family") & (summ.contrast == "P")].set_index("group").estimate.apply(np.sign)
    ep = pd.read_csv(TABLES / f"{model}_core_turns.csv")
    last = ep.turn.max()
    deb = ep[ep.cond == "debate"].pivot_table(index="item_id", columns="turn", values="majority_correct", aggfunc="first")
    gap = pd.read_csv(TABLES / f"private_gap_{model}_core.csv") if (TABLES / f"private_gap_{model}_core.csv").exists() else None
    cards = []
    for _, fam, lab in FAMILY_ORDER:
        ids = [i for i in sprint if family_of(core[i]) == fam and i in cpt.index]
        if not ids:
            continue
        C = cpt.loc[ids].copy()
        C["score"] = C.P.abs() + C.IQP.abs()
        C["hash"] = [h64("casebook", i) for i in C.index]
        pos = C[(np.sign(C.P) == fam_sign.get(fam, 1)) & (C.score >= C.score.quantile(0.75))].sort_values("hash")
        chosen_pos = pos.index[0] if len(pos) else None
        fail = []
        for i in ids:
            if i in deb.index and deb.loc[i].get(0) in (True, "True") and deb.loc[i].get(last) in (False, "False"):
                fail.append(i)
        rule = "debate turned a correct turn-0 majority wrong"
        if not fail and gap is not None:
            g = gap[(gap.item_id.isin(ids)) & (gap.cond == "debate")]
            if len(g):
                thr = g.gap.quantile(0.9)
                fail = list(g[g.gap >= max(thr, 1e-9)].item_id)
                rule = "top-decile public/private mismatch"
        if not fail:
            fail = list(C[C.score <= C.score.quantile(0.25)].index)
            rule = "bottom-quartile CPT sensitivity (inert)"
        fail = sorted([i for i in fail if i != chosen_pos], key=lambda i: h64("casebook", i))
        for kind, iid, why in (("positive", chosen_pos, "top-quartile provenance/interaction sensitivity"), ("failure", fail[0] if fail else None, rule)):
            if iid is None:
                continue
            r = core[iid]
            row = cpt.loc[iid]
            traj = ep[(ep.item_id == iid) & (ep.cond == "debate")].sort_values("turn")
            cards.append(dict(family=lab, kind=kind, rule=why, item_id=iid, prompt=r["prompt"][:400], target=r["ground_truth"],
                              public_majority_by_turn=list(traj.majority), agreement_by_turn=list(traj.agreement.round(2)),
                              pcif=dict(Q=round(row.Q, 3), P=round(row.P, 3), IQP=round(row.IQP, 3), S=round(row.S, 3))))
    write_jsonl(TABLES / f"casebook_{model}.jsonl", cards)
    print(f"{len(cards)} cards")


if __name__ == "__main__":
    main()
