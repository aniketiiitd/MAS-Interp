"""The set of concepts every named model validates, written down with its provenance.

Claim 8 is measured on "the 104 concepts both models validate", and that list was built by hand. A hand-built
list is the same hazard as a hand-computed number: nothing regenerates it, nothing checks it, and when a third
model arrives it is not obvious that the list has to change. This campaign has already paid for that once --
Gemma's twelve-turn analysis ran on its own 105 concepts against Qwen's shared 104, because the flag that
restricts to the shared set was added for one model and not the other.

Adding a model SHRINKS the set, so every model must be re-analysed on the new intersection. Bolting the new one
onto old tables compares different instruments. This script therefore refuses to overwrite an existing list with
a different one unless --force is given, so that the re-analysis is a deliberate act.

Out: results/shared_concepts_<M1>_<M2>[_<M3>].txt and a provenance line per model on stdout.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import *  # noqa

ROOT = Path(__file__).resolve().parents[2]


def validated(model):
    f = TABLES / f"concept_validation_{model}.csv"
    if not f.exists():
        raise SystemExit(f"[shared_concepts] {model} has no concept_validation table ({f}). "
                         f"Run `interp/emotion_vectors.py extract --model {model} && ... fit --model {model}` first.")
    V = pd.read_csv(f)
    return set(V[V.passes].concept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Q9,G4")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing list whose contents differ; required because every downstream "
                         "table must then be recomputed")
    a = ap.parse_args()

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    if len(models) < 2:
        raise SystemExit("[shared_concepts] need at least two models to intersect")

    sets = {m: validated(m) for m in models}
    shared = set.intersection(*sets.values())
    for m in models:
        print(f"  {m:4s} validates {len(sets[m]):4d}   loses {len(sets[m] - shared):4d} to the intersection")
    print(f"  shared across {len(models)} models: {len(shared)}")

    out = ROOT / f"results/shared_concepts_{'_'.join(models)}.txt"
    body = "\n".join(sorted(shared)) + "\n"
    if out.exists() and out.read_text() != body and not a.force:
        old = {ln.strip() for ln in out.read_text().split() if ln.strip()}
        raise SystemExit(
            f"[shared_concepts] {out.name} already exists and differs: {len(old)} concepts on disk against "
            f"{len(shared)} now ({len(old - shared)} would be dropped, {len(shared - old)} added). Every table "
            f"built on the old list would have to be recomputed, so pass --force to say that is the intent.")
    out.write_text(body)
    print(f"[done] {out} ({len(shared)} concepts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
