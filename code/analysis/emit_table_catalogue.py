"""Catalogue the result tables: what each contains, and which of them the paper actually stands on.

422 tables exist. The paper depends on a subset, and a reader handed all of them has no way to tell which. This
writes two things: a catalogue of the cited tables with their columns and sizes, and a manifest of every table
so nothing is hidden.

Out: results/TABLE_CATALOGUE.md and results/TABLE_MANIFEST.csv
"""
from __future__ import annotations

import csv
import glob
import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TAB = f"{ROOT}/outputs/tables"

# Families of tables, keyed by filename prefix, with what the family measures and which claim leans on it.
FAMILIES = [
    ("lit_commit_",      "Forced-answer curves: how often a truncated response's forced answer equals the final one", "1"),
    ("lit_filler_",      "Controls for the truncation result: neutral filler and another item's reasoning",           "1"),
    ("lit_source_",      "Source attribution, standing crossed with track record, and the amplification check",       "2"),
    ("lit_credential_",  "Probe legibility of the credential inside the model",                                        "2"),
    ("lit_routing_",     "Attention knockout by layer and its controls; residual patching",                            "3"),
    ("lit_carrier_",     "Message split into bare answer and reasoning; additivity residuals",                         "4"),
    ("lit_debate_",      "Vote, majority and unanimity outcomes per scope and domain",                                 "5"),
    ("lit_turns_",       "Turn-by-turn switch rate, unanimity, wrong unanimity over twelve turns",                     "5"),
    ("lit_freeze_",      "Share of groups whose last answer change has already happened, by turn",                     "5"),
    ("lit_independence_","Correlation of being right or wrong together, as effective independent voters",              "5"),
    ("response_length_", "Response growth over a long debate, and peer-citation rate",                                 "5"),
    ("lit_frames_",      "Attribution frames: own-label effect and the position control",                              "6"),
    ("concept_validation_", "Which concepts a model validates, with specificity and leakage checks",                  "7, 8"),
    ("concept_null_",    "Permuted-label null for each concept probe",                                                 "7"),
    ("shuffled_topic_auroc_", "How well a direction carrying no concept information separates question types",         "7"),
    ("concept_drift_crossmodel_", "Two fixed moral concepts, drift compared across models and read-out sites",         "7"),
    ("emotion_stories_", "Concept probe fitting and out-of-fold validation on the story corpus",                       "7, 8"),
    ("moral_stories_",   "The same for the moral concept set",                                                          "7, 8"),
    ("traj_domain_",     "Affect trajectories by domain, with shuffled-domain nulls",                                  "8"),
    ("prompt_lengths_",  "Prompt token lengths against the sliding-window size",                                       "3"),
    ("lit_latentx_",     "Cross-model latent transfer and its controls",                                               "-"),
    ("portability",      "Cross-model dispersion of levels against contrasts (methodological)",                        "-"),
]


def family_of(name):
    for pre, what, claim in FAMILIES:
        if name.startswith(pre):
            return what, claim
    return None, None


def main():
    cited = [l.strip() for l in open(sys.argv[1]) if l.strip()] if len(sys.argv) > 1 else []
    cited = [c for c in cited if os.path.exists(f"{TAB}/{c}")]
    allt = sorted(os.path.basename(p) for p in glob.glob(f"{TAB}/*.csv"))

    # manifest of everything
    with open(f"{ROOT}/results/TABLE_MANIFEST.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["table", "cited_by_paper", "rows", "columns", "size_kb", "family", "claim"])
        for n in allt:
            p = f"{TAB}/{n}"
            try:
                d = pd.read_csv(p, nrows=5000)
                rows, cols = len(d), ";".join(map(str, d.columns))
            except Exception as e:
                rows, cols = -1, f"(unreadable: {type(e).__name__})"
            what, claim = family_of(n)
            wr.writerow([n, int(n in cited), rows, cols, round(os.path.getsize(p) / 1024, 1),
                         what or "", claim or ""])

    L = ["# Result tables: what the paper stands on", "",
         f"{len(allt)} tables exist under `outputs/tables/`. **{len(cited)}** of them are read by the claims, "
         f"the figure captions or the figure scripts, and those are the ones shipped in this package. "
         f"`TABLE_MANIFEST.csv` lists all {len(allt)} with columns and sizes so nothing is hidden.", "",
         "Tables are grouped by family below. Within a family the filename encodes the model and the dataset: "
         "`_Q9_`, `_L31_`, `_G4_` for the models, and `full2400`, `core`, `lit1200`, `release_600_t12` for the "
         "dataset. A half-benchmark table (`lit1200`, `lit1200b`) is never quoted beside a pooled one.", ""]

    seen = set()
    for pre, what, claim in FAMILIES:
        members = [n for n in cited if n.startswith(pre)]
        if not members:
            continue
        seen |= set(members)
        L += [f"## `{pre}*`", "", f"{what}", "", f"Backs claim {claim}." if claim != "-" else
              "Methodological; not quoted in a claim.", "",
              "| table | rows | columns |", "|---|---|---|"]
        for n in sorted(members):
            try:
                d = pd.read_csv(f"{TAB}/{n}", nrows=5000)
                L.append(f"| {n} | {len(d)} | {', '.join(map(str, d.columns))[:78]} |")
            except Exception as e:
                L.append(f"| {n} | ? | (unreadable: {type(e).__name__}) |")
        L.append("")

    rest = sorted(set(cited) - seen)
    if rest:
        L += ["## Other cited tables", "",
              "Read by the claims or figures but not part of a named family above.", "",
              "| table | rows | columns |", "|---|---|---|"]
        for n in rest:
            try:
                d = pd.read_csv(f"{TAB}/{n}", nrows=5000)
                L.append(f"| {n} | {len(d)} | {', '.join(map(str, d.columns))[:78]} |")
            except Exception:
                L.append(f"| {n} | ? | (unreadable) |")
        L.append("")

    open(f"{ROOT}/results/TABLE_CATALOGUE.md", "w").write("\n".join(L) + "\n")
    print(f"[done] results/TABLE_CATALOGUE.md ({len(cited)} cited tables in "
          f"{sum(1 for pre, _, _ in FAMILIES if any(n.startswith(pre) for n in cited))} families, "
          f"{len(rest)} ungrouped) and results/TABLE_MANIFEST.csv ({len(allt)} tables)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
