#!/usr/bin/env python3
"""Build only the 2,400-item MADBench v2 core dataset.

This script reuses the pinned source adapters and deterministic sampling logic
from ``build_madbench_v2.py`` but writes only ``madbench_v2_core.jsonl``.
Run it directly; the original builder must remain alongside this file.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path

if __package__:
    from . import build_madbench_v2 as builder
else:
    import build_madbench_v2 as builder


def build_core(input_path: Path, output_path: Path, cache_dir: Path | None) -> dict:
    if cache_dir is None:
        with tempfile.TemporaryDirectory(prefix="madbench-v2-core-") as temporary_dir:
            return _build_core(input_path, output_path, Path(temporary_dir))
    return _build_core(input_path, output_path, cache_dir)


def _build_core(input_path: Path, output_path: Path, cache_dir: Path) -> dict:
    paths, _source_audit = builder.download_sources(cache_dir)
    original_raw = builder.load_jsonl(input_path)
    original_deduped, _duplicate_log = builder.dedupe_by_prompt(original_raw)
    normalized_original = [builder.normalize_existing(record) for record in original_deduped]

    current_by_stratum: dict[str, list[dict]] = defaultdict(list)
    for record in normalized_original:
        current_by_stratum[record["stratum"]].append(record)

    hard_core = builder.proportional_sample(
        current_by_stratum["hard_reasoning"], 310, lambda r: r["source"], "current-hard"
    )
    verifiable_core = builder.proportional_sample(
        current_by_stratum["verifiable"], 150, lambda r: r["source"], "current-verifiable"
    )
    factual_core = list(current_by_stratum["factual_contested"])
    existing_knowledge_core = hard_core + verifiable_core + factual_core
    existing_values_core = (
        current_by_stratum["debate_open"]
        + current_by_stratum["clear_moral"]
        + current_by_stratum["grey_moral"]
    )
    existing_safety_core = current_by_stratum["attack"] + current_by_stratum["refusal_boundary"]

    abstention = builder.build_abstention(paths)
    persuasion = builder.build_persuasion(paths["persuasion"])
    ethics = builder.build_ethics(paths)
    deception = builder.build_deception(paths["deceptionbench"])
    xstest = builder.build_xstest(paths["xstest"])
    core = (
        existing_knowledge_core
        + abstention
        + existing_values_core
        + persuasion
        + ethics
        + existing_safety_core
        + deception
        + xstest
    )

    if (len(existing_knowledge_core), len(existing_values_core), len(existing_safety_core)) != (600, 400, 360):
        raise ValueError("Existing-domain sampling did not produce 600/400/360")
    if [len(abstention), len(persuasion), len(ethics), len(deception), len(xstest)] != [200, 200, 200, 180, 260]:
        raise ValueError("New-source sampling did not produce 200/200/200/180/260")

    # Check cross-source duplicates and add the same membership field as the
    # full builder so the resulting core rows have an identical schema.
    all_records, duplicate_log = builder.dedupe_by_prompt(normalized_original + abstention + persuasion + ethics + deception + xstest)
    removed_ids = {entry["removed_id"] for entry in duplicate_log}
    if removed_ids:
        raise ValueError(f"New-source prompt duplicates detected: {sorted(removed_ids)}")

    core_ids = {record["id"] for record in core}
    builder.assign_membership(all_records, core_ids)
    by_id = {record["id"]: record for record in all_records}
    core = [by_id[record_id] for record_id in sorted(core_ids)]
    core = builder.shuffled(core, "core-output")

    expected_domains = {
        "knowledge_reasoning": 800,
        "safety_refusal": 800,
        "values_moral_judgment": 800,
    }
    summary = builder.validate(core, 2400, expected_domains)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    builder.write_jsonl(output_path, core)
    return summary


def main() -> None:
    default_dataset_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_dataset_dir / "madbench.jsonl")
    parser.add_argument("--output-dir", type=Path, default=default_dataset_dir)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    output_path = args.output_dir / "madbench_v2_core.jsonl"
    summary = build_core(args.input, output_path, args.cache_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
