# MAD-Interp research code

This repository contains the analysis and experiment scripts for the MAD-Interp paper, plus the 2,400-item MADBench v2 core dataset used by those scripts. It is a research-code package, not a complete reproduction bundle: model checkpoints, generated activations and responses, configuration files, and derived results are not included.

## Repository layout

- `code/analysis/` — analysis and verification scripts.
- `code/figures/` — figure generation scripts.
- `code/interp/` — interpretability and intervention experiments.
- `code/runner/` — debate, generation, and judging runners.
- `code/ops/` — local job and experiment utilities.
- `madbench/madbench_v2_core.jsonl` — the core benchmark used by the paper.
- `madbench/scripts/` — benchmark build utilities.

Other top-level folders and local runtime data are intentionally excluded from Git by `.gitignore`; they are left untouched on disk.

## Setup

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For model-inference scripts, install the optional `inference` dependencies in an environment compatible with your GPU and CUDA setup. vLLM and PyTorch installation requirements vary by platform; use the appropriate upstream installation instructions if the optional install does not match your machine:

```bash
python -m pip install -e ".[inference]"
```

## Paths and local files

Scripts derive the repository root from their own location, so they can be run from any working directory. Generated experiment outputs go under `outputs/`. The Hugging Face cache defaults to `.cache/huggingface/` inside the repository and is ignored by Git; set `HF_HOME` to use a different cache location. Paths supplied through `HF_HOME` may be absolute or relative to the repository root.

Full experiments also expect local assets under `data/` and `configs/` (for example split definitions and prompt/configuration files), as well as model weights accessible through Hugging Face. These are not part of this public source package. Keep credentials in environment variables or a local, ignored `.env` file; never put API keys or tokens in source files.

## Core-data scope

The experiment code uses `madbench_v2_core.jsonl`. The former `load_extended()` path and the `extended_extra` runner mode have been removed. The benchmark builder may still contain utilities for rebuilding upstream-derived data; the paper analysis scripts do not load an extended benchmark.

The core JSONL records include source and license metadata. Review the licenses and attribution requirements for all upstream data before redistributing the dataset publicly.

## Quick checks

```bash
python code/analysis/behavior.py --help
python code/runner/debate_vllm.py --help
python madbench/scripts/build_madbench_v2_core.py --help
```

The inference runners require a compatible GPU stack; the help commands do not launch a model.
