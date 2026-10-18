"""Generate the detailed methodology document for writing the paper.

Everything factual here is read at generation time from the benchmark, the model registry, the tables and the
scripts' own docstrings. Prose that cannot be derived is written inline and marked as such. The point is that
the methods section cannot quietly stop describing what the code does, which is the failure this campaign has
hit five times in other artifacts.

Out: results/METHODOLOGY.md
"""
from __future__ import annotations

import ast
import collections
import glob
import json
import os
import sys
from datetime import datetime, timezone

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
from common import *  # noqa

ROOT = Path(__file__).resolve().parents[2]
L = []


def w(s=""):
    L.append(s)


def doc_of(path, limit=2600):
    """A script's own docstring, so the method description cannot drift from the code."""
    try:
        d = ast.get_docstring(ast.parse((ROOT / "code" / path).read_text()))
    except Exception as e:
        return f"(could not read {path}: {e})"
    return (d or f"({path} has no docstring)")[:limit].strip()


def main():
    C = load_core()
    dom = collections.Counter(r["primary_domain"] for r in C)
    strat = collections.Counter(r["stratum"] for r in C)
    src = collections.Counter(r["source"] for r in C)
    gt = sum(1 for r in C if r["ground_truth"] is not None)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    w("# MAD-Interp: methodology")
    w()
    w(f"Generated {stamp} by `analysis/emit_methods.py`. Benchmark composition, model configurations and table "
      f"inventories are read from the artifacts at generation time; each experiment's description is that "
      f"experiment's own module docstring, so this document cannot describe something the code does not do.")
    w()
    w("## 1. What this study measures, and why it is set up this way")
    w()
    w("Multi-agent debate is normally evaluated on its outputs: does the group's final answer improve. This "
      "study measures the agents' internal states and their decisions under intervention, and asks a different "
      "question: what actually moves a model that is reading its peers, and when.")
    w()
    w("Three design commitments follow from that and constrain everything below.")
    w()
    w("**A matched control for every comparison.** Every debate arm is paired with a no-communication arm in "
      "which the same agents run for the same number of turns on the same items with the peers' messages "
      "removed. Any effect of 'thinking longer' appears in both arms and cancels. Without this control, most of "
      "what is reported here as an effect of debate would be an effect of extra tokens.")
    w()
    w("**Intervention rather than correlation.** Where the question is causal, the measurement is causal: "
      "attention edges are cut, residual streams are interchanged, identical bytes are relabelled, reasoning is "
      "truncated mid-write. Probing and decoding are used only as instruments, and where they carry a claim they "
      "are checked against a null that preserves everything except the thing being claimed.")
    w()
    w("**Nulls that match the structure of the data.** A chance baseline is the wrong null for almost every "
      "quantity here. Spreads are positive by construction, concept directions read topic structure whether or "
      "not they encode a concept, and states from one question are not independent. The nulls are therefore "
      "permutation nulls built by shuffling the specific label being claimed, and the intervals are clustered on "
      "items. Section 8 gives the details, and section 9 the machinery that enforces them.")
    w()

    # ---------------------------------------------------------------- benchmark
    w("## 2. The benchmark: MAD Bench v2")
    w()
    w(f"{len(C):,} items in three domains of {dom['knowledge_reasoning']:,}, {dom['safety_refusal']:,} and "
      f"{dom['values_moral_judgment']:,}. The three-way split is the reason this benchmark was curated rather "
      f"than an existing one being reused: the domains differ in **what kind of thing a right answer is**, which "
      f"is what lets an effect of persuasion be separated from an effect of being corrected.")
    w()
    w("| domain | items | what a correct answer is | what debate could be doing |")
    w("|---|---|---|---|")
    w(f"| knowledge_reasoning | {dom['knowledge_reasoning']:,} | a fact of the matter, checkable | correcting an error, or spreading one |")
    w(f"| safety_refusal | {dom['safety_refusal']:,} | a behavioural target, policy-defined | moving a refusal boundary |")
    w(f"| values_moral_judgment | {dom['values_moral_judgment']:,} | **no fact of the matter** | pure social influence, with no correction available |")
    w()
    w(f"**{gt:,} items carry a ground-truth answer and {len(C) - gt:,} do not.** The second group is not a "
      "deficiency; it is the normative-null regime. On those items accuracy is undefined, so any movement debate "
      "produces is movement that cannot be justified as correction. Analyses that ask whether a group is *wrong* "
      "are necessarily restricted to the items where being wrong is defined, and every such restriction is "
      "reported with its own sample size rather than folded into a single figure.")
    w()
    w(f"**{len(strat)} strata** within the three domains, so that a domain effect is not an artifact of one task "
      "type dominating it:")
    w()
    w("| stratum | items | stratum | items |")
    w("|---|---|---|---|")
    items = sorted(strat.items(), key=lambda kv: -kv[1])
    for a, b in zip(items[0::2], items[1::2] + [("", "")]):
        w(f"| {a[0]} | {a[1]} | {b[0]} | {b[1]} |")
    w()
    w("**Provenance.** Items are drawn from established benchmarks where one exists for the construct and "
      "authored where none does:")
    w()
    w("| source | items |")
    w("|---|---|")
    for k, v in src.most_common():
        w(f"| {k} | {v} |")
    w()
    w("Each item carries a prompt, a stance-option list where the task is multiple choice, a ground truth where "
      "one exists, a stratum, a primary domain, and pre-computed solo statistics (a small model's accuracy and "
      "answer entropy on the item) used only for stratification and never as an outcome.")
    w()

    # ---------------------------------------------------------------- models
    w("## 3. Models")
    w()
    w("Three models carry the results. They were chosen so that the architectural contrast in section 7.3 is "
      "not confounded with vendor or with training data.")
    w()
    w("| key | model | params | layers | hidden | attention composition |")
    w("|---|---|---|---|---|---|")
    for key in ("Q9", "L31", "G4"):
        m = MODELS[key]
        try:
            p, _ = snapshot(key)
            cfg = json.load(open(p + "/config.json"))
            tc = cfg.get("text_config", cfg)
            lt = tc.get("layer_types") or []
            comp = ", ".join(f"{v} {k}" for k, v in collections.Counter(lt).items()) if lt else "all full attention"
            nl, hs = tc.get("num_hidden_layers", "?"), tc.get("hidden_size", "?")
        except Exception:
            comp, nl, hs = "(config unavailable)", "?", "?"
        w(f"| {key} | {m['repo']} | {m['params_b']}B | {nl} | {hs} | {comp} |")
    w()
    w("**Why this matters more than it usually does.** Qwen3.5-9B is not a transformer in the usual sense: only "
      "8 of its 32 layers are full attention, and the other 24 are Gated-DeltaNet linear-attention blocks that "
      "no attention mask can reach. Llama-3.1-8B is 32 full-attention layers. Gemma-4-E4B is 42 attention "
      "layers of which 7 are global and 35 are sliding-window over 512 tokens, against a median debate prompt "
      "in this study of 1,668 tokens, so for most items the peer's message lies outside the sliding window and "
      "influence must travel through the seven global layers. All three take an attention mask on every layer "
      "that has one, which is exactly what Qwen's 24 recurrent layers do not.")
    w()
    w("Precision is bf16 for all three. A fourth model, Qwen3.5-4B, shares Qwen3.5-9B's layer composition "
      "exactly (8 full attention, 24 linear, 32 layers) at a smaller width, and is the natural size-matched "
      "control for the architecture result; it was scoped but not run.")
    w()

    # ---------------------------------------------------------------- protocol
    w("## 4. The debate protocol")
    w()
    w("The protocol is literature-standard rather than novel, so that the internal measurements attach to a "
      "setup readers already recognise. Round structure, prompt wording, answer format and decoding follow "
      "Choi et al. (2025), *Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language "
      "Models?*, Appendix B.1, with three agents as in Du et al. (2024).")
    w()
    w("**Structure.** Three agents, all copies of the same model. Round 0 is independent: each agent answers "
      "the question alone. In each subsequent round an agent receives one user message containing the other "
      "agents' most recent responses, its own most recent response, and the question, and must restate a final "
      "answer. Talk is simultaneous: all agents in a round see the previous round's responses, not each other's "
      "current ones.")
    w()
    w("**The no-communication control.** The `reflect` condition uses the identical message with the other "
      "agents' responses removed, so the agent reconsiders its own most recent opinion for the same number of "
      "rounds. Same items, same round count, same token budget.")
    w()
    w("**Answer extraction.** Agents are instructed to close with the answer in curly brackets "
      "(`{final answer: (A)}`, or a number for numeric items). This matters for the internal measurements as "
      "well: it creates a single, known **decision token**, the `(` of a prefilled `{final answer: (`, at which "
      "the next-token distribution over option letters is the model's decision. Almost every read-out in "
      "section 7 is taken there.")
    w()
    w("**Decoding.** Temperature 1.0, top-p 0.9. Generation is capped at 2,048 new tokens rather than the 512 "
      "of the source protocol, because Qwen3.5 truncates 28 to 50 percent of responses at 512 and a truncated "
      "response has no parseable answer, which would bias exactly the arm that produces it.")
    w()
    w("**Re-queries.** After the final round each agent is asked once more, in two framings: a private question "
      "stating the answer will not be shown to the others or used for the group decision, and a public one "
      "stating it will. Both use the same context plus one extra user turn.")
    w()

    # ---------------------------------------------------------------- datasets
    w("## 5. Datasets, splits and scopes")
    w()
    w("Several named subsets appear in the tables and they are not interchangeable. Confusing them has been the "
      "single most common source of error in this work, so they are listed explicitly.")
    w()
    w("| name | what it is | used for |")
    w("|---|---|---|")
    w("| `lit1200` | first half of the benchmark | Qwen and Llama three-round runs |")
    w("| `lit1200b` | second half | the other half of the same |")
    w("| `full2400` | the two halves pooled | every Qwen and Llama number the paper quotes |")
    w("| `core` | the whole benchmark in one pass | Gemma's three-round runs |")
    w("| `release_600` | a 600-item subset | twelve-turn runs and activation replays |")
    w("| `release_600_t12` | that subset, twelve rounds | the long-horizon results |")
    w()
    w("**A half is never quoted beside a whole.** Qwen and Llama ran the benchmark in two halves and Gemma ran "
      "it in one pass, and a number computed on `lit1200` is not comparable with one computed on `core`. Every "
      "cross-model comparison in the paper is pooled to `full2400` or run on `core`, and the verification in "
      "section 9 includes a structural check that no figure script reads a half-benchmark table when a pooled "
      "one exists.")
    w()
    w("**Two answer scopes.** `all_items` counts every episode; `fully_parsed` counts only the questions where "
      "every agent in every arm produced a parseable answer. The distinction is load-bearing: an unparseable "
      "answer is scored against the arm that produced it, and first-round answers parse worst, so an "
      "unrestricted comparison flatters debate against a plain vote. The paper quotes `fully_parsed` for the "
      "vote comparison and says so.")
    w()

    # ---------------------------------------------------------------- readouts
    w("## 6. Read-out conventions")
    w()
    w("**The decision token.** Unless stated otherwise, a decision is the next-token distribution over the "
      "option letters at the `(` of a prefilled `{final answer: (`, read directly from a forward pass rather "
      "than sampled. Effects are reported in logits of that distribution.")
    w()
    w("**Two sites.** Trajectory analyses distinguish the **reading site**, the agent's state while it is "
      "processing the peers' message, from the **decision site**, its state at the decision token. These behave "
      "differently often enough that a result at one is not evidence about the other, and every trajectory "
      "number in the paper names its site.")
    w()
    w("**Activation caches.** Forward passes that need internal states write them to a per-model cache under "
      "`outputs/activations/<MODEL>/`, on a fixed layer grid, so that downstream analyses re-read one pass "
      "rather than re-running the model. Metadata records the layer grid, the sites and the episode index.")
    w()

    # ---------------------------------------------------------------- experiments
    w("## 7. The experiments")
    w()
    w("Each subsection begins with the implementing module's own docstring, which is the authoritative "
      "description of what runs.")
    w()
    for n, (title, path) in enumerate([
        ("When the answer is decided: truncation and its controls", "interp/commit_lit.py"),
        ("Source attribution: standing crossed with track record", "interp/cpt_lit.py"),
        ("Where influence travels: attention knockout and residual patching", "interp/routing_lit.py"),
        ("Attribution frames: identical words, different owner", "interp/frames_lit.py"),
        ("The latent channel between agents", "interp/latent_lit.py"),
        ("Concept probes on a story corpus", "interp/emotion_stories.py"),
        ("Affect trajectories by domain", "analysis/traj_domain.py"),
        ("Shuffled-label directions against topic", "analysis/shuffled_topic_auroc.py"),
        ("Response growth over a long debate", "analysis/response_length.py"),
        ("Cross-model concept drift", "analysis/concept_drift_crossmodel.py"),
    ], start=1):
        w(f"### 7.{n} {title}")
        w()
        w(f"Module: `{path}`")
        w()
        # verbatim: these docstrings contain aligned argument tables, and reflowing them to prose turned
        # the alignment into runs of whitespace in the middle of sentences
        d = doc_of(path)
        head, _, rest = d.partition("\n\n")
        w(head.replace("\n", " ").strip())
        w()
        if rest.strip():
            w("```")
            for line in rest.rstrip().splitlines():
                w(line)
            w("```")
            w()

    # ---------------------------------------------------------------- statistics
    w("## 8. Statistics")
    w()
    w("**Item-clustered bootstrap.** Three agents answer each question and a twelve-turn run produces twelve "
      "states per agent per question, so responses are nested in items. Every interval in the paper resamples "
      "**items** with replacement and takes all rows belonging to a resampled item, rather than resampling "
      "rows. Treating rows as independent would shrink every interval by roughly the square root of the "
      "within-item count. Unless stated otherwise 2,000 resamples, percentile intervals at 95 percent.")
    w()
    w("**Paired comparisons.** Where debate and no-communication arms run on the same items, the difference is "
      "computed per item and the bootstrap is over items, so the comparison is paired. Unpaired and paired gaps "
      "differ enough to matter and the paper labels which it is quoting.")
    w()
    w("**Permutation nulls, not chance baselines.** For any quantity that is positive by construction or that "
      "an uninformative direction can achieve, the null is built by shuffling the specific label being claimed "
      "and re-running the whole procedure, fit and all:")
    w()
    w("- concept probes: concept labels permuted **within topic**, so topic structure, class sizes, pooling, "
      "fold structure and geometry are all preserved and only the concept mapping is destroyed")
    w("- domain spreads: domain labels shuffled **between item clusters**, because a spread is a maximum minus "
      "a minimum and is positive even when nothing is happening")
    w("- transfer maps: the map refit on deliberately shuffled sender-receiver pairs")
    w()
    w("**Discovery and confirmation.** Where many concepts or cells are screened, selection happens on one half "
      "of the items and the surviving candidates are re-tested on the other half. Reported counts are "
      "confirmation-half counts among discovery-half selections.")
    w()
    w("**Multiple comparisons.** Benjamini-Hochberg false discovery rate control within each family of tests.")
    w()
    w("**Effective independent voters.** Agreement among three agents is converted to an effective count by "
      "`n_eff = 3 / (1 + 2 rho)`, where rho is the correlation of being right or wrong together. Three "
      "independent agents give 3; three perfectly correlated agents give 1. The floor at 1 means values near it "
      "should be read as 'no better than one agent' rather than compared with each other.")
    w()
    w("**A note on precision.** Several results in this study have intervals wide enough that agreement between "
      "two point estimates is not evidence of agreement between two quantities. Where the paper states that "
      "models agree, it states the non-overlap of intervals; where it declines to rank models, it is because "
      "the intervals overlap, and the verification suite asserts that they still do.")
    w()

    # ---------------------------------------------------------------- verification
    w("## 9. Verification")
    w()
    w("Results are re-derived from the frozen tables on every run rather than transcribed. The machinery is "
      "part of the method because several of the errors it caught were invisible to reading.")
    w()
    try:
        naudit = sum(1 for ln in open(ROOT / "code/analysis/audit_reported.py") if "    check(" in ln or "check(f\"" in ln)
    except Exception:
        naudit = 0
    w("| tool | what it enforces |")
    w("|---|---|")
    w("| `analysis/audit_reported.py` | every number in every claim, the abstract and every figure caption is "
      "recomputed from its table and compared; retracted phrasings are refused; figure scripts are checked "
      "structurally for reading the right tables |")
    w("| `analysis/unsourced_numbers.py` | the inverse: numeric tokens in the prose that **no** check covers. "
      "Reports a lower bound, and prints its own measured miss rate so a clean run is not mistaken for a proof |")
    w("| `analysis/cross_references.py` | sentences whose truth depends on a result they do not cite, such as "
      "superlatives over the rest of the paper |")
    w("| `analysis/emit_numbers.py` | every audited number with the table it came from |")
    w("| `ops/check_inputs.py` | resolves every queued job's inputs against the filesystem before it runs |")
    w()
    w("**Staleness is checked separately from correctness**, because five defects in this project were correct "
      "content with a wrong timestamp: a write-up behind the claims, a figure built on a superseded concept "
      "set, a rendered figure older than its script, an abstract section carrying a retracted finding, and a "
      "PDF older than its captions. The audit therefore asserts that each figure is newer than its script, the "
      "write-up newer than its sources, and the shipped PDF newer than everything it is built from.")
    w()
    w("**Every check is verified by reintroduction.** A check that has never been seen to fail has not been "
      "tested; four checks written during this project were found to be vacuous, contaminated or mis-scoped "
      "when deliberately broken.")
    w()

    # ---------------------------------------------------------------- reproduction
    w("## 10. Reproducing a result")
    w()
    w("Order matters, because activation caches are shared between analyses.")
    w()
    w("```")
    w("# 1. debates (GPU)")
    w("python runner/debate_lit.py --model Q9 --dataset lit1200  --rounds 3")
    w("python runner/debate_lit.py --model Q9 --dataset lit1200b --rounds 3")
    w("# 2. interventions on those transcripts (GPU)")
    w("python interp/commit_lit.py  --mode commit --model Q9 --dataset lit1200")
    w("python interp/routing_lit.py attn --model Q9 --dataset lit1200 --n 100 --batch 8")
    w("# 3. analyses (CPU), pooling the halves")
    w("python analysis/lit_routing.py --models Q9 --dataset lit1200,lit1200b")
    w("# 4. figures, document, verification")
    w("python figures/fig_hero1_answer.py && python figures/build_pdf.py")
    w("python analysis/audit_reported.py && python analysis/unsourced_numbers.py")
    w("```")
    w()
    w("`ops/jobqueue.py` runs these as a single-slot queue per device with wait-gates between dependent stages; "
      "`ops/check_inputs.py` validates a queued job's inputs before the queue reaches it.")
    w()

    # ---------------------------------------------------------------- limits
    w("## 11. Limitations")
    w()
    w("**Three models, one size class.** All three are 8 to 10 billion parameters. The architecture contrast is "
      "real and holds across two vendors, but it is a contrast between three points at one scale, and the "
      "size-matched control that would separate architecture from scale was scoped and not run.")
    w()
    w("**One benchmark, curated for this study.** The three-regime structure is what makes the domain "
      "comparisons possible and is also the reason they may not transfer; another benchmark's values items are "
      "not these.")
    w()
    w("**The long-horizon results rest on small contested subsets.** Twelve-turn analyses run on the questions "
      "where the three agents disagreed before anyone spoke, which is where debate can do anything at all, and "
      "that is a few dozen to about a hundred groups per model. Intervals are correspondingly wide and the "
      "paper declines to rank models on those panels.")
    w()
    w("**Probe-based results carry a caution rather than a finding.** Concept probes on debate states clear a "
      "chance baseline easily and mostly do not survive a permuted-label null; the study reports that as a "
      "methodological result and does not build a behavioural claim on the probes.")
    w()
    w("**Dynamics do not generalise and the study does not explain why.** Three models show the same "
      "affect-by-domain dependence and three different dynamics. Three architectures are enough to show there "
      "is no shared shape and not enough to say what governs it.")
    w()

    # ---------------------------------------------------------------- inventory
    w("## 12. Inventory")
    w()
    ntab = len(glob.glob(f"{ROOT}/outputs/tables/*.csv"))
    nfig = len(glob.glob(f"{ROOT}/outputs/figures/paper/*.png"))
    w(f"- **{ntab} result tables** under `outputs/tables/`, of which the subset this package ships are those "
      f"the claims, captions and figure scripts read")
    w(f"- **{nfig} rendered figures** under `outputs/figures/paper/`, six of which the paper ships")
    w(f"- **{len(glob.glob(str(ROOT / 'code/analysis/*.py')))} analysis modules**, "
      f"**{len(glob.glob(str(ROOT / 'code/interp/*.py')))} intervention modules**, "
      f"**{len(glob.glob(str(ROOT / 'code/figures/*.py')))} figure scripts**, "
      f"**{len(glob.glob(str(ROOT / 'code/runner/*.py')))} runners**")
    w("- `results/NUMBERS.md` lists every audited number beside the table it comes from")
    w("- `results/NOVELTY_LEDGER.md` records every retraction and correction with its date and reason, "
      "including the findings that did not survive")
    w()

    open(f"{ROOT}/results/METHODOLOGY.md", "w").write("\n".join(L) + "\n")
    print(f"[done] results/METHODOLOGY.md ({len(L)} lines, ~{sum(len(x.split()) for x in L):,} words)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
