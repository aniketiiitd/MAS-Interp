"""Does every queued job have inputs that exist, or that something upstream will actually create?

Written after the fourth instance of one failure: a job whose input path is wrong, discovered only when the job
runs — or worse, not discovered, because a wait-gate sat on the wrong path forever and the queue called it
"running". The instances so far:

  1. six Gemma analysis jobs gated on `lit1200` paths while the chain ran `--dataset core`
  2. `LITSRC_G4` pointed at `--dir cptlit_core_n600`, which that model never produced
  3. `FULL_CPTSWAP_G4` given a fit dir covering the whole split, leaving an empty holdout
  4. `LATENTX_Q9_G4` reading `raw_lit/G4/lit1200`, which does not exist because Gemma only ran `core`

The instinct after each was to fix the instance. What connects them is that Gemma runs the whole benchmark in one
pass while Qwen and Llama ran halves, so every path convention built around the two-half layout is wrong for one
model and right for the others.

Case 4 is the one that says why a consistency check is not enough: its job and its wait-gate agreed exactly with
each other, because both were written from the same wrong assumption. Two artifacts written together agree whether
or not they are right. So this resolves paths against the filesystem, which cannot share the mistake.

A missing input is only a bug if nothing upstream creates it, so each miss is classified:
  BROKEN   nothing in any queue writes this path -- it will never appear
  UPSTREAM some pending job produces it; fine, that is the pipeline working

Usage: python ops/check_inputs.py [--verbose]
Exit code is 1 if anything is BROKEN, so it can gate a queue restart.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUEUES = ["queue", "cpu_queue", "cpu2", "cpu3", "cpu4", "cpu5"]

# How each script names the directory it reads, given the flags on the command line. Only conventions that have
# actually bitten are encoded; guessing at others would produce false alarms nobody reads.
def implied_inputs(cmd: str) -> list[tuple[str, str]]:
    """Return (path, why) pairs the command will read, inferred from its flags."""
    out = []
    model = re.search(r"--model\s+(\S+)", cmd)
    sender = re.search(r"--sender\s+(\S+)", cmd)
    receiver = re.search(r"--receiver\s+(\S+)", cmd)
    transcripts = re.search(r"--transcripts\s+(\S+)", cmd)
    dataset = re.search(r"--dataset\s+(\S+)", cmd)
    split = re.search(r"--split\s+(\S+)", cmd)

    # debate transcripts: RAW_LIT/<model>/<dataset>
    if dataset and "latent_cross" in cmd:
        # tm = transcripts or receiver, which is why one direction of a cross-model pair can be right and the
        # other wrong. Getting this wrong in the safe direction would have changed the experiment silently.
        tm = transcripts.group(1) if transcripts else (receiver.group(1) if receiver else None)
        if tm:
            out.append((f"outputs/raw_lit/{tm}/{dataset.group(1)}", f"transcripts for {tm}"))
    elif model and dataset and re.search(r"(interp|analysis)/(lit_|commit_lit|replay_lit|frames_lit|routing_lit|carrier)", cmd):
        out.append((f"outputs/raw_lit/{model.group(1)}/{dataset.group(1)}", "debate transcripts"))

    # There are TWO debate runners writing TWO layouts, and nothing on a consumer's command line says which it
    # wants. `runner/debate_lit.py` writes outputs/raw_lit/<model>/<dataset+tag>; `runner/debate_vllm.py` writes
    # outputs/raw/<model>/<dataset>/t0.jsonl. `interp/replay_acts.py` reads the SECOND, and a job that had the
    # first spent two attempts loading 8B of weights before dying on a missing t0.jsonl. The gate on that job was
    # satisfied and this checker said "0 broken", because it knew the gate and not the script.
    if model and dataset and "replay_acts.py" in cmd:
        out.append((f"outputs/raw/{model.group(1)}/{dataset.group(1)}/t0.jsonl",
                    "debate_vllm episodes (NOT the debate_lit raw_lit layout)"))

    # activation dirs passed explicitly
    for flag in ("dir", "fit_dir", "swap_dir"):
        mm = re.search(rf"--{flag}\s+(\S+)", cmd)
        if mm and model:
            out.append((f"outputs/activations/{model.group(1)}/{mm.group(1)}", f"--{flag}"))

    # a split must correspond to transcripts for that model when the script reads debates
    if model and split and not dataset and "cpt_lit.py" in cmd and " swap" in cmd:
        pass  # cmd_swap reads only the fit dir, already covered above
    return out


def produces(cmd: str) -> list[str]:
    """Union of what every segment of a compound command writes.

    A job often runs a smoke pass and the real pass in one line joined by `&&`. Regexing the whole string takes the
    first `--tag`, which is the smoke one, and the real output goes unaccounted for -- this checker's own first run
    reported a healthy job as BROKEN for exactly that reason. Each segment is its own command; treat it as one.
    """
    return [p for seg in re.split(r"&&|;", cmd) for p in _produces_one(seg)]


def _produces_one(cmd: str) -> list[str]:
    """Directories this command writes, by each script's actual naming line (not a guess).

    Transcribed from the source so the checker and the pipeline cannot drift:
      commit_lit    ACTS/<model>/lit_<mode>_<dataset>[_<tm>text]
      replay_lit    ACTS/<model>/lit_replay_<dataset>[_<tm>text]
      frames_lit    ACTS/<model>/lit_frames[_extra]_<dataset>[_<tm>text]
      routing_lit   ACTS/<model>/routing_attn_<dataset>  |  routing_span_<split>
      emotion_stories  ACTS/<model>/<tag>
      latent_cross  ACTS/<receiver>/lit_latentx_<sender>to<receiver>_<dataset><tag>
      cpt_lit swap  ACTS/<model>/cptlit_swap_<split>
    """
    g = lambda f: (m.group(1) if (m := re.search(rf"--{f}\s+(\S+)", cmd)) else None)
    model, dataset, split, tag = g("model"), g("dataset"), g("split"), g("tag")
    tm, sender, receiver = g("transcripts"), g("sender"), g("receiver")
    suffix = f"_{tm}text" if tm else ""
    out = []
    if "latent_cross.py" in cmd and sender and receiver and dataset:
        out.append(f"outputs/activations/{receiver}/lit_latentx_{sender}to{receiver}_{dataset}{tag or ''}")
    if model:
        A = f"outputs/activations/{model}"
        if "commit_lit.py" in cmd and dataset:
            out.append(f"{A}/lit_{g('mode') or 'commit'}_{dataset}{suffix}")
        if "replay_lit.py" in cmd and dataset:
            out.append(f"{A}/lit_replay_{dataset}{suffix}")
        if "frames_lit.py" in cmd and dataset:
            out.append(f"{A}/lit_frames{'_extra' if '--extra' in cmd else ''}_{dataset}{suffix}")
        if "routing_lit.py" in cmd:
            if dataset:
                out.append(f"{A}/routing_attn_{dataset}")
            if split:
                out.append(f"{A}/routing_span_{split}")
        if "emotion_stories.py" in cmd:
            # the tag defaults by which story set is passed; both spellings are cheap to accept
            out.append(f"{A}/{tag or ('moral_stories' if '--stories' in cmd else 'emotion_stories')}")
        if "cpt_lit.py" in cmd and " swap" in cmd and split:
            out.append(f"{A}/cptlit_swap_{split}")
    # emotion_vectors.py writes tables rather than activation dirs:
    #   fit             -> tables/concept_validation_<model>.csv          (emotion_vectors.py:180)
    #   debate --replay -> tables/concept_traj_<model>_<replay>.parquet   (emotion_vectors.py:210)
    #   observer        -> tables/concept_obs_<model>_<observer>.parquet  (emotion_vectors.py:241)
    if "emotion_vectors.py" in cmd and model:
        if " fit" in cmd:
            out.append(f"outputs/tables/concept_validation_{model}.csv")
        rp = g("replay")
        if " debate" in cmd and rp:
            out.append(f"outputs/tables/concept_traj_{model}_{rp}.parquet")
        ob = g("observer")
        if " observer" in cmd and ob:
            out.append(f"outputs/tables/concept_obs_{model}_{ob}.parquet")
    return out


def produced_by(path: str, jobs: list[dict], asker: dict) -> bool:
    """Will some other job create this path?

    The first version of this asked whether the directory's name or its parent's appeared anywhere in the command
    set. That is far too loose: for `outputs/raw_lit/G4/lit1200` the two tokens are "lit1200" and "G4", both of
    which occur in dozens of unrelated commands, so the checker called the bug it was written for "upstream" and
    exited 0. Token overlap is not production. Match the exact path, or name the producer by its real convention.
    """
    for j in jobs:
        # A job that will never run cannot produce anything. Only `done` was excluded here, so the 69 SKIPPED
        # jobs in the main queue each counted as a live producer, and any input one of them would have written
        # was reported as "waiting on upstream" forever. Found by a negative test whose fixture was a model
        # whose debate job had been skipped months ago.
        if j is asker or j["status"] in ("done", "skipped", "failed"):
            continue
        cmd = j["cmd"]
        if path in cmd:                       # a gate or an explicit output path, matched whole
            return True
        # a file inside a directory this job creates counts as produced
        if any(path == d or path.startswith(d + "/") for d in produces(cmd)):
            return True
        # runner/debate_vllm.py writes outputs/raw/<model>/<dataset>/ -- the layout replay_acts.py reads. Without
        # this the checker flags a job that satisfies its OWN dependency in an earlier segment, which is worse
        # than silence: a false alarm trains you to ignore the real ones.
        m2 = re.match(r"outputs/raw/([^/]+)/([^/]+)(/.*)?$", path)
        if m2 and "debate_vllm.py" in cmd:
            mm2 = re.search(r"--model\s+(\S+)", cmd)
            dd2 = re.search(r"--dataset\s+(\S+)", cmd)
            if mm2 and dd2 and mm2.group(1) == m2.group(1) and dd2.group(1) == m2.group(2):
                return True
        # runner/debate_lit.py writes RAW_LIT/<model>/<dataset+tag>; that is the only producer of transcripts
        m = re.match(r"outputs/raw_lit/([^/]+)/(.+)$", path)
        if m and "debate_lit.py" in cmd:
            mm = re.search(r"--model\s+(\S+)", cmd)
            dd = re.search(r"--dataset\s+(\S+)", cmd)
            tt = re.search(r"--tag\s+(\S+)", cmd)
            # the gate usually names a FILE inside that directory (gate on the last thing written, not the
            # first thing needed), so compare the directory prefix rather than the whole path
            if mm and dd and mm.group(1) == m.group(1):
                d = dd.group(1) + (tt.group(1) if tt else "")
                if m.group(2) == d or m.group(2).startswith(d + "/"):
                    return True
    return False


def gate_paths(cmd: str) -> list[str]:
    """Explicit paths named inside a wait-gate, which are checkable exactly."""
    gate = cmd.split("; do sleep", 1)[0] if "; do sleep" in cmd else ""
    return re.findall(r"\[\s*-[sf]\s+(\S+?)\s*\]", gate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    jobs = []
    for q in QUEUES:
        cf, sf = ROOT / f"configs/{q}.jsonl", ROOT / f"outputs/logs/{q}_state.json"
        if not cf.exists():
            continue
        st = json.load(open(sf)) if sf.exists() else {}
        for line in open(cf):
            if not line.strip():
                continue
            j = json.loads(line)
            j["queue"], j["status"] = q, st.get(j["name"], {}).get("status", "pending")
            jobs.append(j)


    live = [j for j in jobs if j["status"] in ("pending", "running", "retry")]
    print(f"[check_inputs] {len(live)} live jobs across {len(QUEUES)} queues\n")

    broken, upstream = [], []
    for j in live:
        wants = [(p, "wait-gate") for p in gate_paths(j["cmd"])] + implied_inputs(j["cmd"])
        for p, why in wants:
            if (ROOT / p).exists():
                if a.verbose:
                    print(f"  ok       {j['name']:24s} {p}")
                continue
            # A job may satisfy its own dependency in an earlier segment: `debate && replay && analyse` is one
            # line. produced_by() skips the asker by design (so a job cannot vouch for itself across queues),
            # which made a self-contained chain read as BROKEN. Self-production is checked separately and only
            # counts if the producing segment comes BEFORE the consuming one.
            segs = re.split(r"&&|;", j["cmd"])
            consumer = next((i for i, sg in enumerate(segs) if any(pp == p for pp, _ in implied_inputs(sg))), None)
            self_made = consumer is not None and any(
                produced_by(p, [dict(j, cmd=sg, status="pending", name=j["name"] + f"#seg{i}")], {})
                for i, sg in enumerate(segs) if i < consumer)
            if self_made:
                if a.verbose:
                    print(f"  self     {j['name']:24s} {p} (produced by an earlier segment)")
                continue
            (upstream if produced_by(p, jobs, j) else broken).append((j, p, why))

    # Head-of-line blocking. These queues run one job at a time and pick the first that is not done, and a job
    # that waits inside its own process cannot be preempted. So an analysis whose input has not arrived blocks
    # every ready job behind it, for as long as its own input takes. That has happened three times today:
    # LITSRC_G4 ahead of LITFR_G4, TRAJDOM_G4 ahead of CONCEPT_AN_G4, LATENTX_AN_Q9toG4 ahead of AN_G4toQ9.
    # The fix each time was to order analyses by when their inputs arrive, which is what this reports.
    from collections import defaultdict
    per_q = defaultdict(list)
    for j in jobs:
        if j["status"] not in ("done", "failed", "skipped"):
            per_q[j["queue"]].append(j)
    blocks = []
    for qn, js in per_q.items():
        # `a` is the argparse namespace in this function; naming a loop variable `a` here silently clobbered it
        # and the crash surfaced forty lines later at `a.verbose`. Loop names are prefixed.
        for hi, head in enumerate(js):
            head_missing = [p for p in gate_paths(head["cmd"]) if not (ROOT / p).exists()]
            if not head_missing:
                continue
            for tail in js[hi + 1:]:
                gp = gate_paths(tail["cmd"])
                if gp and all((ROOT / p).exists() and (ROOT / p).stat().st_size > 0 for p in gp):
                    blocks.append((qn, head["name"], head_missing[0], tail["name"]))
                    break
    for qn, blocker, waiting_on, blocked in blocks:
        print(f"  BLOCKED  {qn}: {blocker} waits on {waiting_on}")
        print(f"           and sits ahead of {blocked}, whose inputs are ready now")

    for j, p, why in broken:
        print(f"  BROKEN   {j['queue']}/{j['name']:24s} [{why}] {p}")
        print(f"           nothing in any queue produces this path")
    if a.verbose:
        for j, p, why in upstream:
            print(f"  upstream {j['queue']}/{j['name']:24s} [{why}] {p}")

    print(f"\n{len(broken)} broken, {len(upstream)} waiting on upstream jobs")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
