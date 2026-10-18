"""Report the answer-parse rate of a debate run as soon as its first round lands.

A replication whose prompts or chat template do not suit a new model shows up first as a collapsed parse rate. Q9's
baseline is 97.1% at round 0 and 99.6% by round 3, so anything far below that means the chain is producing unusable
transcripts and should be stopped rather than left to run for hours.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, read_jsonl

model, dataset = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "lit1200")
f = ROOT / "outputs/raw_lit" / model / dataset / "r0.jsonl"
while not f.exists():
    time.sleep(60)
time.sleep(120)  # let the file finish being written
rows = [x for x in read_jsonl(f)]
ok = sum(1 for x in rows if x.get("answer") is not None)
pct = 100 * ok / max(1, len(rows))
empty = sum(1 for x in rows if not (x.get("text") or "").strip())
print(f"[parsecheck] {model} {dataset} r0: parsed {ok}/{len(rows)} = {pct:.1f}%  (empty texts {empty})", flush=True)
print(f"[parsecheck] Q9 baseline was 97.1% at r0", flush=True)
if pct < 85:
    print(f"[parsecheck] *** PARSE RATE COLLAPSED for {model} -- inspect the prompts/chat template before the chain continues ***", flush=True)
    for x in rows[:3]:
        print("  sample:", repr((x.get("text") or "")[:300]), flush=True)
    sys.exit(3)
