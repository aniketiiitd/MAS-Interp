"""Emit one line per meaningful queue event: job status changes, [done] round lines, errors, stalls."""
import json, time, os, re
from pathlib import Path
L = Path(__file__).resolve().parents[2] / "outputs/logs"
seen_status, offsets, last_progress = {}, {}, time.time()
# start from the current end of every log and the current queue state (no replay of history on re-arm)
for f in L.glob("*.log"):
    offsets[f] = f.stat().st_size
for sf in L.glob("*_state.json"):
    for name, s in json.load(open(sf)).items():
        seen_status[name] = (s.get("status"), s.get("attempts"))
PAT = re.compile(r"^\[(done|complete|skip)\]|Traceback|CUDA out of memory|OutOfMemory|Killed|RuntimeError|Error:|NaN|STAGE", re.I)
while True:
    try:
        st = {}
        for sf in L.glob("*_state.json"):
            try:
                st.update(json.load(open(sf)))
            except Exception:
                pass
        for name, s in st.items():
            key = (s.get("status"), s.get("attempts"))
            if seen_status.get(name) != key:
                seen_status[name] = key
                print(f"QUEUE {name}: {s.get('status')} (attempt {s.get('attempts')}, rc={s.get('returncode')}, {s.get('elapsed_min','-')} min)", flush=True)
                last_progress = time.time()
        for f in L.glob("*.log"):
            if f.name.endswith("daemon.log"):
                continue
            off = offsets.get(f, 0)
            size = f.stat().st_size
            if size < off:
                off = 0
            if size > off:
                with open(f, errors="ignore") as fh:
                    fh.seek(off)
                    for line in fh:
                        if re.match(r"^\[(done|skip)\] (debate|pool)_t\d+", line) or line.startswith("[progress]"):
                            continue  # per-round/progress lines are too chatty; completion and errors still surface
                        if PAT.search(line) and "EngineCore" not in line[:40] or ("ERROR" in line and "EngineCore" in line and "Traceback" in line):
                            print(f"{f.stem}: {line.strip()[:220]}", flush=True)
                            last_progress = time.time()
                offsets[f] = size
        running = [n for n, s in st.items() if s.get("status") == "running"]
        if running and time.time() - last_progress > 3600:
            print(f"STALL? no events for 60 min while running {running}", flush=True)
            last_progress = time.time()
    except Exception as e:
        print("watch error", e, flush=True)
    time.sleep(30)
