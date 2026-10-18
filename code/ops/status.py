"""Queue/infra snapshot for the 2-hourly reports (the narrative and analysis are written by hand on top of this)."""
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
L = ROOT / "outputs/logs"


def main():
    out = [f"## Queue snapshot ({time.strftime('%Y-%m-%d %H:%M UTC')})", ""]
    for q in ("queue", "cpu_queue"):
        jobs = [json.loads(l) for l in open(ROOT / f"configs/{q}.jsonl") if l.strip()]
        st = json.load(open(L / f"{q}_state.json")) if (L / f"{q}_state.json").exists() else {}
        done = [j["name"] for j in jobs if st.get(j["name"], {}).get("status") == "done"]
        failed = [j["name"] for j in jobs if st.get(j["name"], {}).get("status") == "failed"]
        running = [j["name"] for j in jobs if st.get(j["name"], {}).get("status") in ("running", "retry")]
        queued = [j["name"] for j in jobs if j["name"] not in st]
        out.append(f"**{q}**: {len(done)} done, {len(running)} running, {len(queued)} queued, {len(failed)} failed")
        for n in done:
            s = st[n]
            out.append(f"- done `{n}` ({s.get('elapsed_min')} min, attempts {s.get('attempts')})")
        for n in running:
            s = st[n]
            out.append(f"- running `{n}` since {s.get('start')} (attempt {s.get('attempts')})")
        for n in failed:
            out.append(f"- FAILED `{n}` rc={st[n].get('returncode')}")
        if queued:
            out.append(f"- next: {', '.join('`'+n+'`' for n in queued[:8])}{' …' if len(queued) > 8 else ''}")
        out.append("")
    smi = subprocess.run("nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader", shell=True,
                         capture_output=True, text=True).stdout.strip()
    du = shutil.disk_usage(ROOT)
    out.append(f"GPU: {smi}; project filesystem free {du.free/1e9:.0f} GB")
    print("\n".join(out))


if __name__ == "__main__":
    main()
