"""Sequential GPU job queue. Jobs are appended to configs/queue.jsonl ({"name","cmd","retries","timeout_h"}).
State lives in outputs/logs/queue_state.json. The job file is re-read after every job, so jobs can be appended live.
Each job runs in its own process group, which is killed after the job ends (frees GPU memory from stray workers)."""
import json, os, signal, subprocess, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
QNAME = sys.argv[1] if len(sys.argv) > 1 else "queue"
QF, SF, LOGS = ROOT / f"configs/{QNAME}.jsonl", ROOT / f"outputs/logs/{QNAME}_state.json", ROOT / "outputs/logs"
# There is no argument parsing here, so any word becomes a queue name. `jobqueue.py --help` and `jobqueue.py status`
# each started a daemon for a queue that does not exist and looped on an empty job list for over an hour, which
# looks exactly like a hung command. A missing config is a typo, not an empty queue.
if not QF.exists():
    raise SystemExit(f"[jobqueue] no such queue {QNAME!r}: {QF} does not exist. "
                     f"Queues are: {', '.join(sorted(p.stem for p in (ROOT / 'configs').glob('*.jsonl')))}")
env = dict(os.environ)
env.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
env.setdefault("HF_TOKEN_PATH", str(Path(env["HF_HOME"]) / "token"))
env["PYTHONUNBUFFERED"] = "1"
def log(msg):
    print(time.strftime("[%H:%M:%S] "), msg, flush=True)
def load_state():
    return json.load(open(SF)) if SF.exists() else {}
def save_state(s):
    tmp = SF.with_suffix(".tmp"); json.dump(s, open(tmp, "w"), indent=1); os.replace(tmp, SF)
def run_job(j):
    state = load_state()
    st = state.get(j["name"], {"attempts": 0})
    st.update(status="running", start=time.strftime("%Y-%m-%d %H:%M:%S"), cmd=j["cmd"]); st["attempts"] += 1
    state[j["name"]] = st; save_state(state)
    t = time.time()
    with open(LOGS / f"{j['name']}.log", "a") as lf:
        lf.write(f"\n===== attempt {st['attempts']} {st['start']} =====\n"); lf.flush()
        p = subprocess.Popen(["bash", "-c", j["cmd"]], cwd=ROOT / "code", env=env, stdout=lf, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        st["pid"] = p.pid; state = load_state(); state[j["name"]] = st; save_state(state)
        log(f"started {j['name']} pid={p.pid}")
        try:
            rc = p.wait(timeout=j.get("timeout_h", 12) * 3600)
        except subprocess.TimeoutExpired:
            rc = -999
    try:
        os.killpg(p.pid, signal.SIGTERM); time.sleep(5); os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    state = load_state(); st = state[j["name"]]
    st.update(elapsed_min=round((time.time() - t) / 60, 1), returncode=rc, end=time.strftime("%Y-%m-%d %H:%M:%S"))
    st["status"] = "done" if rc == 0 else ("retry" if st["attempts"] <= j.get("retries", 1) else "failed")
    state[j["name"]] = st; save_state(state)
    log(f"finished {j['name']} rc={rc} status={st['status']} {st['elapsed_min']}min")
def alive(pid):
    try:
        os.kill(pid, 0); return True
    except Exception:
        return False
# adopt jobs left "running" by a previous daemon: wait for them, then mark for (resumable) retry
for name, st in load_state().items():
    if st.get("status") == "running" and st.get("pid"):
        log(f"adopting running job {name} pid={st['pid']}")
        while alive(st["pid"]):
            time.sleep(30)
        state = load_state(); state[name]["status"] = "retry"; save_state(state)
        log(f"adopted job {name} exited; marked retry (runner resumes finished rounds)")
while True:
    try:
        if (ROOT / f"configs/HOLD_{QNAME}").exists() or (QNAME == "queue" and (ROOT / "configs/HOLD").exists()):
            time.sleep(15); continue
        state = load_state()
        jobs = [json.loads(l) for l in open(QF) if l.strip()] if QF.exists() else []
        todo = [j for j in jobs if state.get(j["name"], {}).get("status") not in ("done", "failed", "skipped")]
        if not todo:
            time.sleep(20); continue
        run_job(todo[0])
        time.sleep(10)
    except Exception:
        log(traceback.format_exc()); time.sleep(30)
