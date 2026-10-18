"""Release page cache for large files we own (model weights already on GPU, written activation memmaps).
Clean pages are dropped immediately; dirty pages are fsync'ed first. Safe for running readers/writers (pages refault on demand)."""
import os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
HF_HOME = Path(os.environ.get("HF_HOME", ROOT / ".cache" / "huggingface")).expanduser()
if not HF_HOME.is_absolute():
    HF_HOME = ROOT / HF_HOME
ROOTS = [ROOT / "outputs/activations"]
HUB = HF_HOME / "hub"  # dropped only once the running GPU job is past its model load (mid-load drops made loads 10x slower)


def hub_safe():
    import json
    try:
        st = json.load(open(ROOT / "outputs/logs/queue_state.json"))
        run = [v for v in st.values() if v.get("status") == "running"]
        if not run:
            return True
        t = time.mktime(time.strptime(run[0]["start"], "%Y-%m-%d %H:%M:%S"))
        return time.time() - t > 600
    except Exception:
        return False
MIN = 256 * 1024 * 1024

def drop():
    n = 0
    for root in ROOTS + ([HUB] if hub_safe() else []):
        for p in root.rglob("*"):
            try:
                if p.is_symlink() or not p.is_file() or p.stat().st_size < MIN:
                    continue
                fd = os.open(p, os.O_RDONLY)
                try:
                    os.fsync(fd) if os.access(p, os.W_OK) else None
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    n += 1
                finally:
                    os.close(fd)
            except Exception:
                pass
    return n

if __name__ == "__main__":
    loop = len(sys.argv) > 1 and sys.argv[1] == "loop"
    while True:
        n = drop()
        if not loop:
            print("dropped", n); break
        time.sleep(180)
