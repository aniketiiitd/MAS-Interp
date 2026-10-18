"""Block until the GPU is actually free, then exit 0.

The queue daemon SIGTERMs a finished job's process group and waits 5 s before starting the next one. Releasing 40+ GB
takes longer than that, so a job that follows a large one -- or the automatic retry of a job that just OOMed -- can start
while the card is still full and die immediately with a misleading OOM. Prepending this to a GPU job's command removes
that failure mode. It never fails the job: on timeout it gives up and lets the job try anyway.
"""
from __future__ import annotations

import subprocess
import sys
import time

FREE_MB = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
TIMEOUT = int(sys.argv[2]) if len(sys.argv) > 2 else 240


def used_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return 0


t0 = time.time()
while time.time() - t0 < TIMEOUT:
    u = used_mb()
    if u <= FREE_MB:
        if time.time() - t0 > 1:
            print(f"[gpuwait] free after {time.time()-t0:.0f}s ({u} MiB in use)", flush=True)
        sys.exit(0)
    time.sleep(5)
print(f"[gpuwait] timed out after {TIMEOUT}s with {used_mb()} MiB still in use; starting anyway", flush=True)
sys.exit(0)
