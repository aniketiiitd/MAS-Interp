"""One command: rebuild every analysis table and figure from frozen raw outputs (skips steps whose inputs are missing)."""
import subprocess
import sys
import time
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "outputs/activations"
R = ROOT / "outputs/raw"
T = ROOT / "outputs/tables"
PY = sys.executable

STEPS = [
    # (name, required input path, command)
    ("behavior Q9 core", R / "Q9/core/behavior.jsonl", "analysis/behavior.py --model Q9 --dataset core"),
]
for m in ["Q4", "G4", "Q27", "Q35", "G31", "G26", "G12", "L31", "M7", "O7", "P4"]:
    STEPS.append((f"behavior {m} release", R / f"{m}/release_600/behavior.jsonl", f"analysis/behavior.py --model {m} --dataset release_600"))
STEPS += [
    ("behavior Q4 original", R / "Q4/original/debate_t11.jsonl", "analysis/behavior.py --model Q4 --dataset original"),
    ("cpt Q9", A / "Q9/cpt_core/meta.json", "analysis/cpt_analysis.py --model Q9 --run cpt_core"),
    ("cpt Q4", A / "Q4/cpt_core/meta.json", "analysis/cpt_analysis.py --model Q4 --run cpt_core"),
    ("cpt G4", A / "G4/cpt_core/meta.json", "analysis/cpt_analysis.py --model G4 --run cpt_core"),
    ("cpt Q9 hero all templates", A / "Q9/cpt_hero_48_alltpl/meta.json", "analysis/cpt_analysis.py --model Q9 --run cpt_hero_48_alltpl"),
]
for m in ["Q9", "Q27", "Q35", "G31", "G26", "G12", "L31", "M7", "O7", "P4"]:
    STEPS.append((f"cpt vllm {m}", A / f"{m}/cpt_vllm_release_600/worlds.jsonl", f"analysis/cpt_analysis.py --model {m} --run cpt_vllm_release_600"))
STEPS += [
    ("patch Q9", A / "Q9/patch_hero_48/meta.json", "analysis/patch_analysis.py --model Q9"),
    ("patch Q4", A / "Q4/patch_hero_48/meta.json", "analysis/patch_analysis.py --model Q4"),
    ("patch G4", A / "G4/patch_hero_48/meta.json", "analysis/patch_analysis.py --model G4"),
    ("dispersion Q9", A / "Q9/replay_core_release_600/meta.json", "analysis/dispersion.py --model Q9"),
    ("monitor Q9", A / "Q9/probes_frozen_cpt_core.json", "analysis/monitor.py --model Q9"),
    ("edges", A / "edges", "analysis/edges.py"),
    ("interventions", A / "Q9/steer/steer.jsonl", "analysis/interventions.py"),
    ("emotion", T / "emotion_validation_Q9.csv", "analysis/emotion_analysis.py"),
    ("casebook", T / "cpt_Q9_cpt_core_m_adv_items.csv", "analysis/casebook.py"),
    # figures
    ("F3", T / "Q9_core_family.csv", "figures/fig_behavior.py --model Q9 --dataset core"),
    ("F4+F6", T / "cpt_Q9_cpt_core_m_adv_summary.csv", "figures/fig_cpt.py f4 f6 --model Q9"),
    ("F5 Q9", T / "patch_recovery_Q9_hero_48.csv", "figures/fig_cpt.py f5 --model Q9"),
    ("F5 Q4", T / "patch_recovery_Q4_hero_48.csv", "figures/fig_cpt.py f5 --model Q4"),
    ("F5 G4", T / "patch_recovery_G4_hero_48.csv", "figures/fig_cpt.py f5 --model G4"),
    ("F2", T / "dispersion_Q9_replay_core_release_600.csv", "figures/fig_misc.py f2"),
    ("F7", T / "edges_influence.csv", "figures/fig_misc.py f7"),
    ("F9", T / "recourse_Q9.csv", "figures/fig_misc.py f9"),
    ("F10", T / "monitor_Q9.csv", "figures/fig_misc.py f10"),
]

if __name__ == "__main__":
    only = sys.argv[1:]
    for name, need, cmd in STEPS:
        if only and not any(o.lower() in name.lower() for o in only):
            continue
        if not need.exists():
            print(f"[skip] {name} (missing {need.relative_to(ROOT)})")
            continue
        t = time.time()
        script, *args = shlex.split(cmd)
        p = subprocess.run([PY, str(ROOT / "code" / script), *args], cwd=ROOT, capture_output=True, text=True)
        status = "ok" if p.returncode == 0 else "FAIL"
        print(f"[{status}] {name} ({time.time()-t:.0f}s)")
        if p.returncode != 0:
            print(p.stderr[-1500:])
