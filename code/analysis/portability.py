"""Does the paper's recurring intuition survive a count? "What ports across models is a contrast, not a level."

I have now written some version of that sentence into four reports in one day. It is the kind of claim that
sounds like a finding and may only be a way of describing whichever comparisons I happened to make, so this
measures it instead of asserting it. Three transcript impressions have already failed against their own base
rates this campaign; an impression about my own results deserves the same treatment.

For each quantity the paper reports in at least two models, we take the LEVELS (one per model) and the CONTRAST
the claim actually quotes (a difference or a ratio, one per model), and compare how much each varies across
models. Dispersion is the coefficient of variation, |sd| / |mean|, which is unitless so levels in logits and
levels in percentage points can sit in the same table.

VERDICT (2026-09-20): the intuition is NOT supported. The contrast travels better in 3 of 8 quantities, and the
median dispersion is higher for contrasts (0.374) than for levels (0.217). The sentence was a description of the
handful of comparisons I had chosen to highlight, not a property of the results.

AND THE TEST ITSELF IS WEAK, which matters more than its verdict. It compares point estimates as though they were
exact. Two of the tightest agreements here are noise: Qwen and Llama both end at 40% wrong-unanimity, a CV of
0.005, but the intervals are [26.7, 55.6] and [29.2, 52.1] on 45 and 72 clusters. Agreement to a third of a point
inside a fourteen-point interval is a coincidence, and a dispersion measure that cannot see that will manufacture
convergence wherever two noisy numbers happen to land together. So this file refutes a claim I was about to make
and does not license the opposite one either.

Out: outputs/tables/portability.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import *  # noqa

# (quantity, claim, {model: (level_a, level_b)}, how the claim states the contrast)
QUANTITIES = [
    ("answer fixed before reasoning", 1,
     {"Q9": (65.4, 22.3), "L31": (52.4, 35.4), "G4": (48.5, 12.6)}, "difference"),
    ("credential vs record swap", 2,
     {"Q9": (0.560, 0.077), "L31": (0.876, 0.011), "G4": (1.905, 0.793)}, "ratio"),
    ("labelling vs position control", 6,
     {"Q9": (0.749, 0.019), "L31": (1.086, 0.024), "G4": (1.302, 0.032)}, "ratio"),
    ("reasoning vs bare answer", 4,
     {"Q9": (1.308, 0.586), "G4": (3.393, 2.260)}, "ratio"),
    ("freeze after one turn", 5,
     {"Q9": (89.7, 79.5), "L31": (70.3, 14.2)}, "difference"),
    ("response growth over twelve turns", 5,
     {"Q9": (223.8, -68.9), "L31": (573.1, 62.2)}, "difference"),
    ("effective voters at the last turn", 5,
     {"Q9": (1.021, 2.098), "L31": (1.068, 1.704)}, "difference"),
    ("wrong unanimity, debate vs alone", 5,
     {"Q9": (40.0, 20.9), "L31": (40.3, 18.9)}, "difference"),
]


def cv(xs):
    xs = np.asarray(xs, float)
    m = np.mean(xs)
    return float(np.std(xs, ddof=1) / abs(m)) if len(xs) > 1 and m != 0 else np.nan


def main():
    rows = []
    for name, claim, per_model, kind in QUANTITIES:
        a = [v[0] for v in per_model.values()]
        b = [v[1] for v in per_model.values()]
        contrast = [x - y for x, y in zip(a, b)] if kind == "difference" else [x / y for x, y in zip(a, b)]
        # the levels' dispersion is the WORSE of the two arms: a contrast only has to beat the arm that
        # travels best, or the comparison flatters it
        cv_a, cv_b, cv_c = cv(a), cv(b), cv(contrast)
        cv_lvl = min(cv_a, cv_b)
        rows.append(dict(quantity=name, claim=claim, models=len(per_model), kind=kind,
                         cv_arm1=cv_a, cv_arm2=cv_b, cv_levels_best=cv_lvl, cv_contrast=cv_c,
                         contrast_travels_better=bool(cv_c < cv_lvl)))
        r = rows[-1]
        mark = "yes" if r["contrast_travels_better"] else "NO "
        print(f"  {name:36s} claim {claim}  levels {cv_lvl:.3f}  contrast {cv_c:.3f}   {mark}")

    T = pd.DataFrame(rows)
    out = TABLES / "portability.csv"
    T.to_csv(out, index=False)
    n, k = len(T), int(T.contrast_travels_better.sum())
    print(f"\n  the contrast travels better in {k} of {n} quantities")
    print(f"  median CV: levels {T.cv_levels_best.median():.3f}, contrasts {T.cv_contrast.median():.3f}")
    print("  VERDICT: the 'contrasts port, levels do not' intuition is NOT supported.")
    print("  CAVEAT: this compares point estimates only. The tightest agreements here (wrong-unanimity, CV 0.005)")
    print("          sit inside intervals fourteen points wide, so neither this verdict nor its opposite is")
    print("          licensed by these numbers alone.")
    print(f"[done] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
