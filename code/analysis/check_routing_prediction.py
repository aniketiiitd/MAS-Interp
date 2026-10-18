"""Evaluate Gemma's attention-block result against the prediction registered before the data existed.

Written at 21:25 on 2026-09-19, while `FULL_ROUTING_G4` was at 300 of 400 episodes and no number yet existed, so
that the decision rule is fixed before the result can influence it. Reading a number and then deciding what would
have counted as confirmation is how a falsification test becomes a formality.

The claim under test: a complete attention block removes all of a peer's influence in a model built entirely from
attention, and only about four fifths of it in a hybrid, because the hybrid's recurrent layers carry influence no
attention mask can reach. That requires the shortfall to track the architecture, not the method.

  Llama-3.1-8B   32 of 32 attention   removes 131.7% [121.6, 144.4]
  Qwen3.5-9B      8 of 32 attention   removes  79.7% [ 75.2,  84.4]
  Gemma-4-E4B    42 of 42 attention   removes ?

Gemma is the second fully-maskable control, from a different vendor, size and depth.

  CONFIRMS    the lower bound of Gemma's interval sits at or above 95%. Two independent all-attention models then
              lose everything to the mask while the hybrid does not, on 2400 items each.
  FALSIFIES   Gemma's interval overlaps Qwen's [75.2, 84.4]. The shortfall would then be a property of the masking
              procedure that Llama happens not to expose, and the claim is retracted rather than caveated, per the
              standing rule that failing claims are replaced.
  AMBIGUOUS   anything between: clear of Qwen but short of 95%. Reported as ambiguous and the claim is weakened to
              what the data supports, not defended.

Usage: python analysis/check_routing_prediction.py
"""
from __future__ import annotations

import sys

import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from common import TABLES  # noqa

ARM = "peer keys blocked for every later query at every attention layer"
QWEN = (75.2, 84.4)
CONFIRM_FLOOR = 95.0


def main():
    try:
        S = pd.read_csv(TABLES / "lit_routing_G4_core_summary.csv").set_index("arm")
    except FileNotFoundError:
        print("[prediction] lit_routing_G4_core_summary.csv does not exist yet; the test has not resolved.")
        return 2
    if ARM not in S.index:
        print(f"[prediction] the arm {ARM!r} is absent; arms present: {list(S.index)}")
        return 2
    r = S.loc[ARM]
    est, lo, hi = 100 * r.share, 100 * r.share_lo, 100 * r.share_hi
    print(f"  Gemma-4-E4B, complete attention block: {est:.1f}% [{lo:.1f}, {hi:.1f}]")
    print(f"  Qwen (hybrid) was 79.7% [75.2, 84.4]; Llama (all attention) 131.7% [121.6, 144.4]")
    overlaps_qwen = lo <= QWEN[1] and QWEN[0] <= hi
    if overlaps_qwen:
        print("\n  FALSIFIED. Gemma's interval overlaps Qwen's, so a fully maskable model also stops near four "
              "fifths.\n  The shortfall is a property of the masking procedure, not the hybrid architecture.\n"
              "  Action: retract the claim. Do not caveat it.")
        return 1
    if lo >= CONFIRM_FLOOR:
        print(f"\n  CONFIRMED. The lower bound {lo:.1f}% clears the {CONFIRM_FLOOR:.0f}% floor, so two independent "
              "all-attention\n  models lose everything to the mask while the hybrid keeps about a fifth.")
        return 0
    print(f"\n  AMBIGUOUS. Clear of Qwen but the lower bound {lo:.1f}% is short of {CONFIRM_FLOOR:.0f}%.\n"
          "  Action: weaken the claim to what this supports; do not defend the original wording.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
