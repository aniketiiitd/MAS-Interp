"""Find numbers the paper states that the audit never checks.

`audit_reported.py` proves that every number it tests matches a table. It cannot prove the converse: that every
number in the prose is tested. Claim 7 quoted "a median AUROC of 0.66 and as high as 0.95" for days with no table
behind it at all -- the audit was green the whole time, because a number it does not know about is a number it
cannot find fault with. Reproducing it moved it to 0.685 and 0.993.

This closes that gap from the other side. It pulls every numeric token out of the claims, the abstract and the
figure captions, subtracts the ones the audit actually tested, and subtracts an explicit allowlist of numbers
that are not results (layer indices, model names, benchmark sizes, sample counts). What is left is a number
stated as a finding that nothing verifies.

The allowlist is the point of the design: every entry is a decision somebody made in writing, rather than a
number that slipped through because no one was looking.

What this can and cannot do: a number is called accounted-for if SOME audited value rounds to it, not if the
audited value for that same quantity does. With several hundred audited values, coincidental matches happen, so
the count here is a LOWER BOUND on unsourced numbers rather than a certificate for the rest. It is still worth
running -- it found three claim numbers that had no table at all, two of which changed when reproduced.

HOW WEAK, EXACTLY: `--calibrate` injects random plausible numbers and measures how often they slip through. At
704 audited values the answer is humbling -- about 37% for a one-decimal percentage and **99% for an integer
percentage**, because a tolerance of half a unit against 704 values scattered over 0-100 covers nearly the whole
range. So a clean run means "no number failed a weak test", and every run says so out loud rather than printing
a zero that reads like a guarantee. This is the paper's own probe lesson turned on the tooling: a test is only
as good as the null it is measured against, and I nearly shipped this one without measuring it at all.

Out: results/UNSOURCED.md, and a nonzero exit if anything is unaccounted for.
"""
from __future__ import annotations

import io
import re
import sys
from contextlib import redirect_stdout

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))

ROOT = Path(__file__).resolve().parents[2]
ALLOW_PATH = f"{ROOT}/results/not_results.txt"
NUM = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![\w])")
# Model names are full of digits that are not results. Strip them before tokenising rather than allowlisting
# 4, 3.1, 8, 9 and 5 globally, which would also hide real numbers that happen to equal them.
MODELNAME = re.compile(r"\b(?:Gemma-4-E4B|Llama-3\.1-8B|Qwen3\.5-9B|Gemma-4|Llama-3\.1|Qwen3\.5)\b")


def matches(v, tok, seen):
    """Does an audited value round to the number as written?

    A claim writes 132% where the table says 1.3168, and 0.66 where it says 0.6598. Comparing exact values calls
    both of those unsourced, which buries the one number that really is. The tolerance is the rounding the
    author did: a token with d decimals could have come from anything within half a unit in the last place.

    The tolerance has to travel with the unit conversion. "132" as a percent means 1.32 as a fraction, and its
    rounding slack is then 0.005, not 0.5 -- applying the percent-scale tolerance to the fraction-scale
    candidate made an integer match almost any audited value, and briefly turned this whole file green.
    """
    tok = tok.replace(",", "")
    d = len(tok.split(".")[1]) if "." in tok else 0
    tol = 0.5 * 10 ** (-d) + 1e-9
    for cand, ctol in ((v, tol), (v / 100.0, tol / 100.0), (v * 100.0, tol * 100.0)):
        for w in seen:
            if abs(cand - w) <= ctol:
                return True
    return False


def audited_values():
    """Every value the audit tested today, as the claim states it and as the table gives it."""
    import audit_reported as A
    got = set()
    orig = A.check

    def rec(label, est, lower, upper, claimed, scale=1.0, tol=0.006):
        for v in (claimed, est * scale, lower * scale, upper * scale):
            got.add(abs(float(v)))
        return orig(label, est, lower, upper, claimed, scale, tol)

    A.check = rec
    with redirect_stdout(io.StringIO()):
        try:
            A.main()
        except SystemExit:
            pass
    A.check = orig
    return got


def prose():
    import build_pdf as B
    out = []
    for kind, title, body in B.CONTRIBUTIONS:
        out.append((f"claim: {title[:48]}", body))
    if hasattr(B, "ABSTRACT"):
        out.append(("abstract", B.ABSTRACT))
    try:
        out.append(("captions", open(f"{ROOT}/outputs/figures/paper/HERO_CAPTIONS.md").read()))
    except FileNotFoundError:
        pass
    return out


def calibrate(seen, n=20000, seed=0):
    """How often does a number that should be flagged slip through? Measure it rather than assume it."""
    import random
    rng = random.Random(seed)
    out = []
    for desc, lo, hi, fmt in (("percentage, one decimal", 0, 100, "{:.1f}"),
                              ("percentage, integer", 0, 100, "{:.0f}"),
                              ("logit or z, two decimals", 0, 5, "{:.2f}"),
                              ("proportion, three decimals", 0, 1, "{:.3f}")):
        hits = sum(matches(abs(float(fmt.format(rng.uniform(lo, hi)))),
                           fmt.format(rng.uniform(lo, hi)), seen) for _ in range(n))
        out.append((desc, 100.0 * hits / n))
    return out


def main():
    if "--calibrate" in sys.argv:
        seen = audited_values()
        print(f"Miss rate against {len(seen)} audited values, {20000} injected numbers each:")
        for desc, pct in calibrate(seen):
            print(f"  {desc:30s} slips through {pct:5.1f}% of the time")
        return 0
    allow = set()
    try:
        for ln in open(ALLOW_PATH):
            ln = ln.split("#")[0].strip()
            if ln:
                allow.add(round(abs(float(ln)), 3))
    except FileNotFoundError:
        pass

    seen = audited_values()
    rows, n_tok = [], 0
    for where, text in prose():
        text = MODELNAME.sub("<model>", text)
        for m in NUM.finditer(text):
            tok = m.group(1).replace(",", "")   # "1,668" is one number, not 1 and 668
            v = abs(float(tok))
            n_tok += 1
            if round(v, 3) in allow or matches(v, tok, seen):
                continue
            ctx = " ".join(text[max(0, m.start() - 70):m.end() + 40].split())
            rows.append((where, m.group(1), ctx))

    lines = ["# Numbers stated in the paper that the audit does not check", "",
             f"{n_tok} numeric tokens in the claims, abstract and captions. "
             f"{len(seen)} distinct values are verified against tables by `analysis/audit_reported.py`; "
             f"{len(allow)} more are listed in `results/not_results.txt` as not being results.", "",
             f"**{len(rows)} unaccounted for.**", ""]
    for where, tok, ctx in rows:
        lines.append(f"- `{tok}` in **{where}** -- ...{ctx}...")
    open(f"{ROOT}/results/UNSOURCED.md", "w").write("\n".join(lines) + "\n")
    print(f"[unsourced] {n_tok} tokens, {len(seen)} audited values, {len(allow)} allowlisted, "
          f"{len(rows)} UNACCOUNTED")
    if not rows:
        print("  NOTE: a clean run is a LOWER BOUND, not a certificate. Run --calibrate for the miss rate;")
        print("        at this pool size an integer percentage slips through about 99% of the time.")
    for where, tok, ctx in rows[:40]:
        print(f"  {tok:>8s}  {where:<56s} ...{ctx[:90]}...")
    if len(rows) > 40:
        print(f"  ... and {len(rows) - 40} more in results/UNSOURCED.md")
    return 1 if rows else 0


if __name__ == "__main__":
    sys.exit(main())
