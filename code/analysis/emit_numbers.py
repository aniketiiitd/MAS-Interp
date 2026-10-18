"""Emit every audited number with the table it comes from, as a single traceable list.

The audit proves each claim number matches a table, but it proves it in passing: the mapping lives inside the
checking code and nowhere a reader can see. Writing a Results section or an appendix needs the inverse view, one
row per number saying what it is, what it equals, and which file produced it.

This runs the audit with a recording `check`, so the list cannot drift from what is actually verified: a number
appears here only if the audit tested it, and it appears with the value the audit read today.

Out: results/NUMBERS.md
"""
from __future__ import annotations

import io
import re
import sys
from contextlib import redirect_stdout

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import audit_reported as A  # noqa: E402

ROWS = []
SECTION = {"cur": "(unsectioned)"}
_orig_check = A.check
_orig_print = print


def recording_check(label, est, lower, upper, claimed, scale=1.0, tol=0.006):
    ROWS.append(dict(section=SECTION["cur"], label=label, est=est * scale,
                     lo=lower * scale, hi=upper * scale, claimed=claimed))
    return _orig_check(label, est, lower, upper, claimed, scale, tol)


def main():
    A.check = recording_check
    # section headers are plain prints inside main(); capture them to attribute each row
    buf = io.StringIO()

    class Tee(io.StringIO):
        def write(self, s):
            if s.strip() and not s.startswith("  ") and "MATCH THE TABLES" not in s and "DRIFT" not in s:
                SECTION["cur"] = s.strip()
            return buf.write(s)

    with redirect_stdout(Tee()):
        A.main()

    by_sec = {}
    for r in ROWS:
        by_sec.setdefault(r["section"], []).append(r)

    out = ["# Every number the paper states, and the table it comes from", "",
           f"Generated from `analysis/audit_reported.py` by running it with a recording check, so this list cannot",
           f"drift from what is verified: **{len(ROWS)} numbers**, each one tested today against the current tables.",
           "Values are as read at generation time; `claimed` is what the paper says.", ""]
    for sec, rows in by_sec.items():
        out.append(f"## {sec}")
        out.append("")
        out.append("| quantity | value | interval | in the paper |")
        out.append("|---|---|---|---|")
        for r in rows:
            iv = "" if (r["lo"] == r["est"] and r["hi"] == r["est"]) else f"[{r['lo']:.3f}, {r['hi']:.3f}]"
            out.append(f"| {r['label']} | {r['est']:.3f} | {iv} | {r['claimed']:.3f} |")
        out.append("")
    output_path = ROOT / "results/NUMBERS.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(out))
    print(f"[done] results/NUMBERS.md with {len(ROWS)} numbers across {len(by_sec)} sections")
    return 0


if __name__ == "__main__":
    sys.exit(main())
