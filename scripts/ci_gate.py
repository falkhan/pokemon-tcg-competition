"""The one command to run before anything consequential (M41b § II.3e).

This repo has no hosted CI. "CI" here is the pytest suite plus the QC replay
review, and both were things a human remembered to run — which is exactly how
M41 exported mid-edit and shipped a half-finished encoder. This makes the
remembering unnecessary: one command, one PASS/FAIL line.

    uv run python scripts/ci_gate.py

Three tiers, cheapest first, and it stops at the first tier that fails —
there is no point pricing a bundle whose suite is red:

  1. suite      the full pytest run. The 4 known-failing tests
                (3x test_meta_eval drift, 1x test_network Windows tempfile
                PermissionError) are tolerated BY NAME, never by count, so a
                new failure cannot hide inside the allowance.
  2. bundle     tcg.shipping.verify_bundle on submission/ — twin parity,
                artifacts present, deck size, module-level bundle purity.
  3. instruments the probe fixtures and the coverage registry, i.e. the § II.3a
                guarantee that every measuring tool has a fires-and-guards
                fixture. Included here because a green suite with an untested
                instrument is the state that produced six corrections in M42.

`scripts/qc_battery.py` runs this first, so a pre-ship replay review can never
be spent reviewing a broken artifact.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: Tolerated by NAME. Each needs a reason and an owner, not a shrug.
#: EMPTY since M43.1 (2026-08-12, the Linux box migration): the 3
#: test_meta_eval drift failures and the Windows-only test_network tempfile
#: PermissionError all pass here — the gate itself flagged them for removal.
KNOWN_FAILURES: set[str] = set()


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    return proc.returncode, proc.stdout + proc.stderr


def tier_suite() -> tuple[bool, list[str]]:
    """The full suite, with known failures tolerated by name."""
    _rc, out = _run([sys.executable, "-m", "pytest", "tests/", "-q",
                     "--no-header", "-rf"])
    failed = {line.split(" ", 2)[1] for line in out.splitlines()
              if line.startswith("FAILED ")}
    unexpected = sorted(failed - KNOWN_FAILURES)
    fixed = sorted(KNOWN_FAILURES - failed)
    notes = []
    for name in unexpected:
        notes.append(f"NEW failure: {name}")
    for name in fixed:
        # not a failure — but the allowance must not outlive its reason
        notes.append(f"known failure now PASSES, drop it from KNOWN_FAILURES: "
                     f"{name}")
    tail = [ln for ln in out.splitlines() if " passed" in ln or " failed" in ln]
    if tail:
        notes.append(f"pytest: {tail[-1].strip()}")
    return not unexpected, notes


def tier_bundle() -> tuple[bool, list[str]]:
    """submission/ must satisfy its own ship invariants right now."""
    from tcg.shipping import SUBMISSION, verify_bundle
    if not SUBMISSION.exists():
        return True, ["no submission/ directory — nothing to verify"]
    fails = verify_bundle(base=SUBMISSION)
    return not fails, fails or ["bundle invariants hold"]


def tier_instruments() -> tuple[bool, list[str]]:
    """Every probe has fixtures, and those fixtures pass."""
    rc, out = _run([sys.executable, "-m", "pytest",
                    "tests/test_probe_fixture_coverage.py",
                    "tests/test_instrument_agreement.py",
                    "tests/test_gate_estimator.py",
                    "tests/test_gate_spec.py",
                    "-q", "--no-header"])
    tail = [ln for ln in out.splitlines() if " passed" in ln or " failed" in ln]
    return rc == 0, [f"instruments: {tail[-1].strip()}" if tail else out[-200:]]


TIERS = (("suite", tier_suite),
         ("bundle", tier_bundle),
         ("instruments", tier_instruments))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-suite", action="store_true",
                    help="tiers 2-3 only (for a fast re-check after a fix)")
    a = ap.parse_args()

    print("=== ci_gate ===")
    ok_all = True
    for name, fn in TIERS:
        if a.skip_suite and name == "suite":
            print(f"[skip] {name}")
            continue
        ok, notes = fn()
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        for note in notes:
            print(f"       {note}")
        if not ok:
            ok_all = False
            print("\nstopping here — later tiers measure artifacts this one "
                  "just called untrustworthy.")
            break

    print("\nPASS — safe to export, gate, or ship." if ok_all else
          "\nFAIL — do not export, gate, or ship until this is green.")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
