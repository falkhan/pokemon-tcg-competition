"""Every probe script has golden fixtures — asserted, not remembered.

docs/M41b-plan.md § II.3a asks for a fires-and-guards fixture per probe. That
is a state, and states rot: the next probe someone adds is exactly the one
nobody writes fixtures for, and the completion criterion still reads "done".

So the criterion is a test. `scripts/*_probe.py` is enumerated at run time and
each entry must be exercised by one of the fixture files. A new probe fails
this until its fixtures exist.

The plan's own criterion had this bug: it said "every `scripts/*_probe.py`",
which silently missed `pm_probe2.py` (renamed to `pm2_probe.py` in this
milestone so the glob covers it). Enumerating from disk is what makes the
miss impossible rather than unlikely.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
TESTS = ROOT / "tests"

# The fixture files § II.3a produced. Listed explicitly: a glob over
# tests/test_probes_*.py would let a file that stops importing a probe still
# satisfy the check by existing.
FIXTURE_FILES = (
    "test_probes_encoding.py",
    "test_probes_engine_side.py",
    "test_probes_mechanism.py",
    "test_probes_replay_analyzers.py",
    "test_probes_rule_mechanism.py",
    "test_probes_stageb.py",
)


def _probe_scripts() -> list[str]:
    return sorted(p.stem for p in SCRIPTS.glob("*_probe.py"))


def _fixture_text() -> str:
    return "\n".join((TESTS / name).read_text(encoding="utf-8")
                     for name in FIXTURE_FILES)


def test_the_fixture_files_all_exist():
    """A renamed or deleted fixture file must fail loudly here, not silently
    shrink the coverage the test below reports."""
    missing = [n for n in FIXTURE_FILES if not (TESTS / n).is_file()]
    assert not missing, f"fixture files vanished: {missing}"


def test_no_probe_escapes_the_glob():
    """The plan's own miss, pinned: any probe whose name does not end in
    `_probe.py` is invisible to `scripts/*_probe.py` and therefore to the
    coverage test below. `pm_probe2.py` was exactly that and was renamed."""
    stragglers = [p.name for p in SCRIPTS.glob("*probe*.py")
                  if not p.name.endswith("_probe.py")]
    assert not stragglers, (
        f"probe scripts outside the *_probe.py glob: {stragglers} — rename "
        "them so the coverage check can see them")


@pytest.mark.parametrize("probe", _probe_scripts())
def test_every_probe_has_fixtures(probe):
    """`probe` is imported by at least one § II.3a fixture file.

    Import is the weakest honest signal — it proves the fixtures address THIS
    instrument. Whether they fire and guard is the business of the fixture
    files themselves, which is where those assertions live.
    """
    text = _fixture_text()
    name = re.escape(probe)
    # the three import forms the fixture files actually use:
    #   import scripts.x_probe          from scripts.x_probe import f
    #   from scripts import x_probe as p
    imported = (re.search(rf"(?:import|from)\s+scripts\.{name}\b", text)
                or re.search(rf"from\s+scripts\s+import\s+(?:[\w,\s]*\b){name}\b",
                             text)
                or re.search(rf"\bscripts\.{name}\b", text))
    assert imported, (
        f"scripts/{probe}.py has no golden fixtures. docs/M41b-plan.md "
        "§ II.3a: every probe gets a fires case and a guards case, ground "
        f"truth true by construction. Add them to one of {FIXTURE_FILES}.")


def test_the_coverage_count_is_what_the_milestone_claimed():
    """22 probes: the 20 § II.3a scoped, plus the two Stage B probes this
    milestone added (the registry caught them missing fixtures). A new probe SHOULD break this — update it in the
    same commit that adds the probe's fixtures, so the count stays a decision
    rather than a drift."""
    assert len(_probe_scripts()) == 22, (
        f"probe count moved to {len(_probe_scripts())}: {_probe_scripts()}")
