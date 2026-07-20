"""Old-vs-new parity: deck-file loading and the isolated teacher loader.

The teacher tests exec a SYNTHETIC teacher module (the real sample-agent
files need the actual engine), shaped to exercise both loader quirks: the
cwd-relative "../deck.csv" read and the module-level mutable state.
"""
import os
import sys
import textwrap
from pathlib import Path

import rl.teacher as old_teacher
import tcg.decks as decks
import tcg.teachers as new_teacher

REPO_ROOT = Path(__file__).resolve().parent.parent


def old_style_deck_read(path: Path) -> list[int]:
    # transcription of the old inline expression (rl/teacher.py line 53)
    return [int(x) for x in path.read_text().split() if x.strip()]


def test_load_deck_file_parity_on_repo_decks():
    deck_files = sorted((REPO_ROOT / "decks").glob("*.csv"))
    deck_files.append(REPO_ROOT / "deck.csv")
    assert deck_files, "expected deck csvs in the repo"
    for path in deck_files:
        assert decks.load_deck_file(path) == old_style_deck_read(path)


def test_load_deck_matches_load_deck_file():
    for name in ("lucario", "iono", "kyogre", "dragapult"):
        assert decks.load_deck(name) == decks.load_deck_file(
            REPO_ROOT / "decks" / f"{name}.csv")


def test_teacher_and_deck_tables_parity():
    assert new_teacher.TEACHER_PATHS == old_teacher.TEACHER_PATHS
    assert new_teacher.DECK_PATHS == old_teacher.DECK_PATHS


# --- the isolated loader ------------------------------------------------------

SYNTHETIC_TEACHER = textwrap.dedent("""\
    from pathlib import Path

    # Quirk 1: deck loaded relative to the CURRENT WORKING DIRECTORY.
    my_deck = [int(x) for x in Path("../deck.csv").read_text().split() if x.strip()]

    # Quirk 2: module-level mutable state.
    calls = []
    aggression = 1

    def agent(obs):
        calls.append(obs)
        return [len(calls), aggression, sum(my_deck)]
    """)


def make_teacher_dir(tmp_path: Path, deck: list[int]) -> tuple[Path, Path]:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    main_py = agent_dir / "main.py"
    main_py.write_text(SYNTHETIC_TEACHER)
    (tmp_path / "deck.csv").write_text("\n".join(str(card) for card in deck))
    override_deck = tmp_path / "override.csv"
    override_deck.write_text("7\n8\n9\n")
    return main_py, override_deck


def patch_tables(monkeypatch, module, main_py, override_deck):
    monkeypatch.setitem(module.TEACHER_PATHS, "synthetic", main_py)
    monkeypatch.setitem(module.DECK_PATHS, "synthetic", override_deck)


def test_load_teacher_parity(tmp_path, monkeypatch):
    main_py, override_deck = make_teacher_dir(tmp_path, deck=[1, 2, 3])
    patch_tables(monkeypatch, old_teacher, main_py, override_deck)
    patch_tables(monkeypatch, new_teacher, main_py, override_deck)

    cwd_before = os.getcwd()
    old_agent = old_teacher.load_teacher("po", agent="synthetic")
    new_agent = new_teacher.load_teacher("pn", agent="synthetic")
    assert os.getcwd() == cwd_before  # cwd restored after the chdir dance

    # Same behavior, including the my_deck override to DECK_PATHS[deck].
    assert old_agent("x") == new_agent("x")
    assert new_agent("x") == [2, 1, 7 + 8 + 9]

    # Fresh module instance per call: mutable globals must NOT leak.
    another = new_teacher.load_teacher("pn2", agent="synthetic")
    assert another("x") == [1, 1, 7 + 8 + 9]
    assert new_agent("x") == [3, 1, 7 + 8 + 9]

    # Each instance is registered under its own sys.modules name.
    assert sys.modules["_teacher_pn"] is not sys.modules["_teacher_pn2"]


def test_load_teacher_weights_override_parity(tmp_path, monkeypatch):
    main_py, override_deck = make_teacher_dir(tmp_path, deck=[1, 2, 3])
    patch_tables(monkeypatch, old_teacher, main_py, override_deck)
    patch_tables(monkeypatch, new_teacher, main_py, override_deck)

    old_agent = old_teacher.load_teacher("wo", agent="synthetic",
                                         weights={"aggression": 5})
    new_agent = new_teacher.load_teacher("wn", agent="synthetic",
                                         weights={"aggression": 5})
    assert old_agent("x") == new_agent("x") == [1, 5, 7 + 8 + 9]
