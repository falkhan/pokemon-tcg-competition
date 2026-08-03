"""Deck operations for the deck lab (tcg/decklab.py).

The evolution cases are the ones that matter: two different plausible
implementations (by card id, and without the self-start exemption) each produce
a confident wrong answer on a deck this repo actually ships.
"""
import subprocess
from pathlib import Path

import pytest

from tcg import decklab as dl
from tcg.cardpool import RARE_CANDY_ID, cards, self_starting_ids

ROOT = Path(__file__).resolve().parent.parent
DECK_FILES = sorted((ROOT / "decks").glob("*.csv"))


def _id_for(name: str) -> int:
    return next(cid for cid, r in cards().items() if r["name_norm"] == name)


# --- counts <-> ids --------------------------------------------------------

@pytest.mark.parametrize("path", DECK_FILES, ids=lambda p: p.stem)
def test_counts_round_trip_and_hash_is_order_invariant(path):
    from tcg.decks import load_deck_file
    ids = load_deck_file(path)
    counts = dl.counts_from_ids(ids)
    assert dl.deck_size(counts) == len(ids) == 60
    assert sorted(dl.deck_ids(counts)) == sorted(ids)
    assert dl.deck_hash(dl.deck_ids(counts)) == dl.deck_hash(ids)


def test_deck_hash_matches_the_project_definition():
    """Pinned to rl.kaggle_ingest so a deck built here joins to the harvest and
    census tables."""
    from rl.kaggle_ingest import deck_hash as ingest_hash
    from tcg.decks import load_deck_file
    ids = load_deck_file(ROOT / "decks/lucario.csv")
    assert dl.deck_hash(ids) == ingest_hash(ids)


def test_copies_by_name_totals_across_printings_and_exempts_basic_energy():
    """The 4-copy rule is per NAME; 154 names have more than one printing."""
    riolu = [cid for cid, r in cards().items() if r["name_norm"] == "Riolu"]
    assert len(riolu) > 1
    counts = {riolu[0]: 3, riolu[1]: 2, 1: 12}      # card 1 = Basic {G} Energy
    by_name = dl.copies_by_name(counts)
    assert by_name["Riolu"] == 5
    assert "Basic {G} Energy" not in by_name


# --- legality --------------------------------------------------------------

@pytest.mark.parametrize("path", DECK_FILES, ids=lambda p: p.stem)
def test_every_shipped_deck_is_legal(path):
    ok, reasons = dl.legality(dl.load_counts(path))
    assert ok, reasons


def test_legality_reports_the_real_rules():
    ok, reasons = dl.legality({_id_for("Riolu"): 5})
    assert not ok
    joined = " ".join(reasons)
    assert "must be exactly 60" in joined and ">4 copies" in joined


# --- evolution notes: the two traps ----------------------------------------

@pytest.mark.parametrize("path", DECK_FILES, ids=lambda p: p.stem)
def test_no_shipped_deck_reports_an_error(path):
    """The load-bearing test. Two plausible implementations fail it:

    * matching evolution by card ID errors on decks/lucario.csv, which runs the
      off-printing Riolu #677 while Mega Lucario ex points at #974;
    * omitting the self-start exemption errors on decks/archaludon.csv, whose
      4x Cinderace behind no Raboot is how all 237 harvested seats build it.
    """
    errors = [n for n in dl.deck_notes(dl.load_counts(path)) if n.level == "error"]
    assert errors == [], [n.message for n in errors]


def test_lucario_line_resolves_by_name_not_id():
    """decks/lucario.csv runs Riolu #677; Mega Lucario ex points at #974."""
    notes = dl.evolution_notes(dl.load_counts(ROOT / "decks/lucario.csv"))
    assert not [n for n in notes if n.code == "missing_preevo"]


def test_cinderace_is_exempt_because_it_starts_from_hand():
    assert _id_for("Cinderace") in self_starting_ids()
    notes = dl.evolution_notes(dl.load_counts(ROOT / "decks/archaludon.csv"))
    assert not [n for n in notes if n.code == "missing_preevo"]


def test_missing_preevolution_is_reported():
    counts = {_id_for("Marnie's Grimmsnarl ex"): 3, _id_for("Marnie's Impidimp"): 4}
    codes = {(n.code, n.level) for n in dl.evolution_notes(counts)}
    assert ("missing_preevo", "warn") in codes      # no Morgrem in between


def test_rare_candy_downgrades_a_skipped_middle_stage():
    counts = {_id_for("Marnie's Grimmsnarl ex"): 3,
              _id_for("Marnie's Impidimp"): 4,
              RARE_CANDY_ID: 4}
    notes = [n for n in dl.evolution_notes(counts) if n.code == "missing_preevo"]
    assert notes and all(n.level == "info" for n in notes)
    assert "Rare Candy" in notes[0].message


def test_inverted_line_is_flagged():
    counts = {_id_for("Marnie's Grimmsnarl ex"): 4,
              _id_for("Marnie's Morgrem"): 3,
              _id_for("Marnie's Impidimp"): 4}
    assert [n for n in dl.evolution_notes(counts) if n.code == "line_inverted"]


# --- energy notes ----------------------------------------------------------

def test_special_energy_is_credited_for_an_uncovered_type():
    """decks/greattusk_wall.csv runs 0 basic Fighting behind 11 Fighting symbols
    and 8 special energy — warning there would be wrong."""
    notes = dl.energy_notes(dl.load_counts(ROOT / "decks/greattusk_wall.csv"))
    assert not [n for n in notes if n.code == "no_energy_for_type"]


def test_uncovered_type_without_special_energy_warns():
    counts = {_id_for("Blaziken ex"): 4, 5: 10}          # Fire attacker, Psychic energy
    codes = {n.code for n in dl.energy_notes(counts)}
    assert "no_energy_for_type" in codes


# --- summary ---------------------------------------------------------------

@pytest.mark.parametrize("path", DECK_FILES, ids=lambda p: p.stem)
def test_summary_kinds_sum_to_the_deck(path):
    counts = dl.load_counts(path)
    s = dl.summarize(counts)
    assert sum(s.kind_counts.values()) == 60
    assert sum(s.type_counts.values()) == 60
    assert sum(s.stage_counts.values()) == s.kind_counts.get("Pokemon", 0)


def test_summary_pins_a_known_deck():
    s = dl.summarize(dl.load_counts(ROOT / "decks/lucario.csv"))
    assert s.kind_counts == {"Pokemon": 16, "Trainer": 31, "Energy": 13}
    assert s.pokemon_energy.get("Fighting", 0) > 0
    assert s.ex_count == 4 and s.prize_liability == 24
    assert s.top_attackers[0][0] == "Mega Lucario ex"


def test_empty_deck_summarizes_without_crashing():
    s = dl.summarize({})
    assert s.n_cards == 0 and s.kind_counts == {} and s.top_attackers == []


# --- fill with energy ------------------------------------------------------

def test_only_the_eight_printed_basic_energies_exist():
    """There is no basic Colorless or Dragon energy, so those types can never
    be the fill target however much the deck demands them."""
    ids = dl.basic_energy_ids()
    assert set(ids) == {"Grass", "Fire", "Water", "Lightning", "Psychic",
                        "Fighting", "Darkness", "Metal"}
    assert all(cid == cards()[cid]["energy_type_id"] for cid in ids.values())


@pytest.mark.parametrize("deck,expected", [
    ("decks/lucario.csv", "Fighting"),
    ("decks/grim_live.csv", "Darkness"),
    ("decks/alakazam_v2_h4.csv", "Psychic"),
    ("decks/iono.csv", "Lightning"),
    ("decks/archaludon.csv", "Metal"),
    ("decks/kyogre.csv", "Water"),
])
def test_dominant_energy_recovers_what_the_deck_actually_runs(deck, expected):
    """Strip a deck's energy entirely and the attack costs alone must lead back
    to the type its author chose."""
    ft = cards()
    counts = {c: n for c, n in dl.load_counts(ROOT / deck).items()
              if not ft[c]["is_basic_energy"]}
    assert dl.dominant_energy(counts) == expected


def test_fill_with_energy_completes_a_stripped_deck_legally():
    ft = cards()
    full = dl.load_counts(ROOT / "decks/lucario.csv")
    stripped = {c: n for c, n in full.items() if not ft[c]["is_basic_energy"]}
    filled, etype, added = dl.fill_with_energy(stripped)
    assert etype == "Fighting"
    assert added == dl.DECK_SIZE - dl.deck_size(stripped)
    assert dl.deck_size(filled) == dl.DECK_SIZE
    assert dl.legality(filled)[0]


def test_fill_exceeds_the_four_copy_limit_because_energy_is_exempt():
    ft = cards()
    stripped = {c: n for c, n in dl.load_counts(ROOT / "decks/kyogre.csv").items()
                if not ft[c]["is_basic_energy"]}
    filled, _etype, added = dl.fill_with_energy(stripped)
    assert added > dl.MAX_COPIES          # 35 for this deck
    assert dl.legality(filled)[0]


def test_fill_adds_to_an_existing_energy_stack():
    counts = {_id_for("Mega Lucario ex"): 4, 6: 2}      # 6 = Basic {F} Energy
    filled, etype, added = dl.fill_with_energy(counts)
    assert etype == "Fighting"
    assert filled[6] == 2 + added
    assert dl.deck_size(filled) == dl.DECK_SIZE


def test_fill_is_a_no_op_on_a_full_deck():
    full = dl.load_counts(ROOT / "decks/lucario.csv")
    filled, etype, added = dl.fill_with_energy(full)
    assert (filled, etype, added) == (full, None, 0)


def test_fill_is_a_no_op_when_no_type_can_be_inferred():
    """An empty deck, or one whose attacks are all Colorless, has nothing to
    infer from — returning silently beats inventing a type."""
    filled, etype, added = dl.fill_with_energy({})
    assert (etype, added) == (None, 0) and filled == {}


def test_fill_does_not_mutate_the_input():
    counts = {_id_for("Mega Lucario ex"): 4}
    before = dict(counts)
    dl.fill_with_energy(counts)
    assert counts == before


# --- file IO ---------------------------------------------------------------

def test_save_deck_writes_a_loadable_file(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CUSTOM_DIR", tmp_path)
    counts = dl.load_counts(ROOT / "decks/lucario.csv")
    path = dl.save_deck("m41_probe", counts)
    from tcg.decks import load_deck_file
    assert path.parent == tmp_path
    assert sorted(load_deck_file(path)) == sorted(dl.deck_ids(counts))


@pytest.mark.parametrize("bad", ["", "../evil", "a/b", "a\\b", "with space",
                                 "x" * 49, "-leading"])
def test_save_deck_rejects_unsafe_names(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(dl, "CUSTOM_DIR", tmp_path)
    with pytest.raises(ValueError, match="invalid deck name"):
        dl.save_deck(bad, dl.load_counts(ROOT / "decks/lucario.csv"))


def test_save_deck_refuses_to_clobber_without_being_told(tmp_path, monkeypatch):
    """Load decks/lucario.csv, tweak, save: the name stays "lucario" and lands
    on decks/custom/lucario.csv. Harmless once, a silent loss of work twice."""
    monkeypatch.setattr(dl, "CUSTOM_DIR", tmp_path)
    counts = dl.load_counts(ROOT / "decks/lucario.csv")
    first = dl.save_deck("lucario", counts)
    assert first.exists()

    edited = dict(counts)
    edited[_id_for("Riolu")] = 1
    with pytest.raises(FileExistsError, match="already exists"):
        dl.save_deck("lucario", edited, allow_illegal=True)
    # the original is untouched
    from tcg.decks import load_deck_file
    assert len(load_deck_file(first)) == 60

    dl.save_deck("lucario", edited, allow_illegal=True, overwrite=True)
    assert len(load_deck_file(first)) != 60


def test_save_never_escapes_the_custom_dir(tmp_path, monkeypatch):
    """The curated decks/ and harvested data/kaggle/ lists must be unreachable."""
    monkeypatch.setattr(dl, "CUSTOM_DIR", tmp_path)
    assert dl.custom_deck_path("ok_name").parent == tmp_path
    for bad in ("../lucario", "..\\lucario", "a/b", "sub/../../escape"):
        with pytest.raises(ValueError):
            dl.custom_deck_path(bad)


def test_suggest_name_avoids_existing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CUSTOM_DIR", tmp_path)
    assert dl.suggest_name("untitled") == "untitled"
    (tmp_path / "untitled.csv").write_text("1\n", encoding="utf-8")
    assert dl.suggest_name("untitled") == "untitled_2"
    (tmp_path / "untitled_2.csv").write_text("1\n", encoding="utf-8")
    assert dl.suggest_name("untitled") == "untitled_3"


def test_save_deck_refuses_an_illegal_deck_unless_forced(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CUSTOM_DIR", tmp_path)
    counts = {_id_for("Riolu"): 4}
    with pytest.raises(ValueError, match="not legal"):
        dl.save_deck("stub", counts)
    assert dl.save_deck("stub", counts, allow_illegal=True).exists()


def test_list_deck_files_finds_the_shipped_decks():
    labels = {label for label, _ in dl.list_deck_files()}
    assert "decks/lucario.csv" in labels and "decks/grim_live.csv" in labels


# --- smoke runner ----------------------------------------------------------

def test_deck_spec_is_root_relative_posix():
    """parse_spec splits on ':', so a Windows absolute path breaks the spec."""
    spec = dl.deck_spec(ROOT / "decks" / "custom" / "foo.csv")
    assert spec == "generic:decks/custom/foo.csv"
    assert "\\" not in spec and spec.count(":") == 1


def test_smoke_command_shape_and_worker_cap():
    cmd = dl.smoke_command("generic:lucario", "generic:grim_live", games=40)
    assert "-m" in cmd and "rl.matchrunner" in cmd and "play" in cmd
    assert cmd[cmd.index("-n") + 1] == "40"
    with pytest.raises(ValueError, match="workers must be"):
        dl.smoke_command("generic:a", "generic:b", workers=12)
    with pytest.raises(ValueError):
        dl.smoke_command("generic:a", "generic:b", games=0)


def test_parse_series_line_takes_the_last_summary():
    """--diag prints per-game lines first; first-match parsing reads a game as
    the whole series."""
    out = ("game   7: L  moves= 42  deck a/b=13/21\n"
           "generic:x vs generic:y: 1W 0L 0D over 1 (wr=1.000)\n"
           "generic:x vs generic:y: 26W 13L 1D over 40 (wr=0.663)\n")
    got = dl.parse_series_line(out)
    assert got["n"] == 40 and got["w"] == 26 and got["wr"] == pytest.approx(0.663)
    assert dl.parse_series_line("nothing here") is None


def test_run_smoke_surfaces_failure(monkeypatch):
    def boom(*a, **k):
        return subprocess.CompletedProcess(a[0], 1, stdout="", stderr="engine died")
    monkeypatch.setattr(dl.subprocess, "run", boom)
    got = dl.run_smoke("generic:a", "generic:b", games=2)
    assert got["ok"] is False and "engine died" in got["stderr"]


def test_run_smoke_handles_timeout(monkeypatch):
    def slow(*a, **k):
        raise subprocess.TimeoutExpired(a[0], 1)
    monkeypatch.setattr(dl.subprocess, "run", slow)
    got = dl.run_smoke("generic:a", "generic:b", games=2, timeout_s=1)
    assert got["ok"] is False and "timed out" in got["error"]


def test_run_smoke_parses_a_successful_run(monkeypatch):
    line = "generic:a vs generic:b: 26W 13L 1D over 40 (wr=0.663)\n"
    monkeypatch.setattr(dl.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, line, ""))
    got = dl.run_smoke("generic:a", "generic:b", games=40)
    assert got["ok"] and got["result"]["n"] == 40
