"""M7.1 deck factory: line enumeration, scoring, shells, assembly, field fitness.

Everything here is engine-free — deck_build works from the committed parquets and
field_fitness's aggregation is tested with an injected play_fn (its default engine
loop, like matchup(), is [ENGINE]-only).
"""
import json
from pathlib import Path

import pytest

pytest.importorskip("polars")
pytest.importorskip("numpy")

import rl.deck_build as db  # noqa: E402
import rl.deck_search as ds  # noqa: E402
from rl.kaggle_ingest import deck_hash  # noqa: E402

DECKS = Path(__file__).resolve().parent.parent / "decks"
LUCARIO = [int(x) for x in (DECKS / "lucario.csv").read_text().split()]


# --- data contracts ----------------------------------------------------------

def test_attack_table_covers_every_attack_and_pins_mega_lucario():
    assert sum(len(a) for a in db.ATTACKS.values()) == 1556
    best = max(db.ATTACKS[678], key=lambda a: a["damage"])  # Mega Lucario ex
    assert (best["attack_name"], best["damage"], best["cost_total"]) == ("Mega Brave", 270, 2)


def test_basic_energy_ids_equal_their_type_ids():
    assert db._BASIC_ENERGY == {t: t for t in range(1, 9)}


def test_pool_weakness_histogram_pins_the_m6_measurement():
    weak = db.meta_weakness_histogram()
    top3 = sorted(weak, key=weak.get, reverse=True)[:3]
    assert top3 == [2, 6, 4]  # Fire, Fighting, Lightning (220/188/155 in M6)


# --- line enumeration --------------------------------------------------------

def test_lines_chain_by_name_and_finals_attack():
    lines = db.enumerate_lines()
    assert len(lines) > 100
    for ln in lines:
        assert 1 <= len(ln.stages) <= 3
        assert ln.stages[-1] == ln.final and ln.peak_damage > 0
        for lower, upper in zip(ln.stages, ln.stages[1:]):
            evo_from = db._ft[upper]["evolves_from_id"]
            assert db._ft[evo_from]["name"] == db._ft[lower]["name"]  # by NAME (fact 1)


def test_lucario_line_spot_check():
    """Verification item 2: the known Lucario line is enumerated as played —
    Riolu #677 (best printing, not the #974 the id points at) -> Mega Lucario ex."""
    (line,) = [ln for ln in db.enumerate_lines() if ln.final == 678]
    assert line.stages == [677, 678]
    assert (line.energy_type, line.prize_liability, line.best_cost) == (6, 3, 2)
    assert line.setup_cost == 3  # 2 energy + 1 evolution step


# --- scoring -----------------------------------------------------------------

def test_score_line_rewards_coverage_and_charges_liabilities():
    (lucario,) = [ln for ln in db.enumerate_lines() if ln.final == 678]
    hot = db.score_line(lucario, {6: 0.5})
    cold = db.score_line(lucario, {6: 0.0})
    assert hot > cold  # weakness coverage pays
    single_prize = db.dataclasses.replace(lucario, prize_liability=1)
    assert db.score_line(single_prize, {}) > db.score_line(lucario, {})
    slow = db.dataclasses.replace(lucario, setup_cost=lucario.setup_cost + 2)
    assert db.score_line(slow, {}) < db.score_line(lucario, {})


def test_harvested_meta_reweights_the_histogram(tmp_path):
    """The M7.0 -> deck-building feedback edge: a meta of Fighting-weak decks
    should shift coverage toward Fighting."""
    import polars as pl
    fighting_weak = [cid for cid, r in db._ft.items()
                     if r["is_pokemon"] and r["weakness_id"] == 6][:5]
    pq = tmp_path / "opp_decks.parquet"
    pl.DataFrame({"deck": [fighting_weak], "deck_hash": ["x"]}).write_parquet(pq)
    weak = db.meta_weakness_histogram(pq)
    assert weak == {6: 1.0}


# --- shells ------------------------------------------------------------------

def test_mine_shells_extracts_exactly_the_trainers(tmp_path):
    shells = db.mine_shells(out=tmp_path / "shells.json")
    lucario_trainers = [i for i in LUCARIO if db._is_trainer(i)]
    assert shells["lucario_engine"] == lucario_trainers
    assert len(lucario_trainers) == 31
    assert json.loads((tmp_path / "shells.json").read_text()).keys() == shells.keys()


def test_harvested_shell_takes_top_trainers_capped_at_four(tmp_path):
    import polars as pl
    trainers = [i for i in LUCARIO if db._is_trainer(i)]
    pq = tmp_path / "opp.parquet"
    pl.DataFrame({"deck": [LUCARIO, LUCARIO], "deck_hash": ["a", "b"]}).write_parquet(pq)
    shell = db.harvested_shell(pq, size=10)
    assert len(shell) == 10
    assert all(db._is_trainer(i) for i in shell)
    assert all(shell.count(i) <= 4 for i in set(shell))
    assert set(shell) <= set(trainers)


# --- assembly + generation ---------------------------------------------------

def _line(final_id):
    return next(ln for ln in db.enumerate_lines() if ln.final == final_id)


def test_build_deck_is_60_legal_with_template_counts():
    lucario = _line(678)
    shell = [i for i in LUCARIO if db._is_trainer(i)]
    deck = db.build_deck(lucario, shell)
    assert len(deck) == 60 and ds.validate_deck(deck)[0]
    assert deck.count(677) == 4 and deck.count(678) == 3  # 4-3 stage-1 line
    assert deck.count(6) >= 12  # Fighting energy, cost-2 attacker -> 12 base


def test_build_deck_dual_type_gives_the_tech_its_energy():
    lucario = _line(678)
    meta = db.meta_weakness_histogram()
    tech = db.pick_tech(db.enumerate_lines(), lucario, meta)
    assert tech is not None and tech.energy_type != lucario.energy_type
    deck = db.build_deck(lucario, [i for i in LUCARIO if db._is_trainer(i)], tech)
    assert ds.validate_deck(deck)[0]
    assert deck.count(db._BASIC_ENERGY[tech.energy_type]) == 4


def test_generate_dedupes_validates_and_diversifies(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "SHELLS_JSON", tmp_path / "shells.json")
    out = tmp_path / "gen"
    results = db.generate(n=24, top_k=8, out=out)
    assert 0 < len(results) <= 24
    hashes = {deck_hash(deck) for _, deck in results}
    assert len(hashes) == len(results)  # deduped
    for csv, deck in results:
        ids = [int(x) for x in csv.read_text().split()]
        assert ids == deck and ds.validate_deck(ids)[0]
    manifest = json.loads((out / "manifest.json").read_text())
    archetypes = {d["line"] for d in manifest["decks"]}
    assert len(archetypes) >= 3  # round-robin keeps archetype diversity


# --- field_fitness aggregation (engine loop itself is [ENGINE]) ---------------

def test_field_fitness_weighted_mean_and_breakdown():
    field = [("generic", "lucario"), ("random", "kyogre")]
    results = {"generic": [0, 0, 1, 2], "random": [0, 0, 0, 1]}  # wr .625 / .75

    def play_fn(deck, spec, n, pilot):
        assert pilot == "generic" and n == 4
        return results[spec[0]]

    wr, per_opp = ds.field_fitness([0] * 60, field, games_per_opp=4, play_fn=play_fn)
    assert wr == pytest.approx((0.625 + 0.75) / 2)
    assert per_opp[str(field[0])] == {"wr": 0.625, "wins": 2, "draws": 1, "n": 4}
    weighted, _ = ds.field_fitness([0] * 60, field, games_per_opp=4,
                                   weights=[3, 1], play_fn=play_fn)
    assert weighted == pytest.approx((3 * 0.625 + 0.75) / 4)


def test_field_fitness_rejects_mismatched_weights():
    with pytest.raises(ValueError, match="weights"):
        ds.field_fitness([0] * 60, [("random", "kyogre")], weights=[1, 2],
                         play_fn=lambda *a: [0])


def test_resolve_deck_accepts_name_path_and_ids():
    assert ds._resolve_deck("lucario") == LUCARIO
    assert ds._resolve_deck(str(DECKS / "lucario.csv")) == LUCARIO
    assert ds._resolve_deck(LUCARIO) == LUCARIO


# --- floor deck (M7.2b gate fixture) -------------------------------------------

def test_build_floor_deck_is_legal_deterministic_and_harmless():
    deck = db.build_floor_deck()
    assert len(deck) == 60 and ds.validate_deck(deck)[0]
    assert deck == db.build_floor_deck()  # deterministic
    for cid in deck:
        r = db._ft[cid]
        if r["is_pokemon"]:
            assert r["max_damage"] == 0 and not r["has_variable_attack"]


def test_committed_floor_deck_matches_the_builder():
    committed = [int(x) for x in (DECKS / "floor_zero_damage.csv").read_text().split()]
    assert committed == db.build_floor_deck()  # regeneration tripwire


def test_field_hill_climb_accepts_only_clear_field_gains():
    fitness = {"seed": 0.50, "better": 0.55, "worse": 0.48, "marginal": 0.51}
    decks = {"seed": [1] * 60, "better": [2] * 60, "worse": [3] * 60, "marginal": [4] * 60}
    names = {tuple(v): k for k, v in decks.items()}
    proposals = iter([decks["worse"], decks["marginal"], decks["better"], decks["worse"]])

    def fitness_fn(deck, field, games_per_opp, pilot):
        return fitness[names[tuple(deck)]], {}

    import unittest.mock as mock
    with mock.patch.object(ds, "mutate_flex", side_effect=lambda d, n_swaps: next(proposals)):
        champ, best, history = ds.field_hill_climb(
            decks["seed"], [("random", "kyogre")], proposals=4, min_gain=0.02,
            fitness_fn=fitness_fn)
    assert champ == decks["better"] and best == 0.55
    tags = [t for _, _, t in history]
    assert tags == ["SEED", "reject", "reject", "ACCEPT", "reject"]  # +0.01 < min_gain
