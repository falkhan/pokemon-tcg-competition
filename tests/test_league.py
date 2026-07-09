"""M7.2 league: persistence, anchors, rating, scheduling, gates-as-data, promotion.

All engine-flavored paths run through injected game_fn / play / fitness_fn; the
real anchor ordering self-test is the [ENGINE] runbook step in docs/M7.md.
"""
import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("openskill")

import rl.league as lg  # noqa: E402

DECKS = Path(__file__).resolve().parent.parent / "decks"
LUCARIO = [int(x) for x in (DECKS / "lucario.csv").read_text().split()]


@pytest.fixture
def fake_pilots(monkeypatch):
    """play_series with an injected game_fn still builds pilots — fake them so
    scripted league runs don't exec real teacher modules (engine-only)."""
    import rl.matchrunner as mr
    monkeypatch.setattr(mr, "make_pilot",
                        lambda spec, instance: (instance, mr.resolve_deck(mr.spec_deck(spec))))


@pytest.fixture
def ldirs(tmp_path, monkeypatch):
    monkeypatch.setattr(lg, "LEAGUE_DIR", tmp_path)
    monkeypatch.setattr(lg, "LEAGUE_JSON", tmp_path / "league.json")
    monkeypatch.setattr(lg, "LEAGUE_DECKS", tmp_path / "decks")
    monkeypatch.setattr(lg, "GATES_DIR", tmp_path / "gates")
    monkeypatch.setattr(lg, "SUBMISSIONS_JSON", tmp_path / "submissions.json")
    return tmp_path


def _fresh(with_anchors=True) -> lg.League:
    league = lg.new_league()
    if with_anchors:
        lg.bootstrap_anchors(league)
    return league


# --- entries + persistence ---------------------------------------------------

def test_add_entry_registers_deck_and_normalizes_spec(ldirs):
    league = _fresh(with_anchors=False)
    e = lg.add_entry(league, ("generic", "lucario"))
    csv = lg.LEAGUE_DECKS / f"{e.deck_hash}.csv"
    assert csv.exists()
    assert e.pilot == ("generic", str(csv))  # deck slot normalized to the snapshot
    assert lg.add_entry(league, ("generic", "lucario"), entry_id=e.entry_id) is e  # idempotent


def test_add_entry_rejects_illegal_deck(ldirs, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("\n".join(["1"] * 60))
    with pytest.raises(ValueError, match="illegal deck"):
        lg.add_entry(lg.new_league(), ("generic", str(bad)))


def test_save_load_round_trip_restores_tuples(ldirs):
    league = _fresh()
    league.meta["champion_deck"] = "abc123"
    lg.save_league(league)
    loaded = lg.load_league()
    assert set(loaded.entries) == set(league.entries)
    for eid, e in loaded.entries.items():
        assert isinstance(e.pilot, tuple)  # JSON lists -> tuples (hashable specs)
        assert e.pilot == league.entries[eid].pilot
    assert loaded.meta["champion_deck"] == "abc123"


def test_bootstrap_anchors_seven_frozen_idempotent(ldirs):
    league = _fresh()
    assert len(league.entries) == 7
    assert all(e.frozen and e.origin == "anchor" for e in league.entries.values())
    lg.bootstrap_anchors(league)  # idempotent
    assert len(league.entries) == 7


def test_bootstrap_skips_missing_checkpoint(ldirs, monkeypatch, capsys):
    monkeypatch.setattr(lg, "ANCHORS", [
        ("random+kyogre", ("random", "kyogre")),
        ("ghost+model", ("model", "checkpoints/does_not_exist.pt", "kyogre")),
    ])
    league = lg.new_league()
    lg.bootstrap_anchors(league)
    assert set(league.entries) == {"random+kyogre"}
    assert "skipped" in capsys.readouterr().out


# --- rating ------------------------------------------------------------------

def test_record_result_matches_direct_openskill(ldirs):
    from openskill.models import PlackettLuce
    league = _fresh()
    a, b = league.entries["lucario_expert"], league.entries["random+kyogre"]
    model = PlackettLuce()
    expect = model.rate([[model.rating(mu=a.mu, sigma=a.sigma)],
                         [model.rating(mu=b.mu, sigma=b.sigma)]])
    lg.record_result(league, a.entry_id, b.entry_id, 0)
    assert (a.mu, a.sigma) == pytest.approx((expect[0][0].mu, expect[0][0].sigma))
    assert (b.mu, b.sigma) == pytest.approx((expect[1][0].mu, expect[1][0].sigma))
    assert a.games == b.games == 1


def test_record_result_draw_counts_game_but_not_rating(ldirs):
    league = _fresh()
    a, b = league.entries["lucario_expert"], league.entries["random+kyogre"]
    mu_before = (a.mu, b.mu)
    lg.record_result(league, a.entry_id, b.entry_id, 2)
    assert (a.mu, b.mu) == mu_before
    assert a.games == b.games == 1


# --- scheduling + run ----------------------------------------------------------

def test_schedule_phases(ldirs):
    league = _fresh()
    cand = lg.add_entry(league, ("generic", "kyogre"), entry_id="cand1")
    pairs = lg.schedule(league, games_per_anchor=10)
    anchors = {e.entry_id for e in league.entries.values() if e.frozen}
    assert {(a, b) for a, b, _ in pairs} == {("cand1", x) for x in anchors}
    assert all(n == 10 for _, _, n in pairs)

    # anchor phase complete -> nearest peers only, never frozen, never self
    cand.games = 10 * len(anchors)
    peer = lg.add_entry(league, ("generic", str(DECKS / "iono.csv")), entry_id="cand2")
    peer.games = 10 * len(anchors)
    pairs = lg.schedule(league, games_per_anchor=10, top_peers=3)
    assert ("cand1", "cand2") in {(a, b) for a, b, _ in pairs}
    assert all(not league.entries[b].frozen for _, b, _ in pairs)
    assert all(a != b for a, b, _ in pairs)


def test_run_updates_ratings_and_persists(ldirs, fake_pilots):
    league = _fresh()
    lg.add_entry(league, ("generic", "kyogre"), entry_id="cand1")
    lg.run(league, games_per_anchor=4, workers=1, game_fn=lambda *a: 0)  # seat-0 wins
    cand = league.entries["cand1"]
    assert cand.games == 4 * 7
    assert lg.LEAGUE_JSON.exists()
    # slot-fair scripted wins split 50/50 -> candidate stays near the pack; the
    # ratings did move off their defaults though
    assert any(abs(e.mu - 25.0) > 1e-9 for e in league.entries.values())


def test_standings_sorted_by_ordinal(ldirs, capsys):
    league = _fresh()
    league.entries["lucario_expert"].mu = 40.0
    league.entries["random+kyogre"].mu = 5.0
    ordered = lg.standings(league)
    assert ordered[0].entry_id == "lucario_expert"
    assert ordered[-1].entry_id == "random+kyogre"
    assert "lucario_expert" in capsys.readouterr().out


# --- gates as data -------------------------------------------------------------

def make_play(wr_by_kind, stats_ms=10.0):
    """play(spec_a, spec_b, n, seed, stats): scripted wr per opponent kind."""
    def play(spec_a, spec_b, n, seed, stats):
        if stats is not None:
            side = stats.setdefault("a", {"moves": 0, "time_s": 0.0, "errors": 0})
            side["moves"] += n * 10
            side["time_s"] += n * 10 * stats_ms / 1000.0
        key = spec_b[1] if spec_b[0] == "rule" else spec_b[0]
        wins = round(wr_by_kind[key] * n)
        return [0] * wins + [1] * (n - wins)
    return play


def test_run_gates_shape_thresholds_and_file(ldirs):
    league = _fresh()
    # a strong pilot: 95% vs random, 60% vs generic, 40% vs lucario expert
    play = make_play({"random": 0.95, "generic": 0.60, "lucario": 0.40})
    rec = lg.run_gates(league, "generic+lucario", play=play, held_out=[], n_scale=0.1)
    g = rec["gates"]
    assert [g[k]["pass"] for k in ("G1", "G2", "G3", "G4")] == [True] * 4
    assert (g["G2"]["threshold"], g["G3"]["threshold"], g["G4"]["threshold"]) == (0.90, 0.55, 0.35)
    assert g["G5"]["pass"] is False and g["G5"]["reason"] == "no held-out decks"
    assert g["G6"]["pass"] is True and g["G6"]["value"] == pytest.approx(10.0)
    assert rec["n_scale"] == 0.1 and rec["field_version"] == "base_v1"
    on_disk = json.loads((lg.GATES_DIR / "generic+lucario.json").read_text())
    assert on_disk["gates"]["G2"]["value"] == g["G2"]["value"]


def test_run_gates_g5_min_delta_and_g6_slow_fail(ldirs):
    league = _fresh()

    def play(spec_a, spec_b, n, seed, stats):
        if stats is not None:
            side = stats.setdefault("a", {"moves": 0, "time_s": 0.0, "errors": 0})
            side["moves"] += n
            side["time_s"] += n * 0.080          # 80ms/move -> G6 fails
        # held-out phase: the re-decked pilot slightly beats generic on deck 1,
        # loses by 5pp on deck 2 -> min delta -0.05 -> G5 fails
        if spec_a[0] == "generic" and spec_b == ("generic", "lucario"):
            return [0] * (n // 2) + [1] * (n - n // 2)          # generic baseline 0.50
        if spec_b == ("generic", "lucario"):
            return [0] * (n // 2) + [1] * (n - n // 2)
        return [0] * n
    # two held-out "decks" with scripted asymmetry via a closure counter
    calls = {"i": 0}

    def play2(spec_a, spec_b, n, seed, stats):
        if spec_b == ("generic", "lucario"):
            calls["i"] += 1
            wr = {1: 0.55, 2: 0.50, 3: 0.45, 4: 0.50}[calls["i"]]  # pilot,base,pilot,base
            wins = round(wr * n)
            return [0] * wins + [1] * (n - wins)
        return play(spec_a, spec_b, n, seed, stats)

    rec = lg.run_gates(league, "generic+lucario", play=play2,
                       held_out=["lucario", "iono"], n_scale=0.1)
    g5 = rec["gates"]["G5"]
    assert g5["value"] == pytest.approx(-0.05)
    assert g5["pass"] is False and g5["reference"] == str(("generic", "lucario"))
    assert set(g5["per_deck"]) == {"lucario", "iono"}
    assert rec["gates"]["G6"]["pass"] is False  # 80ms mean move


# --- promotion + submissions ----------------------------------------------------

def test_promote_deck_matrix(ldirs):
    league = _fresh()

    def run_case(fitness, vs_champ):
        fitness_fn = lambda ids, field, games_per_opp: (fitness, {})  # noqa: E731
        wins = round(vs_champ * 400)
        play = lambda a, b, n, seed, stats: [0] * wins + [1] * (n - wins)  # noqa: E731
        return lg.promote_deck(league, "lucario", fitness_fn=fitness_fn, play=play)

    champion_before = league.meta["champion_deck"]
    assert run_case(0.54, 0.60)["promoted"] is False
    assert run_case(0.60, 0.49)["promoted"] is False
    assert league.meta["champion_deck"] == champion_before
    rec = run_case(0.56, 0.51)
    assert rec["promoted"] is True
    assert league.meta["champion_deck"] == rec["deck_hash"]


def test_ship_eligible_reasons():
    gates = {"gates": {"G1": {"pass": True}, "G2": {"pass": False}}}
    ok, reasons = lg.ship_eligible(gates, {"promoted": True}, 0.60)
    assert not ok and any("G2" in r for r in reasons)
    gates = {"gates": {"G1": {"pass": True}}}
    ok, reasons = lg.ship_eligible(gates, {"promoted": True}, 0.56)
    assert ok and "rl.gate" in reasons[0]


def test_log_submission_appends(ldirs):
    league = _fresh()
    lg.log_submission(111, "generic+lucario", league, note="probe")
    lg.log_submission(222, "generic+lucario", league)
    rows = json.loads(lg.SUBMISSIONS_JSON.read_text())
    assert [r["submission_id"] for r in rows] == [111, 222]
    assert rows[0]["deck_hash"] == league.entries["generic+lucario"].deck_hash


# --- self-test ordering --------------------------------------------------------

def test_check_anchor_ordering_m6_shape_passes():
    ok, problems = lg.check_anchor_ordering({
        "lucario_expert": 8.0, "tuned_lucario": 8.5,   # tuned above expert = soft tie, fine
        "generic+lucario": 5.0, "bc_v1+kyogre": 3.0, "random+kyogre": -4.0,
        "iono_expert": 6.0, "generic+iono": 4.0,
    })
    assert ok and not problems


def test_check_anchor_ordering_flags_inversions():
    ok, problems = lg.check_anchor_ordering({
        "lucario_expert": 5.0, "tuned_lucario": 5.0,
        "generic+lucario": 3.0, "bc_v1+kyogre": 4.0, "random+kyogre": -1.0,
    })
    assert not ok
    assert any("bc_v1" in p for p in problems)


def test_anchor_round_robin_covers_all_pairs(ldirs, fake_pilots):
    league = _fresh()
    lg.anchor_round_robin(league, games=2, workers=1, game_fn=lambda *a: 0)
    n_anchors = 7
    expected_games_total = 2 * (n_anchors * (n_anchors - 1) // 2) * 2  # per-entry sum
    assert sum(e.games for e in league.entries.values()) == expected_games_total


def test_add_batch_enrolls_a_directory_idempotently(ldirs, tmp_path):
    gen = tmp_path / "gen"
    gen.mkdir()
    for name, deck in (("deck_aaa.csv", LUCARIO),
                       ("deck_bbb.csv", [int(x) for x in (DECKS / "iono.csv").read_text().split()])):
        (gen / name).write_text("\n".join(map(str, deck)))
    league = _fresh(with_anchors=False)
    added = lg.add_batch(league, gen)
    assert len(added) == 2 and len(league.entries) == 2
    again = lg.add_batch(league, gen)  # idempotent: same entries returned
    assert len(league.entries) == 2 and {e.entry_id for e in again} == set(league.entries)
