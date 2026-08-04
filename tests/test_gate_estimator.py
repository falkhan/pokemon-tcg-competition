"""Golden fixtures for scripts/m41b_gate_estimator.py (M41b § II.3c).

The instrument that decides a methodology needs fixtures more than most: its
whole job is to tell us that a tempting variance reduction is an illusion, and
a bug in it would either sell us the illusion or hide a real gain.

Ground truth here is arithmetic rather than a constructed game — the no-op is
an algebraic identity, so it can be asserted exactly, and stratification has a
closed form on a hand-built cell.
"""
import json

import pytest

import scripts.m41b_gate_estimator as gest


def _checkpoint(tmp_path, games, chunk=2):
    """[(result_int, a_prizes, b_prizes)] -> a run_pairs-shaped jsonl.

    Games are laid down in chunk order, so index parity within a chunk is the
    seat — the same recovery the estimator does.
    """
    path = tmp_path / "cell.jsonl"
    lines = [json.dumps({"pairs": [], "workers": 2, "seed": 0})]
    for start in range(0, len(games), chunk):
        block = games[start:start + chunk]
        lines.append(json.dumps({
            "job": start, "pair": 0,
            "results": [g[0] for g in block],
            "margins": [[g[1], g[2], 30, 30] for g in block]}))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_load_games_reads_outcome_margin_and_seat(tmp_path):
    """Side a's score is 1/0/0.5 for win/loss/draw, margin is b's prizes left
    minus a's (positive = a ahead), seat is chunk-index parity."""
    path = _checkpoint(tmp_path, [(0, 1, 5), (1, 6, 2), (2, 3, 3)], chunk=2)
    games = gest.load_games(path)
    assert [g[0] for g in games] == [1.0, 0.0, 0.5]
    assert [g[1] for g in games] == [4.0, -4.0, 0.0]
    assert [g[2] for g in games] == [0, 1, 0]        # parity within each chunk


def test_the_margin_adjustment_is_an_exact_no_op(tmp_path):
    """THE result § II.3c turns on. With E[X] taken in-sample the adjustment
    cancels term by term, so the 'adjusted' win rate IS the raw one — no
    matter how strongly the margin correlates with the outcome."""
    games = [(0, 0, 6), (1, 6, 0), (0, 1, 5), (1, 5, 1),
             (0, 0, 4), (1, 4, 0), (0, 2, 6), (1, 6, 2)]
    path = _checkpoint(tmp_path, games)
    loaded = gest.load_games(path)
    p_raw, h_raw = gest.raw_estimate(loaded)
    p_adj, h_adj, theta, rho = gest.adjusted_estimate(loaded)

    assert p_adj == pytest.approx(p_raw, abs=1e-12)
    # the fixture is built so the margin predicts the outcome nearly perfectly
    assert rho > 0.9 and theta > 0
    # ...and precisely because of that, the residual interval looks much
    # tighter. That gap is the illusion the instrument exists to expose.
    assert h_adj < h_raw * 0.5


def test_bootstrap_refuses_to_credit_the_illusion(tmp_path):
    """The referee: resampling moves the two estimators identically, because
    they are the same number. A future 'improvement' that made these differ
    would be reporting a variance reduction that does not exist."""
    games = [(i % 2, (i % 2) * 6, 6 - (i % 2) * 6) for i in range(40)]
    path = _checkpoint(tmp_path, games)
    spread = gest.bootstrap_spread(gest.load_games(path), reps=200)
    assert spread["raw"] == pytest.approx(spread["adjusted"], rel=1e-9)


def test_seat_stratification_uses_the_designs_balance(tmp_path):
    """Seat is fixed by the slot-fair design, so its variance contribution is
    within-stratum only. On a cell where the seats differ sharply, the
    stratified interval must be strictly tighter than the naive binomial one
    while reporting the same win rate."""
    # seat 0 always wins, seat 1 always loses: all variance is BETWEEN seats,
    # which the design has already removed.
    games = [(i % 2, 0, 0) for i in range(40)]
    loaded = gest.load_games(_checkpoint(games=games, tmp_path=tmp_path))
    p_raw, h_raw = gest.raw_estimate(loaded)
    p_seat, h_seat, detail = gest.seat_estimate(loaded)

    assert p_seat == pytest.approx(p_raw)
    assert detail[0][1] == 1.0 and detail[1][1] == 0.0
    assert h_seat == 0.0 < h_raw          # zero within-seat variance


def test_seat_stratification_is_silent_when_seats_agree(tmp_path):
    """The guard: when the seats behave identically there is nothing to
    stratify away, and the two intervals must essentially coincide."""
    # Both games in a chunk share an outcome, and chunks alternate, so each
    # SEAT sees the same 50/50 mix: all the variance is within-stratum and
    # there is nothing for stratification to remove.
    games = [((i // 2) % 2, 0, 0) for i in range(40)]
    loaded = gest.load_games(_checkpoint(tmp_path, games))
    assert {g[2] for g in loaded} == {0, 1}
    _p_raw, h_raw = gest.raw_estimate(loaded)
    _p_seat, h_seat, _d = gest.seat_estimate(loaded)
    assert h_seat == pytest.approx(h_raw, rel=0.05)


def test_a_checkpoint_without_margins_fails_loudly(tmp_path, capsys):
    """Every pre-M41b battery on disk is margin-free. Reading one must be a
    FAIL with an instruction, never a silent zero that looks like a verdict."""
    path = tmp_path / "legacy.jsonl"
    path.write_text("\n".join([
        json.dumps({"pairs": [], "workers": 2, "seed": 0}),
        json.dumps({"job": 0, "pair": 0, "results": [0, 1]})]) + "\n")
    assert gest.report(path, reps=10) == 1
    assert "no margins" in capsys.readouterr().out
