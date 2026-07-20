"""M22a: the live monitor must REFUSE to call a winner below its resolution.

These tests pin the refusal, not the reporting. The M21 failure mode was a
plausible-looking number quoted without its power, so the property under test is
that `compare` returns decidable=False whenever the gap is unresolvable.
"""
import math

from rl.live_monitor import (MIN_N_FOR_ANY_READ, Arm, compare, mde, n_needed)


def arm(sub: int, n: int, wr: float, score: float = 550.0) -> Arm:
    wins = round(n * wr)
    return Arm(sub=sub, n=n, wins=wins, losses=n - wins, draws=0,
               score=score, last_seen="2026-07-20T12:00")


class TestPowerMath:
    def test_mde_shrinks_with_n(self):
        assert mde(100, 100) > mde(400, 400) > mde(1600, 1600)

    def test_mde_matches_closed_form(self):
        # 80% power, alpha=.05 two-sided, both arms n=800 -> ~7.0pp
        assert math.isclose(mde(800, 800), 0.0700, abs_tol=5e-4)

    def test_mde_infinite_on_empty_arm(self):
        assert mde(0, 100) == float("inf")

    def test_n_needed_inverse_square(self):
        # halving the detectable gap quadruples the games required
        assert math.isclose(n_needed(0.05) / n_needed(0.10), 4.0, rel_tol=0.02)

    def test_n_needed_known_value(self):
        assert n_needed(0.05) == 1570      # the 5pp design point


class TestArm:
    def test_draws_count_half(self):
        a = Arm(1, n=10, wins=4, losses=4, draws=2, score=600.0, last_seen="x")
        assert a.wr == 0.5

    def test_ci_clamped_to_unit_interval(self):
        lo, hi = arm(1, n=30, wr=0.99).ci95()
        assert lo >= 0.0 and hi <= 1.0


class TestCompareRefuses:
    def test_refuses_below_min_n(self):
        """Fresh subs ride the mu=600 prior — no read at any gap size."""
        a, b = arm(1, n=MIN_N_FOR_ANY_READ - 1, wr=0.90), arm(2, n=500, wr=0.10)
        decidable, report = compare([a, b], mde_pp=10.0)
        assert decidable is False
        assert "NO READ" in report

    def test_refuses_when_gap_inside_noise(self):
        """The M21 error: a real-looking gap smaller than the MDE."""
        a, b = arm(1, n=45, wr=0.489), arm(2, n=34, wr=0.412)
        decidable, report = compare([a, b], mde_pp=10.0)
        assert decidable is False
        assert "INSIDE the noise floor" in report
        assert "needs n=" in report

    def test_refuses_with_single_arm(self):
        decidable, _ = compare([arm(1, n=500, wr=0.6)], mde_pp=10.0)
        assert decidable is False

    def test_names_no_winner_when_undecidable(self):
        a, b = arm(111, n=45, wr=0.489), arm(222, n=34, wr=0.412)
        _, report = compare([a, b], mde_pp=10.0)
        assert "READ:" not in report          # the winner-announcing line


class TestCompareReads:
    def test_emits_read_when_gap_exceeds_mde(self):
        a, b = arm(111, n=800, wr=0.62), arm(222, n=800, wr=0.38)
        decidable, report = compare([a, b], mde_pp=10.0)
        assert decidable is True
        assert "READ: 111 > 222" in report

    def test_read_orients_winner_regardless_of_arg_order(self):
        weak, strong = arm(111, n=800, wr=0.38), arm(222, n=800, wr=0.62)
        decidable, report = compare([weak, strong], mde_pp=10.0)
        assert decidable is True
        assert "READ: 222 > 111" in report

    def test_gap_just_under_mde_refuses(self):
        """The boundary is where over-reading starts — it must refuse."""
        n = 800
        half = 0.9 * mde(n, n) / 2          # gap = 90% of the MDE
        a, b = arm(111, n=n, wr=0.5 + half), arm(222, n=n, wr=0.5 - half)
        decidable, report = compare([a, b], mde_pp=10.0)
        assert decidable is False
        assert "INSIDE the noise floor" in report

    def test_gap_just_over_mde_reads(self):
        n = 800
        half = 1.1 * mde(n, n) / 2          # gap = 110% of the MDE
        a, b = arm(111, n=n, wr=0.5 + half), arm(222, n=n, wr=0.5 - half)
        decidable, _ = compare([a, b], mde_pp=10.0)
        assert decidable is True
