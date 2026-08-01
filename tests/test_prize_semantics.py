"""Prize-array semantics: the engine fact, and the PKM_PRIZE_FIX switch.

ENGINE FACT (scripts/prize_semantics_probe.py, and M36's diag end-states):
`player.prize` is that player's OWN remaining prizes — it drains as THEY take
prizes, and the winner's own list ends at 0.

Three consumers were written against the opposite convention and are corrected
behind `PKM_PRIZE_FIX` (default OFF, so the frozen net's features, every
`solver:` bed number and past PPO runs are untouched):
  * rl.plan._make_plan   -> the `wins` / `concedes` plan features
  * rl.turn_solver       -> score_leaf's two prize terms + the T4 trigger
  * rl.collector / tcg.selfplay -> the PPO prize-shaping delta

These cases pin BOTH branches, so flipping the default in the M38 arm is a
one-line change with the expected behaviour already written down
(docs/m38-code-audit.md).
"""
import os

import pytest

pytest.importorskip("numpy")

import rl.plan as rp
import rl.turn_solver as ts
from tests import builders as b
from tests.fake_cg import EnergyType, OptionType, SelectContext

F = EnergyType.FIGHTING
W = EnergyType.WATER


def flipped(my_p, op_p):
    """A turn-passed observation, seat 0 = me (tests/test_turn_solver twin)."""
    return b.observation(op_p, my_p, your_index=1)


# --- rl.plan._make_plan: wins / concedes ------------------------------------
def _lethal_plan(my_prizes, opp_prizes):
    """My card-1 attacker vs a 40hp card-5 target it one-shots. Card 5 is worth
    1 prize; my attacker is worth 1 and cannot be return-KO'd here."""
    me = b.player(active=b.pokemon(1, energies=[F, F]),
                  prizes_remaining=my_prizes)
    op = b.player(active=b.pokemon(5, hp=40), prizes_remaining=opp_prizes)
    obs = b.observation(me=me, opponent=op)
    cands = [c for c in rp.enumerate_plans(obs)[1:] if c.lethal]
    assert cands, "expected a lethal candidate"
    return cands[0]


class TestPlanWins:
    def test_fix_reads_my_own_remaining_prizes(self, monkeypatch):
        monkeypatch.setattr(rp, "PRIZE_FIX", True)
        # I need 1 more prize; this 1-prize KO ends the game.
        assert _lethal_plan(my_prizes=1, opp_prizes=6).wins
        # I need 4; the same KO does not.
        assert not _lethal_plan(my_prizes=4, opp_prizes=1).wins

    def test_legacy_default_reads_the_opponents_array(self, monkeypatch):
        monkeypatch.setattr(rp, "PRIZE_FIX", False)
        assert _lethal_plan(my_prizes=4, opp_prizes=1).wins
        assert not _lethal_plan(my_prizes=1, opp_prizes=6).wins


class TestPlanConcedes:
    def _sacrifice(self, my_prizes, opp_prizes):
        # My 3-prize mega at 50hp attacks into an active that return-KOs it.
        me = b.player(active=b.pokemon(3, hp=50, energies=[F, F]),
                      prizes_remaining=my_prizes)
        op = b.player(active=b.pokemon(1, hp=400, energies=[F, F]),
                      prizes_remaining=opp_prizes)
        cands = rp.enumerate_plans(b.observation(me=me, opponent=op))
        atk = [c for c in cands[1:] if c.attacker_slot == 0]
        assert atk
        return atk

    def test_fix_charges_the_prizes_the_opponent_still_needs(self, monkeypatch):
        monkeypatch.setattr(rp, "PRIZE_FIX", True)
        # They need 2 and my attacker hands them 3 -> losing it loses the game.
        assert all(c.return_ko and c.concedes
                   for c in self._sacrifice(my_prizes=6, opp_prizes=2))
        # They need 6 -> the same trade is survivable.
        assert not any(c.concedes
                       for c in self._sacrifice(my_prizes=2, opp_prizes=6))


# --- rl.turn_solver: score_leaf prize terms + the T4 trigger -----------------
def _leaf(snap, my_prizes, opp_prizes):
    mine = b.player(active=b.pokemon(1, hp=120, energies=[F, F]),
                    prizes_remaining=my_prizes, deck_count=30)
    opp = b.player(active=b.pokemon(2, hp=130), prizes_remaining=opp_prizes)
    return ts.score_leaf(snap, flipped(mine, opp))


class TestScoreLeafPrizeTerms:
    SNAP = ts._Snap(me=0, my_prizes=4, op_prizes=4, op_active_hp=130,
                    my_deck_count=30)

    def test_fix_pays_for_prizes_i_took_and_charges_for_conceded(self, monkeypatch):
        monkeypatch.setattr(ts, "PRIZE_FIX", True)
        base = _leaf(self.SNAP, 4, 4)
        took_two = _leaf(self.SNAP, 2, 4)        # my array drained by 2 = I took 2
        conceded = _leaf(self.SNAP, 4, 3)        # their array drained = they took 1
        assert took_two - base == pytest.approx(2 * ts.W_PRIZE)
        assert conceded - base == pytest.approx(ts.W_MY_PRIZE)
        assert took_two > base > conceded

    def test_legacy_default_has_the_two_terms_swapped(self, monkeypatch):
        """Regression pin for the historical beds: taking prizes SCORES the
        concede penalty, which is why the lethal tier never cleared
        MIN_OVERRIDE_SCORE on a prize line (docs/m38-code-audit.md)."""
        monkeypatch.setattr(ts, "PRIZE_FIX", False)
        base = _leaf(self.SNAP, 4, 4)
        took_two = _leaf(self.SNAP, 2, 4)
        assert took_two - base == pytest.approx(2 * ts.W_MY_PRIZE)
        assert took_two < base
        assert took_two < ts.MIN_OVERRIDE_SCORE

    def test_fixed_prize_lines_reach_the_override_bar(self, monkeypatch):
        monkeypatch.setattr(ts, "PRIZE_FIX", True)
        assert _leaf(self.SNAP, 2, 4) >= ts.MIN_OVERRIDE_SCORE   # 2 prizes
        # ...but a ONE-prize line does not, because the leaf's negative
        # tiebreak tail (W_RACE here; W_COUNTER/W_DECK_LOW in real positions)
        # eats the last dollar of MIN_OVERRIDE_SCORE = W_PRIZE - 1. The gate is
        # therefore stricter than its "override for >= 1 prize" docstring —
        # dormant while the terms were swapped, live once they are not
        # (docs/m38-code-audit.md, follow-on F1).
        one_prize = _leaf(self.SNAP, 3, 4)
        assert ts.W_PRIZE > one_prize >= ts.W_PRIZE - 100
        assert one_prize < ts.MIN_OVERRIDE_SCORE


class TestT4Trigger:
    def _obs(self, my_prizes, opp_prizes):
        # A board with damage in reach but no KO: only T4 can fire.
        me = b.player(active=b.pokemon(4, energies=[F]),
                      prizes_remaining=my_prizes)
        op = b.player(active=b.pokemon(5, hp=300), prizes_remaining=opp_prizes)
        obs = b.observation(me=me, opponent=op,
                            context=SelectContext.MAIN,
                            options=[b.option(OptionType.ATTACK),
                                     b.option(OptionType.END)])
        obs.search_begin_input = object()
        return obs

    def test_fix_fires_when_i_am_two_prizes_from_winning(self, monkeypatch):
        monkeypatch.setattr(ts, "PRIZE_FIX", True)
        assert ts.solve_trigger(self._obs(my_prizes=2, opp_prizes=6)) == "T4_closing"
        assert ts.solve_trigger(self._obs(my_prizes=6, opp_prizes=2)) is None

    def test_legacy_default_fires_when_the_opponent_is_closing(self, monkeypatch):
        monkeypatch.setattr(ts, "PRIZE_FIX", False)
        assert ts.solve_trigger(self._obs(my_prizes=6, opp_prizes=2)) == "T4_closing"
        assert ts.solve_trigger(self._obs(my_prizes=2, opp_prizes=6)) is None


# --- PPO prize shaping ------------------------------------------------------
class TestPrizeShapingDelta:
    """The dense term is +PRIZE_SHAPING per prize gained on the race; with the
    arrays swapped it paid the learner for the OPPONENT's prizes instead."""

    def _sides(self, my_prizes, opp_prizes):
        return (b.player(prizes_remaining=my_prizes),
                b.player(prizes_remaining=opp_prizes))

    @pytest.mark.parametrize("module_name", ["rl.collector", "tcg.selfplay"])
    def test_sign(self, module_name, monkeypatch):
        import importlib
        mod = importlib.import_module(module_name)
        me, op = self._sides(my_prizes=3, opp_prizes=6)   # I took 3, they took 0
        monkeypatch.setattr(mod, "PRIZE_FIX", True)
        assert mod.prize_delta(me, op) == 3
        monkeypatch.setattr(mod, "PRIZE_FIX", False)
        assert mod.prize_delta(me, op) == -3


# --- the switch itself ------------------------------------------------------
SWITCH_MODULES = ["rl.plan", "rl.turn_solver", "rl.collector", "tcg.selfplay"]


def test_every_consumer_reads_the_same_switch():
    """One env var flips all four, or an arm ships half-corrected features."""
    import importlib
    values = {name: importlib.import_module(name).PRIZE_FIX
              for name in SWITCH_MODULES}
    assert len(set(values.values())) == 1, values


@pytest.mark.skipif(os.environ.get("PKM_PRIZE_FIX") == "1",
                    reason="the run explicitly opted in")
def test_default_is_off():
    """The bundle must not change behaviour without an explicit opt-in — the
    frozen net was trained on the legacy features."""
    import importlib
    for name in SWITCH_MODULES:
        assert importlib.import_module(name).PRIZE_FIX is False, name
