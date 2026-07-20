"""M22b: the corrected gust/retreat denominators.

These tests exist so the M21 artifact cannot recur silently. The load-bearing
case is `test_m21_artifact_*`: a 1-prize bench target while the opponent's
active is also KO-able is **neutral**, not a missed gust. M21 counted 27 of
those as opportunities, reported 0/27, and spent a milestone leg on it.
"""
import pytest

from tests import builders as b
from tests.fake_cg import OptionType, SelectContext

import rl.behavior as bh

FIGHTING = 6        # fake-pool energy ids (tests/test_combat.py:7); card 1
                    # attacks for 50 with one FIGHTING energy


def _obs(me, opp, options, *, supporter_played=False, context=SelectContext.MAIN):
    return b.observation(me=me, opponent=opp, options=options,
                         supporter_played=supporter_played, context=context)


@pytest.fixture
def gust_in_hand(monkeypatch):
    """Repoint GUST_IDS at the fake engine's small card pool (test_plan.py idiom)."""
    monkeypatch.setattr(bh, "GUST_IDS", frozenset({7}))
    return b.option(OptionType.PLAY, card_id=7)


class TestPrizeValue:
    def test_prize_cap_makes_a_mega_worth_one_when_one_prize_wins(self):
        """A 3-prize Mega is worth only 1 when 1 prize closes the game."""
        assert bh._val(999, prizes_left=1) == 1

    def test_no_cap_when_prizes_remain(self, monkeypatch):
        monkeypatch.setitem(bh._CARD, 999, (None, None, 0, [], 3))
        assert bh._val(999, prizes_left=6) == 3

    def test_zero_prizes_left_floors_at_zero(self):
        assert bh._val(999, prizes_left=0) == 0


class TestGustNotAnOpportunity:
    def test_no_boss_in_menu_is_not_a_candidate(self, monkeypatch):
        monkeypatch.setattr(bh, "GUST_IDS", frozenset({7}))
        me = b.player(active=b.pokemon(1, energies=[FIGHTING]))
        opp = b.player(active=b.pokemon(4, hp=300), bench=[b.pokemon(5, hp=10)])
        opt = b.option(OptionType.ATTACK)
        assert bh.judge_gust(_obs(me, opp, [opt]), opt) is None

    def test_supporter_already_played_is_not_a_candidate(self, gust_in_hand):
        me = b.player(active=b.pokemon(1, energies=[FIGHTING]), hand=[b.hand_card(7)])
        opp = b.player(active=b.pokemon(4, hp=300), bench=[b.pokemon(5, hp=10)])
        obs = _obs(me, opp, [gust_in_hand], supporter_played=True)
        assert bh.judge_gust(obs, gust_in_hand) is None

    def test_non_main_context_yields_no_verdicts(self, gust_in_hand):
        me = b.player(active=b.pokemon(1, energies=[FIGHTING]), hand=[b.hand_card(7)])
        opp = b.player(active=b.pokemon(4, hp=300), bench=[b.pokemon(5, hp=10)])
        obs = _obs(me, opp, [gust_in_hand], context=SelectContext.SWITCH)
        assert bh.judge_state(obs, gust_in_hand) == []


class TestGustTiers:
    def test_m21_artifact_equal_prizes_is_NEUTRAL_not_a_miss(self, gust_in_hand):
        """THE regression case. Both 1-prize and the active is also KO-able:
        gusting gains nothing, so declining is correct play, not a defect."""
        me = b.player(active=b.pokemon(1, energies=[FIGHTING]), hand=[b.hand_card(7)])
        opp = b.player(active=b.pokemon(4, hp=10), bench=[b.pokemon(5, hp=10)])
        v = bh.judge_gust(_obs(me, opp, [gust_in_hand]), gust_in_hand)
        assert v.tier == "neutral"
        assert v.gain <= 0
        # Guard against this passing VACUOUSLY: it must be neutral because both
        # lines are KO-able and equal-prize, NOT because no KO was found at all.
        assert v.detail["gust_best"] > 0 and v.detail["stay_best"] > 0

    def test_unaffordable_bench_ko_is_not_an_opportunity(self, gust_in_hand):
        """The second M21 defect: _charged_best counted KOs we cannot pay for."""
        me = b.player(active=b.pokemon(1, energies=[]), hand=[b.hand_card(7)])
        opp = b.player(active=b.pokemon(4, hp=300), bench=[b.pokemon(5, hp=10)])
        v = bh.judge_gust(_obs(me, opp, [gust_in_hand]), gust_in_hand)
        assert v.tier == "neutral"
        assert v.detail["gust_best"] == 0

    def test_records_whether_the_line_was_taken(self, gust_in_hand):
        me = b.player(active=b.pokemon(1, energies=[FIGHTING]), hand=[b.hand_card(7)])
        opp = b.player(active=b.pokemon(4, hp=10), bench=[b.pokemon(5, hp=10)])
        other = b.option(OptionType.ATTACK)
        obs = _obs(me, opp, [gust_in_hand, other])
        assert bh.judge_gust(obs, gust_in_hand).taken is True
        assert bh.judge_gust(obs, other).taken is False

    def test_scans_every_bench_slot_not_just_the_first(self, gust_in_hand, monkeypatch):
        """M21 broke after slot 0, letting bench ordering decide what got measured."""
        monkeypatch.setitem(bh._CARD, 5, (None, None, 0, [], 1))
        monkeypatch.setitem(bh._CARD, 6, (None, None, 0, [], 3))
        me = b.player(active=b.pokemon(1, energies=[FIGHTING]), hand=[b.hand_card(7)])
        # slot 0 is a KO-able 1-prize; slot 1 is a KO-able 3-prize behind it
        opp = b.player(active=b.pokemon(4, hp=300),
                       bench=[b.pokemon(5, hp=10), b.pokemon(6, hp=10)])
        v = bh.judge_gust(_obs(me, opp, [gust_in_hand]), gust_in_hand)
        assert v.detail["target_id"] == 6, "must pick the best target, not the first"


class TestRetreat:
    def test_not_a_candidate_without_a_retreat_option(self):
        me = b.player(active=b.pokemon(1, hp=10, energies=[FIGHTING]),
                      bench=[b.pokemon(2, energies=[FIGHTING])])
        opp = b.player(active=b.pokemon(4, hp=300))
        opt = b.option(OptionType.ATTACK)
        assert bh.judge_retreat(_obs(me, opp, [opt]), opt) is None

    def test_no_bench_to_promote_is_neutral(self):
        me = b.player(active=b.pokemon(1, hp=10, energies=[FIGHTING]))
        opp = b.player(active=b.pokemon(4, hp=300))
        opt = b.option(OptionType.RETREAT)
        v = bh.judge_retreat(_obs(me, opp, [opt]), opt)
        assert v.tier == "neutral"

    def test_unpayable_retreat_cost_is_neutral(self, monkeypatch):
        """retreatCost had zero readers before M22b; an unpayable retreat is
        not a missed opportunity."""
        monkeypatch.setattr(bh, "_retreat_cost", lambda _cid: 3)
        me = b.player(active=b.pokemon(1, hp=10, energies=[FIGHTING]),
                      bench=[b.pokemon(2, energies=[FIGHTING])])
        opp = b.player(active=b.pokemon(4, hp=300))
        opt = b.option(OptionType.RETREAT)
        v = bh.judge_retreat(_obs(me, opp, [opt]), opt)
        assert v.tier == "neutral"
        assert v.detail["why"] == "cannot pay retreat cost"

    def test_scuffed_but_surviving_active_is_not_a_defect(self, monkeypatch):
        """The old metric fired on hp/maxHp <= 0.5 — cosmetics. What matters is
        whether the active actually dies and whether promoting gains prizes."""
        monkeypatch.setattr(bh, "_retreat_cost", lambda _cid: 0)
        me = b.player(active=b.pokemon(1, hp=50, max_hp=100, energies=[FIGHTING]),
                      bench=[b.pokemon(2, energies=[])])
        opp = b.player(active=b.pokemon(4, hp=300, energies=[]))
        opt = b.option(OptionType.RETREAT)
        v = bh.judge_retreat(_obs(me, opp, [opt]), opt)
        assert v.tier in ("neutral", "tempo")
        assert v.tier != "strict"


class TestReport:
    def test_headline_prints_neutral_adjacent_to_strict(self):
        acc = bh.Counter({"games": 5, "gust_strict": 2, "gust_strict_taken": 1,
                          "gust_neutral": 26})
        out = bh.report(acc)
        assert "1/2" in out and "neutral 26" in out, \
            "n_neutral must sit beside the headline — M21 reported 0/27 where 26 were neutral"

    def test_voids_the_metric_above_20pct_undecidable(self):
        acc = bh.Counter({"games": 1, "gust_strict": 1, "gust_undecidable": 9})
        assert "VOID" in bh.report(acc)

    def test_zero_strict_is_labelled_not_a_defect(self):
        acc = bh.Counter({"games": 1, "gust_neutral": 10})
        assert "not a defect" in bh.report(acc)
