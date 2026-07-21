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


class TestSupporterAvailability:
    """M23: an AVAILABILITY rate by design — the module's one sanctioned
    exception to the strictly-worse denominator rule; the reference pilot is
    the pricing. These tests pin the candidate definition and the caveat."""

    @pytest.fixture
    def sup_option(self, monkeypatch):
        monkeypatch.setattr(bh, "_SUPPORTER_IDS", frozenset({9}))
        return b.option(OptionType.PLAY, card_id=9)

    def _obs_at_turn(self, options, turn, supporter_played=False):
        me = b.player(active=b.pokemon(1), hand=[b.hand_card(9)])
        opp = b.player(active=b.pokemon(4))
        return b.observation(me=me, opponent=opp, options=options, turn=turn,
                             supporter_played=supporter_played)

    def test_playable_supporter_declined_counts_availability_not_taken(self, sup_option):
        end = b.option(OptionType.END)
        v = bh.judge_supporter(self._obs_at_turn([sup_option, end], turn=5), end)
        assert v.tier == "avail_t3" and not v.taken     # raw turn 5 = our turn 3

    def test_playing_the_supporter_is_taken(self, sup_option):
        v = bh.judge_supporter(self._obs_at_turn([sup_option], turn=1), sup_option)
        assert v.tier == "avail_t1" and v.taken

    def test_supporter_already_played_is_not_a_candidate(self, sup_option):
        obs = self._obs_at_turn([sup_option], turn=1, supporter_played=True)
        assert bh.judge_supporter(obs, sup_option) is None

    def test_no_supporter_on_menu_is_not_a_candidate(self, sup_option):
        end = b.option(OptionType.END)
        assert bh.judge_supporter(self._obs_at_turn([end], turn=1), end) is None

    def test_late_turns_bucket_separately(self, sup_option):
        v = bh.judge_supporter(self._obs_at_turn([sup_option], turn=15), sup_option)
        assert v.tier == "avail_late"

    def test_non_play_option_with_supporter_card_id_is_ignored(self, sup_option):
        discard = b.option(OptionType.DISCARD, card_id=9)
        end = b.option(OptionType.END)
        assert bh.judge_supporter(self._obs_at_turn([discard, end], turn=1), end) is None

    def test_report_prints_rates_and_the_availability_caveat(self):
        acc = bh.Counter({"games": 2, "supporter_avail_t2": 4,
                          "supporter_avail_t2_taken": 1, "supporter_avail_late": 3})
        out = bh.report(acc)
        assert "t2 1/4" in out and "late 0/3" in out
        assert "AVAILABILITY rate, not a defect counter" in out


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


class TestFisher:
    def test_symmetric_table_is_not_significant(self):
        assert bh._fisher_two_sided(5, 5, 5, 5) == pytest.approx(1.0)

    def test_complete_separation_is_significant(self):
        assert bh._fisher_two_sided(20, 0, 0, 20) < 0.001

    def test_empty_table_does_not_divide_by_zero(self):
        assert bh._fisher_two_sided(0, 0, 0, 0) == 1.0

    def test_known_value_matches_hand_computation(self):
        # 2x2 [[3,1],[1,3]] -> exact two-sided p = 0.4857...
        assert bh._fisher_two_sided(3, 1, 1, 3) == pytest.approx(0.4857, abs=1e-3)


class TestFlagContrastReport:
    def _res(self, rows, n_win=20, n_loss=20):
        return {"n_win": n_win, "n_loss": n_loss, "n_kinds": len(rows), "rows": rows}

    def test_bonferroni_alpha_divides_by_kinds_tested(self):
        rows = [{"kind": f"k{i}", "win_rate": 0.1, "loss_rate": 0.1, "w": 2, "l": 2,
                 "p": 0.02} for i in range(6)]
        out = bh.report_flags(self._res(rows))
        assert "0.0083" in out, "alpha must be 0.05/6"
        assert "ENRICHED" not in out, "p=0.02 must NOT survive correction at k=6"

    def test_flag_more_common_in_wins_is_never_called_a_defect(self):
        """attach-off-racer dominated the loss-only taxonomy but is MORE common
        in wins — the phantom this contrast exists to kill."""
        rows = [{"kind": "attach-off-racer", "win_rate": 0.60, "loss_rate": 0.50,
                 "w": 12, "l": 10, "p": 0.734}]
        out = bh.report_flags(self._res(rows))
        assert "not established" in out and "ENRICHED" not in out

    def test_genuinely_enriched_flag_is_reported(self):
        rows = [{"kind": "real-defect", "win_rate": 0.0, "loss_rate": 0.9,
                 "w": 0, "l": 18, "p": 1e-6}]
        assert "ENRICHED in losses" in bh.report_flags(self._res(rows))

    def test_report_always_carries_the_confounding_caveat(self):
        rows = [{"kind": "x", "win_rate": 0.1, "loss_rate": 0.5, "w": 2, "l": 10,
                 "p": 1e-9}]
        out = bh.report_flags(self._res(rows))
        assert "cannot show that X helps or hurts" in out, \
            "a negative delta must never be readable as 'this behaviour helps'"
