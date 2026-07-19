"""M9 Leg 1 pilot-v2 fixes (rl side): default-off flag identity + Fix A/B.

The tcg/ twins get mirrored only after measurement (docs/M9-plan.md), so this
file exercises rl.generic_pilot directly. Fake-cg card ids follow tests/fake_cg
(1 = charged Fighting attacker, 4 = 30-dmg attacker, 5 = no damaging attack,
7 = trainer).
"""
import pytest

from tests import builders as b
from tests.fake_cg import AreaType, EnergyType, OptionType

import rl.generic_pilot as rlp

FIGHTING = EnergyType.FIGHTING


def _play_opt(index=0):
    return b.option(OptionType.PLAY, area=AreaType.HAND, index=index)


class TestFixAHandDiscard:
    @pytest.fixture(autouse=True)
    def _trainer_seven_is_hand_discard(self, monkeypatch):
        monkeypatch.setattr(rlp, "_HAND_DISCARD_IDS", {7})

    def _obs(self):
        # Keeper in hand (card 3 is a Pokémon => keeper under fake data).
        me = b.player(active=b.pokemon(1), hand=[b.hand_card(7), b.hand_card(3)])
        return b.observation(me=me)

    def test_default_off_keeps_blocked_score(self):
        assert rlp.score_play(_play_opt(), self._obs()) == 150

    def test_fix_refuses_below_end(self):
        score = rlp.score_play(_play_opt(), self._obs(), frozenset({"handdiscard"}))
        assert score == -100
        end_score = rlp.score_option(b.option(OptionType.END), self._obs(),
                                     frozenset({"handdiscard"}))
        assert score < end_score

    def test_fix_inapplicable_when_hand_is_chaff(self):
        # No keepers: the fix must not change the normal trainer path.
        me = b.player(active=b.pokemon(1), hand=[b.hand_card(7)])
        obs = b.observation(me=me)
        assert rlp.score_play(_play_opt(), obs, frozenset({"handdiscard"})) \
            == rlp.score_play(_play_opt(), obs)


class TestFixBGust:
    @pytest.fixture(autouse=True)
    def _trainer_seven_is_gust(self, monkeypatch):
        monkeypatch.setattr(rlp, "_GUST_IDS", {7})

    def _obs(self, bench):
        me = b.player(active=b.pokemon(1, energies=[FIGHTING, FIGHTING]),
                      hand=[b.hand_card(7)])
        # Opp active is slow to KO (300 hp), bench dies in one hit (50 hp).
        opp = b.player(active=b.pokemon(4, hp=300), bench=bench)
        return b.observation(me=me, opponent=opp)

    def test_fix_scores_killshot_tier(self):
        obs = self._obs(bench=[b.pokemon(5, hp=50)])
        assert rlp.score_play(_play_opt(), obs, frozenset({"gust"})) == 2450

    def test_default_off_keeps_taper(self):
        obs = self._obs(bench=[b.pokemon(5, hp=50)])
        assert rlp.score_play(_play_opt(), obs) == 2200  # hand 1, deck 30

    def test_no_better_target_falls_through(self):
        # Bench clone of the active: not STRICTLY faster -> normal taper.
        obs = self._obs(bench=[b.pokemon(4, hp=300)])
        assert rlp.score_play(_play_opt(), obs, frozenset({"gust"})) == 2200

    def test_empty_bench_falls_through(self):
        obs = self._obs(bench=[])
        assert rlp.score_play(_play_opt(), obs, frozenset({"gust"})) == 2200

    def test_helper_predicate(self):
        me = b.player(active=b.pokemon(1, energies=[FIGHTING, FIGHTING]))
        opp_fast_bench = b.player(active=b.pokemon(4, hp=300),
                                  bench=[b.pokemon(5, hp=50)])
        assert rlp._gust_has_better_target(me, opp_fast_bench)
        opp_no_bench = b.player(active=b.pokemon(4, hp=300))
        assert not rlp._gust_has_better_target(me, opp_no_bench)
        no_board = b.player()
        assert not rlp._gust_has_better_target(no_board, opp_fast_bench)


class TestFixB2GustTargeting:
    def _target_opt(self, index):
        return b.option(OptionType.CARD, area=AreaType.BENCH, index=index,
                        player_index=1)

    def _obs(self):
        from tests.fake_cg import SelectContext
        me = b.player(active=b.pokemon(1, energies=[FIGHTING, FIGHTING]))
        # Bench 0: mega (3 prizes) but 300 hp — slow. Bench 1: 50 hp — 1 hit.
        opp = b.player(active=b.pokemon(4, hp=300),
                       bench=[b.pokemon(3, hp=300), b.pokemon(5, hp=50)])
        return b.observation(me=me, opponent=opp,
                             context=SelectContext.EFFECT_TARGET)

    def test_default_keeps_prize_ranking(self):
        obs = self._obs()
        mega = rlp.score_card(self._target_opt(0), obs)
        fast = rlp.score_card(self._target_opt(1), obs)
        assert mega > fast                        # 3 prizes beat 1 prize

    def test_fix_prefers_fastest_ko(self):
        obs = self._obs()
        gust = frozenset({"gust"})
        mega = rlp.score_card(self._target_opt(0), obs, gust)
        fast = rlp.score_card(self._target_opt(1), obs, gust)
        assert fast > mega                        # 1-hit KO beats slow mega


class TestEmptyFixesIdentity:
    def test_score_option_identity_across_branches(self):
        me = b.player(active=b.pokemon(1, energies=[FIGHTING]),
                      hand=[b.hand_card(7), b.hand_card(3)])
        opp = b.player(active=b.pokemon(4, hp=300), bench=[b.pokemon(5, hp=50)])
        obs = b.observation(me=me, opponent=opp)
        options = [b.option(OptionType.END), b.option(OptionType.ABILITY),
                   _play_opt(0), _play_opt(1),
                   b.option(OptionType.ATTACK, attack_id=101)]
        for o in options:
            assert rlp.score_option(o, obs) == rlp.score_option(o, obs, frozenset())
