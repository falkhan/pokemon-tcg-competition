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


class TestM42ScalingReachesTheAttachPath:
    """M41 threaded `scaling` through score_attack and the card ladders but not
    through score_attach — score_option called it with no fixes at all, so the
    one path that decides where energy GOES still asked the printed-damage
    question. fake_cg card 5's only attack (103) prints 0, which is the shape
    the whole problem lives in."""

    ZERO_PRINTED_ATTACK = 103
    SCALER_CARD = 5

    @pytest.fixture
    def hand_scaler(self, monkeypatch):
        """Make 103 a hand-scaler, the Powerful Hand shape."""
        import rl.scaling as sc
        monkeypatch.setitem(sc.SCALING_ATTACKS, self.ZERO_PRINTED_ATTACK,
                            ("hand", 20, 0))

    def _obs(self, energies=()):
        me = b.player(active=b.pokemon(self.SCALER_CARD, energies=list(energies)))
        return b.observation(me=me, opponent=b.player(active=b.pokemon(4, hp=300)))

    def _attach_opt(self):
        return b.option(OptionType.ATTACH, in_play_area=AreaType.ACTIVE,
                        in_play_index=0)

    def test_default_still_calls_it_a_non_attacker(self):
        """Unchanged without the flag: the probe measures the DEFAULT rule
        pilot over-attaching 0.0%, so there is no defect here worth moving the
        shipped rules agent or the corpus the net learns from."""
        assert rlp.score_attach(self._attach_opt(), self._obs(), 
                                self._obs().current.players[0]) == 400

    def test_scaling_arm_sees_a_real_attacker(self, hand_scaler):
        obs = self._obs()
        me = obs.current.players[0]
        plain = rlp.score_attach(self._attach_opt(), obs, me)
        scaled = rlp.score_attach(self._attach_opt(), obs, me,
                                  frozenset({"scaling"}))
        assert plain == 400                      # "no damaging attack"
        assert scaled > plain                    # reaches the charged tier

    def test_surplus_penalty_becomes_reachable(self, hand_scaler):
        """The anti-over-attach cap is the only one in the codebase and it
        could never fire on a printed-0 attacker: _turns_to_ready routes
        through `if dmg <= 0` and returns UNREACHABLE at every energy count."""
        scaling = frozenset({"scaling"})
        lean = self._obs(energies=[FIGHTING])
        fat = self._obs(energies=[FIGHTING] * 4)
        lean_score = rlp.score_attach(self._attach_opt(), lean,
                                      lean.current.players[0], scaling)
        fat_score = rlp.score_attach(self._attach_opt(), fat,
                                     fat.current.players[0], scaling)
        assert fat_score < lean_score            # surplus is now penalised

    def test_own_energy_scalers_are_exempt_from_the_penalty(self, hand_scaler,
                                                            monkeypatch):
        """The other half of the M41 confusion, killed in the forensics on
        2026-08-03 and never carried into the pilot: Ogerpon is paid at 3
        energy and gains +30 per further attach, so "charged" is not
        "saturated" and the penalty would punish correct play."""
        import rl.combat as rc
        monkeypatch.setattr(rc, "OWN_ENERGY_SCALERS",
                            frozenset({self.ZERO_PRINTED_ATTACK}))
        scaling = frozenset({"scaling"})
        lean = self._obs(energies=[FIGHTING])
        fat = self._obs(energies=[FIGHTING] * 4)
        assert rlp.score_attach(self._attach_opt(), fat,
                                fat.current.players[0], scaling) == \
            rlp.score_attach(self._attach_opt(), lean,
                             lean.current.players[0], scaling)

    def test_score_option_actually_forwards_the_flag(self, hand_scaler):
        """The bug was one missing argument at the dispatcher, so pin it."""
        obs = self._obs()
        assert rlp.score_option(self._attach_opt(), obs, frozenset({"scaling"})) \
            != rlp.score_option(self._attach_opt(), obs)

    def test_attach_recipient_value_forwards_scaling(self, hand_scaler):
        """Same drop, one function over: `scaling` was accepted and then not
        passed to _turns_to_ready."""
        me = b.player(active=b.pokemon(self.SCALER_CARD),
                      bench=[b.pokemon(self.SCALER_CARD)])
        card = me.bench[0]
        opp_active = b.pokemon(4, hp=300)
        assert rlp._attach_recipient_value(card, me, opp_active, True) != \
            rlp._attach_recipient_value(card, me, opp_active, False)
