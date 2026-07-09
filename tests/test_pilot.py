"""The pilot's agent contract and each scorer's branches."""
from tests import builders as b
from tests.fake_cg import AreaType, OptionType, SelectContext

from tcg import constants
from tcg.combat import best_damage
from tcg.pilot import (card_at, make_generic_pilot, score_attack, score_attach,
                       score_card, score_option, score_play, score_retreat)

FIGHTING, WATER, PSYCHIC = 6, 3, 5


class TestAgentContract:
    def test_first_call_returns_the_deck(self):
        deck = [1] * 60
        agent = make_generic_pilot(deck)
        assert agent(b.observation(select=False)) is deck

    def test_returns_top_max_count_indices(self):
        options = [b.option(OptionType.END), b.option(OptionType.ABILITY),
                   b.option(OptionType.EVOLVE)]
        agent = make_generic_pilot([1] * 60)
        picks = agent(b.observation(options=options, max_count=2))
        assert picks == [1, 2]  # ability 3000, evolve 2800, end 0

    def test_score_ties_keep_ascending_index_order(self):
        options = [b.option(OptionType.YES)] * 3
        agent = make_generic_pilot([1] * 60)
        assert agent(b.observation(options=options, max_count=3)) == [0, 1, 2]


class TestCardAt:
    def test_resolves_hand_and_bench(self):
        card = b.hand_card(7)
        benched = b.pokemon(1)
        me = b.player(hand=[card], bench=[benched])
        obs = b.observation(me=me)
        assert card_at(obs, AreaType.HAND, 0, 0) is card
        assert card_at(obs, AreaType.BENCH, 0, 0) is benched

    def test_unknown_area_index_or_none_area(self):
        obs = b.observation(me=b.player(hand=[b.hand_card(7)]))
        assert card_at(obs, AreaType.PRIZE, 0, 0) is None  # unmapped zone
        assert card_at(obs, AreaType.HAND, 5, 0) is None   # out of range
        assert card_at(obs, None, 0, 0) is None


class TestScoreAttack:
    def test_no_opponent_active_just_attack(self):
        assert score_attack(b.option(attack_id=101), b.pokemon(1), None) \
            == constants.SCORE_ATTACK_NO_TARGET

    def test_ko_scores_by_prize_count(self):
        attacker = b.pokemon(1)
        ko_basic = score_attack(b.option(attack_id=101), attacker, b.pokemon(5, hp=50))
        ko_mega = score_attack(b.option(attack_id=101), attacker, b.pokemon(3, hp=50))
        assert ko_basic == constants.SCORE_KO_BASE + constants.KO_PRIZE_BONUS
        assert ko_mega == constants.SCORE_KO_BASE + 3 * constants.KO_PRIZE_BONUS

    def test_chip_vs_harmless_board_enters_close_mode(self):
        # M7.2b: card 5 has no damaging attack -> the race is won; chip jumps
        # above trainers so the pilot attacks every turn instead of milling.
        score = score_attack(b.option(attack_id=101), b.pokemon(1), b.pokemon(5, hp=200))
        assert score == constants.SCORE_CHIP_CLOSE_BASE + 5.0  # 2300 + 50/10

    def test_chip_damage_scores_low_when_opponent_threatens(self):
        # A real attacker on the opponent's bench keeps close mode OFF.
        score = score_attack(b.option(attack_id=101), b.pokemon(1),
                             b.pokemon(5, hp=200), opponent_bench=(b.pokemon(1),))
        assert score == constants.SCORE_CHIP_BASE + 5.0  # 1000 + 50/10

    def test_close_mode_never_touches_the_ko_tier(self):
        score = score_attack(b.option(attack_id=101), b.pokemon(1), b.pokemon(5, hp=40))
        assert score == constants.SCORE_KO_BASE + constants.KO_PRIZE_BONUS

    def test_resistance_goes_negative_unlike_best_damage(self):
        # The pinned asymmetry: score_attack does NOT floor resisted damage at 0.
        attacker, defender = b.pokemon(10, energies=[FIGHTING]), b.pokemon(2, hp=200)
        assert best_damage(attacker, defender) == 0
        assert score_attack(b.option(attack_id=107), attacker, defender) \
            == constants.SCORE_CHIP_BASE + (20 - 30) / 10  # 999.0


class TestScoreAttach:
    def _obs(self, target, opponent_active=None):
        me = b.player(active=target)
        opponent = b.player(active=opponent_active)
        return b.observation(me=me, opponent=opponent)

    def _attach_option(self):
        return b.option(OptionType.ATTACH, in_play_area=AreaType.ACTIVE,
                        in_play_index=0)

    def test_unresolved_target(self):
        obs = self._obs(target=None)
        assert score_attach(self._attach_option(), obs) \
            == constants.SCORE_ATTACH_NO_TARGET

    def test_attach_that_unblocks_a_ko_tops_everything(self):
        # card 3 with one energy: 0 damage now, 270 after one more attach
        obs = self._obs(b.pokemon(3, energies=[FIGHTING]),
                        opponent_active=b.pokemon(5, hp=100))
        assert score_attach(self._attach_option(), obs) \
            == constants.SCORE_ATTACH_UNBLOCKS_KO_ACTIVE

    def test_non_attacker_scores_lowest(self):
        obs = self._obs(b.pokemon(5))
        assert score_attach(self._attach_option(), obs) \
            == constants.SCORE_ATTACH_NON_ATTACKER

    def test_already_loaded_means_the_best_attack_charged(self):
        # M7.2b: card 1's cheapest attack costs 1 but its BEST (120) costs 2 —
        # one energy is no longer "loaded"; two are.
        needs_more = self._obs(b.pokemon(1, energies=[FIGHTING]))
        assert score_attach(self._attach_option(), needs_more) \
            == constants.SCORE_ATTACH_ACTIVE_BASE + 1
        loaded = self._obs(b.pokemon(1, energies=[FIGHTING, WATER]))
        assert score_attach(self._attach_option(), loaded) \
            == constants.SCORE_ATTACH_ALREADY_LOADED

    def test_loading_an_attacker_that_needs_energy(self):
        obs = self._obs(b.pokemon(1))  # best damage 120 -> +1 bonus
        assert score_attach(self._attach_option(), obs) \
            == constants.SCORE_ATTACH_ACTIVE_BASE + 1


class TestScorePlay:
    def _obs(self, hand, active=None, hand_count=None, deck_count=30):
        me = b.player(active=active, hand=hand, hand_count=hand_count,
                      deck_count=deck_count)
        return b.observation(me=me)

    def _play_option(self, index=0):
        return b.option(OptionType.PLAY, area=AreaType.HAND, index=index)

    def test_pokemon_always_worth_playing(self):
        obs = self._obs(hand=[b.hand_card(1)])
        assert score_play(self._play_option(), obs) == constants.SCORE_PLAY_POKEMON

    def test_trainer_useless_without_a_board(self):
        obs = self._obs(hand=[b.hand_card(7)])
        assert score_play(self._play_option(), obs) \
            == constants.SCORE_PLAY_TRAINER_NO_BOARD

    def test_trainer_suppressed_near_deckout(self):
        obs = self._obs(hand=[b.hand_card(7)], active=b.pokemon(1), deck_count=6)
        assert score_play(self._play_option(), obs) \
            == constants.SCORE_PLAY_NEAR_DECKOUT

    def test_trainer_tapers_with_hand_size_down_to_the_floor(self):
        obs = self._obs(hand=[b.hand_card(7)], active=b.pokemon(1), hand_count=7)
        assert score_play(self._play_option(), obs) == 2200 - 200 * 3
        obs = self._obs(hand=[b.hand_card(7)], active=b.pokemon(1), hand_count=20)
        assert score_play(self._play_option(), obs) == constants.SCORE_TRAINER_FLOOR


class TestScoreRetreat:
    def test_never_without_active_or_bench(self):
        obs = b.observation(me=b.player(active=b.pokemon(1)))
        assert score_retreat(obs) == constants.SCORE_RETREAT_NEVER

    def test_promotes_a_lethal_bench_attacker(self):
        me = b.player(active=b.pokemon(5),  # can't KO
                      bench=[b.pokemon(1, energies=[FIGHTING, FIGHTING])])
        obs = b.observation(me=me, opponent=b.player(active=b.pokemon(5, hp=100)))
        assert score_retreat(obs) == constants.SCORE_RETREAT_PROMOTE_LETHAL

    def test_escapes_a_ko(self):
        me = b.player(active=b.pokemon(5, hp=40, max_hp=100), bench=[b.pokemon(1)])
        opponent = b.player(active=b.pokemon(1, energies=[FIGHTING], hp=999))
        obs = b.observation(me=me, opponent=opponent)
        assert score_retreat(obs) == constants.SCORE_RETREAT_ESCAPE_KO

    def test_healthy_active_stays(self):
        me = b.player(active=b.pokemon(1), bench=[b.pokemon(5)])
        obs = b.observation(me=me, opponent=b.player(active=b.pokemon(5, hp=999)))
        assert score_retreat(obs) == constants.SCORE_RETREAT_NEVER


class TestScoreCard:
    def test_promote_prefers_ready_attackers(self):
        ready = b.pokemon(1, energies=[FIGHTING])
        me = b.player(bench=[ready], hand=[b.hand_card(1)])
        opponent = b.player(active=b.pokemon(5, hp=999))
        obs = b.observation(me=me, opponent=opponent, context=SelectContext.SWITCH)
        on_bench = score_card(b.option(OptionType.CARD, area=AreaType.BENCH, index=0), obs)
        in_hand = score_card(b.option(OptionType.CARD, area=AreaType.HAND, index=0), obs)
        # M7.2b race term: the benched copy holds 1 of the 2 energies its best
        # attack needs (one attach closer than the hand copy) + can hit now.
        assert on_bench == in_hand + constants.PROMOTE_READY_BONUS \
            + constants.PROMOTE_TURN_PENALTY

    def test_keep_and_discard_mirror_usefulness(self):
        me = b.player(hand=[b.hand_card(6)])  # energy: usefulness 250
        keep_obs = b.observation(me=me, context=SelectContext.TO_HAND)
        drop_obs = b.observation(me=me, context=SelectContext.DISCARD)
        card_option = b.option(OptionType.CARD, area=AreaType.HAND, index=0)
        assert score_card(card_option, keep_obs) == constants.USEFULNESS_ENERGY
        assert score_card(card_option, drop_obs) == -constants.USEFULNESS_ENERGY

    def test_target_the_highest_prize_pokemon(self):
        opponent = b.player(active=b.pokemon(3), bench=[b.pokemon(5)])
        obs = b.observation(opponent=opponent, context=SelectContext.DAMAGE)
        active_opt = b.option(OptionType.CARD, area=AreaType.ACTIVE, index=0,
                              player_index=1)
        bench_opt = b.option(OptionType.CARD, area=AreaType.BENCH, index=0,
                             player_index=1)
        assert score_card(active_opt, obs) == 3 * constants.TARGET_PRIZE_WEIGHT
        assert score_card(bench_opt, obs) == 1 * constants.TARGET_PRIZE_WEIGHT

    def test_unknown_context_is_neutral_and_unresolved_card_is_zero(self):
        me = b.player(hand=[b.hand_card(7)])
        obs = b.observation(me=me, context=SelectContext.MAIN)
        card_option = b.option(OptionType.CARD, area=AreaType.HAND, index=0)
        assert score_card(card_option, obs) == constants.SCORE_CARD_NEUTRAL
        missing = b.option(OptionType.CARD, area=AreaType.HAND, index=9)
        assert score_card(missing, obs) == 0


def test_score_option_dispatches_flat_tiers_and_fallbacks():
    obs = b.observation()
    assert score_option(b.option(OptionType.ABILITY), obs) == constants.SCORE_ABILITY
    assert score_option(b.option(OptionType.EVOLVE), obs) == constants.SCORE_EVOLVE
    assert score_option(b.option(OptionType.YES), obs) == 40
    assert score_option(b.option(OptionType.TOOL_CARD), obs) == 0  # not in the dict


class TestRaceScoring:
    """M7.2b: charge the attacker that wins the race; attack when the race is won."""

    def _attach_obs(self, active, bench, opponent_active):
        me = b.player(active=active, bench=list(bench))
        return b.observation(me=me, opponent=b.player(active=opponent_active))

    def test_attach_prefers_the_race_winning_active(self):
        # card 3 (270, needs FF) closes vs a 999hp wall in 5 turns; card 1 (120)
        # needs 10 — the active is the closer, the bench copy is not.
        obs = self._attach_obs(b.pokemon(3), [b.pokemon(1)], b.pokemon(5, hp=999))
        active_attach = b.option(OptionType.ATTACH, in_play_area=AreaType.ACTIVE,
                                 in_play_index=0)
        bench_attach = b.option(OptionType.ATTACH, in_play_area=AreaType.BENCH,
                                in_play_index=0)
        assert score_attach(active_attach, obs) \
            == constants.SCORE_ATTACH_RACE_CLOSER_ACTIVE + 2
        assert score_attach(bench_attach, obs) == constants.SCORE_ATTACH_BENCH_BASE + 1

    def test_attach_prefers_the_race_winning_bench_once_the_active_is_ready(self):
        # With an attack-READY active (card 1 at [F,F]: its best attack is
        # charged), the benched closer outranks further active investment:
        # keep charging THE ONE attacker (the attach doesn't end the turn).
        obs = self._attach_obs(b.pokemon(1, energies=[FIGHTING, FIGHTING]),
                               [b.pokemon(3)], b.pokemon(5, hp=999))
        bench_attach = b.option(OptionType.ATTACH, in_play_area=AreaType.BENCH,
                                in_play_index=0)
        assert score_attach(bench_attach, obs) \
            == constants.SCORE_ATTACH_RACE_CLOSER_BENCH + 2

    def test_bench_closer_suppressed_while_the_active_starves(self):
        # The measured floor-test failure (0.715): an uncharged active can
        # neither attack nor pay retreat, so the bench tier must wait — the
        # active gets fed first.
        obs = self._attach_obs(b.pokemon(1), [b.pokemon(3)], b.pokemon(5, hp=999))
        active_attach = b.option(OptionType.ATTACH, in_play_area=AreaType.ACTIVE,
                                 in_play_index=0)
        bench_attach = b.option(OptionType.ATTACH, in_play_area=AreaType.BENCH,
                                in_play_index=0)
        assert score_attach(bench_attach, obs) == constants.SCORE_ATTACH_BENCH_BASE + 2
        assert score_attach(active_attach, obs) == constants.SCORE_ATTACH_ACTIVE_BASE + 1
        assert score_attach(active_attach, obs) > score_attach(bench_attach, obs)

    def test_promote_ranks_by_attaches_still_needed(self):
        charged = b.pokemon(3, energies=[FIGHTING, FIGHTING])
        uncharged = b.pokemon(3)
        me = b.player(bench=[charged, uncharged])
        obs = b.observation(me=me, opponent=b.player(active=b.pokemon(5, hp=999)),
                            context=SelectContext.SWITCH)
        pick = lambda i: score_card(  # noqa: E731
            b.option(OptionType.CARD, area=AreaType.BENCH, index=i), obs)
        assert pick(0) == 270 + constants.PROMOTE_READY_BONUS  # q + ready, gap 0
        assert pick(1) == 270 - 2 * constants.PROMOTE_TURN_PENALTY  # gap 2

    def test_close_mode_attacks_instead_of_milling(self):
        # Agent-level: vs an all-harmless board, the chip attack outranks the
        # draw trainer that used to mill the deck (the M6 floor-test loss).
        me = b.player(active=b.pokemon(1, energies=[FIGHTING]),
                      hand=[b.hand_card(7)], hand_count=5, deck_count=20)
        obs = b.observation(me=me, opponent=b.player(active=b.pokemon(5, hp=200)))
        attack = score_option(b.option(OptionType.ATTACK, attack_id=101), obs)
        trainer = score_option(b.option(OptionType.PLAY, area=AreaType.HAND, index=0), obs)
        assert attack == constants.SCORE_CHIP_CLOSE_BASE + 5.0
        assert attack > trainer
