"""Damage math: can_afford / best_damage, and the M7.2b race primitives."""
from tcg.combat import (best_damage, can_afford, charged_best, hits_to_ko,
                        turns_to_first_ko, turns_to_ready)
from tcg.constants import UNREACHABLE_TURNS
from tests import builders as b

FIGHTING, WATER, PSYCHIC, COLORLESS = 6, 3, 5, 0


class TestCanAfford:
    def test_empty_cost_is_free(self):
        assert can_afford([], [])
        assert can_afford([FIGHTING], [])

    def test_typed_slot_needs_matching_energy(self):
        assert can_afford([FIGHTING], [FIGHTING])
        assert not can_afford([WATER], [FIGHTING])
        assert not can_afford([], [FIGHTING])

    def test_colorless_paid_by_any_energy(self):
        assert can_afford([WATER, PSYCHIC], [COLORLESS, COLORLESS])
        assert not can_afford([WATER], [COLORLESS, COLORLESS])

    def test_colorless_paid_by_leftover_typed_energy(self):
        assert can_afford([FIGHTING, FIGHTING], [FIGHTING, COLORLESS])
        assert not can_afford([FIGHTING], [FIGHTING, COLORLESS])

    def test_typed_energy_not_double_spent(self):
        assert not can_afford([FIGHTING], [FIGHTING, FIGHTING])


class TestBestDamage:
    def test_none_participants(self):
        target = b.pokemon(5)
        assert best_damage(None, target) == 0
        assert best_damage(target, None) == 0

    def test_unknown_attacker_id(self):
        assert best_damage(b.pokemon(12345), b.pokemon(5)) == 0

    def test_picks_best_affordable_attack(self):
        attacker = b.pokemon(1, energies=[FIGHTING])          # only the 50 hit
        assert best_damage(attacker, b.pokemon(5)) == 50
        attacker = b.pokemon(1, energies=[FIGHTING, WATER])   # 120 affordable too
        assert best_damage(attacker, b.pokemon(5)) == 120

    def test_weakness_doubles(self):
        attacker = b.pokemon(4, energies=[PSYCHIC])  # 30, colorless cost
        assert best_damage(attacker, b.pokemon(1)) == 60  # card 1 weak to Psychic

    def test_resistance_subtracts_30(self):
        attacker = b.pokemon(1, energies=[FIGHTING, FIGHTING])
        assert best_damage(attacker, b.pokemon(2)) == 90  # 120 - 30

    def test_resistance_is_floored_at_zero(self):
        attacker = b.pokemon(10, energies=[FIGHTING])  # 20 dmg into -30 resistance
        assert best_damage(attacker, b.pokemon(2)) == 0

    def test_zero_damage_attacks_ignored(self):
        assert best_damage(b.pokemon(5, energies=[FIGHTING] * 3), b.pokemon(1)) == 0

    def test_attack_ids_missing_from_table_ignored(self):
        assert best_damage(b.pokemon(9, energies=[FIGHTING] * 3), b.pokemon(1)) == 0

    def test_extra_energy_simulates_an_attach(self):
        attacker = b.pokemon(3, energies=[FIGHTING])  # 270 needs FF
        assert best_damage(attacker, b.pokemon(5)) == 0
        assert best_damage(attacker, b.pokemon(5), extra_energy=1) == 270

    def test_unknown_target_takes_plain_damage(self):
        attacker = b.pokemon(1, energies=[FIGHTING])
        assert best_damage(attacker, b.pokemon(12345)) == 50


class TestRaceMath:
    """M7.2b primitives: what a Pokémon is worth at full charge, and how many
    turns until it takes its first KO (attach-1/turn model)."""

    def test_charged_best_ignores_affordability(self):
        assert charged_best(b.pokemon(3), b.pokemon(5)) == (270, 2)  # no energy attached
        assert charged_best(b.pokemon(3), b.pokemon(2)) == (240, 2)  # resisted -30

    def test_charged_best_applies_weakness_and_prefers_best_not_cheapest(self):
        assert charged_best(b.pokemon(4), b.pokemon(1)) == (60, 1)   # 30 x2 weakness
        assert charged_best(b.pokemon(1), None) == (120, 2)          # raw printed, best attack

    def test_charged_best_degenerate_cases(self):
        assert charged_best(b.pokemon(5), b.pokemon(1)) == (0, 0)    # zero-damage only
        assert charged_best(b.pokemon(9), b.pokemon(1)) == (0, 0)    # attacks missing from table
        assert charged_best(b.pokemon(12345), b.pokemon(1)) == (0, 0)
        assert charged_best(None, b.pokemon(1)) == (0, 0)

    def test_turns_to_ready_counts_missing_attaches(self):
        assert turns_to_ready(b.pokemon(3)) == 2                     # 270 needs FF
        assert turns_to_ready(b.pokemon(3, energies=[FIGHTING])) == 1
        assert turns_to_ready(b.pokemon(3, energies=[FIGHTING] * 2)) == 0
        assert turns_to_ready(b.pokemon(3, energies=[FIGHTING] * 3)) == 0

    def test_turns_to_ready_sentinel_and_hand_cards(self):
        assert turns_to_ready(b.pokemon(5)) == UNREACHABLE_TURNS     # can never damage
        assert turns_to_ready(b.hand_card(1)) == 2                   # no .energies -> 0 attached

    def test_hits_to_ko_ceils(self):
        assert hits_to_ko(b.pokemon(3), b.pokemon(1, hp=100)) == 1
        assert hits_to_ko(b.pokemon(3), b.pokemon(1, hp=300)) == 2   # ceil(300/270)
        assert hits_to_ko(b.pokemon(1), b.pokemon(2, hp=200)) == 3   # ceil(200/90) resisted

    def test_hits_to_ko_sentinel(self):
        assert hits_to_ko(b.pokemon(10), b.pokemon(2, hp=50)) == UNREACHABLE_TURNS  # 20-30 -> 0
        assert hits_to_ko(b.pokemon(5), b.pokemon(1, hp=10)) == UNREACHABLE_TURNS

    def test_turns_to_first_ko_attach_fires_same_turn(self):
        # One energy short still fires THIS turn (attach happens before the
        # attack) — the same semantics as the pilot's 4000 unblock tier.
        assert turns_to_first_ko(b.pokemon(3, energies=[FIGHTING]), b.pokemon(1, hp=100)) == 1
        assert turns_to_first_ko(b.pokemon(3), b.pokemon(1, hp=100)) == 2          # gap 2
        assert turns_to_first_ko(b.pokemon(1), b.pokemon(5, hp=240)) == 3          # g=2,h=2
        assert turns_to_first_ko(b.pokemon(5), b.pokemon(1, hp=10)) == UNREACHABLE_TURNS


class TestConditionalAttacks:
    """M13 0a: CONDITIONAL_ATTACKS gate attacks on own-board presence —
    rl/tcg twins agree; board_ids=None keeps legacy behavior byte-identical."""

    def _patch(self, monkeypatch):
        import rl.combat as rc
        import tcg.combat as tc
        # fake fact: attack 101 (card 1's 50-dmg) requires card 5 on board
        monkeypatch.setattr(rc, "CONDITIONAL_ATTACKS", {101: 5})
        monkeypatch.setattr(tc, "CONDITIONAL_ATTACKS", {101: 5})

    def test_gated_when_requirement_missing(self, monkeypatch):
        self._patch(monkeypatch)
        import rl.combat as rc
        attacker = b.pokemon(1, energies=[FIGHTING])
        target = b.pokemon(5, hp=100)
        # requirement absent: attack 101 unavailable -> falls to 102 (needs
        # F+C, unaffordable with one energy) -> 0 damage
        assert best_damage(attacker, target, board_ids=set()) == 0
        assert rc._best_damage(attacker, target, board_ids=set()) == 0
        # requirement on board: 101 available again
        assert best_damage(attacker, target, board_ids={5}) == 50
        assert rc._best_damage(attacker, target, board_ids={5}) == 50

    def test_none_board_ids_is_legacy(self, monkeypatch):
        self._patch(monkeypatch)
        import rl.combat as rc
        attacker = b.pokemon(1, energies=[FIGHTING])
        target = b.pokemon(5, hp=100)
        assert best_damage(attacker, target) == 50
        assert rc._best_damage(attacker, target) == 50
        assert turns_to_ready(attacker, target) == \
            rc._turns_to_ready(attacker, target)

    def test_charged_best_and_readiness_gated(self, monkeypatch):
        self._patch(monkeypatch)
        import rl.combat as rc
        attacker = b.pokemon(1)                      # no energy
        target = b.pokemon(5, hp=100)
        # without card 5: only attack 102 (120 dmg, 2-cost) remains charged-best
        assert charged_best(attacker, target, board_ids=set()) == (120, 2)
        assert rc._charged_best(attacker, target, board_ids=set()) == (120, 2)
        assert turns_to_ready(attacker, target, board_ids=set()) == 2

    def test_attach_scorer_respects_conditional_attacks(self, monkeypatch):
        # M14 replay fix: a conditional attacker missing its requirement hits
        # the non-attacker floor in BOTH twins' attach scorers.
        self._patch_single(monkeypatch)
        import rl.generic_pilot as rlp
        import tcg.pilot as tp
        from tcg import constants as tc
        from tests import builders as b
        from tests.fake_cg import AreaType, OptionType
        me = b.player(active=b.pokemon(10), hand=[b.hand_card(6)])
        obs = b.observation(me=me, opponent=b.player(active=b.pokemon(5, hp=200)))
        opt = b.option(OptionType.ATTACH, area=AreaType.HAND, index=0,
                       in_play_area=AreaType.ACTIVE, in_play_index=0)
        assert tp.score_attach(opt, obs) == tc.SCORE_ATTACH_NON_ATTACKER
        assert rlp.score_attach(opt, obs, me) == 400
        # requirement on board: attacker again (scores above the floor)
        me2 = b.player(active=b.pokemon(10), bench=[b.pokemon(5)],
                       hand=[b.hand_card(6)])
        obs2 = b.observation(me=me2, opponent=b.player(active=b.pokemon(5, hp=200)))
        assert tp.score_attach(opt, obs2) > tc.SCORE_ATTACH_NON_ATTACKER
        assert rlp.score_attach(opt, obs2, me2) > 400

    def _patch_single(self, monkeypatch):
        import rl.combat as rc
        import tcg.combat as tc_
        # card 10's only attack (107) requires card 5 on board
        monkeypatch.setattr(rc, "CONDITIONAL_ATTACKS", {107: 5})
        monkeypatch.setattr(tc_, "CONDITIONAL_ATTACKS", {107: 5})
