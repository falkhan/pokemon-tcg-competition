"""Damage math: can_afford / best_damage."""
from tcg.combat import best_damage, can_afford
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
