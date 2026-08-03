"""The M41 energy-ceiling primitives (docs/M41.md, 2026-08-03).

Two properties are load-bearing, and both are asserted without the real engine
wherever possible — `tests/conftest.py` installs a 10-card stub, so anything
that needs the true pool reads the PARQUETS via `tcg.cardpool` (the
`tests/test_scaling.py` pattern) rather than skipping.

1. `energy_is_dead` must never ban a charge that a real attack or retreat needs.
   Two earlier drafts did. Both asked a DAMAGE question and both were falsified
   on live replays by Fezandipiti ex's Cruel Arrow — cost 3, printed damage 0,
   because all 100 of its damage lives in the effect text. The stub happens to
   contain a card with exactly that shape (card 1: a 1-cost and a 2-cost
   attack), so the regression runs in CI.
2. The shipped policies stay untouched: `scaling` defaults to False everywhere,
   the M19 block keeps its exact (blind) values, and the new option columns are
   APPENDED so every trained checkpoint truncates them away.
"""
import re

import pytest

from rl import combat
from tests.fake_cg import EnergyType

FIGHTING, WATER, COLORLESS = (int(EnergyType.FIGHTING), int(EnergyType.WATER),
                              int(EnergyType.COLORLESS))


# --- energy_is_dead on the stub: the incremental-charging regression ---------
# Stub card 1 has attacks 101 {F} and 102 {F}{C}; card 4 has 104 {C}; card 5's
# only attack costs nothing; every stub card's retreatCost is absent (-> 0).

def test_a_pokemon_is_not_finished_while_a_costlier_attack_is_unpaid():
    """THE Cruel Arrow regression. Draft v2 asked "does the NEXT energy unlock
    an attack?" — at 1 energy nothing unlocks a 3-cost attack in one step, so it
    declared the Pokemon finished and would have banned the charge. Here: card 1
    at one {F} can pay 101 but not 102, so it must stay open."""
    assert not combat.energy_is_dead(1, [])
    assert not combat.energy_is_dead(1, [FIGHTING]), \
        "banned charging toward the 2-cost attack from 1 energy"
    assert combat.energy_is_dead(1, [FIGHTING, FIGHTING])


def test_off_type_energy_does_not_pay_a_typed_cost():
    """Counting attachments instead of checking affordability would call this
    Pokemon finished while holding energy that cannot pay for anything."""
    assert not combat.energy_is_dead(1, [WATER, WATER])


def test_a_free_attack_is_paid_from_the_start():
    assert combat.energy_is_dead(5, [])          # card 5's only attack costs {}


def test_a_colorless_cost_is_paid_by_any_type():
    assert not combat.energy_is_dead(4, [])
    assert combat.energy_is_dead(4, [WATER])


def test_an_unknown_card_has_no_attacks_to_pay_for():
    assert combat.energy_is_dead(-1, [])


def test_a_pokemon_whose_attacks_are_all_missing_from_the_table_is_dead():
    """Card 9's only attack id is absent from _ATK — nothing to pay for."""
    assert combat.energy_is_dead(9, [])


def test_retreat_cost_holds_the_ceiling_open(monkeypatch):
    """The one legitimate reason to charge past the attack cost. Stub cards
    carry no retreatCost, so this pins the branch directly."""
    monkeypatch.setitem(combat._RETREAT, 4, 3)
    assert not combat.energy_is_dead(4, [WATER])          # attack paid, retreat not
    assert not combat.energy_is_dead(4, [WATER] * 2)
    assert combat.energy_is_dead(4, [WATER] * 3)


def test_an_own_energy_scaler_never_reads_as_dead(monkeypatch):
    """More energy really is more damage, so there is no ceiling at all."""
    monkeypatch.setattr(combat, "OWN_ENERGY_SCALERS", frozenset({104}))
    assert combat.scales_on_own_energy(4)
    for n in range(8):
        assert not combat.energy_is_dead(4, [WATER] * n)


# --- the frozen scaler set, checked against the pool -------------------------

def test_own_energy_scalers_are_re_derivable_from_the_engine():
    """The set is frozen so a pool update cannot silently re-point it (the
    SPREAD_ATTACKS doctrine); this is the re-derivation a hand-curated table
    always needs. Rules text lives only in the engine, not the parquets."""
    from tcg.cardpool import attack_texts
    texts = attack_texts()
    if not any(t.get("text") for t in texts.values()):
        pytest.skip("needs the real engine's rules text (conftest installs a stub)")
    own = re.compile(r"for each .{0,40}energy attached to "
                     r"(this|your active|both|all of your|each of your)",
                     re.IGNORECASE)
    derived = {aid for aid, t in texts.items()
               if t.get("text") and own.search(t["text"])}
    assert derived == combat.OWN_ENERGY_SCALERS


def test_the_audited_near_misses_stay_out():
    """Torrential Pump / Jungle Whip / Chrono Burst / Sonic Ripper mention their
    own attached energy but pay a FLAT bonus for shuffling it away, so their
    printed cost already covers what they consume. Listing them would make the
    rule inert on four decks for no reason."""
    for aid in (136, 236, 1009, 1537):
        assert aid not in combat.OWN_ENERGY_SCALERS


def test_the_scaler_set_is_not_accidentally_empty_or_the_whole_pool():
    assert 20 < len(combat.OWN_ENERGY_SCALERS) < 60


# --- the card facts the rule depends on, pinned from the parquets ------------

def test_powerful_hand_is_a_one_energy_attack_that_prints_zero():
    """The whole defect in one row: cost 1, damage 0. If a pool update ever
    changes either number, the Alakazam ceiling changes with it."""
    from tcg.cardpool import attacks_by_card, cards
    (atk,) = attacks_by_card()[743]
    assert (atk["attack_name"], atk["damage"], atk["cost_total"]) == \
        ("Powerful Hand", 0, 1)
    assert cards()[743]["retreat_cost"] == 1        # ceiling stays at 1


def test_cruel_arrow_prints_zero_yet_costs_three():
    """The falsifier card. A damage-based ceiling reads this as harmless and
    caps Fezandipiti ex at its retreat cost of 1; live replays used it 7 times
    in one ship."""
    from tcg.cardpool import attacks_by_card, cards
    (atk,) = attacks_by_card()[140]
    assert (atk["attack_name"], atk["damage"], atk["cost_total"]) == \
        ("Cruel Arrow", 0, 3)
    assert cards()[140]["retreat_cost"] == 1


# --- scaling stays OFF for every shipped consumer ---------------------------

@pytest.mark.parametrize("fn", ["_charged_best", "_turns_to_ready", "_best_damage"])
def test_scaling_defaults_to_off(fn):
    """Default-off is what keeps the live bundles byte-identical."""
    import inspect
    assert inspect.signature(getattr(combat, fn)).parameters["scaling"].default \
        is False


def test_printed_damage_is_untouched_when_scaling_is_off():
    class Mon:
        def __init__(self, cid, energies=()):
            self.id, self.energies, self.hp, self.maxHp = cid, list(energies), 100, 100

    attacker, target = Mon(1, [FIGHTING, FIGHTING]), Mon(4)
    assert combat._best_damage(attacker, target) == \
        combat._best_damage(attacker, target, scaling=False)
    assert combat._charged_best(attacker, target)[0] == 120


def test_scaling_credits_a_zero_printed_attack(monkeypatch):
    """With scaling on, a zero-printed attack stops reading as harmless — which
    is what un-pins _turns_to_ready from UNREACHABLE on our own win condition."""
    from rl import scaling

    class Mon:
        def __init__(self, cid, energies=()):
            self.id, self.energies, self.hp, self.maxHp = cid, list(energies), 100, 100

    monkeypatch.setitem(scaling.SCALING_ATTACKS, 103, ("hand", 20, 0))
    card5 = Mon(5)                                   # only attack 103, prints 0
    assert combat._charged_best(card5, Mon(1)) == (0, 0)
    assert combat._turns_to_ready(card5, Mon(1)) == combat.UNREACHABLE
    assert combat._charged_best(card5, Mon(1), scaling=True)[0] > 0
    assert combat._turns_to_ready(card5, Mon(1), scaling=True) == 0


# --- the encoder block is APPENDED, never in-place --------------------------

def test_the_new_option_columns_are_appended_after_the_m28_width():
    from rl import encoders
    assert encoders.OPTION_M41_DIM == \
        encoders.OPTION_M28_DIM + encoders.N_OPTION_ENERGY
    assert encoders.N_OPTION_ENERGY == 5


def test_marginal_damage_is_zero_for_a_flat_attacker():
    """Slot 4 only ever speaks about the scaling exception; a flat attacker
    must contribute nothing to it."""
    from rl.encoders import _marginal_energy_damage

    class Mon:
        def __init__(self, cid, energies=()):
            self.id, self.energies, self.hp, self.maxHp = cid, list(energies), 100, 100

    assert _marginal_energy_damage(Mon(1, [FIGHTING]), Mon(2)) == 0


def test_marginal_damage_reports_the_per_energy_step(monkeypatch):
    """With a curated magnitude the slot must report what one more energy buys,
    which is what keeps an own-energy scaler from reading as 'finished' once its
    printed cost is paid."""
    from rl import scaling
    from rl.encoders import _marginal_energy_damage

    class Mon:
        def __init__(self, cid, energies=()):
            self.id, self.energies, self.hp, self.maxHp = cid, list(energies), 100, 100

    monkeypatch.setitem(scaling.SCALING_ATTACKS, 101, ("my_nrg", 30, 0))
    # `effective_damage` floors at the PRINTED number (50 here), so while the
    # scaling term is still under that floor the marginal gain is damped — at 1
    # energy it is max(50,60)-max(50,30) = 10, not the full 30. That damping is
    # a property of the estimate, not of the slot, and it is in the right
    # direction: it under-promises exactly where the estimate is least certain.
    assert _marginal_energy_damage(Mon(1, [FIGHTING]), Mon(2)) == 10
    # once the scaling term clears the floor the slot reports the true step,
    # and keeps reporting it far past the printed cost — the point of the slot
    assert _marginal_energy_damage(Mon(1, [FIGHTING] * 6), Mon(2)) == 30
    assert _marginal_energy_damage(Mon(1, [FIGHTING] * 9), Mon(2)) == 30


def test_an_uncurated_scaler_reports_zero_margin_but_still_flags_itself():
    """OWN_ENERGY_SCALERS knows an attack scales; only SCALING_ATTACKS knows by
    how much. The 19 uncurated ones must fall back on the boolean rather than
    invent a magnitude — every damage guess in this lane has been falsified."""
    from rl.encoders import _marginal_energy_damage
    from rl.scaling import SCALING_ATTACKS

    uncurated = combat.OWN_ENERGY_SCALERS - set(SCALING_ATTACKS)
    assert uncurated, "expected scalers with no curated magnitude"

    class Mon:
        def __init__(self, cid, energies=()):
            self.id, self.energies, self.hp, self.maxHp = cid, list(energies), 100, 100

    assert _marginal_energy_damage(Mon(1, [FIGHTING]), Mon(2)) == 0


def test_the_m19_attach_block_is_untouched():
    """`_attach_extra` feeds live bundles at a slice they still read, so its
    three slots must keep their exact (blind) values. The truthful gap goes in
    the appended block, where trained widths truncate it away."""
    import inspect

    from rl import encoders
    assert "scaling=True" not in inspect.getsource(encoders._attach_extra)
    assert "scaling=True" in inspect.getsource(encoders._attach_energy_ceiling)


def test_scaling_ships_in_the_bundle():
    """encode_option_v2 now reaches rl.scaling through combat on every ATTACH
    option, so the module has to be in the copied package or the agent crashes
    on Kaggle at the first energy card."""
    import inspect

    from tcg import shipping
    assert '"scaling.py"' in inspect.getsource(shipping.export)
