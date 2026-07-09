"""Old-vs-new parity: rl/encoders.py vs tcg/encoders.py.

Both modules build their FEAT matrix from the real data/cards_features.parquet
and their combat tables from the fake engine; every encoding must be
bit-identical (np.array_equal, no tolerance).
"""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("polars")

import rl.encoders as old
import tcg.encoders as new
from tests.builders import hand_card, observation, option, player, pokemon
from tests.fake_cg import AreaType, EnergyType, OptionType, SelectContext

FIGHTING = EnergyType.FIGHTING
WATER = EnergyType.WATER
PSYCHIC = EnergyType.PSYCHIC


def test_feature_table_parity():
    assert np.array_equal(old.FEAT, new.FEAT)
    assert old.FEAT_DIM == new.FEAT_DIM
    assert old.SLOT_DIM == new.SLOT_DIM
    assert old.STATE_DIM == new.STATE_DIM
    assert old.OPTION_DIM == new.OPTION_DIM
    assert (old.N_BENCH, old.N_CONTEXTS, old.N_OPTION_TYPES, old.N_ENERGY,
            old.N_STATUS, old.N_COMBAT) == (new.N_BENCH, new.N_CONTEXTS,
                                            new.N_OPTION_TYPES, new.N_ENERGY,
                                            new.N_STATUS, new.N_COMBAT)
    assert old.BASIC_FIGHTING_ENERGY == new.BASIC_FIGHTING_ENERGY


# --- state encodings over a battery of boards --------------------------------

def board_empty():
    return observation()


def board_full():
    me = player(
        active=pokemon(1, hp=120, max_hp=140, energies=[FIGHTING, FIGHTING],
                       tools=[hand_card(7)]),
        bench=[pokemon(3, hp=220, energies=[FIGHTING]),
               pokemon(5, hp=60), None,
               pokemon(10, hp=30, energies=[WATER, PSYCHIC])],
        hand=[hand_card(6), hand_card(7), hand_card(2)],
        discard=[hand_card(6), hand_card(6), hand_card(1)],
        deck_count=17, prizes_remaining=4,
    )
    opponent = player(
        active=pokemon(2, hp=90, max_hp=180, energies=[WATER]),
        bench=[pokemon(4, hp=70, energies=[PSYCHIC, PSYCHIC])],
        discard=[hand_card(8)],
        hand_count=9, deck_count=31, prizes_remaining=6,
    )
    return observation(me, opponent, turn=12, energy_attached=True,
                       stadium=[hand_card(7)])


def board_statuses():
    me = player(active=pokemon(1, energies=[FIGHTING]),
                poisoned=1, burned=1, asleep=1)
    opponent = player(active=pokemon(3, hp=10), paralyzed=1, confused=1,
                      prizes_remaining=1)
    return observation(me, opponent, turn=40, supporter_played=True)


def board_seat_swapped():
    me = player(active=pokemon(4, hp=50, energies=[PSYCHIC]),
                discard=[hand_card(6)] * 4)
    opponent = player(active=pokemon(1, hp=140, energies=[FIGHTING] * 3),
                      bench=[pokemon(9), pokemon(5)])
    return observation(me, opponent, your_index=1, turn=7)


def board_no_actives():
    return observation(player(bench=[pokemon(5)]), player(deck_count=0))


@pytest.mark.parametrize("board", [board_empty, board_full, board_statuses,
                                   board_seat_swapped, board_no_actives])
def test_encode_state_parity(board):
    obs = board()
    assert np.array_equal(old.encode_state(obs.current),
                          new.encode_state(obs.current))


@pytest.mark.parametrize("board", [board_empty, board_full, board_statuses,
                                   board_seat_swapped, board_no_actives])
def test_combat_features_parity(board):
    obs = board()
    assert np.array_equal(old._combat_features(obs.current),
                          new.combat_features(obs.current))


def test_combat_features_one_energy_from_ko():
    # Card 3's 270-damage attack needs FF; with one F attached, one more
    # energy unlocks a KO — the `one_energy_from_ko` lookahead feature.
    me = player(active=pokemon(3, energies=[FIGHTING]))
    opponent = player(active=pokemon(2, hp=100))
    obs = observation(me, opponent)
    old_vec = old._combat_features(obs.current)
    new_vec = new.combat_features(obs.current)
    assert np.array_equal(old_vec, new_vec)
    assert new_vec[3] == 1.0  # the scenario actually exercises the feature


# --- option encodings ---------------------------------------------------------

def selection_board():
    me = player(
        active=pokemon(1, energies=[FIGHTING]),
        bench=[pokemon(3), None, pokemon(5)],
        hand=[hand_card(6), hand_card(2)],
        discard=[hand_card(7)],
    )
    opponent = player(active=pokemon(2), hand=[hand_card(8)])
    return observation(me, opponent, select_deck=[hand_card(4), hand_card(9)],
                       stadium=[hand_card(7)], looking=[hand_card(10), None])


OPTION_CASES = [
    # explicit cardId wins over (area, index)
    option(OptionType.CARD, card_id=3, area=int(AreaType.HAND), index=0),
    # resolved through every zone the encoder knows
    option(OptionType.CARD, area=int(AreaType.DECK), index=1),
    option(OptionType.CARD, area=int(AreaType.HAND), index=1),
    option(OptionType.CARD, area=int(AreaType.DISCARD), index=0),
    option(OptionType.CARD, area=int(AreaType.ACTIVE), index=0),
    option(OptionType.CARD, area=int(AreaType.BENCH), index=0),
    option(OptionType.CARD, area=int(AreaType.PRIZE), index=0),      # hidden -> None
    option(OptionType.CARD, area=int(AreaType.STADIUM), index=0),
    option(OptionType.CARD, area=int(AreaType.LOOKING), index=0),
    option(OptionType.CARD, area=int(AreaType.LOOKING), index=1),    # None slot
    option(OptionType.CARD, area=8, index=0),                        # unknown area
    option(OptionType.CARD, area=int(AreaType.BENCH), index=99),     # out of range
    option(OptionType.CARD, area=int(AreaType.BENCH), index=1),      # empty bench slot
    # opponent's zone via playerIndex
    option(OptionType.CARD, area=int(AreaType.HAND), index=0, player_index=1),
    # ATTACH/EVOLVE target encoding, active vs bench
    option(OptionType.ATTACH, area=int(AreaType.HAND), index=0,
           in_play_area=int(AreaType.ACTIVE), in_play_index=0),
    option(OptionType.ATTACH, area=int(AreaType.HAND), index=0,
           in_play_area=int(AreaType.BENCH), in_play_index=0),
    option(OptionType.EVOLVE, card_id=3, in_play_area=int(AreaType.BENCH),
           in_play_index=99),                                        # unresolved target
    option(OptionType.END),
    option(OptionType.ATTACK, attack_id=101),
]


@pytest.mark.parametrize("opt", OPTION_CASES)
def test_encode_option_parity(opt):
    obs = selection_board()
    assert np.array_equal(old.encode_option(opt, obs), new.encode_option(opt, obs))


def test_encode_option_parity_seat_swapped():
    me = player(hand=[hand_card(3)])
    opponent = player(active=pokemon(2))
    obs = observation(me, opponent, your_index=1)
    opt = option(OptionType.CARD, area=int(AreaType.HAND), index=0)
    assert np.array_equal(old.encode_option(opt, obs), new.encode_option(opt, obs))


def test_card_id_at_parity_over_all_areas():
    obs = selection_board()
    for area in [1, 2, 3, 4, 5, 6, 7, 8, 12]:
        for index in [0, 1, 5]:
            for player_index in [0, 1]:
                assert (old._card_id_at(obs, area, index, player_index)
                        == new.card_id_at(obs, area, index, player_index))


@pytest.mark.parametrize("context", list(SelectContext))
def test_encode_context_parity(context):
    assert np.array_equal(old.encode_context(context), new.encode_context(context))
