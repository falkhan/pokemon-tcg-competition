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


def test_combat_slice_locates_the_m3_block():
    """COMBAT_SLICE must point exactly at the combat features inside
    encode_state's output — the pre-M3 checkpoint compat loader (bc_v1)
    slices this range out to reconstruct the old encoding."""
    assert old.COMBAT_SLICE == new.COMBAT_SLICE
    start, end = old.COMBAT_SLICE
    assert end - start == old.N_COMBAT
    assert end + 2 * (1 + old.N_BENCH) * old.SLOT_DIM == old.STATE_DIM

    me = player(active=pokemon(1, energies=[FIGHTING]), bench=[pokemon(3)])
    opponent = player(active=pokemon(5, hp=80))
    obs = observation(me=me, opponent=opponent)
    state = old.encode_state(obs.current)
    assert np.array_equal(state[start:end], old._combat_features(obs.current))


# --- Encoders v2 (M7.3): parity + behavior ------------------------------------

V2_DECK = [1] * 10 + [3] * 4 + [6] * 40 + [7] * 6  # a synthetic legal-ish 60


def _v2_boards():
    yield observation(me=player(), opponent=player())                       # empty
    yield observation(                                                      # full-ish
        me=player(active=pokemon(1, energies=[FIGHTING]),
                  bench=[pokemon(3), pokemon(5)],
                  hand=[hand_card(6), hand_card(7)],
                  discard=[hand_card(1)]),
        opponent=player(active=pokemon(2, hp=150), bench=[pokemon(4)]))
    yield observation(                                                      # harmless opp
        me=player(active=pokemon(3, energies=[FIGHTING, FIGHTING])),
        opponent=player(active=pokemon(5, hp=200)))


def test_encode_state_v2_parity_and_shape():
    assert old.STATE_V2_DIM == new.STATE_V2_DIM
    assert (old.N_STATE_IDS, old.N_OPTION_IDS, old.EMBED_DIM, old.N_CARD_IDS) \
        == (new.N_STATE_IDS, new.N_OPTION_IDS, new.EMBED_DIM, new.N_CARD_IDS)
    for obs in _v2_boards():
        a_num, a_ids = old.encode_state_v2(obs.current, V2_DECK)
        b_num, b_ids = new.encode_state_v2(obs.current, V2_DECK)
        assert np.array_equal(a_num, b_num) and np.array_equal(a_ids, b_ids)
        assert a_num.shape == (old.STATE_V2_DIM,) and a_ids.shape == (old.N_STATE_IDS,)
        assert a_num.dtype == np.float32 and a_ids.dtype == np.int32
        # v1 prefix is byte-identical to encode_state (v2 is additive)
        assert np.array_equal(a_num[:old.STATE_DIM], old.encode_state(obs.current))


def test_encode_option_v2_parity_and_ids():
    me = player(active=pokemon(1), hand=[hand_card(7)])
    obs = observation(me=me, opponent=player(active=pokemon(2)))
    attach = option(OptionType.ATTACH, in_play_area=AreaType.ACTIVE, in_play_index=0)
    play = option(OptionType.PLAY, area=AreaType.HAND, index=0)
    for opt in (attach, play):
        a_num, a_ids = old.encode_option_v2(opt, obs)
        b_num, b_ids = new.encode_option_v2(opt, obs)
        assert np.array_equal(a_num, b_num) and np.array_equal(a_ids, b_ids)
        assert a_num.shape == (old.OPTION_M41_DIM,)
        # the v1 prefix stays byte-identical (M16 block is additive)
        assert np.array_equal(a_num[:old.OPTION_DIM], old.encode_option(opt, obs))
    assert old.encode_option_v2(attach, obs)[1].tolist() == [0, 1]  # target = active card 1
    assert old.encode_option_v2(play, obs)[1].tolist() == [7, 0]    # acted card 7


# --- M16 option-identity block ----------------------------------------------

def test_play_option_resolves_hand_card():
    """Live PLAY prompts carry ONLY a hand index (no area) — pre-M16 every
    trainer in hand encoded to the same blank vector (the aliasing defect)."""
    me = player(active=pokemon(1), hand=[hand_card(7), hand_card(6)])
    obs = observation(me=me, opponent=player(active=pokemon(2)))
    p0 = option(OptionType.PLAY, index=0)
    p1 = option(OptionType.PLAY, index=1)
    lo, hi = old.N_OPTION_TYPES, old.N_OPTION_TYPES + old.FEAT_DIM
    for mod in (old, new):
        n0, i0 = mod.encode_option_v2(p0, obs)
        n1, i1 = mod.encode_option_v2(p1, obs)
        assert i0.tolist() == [7, 0] and i1.tolist() == [6, 0]
        assert not np.array_equal(n0, n1)          # distinguishable now
        assert np.array_equal(n0[lo:hi], old.FEAT[7])
        assert not mod.encode_option(p0, obs)[lo:hi].any()  # v1 stays blank
        # legacy path = exact pre-M16 encoding (pinned checkpoints)
        ln, li = mod.encode_option_v2_legacy(p0, obs)
        assert ln.shape == (old.OPTION_DIM,) and li.tolist() == [0, 0]
        assert not ln[lo:hi].any()


def test_attack_option_identity_block():
    """ATTACK options encode (printed dmg, cost, effective dmg vs opp active)
    — attackId was previously not encoded at all."""
    base = old.OPTION_DIM
    # card 1 (Fighting) attacks card 2, which RESISTS Fighting (-30)
    me = player(active=pokemon(1, energies=[FIGHTING, FIGHTING]))
    obs = observation(me=me, opponent=player(active=pokemon(2)))
    a101 = option(OptionType.ATTACK, attack_id=101)   # 50 dmg, 1 cost
    a102 = option(OptionType.ATTACK, attack_id=102)   # 120 dmg, 2 cost
    for mod in (old, new):
        n1, _ = mod.encode_option_v2(a101, obs)
        n2, _ = mod.encode_option_v2(a102, obs)
        assert not np.array_equal(n1, n2)          # distinguishable now
        assert n1[base] == np.float32(50 / 300) and n1[base + 1] == np.float32(1 / 5)
        assert n1[base + 2] == np.float32((50 - 30) / 300)      # resisted
        assert n2[base] == np.float32(120 / 300)
        assert n2[base + 2] == np.float32((120 - 30) / 300)
    # weakness doubling: card 4 (Psychic) attacks card 1 (weak to Psychic)
    obs2 = observation(me=player(active=pokemon(4)),
                       opponent=player(active=pokemon(1)))
    a104 = option(OptionType.ATTACK, attack_id=104)   # 30 dmg
    for mod in (old, new):
        n, _ = mod.encode_option_v2(a104, obs2)
        assert n[base + 2] == np.float32(60 / 300)


def test_number_option_identity_block():
    o = option(OptionType.NUMBER)
    o.number = 3
    obs = observation()
    for mod in (old, new):
        n, _ = mod.encode_option_v2(o, obs)
        assert n[old.OPTION_DIM + 3] == np.float32(0.3)


# --- M19 attach/retreat extra block ------------------------------------------

def test_attach_option_energy_sufficiency_block():
    """ATTACH options encode the target's CURRENT energy state — previously
    printed features only (the live Solrock over-attach blind spot)."""
    base = old.OPTION_DIM
    # active card 1 with 2 energies: charged-best (102, cost 2) is paid
    me = player(active=pokemon(1, energies=[FIGHTING, FIGHTING]),
                bench=[pokemon(4)])                      # bench card 4: 1-cost, empty
    obs = observation(me=me, opponent=player(active=pokemon(2)))
    sat = option(OptionType.ATTACH, in_play_area=AreaType.ACTIVE, in_play_index=0)
    hungry = option(OptionType.ATTACH, in_play_area=AreaType.BENCH, in_play_index=0)
    for mod in (old, new):
        n_sat, _ = mod.encode_option_v2(sat, obs)
        assert n_sat[base:base + 3].tolist() == [
            np.float32(2 / 5), np.float32(0.0), np.float32(1.0)]
        n_hungry, _ = mod.encode_option_v2(hungry, obs)
        assert n_hungry[base:base + 3].tolist() == [
            np.float32(0.0), np.float32(1 / 5), np.float32(0.0)]


def test_retreat_option_utility_block():
    """RETREAT options encode (damage fraction, prizes at risk, bench-ready)
    — previously a bare type one-hot (the save-the-active blind spot)."""
    base = old.OPTION_DIM
    me = player(active=pokemon(3, hp=100, max_hp=340),   # damaged 3-prize Mega
                bench=[pokemon(1, energies=[FIGHTING, FIGHTING])])
    obs = observation(me=me, opponent=player(active=pokemon(5, hp=999)))
    o = option(OptionType.RETREAT)
    for mod in (old, new):
        n, _ = mod.encode_option_v2(o, obs)
        assert n[base] == np.float32(1.0 - 100 / 340)
        assert n[base + 1] == np.float32(1.0)            # 3 prizes / 3
        assert n[base + 2] == np.float32(1.0)            # ready bench
    # unready bench zeroes the flag
    me2 = player(active=pokemon(3, hp=100, max_hp=340), bench=[pokemon(1)])
    obs2 = observation(me=me2, opponent=player(active=pokemon(5, hp=999)))
    for mod in (old, new):
        n, _ = mod.encode_option_v2(o, obs2)
        assert n[base + 2] == np.float32(0.0)


def test_board_ids_layout_and_padding():
    me = player(active=pokemon(1), bench=[pokemon(3), None, pokemon(5)])
    obs = observation(me=me, opponent=player())
    _, ids = old.encode_state_v2(obs.current, V2_DECK)
    assert ids.tolist() == [1, 3, 0, 5, 0, 0] + [0] * 6  # my slots then opp's, 0-padded


def test_deck_pools_subtract_observed_zones():
    deck = [1, 1, 6, 6, 7]
    me = player(active=pokemon(1), hand=[hand_card(6)], discard=[hand_card(7)])
    obs = observation(me=me, opponent=player())
    pools = old._deck_pools(obs.current, deck)
    full, rest = pools[:old.FEAT_DIM], pools[old.FEAT_DIM:]
    expect_full = (old.FEAT[[1, 1, 6, 6, 7]].sum(axis=0) * 0.1).astype(np.float32)
    expect_rest = (old.FEAT[[1, 6]].sum(axis=0) * 0.1).astype(np.float32)  # minus seen
    assert np.allclose(full, expect_full) and np.allclose(rest, expect_rest)


def test_race_features_reflect_the_won_race():
    # my charged card 3 (270 dmg) vs a harmless card 5 wall: race delta positive
    obs = observation(me=player(active=pokemon(3, energies=[FIGHTING, FIGHTING])),
                      opponent=player(active=pokemon(5, hp=200)))
    race = old._race_features(obs.current)
    assert race.shape == (old.N_RACE,)
    assert race[0] == 0.0            # my active ready
    assert race[1] == 0.1            # 1 turn to first KO
    assert race[6] == 1.0            # opponent can never KO (capped)
    assert race[7] > 0.8             # race clearly won


# --- M27 play-precondition block ---------------------------------------------
# docs/M27.md Probe 3: on identical menus the clone plays supporters at 0.51x
# and stadiums at 0.06x the teacher's rate. The 0.505 sample agent expresses
# these as binary preconditions, not preferences (docs/M27-sample-agent-diff.md).

BOSS_ORDERS = 1182   # supporter AND the gust card (rl.plan.GUST_IDS)
HILDA = 1225         # plain supporter
NIGHTTIME_MINE = 1266  # stadium


def _kind_patch(monkeypatch, mod, mapping):
    """Point the module's card-kind table at `mapping`. The test suite runs on
    the fake engine, whose card DB has no supporters or stadiums at all."""
    monkeypatch.setattr(mod, "_PLAY_KIND", dict(mapping), raising=True)


def _play_block(mod, card_id, obs):
    """The M27 tail of encode_option_v2 for a PLAY of `card_id` from hand."""
    me = obs.current.players[obs.current.yourIndex]
    idx = [c.id for c in me.hand].index(card_id)
    opt = option(OptionType.PLAY, area=AreaType.HAND, index=idx)
    num, _ = mod.encode_option_v2(opt, obs)
    return num[mod.OPTION_V3_DIM:]


def test_supporter_slot_tracks_the_once_per_turn_budget(monkeypatch):
    me = player(active=pokemon(1), hand=[hand_card(HILDA)])
    fresh = observation(me=me, opponent=player(active=pokemon(2)),
                        supporter_played=False)
    spent = observation(me=me, opponent=player(active=pokemon(2)),
                        supporter_played=True)
    for mod in (old, new):
        _kind_patch(monkeypatch, mod, {HILDA: "SUPPORTER"})
        assert _play_block(mod, HILDA, fresh)[0] == 1.0
        assert _play_block(mod, HILDA, spent)[0] == 0.0


def test_stadium_slot_tracks_an_occupied_stadium(monkeypatch):
    me = player(active=pokemon(1), hand=[hand_card(NIGHTTIME_MINE)])
    empty = observation(me=me, opponent=player(active=pokemon(2)))
    occupied = observation(me=me, opponent=player(active=pokemon(2)),
                           stadium=[hand_card(NIGHTTIME_MINE)])
    for mod in (old, new):
        _kind_patch(monkeypatch, mod, {NIGHTTIME_MINE: "STADIUM"})
        assert _play_block(mod, NIGHTTIME_MINE, empty)[1] == 1.0
        assert _play_block(mod, NIGHTTIME_MINE, occupied)[1] == 0.0


def test_card_kind_lookup_is_keyed_on_the_enum_NAME():
    """Guards the collision that ints would allow: the fake engine numbers
    CardType 0..3 (TRAINER=1), the real one 0..6 (SUPPORTER=3) — an int-keyed
    table would encode a special energy as a supporter."""
    for mod, table in ((old, old._play_kind()), (new, new.play_kind())):
        assert all(isinstance(v, str) for v in table.values())


def test_gust_slot_fires_when_the_bench_is_the_better_target(monkeypatch):
    """Boss's Orders is worth playing when a BENCHED Pokemon outscores the
    active — a port of the sample agent's `pokemon_score` argmax
    (`plan.target >= 1`), computed from the observation because the plan block
    is inert (docs/M27.md Probe 4). Same card id on both sides, so the
    comparison turns purely on attached energy."""
    def board(active_energies, bench):
        me = player(active=pokemon(1), hand=[hand_card(BOSS_ORDERS)])
        return observation(
            me=me,
            opponent=player(active=pokemon(2, hp=100, energies=active_energies),
                            bench=bench))

    juicy = board([], [pokemon(2, hp=100, energies=[FIGHTING] * 3)])
    worthless = board([FIGHTING] * 3, [pokemon(2, hp=100)])
    empty_bench = board([], [])

    for mod, dmg_name in ((old, "_best_damage"), (new, "best_damage")):
        _kind_patch(monkeypatch, mod, {BOSS_ORDERS: "SUPPORTER"})
        monkeypatch.setattr(mod, dmg_name, lambda a, b, **k: 300)  # lethal either way
        assert _play_block(mod, BOSS_ORDERS, juicy)[2] == 1.0
        assert _play_block(mod, BOSS_ORDERS, worthless)[2] == 0.0
        assert _play_block(mod, BOSS_ORDERS, empty_bench)[2] == 0.0


def test_gust_slot_is_off_for_non_gust_supporters(monkeypatch):
    """Only Boss's Orders (rl.plan.GUST_IDS) gets the bench-target signal."""
    me = player(active=pokemon(1), hand=[hand_card(HILDA)])
    obs = observation(me=me,
                      opponent=player(active=pokemon(2, hp=100),
                                      bench=[pokemon(2, hp=100,
                                                     energies=[FIGHTING] * 3)]))
    for mod in (old, new):
        _kind_patch(monkeypatch, mod, {HILDA: "SUPPORTER"})
        assert _play_block(mod, HILDA, obs)[2] == 0.0


def test_m27_block_is_additive_so_narrower_checkpoints_are_unaffected():
    """The whole width-shim contract: every earlier width must be a byte-identical
    PREFIX of the current one, so a checkpoint trained at any of them stays
    reproducible by truncation. Each appended block extends this chain."""
    me = player(active=pokemon(1), hand=[hand_card(BOSS_ORDERS)])
    obs = observation(me=me, opponent=player(active=pokemon(2)),
                      supporter_played=False)
    opt = option(OptionType.PLAY, area=AreaType.HAND, index=0)
    num, _ = old.encode_option_v2(opt, obs)
    assert num.shape == (old.OPTION_M41_DIM,)
    assert np.array_equal(num[:old.OPTION_DIM], old.encode_option(opt, obs))
    assert old.OPTION_M27_DIM == old.OPTION_V3_DIM + old.N_OPTION_PLAY_PRE
    assert old.OPTION_M28_DIM == old.OPTION_M27_DIM + old.N_OPTION_PHASE
    assert old.OPTION_M41_DIM == old.OPTION_M28_DIM + old.N_OPTION_ENERGY


def test_the_m41_energy_block_is_zero_on_every_non_attach_option():
    """A PLAY/ATTACK/RETREAT option must not pick up energy-ceiling values —
    the block is ATTACH-only, and a stray write there would shift the meaning of
    the columns for the next corpus."""
    me = player(active=pokemon(1), hand=[hand_card(BOSS_ORDERS)])
    obs = observation(me=me, opponent=player(active=pokemon(2)))
    for opt in (option(OptionType.PLAY, area=AreaType.HAND, index=0),
                option(OptionType.RETREAT),
                option(OptionType.ATTACK, attack_id=101)):
        num, _ = old.encode_option_v2(opt, obs)
        assert not num[old.OPTION_M28_DIM:old.OPTION_M41_DIM].any()


def test_the_m41_energy_block_fires_on_an_attach():
    """Stub card 1 holds attacks {F} and {F}{C}: at one Fighting it can pay the
    cheap one but not the dear one, so the ceiling must read NOT dead, and the
    scaling-aware gap must be the one remaining attach."""
    me = player(active=pokemon(1, energies=[FIGHTING]), hand=[hand_card(7)])
    obs = observation(me=me, opponent=player(active=pokemon(2)))
    opt = option(OptionType.ATTACH, in_play_area=AreaType.ACTIVE, in_play_index=0)
    num, _ = old.encode_option_v2(opt, obs)
    block = num[old.OPTION_M28_DIM:old.OPTION_M41_DIM]
    assert block[0] == 0.0                       # not dead: {F}{C} still unpaid
    assert block[1] == pytest.approx(1 / 5.0)    # one attach short of the best
    assert block[3] == 0.0                       # not an own-energy scaler
