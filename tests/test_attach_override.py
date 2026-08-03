"""M26 attach-override + M30 deck-economy arms (rl/plan.apply_*_overrides).

Both pilots (rl/matchrunner fn3/fn4 and submission/main.py) import THESE
functions and feed them the model's full greedy order — each predicate has a
single source of truth, so parity reduces to (a) these behavioral cases and
(b) the twin files staying byte-identical (test_bundle_twins below).
"""
from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")

from cg.api import AreaType, EnergyType, OptionType, SelectContext

FIGHTING = EnergyType.FIGHTING
from rl.plan import (ATTACH_FIX_BACKSTOP, ATTACH_FIX_DEADENERGY,
                     ATTACH_FIX_TELEPATH,
                     ENHANCED_HAMMER_ID, FEZANDIPITI_ID,
                     HILDA_ID, POFFIN_ID, PLAY_FIX_ASH, PLAY_FIX_ASHGUARD,
                     PLAY_FIX_BENCHFLOOR, PLAY_FIX_CONSERVE,
                     PLAY_FIX_DECKGUARD, PLAY_FIX_DRAWFLOOR,
                     PLAY_FIX_GUSTVETO, PLAY_FIX_HAMMER,
                     PLAY_FIX_POFFINFLOOR, PLAY_FIX_RACEMODE,
                     PLAY_FIX_RACEMODER, PLAY_FIX_TEMPO, SACRED_ASH_ID,
                     TELEPATH_ID, apply_attach_overrides, apply_play_overrides)

BASIC_P = 4          # Basic {P} Energy — any basic energy id works for tests
NON_ENERGY = 741     # Abra
DUDUNSPARCE = 66


@pytest.fixture(autouse=True)
def _energy_ids(monkeypatch):
    """The suite runs on tests/fake_cg, whose card table doesn't carry the
    real energy ids — pin the card-fact set the predicate reads."""
    import rl.plan as rp
    monkeypatch.setattr(rp, "_IS_ENERGY", {BASIC_P, TELEPATH_ID})
    # fake_cg lacks the real basic-Pokemon table; pin Abra (NON_ENERGY) as one.
    monkeypatch.setattr(rp, "_IS_BASIC_POKEMON", {NON_ENERGY})


def _card(cid):
    return SimpleNamespace(id=cid)


def _opt(otype, index=None, area=AreaType.HAND, card_id=None):
    return SimpleNamespace(type=otype, index=index, area=area, cardId=card_id)


def _obs(options, hand, bench=(None,) * 5, energy_attached=False,
         context=SelectContext.MAIN, bench_max=5, deck_count=40, active=(),
         opp_prizes=6, opp_active=(), opp_bench=(), opp_deck_count=40):
    me = SimpleNamespace(hand=[_card(c) for c in hand],
                         bench=list(bench), benchMax=bench_max,
                         deckCount=deck_count, active=list(active))
    op = SimpleNamespace(prize=[object()] * opp_prizes,
                         active=[_card(c) for c in opp_active],
                         bench=[_card(c) if c is not None else None
                                for c in opp_bench],
                         deckCount=opp_deck_count)
    current = SimpleNamespace(players=[me, op], yourIndex=0,
                              energyAttached=energy_attached)
    select = SimpleNamespace(context=context, option=options)
    return SimpleNamespace(current=current, select=select)


BOTH = frozenset({ATTACH_FIX_TELEPATH, ATTACH_FIX_BACKSTOP})


def test_o1_moves_telepath_attach_to_front():
    opts = [_opt(OptionType.ATTACK), _opt(OptionType.ATTACH, 0),
            _opt(OptionType.ATTACH, 1), _opt(OptionType.END)]
    obs = _obs(opts, hand=[BASIC_P, TELEPATH_ID])
    out = apply_attach_overrides(obs, [0, 1, 2, 3],
                                 frozenset({ATTACH_FIX_TELEPATH}))
    assert out == [2, 0, 1, 3]          # Telepath attach (idx 2) forced first


def test_o1_respects_full_bench_and_energy_attached():
    opts = [_opt(OptionType.ATTACK), _opt(OptionType.ATTACH, 0)]
    full = _obs(opts, hand=[TELEPATH_ID], bench=[_card(NON_ENERGY)] * 5)
    assert apply_attach_overrides(
        full, [0, 1], frozenset({ATTACH_FIX_TELEPATH})) == [0, 1]
    attached = _obs(opts, hand=[TELEPATH_ID], energy_attached=True)
    assert apply_attach_overrides(attached, [0, 1], BOTH) == [0, 1]


def test_o2_backstop_fires_on_turn_ending_choice():
    opts = [_opt(OptionType.ATTACK), _opt(OptionType.ATTACH, 0),
            _opt(OptionType.END)]
    obs = _obs(opts, hand=[BASIC_P])
    fixes = frozenset({ATTACH_FIX_BACKSTOP})
    # model wants ATTACK -> attach (model's best attach) goes first
    assert apply_attach_overrides(obs, [0, 1, 2], fixes) == [1, 0, 2]
    # model wants END -> same
    assert apply_attach_overrides(obs, [2, 0, 1], fixes) == [1, 2, 0]
    # model already picked a non-terminal action -> untouched
    play = [_opt(OptionType.PLAY, 0)] + opts
    obs2 = _obs(play, hand=[NON_ENERGY, BASIC_P])
    assert apply_attach_overrides(obs2, [0, 2, 1, 3], fixes) == [0, 2, 1, 3]


def test_o2_ignores_non_energy_attach():
    # ATTACH of a TOOL (non-energy card) must not satisfy the backstop
    opts = [_opt(OptionType.ATTACK), _opt(OptionType.ATTACH, 0)]
    obs = _obs(opts, hand=[NON_ENERGY])
    assert apply_attach_overrides(
        obs, [0, 1], frozenset({ATTACH_FIX_BACKSTOP})) == [0, 1]


def test_off_by_default_and_non_main_untouched():
    opts = [_opt(OptionType.ATTACK), _opt(OptionType.ATTACH, 0)]
    obs = _obs(opts, hand=[TELEPATH_ID])
    assert apply_attach_overrides(obs, [0, 1], frozenset()) == [0, 1]
    sub = _obs(opts, hand=[TELEPATH_ID], context=SelectContext.DISCARD)
    assert apply_attach_overrides(sub, [0, 1], BOTH) == [0, 1]


def test_o1_wins_over_o2_when_both_on():
    opts = [_opt(OptionType.ATTACK), _opt(OptionType.ATTACH, 0),
            _opt(OptionType.ATTACH, 1)]
    obs = _obs(opts, hand=[BASIC_P, TELEPATH_ID])
    # composed arm: telepath (idx 2) preferred over the model's best attach (1)
    assert apply_attach_overrides(obs, [0, 1, 2], BOTH) == [2, 0, 1]


# --- M30 deck-economy arms (apply_play_overrides) ---------------------------

ECO = frozenset({PLAY_FIX_TEMPO, PLAY_FIX_DECKGUARD, PLAY_FIX_ASH})


def test_o3_tempo_fires_only_on_end_pick():
    opts = [_opt(OptionType.END), _opt(OptionType.PLAY, 0),
            _opt(OptionType.ATTACK)]
    obs = _obs(opts, hand=[POFFIN_ID])
    fixes = frozenset({PLAY_FIX_TEMPO})
    # model wants END with a Poffin play on the menu -> item forced first
    assert apply_play_overrides(obs, [0, 1, 2], fixes) == [1, 0, 2]
    # model wants ATTACK -> untouched (tempo is an END backstop only)
    assert apply_play_overrides(obs, [2, 0, 1], fixes) == [2, 0, 1]
    # END pick but the playable card is not a tempo item -> untouched
    obs2 = _obs(opts, hand=[NON_ENERGY])
    assert apply_play_overrides(obs2, [0, 1, 2], fixes) == [0, 1, 2]


def test_o15_ashguard_demotes_early_ash_only():
    opts = [_opt(OptionType.PLAY, 0), _opt(OptionType.ATTACK),
            _opt(OptionType.END)]
    fixes = frozenset({PLAY_FIX_ASHGUARD})
    # deck fat (>12) and the model wants the Ash play -> demoted below rest
    fat = _obs(opts, hand=[SACRED_ASH_ID], deck_count=30)
    assert apply_play_overrides(fat, [0, 1, 2], fixes) == [1, 2, 0]
    # deck at/below the guard -> untouched (O5 ash owns the low regime)
    low = _obs(opts, hand=[SACRED_ASH_ID], deck_count=12)
    assert apply_play_overrides(low, [0, 1, 2], fixes) == [0, 1, 2]
    # model didn't pick Ash -> untouched (demote only reorders the top pick)
    assert apply_play_overrides(fat, [1, 0, 2], fixes) == [1, 0, 2]


def test_o15_ashguard_composes_with_o5_into_a_window():
    """ash+ashguard = play Sacred Ash in the deck 4-11 window the top band
    uses: forced at <=10 (O5), forbidden-first at >12 (O15), free between."""
    opts = [_opt(OptionType.PLAY, 0), _opt(OptionType.ATTACK),
            _opt(OptionType.END)]
    both = frozenset({PLAY_FIX_ASH, PLAY_FIX_ASHGUARD})
    low = _obs(opts, hand=[SACRED_ASH_ID], deck_count=8)
    assert apply_play_overrides(low, [1, 0, 2], both) == [0, 1, 2]  # promoted
    fat = _obs(opts, hand=[SACRED_ASH_ID], deck_count=30)
    assert apply_play_overrides(fat, [0, 1, 2], both) == [1, 2, 0]  # demoted


def test_o16_hammer_fires_only_on_end_pick():
    opts = [_opt(OptionType.END), _opt(OptionType.PLAY, 0),
            _opt(OptionType.ATTACK)]
    obs = _obs(opts, hand=[ENHANCED_HAMMER_ID])
    fixes = frozenset({PLAY_FIX_HAMMER})
    # about to END with a hammer play on the menu -> hammer forced first
    assert apply_play_overrides(obs, [0, 1, 2], fixes) == [1, 0, 2]
    # model wants ATTACK -> untouched (END backstop only, like tempo)
    assert apply_play_overrides(obs, [2, 0, 1], fixes) == [2, 0, 1]
    # END pick but the playable card is not the hammer -> untouched
    obs2 = _obs(opts, hand=[NON_ENERGY])
    assert apply_play_overrides(obs2, [0, 1, 2], fixes) == [0, 1, 2]


def test_o4_deckguard_demotes_dudunsparce_at_low_deck():
    opts = [_opt(OptionType.ABILITY, 0, area=AreaType.BENCH),
            _opt(OptionType.PLAY, 0), _opt(OptionType.END)]
    fixes = frozenset({PLAY_FIX_DECKGUARD})
    low = _obs(opts, hand=[NON_ENERGY], bench=[_card(DUDUNSPARCE)],
               deck_count=6)
    assert apply_play_overrides(low, [0, 1, 2], fixes) == [1, 2, 0]
    # deck above the threshold -> untouched
    high = _obs(opts, hand=[NON_ENERGY], bench=[_card(DUDUNSPARCE)],
                deck_count=7)
    assert apply_play_overrides(high, [0, 1, 2], fixes) == [0, 1, 2]
    # ability belongs to a non-Dudunsparce body -> untouched
    other = _obs(opts, hand=[NON_ENERGY], bench=[_card(NON_ENERGY)],
                 deck_count=6)
    assert apply_play_overrides(other, [0, 1, 2], fixes) == [0, 1, 2]


def test_o4_deckguard_resolves_active_area():
    opts = [_opt(OptionType.ABILITY, 0, area=AreaType.ACTIVE),
            _opt(OptionType.END)]
    obs = _obs(opts, hand=[], active=[_card(DUDUNSPARCE)], deck_count=5)
    assert apply_play_overrides(
        obs, [0, 1], frozenset({PLAY_FIX_DECKGUARD})) == [1, 0]


def test_o6_conserve_demotes_fezandipiti_at_low_deck():
    opts = [_opt(OptionType.ABILITY, 0, area=AreaType.BENCH),
            _opt(OptionType.PLAY, 0), _opt(OptionType.END)]
    fixes = frozenset({PLAY_FIX_CONSERVE})
    low = _obs(opts, hand=[NON_ENERGY], bench=[_card(FEZANDIPITI_ID)],
               deck_count=6)
    assert apply_play_overrides(low, [0, 1, 2], fixes) == [1, 2, 0]
    # deck above the threshold -> untouched
    high = _obs(opts, hand=[NON_ENERGY], bench=[_card(FEZANDIPITI_ID)],
                deck_count=7)
    assert apply_play_overrides(high, [0, 1, 2], fixes) == [0, 1, 2]
    # conserve off -> deckguard alone leaves Fezandipiti untouched
    assert apply_play_overrides(
        low, [0, 1, 2], frozenset({PLAY_FIX_DECKGUARD})) == [0, 1, 2]


def test_o4_o6_compose_demoting_both_draw_abilities():
    opts = [_opt(OptionType.ABILITY, 0, area=AreaType.BENCH),
            _opt(OptionType.ABILITY, 1, area=AreaType.BENCH),
            _opt(OptionType.END)]
    obs = _obs(opts, hand=[],
               bench=[_card(DUDUNSPARCE), _card(FEZANDIPITI_ID)],
               deck_count=5)
    fixes = frozenset({PLAY_FIX_DECKGUARD, PLAY_FIX_CONSERVE})
    assert apply_play_overrides(obs, [0, 1, 2], fixes) == [2, 0, 1]


def test_o5_ash_fires_at_low_deck_regardless_of_top_pick():
    opts = [_opt(OptionType.ATTACK), _opt(OptionType.PLAY, 0),
            _opt(OptionType.END)]
    obs = _obs(opts, hand=[SACRED_ASH_ID], deck_count=10)
    fixes = frozenset({PLAY_FIX_ASH})
    assert apply_play_overrides(obs, [0, 1, 2], fixes) == [1, 0, 2]
    # deck above the threshold -> untouched
    high = _obs(opts, hand=[SACRED_ASH_ID], deck_count=11)
    assert apply_play_overrides(high, [0, 1, 2], fixes) == [0, 1, 2]


def test_o5_wins_over_o3_when_both_fire():
    opts = [_opt(OptionType.END), _opt(OptionType.PLAY, 0),
            _opt(OptionType.PLAY, 1)]
    obs = _obs(opts, hand=[POFFIN_ID, SACRED_ASH_ID], deck_count=8)
    assert apply_play_overrides(obs, [0, 1, 2], ECO) == [2, 0, 1]


def test_play_overrides_off_and_non_main_untouched():
    opts = [_opt(OptionType.END), _opt(OptionType.PLAY, 0)]
    obs = _obs(opts, hand=[POFFIN_ID], deck_count=5)
    assert apply_play_overrides(obs, [0, 1], frozenset()) == [0, 1]
    sub = _obs(opts, hand=[POFFIN_ID], deck_count=5,
               context=SelectContext.DISCARD)
    assert apply_play_overrides(sub, [0, 1], ECO) == [0, 1]


def test_o1_composes_with_play_overrides():
    # the pilots call attach overrides first, then play overrides: an
    # O1-promoted ATTACH is not turn-ending, so tempo must not re-reorder
    opts = [_opt(OptionType.END), _opt(OptionType.ATTACH, 0),
            _opt(OptionType.PLAY, 1)]
    obs = _obs(opts, hand=[TELEPATH_ID, POFFIN_ID])
    fixes = frozenset({ATTACH_FIX_TELEPATH, PLAY_FIX_TEMPO})
    order = apply_attach_overrides(obs, [0, 1, 2], fixes)
    assert order == [1, 0, 2]                    # telepath attach promoted
    assert apply_play_overrides(obs, order, fixes) == [1, 0, 2]


# --- M31 bench-economy (O7) / supporter (O8) arms + O9 precedence ----------


def test_o7_poffinfloor_digs_at_thin_bench():
    opts = [_opt(OptionType.END), _opt(OptionType.PLAY, 0),
            _opt(OptionType.ATTACK)]
    fixes = frozenset({PLAY_FIX_POFFINFLOOR})
    # bench-alive 0 (thin) and deck >= 10 -> Poffin (idx 1) promoted to front
    thin = _obs(opts, hand=[POFFIN_ID], bench=[None] * 5, deck_count=40)
    assert apply_play_overrides(thin, [0, 1, 2], fixes) == [1, 0, 2]
    # bench-alive 1 still fires (threshold is <= 1)
    one = _obs(opts, hand=[POFFIN_ID],
               bench=[_card(NON_ENERGY), None, None, None, None], deck_count=40)
    assert apply_play_overrides(one, [0, 1, 2], fixes) == [1, 0, 2]


def test_o7_poffinfloor_guards():
    opts = [_opt(OptionType.END), _opt(OptionType.PLAY, 0)]
    fixes = frozenset({PLAY_FIX_POFFINFLOOR})
    # bench-alive 2 -> board not thin -> untouched
    full = _obs(opts, hand=[POFFIN_ID],
                bench=[_card(NON_ENERGY), _card(NON_ENERGY), None, None, None],
                deck_count=40)
    assert apply_play_overrides(full, [0, 1], fixes) == [0, 1]
    # deck below the floor (9) -> untouched (low-deck economy rules own it)
    low = _obs(opts, hand=[POFFIN_ID], bench=[None] * 5, deck_count=9)
    assert apply_play_overrides(low, [0, 1], fixes) == [0, 1]
    # no Poffin on the menu -> untouched
    other = _obs(opts, hand=[NON_ENERGY], bench=[None] * 5, deck_count=40)
    assert apply_play_overrides(other, [0, 1], fixes) == [0, 1]
    # off by default
    assert apply_play_overrides(low, [0, 1], frozenset()) == [0, 1]


def test_o8_drawfloor_forces_hilda_when_hand_starved():
    opts = [_opt(OptionType.END), _opt(OptionType.PLAY, 0)]
    fixes = frozenset({PLAY_FIX_DRAWFLOOR})
    # hand size 4 (<= 4) and deck >= 10 -> Hilda (idx 1) promoted
    starved = _obs(opts, hand=[HILDA_ID, BASIC_P, BASIC_P, BASIC_P],
                   deck_count=40)
    assert apply_play_overrides(starved, [0, 1], fixes) == [1, 0]
    # hand size 5 (> 4) -> not starved -> untouched
    full = _obs(opts, hand=[HILDA_ID, BASIC_P, BASIC_P, BASIC_P, BASIC_P],
                deck_count=40)
    assert apply_play_overrides(full, [0, 1], fixes) == [0, 1]
    # deck below the floor -> untouched (keep clear of conserve's deck-out zone)
    low = _obs(opts, hand=[HILDA_ID, BASIC_P], deck_count=9)
    assert apply_play_overrides(low, [0, 1], fixes) == [0, 1]
    # off by default
    assert apply_play_overrides(starved, [0, 1], frozenset()) == [0, 1]


def test_o7_wins_over_o8_when_both_fire():
    # thin bench AND starved hand -> dig for board (Poffin) before drawing
    opts = [_opt(OptionType.END), _opt(OptionType.PLAY, 0),   # Poffin (hand 0)
            _opt(OptionType.PLAY, 1)]                          # Hilda  (hand 1)
    obs = _obs(opts, hand=[POFFIN_ID, HILDA_ID], bench=[None] * 5,
               deck_count=40)
    fixes = frozenset({PLAY_FIX_POFFINFLOOR, PLAY_FIX_DRAWFLOOR})
    assert apply_play_overrides(obs, [0, 1, 2], fixes) == [1, 0, 2]


# --- M35 bench-floor (O10): bench a basic you HOLD when thin --------------


def test_o10_benchfloor_plays_held_basic_at_thin_bench():
    opts = [_opt(OptionType.END), _opt(OptionType.PLAY, 0),   # Abra (basic)
            _opt(OptionType.ATTACK)]
    fixes = frozenset({PLAY_FIX_BENCHFLOOR})
    # bench-alive 0 -> the basic PLAY (idx 1) is promoted to front
    thin = _obs(opts, hand=[NON_ENERGY], bench=[None] * 5, deck_count=40)
    assert apply_play_overrides(thin, [0, 1, 2], fixes) == [1, 0, 2]
    # bench-alive 1 still fires (threshold <= 1)
    one = _obs(opts, hand=[NON_ENERGY],
               bench=[_card(NON_ENERGY), None, None, None, None], deck_count=40)
    assert apply_play_overrides(one, [0, 1, 2], fixes) == [1, 0, 2]
    # NO deck floor: fires even at low deck (benching costs 0 deck)
    low = _obs(opts, hand=[NON_ENERGY], bench=[None] * 5, deck_count=3)
    assert apply_play_overrides(low, [0, 1, 2], fixes) == [1, 0, 2]


def test_o10_benchfloor_guards():
    opts = [_opt(OptionType.END), _opt(OptionType.PLAY, 0)]
    fixes = frozenset({PLAY_FIX_BENCHFLOOR})
    # bench-alive 2 -> board not thin -> untouched
    full = _obs(opts, hand=[NON_ENERGY],
                bench=[_card(NON_ENERGY), _card(NON_ENERGY), None, None, None],
                deck_count=40)
    assert apply_play_overrides(full, [0, 1], fixes) == [0, 1]
    # no basic-Pokemon PLAY on the menu (POFFIN is an item) -> untouched
    other = _obs(opts, hand=[POFFIN_ID], bench=[None] * 5, deck_count=40)
    assert apply_play_overrides(other, [0, 1], fixes) == [0, 1]
    # off by default
    assert apply_play_overrides(full, [0, 1], frozenset()) == [0, 1]


def test_o10_benchfloor_wins_over_o7_poffinfloor():
    # thin bench, holding BOTH a basic and a Poffin -> bench the held basic
    # (idx 1) before digging with Poffin (idx 2).
    opts = [_opt(OptionType.END), _opt(OptionType.PLAY, 0),   # Abra   (hand 0)
            _opt(OptionType.PLAY, 1)]                          # Poffin (hand 1)
    obs = _obs(opts, hand=[NON_ENERGY, POFFIN_ID], bench=[None] * 5,
               deck_count=40)
    fixes = frozenset({PLAY_FIX_BENCHFLOOR, PLAY_FIX_POFFINFLOOR})
    assert apply_play_overrides(obs, [0, 1, 2], fixes) == [1, 0, 2]


def test_o5_ash_wins_over_o10_benchfloor():
    # ash outranks benchfloor when both fire (deck low, thin bench)
    opts = [_opt(OptionType.END), _opt(OptionType.PLAY, 0),   # Abra
            _opt(OptionType.PLAY, 1)]                          # Sacred Ash
    obs = _obs(opts, hand=[NON_ENERGY, SACRED_ASH_ID], bench=[None] * 5,
               deck_count=8)
    fixes = frozenset({PLAY_FIX_ASH, PLAY_FIX_BENCHFLOOR})
    assert apply_play_overrides(obs, [0, 1, 2], fixes) == [2, 0, 1]


def test_o5_ash_wins_over_o7_at_deck_floor():
    # deck == 10: ash (<= 10) and poffinfloor (>= 10) both eligible; ash wins
    opts = [_opt(OptionType.END), _opt(OptionType.PLAY, 0),   # Poffin
            _opt(OptionType.PLAY, 1)]                          # Sacred Ash
    obs = _obs(opts, hand=[POFFIN_ID, SACRED_ASH_ID], bench=[None] * 5,
               deck_count=10)
    fixes = frozenset({PLAY_FIX_ASH, PLAY_FIX_POFFINFLOOR})
    assert apply_play_overrides(obs, [0, 1, 2], fixes) == [2, 0, 1]


def test_o7_composes_with_gac():
    # gacb arm = gac + poffinfloor: at high deck the demote rules are inert,
    # poffinfloor fires on the thin bench.
    opts = [_opt(OptionType.END), _opt(OptionType.PLAY, 0),
            _opt(OptionType.ATTACK)]
    obs = _obs(opts, hand=[POFFIN_ID], bench=[None] * 5, deck_count=40)
    gacb = frozenset({ATTACH_FIX_TELEPATH, PLAY_FIX_DECKGUARD, PLAY_FIX_ASH,
                      PLAY_FIX_CONSERVE, PLAY_FIX_POFFINFLOOR})
    assert apply_play_overrides(obs, [0, 1, 2], gacb) == [1, 0, 2]


def test_o8_composes_with_gac():
    # gacd arm = gac + drawfloor
    opts = [_opt(OptionType.END), _opt(OptionType.PLAY, 0)]
    obs = _obs(opts, hand=[HILDA_ID, BASIC_P], deck_count=40)
    gacd = frozenset({ATTACH_FIX_TELEPATH, PLAY_FIX_DECKGUARD, PLAY_FIX_ASH,
                      PLAY_FIX_CONSERVE, PLAY_FIX_DRAWFLOOR})
    assert apply_play_overrides(obs, [0, 1], gacd) == [1, 0]


def test_ash_takes_precedence_over_o1_telepath():
    # O9 (m31): the docstring once claimed "O1 keeps precedence"; in fact a
    # PROMOTE (ash) DOES displace an O1-promoted ATTACH in the same prompt.
    # Benign (the manual attach re-offers next MAIN) but pin the REAL
    # precedence so a future edit cannot silently change it.
    opts = [_opt(OptionType.END), _opt(OptionType.ATTACH, 0),   # Telepath
            _opt(OptionType.PLAY, 1)]                            # Sacred Ash
    obs = _obs(opts, hand=[TELEPATH_ID, SACRED_ASH_ID], deck_count=10)
    fixes = frozenset({ATTACH_FIX_TELEPATH, PLAY_FIX_ASH})
    order = apply_attach_overrides(obs, [0, 1, 2], fixes)
    assert order == [1, 0, 2]                    # O1 promotes the telepath attach
    assert apply_play_overrides(obs, order, fixes) == [2, 1, 0]  # ash over it


# --- M36 gust veto (O11): no Boss's Orders while opp needs <= 1 prize ------

GUST = 1182   # Boss's Orders — rl.plan.GUST_IDS


def test_o11_gustveto_demotes_gust_at_opp_match_point():
    opts = [_opt(OptionType.PLAY, 0),      # Boss's Orders (hand 0)
            _opt(OptionType.ATTACK), _opt(OptionType.END)]
    fixes = frozenset({PLAY_FIX_GUSTVETO})
    # opponent needs 1 prize and gust is the TOP pick -> demoted below rest
    hot = _obs(opts, hand=[GUST], opp_prizes=1)
    assert apply_play_overrides(hot, [0, 1, 2], fixes) == [1, 2, 0]
    # opp_prizes 0 edge (terminal-adjacent states) still demotes
    zero = _obs(opts, hand=[GUST], opp_prizes=0)
    assert apply_play_overrides(zero, [0, 1, 2], fixes) == [1, 2, 0]


def test_o11_gustveto_guards():
    opts = [_opt(OptionType.PLAY, 0), _opt(OptionType.ATTACK)]
    fixes = frozenset({PLAY_FIX_GUSTVETO})
    hot = _obs(opts, hand=[GUST], opp_prizes=1)
    # opponent needs 2 -> not match point -> untouched
    cold = _obs(opts, hand=[GUST], opp_prizes=2)
    assert apply_play_overrides(cold, [0, 1], fixes) == [0, 1]
    # top pick is not the gust PLAY -> untouched (demotes only fire on top)
    assert apply_play_overrides(hot, [1, 0], fixes) == [1, 0]
    # a non-gust PLAY on top -> untouched
    other = _obs([_opt(OptionType.PLAY, 0), _opt(OptionType.END)],
                 hand=[POFFIN_ID], opp_prizes=1)
    assert apply_play_overrides(other, [0, 1], fixes) == [0, 1]
    # off by default
    assert apply_play_overrides(hot, [0, 1], frozenset()) == [0, 1]


def test_o11_composes_with_gacf():
    gacfv = frozenset({ATTACH_FIX_TELEPATH, PLAY_FIX_DECKGUARD, PLAY_FIX_ASH,
                       PLAY_FIX_CONSERVE, PLAY_FIX_BENCHFLOOR,
                       PLAY_FIX_GUSTVETO})
    # high deck (demotes inert), healthy bench (benchfloor inert): the gust
    # top pick at opp match point is demoted below the rest of the order.
    opts = [_opt(OptionType.PLAY, 0),      # Boss's Orders
            _opt(OptionType.ATTACK), _opt(OptionType.END)]
    bench2 = [_card(NON_ENERGY), _card(NON_ENERGY), None, None, None]
    hot = _obs(opts, hand=[GUST], bench=bench2, opp_prizes=1)
    assert apply_play_overrides(hot, [0, 1, 2], gacfv) == [1, 2, 0]
    # PROMOTE keeps precedence: thin bench + held basic -> benchfloor fires
    # first and the veto never sees the gust on top.
    opts2 = [_opt(OptionType.PLAY, 0),     # Boss's Orders (hand 0)
             _opt(OptionType.PLAY, 1),     # Abra (hand 1, basic)
             _opt(OptionType.END)]
    thin = _obs(opts2, hand=[GUST, NON_ENERGY], bench=[None] * 5,
                opp_prizes=1)
    assert apply_play_overrides(thin, [0, 1, 2], gacfv) == [1, 0, 2]


# --- M37 race mode (O12): conserve draws vs stall/grim boards --------------

TREVENANT = 879    # Hop's Trevenant — rl.plan._RACEMODE_STALL_IDS
GRIM_EX = 648      # Marnie's Grimmsnarl ex — rl.plan._RACEMODE_GRIM_IDS

# top pick = the Dudunsparce draw ABILITY on our ACTIVE; demote targets it.
_DUD_OPTS = [_opt(OptionType.ABILITY, area=AreaType.ACTIVE),
             _opt(OptionType.ATTACK), _opt(OptionType.END)]


def _race_obs(opts=None, my_deck=15, opp_deck=30, opp_active=(TREVENANT,),
              opp_bench=(), me_active_id=DUDUNSPARCE, hand=()):
    return _obs(opts or _DUD_OPTS, hand=list(hand),
                active=[_card(me_active_id)], deck_count=my_deck,
                opp_active=opp_active, opp_bench=opp_bench,
                opp_deck_count=opp_deck)


def test_o12_racemode_fires_on_stall_and_grim_boards():
    fixes = frozenset({PLAY_FIX_RACEMODE})
    # stall id on opp ACTIVE, behind by 15 on the race, deck inside (6, 25]
    stall = _race_obs(my_deck=15, opp_deck=30, opp_active=(TREVENANT,))
    assert apply_play_overrides(stall, [0, 1, 2], fixes) == [1, 2, 0]
    # grim id on opp BENCH triggers too (grim is a separate, droppable set)
    grim = _race_obs(my_deck=15, opp_deck=30, opp_active=(),
                     opp_bench=(GRIM_EX,))
    assert apply_play_overrides(grim, [0, 1, 2], fixes) == [1, 2, 0]
    # Fezandipiti body is demoted the same way (shares the O4/O6 target set)
    fez = _race_obs(me_active_id=FEZANDIPITI_ID)
    assert apply_play_overrides(fez, [0, 1, 2], fixes) == [1, 2, 0]


def test_o12_racemode_margin_guards():
    fixes = frozenset({PLAY_FIX_RACEMODE})
    # deck above the high floor (26 > 25): early setup digs stay untouched
    high = _race_obs(my_deck=26, opp_deck=40)
    assert apply_play_overrides(high, [0, 1, 2], fixes) == [0, 1, 2]
    # margin exactly 5 does NOT fire (strict <: 20 < 25 - 5 is false)
    edge = _race_obs(my_deck=20, opp_deck=25)
    assert apply_play_overrides(edge, [0, 1, 2], fixes) == [0, 1, 2]
    # deck at the low floor (6): deckguard/conserve territory, racemode out
    low = _race_obs(my_deck=6, opp_deck=30)
    assert apply_play_overrides(low, [0, 1, 2], fixes) == [0, 1, 2]
    # ahead on the race -> untouched
    ahead = _race_obs(my_deck=30, opp_deck=20)
    assert apply_play_overrides(ahead, [0, 1, 2], fixes) == [0, 1, 2]
    # no trigger id on the opponent board (mirror: Abra) -> inert
    mirror = _race_obs(opp_active=(NON_ENERGY,))
    assert apply_play_overrides(mirror, [0, 1, 2], fixes) == [0, 1, 2]
    # top pick is not a targeted ABILITY -> untouched (demote-on-top law)
    not_top = _race_obs()
    assert apply_play_overrides(not_top, [1, 0, 2], fixes) == [1, 0, 2]
    # off by default
    assert apply_play_overrides(_race_obs(), [0, 1, 2], frozenset()) \
        == [0, 1, 2]
    # non-MAIN untouched
    non_main = _obs(_DUD_OPTS, hand=[], active=[_card(DUDUNSPARCE)],
                    opp_active=(TREVENANT,), opp_deck_count=30,
                    deck_count=15, context=SelectContext.DISCARD)
    assert apply_play_overrides(non_main, [0, 1, 2], fixes) == [0, 1, 2]


def test_o12_racemoder_blanket_ignores_margin():
    fixes = frozenset({PLAY_FIX_RACEMODER})
    # even race, deck far above the margin gates -> blanket still demotes
    even = _race_obs(my_deck=40, opp_deck=40)
    assert apply_play_overrides(even, [0, 1, 2], fixes) == [1, 2, 0]
    # but never without a trigger id on the opponent board
    mirror = _race_obs(my_deck=40, opp_deck=40, opp_active=(NON_ENERGY,))
    assert apply_play_overrides(mirror, [0, 1, 2], fixes) == [0, 1, 2]


def test_o12c_racemode2_splits_families():
    from rl.plan import PLAY_FIX_RACEMODE2
    fixes = frozenset({PLAY_FIX_RACEMODE2})
    CRUSTLE = 345   # rl.plan._RACEMODE_WALL_IDS
    # wall family on opp board -> BLANKET demote (no margin needed: even race)
    wall = _race_obs(my_deck=40, opp_deck=40, opp_active=(CRUSTLE,))
    assert apply_play_overrides(wall, [0, 1, 2], fixes) == [1, 2, 0]
    # pressure family (Trevenant) -> margin-gated: even race does NOT fire...
    trev_even = _race_obs(my_deck=40, opp_deck=40, opp_active=(TREVENANT,))
    assert apply_play_overrides(trev_even, [0, 1, 2], fixes) == [0, 1, 2]
    # ...but behind-with-margin does
    trev_behind = _race_obs(my_deck=15, opp_deck=30, opp_active=(TREVENANT,))
    assert apply_play_overrides(trev_behind, [0, 1, 2], fixes) == [1, 2, 0]
    # grim is in the PRESSURE set (margin-gated), not the wall set
    grim_even = _race_obs(my_deck=40, opp_deck=40, opp_active=(GRIM_EX,))
    assert apply_play_overrides(grim_even, [0, 1, 2], fixes) == [0, 1, 2]
    grim_behind = _race_obs(my_deck=15, opp_deck=30, opp_active=(GRIM_EX,))
    assert apply_play_overrides(grim_behind, [0, 1, 2], fixes) == [1, 2, 0]
    # no trigger id (mirror) -> inert
    mirror = _race_obs(my_deck=15, opp_deck=30, opp_active=(NON_ENERGY,))
    assert apply_play_overrides(mirror, [0, 1, 2], fixes) == [0, 1, 2]


def test_o12d_racemode3_wall_only():
    from rl.plan import PLAY_FIX_RACEMODE3
    fixes = frozenset({PLAY_FIX_RACEMODE3})
    CRUSTLE = 345
    # wall id on opp board -> blanket demote, no deck gates at all
    wall = _race_obs(my_deck=40, opp_deck=40, opp_active=(CRUSTLE,))
    assert apply_play_overrides(wall, [0, 1, 2], fixes) == [1, 2, 0]
    # pressure families are NOT in the trigger: Trevenant even-race AND
    # behind-with-margin both stay untouched
    for my, opp in ((40, 40), (15, 30)):
        trev = _race_obs(my_deck=my, opp_deck=opp, opp_active=(TREVENANT,))
        assert apply_play_overrides(trev, [0, 1, 2], fixes) == [0, 1, 2]
    grim = _race_obs(my_deck=15, opp_deck=30, opp_active=(GRIM_EX,))
    assert apply_play_overrides(grim, [0, 1, 2], fixes) == [0, 1, 2]
    mirror = _race_obs(my_deck=15, opp_deck=30, opp_active=(NON_ENERGY,))
    assert apply_play_overrides(mirror, [0, 1, 2], fixes) == [0, 1, 2]


def test_o12_composes_with_gacf():
    gacfr = frozenset({ATTACH_FIX_TELEPATH, PLAY_FIX_DECKGUARD, PLAY_FIX_ASH,
                       PLAY_FIX_CONSERVE, PLAY_FIX_BENCHFLOOR,
                       PLAY_FIX_RACEMODE})
    # healthy bench (benchfloor inert), deck 15 (ash/deckguard/conserve
    # inert), trigger + margin hold -> racemode demotes the dud ability.
    bench2 = [_card(NON_ENERGY), _card(NON_ENERGY), None, None, None]
    hot = _obs(_DUD_OPTS, hand=[], bench=bench2, active=[_card(DUDUNSPARCE)],
               deck_count=15, opp_active=(TREVENANT,), opp_deck_count=30)
    assert apply_play_overrides(hot, [0, 1, 2], gacfr) == [1, 2, 0]
    # PROMOTE keeps precedence: ash at deck <= 10 fires before the demote
    ash_opts = [_opt(OptionType.ABILITY, area=AreaType.ACTIVE),
                _opt(OptionType.PLAY, 0),          # Sacred Ash (hand 0)
                _opt(OptionType.END)]
    ash = _obs(ash_opts, hand=[SACRED_ASH_ID], bench=bench2,
               active=[_card(DUDUNSPARCE)], deck_count=10,
               opp_active=(TREVENANT,), opp_deck_count=30)
    assert apply_play_overrides(ash, [0, 1, 2], gacfr) == [1, 0, 2]
    # racemode composes with deckguard at deck <= 6: both want the demote
    low = _obs(_DUD_OPTS, hand=[], bench=bench2, active=[_card(DUDUNSPARCE)],
               deck_count=6, opp_active=(TREVENANT,), opp_deck_count=30)
    assert apply_play_overrides(low, [0, 1, 2], gacfr) == [1, 2, 0]


# --- M39 P2 ----------------------------------------------------------------

MEGA_KANGA = 756   # rl.plan._RACEMODE_WALL_IDS (P2a: the kanga blindspot)
FAN_ROTOM = 174    # rl.plan._RACEMODE_STALL_IDS -> pressure half (P2a)
ALAKAZAM = 743
ENRICHING = 13
POKE_PAD = 1152
DAWN = 1231


def test_p2a_trigger_coverage_is_enumerated():
    """G-13-adjacent coverage regression: the next stall/wall variant must be
    a ONE-LINE diff caught by review, not a live 0-2. This test pins the exact
    id sets, so adding or dropping a trigger id fails here first and the diff
    states which family it lands in."""
    from rl.plan import (_RACEMODE_GRIM_IDS, _RACEMODE_PRESSURE_IDS,
                         _RACEMODE_STALL_IDS, _RACEMODE_WALL_IDS)
    assert _RACEMODE_WALL_IDS == frozenset({
        58,          # Great Tusk
        344, 532,    # Dwebble
        345, 533,    # Crustle
        607,         # Terrakion
        756,         # Mega Kangaskhan ex (M39 P2a)
    })
    assert _RACEMODE_STALL_IDS - _RACEMODE_WALL_IDS == frozenset({
        878, 879,    # Hop's Phantump / Trevenant
        304,         # Hop's Snorlax
        379, 380, 381, 341, 342, 387,   # Cynthia's line
        174,         # Fan Rotom (M39 P2a)
    })
    # the two halves partition the trigger, and grim rides with pressure
    assert _RACEMODE_PRESSURE_IDS == \
        (_RACEMODE_STALL_IDS - _RACEMODE_WALL_IDS) | _RACEMODE_GRIM_IDS
    assert not _RACEMODE_WALL_IDS & _RACEMODE_PRESSURE_IDS
    # ids deliberately EXCLUDED (documented in rl/plan.py): Team Rocket's
    # Kangaskhan ex and plain Kangaskhan are not wall bodies.
    assert not {24, 472} & (_RACEMODE_WALL_IDS | _RACEMODE_PRESSURE_IDS)


def test_p2a_new_ids_fire_on_their_own_side():
    from rl.plan import PLAY_FIX_RACEMODE2
    fixes = frozenset({PLAY_FIX_RACEMODE2})
    # Mega Kangaskhan is a WALL body -> blanket, no margin needed
    kanga = _race_obs(my_deck=40, opp_deck=40, opp_active=(MEGA_KANGA,))
    assert apply_play_overrides(kanga, [0, 1, 2], fixes) == [1, 2, 0]
    # Fan Rotom is a PRESSURE body -> margin-gated (2 of 83 cached mirror
    # lists run it, so the blanket half would misfire in the mirror)
    rotom_even = _race_obs(my_deck=40, opp_deck=40, opp_active=(FAN_ROTOM,))
    assert apply_play_overrides(rotom_even, [0, 1, 2], fixes) == [0, 1, 2]
    rotom_behind = _race_obs(my_deck=15, opp_deck=30, opp_active=(FAN_ROTOM,))
    assert apply_play_overrides(rotom_behind, [0, 1, 2], fixes) == [1, 2, 0]


def _burn_obs(hand_ids, my_deck=20, opp_deck=40, opp_active=(345,),
              own_bench=(None,) * 5, order=None):
    """MAIN prompt whose top pick is the FIRST hand card (a PLAY, or an
    ATTACH for energies) — the shape racemode4 acts on."""
    opts = [_opt(OptionType.ATTACH if cid in (ENRICHING,) else OptionType.PLAY,
                 i) for i, cid in enumerate(hand_ids)]
    opts.append(_opt(OptionType.END))
    return _obs(opts, hand=list(hand_ids), bench=list(own_bench),
                active=[_card(NON_ENERGY)], deck_count=my_deck,
                opp_active=opp_active, opp_deck_count=opp_deck)


def test_o13_racemode4_demotes_measured_burn_in_a_race():
    from rl.plan import PLAY_FIX_RACEMODE4
    fixes = frozenset({PLAY_FIX_RACEMODE4})
    # Enriching Energy ATTACH (4.0 deck cards/attach) vs a wall board
    enr = _burn_obs([ENRICHING, POFFIN_ID])
    assert apply_play_overrides(enr, [0, 1, 2], fixes) == [1, 2, 0]
    # Poke Pad PLAY, same trigger
    pad = _burn_obs([POKE_PAD, POFFIN_ID])
    assert apply_play_overrides(pad, [0, 1, 2], fixes) == [1, 2, 0]
    # both burn cards demoted together, the non-burn option survives in order
    both = _burn_obs([ENRICHING, POKE_PAD, POFFIN_ID])
    assert apply_play_overrides(both, [0, 1, 2, 3], fixes) == [2, 3, 0, 1]


def test_o13_racemode4_setup_gate_and_inertness():
    from rl.plan import PLAY_FIX_RACEMODE4
    fixes = frozenset({PLAY_FIX_RACEMODE4})
    built = [_card(ALAKAZAM), None, None, None, None]
    # Dawn/Hilda are surplus ONLY once Alakazam is on our board...
    for cid in (DAWN, HILDA_ID):
        empty_board = _burn_obs([cid, POFFIN_ID])
        assert apply_play_overrides(empty_board, [0, 1, 2], fixes) == [0, 1, 2]
        set_up = _burn_obs([cid, POFFIN_ID], own_bench=built)
        assert apply_play_overrides(set_up, [0, 1, 2], fixes) == [1, 2, 0]
    # ...while Enriching/Poke Pad do not wait for the board
    assert apply_play_overrides(_burn_obs([ENRICHING, POFFIN_ID]),
                                [0, 1, 2], fixes) == [1, 2, 0]
    # OUT of a race: no trigger id on the opponent board -> fully inert
    mirror = _burn_obs([ENRICHING, POFFIN_ID], opp_active=(NON_ENERGY,))
    assert apply_play_overrides(mirror, [0, 1, 2], fixes) == [0, 1, 2]
    # pressure family without the margin -> inert (the m37 split holds)
    trev_even = _burn_obs([ENRICHING, POFFIN_ID], my_deck=40, opp_deck=40,
                          opp_active=(TREVENANT,))
    assert apply_play_overrides(trev_even, [0, 1, 2], fixes) == [0, 1, 2]
    # demote-on-top law: burn card not the top pick -> untouched
    not_top = _burn_obs([ENRICHING, POFFIN_ID])
    assert apply_play_overrides(not_top, [1, 0, 2], fixes) == [1, 0, 2]
    # Rare Candy is NOT in the burn set (win condition, excluded by design)
    candy = _burn_obs([1079, POFFIN_ID])
    assert apply_play_overrides(candy, [0, 1, 2], fixes) == [0, 1, 2]
    # off by default
    assert apply_play_overrides(_burn_obs([ENRICHING]), [0, 1],
                                frozenset()) == [0, 1]


def test_o13b_raceash_raises_the_promote_floor_in_a_race():
    from rl.plan import PLAY_FIX_RACEASH
    ash_opts = [_opt(OptionType.ABILITY, area=AreaType.ACTIVE),
                _opt(OptionType.PLAY, 0),      # Sacred Ash (hand 0)
                _opt(OptionType.END)]

    def obs(deck, opp_active=(345,), opp_deck=40):
        return _obs(ash_opts, hand=[SACRED_ASH_ID],
                    active=[_card(DUDUNSPARCE)], deck_count=deck,
                    opp_active=opp_active, opp_deck_count=opp_deck)

    # deck 18 is above the plain `ash` floor (10) but inside the race floor
    assert apply_play_overrides(obs(18), [0, 1, 2],
                                frozenset({PLAY_FIX_ASH})) == [0, 1, 2]
    assert apply_play_overrides(obs(18), [0, 1, 2],
                                frozenset({PLAY_FIX_RACEASH})) == [1, 0, 2]
    # above the race floor -> still off
    assert apply_play_overrides(obs(21), [0, 1, 2],
                                frozenset({PLAY_FIX_RACEASH})) == [0, 1, 2]
    # out of a race, raceash alone does nothing at all (not even `ash`'s job)
    assert apply_play_overrides(obs(8, opp_active=(NON_ENERGY,)), [0, 1, 2],
                                frozenset({PLAY_FIX_RACEASH})) == [0, 1, 2]
    # composed with `ash`, the plain floor still applies outside a race
    assert apply_play_overrides(obs(8, opp_active=(NON_ENERGY,)), [0, 1, 2],
                                frozenset({PLAY_FIX_ASH,
                                           PLAY_FIX_RACEASH})) == [1, 0, 2]


def test_p2_package_leaves_the_ship_a_config_untouched():
    """Ship A is live as `conserve` alone. Every P2 rule must be provably
    inert when its own name is absent, so the P2 gate cells differ from the
    live agent by exactly one rule."""
    from rl.plan import PLAY_FIX_RACEMODE4
    conserve_only = frozenset({PLAY_FIX_CONSERVE})
    # a wall board + a burn card in hand: racemode4 would fire, conserve alone
    # touches only the Fezandipiti ability at deck <= 6
    assert apply_play_overrides(_burn_obs([ENRICHING, POFFIN_ID]),
                                [0, 1, 2], conserve_only) == [0, 1, 2]
    # and the reverse: racemode4 alone does not take over conserve's job
    fez = _race_obs(my_deck=5, opp_deck=40, opp_active=(NON_ENERGY,),
                    me_active_id=FEZANDIPITI_ID)
    assert apply_play_overrides(fez, [0, 1, 2],
                                frozenset({PLAY_FIX_RACEMODE4})) == [0, 1, 2]
    assert apply_play_overrides(fez, [0, 1, 2], conserve_only) == [1, 2, 0]


def test_bundle_twins():
    """Every bundled rl/ module is a build-time copy — no shipped predicate may
    diverge between screen and ship.

    M42: this compared ONLY plan.py, which is why the M41 divergence went
    unnoticed — HEAD carried a new submission/rl/encoders.py against an old
    rl/encoders.py for a whole milestone. It now walks the same list
    `ship_verify` does, so the two guards cannot drift apart from each other
    either."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "scripts"))
    from ship_verify import TWIN_FILES

    stale = [name for name in TWIN_FILES
             if (root / "rl" / name).read_bytes()
             != (root / "submission/rl" / name).read_bytes()]
    assert not stale, f"submission/rl is stale: {stale} — run the export"


def test_matchrunner_spec_kinds_parse():
    from rl.matchrunner import _MODEL_FIX_KINDS, parse_spec, spec_deck
    for kind in _MODEL_FIX_KINDS:
        spec = parse_spec(f"{kind}:checkpoints/x.pt:clone54618168")
        assert spec == (kind, "checkpoints/x.pt", "clone54618168")
        assert spec_deck(spec) == "clone54618168"


# --- M42 O19 `deadenergy` (apply_attach_overrides) --------------------------
# Measured before it was written: the Alakazam ship attaches onto a target
# `energy_is_dead` already covers on 29 of 297 offers (9.8%), 440 damage
# forgone, while the rule pilot on the identical deck and opponent does it
# 0.0% of the time (docs/M42.md 2026-08-04).

def _attach_opt(in_play_area=AreaType.ACTIVE, in_play_index=0, index=0):
    o = _opt(OptionType.ATTACH, index=index)
    o.inPlayArea = in_play_area
    o.inPlayIndex = in_play_index
    return o


def _dead_obs(active_energies, options=None, **kw):
    """fake_cg card 1 holds attacks {F} and {F}{C}: at two energies every
    attack is affordable and the retreat cost is 0, so it is DEAD."""
    opts = options if options is not None else [_attach_opt(),
                                                _opt(OptionType.END)]
    return _obs(opts, hand=[BASIC_P],
                active=[SimpleNamespace(id=1, energies=list(active_energies))],
                **kw)


def test_o19_demotes_an_attach_onto_a_dead_target():
    fixes = frozenset({ATTACH_FIX_DEADENERGY})
    dead = _dead_obs([FIGHTING, FIGHTING])
    assert apply_attach_overrides(dead, [0, 1], fixes) == [1, 0]


def test_o19_guards():
    fixes = frozenset({ATTACH_FIX_DEADENERGY})
    # one energy short of paying {F}{C}: NOT dead, so nothing moves
    assert apply_attach_overrides(_dead_obs([FIGHTING]), [0, 1], fixes) == [0, 1]
    # the dead attach is not the model's top pick -> demote-only, no action
    assert apply_attach_overrides(_dead_obs([FIGHTING, FIGHTING]),
                                  [1, 0], fixes) == [1, 0]
    # OFF by default: an unnamed arm must never act
    assert apply_attach_overrides(_dead_obs([FIGHTING, FIGHTING]),
                                  [0, 1], frozenset()) == [0, 1]
    # non-MAIN contexts are not ours to touch
    off_main = _dead_obs([FIGHTING, FIGHTING], context=SelectContext.TO_HAND)
    assert apply_attach_overrides(off_main, [0, 1], fixes) == [0, 1]
    # the manual attach is already spent this turn
    spent = _dead_obs([FIGHTING, FIGHTING], energy_attached=True)
    assert apply_attach_overrides(spent, [0, 1], fixes) == [0, 1]
    # a non-ATTACH top pick is never demoted
    play_first = _dead_obs([FIGHTING, FIGHTING],
                           options=[_opt(OptionType.PLAY, index=0),
                                    _attach_opt(index=0)])
    assert apply_attach_overrides(play_first, [0, 1], fixes) == [0, 1]


def test_o19_exempts_own_energy_scalers(monkeypatch):
    """The M41 exemption, inherited through energy_is_dead: an attacker whose
    damage grows with its own attached energy has no ceiling, so "charged" is
    never "saturated". Ogerpon at 7 energy is correct play, not a defect."""
    import rl.combat as rc
    monkeypatch.setattr(rc, "OWN_ENERGY_SCALERS", frozenset({101}))
    fixes = frozenset({ATTACH_FIX_DEADENERGY})
    assert apply_attach_overrides(_dead_obs([FIGHTING] * 5), [0, 1], fixes) \
        == [0, 1]


def test_o19_composes_with_backstop_without_fighting_it():
    """O2 promotes an attach when the model is about to end the turn; O19 must
    not then demote the very option O2 just promoted."""
    fixes = frozenset({ATTACH_FIX_DEADENERGY, ATTACH_FIX_BACKSTOP})
    dead = _dead_obs([FIGHTING, FIGHTING],
                     options=[_opt(OptionType.END), _attach_opt(index=0)])
    # O2 fires first and wins: the promoted order is returned unchanged
    assert apply_attach_overrides(dead, [0, 1], fixes) == [1, 0]


def test_o19_is_absent_from_every_shipped_config():
    """Inertness proof: no live fix string names it."""
    from rl.matchrunner import _MODEL_FIX_KINDS
    shipped = ("conserve,planzero,ash,ashguard",      # the M41 ogerpon ship
               "conserve,racemode2,racemode4")        # the M40 Ship B package
    for cfg in shipped:
        assert ATTACH_FIX_DEADENERGY not in cfg.split(",")
    # and it is reachable only through the two arms that name it
    naming = {k for k, v in _MODEL_FIX_KINDS.items()
              if ATTACH_FIX_DEADENERGY in v}
    assert naming == {"model-c-pkg-de", "model-de"}
