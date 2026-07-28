"""M26 attach-override + M30 deck-economy arms (rl/plan.apply_*_overrides).

Both pilots (rl/matchrunner fn3/fn4 and submission/main.py) import THESE
functions and feed them the model's full greedy order — each predicate has a
single source of truth, so parity reduces to (a) these behavioral cases and
(b) the twin files staying byte-identical (test_bundle_twins below).
"""
from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")

from cg.api import AreaType, OptionType, SelectContext
from rl.plan import (ATTACH_FIX_BACKSTOP, ATTACH_FIX_TELEPATH, FEZANDIPITI_ID,
                     HILDA_ID, POFFIN_ID, PLAY_FIX_ASH, PLAY_FIX_BENCHFLOOR,
                     PLAY_FIX_CONSERVE, PLAY_FIX_DECKGUARD, PLAY_FIX_DRAWFLOOR,
                     PLAY_FIX_GUSTVETO, PLAY_FIX_POFFINFLOOR, PLAY_FIX_RACEMODE,
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


def test_bundle_twins():
    """submission/rl/plan.py is a build-time copy of rl/plan.py — the
    override predicate must never diverge between screen and ship."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    assert (root / "rl/plan.py").read_bytes() == \
        (root / "submission/rl/plan.py").read_bytes()


def test_matchrunner_spec_kinds_parse():
    from rl.matchrunner import _MODEL_FIX_KINDS, parse_spec, spec_deck
    for kind in _MODEL_FIX_KINDS:
        spec = parse_spec(f"{kind}:checkpoints/x.pt:clone54618168")
        assert spec == (kind, "checkpoints/x.pt", "clone54618168")
        assert spec_deck(spec) == "clone54618168"
