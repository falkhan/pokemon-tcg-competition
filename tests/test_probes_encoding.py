"""Golden fixtures for the encoding probes (docs/M41b-plan.md § II.3a).

Each instrument gets a `fires` case and a `guards` case whose ground truth is
true BY CONSTRUCTION — synthetic menus and decisions via tests/builders, no
engine, no filesystem replays. The fake ``to_observation_class`` is a
passthrough, so the probe cores receive these objects directly, exactly the
shape their CLI paths produce after conversion.

Covered: scripts/m42_alias_probe.py (the M41b central instrument, including
the Phase-3 kill property at OPTION_M41B_DIM), scripts/m37_pm_probe.py, and
scripts/m41_scaling_probe.py."""
from collections import Counter

from rl.encoders import OPTION_M28_DIM, OPTION_M41B_DIM
from scripts.m37_pm_probe import (BOSS, DUD_IDS, WALL_IDS, adjudication_suspect,
                                  analyze_episode, detect_opp_deck)
from scripts.m41_scaling_probe import summarize_turns, tally_attack_turns
from scripts.m42_alias_probe import probe_aliasing
from tests.builders import hand_card, observation, option, player, pokemon
from tests.fake_cg import AreaType, OptionType, SelectContext

WALL_ID = min(WALL_IDS)   # 58 — any member works, minimum for determinism
DUD_ID = min(DUD_IDS)     # 66 — the clone deck's draw-ability body


# ---------------------------------------------------------------------------
# scripts/m42_alias_probe.py — real aliasing vs harmless duplicates, by width
# ---------------------------------------------------------------------------

def _ability_twin_menu(hp_a, hp_b):
    """Two ABILITY options on two SAME-ID bench Pokémon. At the live width the
    encoding carries only the card identity, so the pair aliases by
    construction; the hp arguments decide whether the alias is real."""
    me = player(active=pokemon(4),
                bench=[pokemon(1, hp=hp_a, max_hp=100),
                       pokemon(1, hp=hp_b, max_hp=100)])
    opts = [option(OptionType.ABILITY, area=AreaType.BENCH, index=0),
            option(OptionType.ABILITY, area=AreaType.BENCH, index=1)]
    return observation(me=me, options=opts)


def test_alias_probe_fires_on_real_aliasing_at_live_width():
    # Same card id, DIFFERENT live hp: identical encodings at the live-bundle
    # width, two distinct game-object fingerprints -> a REAL alias.
    c, examples = probe_aliasing([_ability_twin_menu(100, 30)], OPTION_M28_DIM)
    assert c["menus"] == 1
    assert c["real::ABILITY"] == 1
    assert c["menus_with_REAL_aliasing"] == 1
    assert c["harmless_true_duplicates"] == 0
    (fp_a, fp_b), = examples["ABILITY"]
    assert {fp_a[1][1], fp_b[1][1]} == {100, 30}    # the hp the key cannot see


def test_alias_probe_guards_identical_twins_are_harmless():
    # Same card id AND identical live state: one fingerprint -> interchangeable.
    c, examples = probe_aliasing([_ability_twin_menu(100, 100)], OPTION_M28_DIM)
    assert c["menus"] == 1
    assert c["harmless_true_duplicates"] == 1
    assert c["menus_with_REAL_aliasing"] == 0
    assert not any(k.startswith("real::") for k in c)
    assert not examples


def test_alias_probe_m41b_width_kills_the_real_alias():
    # The Phase-3 kill (plan § 3.1): the M41b board block encodes the subject's
    # live hp and bench slot, so the SAME pair no longer even shares a key —
    # no alias group forms at all, real AND harmless both read zero.
    c, examples = probe_aliasing([_ability_twin_menu(100, 30)], OPTION_M41B_DIM)
    assert c["menus"] == 1
    assert not any(k.startswith("real::") for k in c)
    assert c["harmless_true_duplicates"] == 0
    assert not examples


def test_alias_probe_true_duplicates_stay_harmless_at_m41b_width():
    # The other half of the pre-registered bar: two copies of the same trainer
    # in hand are the SAME move, and must keep aliasing (harmless) at every
    # width — the M41b block may not disturb harmless_true_duplicates.
    me = player(active=pokemon(4), hand=[hand_card(7), hand_card(7)])
    obs = observation(me=me, options=[option(OptionType.PLAY, index=0),
                                      option(OptionType.PLAY, index=1)])
    for width in (OPTION_M28_DIM, OPTION_M41B_DIM):
        c, _ = probe_aliasing([obs], width)
        assert c["harmless_true_duplicates"] == 1
        assert not any(k.startswith("real::") for k in c)


# ---------------------------------------------------------------------------
# scripts/m37_pm_probe.py — racemode3 engagement, boss plays, adjudication
# ---------------------------------------------------------------------------

def _wall_prompt(bench_id, opp_active_id, hand_id=BOSS, opp_prizes=1, turn=5,
                 context=SelectContext.MAIN):
    """One MAIN decision: an ABILITY on our bench Pokémon, a PLAY from hand,
    and END — against an opponent whose active decides the wall trigger."""
    me = player(active=pokemon(4), bench=[pokemon(bench_id)],
                hand=[hand_card(hand_id)])
    opp = player(active=pokemon(opp_active_id), prizes_remaining=opp_prizes)
    opts = [option(OptionType.ABILITY, area=AreaType.BENCH, index=0),
            option(OptionType.PLAY, index=0),
            option(OptionType.END)]
    return observation(me=me, opponent=opp, options=opts, turn=turn,
                       context=context)


def test_m37_wall_engagement_and_boss_play_fire():
    obs = _wall_prompt(DUD_ID, WALL_ID)
    # Two prompts in the SAME turn: the dud ability used, then Boss played.
    res = analyze_episode([(obs, [0]), (obs, [1])], is_wall_opp=True)
    assert res["trig_prompts"] == 2 and res["trig_true"] == 2
    assert res["offer_trig"] == 2 and res["use_trig"] == 1
    assert res["offer_off"] == 0 and res["use_off"] == 0
    assert res["boss_plays"] == [(5, 1)]        # turn 5, opp at 1 prize left
    assert res["traj"] == [(5, 30, 30, 6, 1)]   # the TURN is the unit, once


def test_m37_wall_engagement_guards():
    # No wall on the opponent's board: the offer lands in the OFF-trigger bin,
    # and a played non-Boss trainer records no boss play.
    off = _wall_prompt(DUD_ID, opp_active_id=2, hand_id=7)
    res = analyze_episode([(off, [0])], is_wall_opp=True)
    assert res["trig_true"] == 0
    assert res["offer_trig"] == 0 and res["use_trig"] == 0
    assert res["offer_off"] == 1 and res["use_off"] == 1
    assert res["boss_plays"] == []
    # A non-draw ability body is not an offer in EITHER bin.
    other_body = _wall_prompt(bench_id=1, opp_active_id=WALL_ID)
    res = analyze_episode([(other_body, [0])], is_wall_opp=True)
    assert res["trig_true"] == 1
    assert res["offer_trig"] == 0 and res["offer_off"] == 0
    # A non-MAIN prompt is not counted at all — but last_state still tracks it
    # (adjudication anatomy reads the FINAL decision state).
    sub_prompt = _wall_prompt(DUD_ID, WALL_ID, context=SelectContext.DISCARD)
    res = analyze_episode([(sub_prompt, [0])], is_wall_opp=True)
    assert res["trig_prompts"] == 0 and res["traj"] == []
    assert res["last_state"] is sub_prompt.current


def _end_state(my_prizes, opp_prizes, my_deck):
    return observation(me=player(prizes_remaining=my_prizes, deck_count=my_deck),
                       opponent=player(prizes_remaining=opp_prizes)).current


def test_m37_adjudication_suspect_fires():
    assert adjudication_suspect(_end_state(3, 3, 5))    # tied race, deck alive
    assert adjudication_suspect(_end_state(2, 6, 10))   # ahead outright


def test_m37_adjudication_suspect_guards():
    assert not adjudication_suspect(_end_state(5, 2, 10))  # behind on the race
    assert not adjudication_suspect(_end_state(3, 3, 0))   # decked out: real loss
    assert not adjudication_suspect(_end_state(3, 0, 10))  # opp took all prizes


def test_m37_detect_opp_deck_fires():
    deck = [WALL_ID] * 4 + [7] * 56
    steps = [[{"action": deck}, {}], [{}, {"action": [0]}]]
    assert detect_opp_deck(steps, seat=1) == deck          # opponent is seat 0
    assert bool(WALL_IDS & set(detect_opp_deck(steps, 1)))


def test_m37_detect_opp_deck_guards():
    # A menu answer is not a deck list, and the scan stops after 4 steps.
    assert detect_opp_deck([[{"action": [3]}, {}]] * 4, seat=1) is None
    assert detect_opp_deck([], seat=0) is None
    late = [[{}, {}]] * 4 + [[{"action": [7] * 60}, {}]]
    assert detect_opp_deck(late, seat=1) is None


# ---------------------------------------------------------------------------
# scripts/m41_scaling_probe.py — the TURN is the unit, not the prompt
# ---------------------------------------------------------------------------

ATTACKER = 5   # fake pool: its only attack (103) PRINTS 0 — the scaling shape
OTHER = 1      # fake pool: a plain attacker for the control book


def _attack_menu():
    return [option(OptionType.PLAY, index=0),
            option(OptionType.ATTACK, attack_id=103),
            option(OptionType.END)]


def _attack_prompt(attacker_id, hand_n, opp_hp, turn,
                   context=SelectContext.MAIN, options=None):
    me = player(active=pokemon(attacker_id), hand=[hand_card(7)] * hand_n)
    opp = player(active=pokemon(2, hp=opp_hp))
    return observation(me=me, opponent=opp, turn=turn, context=context,
                       options=_attack_menu() if options is None else options)


def test_scaling_probe_fires_one_turn_one_record():
    turns, other_turns, says = {}, {}, Counter()
    # Two prompts in ONE turn: declined first, attacked at the last chance —
    # the exact shape the probe's first cut miscounted as a missed lethal.
    decisions = [(_attack_prompt(ATTACKER, hand_n=5, opp_hp=100, turn=4), [0]),
                 (_attack_prompt(ATTACKER, hand_n=2, opp_hp=100, turn=4), [1])]
    tally_attack_turns(iter(decisions), 1, ATTACKER, 20,
                       turns, other_turns, says)
    assert list(turns) == [(1, 4)]      # ONE record: the turn, not the prompts
    rec = turns[(1, 4)]
    assert rec["attacked"] is True
    assert rec["hand"] == 2             # the hand at the LAST chance to attack
    assert rec["lethal"] is True        # 20 x 5 >= 100, seen earlier that turn
    assert says[True] == 0 and says[False] == 2   # printed 0: feature-blind
    by_hand, total, lethal = summarize_turns(turns)
    assert total == [1, 1] and lethal == [1, 1]   # attacked, 0 missed lethals
    assert by_hand[2] == [1, 1]


def test_scaling_probe_guards_out_of_scope_prompts():
    turns, other_turns, says = {}, {}, Counter()
    decisions = [
        # a different attacker active -> the control book, never ours
        (_attack_prompt(OTHER, hand_n=4, opp_hp=100, turn=2), [1]),
        # not a MAIN prompt -> not an attack chance
        (_attack_prompt(ATTACKER, hand_n=4, opp_hp=100, turn=3,
                        context=SelectContext.DISCARD), [0]),
        # ATTACK absent from the menu -> attacking was not legal, no record
        (_attack_prompt(ATTACKER, hand_n=8, opp_hp=100, turn=3,
                        options=[option(OptionType.PLAY, index=0),
                                 option(OptionType.END)]), [1]),
        # no active on either side -> skipped
        (observation(options=_attack_menu(), turn=4), [2]),
    ]
    tally_attack_turns(iter(decisions), 9, ATTACKER, 20,
                       turns, other_turns, says)
    assert turns == {}
    assert list(other_turns) == [(9, 2)]
    assert other_turns[(9, 2)]["attacked"] is True
    assert says == Counter()            # counted only on the attacker's turns


def test_scaling_probe_guards_below_lethal_threshold():
    turns, other_turns, says = {}, {}, Counter()
    # 20 x 3 = 60 < 100: attacking, but no lethal was ever on the table.
    decisions = [(_attack_prompt(ATTACKER, hand_n=3, opp_hp=100, turn=5), [1])]
    tally_attack_turns(iter(decisions), 9, ATTACKER, 20,
                       turns, other_turns, says)
    assert turns[(9, 5)]["attacked"] is True
    assert turns[(9, 5)]["lethal"] is False
    _by_hand, total, lethal = summarize_turns(turns)
    assert total == [1, 1] and lethal == [0, 0]
