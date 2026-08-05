"""Cross-instrument agreement: two independent instruments, one situation.

docs/M41b-plan.md § II.3b. The M42 stadium bug surfaced ONLY because
`scripts/m42_perception_probe.py` and `rl/postmortem.py` disagreed on the same
games — one counted per prompt, the other per turn, and the contradiction is
what exposed the wrong one. That was luck. These tests make it deliberate.

THE POINT IS THE INDEPENDENCE. `rl/postmortem.py` reads replay-JSON **dicts**;
the perception probe reads engine **observation objects** through an
instrumented pilot. Neither shares a line of measurement code with the other,
and this file must never introduce a shared one: a common helper would make
the two agree while both were wrong, which is the exact opposite of the
property being bought here. `_scenario` below builds test DATA in both shapes
— it decides nothing, classifies nothing, and counts nothing.

So each test states one situation, hands it to both instruments in their own
native shape, and asserts they reach the same verdict.
"""
from types import SimpleNamespace

import pytest

import rl.postmortem as pm
import scripts.m42_perception_probe as m42_probe
from tests import builders
from tests.fake_cg import AreaType, CardType, OptionType, SelectContext

# Synthetic pool. Card 5's only attack costs nothing and its retreat is 0, so
# any energy on it is dead by construction (`rl.combat.energy_is_dead`) — the
# over-attach ground truth. Cards 20/21 are two DIFFERENT stadiums: a same-name
# stadium cannot legally be played over itself, so only a distinct pair can
# ever be the miss (M42: 0 same-id offers in 20 chances).
DEAD_TARGET = 5
LIVE_TARGET = 1          # attacks 101 {F} and 102 {F}{C} — energy is not dead
OUR_STADIUM, THEIR_STADIUM = 20, 21


@pytest.fixture(autouse=True)
def _stadium_pool(monkeypatch):
    """Register the two stadiums with BOTH instruments' own card tables.

    Deliberately patched twice, once per instrument, rather than through a
    shared fixture object: the two tables are separate module state and the
    test must not quietly unify them.
    """
    ours = SimpleNamespace(cardId=OUR_STADIUM, name="Our Stadium", basic=False,
                           evolvesFrom=None, cardType=CardType.STADIUM)
    theirs = SimpleNamespace(cardId=THEIR_STADIUM, name="Their Stadium",
                             basic=False, evolvesFrom=None,
                             cardType=CardType.STADIUM)
    pool = {c.cardId: c for c in (ours, theirs)}
    # instrument A: rl/postmortem.py
    monkeypatch.setattr(pm, "_CARDS", {**pm._CARDS, **pool})
    monkeypatch.setattr(pm, "_CARDS_BY_NAME",
                        {**pm._CARDS_BY_NAME, **{c.name: c for c in pool.values()}})
    monkeypatch.setattr(pm, "_STADIUM_TYPE", CardType.STADIUM)
    # instrument B: scripts/m42_perception_probe.py
    monkeypatch.setattr(m42_probe, "_NAME",
                        {**m42_probe._NAME, OUR_STADIUM: "Our Stadium",
                         THEIR_STADIUM: "Their Stadium"})
    monkeypatch.setattr(m42_probe, "_IS_STADIUM", {OUR_STADIUM, THEIR_STADIUM})


# --- data rendering, NOT measurement ----------------------------------------

def _poke_dict(card_id, hp=100, energies=()):
    return {"id": card_id, "hp": hp, "maxHp": hp, "energies": list(energies),
            "tools": []}


def _player_dict(active=None, bench=(), hand=(), prizes=6, deck=30):
    return {"active": [active] if active else [], "bench": list(bench),
            "hand": [{"id": c} for c in hand], "handCount": len(hand),
            "discard": [], "prize": [{}] * prizes, "deckCount": deck}


def _steps(prompts, us=0, turn=1):
    """[(options, chosen)] -> replay-JSON steps for rl/postmortem.py.

    Mirrors the shape `_iter_selects` walks: step[seat]['observation']['select']
    plus step[seat]['action'].
    """
    out = []
    for options, chosen, cur in prompts:
        mine = {"observation": {"select": {"context": int(SelectContext.MAIN),
                                           "option": options, "deck": []},
                                "current": {**cur, "turn": turn}},
                "action": list(chosen), "status": "ACTIVE"}
        out.append([mine, {"observation": {}, "action": None}]
                   if us == 0 else
                   [{"observation": {}, "action": None}, mine])
    return out


def _ladder_run(lad, obs_picks):
    """Feed (obs, picks) through one instrumented scripted pilot, then close
    the game so the per-TURN ladders resolve (the semantics at issue)."""
    script = [list(p) for _, p in obs_picks]
    wrapped = m42_probe.instrument(lambda od: script.pop(0), lad)
    for obs, _ in obs_picks:
        wrapped(obs)
    lad.end_game()
    assert lad.c["probe_errors"] == 0        # nothing swallowed into silence


# --- 1. over-attach ---------------------------------------------------------

def test_over_attach_both_instruments_fire_on_a_dead_target():
    """One chosen ATTACH onto a target that can already pay everything it
    has. postmortem emits an [over-attach] line; the probe's ladder counts a
    chose_bad. Ground truth is the pool: card 5 has a free attack and no
    retreat cost, so any energy on it buys nothing that exists in the rules."""
    opt = {"type": int(OptionType.ATTACH), "inPlayArea": int(AreaType.BENCH),
           "inPlayIndex": 0}
    cur = {"players": [_player_dict(active=_poke_dict(LIVE_TARGET),
                                    bench=[_poke_dict(DEAD_TARGET,
                                                      energies=[6])]),
                       _player_dict(active=_poke_dict(LIVE_TARGET))]}
    flag = pm._attach_saturated(opt, cur, 0, 0, 1)

    lad = m42_probe.Ladders()
    me = builders.player(active=builders.pokemon(LIVE_TARGET),
                         bench=[builders.pokemon(DEAD_TARGET, energies=[6])])
    obs = builders.observation(
        me=me, opponent=builders.player(active=builders.pokemon(LIVE_TARGET)),
        options=[builders.option(OptionType.ATTACH,
                                 in_play_area=AreaType.BENCH, in_play_index=0),
                 builders.option(OptionType.END)])
    _ladder_run(lad, [(obs, [0])])

    assert flag is not None, "postmortem missed an over-attach the probe saw"
    assert lad.c["over_attach.chose_bad"] == 1
    assert bool(flag) == bool(lad.c["over_attach.chose_bad"])


def test_over_attach_both_instruments_stay_silent_on_a_live_target():
    """The near-miss both must decline: card 1 still has an unpaid {F}{C}, so
    the energy is live and neither instrument may count it."""
    opt = {"type": int(OptionType.ATTACH), "inPlayArea": int(AreaType.BENCH),
           "inPlayIndex": 0}
    cur = {"players": [_player_dict(active=_poke_dict(LIVE_TARGET),
                                    bench=[_poke_dict(LIVE_TARGET,
                                                      energies=[6])]),
                       _player_dict(active=_poke_dict(LIVE_TARGET))]}
    flag = pm._attach_saturated(opt, cur, 0, 0, 1)

    lad = m42_probe.Ladders()
    me = builders.player(active=builders.pokemon(LIVE_TARGET),
                         bench=[builders.pokemon(LIVE_TARGET, energies=[6])])
    obs = builders.observation(
        me=me, opponent=builders.player(active=builders.pokemon(LIVE_TARGET)),
        options=[builders.option(OptionType.ATTACH,
                                 in_play_area=AreaType.BENCH, in_play_index=0),
                 builders.option(OptionType.END)])
    _ladder_run(lad, [(obs, [0])])

    assert flag is None
    assert lad.c["over_attach.chose_bad"] == 0


# --- 2. the stadium, per TURN — the disagreement that caught the M42 bug -----

def _stadium_prompts_dicts(play_at):
    """Two MAIN prompts in ONE turn, their stadium out, ours in hand.
    `play_at` is the prompt index we play it at, or None to never play it."""
    prompts = []
    for i in range(2):
        cur = {"players": [_player_dict(active=_poke_dict(LIVE_TARGET),
                                        hand=[OUR_STADIUM]),
                           _player_dict(active=_poke_dict(LIVE_TARGET))],
               "stadium": [{"id": THEIR_STADIUM, "playerIndex": 1}]}
        options = [{"type": int(OptionType.PLAY), "index": 0},
                   {"type": int(OptionType.END)}]
        prompts.append((options, [0] if i == play_at else [1], cur))
    return prompts


def _stadium_obs_picks(play_at):
    """The same two prompts as engine observations."""
    out = []
    for i in range(2):
        me = builders.player(active=builders.pokemon(LIVE_TARGET),
                             hand=[builders.hand_card(OUR_STADIUM)])
        obs = builders.observation(
            me=me,
            opponent=builders.player(active=builders.pokemon(LIVE_TARGET)),
            options=[builders.option(OptionType.PLAY, index=0),
                     builders.option(OptionType.END)],
            stadium=[SimpleNamespace(id=THEIR_STADIUM, playerIndex=1)])
        out.append((obs, [0] if i == play_at else [1]))
    return out


def test_stadium_played_late_in_the_turn_is_a_miss_for_NEITHER_instrument():
    """THE regression this file exists for. Playing the stadium at prompt 2 of
    a turn leaves prompt 1 looking like a decline; the probe's first version
    counted per PROMPT and read 58.5%, while postmortem — already per TURN —
    saw 4 for 4. Both are per-turn now, and disagreement here means one of
    them silently reverted."""
    flags = pm._setup_taxonomy_flags(_steps(_stadium_prompts_dicts(play_at=1)), 0)
    lad = m42_probe.Ladders()
    _ladder_run(lad, _stadium_obs_picks(play_at=1))

    pm_missed = any("stadium" in f.lower() for f in flags)
    assert pm_missed is False
    assert lad.c["stadium_ignored.chose_bad"] == 0
    assert pm_missed == bool(lad.c["stadium_ignored.chose_bad"])


def test_stadium_never_played_is_a_miss_for_BOTH_instruments():
    """The other half: the turn ends with their stadium still out and ours
    still in hand. Both instruments must count exactly one miss."""
    flags = pm._setup_taxonomy_flags(
        _steps(_stadium_prompts_dicts(play_at=None)), 0)
    lad = m42_probe.Ladders()
    _ladder_run(lad, _stadium_obs_picks(play_at=None))

    pm_missed = any("stadium" in f.lower() for f in flags)
    assert pm_missed is True
    assert lad.c["stadium_ignored.chose_bad"] == 1
    assert pm_missed == bool(lad.c["stadium_ignored.chose_bad"])


def test_our_own_stadium_in_play_is_a_miss_for_neither():
    """Ownership, the other half of the M42 stadium correction: there is
    nothing to displace when the stadium out there is already ours."""
    prompts = []
    for _ in range(2):
        cur = {"players": [_player_dict(active=_poke_dict(LIVE_TARGET),
                                        hand=[OUR_STADIUM]),
                           _player_dict(active=_poke_dict(LIVE_TARGET))],
               "stadium": [{"id": THEIR_STADIUM, "playerIndex": 0}]}
        prompts.append(([{"type": int(OptionType.PLAY), "index": 0},
                         {"type": int(OptionType.END)}], [1], cur))
    flags = pm._setup_taxonomy_flags(_steps(prompts), 0)

    lad = m42_probe.Ladders()
    obs_picks = []
    for _ in range(2):
        me = builders.player(active=builders.pokemon(LIVE_TARGET),
                             hand=[builders.hand_card(OUR_STADIUM)])
        obs_picks.append((builders.observation(
            me=me,
            opponent=builders.player(active=builders.pokemon(LIVE_TARGET)),
            options=[builders.option(OptionType.PLAY, index=0),
                     builders.option(OptionType.END)],
            stadium=[SimpleNamespace(id=THEIR_STADIUM, playerIndex=0)]), [1]))
    _ladder_run(lad, obs_picks)

    pm_missed = any("stadium" in f.lower() for f in flags)
    assert pm_missed is False
    assert lad.c["stadium_ignored.chose_bad"] == 0
