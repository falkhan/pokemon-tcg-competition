"""SimpleNamespace factories mimicking the engine's observation objects.

Shapes follow docs/pilot-cheatsheet.md. The fake ``to_observation_class`` is a
passthrough, so agents receive these objects directly.
"""
from types import SimpleNamespace

from tests.fake_cg import SelectContext


def pokemon(card_id, hp=100, max_hp=None, energies=(), tools=()):
    """An in-play Pokémon."""
    return SimpleNamespace(id=card_id, hp=hp,
                           maxHp=hp if max_hp is None else max_hp,
                           energies=list(energies),
                           tools=list(tools))


def hand_card(card_id):
    """A card in hand / discard / deck (no in-play fields like energies)."""
    return SimpleNamespace(id=card_id)


def player(active=None, bench=(), hand=(), discard=(), hand_count=None,
           deck_count=30, prizes_remaining=6, poisoned=0, burned=0,
           asleep=0, paralyzed=0, confused=0):
    return SimpleNamespace(
        active=[active] if active is not None else [],
        bench=list(bench),
        hand=list(hand),
        discard=list(discard),
        prize=[None] * prizes_remaining,
        handCount=len(hand) if hand_count is None else hand_count,
        deckCount=deck_count,
        poisoned=poisoned, burned=burned, asleep=asleep,
        paralyzed=paralyzed, confused=confused,
    )


def observation(me=None, opponent=None, *, your_index=0,
                context=SelectContext.MAIN, options=(), max_count=1,
                select_deck=(), stadium=(), select=True, turn=1,
                energy_attached=False, supporter_played=False, looking=(),
                result=-1):
    """A full observation; ``select=False`` models the deck-return first call."""
    me = me if me is not None else player()
    opponent = opponent if opponent is not None else player()
    players = [me, opponent] if your_index == 0 else [opponent, me]
    return SimpleNamespace(
        current=SimpleNamespace(players=players, yourIndex=your_index,
                                stadium=list(stadium), turn=turn,
                                energyAttached=energy_attached,
                                supporterPlayed=supporter_played,
                                looking=list(looking), result=result),
        select=SimpleNamespace(context=context, option=list(options),
                               maxCount=max_count, deck=list(select_deck))
        if select else None,
    )


def search_state(obs, search_id=0):
    """A determinized-search state as returned by cg.api.search_begin/step."""
    return SimpleNamespace(observation=obs, searchId=search_id)


def option(option_type=None, *, area=None, index=None, player_index=None,
           attack_id=None, in_play_area=None, in_play_index=None, card_id=None):
    return SimpleNamespace(type=option_type, area=area, index=index,
                           playerIndex=player_index, attackId=attack_id,
                           inPlayArea=in_play_area, inPlayIndex=in_play_index,
                           cardId=card_id)
