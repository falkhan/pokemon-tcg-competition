"""SimpleNamespace factories mimicking the engine's observation objects.

Shapes follow docs/pilot-cheatsheet.md. The fake ``to_observation_class`` is a
passthrough, so agents receive these objects directly.
"""
from types import SimpleNamespace

from tests.fake_cg import SelectContext


def pokemon(card_id, hp=100, max_hp=None, energies=()):
    """An in-play Pokémon."""
    return SimpleNamespace(id=card_id, hp=hp,
                           maxHp=hp if max_hp is None else max_hp,
                           energies=list(energies))


def hand_card(card_id):
    """A card in hand / discard / deck (no in-play fields like energies)."""
    return SimpleNamespace(id=card_id)


def player(active=None, bench=(), hand=(), discard=(), hand_count=None,
           deck_count=30):
    return SimpleNamespace(
        active=[active] if active is not None else [],
        bench=list(bench),
        hand=list(hand),
        discard=list(discard),
        prize=[None] * 6,
        handCount=len(hand) if hand_count is None else hand_count,
        deckCount=deck_count,
    )


def observation(me=None, opponent=None, *, your_index=0,
                context=SelectContext.MAIN, options=(), max_count=1,
                select_deck=(), stadium=(), select=True):
    """A full observation; ``select=False`` models the deck-return first call."""
    me = me if me is not None else player()
    opponent = opponent if opponent is not None else player()
    players = [me, opponent] if your_index == 0 else [opponent, me]
    return SimpleNamespace(
        current=SimpleNamespace(players=players, yourIndex=your_index,
                                stadium=list(stadium)),
        select=SimpleNamespace(context=context, option=list(options),
                               maxCount=max_count, deck=list(select_deck))
        if select else None,
    )


def option(option_type=None, *, area=None, index=None, player_index=None,
           attack_id=None, in_play_area=None, in_play_index=None):
    return SimpleNamespace(type=option_type, area=area, index=index,
                           playerIndex=player_index, attackId=attack_id,
                           inPlayArea=in_play_area, inPlayIndex=in_play_index)
