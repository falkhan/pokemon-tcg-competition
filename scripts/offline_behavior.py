"""Instrumented offline games: the behavior metrics screens can't see.

Plays side A vs side B on the direct engine loop (single process, slot-swapped)
and reports side A's energy-attach turn rate, Telepath contested-attach share,
deck-out losses, and game length — the M26 defect axes — plus the M30
deck-economy axes: END chosen with a playable ITEM in hand (per END turn, with
the declined items named) and the per-side deck-out split (A decked out vs B
decked out). Screen jsonls store only result codes; this is the offline twin
of scripts/attach_probe.py.

Usage:
    uv run python scripts/offline_behavior.py \
        --a model:checkpoints/m25_bc_alakazam_v3h.pt:clone54618168 \
        --b rule:lucario -n 60 --seed 1
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cg.api import AreaType, OptionType, SelectContext, to_observation_class
from rl.matchrunner import make_pilot, parse_spec
from rl.plan import TELEPATH_ID, _IS_ENERGY, _hand_card_id


def _item_ids() -> frozenset[int]:
    import csv
    with open(ROOT / "data/cards_features.csv") as f:
        return frozenset(int(r["card_id"]) for r in csv.DictReader(f)
                         if r["is_item"] == "true")


_ITEM_IDS = _item_ids()


class BehaviorTap:
    """Wraps a pilot fn; records MAIN-decision behavior for its seat."""

    def __init__(self, fn):
        self.fn = fn
        self.turn_attached: dict[int, bool] = {}
        self.contested = Counter()      # chosen label when telepath contested
        self.tele_opps = 0
        self.tele_attached = 0
        self.turns_seen: set[int] = set()
        self.end_turns = 0
        self.end_with_item = 0
        self.declined_items = Counter()  # item id declined on an END pick
        self.ash_plays = []              # deckCount at each Sacred Ash play
        self.dud_low_opps = 0            # Dudunsparce ability offered, deck<=6
        self.dud_low_used = 0            # ...and chosen

    def reset_game(self):
        self.turn_attached = {}

    def __call__(self, od):
        picks = self.fn(od)
        obs = to_observation_class(od)
        st, sel = obs.current, obs.select
        if st is None or sel is None or sel.context != SelectContext.MAIN:
            return picks
        me = st.players[st.yourIndex]
        hand = me.hand or []
        self.turn_attached.setdefault(st.turn, False)
        attachable = {}
        for j, o in enumerate(sel.option):
            if o.type == OptionType.ATTACH:
                cid = _hand_card_id(o, hand)
                if cid in _IS_ENERGY:
                    attachable[j] = cid
        chosen = sel.option[picks[0]]
        chosen_cid = (_hand_card_id(chosen, hand)
                      if chosen.type in (OptionType.ATTACH, OptionType.PLAY)
                      else None)
        if picks[0] in attachable:
            self.turn_attached[st.turn] = True
        from rl.plan import (DUDUNSPARCE_IDS, SACRED_ASH_ID, _DECKGUARD_AT,
                             _board_pokemon_id)
        if chosen.type == OptionType.PLAY and chosen_cid == SACRED_ASH_ID:
            self.ash_plays.append(me.deckCount)
        if me.deckCount <= _DECKGUARD_AT:
            dud_opts = [j for j, o in enumerate(sel.option)
                        if o.type == OptionType.ABILITY
                        and _board_pokemon_id(o, me) in DUDUNSPARCE_IDS]
            if dud_opts:
                self.dud_low_opps += 1
                if picks[0] in dud_opts:
                    self.dud_low_used += 1
        if chosen.type == OptionType.END:
            self.end_turns += 1
            playable = {_hand_card_id(o, hand) for o in sel.option
                        if o.type == OptionType.PLAY}
            declined = playable & _ITEM_IDS
            if declined:
                self.end_with_item += 1
                for cid in declined:
                    self.declined_items[cid] += 1
        if TELEPATH_ID in attachable.values():
            self.tele_opps += 1
            if chosen_cid == TELEPATH_ID and chosen.type == OptionType.ATTACH:
                self.tele_attached += 1
            if any(c != TELEPATH_ID for c in attachable.values()):
                if chosen.type == OptionType.ATTACH and chosen_cid is not None:
                    self.contested[f"ATTACH {chosen_cid}"] += 1
                else:
                    self.contested[OptionType(chosen.type).name] += 1
        return picks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", required=True, help="instrumented side (spec)")
    ap.add_argument("--b", required=True)
    ap.add_argument("-n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    import random

    from cg.game import battle_finish, battle_select, battle_start
    random.seed(args.seed)

    fn_a, deck_a = make_pilot(parse_spec(args.a), "behav_a")
    fn_b, deck_b = make_pilot(parse_spec(args.b), "behav_b")
    tap = BehaviorTap(fn_a)

    games = []
    our_turns = attach_turns = 0
    for g in range(args.n):
        a_seat = g % 2                        # slot-swap (player-0 advantage)
        fns = (tap, fn_b) if a_seat == 0 else (fn_b, tap)
        decks = (deck_a, deck_b) if a_seat == 0 else (deck_b, deck_a)
        tap.reset_game()
        obs_dict, start = battle_start(*decks)
        if start.errorPlayer >= 0:
            battle_finish()
            raise ValueError(f"battle_start rejected deck ({start.errorType})")
        try:
            while obs_dict["current"]["result"] < 0:
                seat = obs_dict["current"]["yourIndex"]
                picks = fns[seat](obs_dict)
                obs_dict = battle_select([int(i) for i in picks])
            players = obs_dict["current"]["players"]
            res = obs_dict["current"]["result"]        # winner seat, 2 = draw
            games.append(dict(
                won=res == a_seat, draw=res == 2,
                deck_end=players[a_seat].get("deckCount"),
                opp_deck_end=players[1 - a_seat].get("deckCount"),
                turns=obs_dict["current"].get("turn")))
        finally:
            battle_finish()
        our_turns += len(tap.turn_attached)
        attach_turns += sum(tap.turn_attached.values())

    n = len(games)
    wins = sum(g["won"] for g in games)
    losses = sum(1 for g in games if not g["won"] and not g["draw"])
    deckout_losses = sum(1 for g in games
                         if not g["won"] and not g["draw"]
                         and g["deck_end"] == 0)
    deckout_wins = sum(1 for g in games if g["won"] and g["opp_deck_end"] == 0)
    low_deck = sum(1 for g in games if g["deck_end"] <= 3)
    avg_deck_end = sum(g["deck_end"] for g in games) / max(n, 1)
    print(f"A={args.a}\nB={args.b}\n{n} games, {wins}W-{losses}L "
          f"(wr {wins / n:.3f})")
    print(f"  deck-out losses (A decked): {deckout_losses}/{losses}; "
          f"deck-out wins (B decked): {deckout_wins}/{wins}")
    print(f"  A ends with deck<=3: {low_deck}/{n}; "
          f"avg A deck at end {avg_deck_end:.1f}")
    print(f"  END with playable ITEM: {tap.end_with_item}/{tap.end_turns} "
          f"END turns ({tap.end_with_item / max(tap.end_turns, 1):.1%})")
    if tap.declined_items:
        print("  items declined at END:",
              dict(tap.declined_items.most_common(6)))
    print(f"  Sacred Ash plays: {len(tap.ash_plays)} "
          f"(deck at play: {sorted(tap.ash_plays)[:12]})")
    print(f"  Dudunsparce ability at deck<=6: used {tap.dud_low_used}"
          f"/{tap.dud_low_opps} prompts")
    print(f"  energy attached on {attach_turns}/{our_turns} of A's turns "
          f"({attach_turns / max(our_turns, 1):.1%})")
    print(f"  telepath attached {tap.tele_attached}/{tap.tele_opps} "
          f"opportunities ({tap.tele_attached / max(tap.tele_opps, 1):.1%})")
    print("  contested-attach choices:", dict(tap.contested.most_common(8)))
    print(f"  avg turns {sum(g['turns'] for g in games) / n:.1f}")


if __name__ == "__main__":
    main()
