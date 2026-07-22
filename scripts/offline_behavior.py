"""Instrumented offline games: the behavior metrics screens can't see.

Plays side A vs side B on the direct engine loop (single process, slot-swapped)
and reports side A's energy-attach turn rate, Telepath contested-attach share,
deck-out losses, and game length — the M26 defect axes. Screen jsonls store
only result codes; this is the offline twin of scripts/attach_probe.py.

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


class BehaviorTap:
    """Wraps a pilot fn; records MAIN-decision behavior for its seat."""

    def __init__(self, fn):
        self.fn = fn
        self.turn_attached: dict[int, bool] = {}
        self.contested = Counter()      # chosen label when telepath contested
        self.tele_opps = 0
        self.tele_attached = 0
        self.turns_seen: set[int] = set()

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
    print(f"A={args.a}\nB={args.b}\n{n} games, {wins}W-{losses}L "
          f"(wr {wins / n:.3f})")
    print(f"  deck-out losses: {deckout_losses}/{losses}")
    print(f"  energy attached on {attach_turns}/{our_turns} of A's turns "
          f"({attach_turns / max(our_turns, 1):.1%})")
    print(f"  telepath attached {tap.tele_attached}/{tap.tele_opps} "
          f"opportunities ({tap.tele_attached / max(tap.tele_opps, 1):.1%})")
    print("  contested-attach choices:", dict(tap.contested.most_common(8)))
    print(f"  avg turns {sum(g['turns'] for g in games) / n:.1f}")


if __name__ == "__main__":
    main()
