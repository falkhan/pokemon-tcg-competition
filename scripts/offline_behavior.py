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


def _feature_ids():
    """(item ids, basic-Pokemon ids, supporter ids) from the card table."""
    import csv
    items, basics, supporters = set(), set(), set()
    with open(ROOT / "data/cards_features.csv") as f:
        for r in csv.DictReader(f):
            cid = int(r["card_id"])
            if r["is_item"] == "true":
                items.add(cid)
            if r["is_basic"] == "true":
                basics.add(cid)
            if r["is_supporter"] == "true":
                supporters.add(cid)
    return frozenset(items), frozenset(basics), frozenset(supporters)


_ITEM_IDS, _BASIC_IDS, _SUPPORTER_IDS = _feature_ids()
POFFIN_ID = 1086
HILDA_ID = 1225


def _bench_alive(me) -> int:
    return sum(1 for p in (me.bench or []) if p is not None)


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
        # M31 bench-economy + supporter counters
        self.main_prompts = 0
        self.bench_hist = Counter()      # bench-alive size at MAIN prompts
        self.bench_le1_prompts = 0       # MAIN prompts with bench<=1
        self.poffin_le1_offered = 0      # ...Poffin playable
        self.poffin_le1_played = 0       # ...and chosen
        self.basic_le1_offered = 0       # ...a basic PLAY available
        self.basic_le1_played = 0        # ...and chosen
        self.hilda_offered = 0           # MAIN prompts Hilda playable
        self.hilda_played = 0
        self.turn_supp_offered: dict[int, bool] = {}
        self.turn_supp_played: dict[int, bool] = {}
        self.min_bench_after_t3 = 99     # per-game, paired with outcome in main()

    def reset_game(self):
        self.turn_attached = {}
        self.turn_supp_offered = {}
        self.turn_supp_played = {}
        self.min_bench_after_t3 = 99

    def __call__(self, od):
        picks = self.fn(od)
        obs = to_observation_class(od)
        st, sel = obs.current, obs.select
        if st is None or sel is None or sel.context != SelectContext.MAIN:
            return picks
        me = st.players[st.yourIndex]
        hand = me.hand or []
        # --- M31 bench-economy + supporter counters ---
        bench = _bench_alive(me)
        self.main_prompts += 1
        self.bench_hist[bench] += 1
        if st.turn > 3:
            self.min_bench_after_t3 = min(self.min_bench_after_t3, bench)
        chosen0 = sel.option[picks[0]]
        play_ids = {j: _hand_card_id(o, hand) for j, o in enumerate(sel.option)
                    if o.type == OptionType.PLAY}
        chosen_play = play_ids.get(picks[0])
        self.turn_supp_offered.setdefault(st.turn, False)
        self.turn_supp_played.setdefault(st.turn, False)
        if any(c in _SUPPORTER_IDS for c in play_ids.values()):
            self.turn_supp_offered[st.turn] = True
        if chosen0.type == OptionType.PLAY and chosen_play in _SUPPORTER_IDS:
            self.turn_supp_played[st.turn] = True
        if HILDA_ID in play_ids.values():
            self.hilda_offered += 1
            if chosen_play == HILDA_ID:
                self.hilda_played += 1
        if bench <= 1:
            self.bench_le1_prompts += 1
            if POFFIN_ID in play_ids.values():
                self.poffin_le1_offered += 1
                if chosen_play == POFFIN_ID:
                    self.poffin_le1_played += 1
            if any(c in _BASIC_IDS for c in play_ids.values()):
                self.basic_le1_offered += 1
                if chosen_play in _BASIC_IDS:
                    self.basic_le1_played += 1
        # --- end M31 counters ---
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
    supp_turns = supp_played = 0
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
            # RESULT log (type 23): reason 1=prizes, 2=deck-out, 3=benched,
            # 4=card effect (scripts/deck_drain.py:149).
            reason = next((lg.get("reason") for lg in obs_dict.get("logs") or []
                           if lg.get("type") == 23), None)
            games.append(dict(
                won=res == a_seat, draw=res == 2, reason=reason,
                min_bench_t3=tap.min_bench_after_t3,
                deck_end=players[a_seat].get("deckCount"),
                opp_deck_end=players[1 - a_seat].get("deckCount"),
                turns=obs_dict["current"].get("turn")))
        finally:
            battle_finish()
        our_turns += len(tap.turn_attached)
        attach_turns += sum(tap.turn_attached.values())
        supp_turns += sum(tap.turn_supp_offered.values())
        supp_played += sum(tap.turn_supp_played.values())

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

    # --- M31 bench-economy + supporter axes ---
    reason_losses = Counter(g["reason"] for g in games
                            if not g["won"] and not g["draw"])
    deckout = reason_losses.get(2, 0)
    benchout = reason_losses.get(3, 0)
    b0_loss = sum(1 for g in games if not g["won"] and not g["draw"]
                  and g["min_bench_t3"] == 0)
    b0_win = sum(1 for g in games if g["won"] and g["min_bench_t3"] == 0)
    mp = max(tap.main_prompts, 1)
    print("\n  --- M31 bench economy ---")
    print(f"  bench-alive at MAIN: "
          + " ".join(f"{k}:{tap.bench_hist.get(k, 0)}" for k in range(6)))
    print(f"  bench<=1 prompt share: {tap.bench_le1_prompts}/{tap.main_prompts} "
          f"({tap.bench_le1_prompts / mp:.1%})")
    print(f"  Poffin at bench<=1: played {tap.poffin_le1_played}"
          f"/{tap.poffin_le1_offered} offered "
          f"({tap.poffin_le1_played / max(tap.poffin_le1_offered, 1):.1%})")
    print(f"  basic PLAY at bench<=1: played {tap.basic_le1_played}"
          f"/{tap.basic_le1_offered} offered "
          f"({tap.basic_le1_played / max(tap.basic_le1_offered, 1):.1%})")
    print(f"  bench-0-after-t3 games: losses {b0_loss}/{losses}, "
          f"wins {b0_win}/{wins}")
    print(f"  loss reasons (RESULT code): {dict(reason_losses)}; "
          f"deck-out(2) {deckout}/{losses}, bench-out(3) {benchout}/{losses}")
    print("\n  --- M31 supporter axis ---")
    print(f"  supporter-turn utilisation: {supp_played}/{supp_turns} "
          f"({supp_played / max(supp_turns, 1):.1%})")
    print(f"  Hilda per-offer: {tap.hilda_played}/{tap.hilda_offered} "
          f"({tap.hilda_played / max(tap.hilda_offered, 1):.1%})")


if __name__ == "__main__":
    main()
