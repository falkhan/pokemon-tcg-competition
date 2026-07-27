"""Forced-use probe: pin the board-state effect of a card's ABILITY or PLAY.

Always takes the target option when it is offered at a MAIN prompt, then diffs
(deckCount, hand, bench, active, discard) at the NEXT MAIN prompt — the ability's
selection sub-prompts resolve first, so the naive next-decision read shows delta
0 (M30 P0 lesson, docs/M30.md:17). Reuses the DrainTap/BehaviorTap wrapper
pattern (scripts/deck_drain.py:55, scripts/offline_behavior.py:39).

M31 P0.2 / P0.3:
  - Dunsparce (305) ACTIVE-area ABILITY — semantics unpinned; never used by
    anyone across 120 live episodes, declined at bench 0 in bench-out loss
    87676383. If it benches basics it is the in-kit answer to bench economy.
    (Distinct from id 66 Dudunsparce, the deck-draw evolution — DUDUNSPARCE_IDS.)
  - Buddy-Buddy Poffin (1086) PLAY and Hilda (1225) PLAY — deck cost per use and
    what each puts where (P0.3 gates whether O8 drawfloor is carried).

Usage:
    uv run python scripts/ability_probe.py \
        --a model:checkpoints/m28_winners.pt:clone54618168 \
        --b modelt-gac:checkpoints/m28_winners.pt:clone54618168 \
        --kind ability --id 305 -n 80 --seed 1
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cg.api import OptionType, SelectContext, to_observation_class
from rl.matchrunner import make_pilot, parse_spec
from rl.plan import _board_pokemon_id, _hand_card_id


def _card_names() -> dict[int, str]:
    with open(ROOT / "data/cards_features.csv") as f:
        return {int(r["card_id"]): r["name"] for r in csv.DictReader(f)}


NAMES = _card_names()


def _name(cid: int) -> str:
    return NAMES.get(cid, str(cid))


def _basic_ids() -> frozenset[int]:
    with open(ROOT / "data/cards_features.csv") as f:
        return frozenset(int(r["card_id"]) for r in csv.DictReader(f)
                         if r.get("is_basic") == "true"
                         or r.get("stage") == "0")


_BASIC_IDS = _basic_ids()


def _snapshot(obs) -> dict:
    st = obs.current
    me = st.players[st.yourIndex]
    active = me.active[0].id if me.active and me.active[0] is not None else None
    return dict(
        turn=st.turn,
        deck=me.deckCount,
        hand=[c.id for c in (me.hand or [])],
        bench=[p.id for p in (me.bench or []) if p is not None],
        active=active,
        discard=[c.id for c in (me.discard or [])],
    )


def _delta(before: list[int], after: list[int]) -> tuple[Counter, Counter]:
    """(added, removed) as multiset differences after - before / before - after."""
    cb, ca = Counter(before), Counter(after)
    return ca - cb, cb - ca


class ForceTap:
    """Wraps A's pilot; forces the target option whenever offered at a MAIN
    prompt and records the board diff at the next MAIN prompt."""

    def __init__(self, fn, kind: str, target_id: int):
        self.fn = fn
        self.kind = kind                 # "ability" | "play"
        self.target_id = target_id
        self.offered = 0                 # target offered at a MAIN prompt
        self.forced = 0
        self.samples: list[dict] = []
        self.pending = None              # before-snapshot awaiting next MAIN
        # diagnostics for the 0-offer case
        self.main_prompts = 0
        self.target_active = 0           # prompts where target id is active
        self.target_inplay = 0           # prompts where target is active or bench
        self.ability_pokemon = Counter()  # board pokemon offering ANY ability

    def reset_game(self):
        self.pending = None

    def _target_index(self, obs, me):
        for j, o in enumerate(obs.select.option):
            if self.kind == "ability":
                if o.type == OptionType.ABILITY \
                        and _board_pokemon_id(o, me) == self.target_id:
                    return j
            elif o.type == OptionType.PLAY \
                    and _hand_card_id(o, me.hand or []) == self.target_id:
                return j
        return None

    def __call__(self, od):
        obs = to_observation_class(od)
        st, sel = obs.current, obs.select
        is_main = (st is not None and sel is not None
                   and sel.context == SelectContext.MAIN)
        if self.pending is not None and is_main:
            self._record(self.pending, _snapshot(obs))
            self.pending = None
        if not is_main:
            return self.fn(od)
        me = st.players[st.yourIndex]
        self.main_prompts += 1
        active_id = me.active[0].id if me.active and me.active[0] else None
        bench_ids = {p.id for p in (me.bench or []) if p is not None}
        if active_id == self.target_id:
            self.target_active += 1
        if active_id == self.target_id or self.target_id in bench_ids:
            self.target_inplay += 1
        for o in sel.option:
            if o.type == OptionType.ABILITY:
                self.ability_pokemon[_board_pokemon_id(o, me)] += 1
        tgt = self._target_index(obs, me)
        if tgt is None:
            return self.fn(od)
        self.offered += 1
        self.pending = _snapshot(obs)
        self.forced += 1
        picks = [tgt] + [i for i in self.fn(od) if i != tgt]
        return picks[:sel.maxCount]

    def _record(self, before: dict, after: dict):
        b_add, b_rem = _delta(before["bench"], after["bench"])
        h_add, h_rem = _delta(before["hand"], after["hand"])
        d_add, _ = _delta(before["discard"], after["discard"])
        self.samples.append(dict(
            same_turn=after["turn"] == before["turn"],
            ddeck=after["deck"] - before["deck"],
            dhand=len(after["hand"]) - len(before["hand"]),
            dbench=len(after["bench"]) - len(before["bench"]),
            ddiscard=len(after["discard"]) - len(before["discard"]),
            active_change=before["active"] != after["active"],
            bench_added=b_add, bench_removed=b_rem,
            hand_added=h_add, hand_removed=h_rem, disc_added=d_add,
        ))


def _agg_counter(samples, key) -> Counter:
    c = Counter()
    for s in samples:
        c.update(s[key])
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", required=True, help="instrumented side (spec)")
    ap.add_argument("--b", required=True)
    ap.add_argument("--kind", required=True, choices=("ability", "play"))
    ap.add_argument("--id", type=int, required=True, help="target card id")
    ap.add_argument("-n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    import random

    from cg.game import battle_finish, battle_select, battle_start
    random.seed(args.seed)

    fn_a, deck_a = make_pilot(parse_spec(args.a), "probe_a")
    fn_b, deck_b = make_pilot(parse_spec(args.b), "probe_b")
    tap = ForceTap(fn_a, args.kind, args.id)

    for g in range(args.n):
        a_seat = g % 2
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
        finally:
            battle_finish()

    s = tap.samples
    clean = [x for x in s if x["same_turn"]]
    print(f"A={args.a}\nB={args.b}")
    print(f"target {args.kind} id={args.id} ({_name(args.id)}); "
          f"{args.n} games seed {args.seed}")
    print(f"offered/forced at MAIN: {tap.offered}; diffs recorded: {len(s)} "
          f"(same-turn clean: {len(clean)}, cross-turn: {len(s) - len(clean)})")
    print(f"  diag: MAIN prompts {tap.main_prompts}; target active "
          f"{tap.target_active}, target in play {tap.target_inplay}")
    print("  diag: ABILITY options by board pokemon:",
          {_name(k): v for k, v in tap.ability_pokemon.most_common(10)})
    if not clean:
        print("  NO clean same-turn diffs — target never resolved same turn "
              "(check offers; may end the turn).")
        if not s:
            return
    use = clean or s
    tag = "same-turn" if clean else "ALL (cross-turn — confounded)"
    print(f"\n  effect per use [{tag}, n={len(use)}]:")
    print(f"    deckCount delta : {dict(Counter(x['ddeck'] for x in use))} "
          f"(mean {sum(x['ddeck'] for x in use) / len(use):+.2f})")
    print(f"    hand   count Δ  : {dict(Counter(x['dhand'] for x in use))} "
          f"(mean {sum(x['dhand'] for x in use) / len(use):+.2f})")
    print(f"    bench  count Δ  : {dict(Counter(x['dbench'] for x in use))} "
          f"(mean {sum(x['dbench'] for x in use) / len(use):+.2f})")
    print(f"    discard count Δ : {dict(Counter(x['ddiscard'] for x in use))}")
    print(f"    active changed  : {sum(x['active_change'] for x in use)}"
          f"/{len(use)}")
    bench_added = _agg_counter(use, "bench_added")
    if bench_added:
        basics = sum(v for k, v in bench_added.items() if k in _BASIC_IDS)
        print(f"\n  cards ADDED to bench (total {sum(bench_added.values())}, "
              f"basics {basics}):")
        for cid, c in bench_added.most_common(10):
            b = " [basic]" if cid in _BASIC_IDS else ""
            print(f"    {c:3d}  {_name(cid)}{b}")
    hand_added = _agg_counter(use, "hand_added")
    if hand_added:
        print(f"\n  cards ADDED to hand (total {sum(hand_added.values())}):")
        for cid, c in hand_added.most_common(10):
            print(f"    {c:3d}  {_name(cid)}")
    disc_added = _agg_counter(use, "disc_added")
    if disc_added:
        print(f"\n  cards ADDED to discard (total {sum(disc_added.values())}):")
        for cid, c in disc_added.most_common(10):
            print(f"    {c:3d}  {_name(cid)}")


if __name__ == "__main__":
    main()
