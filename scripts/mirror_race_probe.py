"""Mirror deck-race probe (M36 P0.4 — W4 trigger design input).

For each MIRROR game of a sub: per-turn (my_deck, opp_deck) race trajectory,
and every optional draw-ability use (Dudunsparce / Fezandipiti) taken while
BEHIND on the deck race but ABOVE the current demote floors (deck > 6) — i.e.
the exact decisions a race-aware conserve rule would demote. Output sizes the
trigger before any implementation (composition law: gacbf died; mirror bed
resolves only ~7pp, so engagement counters gate this, not strength).

Usage: uv run python scripts/mirror_race_probe.py [<sub_id>]
"""
import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import polars as pl
from cg.api import AreaType, OptionType, SelectContext, to_observation_class
from rl.plan import DUDUNSPARCE_IDS, FEZANDIPITI_ID, _CONSERVE_AT
from rl.replay_bc import iter_replay_decisions

DRAW_IDS = set(DUDUNSPARCE_IDS) | {FEZANDIPITI_ID}

# mirror = the opponent plays the Alakazam archetype (live copies vary a few
# slots, so byte-identity undercounts — live_deck_race.py archetype approach)
_names = {}   # card_id -> name, filled by main(); tests patch module-bound


def is_mirror(opp_deck):
    return any(_names.get(c, "").startswith("Alakazam") for c in opp_deck or [])


def board_id(opt, me):
    if opt.area == AreaType.ACTIVE and me.active and me.active[0]:
        return me.active[0].id
    if opt.area == AreaType.BENCH and opt.index is not None \
            and opt.index < len(me.bench or []) and (me.bench or [])[opt.index]:
        return (me.bench or [])[opt.index].id
    return None


def scan_game(decisions):
    """One game's MAIN decisions -> (traj, uses): the per-prompt
    (turn, my_deck, opp_deck) race trajectory and every draw-ability use.
    decisions: iterable of (converted observation, action) pairs.
    """
    traj = []            # (turn, my_deck, opp_deck)
    uses = []            # (turn, my_deck, opp_deck, card)
    for obs, action in decisions:
        st = obs.current
        if st is None or obs.select is None \
                or obs.select.context != SelectContext.MAIN:
            continue
        me = st.players[st.yourIndex]
        opp = st.players[1 - st.yourIndex]
        traj.append((st.turn, me.deckCount, opp.deckCount))
        chosen = obs.select.option[action[0]]
        if OptionType(chosen.type) == OptionType.ABILITY:
            bid = board_id(chosen, me)
            if bid in DRAW_IDS:
                uses.append((st.turn, me.deckCount, opp.deckCount, bid))
    return traj, uses


def behind_uses(uses):
    """The demote candidates: uses while BEHIND on the race, ABOVE the floor."""
    return [(t, mine, theirs, bid) for t, mine, theirs, bid in uses
            if mine < theirs and mine > _CONSERVE_AT]


def margin_bucket(margin):
    """Race deficit -> the trigger-sizing bucket."""
    return ("margin1-2" if margin <= 2 else
            "margin3-5" if margin <= 5 else "margin6+")


def main(argv):
    sub = int(argv[0]) if argv else 55011605

    with open(ROOT / "data/cards_features.csv") as f:
        for row in csv.DictReader(f):
            _names[int(row["card_id"])] = row["name"]

    df = pl.read_parquet(ROOT / "data/kaggle/episodes.parquet").filter(
        (pl.col("submission_id_0") == sub) | (pl.col("submission_id_1") == sub)
    ).sort("episode_id")

    mirror_games = 0
    uses_behind = Counter()   # margin bucket -> count of draw-ability uses
    for r in df.iter_rows(named=True):
        ep = int(r["episode_id"])
        seat = 0 if r["submission_id_0"] == sub else 1
        path = ROOT / f"data/kaggle/raw/episode_{ep}.json.gz"
        if not path.exists():
            continue
        raw = json.load(gzip.open(path))
        steps = raw["steps"]
        rew = raw.get("rewards") or [0, 0]
        won = rew[seat] > rew[1 - seat]

        opp_deck = None
        for step in steps[:4]:
            if isinstance(step[1 - seat], dict):
                act = step[1 - seat].get("action")
                if isinstance(act, list) and len(act) == 60:
                    opp_deck = [int(a) for a in act]
                    break
        if not is_mirror(opp_deck):
            continue
        mirror_games += 1

        decisions = ((to_observation_class(o), a) for _, o, a
                     in iter_replay_decisions(steps, seat, Counter()))
        traj, uses = scan_game(decisions)

        if not traj:
            continue
        end = traj[-1]
        behind = behind_uses(uses)
        for _t, mine, theirs, _b in behind:
            uses_behind[margin_bucket(theirs - mine)] += 1
        me_end, opp_end = end[1], end[2]
        print(f"ep{ep} {'W' if won else 'L'} t={end[0]:2d} "
              f"deck end {me_end:2d}/{opp_end:2d} "
              f"draw-uses {len(uses):2d} (behind+abovefloor {len(behind):2d}) "
              f"{'DECKOUT' if not won and me_end == 0 else ''}")
        for t, mine, theirs, bid in behind:
            print(f"    t{t:2d} deck {mine:2d} vs {theirs:2d} -> ability {bid} "
                  f"(race-aware conserve would demote)")

    print(f"\nmirror games: {mirror_games}")
    print(f"draw-ability uses while behind-on-race & above deck>{_CONSERVE_AT} "
          f"floor, by margin: {dict(uses_behind)}")


if __name__ == "__main__":
    main(sys.argv[1:])
