"""M37 post-mortem: per-source deck-burn audit in live wall-family games.

For each wall game: our deckCount delta per own-turn, and every action we took
that plausibly consumes deck cards (trainer plays by card, telepath attaches,
ability uses), plus Sacred Ash timing (deck + discard-pokemon count at play).
Burn attribution: measure deckCount drop between consecutive MAIN prompts and
attribute to the action(s) chosen in between.
"""
import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl
from cg.api import AreaType, OptionType, SelectContext, to_observation_class
from rl.replay_bc import iter_replay_decisions
import rl.plan as rlplan

ROOT = Path(__file__).resolve().parent.parent
SUB = int(sys.argv[1]) if len(sys.argv) > 1 else 55065484
WALL_IDS = set(rlplan._RACEMODE_WALL_IDS)
ASH_ID = rlplan.SACRED_ASH_ID
TELEPATH = 19

names = {}
with open(ROOT / "data/cards_features.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        names[int(row["card_id"])] = row["name"]

df = pl.read_parquet(ROOT / "data/kaggle/episodes.parquet").filter(
    (pl.col("submission_id_0") == SUB) | (pl.col("submission_id_1") == SUB)
).sort("episode_id")

# deck-burn attributed to the previously chosen action, pooled over wall games
burn_by_action = defaultdict(lambda: [0, 0])   # label -> [times, total deck drop]
ash_events = []

for r in df.iter_rows(named=True):
    ep = int(r["episode_id"])
    seat = 0 if r["submission_id_0"] == SUB else 1
    path = ROOT / f"data/kaggle/raw/episode_{ep}.json.gz"
    if not path.exists():
        continue
    raw = json.load(gzip.open(path))
    steps = raw["steps"]

    opp_deck = None
    for step in steps[:4]:
        st = step[1 - seat] if 1 - seat < len(step) else None
        if isinstance(st, dict):
            act = st.get("action")
            if isinstance(act, list) and len(act) == 60:
                opp_deck = [int(a) for a in act]
                break
    if not (WALL_IDS & set(opp_deck or [])):
        continue

    drops = Counter()
    prev_deck = None
    prev_label = None
    prev_turn = None
    for i, obs_dict, action in iter_replay_decisions(steps, seat, drops):
        obs = to_observation_class(obs_dict)
        st = obs.current
        if st is None or obs.select is None:
            continue
        if obs.select.context != SelectContext.MAIN:
            continue
        me = st.players[st.yourIndex]
        if prev_deck is not None and prev_label is not None:
            drop = prev_deck - me.deckCount
            # same-turn attribution is clean; across turns the mandatory draw
            # (and opp turn) intervenes -> subtract 1 for the turn draw
            if st.turn != prev_turn:
                drop -= 1
                prev_label = prev_label + "+turndraw" if drop > 0 else prev_label
            if drop > 0 or st.turn == prev_turn:
                burn_by_action[prev_label][0] += 1
                burn_by_action[prev_label][1] += max(drop, 0)
        opts = obs.select.option
        chosen = opts[action[0]]
        ct = OptionType(chosen.type)
        hand = me.hand or []
        label = ct.name
        cid = None
        if ct in (OptionType.PLAY, OptionType.ATTACH, OptionType.EVOLVE):
            idx = chosen.index
            if chosen.area in (AreaType.HAND, None) and idx is not None \
                    and idx < len(hand):
                cid = hand[idx].id
                label = f"{ct.name}:{names.get(cid, cid)}"
        elif ct == OptionType.ABILITY:
            pid = rlplan._board_pokemon_id(chosen, me)
            label = f"ABILITY:{names.get(pid, pid)}"
        if cid == ASH_ID and ct == OptionType.PLAY:
            ash_events.append((ep, st.turn, me.deckCount))
        prev_deck = me.deckCount
        prev_label = label
        prev_turn = st.turn

print("=== deck-burn attribution in wall games (label -> occurrences, avg deck drop) ===")
rows = sorted(burn_by_action.items(), key=lambda kv: -kv[1][1])
for label, (n, tot) in rows:
    if tot == 0 and n < 20:
        continue
    print(f"  {label:40s} n={n:4d} total_drop={tot:4d} avg={tot/max(n,1):.2f}")

print("\n=== Sacred Ash plays in wall games (ep, turn, deck at play) ===")
for e in ash_events:
    print(f"  ep{e[0]} t={e[1]} deck={e[2]}")
