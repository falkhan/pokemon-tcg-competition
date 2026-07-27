"""M29 post-mortem loss forensics: stranded items, declined-item turns, loss modes.

Usage: uv run python m29_loss_forensics.py <submission_id> [max_games]
"""
import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/falkhan/Documents/python_projects/pokemon-tcg-competition")

import polars as pl
from cg.api import AreaType, OptionType, SelectContext, to_observation_class
from rl.replay_bc import iter_replay_decisions

ROOT = Path("/home/falkhan/Documents/python_projects/pokemon-tcg-competition")
SUB = int(sys.argv[1])
MAX_GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 60

names, kinds = {}, {}
with open(ROOT / "data/cards_features.csv") as f:
    for row in csv.DictReader(f):
        names[int(row["card_id"])] = row["name"]
        for k in ("supporter", "item", "stadium", "tool"):
            if row.get(f"is_{k}") == "true":
                kinds[int(row["card_id"])] = k

df = pl.read_parquet(ROOT / "data/kaggle/episodes.parquet").filter(
    (pl.col("submission_id_0") == SUB) | (pl.col("submission_id_1") == SUB)
).sort("episode_id", descending=True)

games = []
decline_examples = Counter()   # item card declined while END chosen
stranded = Counter()           # trainer cards in hand at end of LOST games
n_done = 0
for r in df.iter_rows(named=True):
    if n_done >= MAX_GAMES:
        break
    ep = int(r["episode_id"])
    seat = 0 if r["submission_id_0"] == SUB else 1
    path = ROOT / f"data/kaggle/raw/episode_{ep}.json.gz"
    if not path.exists():
        continue
    raw = json.load(gzip.open(path))
    steps = raw["steps"]
    drops = Counter()
    last_state = None
    end_with_item = 0
    n_end = 0
    my_pts_last = opp_pts_last = None
    for i, obs_dict, action in iter_replay_decisions(steps, seat, drops):
        obs = to_observation_class(obs_dict)
        st = obs.current
        if st is None or obs.select is None:
            continue
        last_state = st
        me = st.players[st.yourIndex]
        opp = st.players[1 - st.yourIndex]
        my_pts_last = getattr(me, "points", None)
        opp_pts_last = getattr(opp, "points", None)
        sel = obs.select
        if sel.context != SelectContext.MAIN:
            continue
        opts = sel.option
        chosen = opts[action[0]]
        hand = me.hand or []
        ct = OptionType(chosen.type)
        if ct == OptionType.END:
            n_end += 1
            avail_items = []
            for o in opts:
                if OptionType(o.type) == OptionType.PLAY and o.index is not None \
                        and o.index < len(hand):
                    cid = hand[o.index].id
                    if kinds.get(cid) == "item":
                        avail_items.append(cid)
            if avail_items:
                end_with_item += 1
                for cid in set(avail_items):
                    decline_examples[names.get(cid, cid)] += 1
    if last_state is None:
        continue
    me = last_state.players[last_state.yourIndex]
    opp = last_state.players[1 - last_state.yourIndex]
    reward = raw["rewards"][seat] if raw.get("rewards") else None
    hand_trainers = Counter()
    if reward == -1:
        for c in (me.hand or []):
            if kinds.get(c.id):
                hand_trainers[names.get(c.id, c.id)] += 1
                stranded[names.get(c.id, c.id)] += 1
    games.append(dict(
        ep=ep, reward=reward, turns=last_state.turn,
        deck_end=me.deckCount, opp_deck_end=opp.deckCount,
        my_pts=my_pts_last, opp_pts=opp_pts_last,
        hand_end=len(me.hand or []),
        end_with_item=end_with_item, n_end=n_end,
        hand_trainers=dict(hand_trainers),
    ))
    n_done += 1

wins = [g for g in games if g["reward"] == 1]
losses = [g for g in games if g["reward"] == -1]
print(f"sub {SUB}: {len(games)} games, {len(wins)}W-{len(losses)}L")
tot_end = sum(g["n_end"] for g in games)
tot_ewi = sum(g["end_with_item"] for g in games)
print(f"  END chosen with a playable ITEM in hand: {tot_ewi}/{tot_end} END turns "
      f"({tot_ewi/max(tot_end,1):.1%})")
print("  items declined at END turns (distinct per turn):")
for k, v in decline_examples.most_common(12):
    print(f"    {k:35s} {v}")
print(f"  avg hand size at end of LOSSES: "
      f"{sum(g['hand_end'] for g in losses)/max(len(losses),1):.1f} "
      f"vs WINS: {sum(g['hand_end'] for g in wins)/max(len(wins),1):.1f}")
print("  trainers stranded in hand at end of lost games:")
for k, v in stranded.most_common(15):
    print(f"    {k:35s} {v}")
print("  loss modes:")
for g in losses:
    mode = "DECK-OUT" if g["deck_end"] == 0 else "prizes/other"
    print(f"    ep {g['ep']} turns={g['turns']:3d} deck={g['deck_end']:2d} "
          f"oppdeck={g['opp_deck_end']:2d} pts={g['my_pts']}-{g['opp_pts']} "
          f"hand={g['hand_end']:2d} {mode}")
