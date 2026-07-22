"""Behavior summary for a submission's seat across cached replays.

Usage: uv run python sub_behavior.py <submission_id> [max_games]
Reports: W/L, deck-out rate, telepath attach rate, action mix, key card play rates,
Dudunsparce ability uses per game, deck cards consumed per turn.
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
TELEPATH = 19
SUB = int(sys.argv[1])
MAX_GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 60

names = {}
with open(ROOT / "data/cards_features.csv") as f:
    for row in csv.DictReader(f):
        names[int(row["card_id"])] = row["name"]

df = pl.read_parquet(ROOT / "data/kaggle/episodes.parquet").filter(
    (pl.col("submission_id_0") == SUB) | (pl.col("submission_id_1") == SUB)
).sort("episode_id", descending=True)

games = []
action_mix = Counter()
played = Counter()
hand_sightings = Counter()
ability_uses = Counter()  # card name -> count

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
    tele_opps = tele_attached = n_main = 0
    deck_traj = []
    last_state = None
    for i, obs_dict, action in iter_replay_decisions(steps, seat, drops):
        obs = to_observation_class(obs_dict)
        st = obs.current
        if st is None or obs.select is None:
            continue
        last_state = st
        me = st.players[st.yourIndex]
        opp = st.players[1 - st.yourIndex]
        deck_traj.append((st.turn, me.deckCount, opp.deckCount))
        sel = obs.select
        if sel.context != SelectContext.MAIN:
            continue
        n_main += 1
        opts = sel.option
        chosen = opts[action[0]]
        hand = me.hand or []
        for cid in set(c.id for c in hand):
            hand_sightings[cid] += 1
        tele_opt = False
        for o in opts:
            ot = OptionType(o.type)
            if ot in (OptionType.ATTACH, OptionType.PLAY):
                idx = o.index
                if (o.area in (AreaType.HAND, None) and idx is not None
                        and idx < len(hand) and hand[idx].id == TELEPATH):
                    tele_opt = True
        ct = OptionType(chosen.type)
        cid = None
        if ct in (OptionType.PLAY, OptionType.ATTACH, OptionType.EVOLVE):
            idx = chosen.index
            if (chosen.area in (AreaType.HAND, None) and idx is not None
                    and idx < len(hand)):
                cid = hand[idx].id
        if ct == OptionType.ABILITY:
            pok = None
            if chosen.area == AreaType.ACTIVE and me.active and me.active[0]:
                pok = me.active[0]
            elif chosen.area == AreaType.BENCH and chosen.index is not None \
                    and chosen.index < len(me.bench):
                pok = me.bench[chosen.index]
            if pok:
                ability_uses[names.get(pok.id, str(pok.id))] += 1
        action_mix[ct.name] += 1
        if cid is not None:
            played[cid] += 1
        if tele_opt:
            tele_opps += 1
            if cid == TELEPATH:
                tele_attached += 1
    if last_state is None:
        continue
    reward = raw["rewards"][seat] if raw.get("rewards") else None
    games.append(dict(
        ep=ep, reward=reward, turns=last_state.turn,
        deck_end=deck_traj[-1][1], opp_deck_end=deck_traj[-1][2],
        tele_opps=tele_opps, tele_attached=tele_attached, n_main=n_main,
    ))
    n_done += 1

wins = sum(1 for g in games if g["reward"] == 1)
losses = sum(1 for g in games if g["reward"] == -1)
deckout_losses = sum(1 for g in games if g["reward"] == -1 and g["deck_end"] == 0)
low_deck = sum(1 for g in games if g["deck_end"] <= 3)
opps = sum(g["tele_opps"] for g in games)
att = sum(g["tele_attached"] for g in games)
turns = sum(g["turns"] for g in games)
consumed = sum(60 - 7 - 6 - g["deck_end"] for g in games)  # rough: 60 minus opening

print(f"sub {SUB}: {len(games)} games, {wins}W-{losses}L")
print(f"  deck-out losses (deck_end==0 & loss): {deckout_losses}/{losses}")
print(f"  games ending with deck<=3: {low_deck}/{len(games)}")
print(f"  telepath: attached {att}/{opps} opportunities ({att/max(opps,1):.1%})")
print(f"  avg turns {turns/max(len(games),1):.1f}; avg deck cards consumed/our-turn "
      f"{consumed/max(turns/2,1):.2f}")
print("  action mix:", dict(action_mix.most_common()))
print("  ability uses per game:",
      {k: round(v/len(games), 2) for k, v in ability_uses.most_common(6)})
key = {19: "Telepath", 25000: None}
print("  key card play/seen:")
for cid in sorted(hand_sightings, key=lambda c: -hand_sightings[c])[:14]:
    print(f"    {names.get(cid, cid):30s} {played[cid]:4d}/{hand_sightings[cid]}")
