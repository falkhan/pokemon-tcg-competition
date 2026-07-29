"""M37 post-mortem probes for sub 55065484 (m28_winners + gacfr3 + alakazam_v2_h4).

1. Wall-family games: racemode3 live engagement (dud/fez ability offers vs uses
   while the wall trigger is TRUE), deck-race trajectory, loss mode.
2. Adjudication anatomy: losses where we led/tied on prizes at a high turn
   count (suspected turn-cap scoring) — final-step dump.
3. Boss's Orders plays at opp-prizes-left <= 1 (gustveto stayed unshipped).
4. WR by opponent-score band (band-escape shape).
"""
import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl
from cg.api import AreaType, OptionType, SelectContext, to_observation_class
from rl.replay_bc import iter_replay_decisions
import rl.plan as rlplan

ROOT = Path(__file__).resolve().parent.parent
SUB = int(sys.argv[1]) if len(sys.argv) > 1 else 55065484
SELF_TEAM = "Team Pierogachu"
BOSS = 1182
WALL_IDS = set(rlplan._RACEMODE_WALL_IDS)
DUD_IDS = set(rlplan.DUDUNSPARCE_IDS)
FEZ_ID = rlplan.FEZANDIPITI_ID

names, is_pokemon = {}, set()
with open(ROOT / "data/cards_features.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        cid = int(row["card_id"])
        names[cid] = row["name"]
        if row.get("is_pokemon") == "true":
            is_pokemon.add(cid)

df = pl.read_parquet(ROOT / "data/kaggle/episodes.parquet").filter(
    (pl.col("submission_id_0") == SUB) | (pl.col("submission_id_1") == SUB)
).sort("episode_id")


def board_ids(p):
    ids = set()
    for pok in (p.active or []) + (p.bench or []):
        if pok:
            ids.add(pok.id)
    return ids


boss_plays = []          # (ep, turn, opp_prize_left, reward)
band_wl = Counter()      # (band, W/L)
wall_rows = []
adjudicated = []

for r in df.iter_rows(named=True):
    ep = int(r["episode_id"])
    seat = 0 if r["submission_id_0"] == SUB else 1
    opp_team = r["team_1"] if seat == 0 else r["team_0"]
    my_team = r["team_0"] if seat == 0 else r["team_1"]
    if str(opp_team) == SELF_TEAM and str(my_team) == SELF_TEAM:
        continue
    path = ROOT / f"data/kaggle/raw/episode_{ep}.json.gz"
    if not path.exists():
        continue
    raw = json.load(gzip.open(path))
    steps = raw["steps"]
    reward = raw["rewards"][seat] if raw.get("rewards") else None
    opp_score = r["updated_score_1"] if seat == 0 else r["updated_score_0"]

    band = int(opp_score // 100) * 100
    band_wl[(band, "W" if reward == 1 else "L")] += 1

    # opponent deck for family tagging
    opp_deck = None
    for step in steps[:4]:
        st = step[1 - seat] if 1 - seat < len(step) else None
        if isinstance(st, dict):
            act = st.get("action")
            if isinstance(act, list) and len(act) == 60:
                opp_deck = [int(a) for a in act]
                break
    is_wall_opp = bool(WALL_IDS & set(opp_deck or []))

    drops = Counter()
    trig_true = trig_prompts = 0
    draw_offer_trig = draw_use_trig = 0
    draw_offer_off = draw_use_off = 0
    traj = []            # (turn, my_deck, opp_deck, my_pz_left, opp_pz_left)
    last_state = None
    for i, obs_dict, action in iter_replay_decisions(steps, seat, drops):
        obs = to_observation_class(obs_dict)
        st = obs.current
        if st is None or obs.select is None:
            continue
        last_state = st
        if obs.select.context != SelectContext.MAIN:
            continue
        me = st.players[st.yourIndex]
        op = st.players[1 - st.yourIndex]
        trig = bool(board_ids(op) & WALL_IDS)
        trig_prompts += 1
        if trig:
            trig_true += 1
        opts = obs.select.option
        chosen = opts[action[0]]
        ct = OptionType(chosen.type)
        chosen_cid = None
        if ct in (OptionType.PLAY, OptionType.ATTACH, OptionType.EVOLVE):
            idx = chosen.index
            hand = me.hand or []
            if chosen.area in (AreaType.HAND, None) and idx is not None \
                    and idx < len(hand):
                chosen_cid = hand[idx].id
        if ct == OptionType.PLAY and chosen_cid == BOSS:
            boss_plays.append((ep, st.turn, len(op.prize or []), reward))
        if is_wall_opp:
            if not traj or traj[-1][0] != st.turn:
                traj.append((st.turn, me.deckCount, op.deckCount,
                             len(me.prize or []), len(op.prize or [])))
            offered = used = 0
            for o in opts:
                if OptionType(o.type) != OptionType.ABILITY:
                    continue
                pid = rlplan._board_pokemon_id(o, me)
                if pid in DUD_IDS or pid == FEZ_ID:
                    offered += 1
                    if (ct == OptionType.ABILITY and chosen.area == o.area
                            and chosen.index == o.index):
                        used += 1
            if trig:
                draw_offer_trig += offered
                draw_use_trig += used
            else:
                draw_offer_off += offered
                draw_use_off += used

    if is_wall_opp:
        wall_rows.append(dict(
            ep=ep, reward=reward, traj=traj, trig_true=trig_true,
            trig_prompts=trig_prompts,
            offer_trig=draw_offer_trig, use_trig=draw_use_trig,
            offer_off=draw_offer_off, use_off=draw_use_off))

    # adjudication suspects: loss, never behind on the prize race at the end
    if reward == -1 and last_state is not None:
        me = last_state.players[last_state.yourIndex]
        op = last_state.players[1 - last_state.yourIndex]
        if len(me.prize or []) <= len(op.prize or []) and me.deckCount > 0 \
                and len(op.prize or []) > 0:
            adjudicated.append((ep, last_state.turn, len(steps),
                                len(me.prize or []), len(op.prize or []),
                                me.deckCount, op.deckCount,
                                raw.get("statuses"), raw.get("rewards")))

print(f"=== 1. wall-family games ({len(wall_rows)}) — racemode3 engagement ===")
for w in wall_rows:
    tag = "W" if w["reward"] == 1 else "L"
    t = w["traj"]
    end = t[-1] if t else None
    print(f"{tag} ep{w['ep']}: trigger true {w['trig_true']}/{w['trig_prompts']} prompts; "
          f"dud/fez offered-while-trig {w['offer_trig']} used {w['use_trig']}; "
          f"offered-while-off {w['offer_off']} used {w['use_off']}")
    if t:
        line = " ".join(f"t{a}:{b}v{c}" for a, b, c, _, _ in t[::4])
        print(f"    deck race (mine v opp): {line}  end t{end[0]} "
              f"deck {end[1]}v{end[2]} pz {end[3]}/{end[4]}")

print(f"\n=== 2. adjudication suspects (loss while not behind on prizes, deck>0) ===")
for ep, turn, nsteps, mpz, opz, mdeck, odeck, statuses, rewards in adjudicated:
    print(f"  ep{ep} t={turn} steps={nsteps} pz me{mpz}/opp{opz} "
          f"deck {mdeck}v{odeck} statuses={statuses} rewards={rewards}")

print("\n=== 3. Boss's Orders plays by opp-prizes-left ===")
c = Counter((pz, "W" if rw == 1 else "L") for _, _, pz, rw in boss_plays)
for (pz, wl), n in sorted(c.items()):
    print(f"  opp_pz={pz} {wl}: {n}")
print("plays at opp_pz<=1:", [(ep, t, pz, "W" if rw == 1 else "L")
                              for ep, t, pz, rw in boss_plays if pz <= 1])

print("\n=== 4. WR by opponent-score band ===")
for band in sorted({b for b, _ in band_wl}):
    w, l = band_wl[(band, "W")], band_wl[(band, "L")]
    print(f"  {band}-{band+99}: {w}-{l} ({w/max(w+l,1):.2f})")
