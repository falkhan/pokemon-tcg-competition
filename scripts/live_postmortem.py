"""Live post-mortem probe for a shipped submission's cached kaggle replays.

Usage: uv run python scripts/live_postmortem.py <submission_id>
(first refresh the cache: uv run python -m rl.kaggle_ingest refresh --subs <id>)

Per game: loss-reason classification from final board state, opponent
archetype (from the replay's deck step), bench trajectory + missed benching,
per-card trainer offer/play rates, O-rule engagement (Dudunsparce/Fez at low
deck, Sacred Ash timing), END-with-item.
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

ROOT = Path(__file__).resolve().parent.parent
SUB = int(sys.argv[1]) if len(sys.argv) > 1 else 54929991
TELEPATH = 19
DUDUNSPARCE_IDS = {1516, 1517}   # verify against plan.py
FEZ_ID = 140
ASH_ID = None                    # resolved below from plan.py constants

import rl.plan as rlplan
DUDUNSPARCE_IDS = set(getattr(rlplan, "DUDUNSPARCE_IDS", DUDUNSPARCE_IDS))
ASH_ID = getattr(rlplan, "SACRED_ASH_ID", None)
POFFIN_ID = getattr(rlplan, "POFFIN_ID", None)

names, kinds, is_pokemon, is_basic = {}, {}, set(), set()
with open(ROOT / "data/cards_features.csv") as f:
    rows = list(csv.DictReader(f))
    fieldnames = rows[0].keys() if rows else []
    for row in rows:
        cid = int(row["card_id"])
        names[cid] = row["name"]
        for k in ("supporter", "item", "stadium", "tool"):
            if row.get(f"is_{k}") == "true":
                kinds[cid] = k
        if row.get("is_pokemon") == "true":
            is_pokemon.add(cid)
            if row.get("is_basic") == "true":
                is_basic.add(cid)
FEZ_ID = getattr(rlplan, "FEZANDIPITI_ID", FEZ_ID)

df = pl.read_parquet(ROOT / "data/kaggle/episodes.parquet").filter(
    (pl.col("submission_id_0") == SUB) | (pl.col("submission_id_1") == SUB)
).sort("episode_id")

def bench_alive(p):
    return [b for b in (p.bench or []) if b]

def active_alive(p):
    return [a for a in (p.active or []) if a]

games = []
trainer_offers = defaultdict(lambda: [0, 0])   # cid -> [prompt-offers, plays]
kind_offers = defaultdict(lambda: [0, 0])
end_with_item = n_end = 0
end_item_declines = Counter()
dud_low, dud_low_used = 0, 0     # dud ability offered at deck<=6 / used
fez_low, fez_low_used = 0, 0
ash_plays = []                    # deck count at Sacred Ash play
tele_opps = tele_att = 0
bench_prompt_hist = Counter()    # our bench size at MAIN prompts
missed_bench = 0                  # MAIN prompt: bench==0, basic PLAY offered, declined
missed_bench_low = 0              # bench<=1 variant
missed_poffin_low = 0             # bench<=1, poffin offered, declined
supporter_turns = set()          # (ep, turn) where a supporter was played
all_turns = set()

for r in df.iter_rows(named=True):
    ep = int(r["episode_id"])
    seat = 0 if r["submission_id_0"] == SUB else 1
    path = ROOT / f"data/kaggle/raw/episode_{ep}.json.gz"
    if not path.exists():
        continue
    raw = json.load(gzip.open(path))
    steps = raw["steps"]
    reward = raw["rewards"][seat] if raw.get("rewards") else None
    opp_team = r["team_1"] if seat == 0 else r["team_0"]
    opp_score = r["updated_score_1"] if seat == 0 else r["updated_score_0"]
    our_score = r["updated_score_0"] if seat == 0 else r["updated_score_1"]

    # opponent deck from the deck step (first action of opp seat)
    opp_deck = None
    for step in steps[:4]:
        st = step[1 - seat] if 1 - seat < len(step) else None
        if isinstance(st, dict):
            act = st.get("action")
            if isinstance(act, list) and len(act) == 60:
                opp_deck = [int(a) for a in act]
                break
    opp_pokemon = sorted({names.get(c, str(c)) for c in (opp_deck or [])
                          if c in is_pokemon})

    drops = Counter()
    last_state = None
    n_our_prompts = 0
    min_bench_after_t3 = 99
    for i, obs_dict, action in iter_replay_decisions(steps, seat, drops):
        obs = to_observation_class(obs_dict)
        st = obs.current
        if st is None or obs.select is None:
            continue
        last_state = st
        me = st.players[st.yourIndex]
        sel = obs.select
        if sel.context != SelectContext.MAIN:
            continue
        n_our_prompts += 1
        opts = sel.option
        chosen = opts[action[0]]
        hand = me.hand or []
        nb = len(bench_alive(me))
        bench_prompt_hist[nb] += 1
        if st.turn > 3:
            min_bench_after_t3 = min(min_bench_after_t3, nb)
        ct = OptionType(chosen.type)
        chosen_cid = None
        if ct in (OptionType.PLAY, OptionType.ATTACH, OptionType.EVOLVE):
            idx = chosen.index
            if (chosen.area in (AreaType.HAND, None) and idx is not None
                    and idx < len(hand)):
                chosen_cid = hand[idx].id

        # trainer offer/play accounting (distinct card per prompt)
        offered_trainers = set()
        offered_basics = set()
        tele_here = False
        item_playable = []
        for o in opts:
            ot = OptionType(o.type)
            if ot in (OptionType.PLAY, OptionType.ATTACH) and o.index is not None \
                    and o.area in (AreaType.HAND, None) and o.index < len(hand):
                cid = hand[o.index].id
                if cid == TELEPATH:
                    tele_here = True
                if ot == OptionType.PLAY:
                    if cid in kinds:
                        offered_trainers.add(cid)
                        if kinds[cid] == "item":
                            item_playable.append(cid)
                    elif cid in is_basic:
                        offered_basics.add(cid)
        for cid in offered_trainers:
            trainer_offers[cid][0] += 1
            kind_offers[kinds[cid]][0] += 1
        if chosen_cid in offered_trainers and ct == OptionType.PLAY:
            trainer_offers[chosen_cid][1] += 1
            kind_offers[kinds[chosen_cid]][1] += 1
            if kinds[chosen_cid] == "supporter":
                supporter_turns.add((ep, st.turn))
            if chosen_cid == ASH_ID:
                ash_plays.append(me.deckCount)
        all_turns.add((ep, st.turn))
        if tele_here:
            tele_opps += 1
            if chosen_cid == TELEPATH:
                tele_att += 1
        if ct == OptionType.END:
            n_end += 1
            if item_playable:
                end_with_item += 1
                for cid in set(item_playable):
                    end_item_declines[names.get(cid, cid)] += 1
        # missed benching
        if offered_basics and chosen_cid not in offered_basics:
            if nb == 0:
                missed_bench += 1
            if nb <= 1:
                missed_bench_low += 1
        if POFFIN_ID in offered_trainers and chosen_cid != POFFIN_ID and nb <= 1:
            missed_poffin_low += 1
        # low-deck draw-ability engagement
        if me.deckCount <= 6:
            for o in opts:
                if OptionType(o.type) != OptionType.ABILITY:
                    continue
                pok = None
                if o.area == AreaType.ACTIVE and active_alive(me):
                    pok = active_alive(me)[0]
                elif o.area == AreaType.BENCH and o.index is not None \
                        and o.index < len(me.bench or []):
                    pok = (me.bench or [])[o.index]
                if pok is None:
                    continue
                if pok.id in DUDUNSPARCE_IDS:
                    dud_low += 1
                    if ct == OptionType.ABILITY and chosen.area == o.area \
                            and chosen.index == o.index:
                        dud_low_used += 1
                elif pok.id == FEZ_ID:
                    fez_low += 1
                    if ct == OptionType.ABILITY and chosen.area == o.area \
                            and chosen.index == o.index:
                        fez_low_used += 1

    if last_state is None:
        continue
    me = last_state.players[last_state.yourIndex]
    opp = last_state.players[1 - last_state.yourIndex]
    my_prize_left = len(me.prize or [])
    opp_prize_left = len(opp.prize or [])
    my_bench = len(bench_alive(me))
    my_active = len(active_alive(me))
    opp_bench = len(bench_alive(opp))
    opp_active = len(active_alive(opp))
    # classify loss reason from final observed state
    reason = None
    if reward == -1:
        if my_active == 0 and my_bench == 0:
            reason = "BENCHED"
        elif me.deckCount == 0:
            reason = "DECK-OUT"
        elif opp_prize_left == 0:
            reason = "PRIZES"
        else:
            reason = f"other(pz me{my_prize_left}/opp{opp_prize_left})"
    games.append(dict(
        ep=ep, reward=reward, turns=last_state.turn,
        opp_team=opp_team, opp_score=opp_score, our_score=our_score,
        opp_pokemon=opp_pokemon,
        deck_end=me.deckCount, opp_deck_end=opp.deckCount,
        hand_end=len(me.hand or []),
        my_prize_left=my_prize_left, opp_prize_left=opp_prize_left,
        my_bench_end=my_bench, my_active_end=my_active,
        opp_bench_end=opp_bench, opp_active_end=opp_active,
        min_bench_after_t3=min_bench_after_t3,
        reason=reason, n_prompts=n_our_prompts,
    ))

wins = [g for g in games if g["reward"] == 1]
losses = [g for g in games if g["reward"] == -1]
print(f"sub {SUB}: {len(games)} games {len(wins)}W-{len(losses)}L\n")

print("=== per-game table (sorted by episode) ===")
for g in games:
    tag = "W" if g["reward"] == 1 else "L"
    poke = ",".join(p[:12] for p in g["opp_pokemon"][:4])
    print(f"{tag} ep{g['ep']} t={g['turns']:3d} opp={str(g['opp_team'])[:16]:16s} "
          f"oppscore={g['opp_score']:6.0f} deck={g['deck_end']:2d}/{g['opp_deck_end']:2d} "
          f"pzleft={g['my_prize_left']}/{g['opp_prize_left']} "
          f"bench_end={g['my_bench_end']} minbench>t3={g['min_bench_after_t3']:2d} "
          f"hand={g['hand_end']:2d} {g['reason'] or '':10s} [{poke}]")

print("\n=== loss reasons ===")
print(Counter(g["reason"] for g in losses))

print("\n=== bench size at our MAIN prompts (all games) ===")
tot = sum(bench_prompt_hist.values())
for k in sorted(bench_prompt_hist):
    print(f"  bench={k}: {bench_prompt_hist[k]:4d} ({bench_prompt_hist[k]/tot:.1%})")
print(f"missed bench (bench==0, basic playable, declined): {missed_bench} prompts")
print(f"missed bench (bench<=1, basic playable, declined): {missed_bench_low} prompts")
print(f"poffin declined at bench<=1: {missed_poffin_low} prompts")

print(f"\n=== END with playable item: {end_with_item}/{n_end} "
      f"({end_with_item/max(n_end,1):.1%}) ===")
for k, v in end_item_declines.most_common(8):
    print(f"  {k:30s} {v}")

print(f"\n=== telepath attach: {tele_att}/{tele_opps} ({tele_att/max(tele_opps,1):.1%}) ===")
print(f"=== dud ability at deck<=6: used {dud_low_used}/{dud_low} offered ===")
print(f"=== fez ability at deck<=6: used {fez_low_used}/{fez_low} offered ===")
print(f"=== sacred ash plays (deck at play): {sorted(ash_plays)} ===")
print(f"=== turns with >=1 supporter played: {len(supporter_turns)}/{len(all_turns)} "
      f"({len(supporter_turns)/max(len(all_turns),1):.1%}) ===")

print("\n=== trainer offer/play per card (prompt-level, distinct card) ===")
print(f"{'card':32s} {'kind':10s} {'offers':>7s} {'plays':>6s} {'rate':>7s}")
for cid, (off, pl_) in sorted(trainer_offers.items(), key=lambda kv: -kv[1][0]):
    print(f"{names.get(cid, str(cid)):32s} {kinds.get(cid,'?'):10s} {off:7d} {pl_:6d} {pl_/max(off,1):7.1%}")
print("\nby kind:")
for k, (off, pl_) in sorted(kind_offers.items(), key=lambda kv: -kv[1][0]):
    print(f"  {k:10s} {off:6d} offers, played {pl_} = {pl_/max(off,1):.1%}")
