"""Targeted probes: brick anatomy, Boss@opp<=1 misfires, our deck verification."""
import csv, gzip, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import polars as pl
from cg.api import AreaType, OptionType, SelectContext, to_observation_class
from rl.replay_bc import iter_replay_decisions

ROOT = Path(__file__).resolve().parent.parent
SUB = 55030954
BOSS = 1182

names, is_basic = {}, set()
with open(ROOT / "data/cards_features.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        cid = int(row["card_id"])
        names[cid] = row["name"]
        if row.get("is_basic") == "true":
            is_basic.add(cid)

df = pl.read_parquet(ROOT / "data/kaggle/episodes.parquet").filter(
    (pl.col("submission_id_0") == SUB) | (pl.col("submission_id_1") == SUB)
).sort("episode_id")

BRICKS = {88459419, 88466529, 88496613, 88590857}
boss_plays = []   # (ep, turn, opp_prize_left, reward)

for r in df.iter_rows(named=True):
    ep = int(r["episode_id"])
    seat = 0 if r["submission_id_0"] == SUB else 1
    path = ROOT / f"data/kaggle/raw/episode_{ep}.json.gz"
    if not path.exists():
        continue
    raw = json.load(gzip.open(path))
    steps = raw["steps"]
    reward = raw["rewards"][seat] if raw.get("rewards") else None

    # our deck (first action of our seat) — verify h4 counts once
    if ep == min(int(x) for x in df["episode_id"]):
        for step in steps[:4]:
            st = step[seat] if seat < len(step) else None
            if isinstance(st, dict):
                act = st.get("action")
                if isinstance(act, list) and len(act) == 60:
                    cnt = Counter(names.get(int(a), str(a)) for a in act)
                    print(f"OUR DECK (ep{ep}):")
                    for n, c in sorted(cnt.items()):
                        print(f"  {c}x {n}")
                    break

    drops = Counter()
    brick_log = []
    for i, obs_dict, action in iter_replay_decisions(steps, seat, drops):
        obs = to_observation_class(obs_dict)
        st = obs.current
        if st is None or obs.select is None or obs.select.context != SelectContext.MAIN:
            continue
        me = st.players[st.yourIndex]
        opp = st.players[1 - st.yourIndex]
        hand = me.hand or []
        opts = obs.select.option
        chosen = opts[action[0]]
        ct = OptionType(chosen.type)
        chosen_cid = None
        if ct in (OptionType.PLAY, OptionType.ATTACH, OptionType.EVOLVE):
            idx = chosen.index
            if chosen.area in (AreaType.HAND, None) and idx is not None and idx < len(hand):
                chosen_cid = hand[idx].id
        if ct == OptionType.PLAY and chosen_cid == BOSS:
            boss_plays.append((ep, st.turn, len(opp.prize or []), reward))
        if ep in BRICKS:
            basics_in_hand = [names[c.id] for c in hand if c.id in is_basic]
            bench_n = len([b for b in (me.bench or []) if b])
            brick_log.append((st.turn, bench_n, len(hand), basics_in_hand,
                              str(OptionType(chosen.type)).split(".")[-1]))
    if ep in BRICKS:
        print(f"\nBRICK ep{ep} (reward {reward}) — turn, bench, handsize, basics-in-hand, chosen:")
        for row in brick_log:
            print("  ", row)

print("\n=== Boss's Orders plays by opp-prizes-left ===")
c = Counter((pz, "W" if rw == 1 else "L") for _, _, pz, rw in boss_plays)
for (pz, wl), n in sorted(c.items()):
    print(f"  opp_pz={pz} {wl}: {n}")
low = [(ep, t, pz, rw) for ep, t, pz, rw in boss_plays if pz <= 1]
print("plays at opp_pz<=1:", low)
