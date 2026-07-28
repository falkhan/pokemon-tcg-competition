"""Post-mortem extras: implied ELO, matchup ledger, trajectory, early-loss statuses."""
import gzip, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
SUB = int(sys.argv[1])
SELF_TEAM = "Team Pierogachu"

df = pl.read_parquet(ROOT / "data/kaggle/episodes.parquet").filter(
    (pl.col("submission_id_0") == SUB) | (pl.col("submission_id_1") == SUB)
).sort("episode_id")

FAMS = [
    ("mirror", {"Abra", "Alakazam"}),
    ("lucario", {"Hariyama", "Makuhita"}),
    ("grim/marnie", {"Froslass", "Marnie's Grimmsnarl", "Marnie's Impidimp", "Marnie's Morgrem"}),
    ("crustle/kanga wall", {"Crustle", "Dwebble"}),
    ("hop stall", {"Hop's Trevenant", "Hop's Phantump", "Hop's Snorlax"}),
    ("garchomp", {"Cynthia's Garchomp ex", "Cynthia's Gible"}),
    ("rocket", {"Team Rocket's Mewtwo ex", "Team Rocket's Wobbuffet"}),
    ("dragapult", {"Dragapult ex", "Dreepy"}),
    ("archaludon", {"Archaludon ex", "Duraludon"}),
    ("starmie", {"Mega Starmie", "Staryu"}),
    ("lucario-solrock", {"Solrock", "Riolu"}),
]

names = {}
import csv
with open(ROOT / "data/cards_features.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        names[int(row["card_id"])] = row["name"]

rows = []
for r in df.iter_rows(named=True):
    ep = int(r["episode_id"])
    seat = 0 if r["submission_id_0"] == SUB else 1
    path = ROOT / f"data/kaggle/raw/episode_{ep}.json.gz"
    if not path.exists():
        continue
    raw = json.load(gzip.open(path))
    reward = raw["rewards"][seat] if raw.get("rewards") else None
    statuses = raw.get("statuses")
    opp_team = r["team_1"] if seat == 0 else r["team_0"]
    my_team = r["team_0"] if seat == 0 else r["team_1"]
    opp_score = r["updated_score_1"] if seat == 0 else r["updated_score_0"]
    our_score = r["updated_score_0"] if seat == 0 else r["updated_score_1"]
    # opp deck
    opp_deck = None
    for step in raw["steps"][:4]:
        st = step[1 - seat] if 1 - seat < len(step) else None
        if isinstance(st, dict):
            act = st.get("action")
            if isinstance(act, list) and len(act) == 60:
                opp_deck = [int(a) for a in act]
                break
    poke = {names.get(c, str(c)) for c in (opp_deck or [])}
    fam = "other"
    for fname, keys in FAMS:
        if keys & poke:
            fam = fname
            break
    nsteps = len(raw["steps"])
    rows.append(dict(ep=ep, reward=reward, opp_team=str(opp_team), fam=fam,
                     opp_score=opp_score, our_score=our_score, statuses=statuses,
                     selfplay=(str(opp_team) == SELF_TEAM and str(my_team) == SELF_TEAM),
                     nsteps=nsteps, create=r["create_time"]))

ladder = [g for g in rows if not g["selfplay"]]
W = sum(1 for g in ladder if g["reward"] == 1)
L = sum(1 for g in ladder if g["reward"] == -1)
print(f"total rows {len(rows)}, selfplay excluded {len(rows)-len(ladder)}, ladder {W}W-{L}L (WR {W/max(W+L,1):.3f})")

opp = [g["opp_score"] for g in ladder]
print(f"avg opp score {sum(opp)/len(opp):.1f}")

# implied ELO: solve sum expected = W
def expected(R):
    return sum(1 / (1 + 10 ** ((o - R) / 400)) for o in opp)
lo, hi = 0, 2000
for _ in range(60):
    mid = (lo + hi) / 2
    if expected(mid) < W:
        lo = mid
    else:
        hi = mid
print(f"implied ELO {(lo+hi)/2:.0f}")

print("\nscore trajectory (our updated_score after each game, episode order):")
print(" ".join(f"{g['our_score']:.0f}{'W' if g['reward']==1 else 'L'}" for g in ladder))

print("\n=== matchup ledger ===")
fam_wl = defaultdict(lambda: [0, 0])
for g in ladder:
    fam_wl[g["fam"]][0 if g["reward"] == 1 else 1] += 1
for fam, (w, l) in sorted(fam_wl.items(), key=lambda kv: -(kv[1][0] + kv[1][1])):
    print(f"  {fam:20s} {w}-{l}")

print("\n=== suspicious early losses (statuses + steps) ===")
for g in ladder:
    if g["reward"] == -1 and g["nsteps"] <= 12:
        print(f"  ep{g['ep']} fam={g['fam']} steps={g['nsteps']} statuses={g['statuses']}")
print("\nall loss statuses:")
print(Counter(tuple(g["statuses"]) if g["statuses"] else None for g in ladder if g["reward"] == -1))
