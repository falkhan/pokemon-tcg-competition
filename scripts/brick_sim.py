"""Tier-0 opening-hand brick simulator for the Alakazam deck (M34).

Combinatorial, no game engine: draws N post-mulligan opening hands from a deck list
and classifies how often the hand cannot get the Abra->Kadabra->Alakazam engine
online turn 1. Ranks candidate deck lists in milliseconds before any game collection.

A "brick opening" = post-mulligan hand of 7 with NO Abra AND NO Buddy-Buddy Poffin
(Poffin searches a basic to bench, so it is the other way to start the line). This is
the live sweep fingerprint from docs/m33-post-mortem.md (bench 0-1, line never online).

Usage: uv run python scripts/brick_sim.py decks/alakazam_v2.csv [decks/clone54618168.csv ...]
"""
import csv
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N = 200_000
SEED = 0

names, is_basic_pokemon = {}, set()
with open(ROOT / "data/cards_features.csv") as f:
    for row in csv.DictReader(f):
        cid = int(row["card_id"])
        names[cid] = row["name"]
        if row.get("is_pokemon") == "true" and row.get("is_basic") == "true":
            is_basic_pokemon.add(cid)


def id_for(substr):
    hits = [c for c, n in names.items() if substr.lower() in (n or "").lower()]
    return set(hits)


ABRA = id_for("Abra")
KADABRA = id_for("Kadabra")
RARECANDY = id_for("Rare Candy")
# SEARCH = any card that can fetch a basic Pokemon (Abra) to hand/bench turn 1.
# Poffin (bench), Ultra/Master/Dusk/Love/Great Ball, Rocket's Great Ball (hand).
# These are the outs that keep a no-Abra opening from bricking.
SEARCH = (id_for("Poffin") | id_for("Ultra Ball") | id_for("Master Ball")
          | id_for("Dusk Ball") | id_for("Love Ball") | id_for("Great Ball"))
POFFIN = SEARCH  # name kept for the report line
# draw supporters/items that can dig toward the line
DIG = id_for("Poké Pad") | id_for("Hilda") | id_for("Dawn") | id_for("Xerosic")


def load(path):
    return [int(x) for x in Path(path).read_text().split()]


def classify(deck, rng):
    """Return flags for one post-mulligan opening hand."""
    mulligans = 0
    while True:
        hand = rng.sample(deck, 7)
        if any(c in is_basic_pokemon for c in hand):
            break
        mulligans += 1
        if mulligans > 20:  # pathological safety
            break
    hset = Counter(hand)
    has_abra = any(c in ABRA for c in hand)
    has_poffin = any(c in POFFIN for c in hand)
    has_dig = any(c in DIG for c in hand)
    # brick: cannot start OR search the line turn 1
    brick = not has_abra and not has_poffin
    # soft brick: no start AND no dig to find one
    hard_brick = brick and not has_dig
    # line-ready: Abra in hand AND (Kadabra or Rare Candy) to evolve next
    line_ready = has_abra and any(c in KADABRA or c in RARECANDY for c in hand)
    return dict(mull=mulligans, brick=brick, hard_brick=hard_brick,
                line_ready=line_ready, has_abra=has_abra, has_poffin=has_poffin)


def run(path):
    deck = load(path)
    assert len(deck) == 60, f"{path}: {len(deck)} cards"
    rng = random.Random(SEED)
    agg = Counter()
    mull_total = 0
    for _ in range(N):
        r = classify(deck, rng)
        mull_total += r["mull"]
        for k in ("brick", "hard_brick", "line_ready", "has_abra", "has_poffin"):
            agg[k] += r[k]
    n = N
    print(f"\n=== {path} (n={n:,}) ===")
    print(f"  P(brick: no Abra & no search)     {agg['brick']/n:6.2%}   <-- primary")
    print(f"  P(hard brick: +no dig to find)    {agg['hard_brick']/n:6.2%}")
    print(f"  P(line-ready: Abra + evolve out)  {agg['line_ready']/n:6.2%}")
    print(f"  P(has Abra)  {agg['has_abra']/n:6.2%}   P(has search) {agg['has_poffin']/n:6.2%}")
    print(f"  avg mulligans/opening  {mull_total/n:.3f}")
    return agg['brick'] / n


if __name__ == "__main__":
    paths = sys.argv[1:] or ["decks/alakazam_v2.csv"]
    results = [(p, run(p)) for p in paths]
    if len(results) > 1:
        print("\n=== brick-rate ranking (lower is better) ===")
        for p, b in sorted(results, key=lambda x: x[1]):
            print(f"  {b:6.2%}  {p}")
