"""Cross-submission live forensics: W/L per opponent FAMILY and score band.

Usage: uv run python scripts/live_family_forensics.py <sub_id> [<sub_id> ...]
(first refresh the cache: uv run python -m rl.kaggle_ingest refresh --subs <ids>)

Complements live_postmortem.py (single-sub, per-decision behaviour): this one
answers "who are we losing to, and does it differ between ships". Families use
m39_live_mix's canonical signature table so the buckets line up with the
G-13 roster's coverage accounting.
"""
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import polars as pl
from m39_live_mix import FAMILY_SIGNATURES, load_pokemon_names, opponent_deck

SCORE_BANDS = ((700, "<700"), (780, "700-780"), (10_000, "780+"))


def family_of(pokemon_names: set[str]) -> str:
    for fam, signatures in FAMILY_SIGNATURES:
        if any(any(sig in name for name in pokemon_names) for sig in signatures):
            return fam
    return "other:" + ",".join(sorted(pokemon_names)[:3])[:30]


def band_of(score: float) -> str:
    for upper, label in SCORE_BANDS:
        if score < upper:
            return label
    return SCORE_BANDS[-1][1]


def report(sub: int, episodes: pl.DataFrame, names: dict) -> None:
    d = episodes.filter(
        (pl.col("submission_id_0") == sub) | (pl.col("submission_id_1") == sub)
    ).sort("episode_id")
    fam_wl = defaultdict(lambda: [0, 0])
    band_wl = defaultdict(lambda: [0, 0])
    rows = []
    our_latest = None
    opp_scores = []
    for r in d.iter_rows(named=True):
        ep = int(r["episode_id"])
        seat = 0 if r["submission_id_0"] == sub else 1
        path = ROOT / f"data/kaggle/raw/episode_{ep}.json.gz"
        if not path.exists():
            continue
        raw = json.load(gzip.open(path))
        reward = raw["rewards"][seat] if raw.get("rewards") else None
        if reward not in (1, -1):
            continue
        opp_score = r["updated_score_1"] if seat == 0 else r["updated_score_0"]
        our_latest = r["updated_score_0"] if seat == 0 else r["updated_score_1"]
        deck = opponent_deck(raw["steps"], seat)
        pokemon = {names[c] for c in (deck or []) if c in names}
        fam = family_of(pokemon)
        won = 1 if reward == 1 else 0
        fam_wl[fam][0] += won
        fam_wl[fam][1] += 1 - won
        band_wl[band_of(opp_score)][0] += won
        band_wl[band_of(opp_score)][1] += 1 - won
        opp_scores.append(opp_score)
        rows.append(won)
    n = len(rows)
    wins = sum(rows)
    print(f"\n=== sub {sub}: {wins}W-{n - wins}L (wr {wins / max(n, 1):.2f}) | "
          f"latest score {our_latest:.1f} | mean opp score "
          f"{sum(opp_scores) / max(n, 1):.0f}")
    print("  by family:")
    for fam, (w, l) in sorted(fam_wl.items(), key=lambda kv: -sum(kv[1])):
        print(f"    {fam:28s} {w:2d}W-{l:2d}L  wr={w / max(w + l, 1):.2f}")
    print("  by opponent score band:")
    for _, label in SCORE_BANDS:
        w, l = band_wl[label]
        print(f"    {label:8s} {w:2d}W-{l:2d}L  wr={w / max(w + l, 1):.2f}")
    print("  last 10: " + "".join("W" if w else "L" for w in rows[-10:]))


def main() -> None:
    subs = [int(a) for a in sys.argv[1:]]
    if not subs:
        sys.exit(__doc__)
    names = load_pokemon_names()
    episodes = pl.read_parquet(ROOT / "data/kaggle/episodes.parquet")
    for sub in subs:
        report(sub, episodes, names)


if __name__ == "__main__":
    main()
