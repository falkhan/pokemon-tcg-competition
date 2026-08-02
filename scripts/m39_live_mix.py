"""M39 P0: build the live opponent mix that weights every ship gate (G-3).

Why this exists: M38's gate was honest and still wrong. The pooled +6.1pp was
carried by the iono/dragapult beds (+14/+17pp) which are ~7% of live games,
while the beds matching the real loss mass (wall, grim, stall) were flat,
stale, or absent. Re-weighted by the live mix the offline delta was ~0 — and
that re-weighting was possible at gate time. From M39 on the WEIGHTED pool is
the gate number, so the weights have to be a committed, reproducible artifact
rather than a number typed into a plan doc.

Source = the cached kaggle replays of the last two ships (M37 55065484 n=94,
M38 55146658 n=55, pooled: 149 ladder-ish games incl. any self-validation
rows, which are dropped here by the mirror rule only if the sub faced itself).
The opponent archetype comes from the replay's deck step — the same fact
scripts/live_postmortem.py reads — classified by an ORDERED signature table.

Usage:
    uv run python scripts/m39_live_mix.py            # rewrite data/m39_live_mix.json
    uv run python scripts/m39_live_mix.py --dry-run  # print, write nothing
"""
import argparse
import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import polars as pl  # noqa: E402

# The two ships whose live samples are pooled into the mix. Keep this at the
# LAST TWO ships (G-6's spirit: an era-old mix is as stale as an era-old bed).
SOURCE_SUBS = (55065484, 55146658)   # M37, M38
OUT = ROOT / "data" / "m39_live_mix.json"
COVERAGE_FLOOR = 0.80                # G-3: a gate roster below this is invalid

# Ordered signature table — FIRST match wins, so precedence is the design.
# Rationale for the contested orderings:
#   mirror first  : any deck with Alakazam/Kadabra is our own family, whatever
#                   else it techs in.
#   wall > kanga  : the Crustle+Mega-Kangaskhan lists play as walls (the
#                   BACKLOG's "Kanga-only wall blindspot" is exactly a wall
#                   list whose Crustle never hits the board).
#   wall > ogerpon: same — Ogerpon is a tech in those lists, not the plan.
#   stall > wall  : Hop's Trevenant / Fan Rotom lists are the deck-out stall
#                   family (M38: 0-2, both deck-outs) and never run Crustle.
FAMILY_SIGNATURES = (
    ("mirror", ("Alakazam", "Kadabra")),
    ("grim", ("Marnie's Grimmsnarl ex",)),
    ("lucario", ("Mega Lucario ex",)),
    ("archaludon", ("Archaludon ex",)),
    ("garchomp", ("Cynthia's Garchomp ex",)),
    ("dragapult", ("Dragapult ex",)),
    ("stall", ("Hop's Trevenant", "Hop's Phantump", "Hop's Snorlax",
               "Fan Rotom")),
    ("wall", ("Crustle", "Dwebble", "Great Tusk", "Terrakion")),
    ("rocket", ("Team Rocket's Mewtwo ex", "Team Rocket's Articuno")),
    ("starmie", ("Mega Starmie ex",)),
    ("froslass", ("Mega Froslass ex",)),
    ("iono", ("Iono's Bellibolt ex",)),
    ("kanga", ("Mega Kangaskhan ex",)),
    ("ogerpon", ("Teal Mask Ogerpon ex", "Cornerstone Mask Ogerpon ex")),
)

# Which gate bed stands in for which live family (G-3 coverage accounting).
# Two beds may share a family — their cells are pooled inside it, and the
# family's live share is what weights the result.
BED_FAMILY = {
    "tuned": "lucario",
    "mirror": "mirror",
    "m28": "mirror",
    "grim": "grim",
    "wall": "wall",
    "stall": "stall",
    "dragapult": "dragapult",
    "iono": "iono",
    "rocket": "rocket",
    "garchomp": "garchomp",
}


def _norm(name: str) -> str:
    """Card names arrive with both straight and curly apostrophes
    (`Hop's Phantump` vs `Hop’s Snorlax`) — match on a normalised form.
    The mojibake form is here because the card CSV is UTF-8 while Windows'
    default locale encoding is cp1252: any reader that forgets `encoding=`
    sees `â€™`, and silently classifies those decks as `other`."""
    return name.replace("â€™", "'").replace("’", "'")


def load_pokemon_names() -> dict:
    names = {}
    with open(ROOT / "data/cards_features.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("is_pokemon") == "true":
                names[int(row["card_id"])] = _norm(row["name"])
    return names


def opponent_deck(steps: list, seat: int) -> list | None:
    """The opponent's 60-card deck from the replay's deck step."""
    for step in steps[:4]:
        st = step[1 - seat] if 1 - seat < len(step) else None
        if isinstance(st, dict):
            action = st.get("action")
            if isinstance(action, list) and len(action) == 60:
                return [int(a) for a in action]
    return None


def classify(pokemon: set) -> str:
    for family, signature in FAMILY_SIGNATURES:
        if any(sig in pokemon for sig in signature):
            return family
    return "other"


def build(subs=SOURCE_SUBS) -> dict:
    names = load_pokemon_names()
    df = pl.read_parquet(ROOT / "data/kaggle/episodes.parquet")
    counts, per_sub, skipped = Counter(), {}, Counter()
    for sub in subs:
        rows = df.filter((pl.col("submission_id_0") == sub)
                         | (pl.col("submission_id_1") == sub))
        sub_counts = Counter()
        for r in rows.iter_rows(named=True):
            seat = 0 if r["submission_id_0"] == sub else 1
            if r["submission_id_0"] == r["submission_id_1"]:
                skipped["self_validation"] += 1
                continue
            path = ROOT / f"data/kaggle/raw/episode_{r['episode_id']}.json.gz"
            if not path.exists():
                skipped["replay_not_cached"] += 1
                continue
            with gzip.open(path) as fh:
                steps = json.load(fh)["steps"]
            deck = opponent_deck(steps, seat)
            if deck is None:
                skipped["no_deck_step"] += 1
                continue
            sub_counts[classify({names[c] for c in deck if c in names})] += 1
        per_sub[str(sub)] = dict(sub_counts)
        counts += sub_counts
    total = sum(counts.values())
    families = {f: {"n": n, "share": round(n / total, 4)}
                for f, n in counts.most_common()}
    return {
        "_doc": "M39 G-3 live opponent mix. Regenerate with "
                "scripts/m39_live_mix.py; the WEIGHTED pool over these shares "
                "is the ship-gate number (scripts/m39_decide.py).",
        "source_subs": list(subs),
        "n_games": total,
        "coverage_floor": COVERAGE_FLOOR,
        "families": families,
        "bed_family": BED_FAMILY,
        "per_sub": per_sub,
        "skipped": dict(skipped),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--subs", type=int, nargs="+", default=list(SOURCE_SUBS))
    args = ap.parse_args()

    mix = build(tuple(args.subs))
    print(f"live mix over subs {mix['source_subs']} — {mix['n_games']} games")
    for family, d in mix["families"].items():
        beds = [b for b, f in BED_FAMILY.items() if f == family]
        print(f"  {family:12s} n={d['n']:3d}  share={d['share']:.4f}  "
              f"beds: {', '.join(beds) or 'NONE'}")
    covered = sum(d["share"] for f, d in mix["families"].items()
                  if f in set(BED_FAMILY.values()))
    print(f"  roster coverage (full bed set): {covered:.1%} "
          f"(floor {COVERAGE_FLOOR:.0%})")
    if mix["skipped"]:
        print(f"  skipped: {mix['skipped']}")
    if args.dry_run:
        return
    OUT.write_text(json.dumps(mix, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
