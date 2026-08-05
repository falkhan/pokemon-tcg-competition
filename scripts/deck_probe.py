"""Fixed-pilot deck probe: measure the DECK, with the pilot held constant.

Every deck comparison this project has run so far confounds deck strength with
pilot strength. M40b's Track A is the cleanest example: a grim clone scored
0.436 against the >=0.55 bar and the deck-switch question was called closed —
but that measured OUR NET PILOTING a grim deck (a net cloned from our own
Alakazam list), while the same note records the grim SHELL beating our
archetype 0.70 head-to-head.

This runs every candidate deck against every opponent deck with the SAME
deck-agnostic rule pilot on both sides (rl/generic_pilot.py, spec `generic:`),
so the only thing varying across a row is the 60 cards. Neural arms are
deliberately unavailable here: our checkpoint is an imitation model of one list
and would play any other deck badly, which is the exact confound being removed.

Opponents are weighted by their share of the LEADERBOARD TOP 250
(data/kaggle/leaderboard_decks.parquet), not by our own live mix — the census
found those are very different populations, and the top-of-ladder mix is the
one a deck decision should optimise against.

KNOWN LIMIT, measured: the rule pilot is a weak generalist. Even with the
scaling fix it gets Alakazam to the Active spot ~1.7 times a game against ~3.9
live, so a deck whose plan needs specific setup play is under-measured here.
Read this probe as "how well does a simple pilot do with these 60 cards",
which is a lower bound on the deck, not a verdict on it.

Sizing (docs/VALIDATION.md G-12): `matchrunner --workers 8` is not
run-reproducible, so n=800 resolves ~10pp and n=2400 ~5pp. Deck-level effects
should be large; if they are not, that is itself the answer. Pre-register the
read before running.

Usage:
    uv run python scripts/deck_probe.py --n 800
    uv run python scripts/deck_probe.py --candidates alakazam_v2_h4 my_grass --n 400
    uv run python scripts/deck_probe.py --pilot solver --n 400
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import polars as pl  # noqa: E402

from tcg.decklab import MAX_WORKERS, parse_series_line  # noqa: E402
from tcg.decks import load_deck_file  # noqa: E402

CENSUS_PQ = ROOT / "data/kaggle/leaderboard_decks.parquet"
OUT_JSON = ROOT / "data/deck_probe.json"

# Opponent roster: one representative deck per top-250 family. Harvested lists
# are preferred over hand-built ones — they are what the ladder actually plays.
OPPONENTS = {
    "grim": "data/kaggle/grim_3121746f_deck.csv",
    "mirror": "decks/alakazam_v2_h4.csv",
    "wall": "decks/greattusk_wall.csv",
    "rocket": "data/kaggle/rocket_59e27a5e_deck.csv",
    "ogerpon": "data/kaggle/ogerpon_356c16bd_deck.csv",
    "dragapult": "data/kaggle/dragapult_3631d393_deck.csv",
    "garchomp": "data/kaggle/garchomp_c7b3253f_deck.csv",
    "festival": "decks/festival_dipplin.csv",
}

# `classify()` has no Dipplin/Festival signature, so the census files these
# seats under `other`. Alias the share rather than editing FAMILY_SIGNATURES,
# which is gate-critical and triplicated across the decide scripts.
SHARE_ALIAS = {"festival": "other"}

DEFAULT_CANDIDATES = {
    "ours(alakazam)": "decks/alakazam_v2_h4.csv",
    "grim": "data/kaggle/grim_3121746f_deck.csv",
    "grass(ogerpon)": "data/kaggle/ogerpon_356c16bd_deck.csv",
    "wall": "decks/greattusk_wall.csv",
}


def ladder_shares() -> dict[str, float]:
    """Family -> share of identified teams in the leaderboard top 250."""
    if not CENSUS_PQ.exists():
        raise SystemExit(
            f"{CENSUS_PQ.relative_to(ROOT)} missing — build it first:\n"
            "  uv run python scripts/leaderboard_decks.py --top 250")
    known = pl.read_parquet(CENSUS_PQ).filter(pl.col("label").is_not_null())
    counts = (known.group_by("family").agg(pl.len().alias("n"))
                   .sort("n", descending=True))
    return {r["family"]: r["n"] / known.height for r in counts.iter_rows(named=True)}


def resolve(rel: str) -> Path:
    p = ROOT / rel
    if not p.exists():
        raise SystemExit(f"deck not found: {rel}\n"
                         "  export harvested lists with scripts/export_opp_deck.py")
    ids = load_deck_file(p)
    if len(ids) != 60:
        raise SystemExit(f"{rel} has {len(ids)} cards, expected 60")
    return p


def spec(rel: str, pilot: str) -> str:
    return f"{pilot}:{Path(rel).as_posix()}"


def ci95(p: float, n: int) -> float:
    return 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / max(n, 1))


def weighted_summary(rows: list[dict], candidates) -> list[tuple]:
    """Share-weighted win rate per candidate: (weighted_wr, name, ci95, wsum).

    Pure aggregation over the per-cell rows, split out of main() so the suite
    can pin it on synthetic cells (M41b § II.3a golden fixtures). Candidates
    with no share-carrying cells are dropped; the caller sorts for display.
    """
    summary = []
    for cname in candidates:
        cells = [r for r in rows if r["candidate"] == cname]
        wsum = sum(r["share"] for r in cells)
        if not wsum:
            continue
        weighted = sum(r["share"] * r["wr"] for r in cells) / wsum
        var = sum((r["share"] / wsum) ** 2 * r["wr"] * (1 - r["wr"]) / r["n"]
                  for r in cells)
        summary.append((weighted, cname, 1.96 * math.sqrt(var), wsum))
    return summary


def run_cell(a: str, b: str, n: int, workers: int, seed: int) -> dict | None:
    cmd = [sys.executable, "-m", "rl.matchrunner", "play", "--a", a, "--b", b,
           "-n", str(n), "--workers", str(workers), "--seed", str(seed)]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print(f"    FAILED (exit {proc.returncode}): "
              f"{(proc.stderr or '').strip().splitlines()[-1:]}", flush=True)
        return None
    return parse_series_line(proc.stdout or "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidates", nargs="*", default=None,
                    help="deck names (decks/<name>.csv) or repo-relative paths")
    ap.add_argument("--opponents", nargs="*", default=None,
                    help="subset of: " + " ".join(OPPONENTS))
    ap.add_argument("--n", type=int, default=800, help="games per cell")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pilot", default="generic-scale",
                    choices=("generic-scale", "solver-scale", "generic", "solver"),
                    help="the fixed pilot used on BOTH sides. The -scale kinds "
                         "read rl/scaling.py's effective damage; the plain ones "
                         "read PRINTED damage and systematically favour "
                         "flat-damage decks")
    a = ap.parse_args()

    if not 1 <= a.workers <= MAX_WORKERS:
        raise SystemExit(f"--workers must be 1..{MAX_WORKERS} "
                         "(12 has deadlocked repeatedly via libcg.so)")

    if a.candidates:
        cands = {}
        for c in a.candidates:
            rel = c if c.endswith(".csv") else f"decks/{c}.csv"
            cands[Path(rel).stem] = rel
    else:
        cands = dict(DEFAULT_CANDIDATES)

    opps = {k: v for k, v in OPPONENTS.items()
            if not a.opponents or k in a.opponents}

    for rel in list(cands.values()) + list(opps.values()):
        resolve(rel)

    shares = ladder_shares()
    covered = sum(shares.get(SHARE_ALIAS.get(f, f), 0.0) for f in opps)
    print(f"deck probe — pilot `{a.pilot}` on BOTH sides, n={a.n}/cell, "
          f"seed {a.seed}, {a.workers} workers", flush=True)
    print(f"opponents cover {covered:.1%} of the identified top-250 field\n", flush=True)

    rows, t0 = [], time.time()
    for cname, crel in cands.items():
        for oname, orel in opps.items():
            res = run_cell(spec(crel, a.pilot), spec(orel, a.pilot),
                           a.n, a.workers, a.seed)
            if res is None:
                continue
            wr = res["wr"]
            rows.append({"candidate": cname, "opponent": oname, "n": res["n"],
                         "wr": wr, "ci": ci95(wr, res["n"]),
                         "w": res["w"], "l": res["l"], "d": res["d"],
                         "share": shares.get(SHARE_ALIAS.get(oname, oname), 0.0),
                         "mirror": crel == orel})
            tag = "  (mirror)" if crel == orel else ""
            print(f"  {cname:16s} vs {oname:10s}  {wr:.3f} ±{ci95(wr, res['n']):.3f}"
                  f"  ({res['w']}W {res['l']}L {res['d']}D){tag}", flush=True)
        print(flush=True)

    if not rows:
        raise SystemExit("no cells completed")

    print(f"\n{'=' * 72}\nWEIGHTED BY TOP-250 SHARE "
          "(mirror cells INCLUDED — see below)\n", flush=True)
    # Mirror cells are included. Excluding them drops a different slice of the
    # field per candidate: grim's own mirror is 49.5% of the top 250 while
    # ogerpon's is 3.6%, so dropping mirrors compared grim-minus-its-hardest-
    # matchup against everyone else's full slate. Facing your own archetype is
    # a real part of the field, and it measures ~0.5 anyway.
    summary = weighted_summary(rows, cands)
    for weighted, cname, ci, wsum in sorted(summary, reverse=True):
        print(f"  {cname:16s} {weighted:.3f} ±{ci:.3f}   "
              f"(covering {wsum:.1%} of the field)", flush=True)

    grim = [r for r in rows if r["opponent"] == "grim" and not r["mirror"]]
    if grim:
        print("\nvs grim alone (49.5% of the top 250):", flush=True)
        for r in sorted(grim, key=lambda r: -r["wr"]):
            print(f"  {r['candidate']:16s} {r['wr']:.3f} ±{r['ci']:.3f}", flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        {"pilot": a.pilot, "n": a.n, "seed": a.seed, "shares": shares,
         "candidates": cands, "opponents": opps, "cells": rows}, indent=2),
        encoding="utf-8")
    print(f"\nwrote {OUT_JSON.relative_to(ROOT)} "
          f"({len(rows)} cells, {time.time() - t0:.0f}s)", flush=True)
    print("\nG-12: n={} resolves roughly ±{:.0f}pp per cell. A delta inside that "
          "is 'no signal', never 'no effect'.".format(a.n, 100 * ci95(0.5, a.n)),
          flush=True)


if __name__ == "__main__":
    main()
