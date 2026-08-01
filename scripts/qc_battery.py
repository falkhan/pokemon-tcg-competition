"""Pre-ship QC battery — the bundle vs a SPREAD of decks, not one opponent.

Piotr's 2026-07-27 feedback after the M36 ship: 3 games vs a single agent
family is not proper QC. This battery runs the actual bundle against every
working company sample agent (tuned Mega-Lucario / Iono / Dragapult — the
base sample-agent crashes as a kaggle opponent) PLUS the PREVIOUS ship bundle
as a mirror opponent — the highest-volume live matchup and the one that
exercises the deck-out race. Replays land in replays/<prefix>_<opp>_NNN.html
for the mandatory human review; the ship gate remains Piotr's explicit go.

Usage:
    uv run python scripts/qc_battery.py --prefix m37_qc \
        [--bundle submission/main.py] [-n 3] \
        [--prev dist/submission_neural_<prev_ship>.tar.gz]
"""
import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLE_AGENTS = {
    "tuned": "sample-agent-tuned/main.py",
    "iono": "sample-agent-iono/main.py",
    "dragapult": "sample-agent-dragapult/main.py",
}

# M39: the loss-family legs. M38 went 11W-1L here against a roster covering
# ZERO of the top-3 live loss families — the QC rule was followed to the
# letter and still measured the wrong thing, so the mandatory replay review
# had nothing relevant to look at. These are the families that actually beat
# us, and QC must cover them permanently (docs/M39-plan.md finding #4).
# Built by scripts/m39_build_qc_beds.py; they run PLAIN (our fix stack keys
# on our own cards), so they are the same agents the gate beds are.
# `stall` is deliberately absent: 9 seats at band, unbuildable (docs/M39.md).
BED_AGENTS = {
    "wall": "dist/qc_beds/wall/main.py",
    "grim": "dist/qc_beds/grim/main.py",
    "archaludon": "dist/qc_beds/archaludon/main.py",
    "top900": "dist/qc_beds/top/main.py",
}
PREV_DIR = ROOT / "dist/prev_ship_agent"


def extract_prev(tarball: Path) -> Path:
    """Unpack the previous ship bundle into dist/prev_ship_agent/."""
    if PREV_DIR.exists():
        subprocess.run(["rm", "-rf", str(PREV_DIR)], check=True)
    PREV_DIR.mkdir(parents=True)
    with tarfile.open(tarball) as tf:
        tf.extractall(PREV_DIR, filter="data")
    main = PREV_DIR / "main.py"
    if not main.exists():
        raise SystemExit(f"{tarball} has no top-level main.py")
    return main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="submission/main.py")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("-n", "--games", type=int, default=3)
    ap.add_argument("--prev", default=None,
                    help="previous ship tarball for the mirror leg "
                         "(default: newest-but-one dist/submission_*.tar.gz)")
    args = ap.parse_args()

    from tcg.evaluation import play_games

    opponents = {k: str(ROOT / v) for k, v in SAMPLE_AGENTS.items()}
    for name, rel in BED_AGENTS.items():
        path = ROOT / rel
        if path.exists():
            opponents[name] = str(path)
        else:
            print(f"WARNING: bed leg {name} missing ({rel}) — run "
                  f"scripts/m39_build_qc_beds.py; QC will NOT cover it")
    tarballs = sorted((ROOT / "dist").glob("submission_*.tar.gz"))
    prev = Path(args.prev) if args.prev else \
        (tarballs[-2] if len(tarballs) >= 2 else None)
    if prev is not None:
        opponents["prevship"] = str(extract_prev(prev))
        print(f"mirror leg: previous ship bundle {prev.name}")
    else:
        print("WARNING: no previous ship tarball found — mirror leg SKIPPED")

    total_w = total_l = 0
    failures = []
    for name, opp in opponents.items():
        wr, results = play_games(args.bundle, opp, args.games,
                                 replay_prefix=f"{args.prefix}_{name}")
        w = sum(1 for r in results if r[0] > r[1])
        l = sum(1 for r in results if r[0] < r[1])
        total_w += w
        total_l += l
        print(f"  vs {name:9s} {w}W-{l}L  ({results})")
        if w == 0:
            failures.append(name)

    print(f"\nQC battery: {total_w}W-{total_l}L over "
          f"{len(opponents)} opponents x {args.games} games; "
          f"replays: replays/{args.prefix}_<opp>_NNN.html")
    if failures:
        print(f"SWEPT 0-{args.games} by: {', '.join(failures)} — "
              f"investigate before asking for the ship go")
    # docs/VALIDATION.md Tier 4: at n=3/opponent the minimum detectable
    # effect is near 100pp. This battery cannot measure strength and its W-L
    # must never be quoted as evidence of it (M38 cited 11W-1L that way).
    print("\nThis is a SMOKE TEST — crashes, timeouts, illegal actions, "
          "catastrophic breakage. It is NOT a strength measurement at n="
          f"{args.games}/opponent; the weighted gate is. Its real output is "
          "the replays, for Piotr's mandatory review.")


if __name__ == "__main__":
    main()
