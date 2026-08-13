"""M44 Step 5 — launch one league leg (docs/M44-plan.md r4).

Builds the exact pre-registered rl.ppo command for pair P from the LIVE
roster: --start is P's current league net (its seed until its first leg
promotes), opponents are the OTHER pairs' current nets at 0.15 each plus
mirror=0.25, past=0.10 and the two rule anchors at 0.10 (sums to 1.00).
Rule anchors are spelled in full (bare `rule:tuned` crashes on the absent
decks/tuned.csv).

Prints the command and execs it with the single-thread BLAS env the M43 legs
ran under. Pause: `touch checkpoints/ppo_pause_m44_<P>_r1`; resume: rerun
with --resume (identical CLI otherwise — rl.ppo enforces it).

Usage:
  uv run python scripts/m44_leg.py K [--iterations 25] [--resume] [--dry-run]
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PAIRS = ("K", "L", "O", "G")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", choices=PAIRS)
    ap.add_argument("--iterations", type=int, default=25,
                    help="25 pre-registered; 15 is the compression setting")
    ap.add_argument("--roster", default=str(ROOT / "docs/specs/m44_roster.json"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    players = json.loads(Path(args.roster).read_text())["players"]
    by_id = {p["id"]: p for p in players}
    if args.pair not in by_id:
        raise SystemExit(f"pair {args.pair} not in roster yet (Step 4 pick?)")
    me = by_id[args.pair]
    others = [by_id[q] for q in PAIRS if q != args.pair and q in by_id]
    if len(others) != 3:
        raise SystemExit(f"expected 3 opponent pairs in roster, found "
                         f"{[p['id'] for p in others]}")

    start = Path(me["net"]).name          # rl.ppo resolves against checkpoints/
    tag = f"m44_{args.pair}_r1"
    cmd = [sys.executable, "-m", "rl.ppo",
           "--start", start,
           "--learn-deck", me["deck"], "--eval-deck", me["deck"],
           "--iterations", str(args.iterations),
           "--games-per-iter", "400", "--workers", "8",
           "--eval-every", "5", "--eval-games", "200",
           "--kl-coef", "0.1", "--device", "auto", "--tag", tag,
           "--opponents",
           *[f"{q['spec']}=0.15" for q in others],
           "mirror=0.25", "past=0.10",
           "rule:tuned:lucario=0.10", "rule:iono=0.10"]
    if args.resume:
        cmd.append("--resume")

    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    print("m44 leg", args.pair, "->", " ".join(cmd), flush=True)
    if args.dry_run:
        return 0
    os.execve(sys.executable, cmd, env)


if __name__ == "__main__":
    raise SystemExit(main())
