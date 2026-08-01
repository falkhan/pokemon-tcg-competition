"""How often does the within-turn solver actually override greedy — and what
does PKM_PRIZE_FIX do to that rate?

rl/turn_solver.score_leaf credited its two prize terms to the wrong arrays
(docs/m38-code-audit.md): W_PRIZE (+100k, "per prize I take") landed on prizes
CONCEDED and W_MY_PRIZE (-150k) on prizes TAKEN. Since the lethal tier only
overrides greedy at MIN_OVERRIDE_SCORE (>= 1 prize), a prize-taking line scored
-150k and could never clear the bar — the solver fired only on lines that won
the game outright, which is the module's whole reason to exist gone missing.

This probe instruments solve_turn and reports, for one seeded series:
trigger fires, overrides, how many overrides were prize-taking lines (as
opposed to outright wins), and the best-score distribution. Run it with and
without PKM_PRIZE_FIX=1 to see the gap.

The W/L line is a 2-digit sample, NOT an A/B result — measure strength with
scripts/strength_gate.sh or a matchrunner battery, never with this.

Usage: PKM_PRIZE_FIX=0 uv run python scripts/solver_prize_probe.py -n 10
       PKM_PRIZE_FIX=1 uv run python scripts/solver_prize_probe.py -n 10
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rl.turn_solver as ts
from rl.matchrunner import parse_spec, play_series

STATS = {"fires": 0, "overrides": 0, "prize_lines": 0, "best": []}
_solve_line = ts.solve_turn_line


def _instrumented_solve_turn(obs, deck, deadline_s=None, dev=False,
                             fixes=frozenset(), leaf_value=None,
                             dev_margin=None, max_depth=None, max_nodes=None):
    """solve_turn with counters. Mirrors the real override gate for the lethal
    tier (the dev tier's stand-pat margin is not instrumented — dev=False in
    every `solver:` spec this probe uses)."""
    best_score, best_line, _ = _solve_line(obs, deck, deadline_s, dev, fixes,
                                           max_depth=max_depth,
                                           max_nodes=max_nodes,
                                           leaf_value=leaf_value)
    STATS["fires"] += 1
    STATS["best"].append(best_score)
    if best_line and best_score >= ts.MIN_OVERRIDE_SCORE:
        STATS["overrides"] += 1
        if best_score < ts.W_WIN:          # a prize haul, not a game-ending line
            STATS["prize_lines"] += 1
        return [int(i) for i in best_line[0]]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="solver:lucario")
    ap.add_argument("--b", default="generic:lucario")
    ap.add_argument("-n", "--games", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    ts.solve_turn = _instrumented_solve_turn
    results = play_series(parse_spec(args.a), parse_spec(args.b), args.games,
                          seed=args.seed)
    wins = sum(1 for r in results if r == 0)
    best = STATS["best"]
    fires = max(1, STATS["fires"])
    print(f"PKM_PRIZE_FIX={os.environ.get('PKM_PRIZE_FIX', '0')}  "
          f"{args.a} {wins}-{args.games - wins} vs {args.b} (sample, not an A/B)")
    print(f"  trigger fires      {STATS['fires']}")
    print(f"  overrides          {STATS['overrides']} "
          f"({STATS['overrides'] / fires:.1%} of fires)")
    print(f"  ... prize-taking   {STATS['prize_lines']} "
          f"(the rest are outright wins)")
    if best:
        print(f"  best_score         max {max(best):.0f}  min {min(best):.0f}")


if __name__ == "__main__":
    main()
