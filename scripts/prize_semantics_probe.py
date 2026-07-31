"""Pin the engine's prize-array semantics against the LIVE engine (M37 audit).

Why this exists
---------------
`players[i].prize` is the prizes THAT player still has to take, and the array
drains for whoever scores the KO. The competition's own reference agent
(`sample-agent/main.py:283`) assumes the opposite, and code in this repo
inherited that inversion three separate times — most damagingly in
`rl/turn_solver.score_leaf`, where it made every knockout the solver could take
score -150,000 (see docs/m37-code-audit.md).

It survived 30 milestones of green tests because `tests/builders.py` builds
synthetic states that can drain the opponent's array during our own turn — a
transition the real engine never produces. So this probe deliberately uses the
real engine and real pilots, and is the instrument to re-run before trusting
any prize-related change.

Usage
-----
  uv run python scripts/prize_semantics_probe.py            # all three checks
  uv run python scripts/prize_semantics_probe.py --games 20

Checks
------
1. end-state    : the WINNER's own array should reach 0.
2. attribution  : the array that drains should belong to the seat that acted.
3. leaf-scoring : every leaf where WE took prizes should score POSITIVE
                  (this is the regression check for the score_leaf inversion;
                  needs the solver pilot, so it is the slow one).
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from rl.matchrunner import make_pilot  # noqa: E402

DEFAULT_A = "generic:alakazam_v2_h4"
DEFAULT_B = "generic:lucario"


def _spec(text: str):
    from rl.matchrunner import parse_spec
    return parse_spec(text)


def _play(spec_a, spec_b, on_step=None):
    """One game. on_step(actor, prev_prizes, now_prizes) after every select."""
    pilot_a, deck_a = make_pilot(spec_a, "a")
    pilot_b, deck_b = make_pilot(spec_b, "b")
    obs_dict, start = battle_start(deck_a, deck_b)
    if start.errorPlayer >= 0:
        raise SystemExit(f"battle_start rejected a deck (errorType={start.errorType})")
    pilots = {0: pilot_a, 1: pilot_b}
    prev = None
    steps = 0
    try:
        while obs_dict["current"]["result"] < 0 and steps < 6000:
            if obs_dict.get("select") is None:
                break
            actor = obs_dict["current"]["yourIndex"]
            obs_dict = battle_select(list(pilots[actor](obs_dict)))
            steps += 1
            cur = obs_dict["current"]
            now = tuple(len(cur["players"][s]["prize"]) for s in (0, 1))
            if on_step is not None and prev is not None and now != prev:
                on_step(actor, prev, now)
            prev = now
    finally:
        cur = obs_dict["current"]
        end = tuple(len(cur["players"][s]["prize"]) for s in (0, 1))
        result = cur["result"]
        battle_finish()
    return result, end


def check_endstate_and_attribution(spec_a, spec_b, games: int) -> bool:
    tally = Counter()

    def on_step(actor, prev, now):
        for seat in (0, 1):
            if now[seat] < prev[seat]:
                tally["actor_drained" if seat == actor else "other_drained"] += 1

    decisive = 0
    for _ in range(games):
        result, end = _play(spec_a, spec_b, on_step)
        if result in (0, 1):
            if end[result] == 0 and end[1 - result] > 0:
                tally["winner_array_zero"] += 1
                decisive += 1
            elif end[1 - result] == 0 and end[result] > 0:
                tally["loser_array_zero"] += 1
                decisive += 1

    print(f"[1] end-state   : winner-array-zero={tally['winner_array_zero']} "
          f"loser-array-zero={tally['loser_array_zero']} "
          f"({decisive} decisive of {games} games)")
    print(f"[2] attribution : acting-seat drained={tally['actor_drained']} "
          f"other-seat drained={tally['other_drained']}")

    ok = tally["loser_array_zero"] == 0 and tally["other_drained"] == 0
    if not ok:
        print("    !! prize semantics do NOT match the pinned convention")
    elif not tally["actor_drained"]:
        print("    ?? no prize changes observed — inconclusive, raise --games")
        ok = False
    return ok


def check_leaf_scoring(spec_a, spec_b, games: int) -> bool:
    """Regression check for the score_leaf inversion: leaves where WE took
    prizes must score positive."""
    import rl.turn_solver as ts

    stats = Counter()
    orig = ts.score_leaf

    def traced(snap, obs, dev=False, leaf_value=None):
        score = orig(snap, obs, dev=dev, leaf_value=leaf_value)
        cur = obs.current
        if cur.result < 0:
            me_p = cur.players[snap.me]
            if snap.my_prizes - len(me_p.prize) > 0:      # we took prizes
                stats["prize_leaves"] += 1
                stats["negative" if score < 0 else "positive"] += 1
        return score

    ts.score_leaf = traced
    try:
        for _ in range(games):
            _play(spec_a, spec_b)
    finally:
        ts.score_leaf = orig

    n = stats["prize_leaves"]
    print(f"[3] leaf scoring: {n} leaves where we took prizes — "
          f"{stats['positive']} positive / {stats['negative']} negative")
    if not n:
        print("    ?? no prize-taking leaves — inconclusive, raise --games "
              "(needs a solver spec, e.g. --a solver:alakazam_v2_h4)")
        return False
    if stats["negative"]:
        print("    !! score_leaf is scoring OUR OWN knockouts as losses "
              "(the M37-audit P0 inversion has regressed)")
        return False
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--a", default=DEFAULT_A, help=f"spec A (default {DEFAULT_A})")
    p.add_argument("--b", default=DEFAULT_B, help=f"spec B (default {DEFAULT_B})")
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--skip-leaf", action="store_true",
                   help="skip check 3 (the slow solver-pilot one)")
    args = p.parse_args()

    spec_a, spec_b = _spec(args.a), _spec(args.b)
    print(f"prize-semantics probe: {args.a} vs {args.b}, {args.games} games\n")
    ok = check_endstate_and_attribution(spec_a, spec_b, args.games)
    if not args.skip_leaf:
        # The solver only searches on a trigger, and only some searches reach a
        # prize line, so this check needs a decent number of games to be
        # conclusive — do not shrink it below the end-state count.
        solver_a = _spec(args.a.replace("generic:", "solver:", 1))
        ok = check_leaf_scoring(solver_a, spec_b, max(8, args.games)) and ok

    print("\nPASS — semantics match docs/m37-code-audit.md" if ok
          else "\nFAIL — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
