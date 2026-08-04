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


# --- measurement core (pure — golden-fixtured in tests/test_probes_engine_side.py).
# The engine runs live below; these functions are the semantics the probe
# ASSERTS about what it observed, so they must be pinnable without the engine.

def prize_transition_keys(actor: int, prev: tuple, now: tuple) -> list:
    """Tally keys for one observed prize-array change. The pinned convention
    (docs/m37-code-audit.md): only the ACTING seat's own array may drain."""
    return ["actor_drained" if seat == actor else "other_drained"
            for seat in (0, 1) if now[seat] < prev[seat]]


def end_state_key(result: int, end: tuple):
    """Which seat's array reached zero in a decisive game, or None when
    neither did (draws, and wins that ended some other way)."""
    if result not in (0, 1):
        return None
    if end[result] == 0 and end[1 - result] > 0:
        return "winner_array_zero"
    if end[1 - result] == 0 and end[result] > 0:
        return "loser_array_zero"
    return None


def endstate_verdict(tally: Counter) -> str:
    """'ok' | 'mismatch' | 'inconclusive'. Silence (no drain ever observed)
    is inconclusive, never a pass."""
    if tally["loser_array_zero"] or tally["other_drained"]:
        return "mismatch"
    if not tally["actor_drained"]:
        return "inconclusive"
    return "ok"


def leaf_keys(score: float, result: int, prizes_taken: int) -> list:
    """Tally keys for one score_leaf call: a non-terminal leaf where WE took
    prizes must score positive (the score_leaf-inversion regression check)."""
    if result >= 0 or prizes_taken <= 0:
        return []
    return ["prize_leaves", "negative" if score < 0 else "positive"]


def leaf_verdict(stats: Counter) -> str:
    """'ok' | 'regressed' | 'inconclusive'."""
    if not stats["prize_leaves"]:
        return "inconclusive"
    if stats["negative"]:
        return "regressed"
    return "ok"


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
        tally.update(prize_transition_keys(actor, prev, now))

    decisive = 0
    for _ in range(games):
        result, end = _play(spec_a, spec_b, on_step)
        key = end_state_key(result, end)
        if key is not None:
            tally[key] += 1
            decisive += 1

    print(f"[1] end-state   : winner-array-zero={tally['winner_array_zero']} "
          f"loser-array-zero={tally['loser_array_zero']} "
          f"({decisive} decisive of {games} games)")
    print(f"[2] attribution : acting-seat drained={tally['actor_drained']} "
          f"other-seat drained={tally['other_drained']}")

    verdict = endstate_verdict(tally)
    if verdict == "mismatch":
        print("    !! prize semantics do NOT match the pinned convention")
    elif verdict == "inconclusive":
        print("    ?? no prize changes observed — inconclusive, raise --games")
    return verdict == "ok"


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
            stats.update(leaf_keys(score, cur.result,
                                   snap.my_prizes - len(me_p.prize)))
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
    verdict = leaf_verdict(stats)
    if verdict == "inconclusive":
        print("    ?? no prize-taking leaves — inconclusive, raise --games "
              "(needs a solver spec, e.g. --a solver:alakazam_v2_h4)")
    elif verdict == "regressed":
        print("    !! score_leaf is scoring OUR OWN knockouts as losses "
              "(the M37-audit P0 inversion has regressed)")
    return verdict == "ok"


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
