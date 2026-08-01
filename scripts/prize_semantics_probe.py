"""Which player's `.prize` array drains when a player takes prizes?

The engine-authoritative answer to the question rl/plan.py, rl/turn_solver.py
and rl/collector.py disagreed about for 37 milestones (docs/m38-code-audit.md).
M36 pinned it from diag end-states + a kyogre probe; this re-derives it from
live games so any future doubt is one command away.

Plays N direct-engine games and prints, per game, the winner seat, both seats'
final prize-array lengths, and the last few prize transitions.

  Hypothesis A (own-needs): `player.prize` = the prizes THAT player still
  needs -> it drains as THEY take prizes -> the WINNER's own array ends at 0.
  Hypothesis B (the swapped convention baked into _make_plan / score_leaf /
  the prize shaping): it drains as their OPPONENT takes prizes -> the LOSER's
  array ends at 0.

Games won by deck-out end with neither array at 0 and are reported as such —
only prize-out games discriminate.

Usage: uv run python scripts/prize_semantics_probe.py [-n 6] [--a generic:lucario]
                                                      [--b generic:kyogre]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.matchrunner import make_pilot, parse_spec


def probe(spec_a: str, spec_b: str, n_games: int) -> list[tuple]:
    from cg.game import battle_finish, battle_select, battle_start

    fn_a, deck_a = make_pilot(parse_spec(spec_a), instance="prz_a")
    fn_b, deck_b = make_pilot(parse_spec(spec_b), instance="prz_b")
    rows = []
    for _ in range(n_games):
        obs_dict, start = battle_start(deck_a, deck_b)
        if start.errorPlayer >= 0:
            battle_finish()
            raise SystemExit(f"battle_start rejected a deck: {start.errorType}")
        prev, moves = None, []
        try:
            while obs_dict["current"]["result"] < 0:
                cur = obs_dict["current"]
                seat = cur["yourIndex"]
                lens = tuple(len(p.get("prize") or []) for p in cur["players"])
                if prev is not None and lens != prev and 0 not in prev:
                    moves.append((cur["turn"], seat, prev, lens))
                prev = lens
                fn = fn_a if seat == 0 else fn_b
                obs_dict = battle_select([int(i) for i in fn(obs_dict)])
            cur = obs_dict["current"]
            rows.append((cur["result"],
                         tuple(len(p.get("prize") or []) for p in cur["players"]),
                         tuple(p.get("deckCount") for p in cur["players"]),
                         moves[-3:]))
        finally:
            battle_finish()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="generic:lucario")
    ap.add_argument("--b", default="generic:kyogre")
    ap.add_argument("-n", "--games", type=int, default=6)
    args = ap.parse_args()

    rows = probe(args.a, args.b, args.games)
    votes = {"A": 0, "B": 0}
    for result, prizes, decks, moves in rows:
        verdict = "deck-out/draw (uninformative)"
        if result in (0, 1):
            if prizes[result] == 0:
                verdict, votes["A"] = "A (own-needs)", votes["A"] + 1
            elif prizes[1 - result] == 0:
                verdict, votes["B"] = "B (opponent-drains)", votes["B"] + 1
        print(f"winner={result} prize_lens={prizes} decks={decks} -> {verdict}")
        for turn, seat, before, after in moves:
            print(f"    t{turn:<3} prompt to seat {seat}: {before} -> {after}")
    print(f"\nverdict: A={votes['A']}  B={votes['B']}  "
          f"(uninformative {len(rows) - votes['A'] - votes['B']})")
    if votes["A"] and not votes["B"]:
        print("A confirmed: player.prize = the prizes THAT player still needs.")


if __name__ == "__main__":
    main()
