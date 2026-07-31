"""M38 Phase 0 merged teacher battery — arms A0-A3 over non-solver beds.

Pre-registered in docs/M38-plan.md (E0a strength + E0b over-greed probe;
E0c label shift lives in scripts/m38_label_shift.py). Arms differ ONLY by
the rl/turn_solver env flags, so each arm invocation is its own process:
this script sets the flags from --arm BEFORE any rl import (spawn workers
re-import modules from disk but inherit the environment — the proven
M22_WHOLE_BOARD idiom; a mid-run edit voided the first C1 battery).

| arm | leaf                       | bar                          |
|-----|----------------------------|------------------------------|
| A0  | old inverted (M38_OLD_LEAF)| current (W_PRIZE - 1)        |
| A1  | corrected                  | current (W_PRIZE - 1)        |
| A2  | corrected                  | lowered constant (W_PRIZE/2) |
| A3  | corrected                  | semantic prize-or-win gate   |

E0a beds are NON-SOLVER only — correcting the solver also changes any
solver-backed bed opponent, and the campaign has two documented bed-fidelity
failures already (garchomp, wall). n=800/cell resolves ~±5pp at 95%; do not
over-read "no difference" on subtle effects (pre-registered MDE).

Usage
-----
  uv run python scripts/m38_battery.py run  --arm A2 [-n 800] [--workers 8]
      [--beds tuned,iono,...] [--deck alakazam_v2_h4] [--seed 0]
  uv run python scripts/m38_battery.py all  [-n 800] [--workers 8] ...
  uv run python scripts/m38_battery.py e0b  --arm A2 [-n 60] [--bed tuned]
  uv run python scripts/m38_battery.py e0b-all [-n 60] [--bed tuned]
  uv run python scripts/m38_battery.py decode

Results: runs/m38_{arm}_{bed}_s{seed}.jsonl (run_pairs chunk checkpoints —
rerunning the same cell RESUMES it). E0b: runs/m38_e0b_{arm}_{bed}.json.
"""
import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

ARMS = {
    "A0": {"M38_OLD_LEAF": "1", "M38_BAR": "current"},
    "A1": {"M38_OLD_LEAF": "0", "M38_BAR": "current"},
    "A2": {"M38_OLD_LEAF": "0", "M38_BAR": "const"},
    "A3": {"M38_OLD_LEAF": "0", "M38_BAR": "semantic"},
}

# Non-solver beds only (see module docstring). Sample agents via rl/teacher
# specs; clones + lineage via model: ckpts; "mirror" = the champion net ON
# THE ARM'S DECK (the deck-race stressor without a solver on the other side).
BEDS = {
    "tuned": "rule:tuned:lucario",
    "iono": "rule:iono",
    "dragapult": "rule:dragapult",
    "grim": "model:checkpoints/m25_bc_grim_54861775.pt:"
            "data/kaggle/grimmsnarl_3121746f_deck.csv",
    "rocket": "model:checkpoints/m30_bc_rocket_54834745.pt:"
              "data/kaggle/rocket_3394cd30_deck.csv",
    "m28": "model:checkpoints/m28_winners.pt:clone54618168",
    "mirror": "model:checkpoints/m28_winners.pt:{deck}",
}
DEFAULT_DECK = "alakazam_v2_h4"


def _set_arm_env(arm: str) -> None:
    """MUST run before any rl import — module flags are read at import time."""
    assert "rl.turn_solver" not in sys.modules, \
        "rl.turn_solver imported before the arm env was set"
    os.environ.update(ARMS[arm])


def _notify(text: str) -> None:
    """Hermes Telegram ping (rl/live_monitor.py pattern) — never fatal."""
    try:
        subprocess.run(["hermes", "send", "-t", "telegram", text],
                       check=False, timeout=30)
    except Exception:
        print("(hermes unavailable — notification skipped)", file=sys.stderr)


def _bed_spec(bed: str, deck: str) -> str:
    return BEDS[bed].format(deck=deck)


# ------------------------------------------------------------------- E0a ---

def run_e0a(arm: str, deck: str, beds: list[str], n: int, workers: int,
            seed: int) -> None:
    from rl.matchrunner import parse_spec, run_pairs, series_wr
    RUNS.mkdir(exist_ok=True)
    arm_spec = parse_spec(f"solver:{deck}")
    print(f"[m38 E0a] arm {arm} env={ARMS[arm]} deck={deck} "
          f"n={n}/bed workers={workers}", flush=True)
    for bed in beds:
        ckpt = RUNS / f"m38_{arm}_{bed}_s{seed}.jsonl"
        results = run_pairs([(arm_spec, parse_spec(_bed_spec(bed, deck)), n)],
                            workers=workers, seed=seed, checkpoint=str(ckpt))
        wr = series_wr(results[0])
        print(f"[m38 E0a] {arm} vs {bed:9s} wr {wr:.4f} n={len(results[0])} "
              f"-> {ckpt.name}", flush=True)
        _notify(f"[m38] E0a {arm} vs {bed}: wr {wr:.3f} n={len(results[0])}")


# ------------------------------------------------------------------- E0b ---

def run_e0b(arm: str, deck: str, bed: str, games: int) -> None:
    """Over-greed probe: how often does the arm END ITS TURN with the active
    exposed to a return-KO (opp active can pay a lethal attack now / with one
    attach), overall and specifically on turns where it took a prize. The
    W-vector was tuned while the prize term was inverted — W_COUNTER has
    never had to restrain a live prize reward (docs/M38-plan.md E0b)."""
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start
    from rl.combat import _CARD, _best_damage
    from rl.matchrunner import make_pilot, parse_spec

    arm_fn, arm_deck = make_pilot(parse_spec(f"solver:{deck}"), "e0b_a")
    bed_fn, bed_deck = make_pilot(parse_spec(_bed_spec(bed, deck)), "e0b_b")
    tally: Counter = Counter()

    def try_exposure(obs_dict, a_seat, took):
        """Exposure at the first post-turn-end prompt where BOTH actives
        exist. On KO turn-ends the opponent's active slot is None until they
        promote — the original version returned early there, which excluded
        every prize-taking turn-end by construction (the E0b v1 bug)."""
        obs = to_observation_class(obs_dict)
        me = obs.current.players[a_seat]
        op = obs.current.players[1 - a_seat]
        my_active = me.active[0] if me.active and me.active[0] is not None else None
        op_active = op.active[0] if op.active and op.active[0] is not None else None
        if my_active is None or op_active is None or op_active.id not in _CARD:
            return False                       # keep pending (e.g. mid-promote)
        now = _best_damage(op_active, my_active) >= my_active.hp
        nxt = _best_damage(op_active, my_active, extra_energy=1) >= my_active.hp
        tally["exposure_measured"] += 1
        tally["exposed_now"] += now
        tally["exposed_attach"] += nxt
        if took:
            tally["prize_exposure_measured"] += 1
            tally["prize_exposed_now"] += now
            tally["prize_exposed_attach"] += nxt
        return True

    for g in range(games):
        a_seat = g % 2
        decks = (arm_deck, bed_deck) if a_seat == 0 else (bed_deck, arm_deck)
        pilots = {a_seat: arm_fn, 1 - a_seat: bed_fn}
        obs_dict, start = battle_start(decks[0], decks[1])
        if start.errorPlayer >= 0:
            raise SystemExit(f"battle_start rejected a deck "
                             f"(errorType={start.errorType})")
        prev_actor, prizes_at_start, steps = None, 6, 0
        pending = None            # (took,) awaiting an exposure-measurable obs
        try:
            while obs_dict["current"]["result"] < 0 and steps < 6000:
                if obs_dict.get("select") is None:
                    break
                actor = obs_dict["current"]["yourIndex"]
                if prev_actor == a_seat and actor != a_seat:
                    me_prize = len(
                        obs_dict["current"]["players"][a_seat]["prize"])
                    took = prizes_at_start - me_prize > 0
                    tally["turn_ends"] += 1
                    tally["prize_turns"] += took
                    pending = (took,)
                if pending is not None and try_exposure(obs_dict, a_seat,
                                                        pending[0]):
                    pending = None
                if actor == a_seat and prev_actor != a_seat:
                    prizes_at_start = len(
                        obs_dict["current"]["players"][a_seat]["prize"])
                    pending = None    # my turn again: stale exposure dropped
                obs_dict = battle_select(list(pilots[actor](obs_dict)))
                prev_actor = actor
                steps += 1
            tally["games"] += 1
            res = obs_dict["current"]["result"]
            tally["wins"] += res == a_seat
        finally:
            battle_finish()

    def rate(k, d):
        return tally[k] / tally[d] if tally[d] else 0.0

    out = {
        "arm": arm, "bed": bed, "deck": deck, **tally,
        "walk_in_rate": rate("exposed_now", "exposure_measured"),
        "walk_in_attach_rate": rate("exposed_attach", "exposure_measured"),
        "prize_walk_in_rate": rate("prize_exposed_now",
                                   "prize_exposure_measured"),
        "prize_walk_in_attach_rate": rate("prize_exposed_attach",
                                          "prize_exposure_measured"),
    }
    RUNS.mkdir(exist_ok=True)
    path = RUNS / f"m38_e0b_{arm}_{bed}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[m38 E0b] {arm} vs {bed}: turn_ends={tally['turn_ends']} "
          f"walk-in now/attach {out['walk_in_rate']:.3f}/"
          f"{out['walk_in_attach_rate']:.3f}  prize-turns "
          f"{tally['prize_turns']} walk-in {out['prize_walk_in_rate']:.3f}/"
          f"{out['prize_walk_in_attach_rate']:.3f} "
          f"(measured {tally['prize_exposure_measured']}) "
          f"(wr {rate('wins', 'games'):.3f} n={tally['games']}) -> {path.name}",
          flush=True)


# ----------------------------------------------------------------- decode ---

def _decode_cell(paths):
    res = []
    for p in paths:
        for line in open(p):
            d = json.loads(line)
            if "results" in d:
                res += d["results"]
    if not res:
        return None
    return (res.count(0) + 0.5 * res.count(2)) / len(res), len(res)


def decode(seedless_beds: list[str]) -> None:
    import glob as g
    print("=== m38 Phase 0 E0a decode (0 = arm win, draws half) ===")
    cells = {arm: {bed: _decode_cell(sorted(
        g.glob(str(RUNS / f"m38_{arm}_{bed}_s*.jsonl"))))
        for bed in seedless_beds} for arm in ARMS}
    header = "bed        " + "".join(f"{a:>18s}" for a in ARMS) + "   z(arm-A0)"
    print(header)
    for bed in seedless_beds:
        row = f"{bed:9s}"
        for arm in ARMS:
            c = cells[arm][bed]
            row += f"  {c[0]:.4f} (n={c[1]:>4d})" if c else "         (pend)  "
        zs = []
        c0 = cells["A0"][bed]
        for arm in ("A1", "A2", "A3"):
            c = cells[arm][bed]
            if c and c0:
                (pa, na), (pc, nc) = c, c0
                se = math.sqrt(pa * (1 - pa) / na + pc * (1 - pc) / nc)
                zs.append(f"{arm}:{(pa - pc) / se:+.2f}" if se else f"{arm}:?")
        print(row + ("   " + " ".join(zs) if zs else ""))
    for arm in ARMS:
        done = [c for c in cells[arm].values() if c]
        if done:
            w = sum(p * n for p, n in done)
            n = sum(n for _, n in done)
            print(f"pooled {arm}: {w / n:.4f} (n={n:.0f}, "
                  f"{len(done)}/{len(seedless_beds)} beds)")
    print("\nE0b:")
    for f in sorted(RUNS.glob("m38_e0b_*.json")):
        d = json.loads(f.read_text())
        print(f"  {d['arm']} vs {d['bed']}: walk-in {d['walk_in_rate']:.3f} "
              f"(attach {d['walk_in_attach_rate']:.3f}), prize-turn walk-in "
              f"{d['prize_walk_in_rate']:.3f} over {d['prize_turns']} "
              f"prize turns, {d['turn_ends']} turn ends")
    print("\nMDE note (pre-registered): n=800/cell resolves ~±5pp at 95% — "
          "do not over-read null deltas below that. E0c outranks E0a as "
          "evidence for H1.")


# ------------------------------------------------------------------- main ---

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s, arm_required):
        if arm_required:
            s.add_argument("--arm", required=True, choices=sorted(ARMS))
        s.add_argument("--deck", default=DEFAULT_DECK)
        s.add_argument("--workers", type=int, default=8)   # HARD CAP 8 (CLAUDE.md)
        s.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("run", help="one arm's E0a cells")
    common(s, True)
    s.add_argument("-n", "--games", type=int, default=800)
    s.add_argument("--beds", default=",".join(BEDS))

    s = sub.add_parser("all", help="every arm sequentially (subprocess/arm)")
    common(s, False)
    s.add_argument("-n", "--games", type=int, default=800)
    s.add_argument("--beds", default=",".join(BEDS))

    s = sub.add_parser("e0b", help="over-greed walk-in probe, one arm")
    common(s, True)
    s.add_argument("-n", "--games", type=int, default=60)
    s.add_argument("--bed", default="tuned")

    s = sub.add_parser("e0b-all", help="walk-in probe, every arm")
    common(s, False)
    s.add_argument("-n", "--games", type=int, default=60)
    s.add_argument("--bed", default="tuned")

    s = sub.add_parser("decode", help="print the battery table")
    s.add_argument("--beds", default=",".join(BEDS))

    a = p.parse_args()
    if getattr(a, "workers", 8) > 8:
        raise SystemExit("--workers 8 is the proven-stable ceiling (CLAUDE.md)")

    if a.cmd == "run":
        _set_arm_env(a.arm)
        run_e0a(a.arm, a.deck, a.beds.split(","), a.games, a.workers, a.seed)
    elif a.cmd == "e0b":
        _set_arm_env(a.arm)
        run_e0b(a.arm, a.deck, a.bed, a.games)
    elif a.cmd in ("all", "e0b-all"):
        child = "run" if a.cmd == "all" else "e0b"
        _notify(f"[m38] Phase 0 battery `{a.cmd}` starting: arms "
                f"{','.join(ARMS)}, n={a.games}"
                + (f"/bed, beds {a.beds}" if a.cmd == "all" else f" vs {a.bed}"))
        for arm in ARMS:
            cmd = [sys.executable, str(Path(__file__).resolve()), child,
                   "--arm", arm, "--deck", a.deck, "-n", str(a.games),
                   "--workers", str(a.workers), "--seed", str(a.seed)]
            cmd += (["--beds", a.beds] if a.cmd == "all"
                    else ["--bed", a.bed])
            print(f"[m38] === arm {arm} ===", flush=True)
            r = subprocess.run(cmd, cwd=str(ROOT))
            if r.returncode != 0:
                _notify(f"[m38] battery FAILED at arm {arm} "
                        f"(exit {r.returncode}) — stopping")
                raise SystemExit(r.returncode)
        _notify(f"[m38] Phase 0 `{a.cmd}` DONE")
        print("DONE", flush=True)          # battery_heartbeat.sh marker
    elif a.cmd == "decode":
        decode(a.beds.split(","))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
