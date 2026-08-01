"""M38 E0c — label-shift differ across the Phase 0 teacher arms (A0-A3).

THE PRIMARY H1 EVIDENCE (docs/M38-plan.md): the student distills per-decision
labels, not the teacher's winrate under its own compute budget, so E0c
outranks E0a. Runs a small plan_iter expert collect per arm (fresh dirs —
collect has NO resume; value_solve is OFF by design, decision 1) and diffs
the label distributions:

  - kill-plan commit rate      stats["kill_plans"] / plan_rows
  - labels/game density        guard vs the M8.1 dev-tier trap (a permissive
                               bar flooding the corpus with marginal lines)
  - attack-when-available      chosen option is ATTACK where one exists
  - end-when-attack-available  turn passed while an attack existed
  - supporter-play rate        chosen where a supporter PLAY exists (the
                               chronic under-play line: 10.2%/prompt)

Arm env flags are set from --arm BEFORE any rl import (spawn workers
re-import modules but inherit the environment — the M22_WHOLE_BOARD idiom).

Usage
-----
  uv run python scripts/m38_label_shift.py collect --arm A1 [-n 200]
      [--workers 8] [--decks data/league/population.json] [--seed 0]
  uv run python scripts/m38_label_shift.py all [-n 200] ...
  uv run python scripts/m38_label_shift.py diff

Output: data/m38_e0c/<arm>/shard_*.npz + stats.json per arm.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_BASE = ROOT / "data" / "m38_e0c"

ARMS = {
    "A0": {"M38_OLD_LEAF": "1", "M38_BAR": "current"},
    "A1": {"M38_OLD_LEAF": "0", "M38_BAR": "current"},
    "A2": {"M38_OLD_LEAF": "0", "M38_BAR": "const"},
    "A3": {"M38_OLD_LEAF": "0", "M38_BAR": "semantic"},
}

# rl/encoders.py option-type one-hot slots (num[int(opt.type)] = 1).
_OT_PLAY, _OT_ATTACK, _OT_END = 7, 13, 14


def _set_arm_env(arm: str) -> None:
    assert "rl.turn_solver" not in sys.modules, \
        "rl.turn_solver imported before the arm env was set"
    os.environ.update(ARMS[arm])


def run_collect(arm: str, games: int, decks: str, workers: int,
                seed: int) -> None:
    out_dir = OUT_BASE / arm
    if out_dir.exists() and any(out_dir.glob("*.npz")):
        raise SystemExit(f"{out_dir} already holds shards — collect has NO "
                         "resume and would clobber them; move or delete the "
                         "dir first (CLAUDE.md), or diff what is there.")
    from rl.plan_iter import collect
    print(f"[m38 E0c] arm {arm} env={ARMS[arm]} n={games} workers={workers}",
          flush=True)
    stats = collect("expert", games, decks, out_dir, workers=workers,
                    seed=seed, value_ckpt=None)   # value_solve OFF (decision 1)
    stats["arm"] = arm
    stats["env"] = ARMS[arm]
    stats["n_games"] = games
    stats["seed"] = seed
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[m38 E0c] {arm}: {stats['games']} games, "
          f"{stats['decisions']} labels, kill_plans {stats['kill_plans']} "
          f"-> {out_dir}", flush=True)


def _shard_rates(out_dir: Path, supporter_ids: frozenset) -> dict:
    import numpy as np
    n = {"rows": 0, "attack_avail": 0, "attack_taken": 0, "end_on_attack": 0,
         "supp_avail": 0, "supp_taken": 0}
    for path in sorted(out_dir.glob("shard_*.npz")):
        s = np.load(path)
        opts, ids = s["options"], s["option_ids"]
        off = 0
        for n_opt, label in zip(s["n_options"], s["labels"]):
            row_opts = opts[off:off + n_opt]
            row_ids = ids[off:off + n_opt]
            off += n_opt
            n["rows"] += 1
            chosen = row_opts[label]
            if (row_opts[:, _OT_ATTACK] == 1).any():
                n["attack_avail"] += 1
                n["attack_taken"] += chosen[_OT_ATTACK] == 1
                n["end_on_attack"] += chosen[_OT_END] == 1
            supp = (row_opts[:, _OT_PLAY] == 1) & np.isin(
                row_ids[:, 0], list(supporter_ids))
            if supp.any():
                n["supp_avail"] += 1
                n["supp_taken"] += bool(
                    chosen[_OT_PLAY] == 1 and row_ids[label, 0]
                    in supporter_ids)
    return n


def diff() -> None:
    from cg.api import CardType, all_card_data
    supporter_ids = frozenset(c.cardId for c in all_card_data()
                              if c.cardType == CardType.SUPPORTER)
    print("=== m38 E0c label shift (expert collect, value_solve OFF) ===")
    print(f"{'metric':28s}" + "".join(f"{a:>10s}" for a in ARMS))
    rows = {}
    for arm in ARMS:
        out_dir = OUT_BASE / arm
        sj = out_dir / "stats.json"
        if not sj.exists():
            rows[arm] = None
            continue
        st = json.loads(sj.read_text())
        sr = _shard_rates(out_dir, supporter_ids)
        rows[arm] = {
            "games": st["games"],
            "labels/game": st["decisions"] / max(1, st["games"]),
            "kill-plan commit rate": st["kill_plans"] / max(1, st["plan_rows"]),
            "plan-null rate": st["plan_null"] / max(1, st["plan_rows"]),
            "derails/game": st["derails"] / max(1, st["games"]),
            "attack-when-available": sr["attack_taken"] / max(1, sr["attack_avail"]),
            "end-when-attack-avail": sr["end_on_attack"] / max(1, sr["attack_avail"]),
            "supporter-play rate": sr["supp_taken"] / max(1, sr["supp_avail"]),
            "supporter prompts": sr["supp_avail"],
        }
    metrics = next(v for v in rows.values() if v)
    for m in metrics:
        line = f"{m:28s}"
        for arm in ARMS:
            v = rows[arm][m] if rows[arm] else None
            line += ("    (pend)" if v is None
                     else f"{v:>10.0f}" if isinstance(v, int) or m in
                     ("games", "supporter prompts") else f"{v:>10.3f}")
        print(line)
    print("\nRead with docs/M38-plan.md E0c: H1 expects kill-plan commit and "
          "attack-when-available UP vs A0, end-when-attack-available DOWN; "
          "watch labels/game for the M8.1 flood trap on permissive bars, "
          "and the R5 early-game profile (setup plans are off).")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("collect", help="one arm's E0c collect")
    s.add_argument("--arm", required=True, choices=sorted(ARMS))
    s.add_argument("-n", "--games", type=int, default=200)
    s.add_argument("--decks", default="data/league/population.json")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("all", help="every arm sequentially (subprocess/arm)")
    s.add_argument("-n", "--games", type=int, default=200)
    s.add_argument("--decks", default="data/league/population.json")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--seed", type=int, default=0)

    sub.add_parser("diff", help="print the label-shift table")

    a = p.parse_args()
    if getattr(a, "workers", 8) > 8:
        raise SystemExit("--workers 8 is the proven-stable ceiling (CLAUDE.md)")

    if a.cmd == "collect":
        _set_arm_env(a.arm)
        run_collect(a.arm, a.games, a.decks, a.workers, a.seed)
    elif a.cmd == "all":
        for arm in ARMS:
            print(f"[m38 E0c] === arm {arm} ===", flush=True)
            r = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "collect",
                 "--arm", arm, "-n", str(a.games), "--decks", a.decks,
                 "--workers", str(a.workers), "--seed", str(a.seed)],
                cwd=str(ROOT))
            if r.returncode != 0:
                raise SystemExit(r.returncode)
        print("DONE", flush=True)
    elif a.cmd == "diff":
        diff()


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
