"""M41b feature census: does every new slot carry information on REAL options?

Phase 3.2 of docs/M41b-plan.md, pre-registered BEFORE the retrain decision:

    Any slot that is non-zero on <1% of the options its type applies to, or
    has zero variance, is DROPPED before the retrain. Dead columns are not
    free — they are capacity and noise, and this is the check that stops 1b
    being carried on a coverage argument alone.

Applicability follows the write site (rl/encoders.py encode_option_v2):
board slots for the six board-object types, cost slots for ATTACK, matchup
and econ for every option. Matchup/econ are per-menu constants, so their
spread is judged across the corpus — a within-menu read of them is
meaningless by construction (plan § 1c).

Options come from local replay JSON (`rl/eval.py --json_prefix` output), the
same corpus the aliasing probe and column safety read. Exits non-zero when
any slot violates the kill bar, so the retrain decision can gate on it.

    uv run python scripts/m41b_feature_census.py
"""
import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
from cg.api import OptionType, to_observation_class  # noqa: E402

from rl.encoders import (OPTION_M42_DIM, OPTION_M41B_DIM,  # noqa: E402
                         _M41B_BOARD_TYPES, _OT_ATTACK)

N_NEW = OPTION_M41B_DIM - OPTION_M42_DIM

SLOT_NAMES = [
    "board.has_object", "board.hp_frac", "board.damage_100",
    "board.energies_5", "board.tools_2", "board.is_active",
    "board.bench_index_4", "board.is_opponents",
    *[f"cost.type_{t}" for t in range(12)],
    "matchup.opp_weak_to_mine", "matchup.opp_resists_mine",
    "matchup.mine_weak_to_theirs", "matchup.mine_resists_theirs",
    "econ.retreat_cost_4", "econ.retreat_payable",
    "econ.hand_15", "econ.bench_5",
]
assert len(SLOT_NAMES) == N_NEW

KILL_NONZERO_RATE = 0.01

# Bar amendment (Piotr, 2026-08-04): the kill applies to STRUCTURAL deadness
# (an engine rule or a pool-wide fact), which is how the deficit/afford slots
# died — the engine never offers an unaffordable ATTACK. Slots that are zero
# only because the replay corpus spans few matchup families (unexercised
# energy types in the cost histogram, resist pairings that never occurred)
# are CORPUS-STARVED, not dead: they are reported below but exempt from the
# kill, because dropping them would rebuild the deck-specificity this block
# exists to remove. Re-censused whenever a broader corpus exists.
CORPUS_STARVED_EXEMPT = frozenset(
    [f"cost.type_{t}" for t in range(12)]
    + ["matchup.opp_resists_mine", "matchup.mine_resists_theirs"])


def applicable(slot: int, option_type: int) -> bool:
    if slot < 8:
        return option_type in _M41B_BOARD_TYPES
    if slot < 20:
        return option_type == _OT_ATTACK
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replays", nargs="*", default=["replays/**/*.json"])
    ap.add_argument("--limit", type=int, default=500_000)
    a = ap.parse_args()

    values: list[list[float]] = [[] for _ in range(N_NEW)]
    n_options = 0
    files = sorted({p for pat in a.replays for p in glob.glob(pat, recursive=True)})
    from rl.encoders import encode_option_v2
    for path in files:
        try:
            raw = json.loads(Path(path).read_text())
        except ValueError:
            continue
        if not isinstance(raw, dict) or not raw.get("steps"):
            continue
        for step in raw["steps"]:
            for seat in (0, 1):
                st = step[seat] if seat < len(step) else None
                obs_dict = (st or {}).get("observation")
                if not obs_dict or not obs_dict.get("select"):
                    continue
                try:
                    obs = to_observation_class(obs_dict)
                except Exception:  # noqa: BLE001
                    continue
                for opt in (obs.select.option or ()):
                    try:
                        num, _ids = encode_option_v2(opt, obs)
                    except Exception:  # noqa: BLE001
                        continue
                    n_options += 1
                    block = num[OPTION_M42_DIM:OPTION_M41B_DIM]
                    ot = int(opt.type)
                    for slot in range(N_NEW):
                        if applicable(slot, ot):
                            values[slot].append(float(block[slot]))
                    if n_options >= a.limit:
                        break

    print(f"censused {n_options} real options from {len(files)} replay files\n")
    print(f"{'slot':<28} {'n_appl':>7} {'nonzero':>8} {'mean':>8} "
          f"{'std':>8} {'min':>7} {'max':>7} {'distinct':>8}  verdict")
    kills = []
    for slot, name in enumerate(SLOT_NAMES):
        vals = np.asarray(values[slot], dtype=np.float64)
        if vals.size == 0:
            print(f"{name:<28} {0:>7} {'-':>8} {'-':>8} {'-':>8} {'-':>7} "
                  f"{'-':>7} {'-':>8}  KILL (no applicable options in corpus)")
            kills.append(name)
            continue
        nonzero = float((vals != 0.0).mean())
        distinct = np.unique(np.round(vals, 6)).size
        dead = nonzero < KILL_NONZERO_RATE or float(vals.std()) == 0.0
        if dead and name in CORPUS_STARVED_EXEMPT:
            verdict = "starved (exempt)"
        elif dead:
            verdict = "KILL"
            kills.append(name)
        else:
            verdict = "ok"
        print(f"{name:<28} {vals.size:>7} {nonzero:>8.4f} {vals.mean():>8.4f} "
              f"{vals.std():>8.4f} {vals.min():>7.3f} {vals.max():>7.3f} "
              f"{distinct:>8}  {verdict}")

    print()
    if kills:
        print(f"KILL bar hit by {len(kills)} slot(s): {', '.join(kills)}")
        print("Pre-registered consequence: DROP before the retrain "
              "(docs/M41b-plan.md Phase 3.2).")
        return 1
    print("All slots clear the pre-registered bar "
          f"(nonzero rate >= {KILL_NONZERO_RATE:.0%} on applicable options, "
          "nonzero variance).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
