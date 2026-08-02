"""M39 P3 screen decoder — kill filter only, never a ranking.

Reads runs/m39_scr_<arm>_<bed>_s1.jsonl against the SAME-BATTERY control
cells from the panel gate (runs/m39_pan_conserve_<bed>_s*.jsonl). At n=400 a
cell resolves ~10pp (G-12), so the only verdict this script is entitled to
emit is KILL — an arm that has collapsed by tens of points on beds where the
control holds. Everything else says "promote to the panel gate".

Usage: uv run python scripts/m39_screen_decode.py vsloss bestresp ...
"""
import glob
import json
import math
import sys

BEDS = ("tuned", "m28", "wall_d1", "grim_d1")
KILL_PP = 0.10        # a >=10pp mean drop is outside what n=400 can be noise


def decode(paths):
    res = []
    for p in paths:
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                if "results" in d:
                    res += d["results"]
    if not res:
        return None
    return (res.count(0) + 0.5 * res.count(2)) / len(res), len(res)


def main() -> int:
    arms = sys.argv[1:]
    ctl = {b: decode(sorted(glob.glob(f"runs/m39_pan_conserve_{b}_s*.jsonl")))
           for b in BEDS}
    head = f"{'arm':<14}" + "".join(f"{b:>12}" for b in BEDS) + f"{'mean d':>9}"
    print("control      " + "".join(f"{ctl[b][0]:>12.3f}" if ctl[b] else
                                    f"{'-':>12}" for b in BEDS))
    print(head)
    print("-" * len(head))
    for arm in arms:
        deltas, row = [], f"{arm:<14}"
        for b in BEDS:
            cell = decode(sorted(glob.glob(f"runs/m39_scr_{arm}_{b}_s1.jsonl")))
            if cell is None or ctl[b] is None:
                row += f"{'(pending)':>12}"
                continue
            d = cell[0] - ctl[b][0]
            deltas.append(d)
            row += f"{cell[0]:>8.3f}{d:>+6.3f}"[-12:]
        mean = sum(deltas) / len(deltas) if deltas else float("nan")
        row += f"{mean:>+9.3f}"
        if deltas and mean <= -KILL_PP:
            row += "   <- KILL"
        print(row)
    print(f"\nKILL bar: mean delta <= -{KILL_PP * 100:.0f}pp across the four "
          f"screen beds. No arm is PROMOTED by this table — n=400 resolves "
          f"~10pp (G-12), so a positive screen number means 'not obviously "
          f"broken', nothing more. The panel gate ranks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
