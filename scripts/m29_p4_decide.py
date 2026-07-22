"""M29 P4: decode the ship-config battery and apply the pre-registered bar.

Bar (docs/M29-plan.md, fixed BEFORE the numbers): a candidate is shippable iff
  1. mirror (rule:lucario) resolved-better than the live pin 0.671 (z > 1.96)
  2. dragapult non-inferior: delta > -MDE (not resolved-worse)
  3. kyogre floor >= 0.95
Grim is advisory (regression watch only). Decode: 0 = side-a WIN.

Usage: uv run python scripts/m29_p4_decide.py
"""
import glob
import json
import math

PINS = {"luc": (0.671, 800), "drag": (0.3375, 400),
        "grim": (0.6875, 400), "kyo": (0.975, 200)}
ARMS = ("m28_winners", "m29_pooled_winners")


def decode(paths):
    res = []
    for p in paths:
        for line in open(p):
            d = json.loads(line)
            if "results" in d:
                res += d["results"]
    if not res:
        return None
    w, _, dr = res.count(0), res.count(1), res.count(2)
    return (w + 0.5 * dr) / len(res), len(res)


def main() -> None:
    for arm in ARMS:
        print(f"=== {arm} (ship config: modelt, O1 on)")
        verdicts = {}
        for tag, (pin, pin_n) in PINS.items():
            fs = [f for f in sorted(glob.glob(f"runs/m29_{arm}_{tag}_s*.jsonl"))
                  if decode([f]) is not None]   # skip header-only checkpoints
            r = decode(fs)
            if r is None:
                print(f"  {tag:5s} (pending)")
                verdicts[tag] = None
                continue
            p, n = r
            se = math.sqrt(p * (1 - p) / n + pin * (1 - pin) / pin_n)
            z = (p - pin) / se
            seeds = " ".join(f"{decode([f])[0]:.3f}" for f in fs)
            print(f"  {tag:5s} {seeds:26s} pooled {p:.4f} n={n}  pin {pin:.4f}"
                  f"  delta {p - pin:+.4f}  z={z:+.2f}  MDE~{2.8 * se:.3f}")
            verdicts[tag] = (p, z)
        if any(v is None for v in verdicts.values()):
            print("  -> INCOMPLETE\n")
            continue
        mirror_ok = verdicts["luc"][1] > 1.96
        drag_ok = verdicts["drag"][1] > -1.96
        kyo_ok = verdicts["kyo"][0] >= 0.95
        ship = mirror_ok and drag_ok and kyo_ok
        print(f"  bar: mirror resolved-better={mirror_ok}  "
              f"dragapult non-inferior={drag_ok}  kyogre floor={kyo_ok}")
        print(f"  -> {'SHIPPABLE' if ship else 'DOES NOT CLEAR THE BAR'}\n")


if __name__ == "__main__":
    main()
