"""M31 P3: decode the battery and apply the pre-registered bar.

Bars (docs/M31-plan.md, fixed BEFORE the numbers). Both arms are RULE arms —
weights frozen at m28_winners, the claimed gain is behavioral (M26 O1 / M30 gac
precedent) — so both take the NON-INFERIORITY bar vs the LIVE gac incumbent
(54929991):
    mirror z > -1.96 vs 0.6750 AND dragapult z > -1.96 vs 0.3750
    AND kyogre >= 0.95.
PLUS (checked in P2, not here) the behavioural claim relative to the P0.6 gac
baseline: the rule's own primary counter strictly improved AND neither deck-out
(reason 2) nor bench-out (reason 3) loss share degraded.
Tie-break among qualifiers: best primary-counter improvement, then the rocket
advisory. Dragapult is NOT a tie-breaker (M29 trap axis). Grim and rocket are
advisory (regression watch; rocket doubles as the deck-out guard). If no arm
clears: evidence to Piotr, no ship. Decode: 0 = side-a WIN.

Usage: uv run python scripts/m31_decide.py
"""
import glob
import json
import math

PINS = {"luc": (0.6750, 800), "drag": (0.3750, 400), "grim": (0.6687, 400),
        "rocket": (0.5675, 400), "kyo": (0.95, 200)}
GATES = ("luc", "drag", "kyo")          # advisory axes excluded from the bar
RULE_ARMS = ("gacb", "gacd")


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
    for arm in RULE_ARMS:
        print(f"=== {arm} — RULE arm (non-inferiority bar vs LIVE gac)")
        verdicts = {}
        for tag, (pin, pin_n) in PINS.items():
            fs = [f for f in sorted(glob.glob(f"runs/m31_{arm}_{tag}_s*.jsonl"))
                  if decode([f]) is not None]
            r = decode(fs)
            if r is None:
                print(f"  {tag:6s} (pending)")
                verdicts[tag] = None
                continue
            p, n = r
            se = math.sqrt(p * (1 - p) / n + pin * (1 - pin) / pin_n)
            z = (p - pin) / se
            seeds = " ".join(f"{decode([f])[0]:.3f}" for f in fs)
            adv = "" if tag in GATES else "  [advisory]"
            print(f"  {tag:6s} {seeds:26s} pooled {p:.4f} n={n}  pin {pin:.4f}"
                  f"  delta {p - pin:+.4f}  z={z:+.2f}  MDE~{2.8 * se:.3f}{adv}")
            verdicts[tag] = (p, z)
        if any(verdicts[t] is None for t in GATES):
            print("  -> INCOMPLETE\n")
            continue
        mirror_ok = verdicts["luc"][1] > -1.96
        drag_ok = verdicts["drag"][1] > -1.96
        kyo_ok = verdicts["kyo"][0] >= 0.95
        ship = mirror_ok and drag_ok and kyo_ok
        print(f"  bar: mirror non-inferior={mirror_ok}  dragapult non-inferior="
              f"{drag_ok}  kyogre floor={kyo_ok}")
        print(f"  -> {'CLEARS ITS STRENGTH BAR' if ship else 'DOES NOT CLEAR'}"
              "\n")
    print("Reminder: the behavioural claim (P2 primary-counter improvement +"
          " deck-out/bench-out non-degradation vs the gac baseline), pre-ship"
          " human QC, and Piotr's explicit go are still required before submit.")


if __name__ == "__main__":
    main()
