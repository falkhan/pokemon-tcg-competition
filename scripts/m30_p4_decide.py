"""M30 P4: decode the battery and apply the pre-registered SPLIT bars.

Bars (docs/M30-plan.md, fixed BEFORE the numbers):
- RULE arms (b1ga, b1tempo, b2ga) — same weights as a live-validated base,
  the claimed gain is behavioral (M26 O1 precedent):
    mirror NON-INFERIOR (z > -1.96 vs 0.671) AND dragapult non-inferior
    (z > -1.96 vs 0.3375) AND kyogre >= 0.95.
- ARM D (darm, weights change — the class that produced the M29 regression):
    mirror RESOLVED-BETTER (z > +1.96) AND dragapult non-inferior AND
    kyogre >= 0.95. Fails => evidence for Piotr only, never a ship candidate.
- Tie-break: a qualifying rule arm beats arm D, always. Among rule arms:
  best rocket-advisory delta, then grim. Dragapult is NOT a tie-breaker.
Grim and rocket are advisory (regression watch). Decode: 0 = side-a WIN.

Usage: uv run python scripts/m30_p4_decide.py
"""
import glob
import json
import math

PINS = {"luc": (0.671, 800), "drag": (0.3375, 400), "grim": (0.6875, 400),
        "rocket": (0.458, 120), "kyo": (0.975, 200)}
GATES = ("luc", "drag", "kyo")          # advisory axes excluded from the bar
RULE_ARMS = ("b1ga", "b1tempo", "b2ga", "gac")   # gac: P5.1 conserve rev
WEIGHT_ARMS = ("darm",)


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
    for arm in RULE_ARMS + WEIGHT_ARMS:
        kind = "RULE arm (non-inferiority bar)" if arm in RULE_ARMS \
            else "WEIGHTS arm (resolved-better bar)"
        print(f"=== {arm} — {kind}")
        verdicts = {}
        for tag, (pin, pin_n) in PINS.items():
            fs = [f for f in sorted(glob.glob(f"runs/m30_{arm}_{tag}_s*.jsonl"))
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
        if arm in RULE_ARMS:
            mirror_ok = verdicts["luc"][1] > -1.96
            mirror_lbl = "mirror non-inferior"
        else:
            mirror_ok = verdicts["luc"][1] > 1.96
            mirror_lbl = "mirror RESOLVED-better"
        drag_ok = verdicts["drag"][1] > -1.96
        kyo_ok = verdicts["kyo"][0] >= 0.95
        ship = mirror_ok and drag_ok and kyo_ok
        print(f"  bar: {mirror_lbl}={mirror_ok}  dragapult non-inferior="
              f"{drag_ok}  kyogre floor={kyo_ok}")
        print(f"  -> {'CLEARS ITS BAR' if ship else 'DOES NOT CLEAR ITS BAR'}"
              "\n")
    print("Reminder: behavioral claim (P3-relative END-item / deck-out /"
          " telepath reads) + pre-ship human QC + Piotr's explicit go are"
          " still required before any submit.")


if __name__ == "__main__":
    main()
