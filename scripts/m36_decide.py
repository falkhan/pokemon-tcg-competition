"""M36: decode the battery and apply the pre-registered bars.

Bars fixed BEFORE the numbers (docs/M36-plan.md execution log, 2026-07-27):

P1 `gacfv` (RULE arm, gustveto) vs the SAME-BATTERY gacf control, 2-prop z:
    non-inferiority z > -1.96 on EVERY bed (pooled n=400 vs n=400)
    AND kyogre absolute >= 0.95.
  No offline strength claim required — the trigger is too rare (engagement
  probe + live A/B carry that); any bed regression kills (composition law).

P2 `h4` (DECK arm, +1 Enhanced Hammer / -1 Hilda on gacf rules) vs gacf+v2:
    wall bed strictly better z > +1.96 (the whole point of the tech — the E4
    unlock mechanism predicts a real move, not a wash)
    AND non-inferiority z > -1.96 on the other 6 beds (rocket/grim watch:
    M33 V1 died on disruption cuts; here hammer goes UP, Hilda goes down)
    AND kyogre absolute >= 0.95.

If no arm clears: evidence to Piotr, no ship. Ship shape stays
single-variable vs 55011605 (rules-only OR deck-only, never both).
Decode law: 0 = side-a WIN, 2 = draw counts half.

Usage: uv run python scripts/m36_decide.py
"""
import glob
import json
import math

BEDS = ("wall", "rocket", "grim", "luc", "mirror", "arch", "kyo")
KYO_FLOOR = 0.95


def decode(paths):
    res = []
    for p in paths:
        for line in open(p):
            d = json.loads(line)
            if "results" in d:
                res += d["results"]
    if not res:
        return None
    w, dr = res.count(0), res.count(2)
    return (w + 0.5 * dr) / len(res), len(res)


def pooled(arm, bed):
    fs = sorted(glob.glob(f"runs/m36_{arm}_{bed}_s*.jsonl"))
    per_seed = [(f, decode([f])) for f in fs]
    tot = decode(fs)
    return tot, per_seed


def compare(arm, control="gacf", strict_beds=()):
    print(f"=== {arm} vs {control} (per-bed 2-prop z; "
          f"strict>{'+1.96' if strict_beds else ''} on {strict_beds}) ===")
    ok = True
    for bed in BEDS:
        (a, seeds_a), (c, _) = pooled(arm, bed), pooled(control, bed)
        if a is None or c is None:
            print(f"  {bed:7s} (pending)")
            ok = False
            continue
        (pa, na), (pc, nc) = a, c
        se = math.sqrt(pa * (1 - pa) / na + pc * (1 - pc) / nc)
        z = (pa - pc) / se if se else 0.0
        seeds = " ".join(f"{r[0]:.3f}" for _, r in seeds_a if r)
        bar = "STRICT" if bed in strict_beds else "noninf"
        if bed in strict_beds:
            verdict = "PASS" if z > 1.96 else "FAIL"
        else:
            verdict = "PASS" if z > -1.96 else "KILL"
        if bed == "kyo" and pa < KYO_FLOOR:
            verdict = "KILL(floor)"
        ok &= verdict.startswith("PASS")
        print(f"  {bed:7s} {seeds:14s} pooled {pa:.4f} n={na} vs {pc:.4f} "
              f"n={nc}  delta {pa - pc:+.4f}  z={z:+.2f}  [{bar}] {verdict}")
    print(f"  => arm {arm}: {'CLEARS the pre-registered bar' if ok else 'DOES NOT CLEAR (or pending)'}")
    return ok


if __name__ == "__main__":
    compare("gacfv")
    print()
    compare("h4", strict_beds=("wall",))
