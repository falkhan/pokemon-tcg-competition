"""M37: decode the racemode battery and apply the pre-registered bars.

Bars fixed BEFORE any battery number (docs/M37-plan.md, 2026-07-28; Piotr's
five kickoff calls recorded there — racemode arm, grim ids IN the trigger):

B1 SHIP bar (per arm, vs the SAME-BATTERY gacf control, 2-prop z):
    pooled-over-TRIGGER-beds wr(arm) >= wr(control) + 0.03
    AND no single trigger bed regresses by more than 0.05
    AND non-inferiority z > -1.96 on every trigger bed individually.
  Trigger beds = beds whose opponent deck contains a _RACEMODE_OPP_IDS
  Pokémon (hop, wall, grim clone, garchomp clone as available). Non-trigger
  beds are NOT run per-arm: O12's predicate is provably inert without a
  trigger id on the opponent board — B4 verifies that claim instead.
B2 VARIANT pick: the higher pooled-trigger-bed wr of {gacfr, gacfrr}; if
    within 1 pooled SE, gacfr (margin-gated — probe-backed default, the
    blanket variant's setup-dig cost was never measured off-mirror).
B3 GRIM rule: if the grim-clone bed shows wr(winner) < wr(control) - 0.05,
    drop _RACEMODE_GRIM_IDS (one-line edit + twin re-export) and rerun
    grim + hop confirm at n=400. If the grim bed is unavailable at decide
    time, ship WITH grim ids (Piotr's call) and flag the grim cell as the
    first M38 watch item.
B4 INERTNESS (AMENDED 2026-07-28 BEFORE any battery decode, evidence in
    docs/M37-plan.md): matchrunner at workers=8 is NOT run-reproducible even
    for an identical config (engine RNG is per worker process; mp.Pool
    assigns chunks by timing — gacfr mirror gave 55-45 then 51-49 at the
    same seed; a gacf rerun happened to reproduce), so "identical W/L" is
    the wrong instrument. AUTHORITATIVE check = scripts/racemode_fire_probe
    --bed mirror (single-process): trigger_true == 0 AND o12_fires == 0.
    Measured: 0/0 over 1313 MAIN prompts, 30 games. The workers-8 smoke
    stays as a sanity band only: |wr(gacfr) - wr(gacf)| <= 0.14 (~2 SE at
    n=100).
B5 BAND gate: band_decode composite(arm) >= composite(same-battery control)
    - 1 SE over the SAME bed set (paired; never compare composites across
    different bed sets).
B6 QC: scripts/qc_battery.py --prev <M36 tarball> + Piotr's replay review +
    explicit go. Not automated here.
B7 AWR done-bar: smoke train completes, non-win row fraction 30-50%; NO
    ship decision from AWR in M37.

Decode law: 0 = side-a WIN, 2 = draw counts half.
Usage: uv run python scripts/m37_decide.py
"""
import glob
import json
import math

TRIGGER_BEDS = ("hop", "wall", "grim", "garchomp")
SHIP_DELTA = 0.03          # B1: pooled trigger-bed improvement required
MAX_BED_REGRESSION = 0.05  # B1: worst single trigger bed allowed


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
    fs = sorted(glob.glob(f"runs/m37_{arm}_{bed}_s*.jsonl"))
    per_seed = [(f, decode([f])) for f in fs]
    tot = decode(fs)
    return tot, per_seed


def bed_rows(arm, control="gacf"):
    rows = {}
    for bed in TRIGGER_BEDS:
        (a, seeds_a), (c, _) = pooled(arm, bed), pooled(control, bed)
        rows[bed] = (a, c, seeds_a)
    return rows


def b1(arm, control="gacf"):
    print(f"=== B1 ship bar: {arm} vs {control} ===")
    rows = bed_rows(arm, control)
    aw = an = cw = cn = 0.0
    ok, have = True, False
    for bed, (a, c, seeds_a) in rows.items():
        if a is None or c is None:
            print(f"  {bed:9s} (pending or bed unavailable)")
            continue
        have = True
        (pa, na), (pc, nc) = a, c
        aw += pa * na
        an += na
        cw += pc * nc
        cn += nc
        se = math.sqrt(pa * (1 - pa) / na + pc * (1 - pc) / nc)
        z = (pa - pc) / se if se else 0.0
        seeds = " ".join(f"{r[0]:.3f}" for _, r in seeds_a if r)
        verdict = "PASS" if z > -1.96 else "KILL(noninf)"
        if pa - pc < -MAX_BED_REGRESSION:
            verdict = "KILL(regression)"
        ok &= verdict == "PASS"
        print(f"  {bed:9s} {seeds:20s} pooled {pa:.4f} n={na:.0f} vs "
              f"{pc:.4f} n={nc:.0f}  delta {pa - pc:+.4f}  z={z:+.2f}  "
              f"{verdict}")
    if not have:
        print("  => no decided beds yet")
        return False, None
    pa, pc = aw / an, cw / cn
    se = math.sqrt(pa * (1 - pa) / an + pc * (1 - pc) / cn)
    hit = pa - pc >= SHIP_DELTA
    ok &= hit
    print(f"  pooled trigger beds: {pa:.4f} (n={an:.0f}) vs {pc:.4f} "
          f"(n={cn:.0f})  delta {pa - pc:+.4f} (need >= +{SHIP_DELTA})  "
          f"{'PASS' if hit else 'FAIL'}")
    print(f"  => arm {arm}: "
          f"{'CLEARS B1' if ok else 'DOES NOT CLEAR B1 (or pending)'}")
    return ok, (pa, se)


def b2():
    print("=== B2 variant pick ===")
    res = {}
    for arm in ("gacfr", "gacfrr"):
        okd, stat = b1(arm)
        res[arm] = (okd, stat)
        print()
    a, b = res["gacfr"][1], res["gacfrr"][1]
    if a and b:
        pick = "gacfr" if a[0] >= b[0] - max(a[1], b[1]) else "gacfrr"
        print(f"B2: gacfr {a[0]:.4f} vs gacfrr {b[0]:.4f} (1 SE "
              f"{max(a[1], b[1]):.4f}) -> pick {pick}")
    else:
        print("B2: pending")


def b3(winner="gacfr", control="gacf"):
    print("=== B3 grim rule ===")
    (a, _), (c, _) = pooled(winner, "grim"), pooled(control, "grim")
    if a is None or c is None:
        print("  grim bed unavailable -> ship WITH grim ids (pre-registered "
              "fallback), flag grim cell for M38")
        return
    delta = a[0] - c[0]
    if delta < -MAX_BED_REGRESSION:
        print(f"  grim delta {delta:+.4f} < -{MAX_BED_REGRESSION}: DROP "
              f"_RACEMODE_GRIM_IDS, re-export twin, rerun grim+hop n=400")
    else:
        print(f"  grim delta {delta:+.4f}: grim ids STAY")


def b4():
    print("=== B4 inertness (amended: probe is authoritative) ===")
    print("  authoritative: racemode_fire_probe --bed mirror -> "
          "trigger_true 0 / o12_fires 0 over 1313 prompts (PASS, "
          "2026-07-28)")
    a = decode(sorted(glob.glob("runs/m37_inert_gacfr_mirror_s*.jsonl")))
    c = decode(sorted(glob.glob("runs/m37_inert_gacf_mirror_s*.jsonl")))
    if a is None or c is None:
        print("  sanity band: pending")
        return
    ok = abs(a[0] - c[0]) <= 0.14
    print(f"  sanity band: gacfr {a[0]:.4f} n={a[1]} vs gacf {c[0]:.4f} "
          f"n={c[1]} |delta| {abs(a[0] - c[0]):.4f} <= 0.14 -> "
          f"{'PASS' if ok else 'FAIL — investigate before decode'}")


if __name__ == "__main__":
    b2()
    print()
    b3()
    print()
    b4()
