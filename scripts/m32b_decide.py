"""M32-B decode: ppo (ppo_current_m32b) vs base (m28_winners), same deck, per
bed, against the PRE-REGISTERED kill-gate (docs/M32.md, fixed BEFORE the run).

This is a TWO-ARM comparison — two-proportion z on (ppo - base), not a pinned
value. Decode: 0 = side-a WIN, 1 = loss, 2 = draw; WR = (w + 0.5*draw)/n.

Pre-registered gate:
  PRIMARY (transfer):  arch (generic:archaludon, SEALED). ppo must beat base
        resolved, z > +1.96. If flat/down (z <= 0) => KILL: the M23 no-transfer
        pattern reproduced; targeted-exogenous PPO overfit the trained pool =>
        pivot to the dragapult archetype (A).
  GUARD (non-inferior): luc (rule:lucario, out-of-loop). ppo >= base - MDE
        (z_luc > -1.96); no catastrophic forgetting of a matchup we win.
  SANITY (trained took): rocket AND grim ppo >= base (expect positive; both in
        pool). mirror reported alongside (trained; base-as-A ~0.5 symmetry).
  FLOOR: kyo (random:kyogre) ppo >= 0.90.
Promising iff PRIMARY passes AND GUARD holds. Then QC + Piotr go. Else KILL.

Usage: uv run python scripts/m32b_decide.py
"""
import glob
import json
import math

# bed -> (label, role). roles: primary | guard | sanity | floor
BEDS = {
    "arch":   ("generic:archaludon  (SEALED transfer)", "primary"),
    "luc":    ("rule:lucario         (non-inf guard)",  "guard"),
    "rocket": ("rocket clone         (trained sanity)", "sanity"),
    "grim":   ("grim clone           (trained sanity)", "sanity"),
    "mirror": ("m28 mirror           (trained)",        "sanity"),
    "kyo":    ("random:kyogre        (floor)",          "floor"),
}


def wr(paths):
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


def arm_bed(arm, bed):
    fs = sorted(glob.glob(f"runs/m32b_{arm}_{bed}_s*.jsonl"))
    fs = [f for f in fs if wr([f]) is not None]
    return wr(fs), fs


def main() -> None:
    print("=== M32-B: ppo (ppo_current_m32b) vs base (m28_winners), "
          "clone54618168 ===\n")
    verdict = {}
    for bed, (label, role) in BEDS.items():
        rb, fb = arm_bed("base", bed)
        rp, fp = arm_bed("ppo", bed)
        if rb is None or rp is None:
            print(f"  {label:34s} (pending  base={rb is not None} "
                  f"ppo={rp is not None})")
            verdict[bed] = None
            continue
        pb, nb = rb
        pp, np_ = rp
        se = math.sqrt(pb * (1 - pb) / nb + pp * (1 - pp) / np_)
        z = (pp - pb) / se if se > 0 else 0.0
        mde = 1.96 * se
        print(f"  {label:34s} base {pb:.4f} (n={nb})  ppo {pp:.4f} (n={np_})  "
              f"delta {pp - pb:+.4f}  z={z:+.2f}  MDE~{mde:.3f}")
        verdict[bed] = (pb, pp, z)

    print()
    if any(verdict[b] is None for b in ("arch", "luc")):
        print("-> INCOMPLETE: run the `primary` phase (arch + luc) first.")
        return
    _, _, z_arch = verdict["arch"]
    _, _, z_luc = verdict["luc"]
    primary_pass = z_arch > 1.96
    primary_kill = z_arch <= 0.0
    guard_ok = z_luc > -1.96
    print(f"  PRIMARY  archaludon transfer: z={z_arch:+.2f}  "
          f"{'PASS (>+1.96)' if primary_pass else ('KILL (<=0)' if primary_kill else 'INCONCLUSIVE (0..+1.96)')}")
    print(f"  GUARD    lucario non-inferior: z={z_luc:+.2f}  "
          f"{'OK' if guard_ok else 'BREACH (<=-1.96)'}")
    for bed in ("rocket", "grim", "mirror", "kyo"):
        if verdict.get(bed):
            pb, pp, z = verdict[bed]
            tag = "floor>=0.90" if bed == "kyo" else "trained>=base"
            ok = (pp >= 0.90) if bed == "kyo" else (pp >= pb)
            print(f"  SANITY   {bed:6s}: {tag} -> {'ok' if ok else 'CHECK'} "
                  f"(base {pb:.3f} ppo {pp:.3f})")
    print()
    if primary_pass and guard_ok:
        print("=> PROMISING: transfer resolved AND guard held. Proceed to QC + "
              "Piotr go. (still need trained-sanity positive + floor.)")
    elif primary_kill:
        print("=> KILL: no out-of-loop transfer (M23 pattern reproduced). "
              "Pivot to the dragapult archetype (option A).")
    else:
        print("=> INCONCLUSIVE on the primary gate (0 < z <= +1.96): extend "
              "seeds or treat as no-transfer per the pre-registered rule.")


if __name__ == "__main__":
    main()
