"""Stage B0: how often does the composite actually DISAGREE with the net?

docs/M41b-plan.md § II.2. BACKLOG #10 says to collect a corpus whose labels
are the composite's actions and fine-tune on it. But `wrap_with_solver` runs
the inner net on EVERY prompt and replaces its answer only when a trigger
fires AND the tier's bar clears — so on every other prompt the composite's
action IS the bare net's. A corpus labelled that way agrees with the student's
own argmax almost everywhere, and the entire +125 ELO lives in the sparse
overridden rows.

That makes this rate the single number the corpus design turns on:

  * if it is small, uniform cross-entropy is mostly the student learning
    itself, and the loss must weight or isolate the divergent rows
  * it also sets the corpus SIZE: the useful signal is (rate x prompts), not
    prompts, so a 1% rate means 100 games buy what 1 game appears to

Everything needed is already counted — `wrap_with_solver` keeps prompts,
solver_fired, changed and per-tier trig_*/fire_* in the stats dict that
`matchrunner.SOLVED_STATS` hands it. This just runs games with that dict
installed and reports. Single process on purpose: the counters are a module
global and would not survive the spawn boundary.

    uv run python scripts/m41b_divergence_probe.py --games 40
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_ARM = "solved:checkpoints/m39_retain_b.pt:alakazam_v2_h4"
DEFAULT_BED = "model:checkpoints/m39_bc_grim.pt:grim_live"

#: Below this share of prompts, uniform cross-entropy spends almost all of its
#: gradient on rows where teacher and student already agree.
SPARSE_AT = 0.02


def summarize(stats: dict) -> dict:
    """The measurement core: wrapper counters -> the rates that decide corpus
    design. Pure, so it can be fixtured without playing a game."""
    prompts = int(stats.get("prompts", 0))
    fired = int(stats.get("solver_fired", 0))
    changed = int(stats.get("changed", 0))
    tiers = {}
    for key in stats:
        if key.startswith("trig_"):
            tier = key[len("trig_"):]
            tiers[tier] = (int(stats[key]), int(stats.get(f"fire_{tier}", 0)))
    return {
        "prompts": prompts, "fired": fired, "changed": changed,
        "fire_rate": fired / prompts if prompts else 0.0,
        "change_rate": changed / prompts if prompts else 0.0,
        "change_given_fire": changed / fired if fired else 0.0,
        "rows_per_1000": (changed / prompts * 1000) if prompts else 0.0,
        "sparse": (changed / prompts if prompts else 0.0) < SPARSE_AT,
        "tiers": tiers,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", default=DEFAULT_ARM,
                    help="a solved: spec — the composite whose labels we would distil")
    ap.add_argument("--bed", default=DEFAULT_BED)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    import rl.matchrunner as mr

    stats: dict = {}
    mr.SOLVED_STATS = stats                 # the wrapper writes here
    print(f"arm  {a.arm}\nbed  {a.bed}\ngames {a.games} "
          f"(single process — the counters are a module global)\n")

    results = mr.play_series(mr.parse_spec(a.arm), mr.parse_spec(a.bed),
                             a.games, seed=a.seed)
    wr = mr.series_wr(results)

    s = summarize(stats)
    prompts, fired, changed = s["prompts"], s["fired"], s["changed"]
    if not prompts:
        print("FAIL: no prompts counted — is --arm really a `solved:` spec? "
              "Only that branch installs SOLVED_STATS.")
        return 1

    print(f"win rate {wr:.4f} over {len(results)} games\n")
    print(f"{'prompts':<22}{prompts:>8}")
    print(f"{'solver fired':<22}{fired:>8}  {fired / prompts:>7.2%} of prompts")
    print(f"{'action CHANGED':<22}{changed:>8}  {changed / prompts:>7.2%} of prompts"
          f"   <-- the distillable signal")
    if fired:
        print(f"{'changed | fired':<22}{'':>8}  {changed / fired:>7.2%} "
              "(when it fires, how often it disagrees)")

    tiers = sorted(k for k in stats if k.startswith("trig_"))
    if tiers:
        print("\nper trigger tier:")
        for key in tiers:
            tier = key[len("trig_"):]
            trig = stats[key]
            fire = stats.get(f"fire_{tier}", 0)
            print(f"  {tier:<20} triggered {trig:>6}   fired {fire:>6}"
                  f"   {fire / trig if trig else 0:>7.2%}")

    print("\n--- what this means for the corpus ---")
    rate = s["change_rate"]
    print(f"Useful rows per 1,000 prompts: ~{rate * 1000:.0f}. "
          f"A corpus of N prompts carries ~{rate:.1%} teacher signal; the rest "
          "is the student's own argmax.")
    if s["sparse"]:
        print("VERDICT: SPARSE. Uniform cross-entropy would spend >98% of its "
              "gradient on rows where teacher and student already agree — the "
              "loss must upweight or isolate the divergent rows, and the "
              "corpus must be sized on the divergent count, not the prompt "
              "count. This is a plausible mechanism for M40 X5's 0.20 "
              "retention that Part I's encoder work does NOT address.")
    else:
        print("VERDICT: NOT SPARSE at this budget — uniform cross-entropy is "
              "defensible, though divergence weighting is still worth an arm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
