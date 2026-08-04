"""M40 S6 mechanism probe — does `planzero` actually act?

G-11 (standing since M39): every shipped config change needs a mechanism proof,
not just a gate delta. M39 earned that rule twice — `conserve` was nearly
discarded on a significance threshold until scripts/conserve_probe.py showed
*where* it fires, and `racemode2` looked positive on a bed its predicate could
never touch. A gate number without a mechanism is a coin flip with a p-value.

For `planzero` the mechanism claim is exact and therefore cheaply falsifiable:
the plan head must NEVER run, and the vector handed to the option scorer must
be all zeros at EVERY prompt. This probe instruments both sides of that claim
directly — it counts `enumerate_plans` calls and records the actual `plan`
argument `OptionScorerV3.act` receives — and asserts the control arm shows the
opposite on the same games (otherwise a pass is vacuous).

Usage:
    uv run python scripts/m40_planzero_probe.py [-n 6] \
        [--checkpoint checkpoints/m38_w9294_cont3.pt] [--deck alakazam_v2_h4] \
        [--bed model:checkpoints/m38_bc_wall.pt:greattusk_wall]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rl.matchrunner as mr  # noqa: E402
import rl.plan as rp  # noqa: E402
import rl.policy  # noqa: E402


def plan_is_nonzero(plan) -> bool:
    """Measurement core, per prompt: is the vector the trunk is FED non-zero?
    (The literal `serve plan=0` claim — golden-fixtured in
    tests/test_probes_rule_mechanism.py.)"""
    return bool(np.asarray(plan).any())


def claim_failures(ctl: dict, arm: dict) -> list:
    """The probe's verdict on the two censuses. Returns the FAIL lines to
    print (empty list = PASS). Golden-fixtured in
    tests/test_probes_rule_mechanism.py."""
    failures = []
    # The claim, both halves. Each is a hard equality: `planzero` is a
    # removal, so "mostly zero" would mean the guard has a hole.
    if arm["plan_enumerations"] != 0:
        failures.append(f"FAIL: planzero arm ran the plan head "
                        f"{arm['plan_enumerations']}x")
    if arm["nonzero_plan_prompts"] != 0:
        failures.append(f"FAIL: planzero arm fed a non-zero plan on "
                        f"{arm['nonzero_plan_prompts']}/{arm['prompts']} prompts")
    # Non-vacuity: the control must show the behaviour we claim to remove,
    # on the same games. Without this a broken harness reads as a pass.
    if ctl["plan_enumerations"] == 0 or ctl["nonzero_plan_prompts"] == 0:
        failures.append("FAIL: control showed no plan-head activity — the "
                        "probe is not measuring anything (harness defect, "
                        "not a result)")
    return failures


def _instrumented_pilot(spec, instance: str, rec: dict):
    """Build one pilot with PER-SIDE instrumentation.

    Attribution is the whole difficulty here, and the first version of this
    probe got it wrong: `rl.plan.enumerate_plans` is module-global, so a naive
    patch counts the OPPONENT bed's plan head too (every `model:` bed is a v3
    pilot running the same machinery) and the arm reads as a failure.

    Both seams below are resolved at make_pilot time, not call time — fn3/fn4
    close over `enumerate_plans` (rl/matchrunner.py, the `from rl.plan import`
    inside the branch) and instantiate their own OptionScorerV3 from
    `rl.policy` — so patching around a SINGLE make_pilot call binds the
    instrumentation to that pilot alone and leaves the other side pristine.
    """
    real_enum = rp.enumerate_plans
    real_cls = rl.policy.OptionScorerV3

    def counting_enum(obs):
        rec["plan_enumerations"] += 1
        return real_enum(obs)

    class _Recording(real_cls):
        """Records what the trunk is actually FED — the literal claim of
        `serve plan=0`, rather than a proxy for it."""

        def act(self, state_ctx, plan, state_ids, options, option_ids, k,
                greedy=True):
            rec["prompts"] += 1
            if plan_is_nonzero(plan):
                rec["nonzero_plan_prompts"] += 1
            return super().act(state_ctx, plan, state_ids, options, option_ids,
                               k, greedy=greedy)

    rp.enumerate_plans = counting_enum
    rl.policy.OptionScorerV3 = _Recording
    try:
        return mr.make_pilot(spec, instance=instance)
    finally:
        rp.enumerate_plans = real_enum
        rl.policy.OptionScorerV3 = real_cls


def probe_arm(kind: str, checkpoint: str, deck: str, bed: str, n_games: int,
              seed: int) -> dict:
    """Drive n_games with `kind` as side a and return side a's census only."""
    rec = {"plan_enumerations": 0, "prompts": 0, "nonzero_plan_prompts": 0}
    spec_a = mr.parse_spec(f"{kind}:{checkpoint}:{deck}")
    fn_a, deck_a = _instrumented_pilot(spec_a, f"pz{seed}_a", rec)
    fn_b, deck_b = mr.make_pilot(mr.parse_spec(bed), instance=f"pz{seed}_b")

    wins = 0
    for g in range(n_games):     # slot-fair, same rotation as mr.play_series
        a_seat = g % 2
        fns = (fn_a, fn_b) if a_seat == 0 else (fn_b, fn_a)
        decks = (deck_a, deck_b) if a_seat == 0 else (deck_b, deck_a)
        res = mr._engine_game(fns[0], fns[1], decks[0], decks[1])
        wins += int(res == a_seat)
    rec["wins"], rec["games"] = wins, n_games
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--games", type=int, default=6)
    ap.add_argument("--checkpoint", default="checkpoints/m38_w9294_cont3.pt")
    ap.add_argument("--deck", default="alakazam_v2_h4")
    ap.add_argument("--bed",
                    default="model:checkpoints/m38_bc_wall.pt:greattusk_wall")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--control", default="model-c-pkg")
    ap.add_argument("--arm", default="model-c-pkgz")
    a = ap.parse_args()

    print(f"probe: {a.arm} vs {a.control}, n={a.games} vs {a.bed}")
    ctl = probe_arm(a.control, a.checkpoint, a.deck, a.bed, a.games, a.seed)
    arm = probe_arm(a.arm, a.checkpoint, a.deck, a.bed, a.games, a.seed)

    print(f"\n{'':<14}{'prompts':>9}{'plan_enum':>11}{'nonzero_plan':>14}")
    for name, s in (("control", ctl), ("planzero", arm)):
        print(f"{name:<14}{s['prompts']:>9}{s['plan_enumerations']:>11}"
              f"{s['nonzero_plan_prompts']:>14}")

    failures = claim_failures(ctl, arm)
    for msg in failures:
        print("\n" + msg)
    ok = not failures

    print("\nPASS: the plan head never runs and the trunk is fed zeros; "
          "the control does both on the same games." if ok else "\nPROBE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
