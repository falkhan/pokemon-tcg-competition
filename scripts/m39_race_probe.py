"""M39 P2: the G-11 mechanism proof for the anti-deck-out package.

Same pattern as scripts/conserve_probe.py, generalised to the three P2 rules
and to per-rule attribution: wrap `rl.plan.apply_play_overrides`, and isolate
a real FIRE by re-running the identical call with that ONE rule removed from
the fix set. If the returned order changes, the rule acted. Single-process,
so unlike `matchrunner --workers 8` it is reproducible (G-12).

Why this runs BEFORE the P2 gate rather than after it. P1-inv is the worked
example: a mechanism probe predicted which beds `conserve` could possibly
affect, that prediction matched the win-rate data it had not seen, and a rule
a significance threshold had discarded got rescued on evidence. The same
question decides P2:

  * fires == 0 on a bed        -> that cell CANNOT move, and a WR delta there
                                  is noise however significant it looks
  * fires on a bed we did not  -> a false positive; the id sets are wrong,
    intend                        not the rule
  * fires where the loss mass  -> the arm is worth n=1200 x 3 draws
    actually is

The false-positive check is the point of running the mirror and rocket beds:
P2a adds ids (Fan Rotom is in 2 of 83 cached mirror lists) and rocket is the
one bed where `conserve` measured negative, so both are places we want a
measured zero rather than an argument.

Usage:
    uv run python scripts/m39_race_probe.py --bed wall -n 30
    uv run python scripts/m39_race_probe.py --all -n 30
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rl.plan as rp  # noqa: E402
from rl.matchrunner import parse_spec, play_series  # noqa: E402

NET = "checkpoints/m38_w9294_cont3.pt"
BEDS = {
    # the deck-race loss mass P2 exists for
    "wall": "model:checkpoints/m38_bc_wall.pt:greattusk_wall",
    "grim": "model:checkpoints/m39_bc_grim.pt:grim_live",
    "archaludon": "model:checkpoints/m39_bc_archaludon.pt:archaludon",
    # inertness controls: no wall/pressure body on either list
    "mirror": "model:checkpoints/m28_winners.pt:clone54618168",
    "rocket": "model:checkpoints/m30_bc_rocket_54834745.pt:"
              "data/kaggle/rocket_3394cd30_deck.csv",
    "top": "model:checkpoints/m39_bc_top.pt:clone54618168",
}
# The Ship B candidate string, probed rule by rule.
PKG = frozenset({rp.PLAY_FIX_CONSERVE, rp.PLAY_FIX_RACEMODE2,
                 rp.PLAY_FIX_RACEMODE4, rp.PLAY_FIX_RACEASH})
RULES = (rp.PLAY_FIX_CONSERVE, rp.PLAY_FIX_RACEMODE2,
         rp.PLAY_FIX_RACEMODE4, rp.PLAY_FIX_RACEASH)


def probe_bed(bed: str, games: int, seed: int, arm: str) -> Counter:
    stats = Counter()
    live = {"own_min": 60, "opp_min": 60}
    orig = rp.apply_play_overrides

    def counting(obs, ranked, fixes):
        out = orig(obs, ranked, fixes)
        if (obs.select is None or obs.current is None
                or obs.select.context != rp.SelectContext.MAIN):
            return out
        stats["main_prompts"] += 1
        st = obs.current
        me = st.players[st.yourIndex]
        op = st.players[1 - st.yourIndex]
        live["own_min"] = min(live["own_min"], me.deckCount)
        live["opp_min"] = min(live["opp_min"], op.deckCount)
        if rp._racemode_engaged(st, me):
            stats["race_engaged"] += 1
        for rule in RULES:
            if rule in fixes and out != orig(obs, ranked, fixes - {rule}):
                stats[f"fire_{rule}"] += 1
        return out

    def on_game(g, result, seat_stats):
        # The CAUSAL claim P2 makes is about the deck race, not about win rate
        # -- so the probe measures the race directly. own_min/opp_min are the
        # lowest deck counts either side was seen at, over our own MAIN
        # prompts; `deckout_risk` counts games where ours reached zero.
        stats["own_min_sum"] += live["own_min"]
        stats["opp_min_sum"] += live["opp_min"]
        stats["deckout_risk"] += live["own_min"] <= 0
        stats["opp_deckout"] += live["opp_min"] <= 0
        live["own_min"] = live["opp_min"] = 60

    rp.apply_play_overrides = counting
    try:
        a = parse_spec(f"{arm}:{NET}:decks/alakazam_v2_h4.csv")
        results = play_series(a, parse_spec(BEDS[bed]), games, seed=seed,
                              on_game=on_game)
    finally:
        rp.apply_play_overrides = orig

    res = list(results["results"] if isinstance(results, dict) else results)
    stats["games"] = len(res)
    stats["wins"] = res.count(0)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bed", default="wall", choices=sorted(BEDS))
    ap.add_argument("--all", action="store_true", help="every bed in turn")
    ap.add_argument("-n", "--games", type=int, default=30)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--arm", default="model-c-pkga",
                    help="matchrunner fix kind, e.g. model-conserve "
                         "(the live config) or model-c-pkg (Ship B)")
    args = ap.parse_args()

    beds = sorted(BEDS) if args.all else [args.bed]
    print(f"probe arm = {args.arm}\n")
    head = (f"{'bed':<12}{'games':>6}{'WR':>7}{'prompts':>9}{'race_on':>9}")
    for rule in RULES:
        head += f"{rule[:9]:>10}"
    head += f"{'own_min':>9}{'opp_min':>9}{'we deck0':>10}{'opp deck0':>10}"
    print(head)
    print("-" * len(head))
    for bed in beds:
        s = probe_bed(bed, args.games, args.seed, args.arm)
        g = max(s["games"], 1)
        line = (f"{bed:<12}{s['games']:>6}{s['wins'] / g:>7.3f}"
                f"{s['main_prompts']:>9}{s['race_engaged']:>9}")
        for rule in RULES:
            line += f"{s[f'fire_{rule}']:>10}"
        line += (f"{s['own_min_sum'] / g:>9.1f}{s['opp_min_sum'] / g:>9.1f}"
                 f"{s['deckout_risk']:>10}{s['opp_deckout']:>10}")
        print(line)
    print("\nown_min / opp_min = mean lowest deck count either side reached "
          "(over our MAIN prompts). The deck race is the CAUSAL claim P2 "
          "makes, so it is measured directly rather than inferred from WR.")
    print("\nA bed with race_on == 0 cannot exercise racemode2/4/raceash at "
          "all -- read its gate cell as an inertness check, not as evidence "
          "for or against the package (G-11).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
