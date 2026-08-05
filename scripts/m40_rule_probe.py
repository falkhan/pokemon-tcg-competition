"""M40 deep-dive rules — G-11 mechanism probe for ashguard / hammer / tempo.

Per arm: play n games vs a bed with the ARM's apply_play_overrides call
instrumented (the planzero-probe attribution pattern — fn3/fn4 resolve the
override functions at make_pilot time, so patching around ONE make_pilot call
binds the counter to that pilot alone). Reports, per arm vs the control:

  - fires: prompts where the override changed the model's top pick
  - Sacred Ash plays by deck count (ashguard's target: no plays at deck>12)
  - Enhanced Hammer / tempo-item plays per game (hammer/tempo's target)
  - END picks per game (passivity)

Usage: uv run python scripts/m40_rule_probe.py [-n 16] [--bed <spec>]
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rl.matchrunner as mr  # noqa: E402
import rl.plan as rp  # noqa: E402
from cg.api import OptionType  # noqa: E402

ARMS = ("model-cz", "model-cz-ashw", "model-cz-ham", "model-cz-tempo")


def new_census(n_games: int = 0) -> dict:
    """One arm's empty census (shape shared with the golden-fixture tests)."""
    return {"prompts": 0, "fires": 0, "ash_deck": [], "hammer_plays": 0,
            "tempo_plays": 0, "end_picks": 0, "wins": 0, "games": n_games}


def record_pick(rec: dict, obs, ranked: list, out: list) -> None:
    """Measurement core: classify ONE apply_play_overrides decision into the
    arm's census. `ranked` is the model's order, `out` the post-override order;
    a fire is the override changing the top pick. Golden-fixtured in
    tests/test_probes_rule_mechanism.py."""
    rec["prompts"] += 1
    if out[0] != ranked[0]:
        rec["fires"] += 1
    st = obs.current
    me = st.players[st.yourIndex]
    hand = me.hand or []
    opt = obs.select.option[out[0]]
    if opt.type == OptionType.END:
        rec["end_picks"] += 1
    if opt.type == OptionType.PLAY and opt.index is not None \
            and opt.index < len(hand):
        cid = hand[opt.index].id
        if cid == rp.SACRED_ASH_ID:
            rec["ash_deck"].append(me.deckCount)
        elif cid == rp.ENHANCED_HAMMER_ID:
            rec["hammer_plays"] += 1
        elif cid in rp._TEMPO_ITEM_IDS:
            rec["tempo_plays"] += 1


def probe_arm(kind: str, checkpoint: str, deck: str, bed: str,
              n_games: int, seed: int) -> dict:
    rec = new_census(n_games)
    real = rp.apply_play_overrides

    def counting(obs, ranked, fixes):
        out = real(obs, ranked, fixes)
        record_pick(rec, obs, ranked, out)
        return out

    rp.apply_play_overrides = counting
    try:
        fn_a, deck_a = mr.make_pilot(
            mr.parse_spec(f"{kind}:{checkpoint}:{deck}"), instance=f"rp{seed}_a")
    finally:
        rp.apply_play_overrides = real
    fn_b, deck_b = mr.make_pilot(mr.parse_spec(bed), instance=f"rp{seed}_b")
    for g in range(n_games):
        fns = (fn_a, fn_b) if g % 2 == 0 else (fn_b, fn_a)
        decks = (deck_a, deck_b) if g % 2 == 0 else (deck_b, deck_a)
        res = mr._engine_game(fns[0], fns[1], decks[0], decks[1])
        rec["wins"] += int(res == g % 2)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--games", type=int, default=16)
    ap.add_argument("--checkpoint", default="checkpoints/m38_w9294_cont3.pt")
    ap.add_argument("--deck", default="alakazam_v2_h4")
    ap.add_argument("--bed",
                    default="model:checkpoints/m28_winners.pt:alakazam_v2_h4")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    print(f"rule probe vs {a.bed}, n={a.games}/arm (small n: behaviour "
          f"counts only, W-L is NOT a result)")
    print(f"{'arm':16s}{'prompts':>8s}{'fires':>7s}{'END':>6s}"
          f"{'hammer':>7s}{'tempo':>6s}  ash plays @deck")
    for kind in ARMS:
        r = probe_arm(kind, a.checkpoint, a.deck, a.bed, a.games, a.seed)
        ash = ",".join(map(str, sorted(r["ash_deck"]))) or "-"
        print(f"{kind:16s}{r['prompts']:>8d}{r['fires']:>7d}"
              f"{r['end_picks']:>6d}{r['hammer_plays']:>7d}"
              f"{r['tempo_plays']:>6d}  {ash}")
    print("\nread: each candidate must FIRE (>0) and move its own metric vs "
          "model-cz; ashguard additionally must show NO ash plays at deck>12.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
