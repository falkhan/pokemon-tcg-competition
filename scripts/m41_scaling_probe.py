"""Did the net learn to attack with a scaling attacker its features call harmless?

`rl/combat.py::_best_damage` reads an attack's PRINTED damage. Our deck's whole
plan, Alakazam #743 `Powerful Hand`, prints **0** and really does 2 damage
counters per card in our hand (20 x hand size). So `rl/encoders.py`'s combat
block hands the net, on every turn our win condition is on the board:

    my_dmg = 0      my_ko = 0.0      my_can_attack = 0.0

i.e. "this Pokemon cannot attack". The rules layer inherits it too
(`rl/plan.py:118` return-KO estimation). Pool-wide, 174 of 1556 attacks scale
and 142 of those print <=30.

BUT the net has a channel the feature does not: it is behaviour-cloned from
pilots who DO attack, and it separately sees hand size and the opponent's
remaining HP. So the feature can be wrong while the policy is right. This
measures which, on our own shipped agents' live games:

  * how often we attack on a TURN where the attacker is active and can attack
  * whether that rate RISES with hand size (the scaling the feature hides)
  * how often a lethal attack was available all turn and never taken

A flat rate across hand sizes means the policy is as blind as its features.
A rising rate means it compensated and this is a cosmetic feature bug.

THE UNIT IS THE TURN, NOT THE PROMPT. A turn holds several prompts but allows
only one attack, so per-prompt accounting scores every play-a-card prompt in a
turn that ENDED in an attack as a declined attack. The first cut of this probe
did exactly that and reported 1,046 "missed lethals", nearly all of them turns
we attacked on.

Usage:
    uv run python scripts/m41_scaling_probe.py
    uv run python scripts/m41_scaling_probe.py --subs 55182097 55185485 --attacker 743
"""
import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import polars as pl  # noqa: E402
from cg.api import OptionType, SelectContext, to_observation_class  # noqa: E402

from rl.combat import _best_damage  # noqa: E402
from rl.kaggle_ingest import EPISODES_PQ, RAW_DIR  # noqa: E402
from rl.replay_bc import iter_replay_decisions  # noqa: E402

# Our shipped agents (M39 Ship A/B and the M40 floor).
OUR_SUBS = [55172160, 55182097, 55185485]
ALAKAZAM_ID = 743                 # `Powerful Hand`: 2 damage counters per card in hand
DAMAGE_PER_HAND_CARD = 20


def our_episodes(subs: list[int]) -> list[tuple[int, int]]:
    """(episode_id, our_seat) for every cached replay of these submissions."""
    df = pl.read_parquet(EPISODES_PQ)
    out = []
    for r in df.iter_rows(named=True):
        for seat in (0, 1):
            if r[f"submission_id_{seat}"] in subs:
                if (RAW_DIR / f"episode_{r['episode_id']}.json.gz").exists():
                    out.append((r["episode_id"], seat))
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subs", type=int, nargs="*", default=OUR_SUBS)
    ap.add_argument("--attacker", type=int, default=ALAKAZAM_ID)
    ap.add_argument("--per-card", type=int, default=DAMAGE_PER_HAND_CARD)
    a = ap.parse_args()

    eps = our_episodes(a.subs)
    print(f"scaling probe: attacker id {a.attacker}, "
          f"{len(eps)} cached episodes over subs {a.subs}\n", flush=True)

    # (episode, turn) -> {attacked, lethal_seen, hand_at_last_chance, ours}
    turns: dict[tuple[int, int], dict] = {}
    other_turns: dict[tuple[int, int], dict] = {}
    model_says_can_attack = Counter()
    drops = Counter()

    for episode_id, seat in eps:
        path = RAW_DIR / f"episode_{episode_id}.json.gz"
        try:
            steps = json.loads(gzip.decompress(path.read_bytes()))["steps"]
        except Exception:  # noqa: BLE001 — a corrupt cache entry is not fatal
            continue
        for _i, obs_dict, action in iter_replay_decisions(steps, seat, drops):
            try:
                obs = to_observation_class(obs_dict)
                st, sel = obs.current, obs.select
                if st is None or sel is None or sel.context != SelectContext.MAIN:
                    continue
                me = st.players[st.yourIndex]
                op = st.players[1 - st.yourIndex]
                mine = [p for p in (me.active or []) if p]
                theirs = [p for p in (op.active or []) if p]
                if not mine or not theirs:
                    continue
                if not any(OptionType(o.type) == OptionType.ATTACK
                           for o in sel.option):
                    continue                    # attacking not legal here

                attacked = OptionType(sel.option[action[0]].type) == OptionType.ATTACK
                key = (episode_id, int(st.turn))
                book = turns if mine[0].id == a.attacker else other_turns
                rec = book.setdefault(key, {"attacked": False, "lethal": False,
                                            "hand": 0})
                rec["attacked"] = rec["attacked"] or attacked
                if mine[0].id != a.attacker:
                    continue

                hand = len(me.hand or [])
                # The hand at the LAST chance to attack this turn is what the
                # attack would actually have done.
                rec["hand"] = hand
                if a.per_card * hand >= (theirs[0].hp or 0):
                    rec["lethal"] = True
                model_says_can_attack[_best_damage(mine[0], theirs[0]) > 0] += 1
            except Exception:  # noqa: BLE001 — forensics, not a rules engine
                continue

    by_hand = defaultdict(lambda: [0, 0])
    total = [len(turns), sum(r["attacked"] for r in turns.values())]
    lethal = [0, 0]
    for rec in turns.values():
        by_hand[rec["hand"]][0] += 1
        by_hand[rec["hand"]][1] += rec["attacked"]
        if rec["lethal"]:
            lethal[0] += 1
            lethal[1] += rec["attacked"]
    other_attacker = [len(other_turns),
                      sum(r["attacked"] for r in other_turns.values())]

    if not total[0]:
        raise SystemExit("no prompts found with that attacker active — "
                         "refresh the replay cache or check --attacker")

    print(f"TURNS with the attacker active and ATTACK legal: {total[0]}")
    print(f"  attacked: {total[1]} ({total[1] / total[0]:.1%})")
    print(f"  what _best_damage said: can-attack True in "
          f"{model_says_can_attack[True]} of {total[0]} "
          f"({model_says_can_attack[True] / total[0]:.1%})\n")

    print("attack rate BY HAND SIZE (the scaling the feature hides):")
    print(f"  {"hand":>5} {"real dmg":>9} {"turns":>7} {'attacked':>9} {'rate':>7}")
    lo_off = lo_atk = hi_off = hi_atk = 0
    for hand in sorted(by_hand):
        offers, atks = by_hand[hand]
        if offers < 3:
            continue
        print(f"  {hand:>5} {a.per_card * hand:>9} {offers:>7} {atks:>9} "
              f"{atks / offers:>6.1%}")
        if hand <= 3:
            lo_off, lo_atk = lo_off + offers, lo_atk + atks
        elif hand >= 6:
            hi_off, hi_atk = hi_off + offers, hi_atk + atks

    if lo_off and hi_off:
        lo, hi = lo_atk / lo_off, hi_atk / hi_off
        print(f"\n  hand<=3: {lo:.1%} ({lo_off} prompts)   "
              f"hand>=6: {hi:.1%} ({hi_off} prompts)   delta {hi - lo:+.1%}")
        print("  A RISING rate means the policy learned the scaling its features "
              "hide.\n  A FLAT rate means it is as blind as they are.")

    if lethal[0]:
        print(f"\nlethal available (real damage >= their remaining HP): {lethal[0]}")
        print(f"  attacked: {lethal[1]} ({lethal[1] / lethal[0]:.1%})  "
              f"MISSED: {lethal[0] - lethal[1]}")

    if other_attacker[0]:
        print(f"\ncontrol — other attackers active, ATTACK legal: "
              f"{other_attacker[0]}, attacked {other_attacker[1]} "
              f"({other_attacker[1] / other_attacker[0]:.1%})")
    if drops:
        print(f"\nwalker drops: {dict(drops)}")


if __name__ == "__main__":
    main()
