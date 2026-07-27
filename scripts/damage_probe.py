"""Variable-attack damage probe (M36 E4 / P0.5).

`rl.combat._ATK` stores damage=0 for text-formula attacks, so `_attack_damage`
models Alakazam 743's Powerful Hand (1072) as 0 — plan features, turn solver
and any KO-gate are blind to our main attacker. The engine card text says
"Place 2 damage counters ... for each card in your hand" => 20 x hand_size,
and counter PLACEMENT should bypass weakness/resistance (which would void the
"Great Tusk is psychic-weak, type math is for us" premise of the wall matchup).

This probe confirms empirically: wraps side A's pilot, records every chosen
ATTACK (attack id, hand size at the prompt, opp active id/remaining hp), then
reads the opp active's hp at A's next observation. Events where the opponent's
active changed or healed are dropped; KOs are counted censored (dmg >= hp).

Usage:
    uv run python scripts/damage_probe.py \
        --b solver:decks/greattusk_wall.csv -n 30 --seed 11
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cg.api import OptionType, SelectContext, to_observation_class
from rl.combat import _ATK, _CARD
from rl.matchrunner import make_pilot, parse_spec, play_series


class AttackTap:
    """Wraps side A's pilot; measures damage its attacks actually deal."""

    def __init__(self, fn):
        self.fn = fn
        self.pending = None    # (attack_id, hand_size, opp_id, opp_hp, opp_en)
        self.events = []       # (attack_id, hand_size, opp_id, delta, opp_en)
        self.censored = Counter()    # attack_id -> KO/target-changed count
        self.healed = 0

    def reset_game(self):
        self.pending = None

    def __call__(self, od):
        picks = self.fn(od)
        obs = to_observation_class(od)
        st, sel = obs.current, obs.select
        if st is None or sel is None:
            return picks
        me = st.players[st.yourIndex]
        op = st.players[1 - st.yourIndex]
        op_active = op.active[0] if op.active and op.active[0] else None

        # resolve a pending attack against the current view of opp's active
        if self.pending is not None:
            aid, hand_n, opp_id, opp_hp, opp_en = self.pending
            self.pending = None
            if op_active is None or op_active.id != opp_id:
                self.censored[aid] += 1          # KO'd or swapped out
            elif (op_active.hp or 0) > opp_hp:
                self.healed += 1                 # healed/evolved: unusable
            else:
                self.events.append(
                    (aid, hand_n, opp_id, opp_hp - (op_active.hp or 0),
                     opp_en))

        if sel.context != SelectContext.MAIN or not picks:
            return picks
        chosen = sel.option[picks[0]]
        if OptionType(chosen.type) != OptionType.ATTACK:
            return picks
        active = me.active[0] if me.active and me.active[0] else None
        if active is None or op_active is None or active.id not in _CARD:
            return picks
        attacks = _CARD[active.id][3]
        idx = chosen.index or 0
        if idx >= len(attacks):
            return picks
        stadium = tuple(c.id for c in (st.stadium or []) if c is not None)
        self.pending = (attacks[idx], len(me.hand or []),
                        op_active.id, op_active.hp or 0,
                        (tuple(getattr(op_active, "energies", ()) or ()),
                         stadium))
        return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="modelt-gacf:checkpoints/m28_winners.pt:"
                                   "decks/alakazam_v2.csv")
    ap.add_argument("--b", required=True)
    ap.add_argument("-n", "--games", type=int, default=30)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    spec_a, spec_b = parse_spec(args.a), parse_spec(args.b)
    fn_a, deck_a = make_pilot(spec_a, instance="dmgprobe_a")
    tap = AttackTap(fn_a)

    def game_fn(fn0, fn1, deck0, deck1, stats, a_seat=None):
        from rl.matchrunner import _engine_game
        tap.reset_game()
        return _engine_game(fn0, fn1, deck0, deck1, stats)

    def swap_pilot(spec, instance):
        fn, deck = make_pilot(spec, instance=instance)
        return (tap, deck_a) if spec == spec_a else (fn, deck)

    # play_series builds its own pilots; monkey-patch make_pilot so side A
    # gets the tap (same trick as the other probe scripts' in-process runs).
    import rl.matchrunner as mr
    orig = mr.make_pilot
    mr.make_pilot = lambda spec, instance: (
        (tap, deck_a) if spec == spec_a else orig(spec, instance))
    try:
        res = play_series(spec_a, spec_b, args.games, seed=args.seed,
                          game_fn=game_fn)
    finally:
        mr.make_pilot = orig

    w, l, d = res.count(0), res.count(1), res.count(2)
    print(f"\nseries {w}W {l}L {d}D vs {args.b}")
    print(f"clean events {len(tap.events)} | censored(KO/swap) "
          f"{dict(tap.censored)} | healed-dropped {tap.healed}")

    by_attack = defaultdict(list)
    for aid, hand_n, opp_id, delta, opp_en in tap.events:
        by_attack[aid].append((hand_n, opp_id, delta, opp_en))
    for aid, rows in sorted(by_attack.items()):
        table_dmg = _ATK.get(aid, (None,))[0]
        print(f"\nattack {aid} (table damage {table_dmg}), {len(rows)} events:")
        for hand_n, opp_id, delta, (opp_en, stadium) in rows:
            weak = _CARD.get(opp_id, (None,))[0]
            per = f"{delta / hand_n:5.1f}" if hand_n else "  n/a"
            print(f"    hand={hand_n:2d} delta={delta:3d} ({per}/card) "
                  f"opp={opp_id} weak={weak} energies={list(opp_en)} "
                  f"stadium={list(stadium)}")


if __name__ == "__main__":
    main()
