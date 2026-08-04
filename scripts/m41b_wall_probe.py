"""What does the solver DO differently in the wall matchup? (M41b, Stage B fallout)

The Stage B teacher gap is matchup-specific: the composite is worth +4.3pp
against the wall clone (twice replicated, z=3.20 and z=3.61) and NEGATIVE
against grim, where most live loss mass sits. So the composite is not a
teacher worth distilling globally — but that wall gain is a real, repeatable
behavioural difference, and a behaviour that can be NAMED can usually be had
for the price of a rule or an encoder feature instead of a corpus and a
retrain.

`m41b_divergence_probe.py` showed the rate is not the story (3.0% wall vs
2.7% grim) but the TIER is: `T1_ko_one_attach` fires 22 times on wall and
once on grim. This probe answers the next question down — on the prompts
where the solver overrides, WHAT does it play instead of the net's pick?

It re-implements the wrapper's decision inline rather than calling
`wrap_with_solver`, because the wrapper deliberately returns only the final
action; the comparison IS the measurement here. `solve_turn`'s own override
bar is reused untouched, so a fire here is a fire there.

    uv run python scripts/m41b_wall_probe.py --games 30
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_NET = "model:checkpoints/m39_retain_b.pt:alakazam_v2_h4"
BEDS = {
    "wall": "model:checkpoints/m38_bc_wall.pt:greattusk_wall",
    "grim": "model:checkpoints/m39_bc_grim.pt:grim_live",
}


def option_label(opt, obs) -> str:
    """A short human label for one option — type plus what it acts on."""
    from cg.api import OptionType
    from rl.encoders import _card_id_at
    try:
        name = OptionType(int(opt.type)).name
    except ValueError:
        return f"TYPE{opt.type}"
    st = obs.current
    if int(opt.type) == int(OptionType.ATTACK):
        from rl.combat import _ATK
        aid = getattr(opt, "attackId", None)
        dmg = _ATK.get(aid, (0, ()))[0] if aid is not None else 0
        return f"ATTACK(dmg={dmg})"
    if int(opt.type) == int(OptionType.ATTACH):
        area = opt.inPlayArea
        where = ("ACTIVE" if area is not None and int(area) == 4
                 else f"BENCH{opt.inPlayIndex}" if area is not None else "?")
        return f"ATTACH->{where}"
    if int(opt.type) == int(OptionType.PLAY):
        cid = opt.cardId
        if cid is None and opt.index is not None:
            cid = _card_id_at(obs, 2, opt.index, st.yourIndex)
        from rl.encoders import _CARD_NAME
        return f"PLAY({_CARD_NAME.get(cid) or cid})"
    return name


def run(net_spec: str, bed_spec: str, games: int, seed: int) -> Counter:
    import rl.matchrunner as mr
    from cg.api import to_observation_class
    from rl.turn_solver import solve_trigger, solve_turn

    inner_fn, deck = mr.make_pilot(mr.parse_spec(net_spec), instance="fx")
    tally: Counter = Counter()
    examples: dict[str, list] = {}

    def probe_agent(obs_dict):
        base = inner_fn(obs_dict)              # ALWAYS — keeps inner state live
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return base
        tally["prompts"] += 1
        tier = solve_trigger(obs)
        if tier is None:
            return base
        tally[f"trig::{tier}"] += 1
        try:
            pick = solve_turn(obs, deck, deadline_s=mr.SOLVED_DEADLINE_S,
                              max_nodes=mr.SOLVED_MAX_NODES)
        except Exception:                      # noqa: BLE001 — forensics
            tally["solver_error"] += 1
            return base
        if pick is None:
            return base
        tally[f"fire::{tier}"] += 1
        if list(pick) == list(base):
            tally["fired_but_same"] += 1
            return pick
        tally["CHANGED"] += 1
        opts = list(obs.select.option or ())
        net_i = base[0] if base else None
        sol_i = pick[0] if pick else None
        net_lbl = option_label(opts[net_i], obs) if net_i is not None and net_i < len(opts) else "?"
        sol_lbl = option_label(opts[sol_i], obs) if sol_i is not None and sol_i < len(opts) else "?"
        key = f"{net_lbl}  ->  {sol_lbl}"
        tally[f"swap::{key}"] += 1
        tally[f"swap_tier::{tier}::{key}"] += 1
        examples.setdefault(tier, [])
        if len(examples[tier]) < 3:
            me = obs.current.players[obs.current.yourIndex]
            opp = obs.current.players[1 - obs.current.yourIndex]
            act = me.active[0] if me.active and me.active[0] else None
            oac = opp.active[0] if opp.active and opp.active[0] else None
            examples[tier].append(
                f"turn {obs.current.turn}: my active {getattr(act, 'id', None)} "
                f"e={len(getattr(act, 'energies', ()) or ())} | "
                f"opp {getattr(oac, 'id', None)} hp={getattr(oac, 'hp', None)} "
                f"| net {net_lbl} -> solver {sol_lbl}")
        return pick

    bed_fn, bed_deck = mr.make_pilot(mr.parse_spec(bed_spec), instance="fxb")
    for g in range(games):
        a_seat = g % 2
        fns = (probe_agent, bed_fn) if a_seat == 0 else (bed_fn, probe_agent)
        decks = (deck, bed_deck) if a_seat == 0 else (bed_deck, deck)
        try:
            mr._engine_game(fns[0], fns[1], decks[0], decks[1], None)
        except Exception:                      # noqa: BLE001
            tally["game_error"] += 1
    tally["_examples"] = 0
    run.examples = examples                    # noqa: B023 — reporting side channel
    return tally


def report(name: str, tally: Counter, examples: dict) -> None:
    prompts = tally.get("prompts", 0)
    changed = tally.get("CHANGED", 0)
    print(f"\n=== {name} ===")
    print(f"prompts {prompts}   changed {changed} "
          f"({changed / prompts if prompts else 0:.2%})   "
          f"fired-but-same {tally.get('fired_but_same', 0)}")
    tiers = sorted(k[len('trig::'):] for k in tally if k.startswith("trig::"))
    for tier in tiers:
        print(f"  {tier:<20} trig {tally[f'trig::{tier}']:>5}  "
              f"fire {tally.get(f'fire::{tier}', 0):>5}")
    swaps = [(v, k[len('swap::'):]) for k, v in tally.items()
             if k.startswith("swap::")]
    if swaps:
        print("  what the solver plays INSTEAD (net -> solver):")
        for count, label in sorted(swaps, reverse=True)[:10]:
            print(f"    {count:>4}  {label}")
    for tier, rows in examples.items():
        if rows:
            print(f"  examples [{tier}]:")
            for r in rows:
                print(f"    {r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default=DEFAULT_NET)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--beds", nargs="*", default=["wall", "grim"])
    a = ap.parse_args()

    for bed in a.beds:
        tally = run(a.net, BEDS[bed], a.games, a.seed)
        report(bed, tally, getattr(run, "examples", {}))
    print("\nRead: a swap that repeats across games is a NAMEABLE behaviour, "
          "and a nameable behaviour is a rule or an encoder feature — not a "
          "corpus and a retrain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
