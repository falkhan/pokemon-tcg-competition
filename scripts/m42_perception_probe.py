"""G-11 mechanism probe for the five M42 perception defects.

docs/VALIDATION.md G-11: every ship carries a fire/behavioural-diff probe, and
"a ship whose mechanism cannot be probed is a ship whose live result cannot be
attributed." This one runs BEFORE any fix, so each defect gets a rate to beat
rather than a story.

Five ladders, each staged `situation -> offered -> chose_bad`:

  over_attach        an offered ATTACH target has energy_is_dead (M41's
                     damage-free predicate); chose_bad = we picked it
  hand_scaler_blind  our active's best attack scales on HAND SIZE while
                     _best_damage reads 0; chose_bad = we attacked anyway,
                     which is the M41 617-of-617 signature that the net
                     learned around a feature its inputs hold constant
  retreat_stranded   the active cannot pay its own retreat while a ready bench
                     member waits; chose_bad = we ENDED the turn like that
  dead_basis_fetch   a fetch/promote menu offers an evolution whose basis is
                     nowhere in play or hand; chose_bad = we took it.
                     `strict` sub-counts Piotr's exact trigger: 0 benched
                     Pokemon AND 0 basics in hand
  wasted_stadium     a stadium PLAY whose id equals the stadium already out

Plus `foreign_target_opts`, which settles whether the rl/encoders.py:207
`your_index`-instead-of-`opt.playerIndex` target resolution is a live defect or
a theoretical one.

`situation == 0` is reported as SITUATION NEVER OCCURS and exits non-zero. That
is not pedantry: M41 killed three levers on exactly this reading, and a ladder
that cannot distinguish "the pilot is clean" from "the board state never
happens" would let a fix be credited for a situation it never faced.

Instrumentation wraps the ARM's pilot callable rather than patching a module
global, so the counters bind to our seat by construction (the m40_planzero_probe
attribution lesson) and the probe works for `generic:` rule arms and `model:`
neural arms alike.

Usage:
    uv run python scripts/m42_perception_probe.py \
        --arm generic:alakazam_v2_h4 --opp generic:grim_live -n 20
    uv run python scripts/m42_perception_probe.py \
        --arm model-pz:checkpoints/m41_ogerpon.pt:decks/ogerpon.csv \
        --opp model:checkpoints/m39_bc_grim.pt:grim_live -n 20 --json out.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cg.api import (CardType, OptionType, SelectContext,  # noqa: E402
                    all_card_data, to_observation_class)

from rl.combat import (_ATK, _CARD, _RETREAT, _best_damage,  # noqa: E402
                       _turns_to_ready, energy_is_dead)
from rl.generic_pilot import (_EVOLVES_FROM, _IS_BASIC, _IS_POKEMON,  # noqa: E402
                              _KEEP_CTX, _NAME, _PROMOTE_CTX)
from rl.matchrunner import _engine_game, make_pilot, parse_spec  # noqa: E402
from rl.scaling import SCALING_ATTACKS  # noqa: E402

_IS_STADIUM = {c.cardId for c in all_card_data()
               if getattr(c, "cardType", None) == CardType.STADIUM}

# Zone ids as plain ints, matching rl/generic_pilot.py:163 — the probe must not
# depend on AreaType surviving a bundle strip.
_AREA_DECK, _AREA_HAND, _AREA_DISCARD, _AREA_ACTIVE, _AREA_BENCH = 1, 2, 3, 4, 5

LADDERS = ("over_attach", "hand_scaler_blind", "retreat_stranded",
           "dead_basis_fetch", "stadium_ignored")

# Per-ladder rung labels. Deliberately NOT a shared schema: on
# `hand_scaler_blind` the bad actor is our damage model and the CHOICE is
# correct, so calling its third rung "chose badly" would be the same kind of
# confident wrong label that four M41 instruments produced. Rung 3's rate is
# reported over rung 2, which is the honest denominator — the situations where
# a different decision was actually available.
RUNGS = {
    "over_attach": (
        "a dead-energy ATTACH target was offered",
        "...and attaching was optional (declining was legal)",
        "...and WE ATTACHED ONTO IT ANYWAY"),
    "hand_scaler_blind": (
        "our active carries a hand-scaling attack",
        "...and OUR OWN damage model reads it as 0 with an ATTACK on the menu",
        "...and the attack really was available — a prompt where 7 features "
        "read 0"),
    "retreat_stranded": (
        "the active cannot pay its own retreat, a ready bench member waits",
        "...and an ATTACH or RETREAT was on the menu to fix it",
        "...and WE ENDED THE TURN in that state"),
    "dead_basis_fetch": (
        "a menu offers an evolution whose basis is nowhere",
        "...and a live (basis-less) Pokemon was on the same menu",
        "...and WE TOOK THE DEAD EVOLUTION"),
    # PER TURN, unlike the four above: playing the stadium at prompt 5 of a
    # turn is not a defect at prompts 1-4. See Ladders.end_game.
    "stadium_ignored": (
        "TURNS where a stadium PLAY was on the menu",
        "...and the OPPONENT'S stadium was in play, so ours would remove it",
        "...and THE TURN ENDED with theirs still out"),
}


def _hand_scaling_attacks() -> dict[int, int]:
    """attack id -> damage per card in hand, for every hand-mode scaler.

    Read from rl.scaling rather than hardcoded, so Phase 2's derived table
    widens this ladder without touching the probe.
    """
    return {aid: per for aid, (mode, per, _b) in SCALING_ATTACKS.items()
            if mode == "hand"}


def _zone(obs, player, area):
    if area is None:
        return None
    return {_AREA_DECK: obs.select.deck, _AREA_HAND: player.hand,
            _AREA_DISCARD: player.discard, _AREA_ACTIVE: player.active,
            _AREA_BENCH: player.bench}.get(int(area))


def _option_card_id(obs, opt, me):
    """Card id an option refers to, or None. Defaults the area to HAND, which is
    the play-tier lesson: real PLAY options carry no `area`."""
    idx = opt.index
    if idx is None:
        return None
    zone = _zone(obs, me, opt.area) if opt.area is not None else me.hand
    try:
        card = zone[idx]
    except (TypeError, IndexError):
        return None
    return getattr(card, "id", None)


def _in_play_target(me, opt):
    """The Pokemon an option addresses through inPlayArea/inPlayIndex."""
    if opt.inPlayArea is None or opt.inPlayIndex is None:
        return None
    zone = {_AREA_ACTIVE: me.active, _AREA_BENCH: me.bench}.get(int(opt.inPlayArea))
    try:
        return zone[opt.inPlayIndex]
    except (TypeError, IndexError):
        return None


def _basis_pool(me):
    """Names of my Pokemon in play and in hand — the evolution-basis pool.
    By NAME, never by id: evolves_from points at one printing and real decks
    run others (the M41 Riolu trap)."""
    in_play = [p for p in (list(me.active or []) + list(me.bench or []))
               if p is not None]
    hand_pokemon = [c for c in (me.hand or [])
                    if c is not None and c.id in _IS_POKEMON]
    return ({_NAME.get(p.id) for p in in_play}
            | {_NAME.get(c.id) for c in hand_pokemon}) - {None}


def _ready_bench(me, op_active):
    return [p for p in (me.bench or [])
            if p is not None and _turns_to_ready(p, op_active) == 0]


class Ladders:
    """The five staged counters. One instance per probe run."""

    def __init__(self):
        self.c = Counter()
        self.detail = {k: Counter() for k in LADDERS}
        self.turn_stadium: dict = {}     # turn -> their stadium name
        self.turn_stadium_offer: set = set()
        self.stadium_played: set = set()

    def end_game(self):
        """Resolve the per-TURN ladders and reset for the next game.

        `stadium_ignored` is per-turn, not per-prompt, and the distinction is
        not cosmetic: a turn has many MAIN prompts, so playing the stadium at
        prompt 5 leaves prompts 1-4 looking like declines. The first version of
        this ladder counted per prompt and read 58.5% on the Ogerpon ship --
        while `rl/postmortem.py` on that ship's own QC replays showed it played
        the stadium on EVERY turn the option existed (turns 7/11 and 5/11, 4
        for 4). The per-turn view is the one `rl/postmortem.py:250` already
        commits to: "fixing it later in the same turn is fine; ending the turn
        in the bad state is the failure."
        """
        self.c["stadium_ignored.situation"] += len(self.turn_stadium_offer)
        for turn, theirs in self.turn_stadium.items():
            self.c["stadium_ignored.offered"] += 1
            if turn not in self.stadium_played:
                self.c["stadium_ignored.chose_bad"] += 1
                self.detail["stadium_ignored"][theirs] += 1
        self.turn_stadium.clear()
        self.turn_stadium_offer.clear()
        self.stadium_played.clear()

    # -- ladder 1 -----------------------------------------------------------
    def over_attach(self, obs, me, opts, chosen, hand_scalers):
        dead = [j for j, o in enumerate(opts)
                if o.type == OptionType.ATTACH
                and (t := _in_play_target(me, o)) is not None
                and energy_is_dead(t.id, list(getattr(t, "energies", ()) or ()))]
        if not dead:
            return
        self.c["over_attach.situation"] += 1
        self.c["over_attach.offered"] += 1
        picked = [j for j in dead if j in chosen]
        if not picked:
            return
        self.c["over_attach.chose_bad"] += 1
        for j in picked:
            t = _in_play_target(me, opts[j])
            self.detail["over_attach"][_NAME.get(t.id) or f"id{t.id}"] += 1
        # The damage this costs TODAY: spending a card from hand shrinks the
        # damage formula of a hand-scaling active by exactly per_unit.
        active = me.active[0] if me.active else None
        if active is not None:
            per = max((hand_scalers[a] for a in _CARD.get(active.id, (0, 0, 0, (), 1))[3]
                       if a in hand_scalers), default=0)
            self.c["over_attach.damage_forgone"] += per * len(picked)

    # -- ladder 2 -----------------------------------------------------------
    def hand_scaler_blind(self, obs, me, op_active, opts, chosen, hand_scalers):
        active = me.active[0] if me.active else None
        if active is None or op_active is None:
            return
        attacks = _CARD.get(active.id, (0, 0, 0, (), 1))[3] or ()
        if not any(a in hand_scalers for a in attacks):
            return
        self.c["hand_scaler_blind.situation"] += 1
        attack_opts = [j for j, o in enumerate(opts)
                       if o.type == OptionType.ATTACK]
        if not attack_opts or _best_damage(active, op_active) > 0:
            return          # no attack on offer, or the model is not blind here
        # The engine is offering an attack our own damage model scores at 0.
        # That contradiction IS the defect — the choice below is correct play.
        self.c["hand_scaler_blind.offered"] += 1
        self.c["hand_scaler_blind.chose_bad"] += 1
        self.detail["hand_scaler_blind"][_NAME.get(active.id)
                                         or f"id{active.id}"] += 1
        if any(j in chosen for j in attack_opts):
            self.c["hand_scaler_blind.attacked_anyway"] += 1

    # -- ladder 3 -----------------------------------------------------------
    def retreat_stranded(self, obs, me, op_active, opts, chosen):
        active = me.active[0] if me.active else None
        if active is None:
            return
        attached = len(getattr(active, "energies", ()) or ())
        if attached >= _RETREAT.get(active.id, 0):
            return          # retreat is payable: not stranded
        if not _ready_bench(me, op_active):
            return          # nothing worth switching to
        self.c["retreat_stranded.situation"] += 1
        fixable = [j for j, o in enumerate(opts)
                   if o.type in (OptionType.ATTACH, OptionType.RETREAT)]
        if not fixable:
            return          # stranded with no legal way out is the board's
                            # fault, not the pilot's — must not count as a miss
        self.c["retreat_stranded.offered"] += 1
        if any(opts[j].type == OptionType.END for j in chosen if j < len(opts)):
            self.c["retreat_stranded.chose_bad"] += 1
            self.detail["retreat_stranded"][_NAME.get(active.id)
                                            or f"id{active.id}"] += 1

    # -- ladder 4 -----------------------------------------------------------
    def dead_basis_fetch(self, obs, me, opts, chosen, ctx):
        pool = _basis_pool(me)
        dead = []
        for j, o in enumerate(opts):
            cid = _option_card_id(obs, o, me)
            if cid is None or cid not in _IS_POKEMON:
                continue
            basis = _EVOLVES_FROM.get(cid)
            if basis is not None and basis not in pool:
                dead.append((j, cid))
        if not dead:
            return
        self.c["dead_basis_fetch.situation"] += 1
        # Piotr's exact trigger: nothing on the bench and no basic in hand, so
        # no evolution card can ever land.
        bare = (not any(p is not None for p in (me.bench or []))
                and not any(c is not None and c.id in _IS_BASIC
                            for c in (me.hand or [])))
        if bare:
            self.c["dead_basis_fetch.strict_situation"] += 1
        alive = [j for j, o in enumerate(opts)
                 if (cid := _option_card_id(obs, o, me)) is not None
                 and cid in _IS_POKEMON and _EVOLVES_FROM.get(cid) is None]
        if alive:
            self.c["dead_basis_fetch.offered"] += 1     # a live basic was there
        picked = [(j, cid) for j, cid in dead if j in chosen]
        if not picked:
            return
        self.c["dead_basis_fetch.chose_bad"] += 1
        if bare:
            self.c["dead_basis_fetch.strict_chose_bad"] += 1
        # Split by context: fetch_value only runs on KEEP contexts, so a pick
        # made under PROMOTE means the guard never ran, which is a different
        # fix from a guard that ran and was outbid.
        where = "KEEP" if ctx in _KEEP_CTX else "PROMOTE"
        self.c[f"dead_basis_fetch.in_{where}"] += 1
        for _j, cid in picked:
            self.detail["dead_basis_fetch"][
                f"{_NAME.get(cid) or f'id{cid}'} [{where}]"] += 1

    # -- ladder 5 -----------------------------------------------------------
    def stadium_ignored(self, obs, me, opts, chosen, turn):
        """The first framing of this ladder — "we replaced the stadium in play
        with the same card" — was RULE-IMPOSSIBLE and had to be thrown away.
        Measured 2026-08-04 over 20 prompts where a stadium was out AND a
        stadium PLAY was offered: same-id offers 0, different-id offers 20. The
        engine enforces the real same-name Stadium rule, so that ladder could
        only ever read 0 and would have credited a fix for a defect the rules
        already prevent. `same_id_offered` stays as a counter so the void
        framing cannot be proposed again from a clean slate.
        """
        plays = [j for j, o in enumerate(opts)
                 if o.type == OptionType.PLAY
                 and (cid := _option_card_id(obs, o, me)) is not None
                 and cid in _IS_STADIUM]
        if not plays:
            return
        self.turn_stadium_offer.add(turn)
        in_play = obs.current.stadium or []
        if not in_play:
            return                      # nothing to displace: declining is fine
        out_id = getattr(in_play[0], "id", None)
        if any(_option_card_id(obs, opts[j], me) == out_id for j in plays):
            self.c["same_id_offered"] += 1
            return
        # Ownership IS readable: cg.api.Card carries playerIndex, and it was
        # verified 2026-08-04 that every stadium we played landed with our own
        # seat index (landed_ok 6, wrong-seat 0). rl/determinize.py:56's
        # "stadium ownership is ambiguous" does not hold for this read.
        owner = getattr(in_play[0], "playerIndex", None)
        if owner is not None and int(owner) == int(obs.current.yourIndex):
            self.c["stadium_ignored.ours_already_out"] += 1
            return                      # displacing our own is not the defect
        # Banked per TURN; end_game() resolves it. See end_game's docstring for
        # why per-prompt counting was wrong by a factor of ~5 here.
        self.turn_stadium[turn] = _NAME.get(out_id) or f"id{out_id}"
        if any(j in chosen for j in plays):
            self.stadium_played.add(turn)


def instrument(fn, lad: Ladders):
    """Wrap a pilot callable so every prompt it answers feeds the ladders.

    Wrapping the callable (rather than patching rl.plan) binds the counters to
    ONE seat by construction and keeps the probe pilot-kind agnostic.
    """
    hand_scalers = _hand_scaling_attacks()

    def wrapped(obs_dict):
        picks = fn(obs_dict)
        try:
            obs = to_observation_class(obs_dict)
            if obs.select is None:
                return picks
            st = obs.current
            me = st.players[st.yourIndex]
            op = st.players[1 - st.yourIndex]
            op_active = op.active[0] if op.active else None
            opts = list(obs.select.option or ())
            chosen = {int(i) for i in (picks or ())}
            ctx = obs.select.context

            lad.c["prompts"] += 1
            for o in opts:
                if o.inPlayArea is not None and o.playerIndex is not None \
                        and int(o.playerIndex) != int(st.yourIndex):
                    lad.c["foreign_target_opts"] += 1

            if ctx == SelectContext.MAIN:
                lad.c["main_prompts"] += 1
                lad.over_attach(obs, me, opts, chosen, hand_scalers)
                lad.hand_scaler_blind(obs, me, op_active, opts, chosen,
                                      hand_scalers)
                lad.retreat_stranded(obs, me, op_active, opts, chosen)
                lad.stadium_ignored(obs, me, opts, chosen, st.turn)
            elif ctx in _KEEP_CTX or ctx in _PROMOTE_CTX:
                lad.c["fetch_prompts"] += 1
                lad.dead_basis_fetch(obs, me, opts, chosen, ctx)
        except Exception:  # noqa: BLE001 — a probe must never break the game
            lad.c["probe_errors"] += 1
        return picks

    return wrapped


def report(lad: Ladders, args) -> int:
    c = lad.c
    main = c["main_prompts"] or 1
    fetch = c["fetch_prompts"] or 1
    print(f"\nM42 perception probe: {args.games} games")
    print(f"  arm {args.arm}\n  opp {args.opp}\n")
    print(f"  prompts answered       {c['prompts']:6d}")
    print(f"  ...MAIN                {c['main_prompts']:6d}")
    print(f"  ...fetch/promote       {c['fetch_prompts']:6d}")
    if c["probe_errors"]:
        print(f"  probe_errors           {c['probe_errors']:6d}  "
              "(swallowed; the ladders below undercount by at most this)")

    dead = []
    for name in LADDERS:
        base = fetch if name == "dead_basis_fetch" else main
        sit, off, bad = (c[f"{name}.situation"], c[f"{name}.offered"],
                         c[f"{name}.chose_bad"])
        r1, r2, r3 = RUNGS[name]
        print(f"\n  [{name}]")
        if name == "stadium_ignored":
            # per TURN, not per prompt — see Ladders.end_game
            print(f"    {sit:6d}  (turns)             {r1}")
        else:
            print(f"    {sit:6d}  ({sit / base:5.1%} of prompts)  {r1}")
        print(f"    {off:6d}                    {r2}")
        # Rate over rung 2: rung 1 counts situations we may have had no
        # alternative to, which would flatter or damn the pilot for the board's
        # shape rather than its choices.
        print(f"    {bad:6d}  ({bad / (off or 1):5.1%} of those)  {r3}")
        if name == "over_attach" and c["over_attach.damage_forgone"]:
            print(f"           damage forgone {c['over_attach.damage_forgone']:6d}"
                  "  — hand-scaling actives only: each surplus attach spends a "
                  "card out of the damage formula")
        if name == "hand_scaler_blind":
            print(f"           of which we attacked anyway: "
                  f"{c['hand_scaler_blind.attacked_anyway']} — the M41 "
                  "617-of-617 signature (correct play through a dead feature)")
        if name == "stadium_ignored":
            print(f"           our own stadium already out (not a defect): "
                  f"{c['stadium_ignored.ours_already_out']}")
        if name == "dead_basis_fetch":
            print(f"           strict (0 bench AND 0 basics in hand): "
                  f"{c['dead_basis_fetch.strict_situation']} situations, "
                  f"{c['dead_basis_fetch.strict_chose_bad']} taken")
            print(f"           where taken: KEEP "
                  f"{c['dead_basis_fetch.in_KEEP']} (fetch_value RAN and was "
                  f"outbid) / PROMOTE {c['dead_basis_fetch.in_PROMOTE']} "
                  "(fetch_value never runs there)")
        if lad.detail[name]:
            for card, n in lad.detail[name].most_common(5):
                print(f"           {n:5d}  {card}")
        if sit == 0:
            dead.append(name)

    if c["same_id_offered"]:
        print(f"\n  same_id_offered        {c['same_id_offered']:6d}  — the "
              "engine DID offer a same-name stadium replacement; the "
              "rule-impossibility finding of 2026-08-04 no longer holds and "
              "the void ladder should be restored.")

    if c["foreign_target_opts"]:
        print(f"\n  foreign_target_opts    {c['foreign_target_opts']:6d}  "
              "— options addressing a board that is NOT ours through "
              "inPlayArea. rl/encoders.py:207 resolves these against "
              "your_index, so these are MIS-ENCODED today.")
    else:
        print("\n  foreign_target_opts         0  — no option addressed a "
              "foreign board through inPlayArea, so the encoders.py:207 "
              "target-resolution bug is latent, not live, on this pairing.")

    if dead:
        print("\n  SITUATION NEVER OCCURS: " + ", ".join(dead))
        print("  These ladders measured nothing. A fix for them cannot be "
              "credited on this pairing — run a deck where the board state "
              "exists, or drop the fix (M41 killed three levers on this "
              "reading).")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True, help="matchrunner spec for OUR seat")
    ap.add_argument("--opp", required=True)
    ap.add_argument("-n", "--games", type=int, default=20)
    ap.add_argument("--json", default=None, help="write raw counters here")
    args = ap.parse_args()

    lad = Ladders()
    fn_a, deck_a = make_pilot(parse_spec(args.arm), "probe_a")
    fn_b, deck_b = make_pilot(parse_spec(args.opp), "probe_b")
    fn_a = instrument(fn_a, lad)

    for g in range(args.games):
        if g % 2 == 0:                       # slot-fair: seat 0 carries ~61%
            _engine_game(fn_a, fn_b, deck_a, deck_b)
        else:
            _engine_game(fn_b, fn_a, deck_b, deck_a)
        lad.end_game()                       # resolve the per-TURN ladders

    rc = report(lad, args)
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"arm": args.arm, "opp": args.opp, "games": args.games,
             "counters": dict(lad.c),
             "detail": {k: dict(v) for k, v in lad.detail.items()}}, indent=2))
        print(f"\n  counters -> {args.json}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
