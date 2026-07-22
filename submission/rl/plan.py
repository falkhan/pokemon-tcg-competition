"""M11 plan primitives — encode / enumerate / derive a turn-scoped attack plan.

The plan is the OBSERVABLE version of buddy's hidden AttackPlan (docs/
buddy-analysis.md): a (attacker, target, attack, needs_attach) commitment made
at a turn's first MAIN prompt and fed to the policy as input features for the
rest of the turn. Making it an explicit input is what breaks the M1-M3/M9
hidden-plan imitation trap (labels stay consistent w.r.t. the inputs).

Bundle-pure on purpose (numpy + cg.api + rl.combat only — ships inside
submission/ like rl/combat.py). No torch, no polars.
"""
from collections import namedtuple

import numpy as np

from cg.api import (AreaType, CardType, OptionType, SelectContext,
                    all_card_data)
from rl.combat import (_ATK, _CARD, _best_damage, _can_afford,
                       _turns_to_first_ko, UNREACHABLE)

PLAN_DIM = 27
MAX_PLAN_CANDS = 48
MAX_ATTACK_IDX = 4                   # one-hot slots for attack position
_DMG_NORM = 340.0                    # encoders' damage normalization
GUST_IDS = frozenset({1182})         # Boss's Orders (U+2019 name) — card fact, id-pinned
_IS_ENERGY = {c.cardId for c in all_card_data()
              if c.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY)}

# attacker_slot/target_slot: 0 = active, 1..5 = bench index + 1.
# attack_idx: position in _CARD[attacker_id][3] (capped at MAX_ATTACK_IDX-1).
# damage is weakness/resistance-adjusted; risk fields are computed vs the
# opponent's ACTUAL board at plan time (buddy's prize-trade logic as features).
Plan = namedtuple("Plan", "attacker_slot target_slot attack_idx attack_id "
                          "needs_attach needs_gust damage target_prize "
                          "lethal wins attacker_prize return_ko concedes opp_ttk")


def _attack_damage(attacker, target, attack_id, extra_energy: int = 0,
                   board_ids=None) -> int:
    """Damage of ONE specific attack (affordability + weak/res — the
    per-attack twin of rl.combat._best_damage's inner loop). board_ids gates
    CONDITIONAL_ATTACKS (M13 0a)."""
    from rl.combat import _attack_available
    if attacker is None or target is None or attacker.id not in _CARD:
        return 0
    if attack_id not in _ATK or not _attack_available(attack_id, board_ids):
        return 0
    _, _, atk_type, _, _ = _CARD[attacker.id]
    energies = list(getattr(attacker, "energies", ())) + [atk_type] * extra_energy
    t_weak, t_res, _, _, _ = _CARD.get(target.id, (None, None, 0, [], 1))
    dmg, cost = _ATK[attack_id]
    if dmg <= 0 or not _can_afford(energies, cost):
        return 0
    if t_weak is not None and int(t_weak) == atk_type:
        dmg *= 2
    elif t_res is not None and int(t_res) == atk_type:
        dmg = max(0, dmg - 30)
    return dmg


def encode_plan(plan: Plan | None) -> np.ndarray:
    """(PLAN_DIM,) float32; all-zeros == "no plan" (the old-shard default)."""
    v = np.zeros(PLAN_DIM, dtype=np.float32)
    if plan is None:
        return v
    v[0] = 1.0
    v[1 + min(int(plan.attacker_slot), 5)] = 1.0
    v[7 + min(int(plan.target_slot), 5)] = 1.0
    v[13 + min(int(plan.attack_idx), MAX_ATTACK_IDX - 1)] = 1.0
    v[17] = float(plan.needs_attach)
    v[18] = float(plan.needs_gust)
    v[19] = min(float(plan.damage), _DMG_NORM) / _DMG_NORM
    v[20] = float(plan.target_prize) / 3.0
    v[21] = float(plan.lethal)
    v[22] = float(plan.wins)
    v[23] = float(plan.attacker_prize) / 3.0
    v[24] = float(plan.return_ko)
    v[25] = float(plan.concedes)
    v[26] = min(float(plan.opp_ttk), 4.0) / 4.0
    return v


def _board(player):
    """[(slot, pokemon)] — 0 = active, 1..5 = bench+1; None holes skipped."""
    slots = []
    active = player.active[0] if player.active and player.active[0] is not None else None
    if active is not None:
        slots.append((0, active))
    for i, p in enumerate(player.bench or []):
        if p is not None:
            slots.append((i + 1, p))
    return slots


def _make_plan(aslot, attacker, tslot, target, aidx, aid, needs_attach,
               me, op, op_active) -> Plan | None:
    my_ids = {p.id for _, p in _board(me)}
    dmg = _attack_damage(attacker, target, aid,
                         extra_energy=1 if needs_attach else 0,
                         board_ids=my_ids)
    if dmg <= 0:
        return None
    target_prize = _CARD.get(target.id, (0, 0, 0, [], 1))[4]
    attacker_prize = _CARD.get(attacker.id, (0, 0, 0, [], 1))[4]
    lethal = dmg >= (target.hp or 0)
    wins = lethal and target_prize >= len(op.prize)
    # Risk block: can the opponent's CURRENT active return-KO my attacker?
    # (+1 energy — assume they attach next turn; conservative like should_solve)
    ret_dmg = _best_damage(op_active, attacker, extra_energy=1)
    return_ko = ret_dmg >= (attacker.hp or 0) and not wins
    # len(me.prize) = prizes the OPPONENT still needs (they take them by
    # KOing my Pokémon) — losing this attacker hands them the game.
    concedes = return_ko and attacker_prize >= len(me.prize)
    opp_ttk = _turns_to_first_ko(op_active, attacker) if op_active is not None \
        else UNREACHABLE
    return Plan(aslot, tslot, aidx, aid, needs_attach, tslot > 0, dmg,
                target_prize, lethal, wins, attacker_prize, return_ko,
                concedes, min(opp_ttk, 4))


def enumerate_plans(obs) -> list:
    """[None] + feasible attack plans for the CURRENT observation, cap
    MAX_PLAN_CANDS. Candidate 0 (null plan = don't commit to an attack) is the
    learnable "don't sacrifice" action. Bench targets only when a gust trainer
    is in hand; needs_attach only when an energy card is in hand and none
    attached yet this turn."""
    st = obs.current
    me = st.players[st.yourIndex]
    op = st.players[1 - st.yourIndex]
    cands: list = [None]
    op_active = op.active[0] if op.active and op.active[0] is not None else None
    if op_active is None:
        return cands
    hand = [c for c in (me.hand or []) if c is not None]
    can_attach = (any(c.id in _IS_ENERGY for c in hand)
                  and not getattr(st, "energyAttached", False))
    gust = any(c.id in GUST_IDS for c in hand)
    targets = [(0, op_active)]
    if gust:
        targets += [(j + 1, p) for j, p in enumerate(op.bench or [])
                    if p is not None]
    my_ids = {p.id for _, p in _board(me)}
    for aslot, attacker in _board(me):
        if attacker.id not in _CARD:
            continue
        attacks = _CARD[attacker.id][3]
        for aidx, aid in enumerate(attacks[:MAX_ATTACK_IDX]):
            if aid not in _ATK:
                continue
            for tslot, target in targets:
                needs = False
                if _attack_damage(attacker, target, aid,
                                  board_ids=my_ids) <= 0:
                    if not can_attach:
                        continue
                    if _attack_damage(attacker, target, aid, extra_energy=1,
                                      board_ids=my_ids) <= 0:
                        continue
                    needs = True
                plan = _make_plan(aslot, attacker, tslot, target, aidx, aid,
                                  needs, me, op, op_active)
                if plan is not None:
                    cands.append(plan)
                    if len(cands) >= MAX_PLAN_CANDS:
                        return cands
    return cands


def _slot_of(player, pokemon) -> int | None:
    """Root-coordinate slot of an in-play pokemon, matched by identity then
    by (id, hp)."""
    for slot, p in _board(player):
        if p is pokemon:
            return slot
    for slot, p in _board(player):
        if p.id == pokemon.id and p.hp == pokemon.hp:
            return slot
    return None


def derive_plan(line: list, per_step_obs: list, root_obs) -> Plan | None:
    """Map a solver line back to a Plan in ROOT coordinates (the anti-aliasing
    contract: this plan conditions every row labeled from this line).

    Finds the first ATTACK step; attacker = the active at that step, mapped to
    its root slot; target = opponent active at that step mapped to the root
    board (a gust promoted it, so match root bench); needs_attach = any ATTACH
    onto the attacker earlier in the line. No ATTACK => null plan (None)."""
    st_root = root_obs.current
    me_root = st_root.players[st_root.yourIndex]
    op_root = st_root.players[1 - st_root.yourIndex]
    op_active_root = op_root.active[0] if op_root.active and \
        op_root.active[0] is not None else None

    atk_step = atk_opt = None
    for k, picks in enumerate(line):
        obs_k = per_step_obs[k]
        if obs_k.select is None:
            continue
        opt = obs_k.select.option[picks[0]]
        if opt.type == OptionType.ATTACK:
            atk_step, atk_opt = k, opt
            break
    if atk_step is None:
        return None

    obs_atk = per_step_obs[atk_step]
    st_atk = obs_atk.current
    me_atk = st_atk.players[st_atk.yourIndex]
    op_atk = st_atk.players[1 - st_atk.yourIndex]
    attacker = me_atk.active[0] if me_atk.active and \
        me_atk.active[0] is not None else None
    target = op_atk.active[0] if op_atk.active and \
        op_atk.active[0] is not None else None
    if attacker is None or target is None or attacker.id not in _CARD:
        return None

    aslot = _slot_of(me_root, attacker)
    if aslot is None:                 # evolved/promoted mid-line: match by id fails
        aslot = 0                     # attacker fights from active; default there
    tslot = _slot_of(op_root, target)
    if tslot is None:
        tslot = 0
    attacks = list(_CARD[attacker.id][3])
    aid = atk_opt.attackId
    aidx = attacks.index(aid) if aid in attacks else 0

    needs_attach = False
    for k in range(atk_step):
        obs_k = per_step_obs[k]
        if obs_k.select is None:
            continue
        opt = obs_k.select.option[line[k][0]]
        if opt.type == OptionType.ATTACH:
            needs_attach = True
            break

    return _make_plan(aslot, attacker, tslot, target, aidx, aid, needs_attach,
                      me_root, op_root, op_active_root)


def match_candidate(plan: Plan | None, cands: list) -> int:
    """Index of `plan` in an enumerate_plans list by (attacker_slot,
    target_slot, attack_idx); 0 for the null plan; -1 when enumeration missed
    it (skip plan loss, count toward the coverage metric)."""
    if plan is None:
        return 0
    for i, c in enumerate(cands):
        if c is None:
            continue
        if (c.attacker_slot == plan.attacker_slot
                and c.target_slot == plan.target_slot
                and c.attack_idx == plan.attack_idx):
            return i
    return -1


# --- M26 attach-override arms (docs/M26-plan.md Phase 4) --------------------
# Override law (M8.1/M12/M13): candidate arms, OFF unless a fix name is passed
# by an opted-in spec kind / build flag — never an unconditional default.
TELEPATH_ID = 19                  # Telepath Psychic Energy — card fact, id-pinned
ATTACH_FIX_TELEPATH = "telepath"  # O1: prefer Telepath on contested attaches
ATTACH_FIX_BACKSTOP = "backstop"  # O2: no turn ends with a legal attach unplayed
_TURN_ENDING = frozenset({OptionType.END, OptionType.ATTACK})


def _hand_card_id(opt, hand):
    """Acted card id of a hand-area option (the encoders' resolution order:
    explicit cardId first, then hand index; None-area options are hand plays)."""
    if opt.cardId:
        return opt.cardId
    if (opt.index is not None and opt.area in (AreaType.HAND, None)
            and opt.index < len(hand)):
        card = hand[opt.index]
        return card.id if card is not None else None
    return None


def apply_attach_overrides(obs, ranked: list, fixes: frozenset) -> list:
    """Reorder the model's MAIN-select option preference per the M26 arms.

    ranked: option indices, model's best first (full order). At most one
    energy-ATTACH index is moved to the front; everything else keeps the
    model's order (target choice inside ATTACH stays model-scored).
    - O1 `telepath`: a Telepath ATTACH is legal and the bench has space ->
      attach it now (the teacher's contested-attach preference).
    - O2 `backstop`: the model is about to end the turn (END, or ATTACK —
      attacking ends the turn) with the manual attach unused and a legal
      energy ATTACH on the menu -> attach first; the next MAIN re-offers
      the turn-ending action.
    Both fire only when this turn's manual energy attach is still unused.
    """
    if not fixes or obs.select is None or obs.current is None:
        return ranked
    st = obs.current
    if (obs.select.context != SelectContext.MAIN
            or getattr(st, "energyAttached", False)):
        return ranked
    me = st.players[st.yourIndex]
    hand = me.hand or []
    opts = obs.select.option
    attach_ranked = [i for i in ranked
                     if opts[i].type == OptionType.ATTACH
                     and _hand_card_id(opts[i], hand) in _IS_ENERGY]
    if not attach_ranked:
        return ranked
    pick = None
    if ATTACH_FIX_TELEPATH in fixes:
        bench = me.bench or []
        bench_space = (sum(p is not None for p in bench)
                       < getattr(me, "benchMax", 5))
        if bench_space:
            telepath = [i for i in attach_ranked
                        if _hand_card_id(opts[i], hand) == TELEPATH_ID]
            if telepath:
                pick = telepath[0]
    if (pick is None and ATTACH_FIX_BACKSTOP in fixes
            and opts[ranked[0]].type in _TURN_ENDING):
        pick = attach_ranked[0]
    if pick is None or pick == ranked[0]:
        return ranked
    return [pick] + [i for i in ranked if i != pick]
