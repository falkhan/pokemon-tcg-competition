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
_IS_BASIC_POKEMON = {c.cardId for c in all_card_data()
                     if c.cardType == CardType.POKEMON
                     and getattr(c, "basic", False)}

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
    # Prize semantics (M37 audit / M38 P2 ride-along, verified live twice):
    # a player's .prize is the prizes THAT player still has to TAKE, and it
    # drains for whoever scores the KO. My KO wins iff its prize haul
    # finishes MY remaining count; their return-KO concedes iff it finishes
    # THEIRS. (Pre-fix this used the opposite arrays; measured disagreement
    # was 0.0%/0.2% over 1,288 real candidates, hence the deferred fix.)
    wins = lethal and target_prize >= len(me.prize)
    # Risk block: can the opponent's CURRENT active return-KO my attacker?
    # (+1 energy — assume they attach next turn; conservative like should_solve)
    ret_dmg = _best_damage(op_active, attacker, extra_energy=1)
    return_ko = ret_dmg >= (attacker.hp or 0) and not wins
    concedes = return_ko and attacker_prize >= len(op.prize)
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


# --- M30 deck-economy override arms (docs/M30-plan.md) ----------------------
# Same override law as the M26 arms: deterministic card-fact rules, OFF unless
# a fix name is passed by an opted-in spec kind / build flag.
POFFIN_ID = 1086       # Buddy-Buddy Poffin — card fact, id-pinned
POKE_PAD_ID = 1152     # Poké Pad
SACRED_ASH_ID = 1129   # Sacred Ash: shuffles up to 5 discard Pokémon into deck
DUDUNSPARCE_IDS = frozenset({66})  # clone deck's only draw-ability body
FEZANDIPITI_ID = 140   # Fezandipiti ex: draw ability, ~2.8 deck cards/use
HILDA_ID = 1225        # Hilda: draw supporter, ~1.72 cards/use (m31 probe)
PLAY_FIX_TEMPO = "tempo"          # O3: no END while Poffin/Poké Pad playable
PLAY_FIX_DECKGUARD = "deckguard"  # O4: no Dudunsparce draw at deck <= 6
PLAY_FIX_ASH = "ash"              # O5: Sacred Ash when the deck runs low
PLAY_FIX_CONSERVE = "conserve"    # O6: no Fezandipiti draw at deck <= 6
PLAY_FIX_POFFINFLOOR = "poffinfloor"  # O7: dig for board when thin (m31)
PLAY_FIX_DRAWFLOOR = "drawfloor"      # O8: force Hilda when hand-starved (m31)
PLAY_FIX_BENCHFLOOR = "benchfloor"    # O10: bench a basic you HOLD when thin (m35)
PLAY_FIX_GUSTVETO = "gustveto"        # O11: no gust at opp-prizes <= 1 (m36)
PLAY_FIX_RACEMODE = "racemode"        # O12: conserve vs stall/grim, margin-gated (m37)
PLAY_FIX_RACEMODER = "racemoder"      # O12b: conserve vs stall/grim, blanket (m37)
PLAY_FIX_RACEMODE2 = "racemode2"      # O12c: blanket vs walls, margin vs pressure-stall (m37)
PLAY_FIX_RACEMODE3 = "racemode3"      # O12d: blanket vs walls ONLY (m37 final synthesis)
PLAY_FIX_RACEMODE4 = "racemode4"      # O13: demote OUR burn sources in a race (m39 P2b)
PLAY_FIX_RACEASH = "raceash"          # O13b: recycle Sacred Ash EARLY in a race (m39 P2b)
_TEMPO_ITEM_IDS = frozenset({POFFIN_ID, POKE_PAD_ID})
_DECKGUARD_AT = 6   # a use draws 3 (net -1); at <=3 it draws the deck to 0
_ASH_AT = 10
_CONSERVE_AT = 6    # measured 2.8 deck cards/use (m30 deck_drain probe);
_CONSERVE_ABILITY_IDS = frozenset({FEZANDIPITI_ID})
# O7/O8 (m31): fire only ABOVE the low-deck economy regime (deckguard/conserve
# <=6, ash <=10) so a board/hand refill never fights those rules or the
# deck-out endgame. bench<=1 is an early-game state (deck >> 10 — the live
# bench-outs were at deck 41/44), so the deck floor barely binds real firings.
_POFFINFLOOR_BENCH_AT = 1   # bench-alive <= 1: board is thin, dig for basics
_POFFINFLOOR_DECK_AT = 10   # ... and deck >= 10 (Poffin costs ~1.27 cards/use)
_BENCHFLOOR_BENCH_AT = 1    # O10 (m35): bench-alive <= 1 -> play a basic you HOLD.
# No deck floor: benching from hand costs 0 deck, so it never fights the low-deck
# economy rules. Ranked ABOVE poffinfloor ("bench what you have before digging"),
# which also fixes the M31 bench-0 missed-basic regression (0->12 prompts).
_DRAWFLOOR_HAND_AT = 4      # hand <= 4: starved, a draw actually helps (not
_DRAWFLOOR_DECK_AT = 10     # ... the full-hand hoarding cases); deck >= 10
# O11 (m36): a player's .prize list is the prizes THEY still need (verified
# empirically vs the diag end-states + a forced kyogre probe, docs/M36-plan.md
# execution log — NOT what the _make_plan comment below says). Opponent at
# match point = len(op.prize) <= 1.
_GUSTVETO_OPP_PRIZES_AT = 1
# O12 (m37): archetype-detected race mode. The 600-band stall/heal-tank lines
# farm us by deck-out (m36 post-mortem: 9/26 losses; hop/garchomp 0-4 live).
# Trigger = a stall-family Pokémon visible on the OPPONENT'S BOARD (public obs
# fact, same class as benchfloor's card facts). Grim ids are a SEPARATE set so
# the m37 pre-registered bar B3 can drop them with a one-line edit. Deliberately
# EXCLUDED: Munkidori 112/139 and Froslass 104 — splashable techs that would
# misfire on non-stall decks.
_RACEMODE_STALL_IDS = frozenset({
    58,              # Great Tusk
    344, 532,        # Dwebble (both printings)
    345, 533,        # Crustle (both printings)
    607,             # Terrakion
    878, 879,        # Hop's Phantump / Trevenant
    304,             # Hop's Snorlax
    379, 380, 381,   # Cynthia's Gible / Gabite / Garchomp ex
    341, 342,        # Cynthia's Roselia / Roserade
    387,             # Cynthia's Spiritomb
    # --- M39 P2a coverage additions, chosen from the harvest, not guessed.
    # 174 Fan Rotom sits in 7 of the 16 cached stall lists and is the family's
    # TURN-1 opener, so it fires the trigger several turns before Trevenant
    # lands. It also appears in 2 of 83 mirror lists — a real false positive,
    # which is why it goes in the PRESSURE (margin-gated) half below and not
    # the blanket wall half: a mirror game where we are >5 cards down on the
    # deck race is a game where conserving the draw is defensible anyway.
    174,             # Fan Rotom (stall opener; 2/83 mirror lists — margin-gated)
})
_RACEMODE_GRIM_IDS = frozenset({646, 647, 648})  # Marnie's Impidimp/Morgrem/Grimmsnarl ex
_RACEMODE_OPP_IDS = _RACEMODE_STALL_IDS | _RACEMODE_GRIM_IDS
# O12c (m37 battery finding): the v1 trigger conflated two archetypes that
# need OPPOSITE gates. The crustle/tusk wall family applies no prize pressure
# — blanket conserve there won the deck-out race outright (+14.8pp z+5.2);
# hop/garchomp/grim DO attack — blanket starves our setup (-12/-18pp), only
# the margin-gated conserve is safe. racemode2 splits the sets.
_RACEMODE_WALL_IDS = frozenset({
    58, 344, 532, 345, 533, 607,
    # M39 P2a: the BACKLOG's "Kanga-only wall blindspot", closed. 756 (Mega
    # Kangaskhan ex, 300 HP) is in 30 of the 65 cached wall lists — where the
    # Crustle ids already fire — AND in all 10 lists of the `kanga` family,
    # which classify as wall-alikes that run no Crustle at all and therefore
    # never triggered. It belongs in the BLANKET half because those lists
    # apply no prize pressure, which is the m37 law that splits these sets.
    # NOT added: 24 Team Rocket's Kangaskhan ex (1 rocket list — rocket
    # attacks, and it is the one bed where conserve measured negative) and
    # 472 plain Kangaskhan (zero cached lists).
    756,             # Mega Kangaskhan ex
})
_RACEMODE_PRESSURE_IDS = (_RACEMODE_STALL_IDS - _RACEMODE_WALL_IDS) \
    | _RACEMODE_GRIM_IDS
# racemode margin gate = the m36 parked raceconserve design (mirror_race_probe:
# behind by >5 on the deck race, below 25 so t3-t6 setup digs stay untouched,
# above 6 where deckguard/conserve already own the endgame).
_RACEMODE_MARGIN = 5
_RACEMODE_DECK_HI = 25
_RACEMODE_DECK_LO = 6
# O13 `racemode4` (m39 P2b). racemode/2/3 demote the two DRAW abilities, which
# the M37 post-mortem's burn audit then showed are not where our cards go:
# measured per-use deck cost is Enriching Energy 4.0/attach (5 attaches in a
# wall game = 20 cards), Rare Candy 3.0, Fezandipiti 2.8, Dawn 2.4, Hilda 2.0,
# Poke Pad 1.0 x19 plays -- against an opponent burning 1.4-1.5/turn while we
# burn 2.5-2.7. racemode4 extends the SAME trigger to those hand cards.
#
# What is deliberately NOT in the set, and why:
#   Rare Candy (1079) and the Alakazam/Kadabra EVOLVE options are the two
#   biggest single burns in the audit and they are our win condition. Demoting
#   them would trade the deck race for the game, which is the failure mode the
#   m37 blanket racemoder already produced against pressure decks (-12/-18pp).
#   Dawn/Hilda are demoted ONLY once the board is built, for the same reason.
ENRICHING_ENERGY_ID = 13    # ACE spec energy: 4.0 deck cards per attach (m37)
DAWN_ID = 1231              # draw supporter, 2.4 cards/play (m37 burn audit)
ALAKAZAM_ID = 743           # stage-2 win condition; its presence == "set up"
_RACEMODE4_ALWAYS_IDS = frozenset({ENRICHING_ENERGY_ID, POKE_PAD_ID})
_RACEMODE4_SETUP_IDS = frozenset({DAWN_ID, HILDA_ID})
_RACEMODE4_SETUP_POKEMON = frozenset({ALAKAZAM_ID})
# O13b `raceash` (m39 P2b, the Sacred Ash half). The audit's finding was the
# OPPOSITE of a demote: Sacred Ash was played at deck 0 and deck 2 in two M37
# losses -- "recycle value at the last possible moment" -- and the ranked
# recommendation is "don't sit on recycle value until deck<=2". So in a race
# the fix RAISES the existing `ash` promote floor (deck <= 10) to deck <= 20,
# rather than demoting anything. Kept a separate fix name from racemode4
# because it is a promote with the opposite sign, and bundling two opposite
# effects behind one name is how a gate cell stops being attributable.
_RACEASH_AT = 20


def _board_pokemon_id(opt, me):
    """Card id of the board Pokémon a board-area option (ABILITY) acts on.
    ABILITY options carry only area+index, never cardId (M30 P0 probe)."""
    if opt.area == AreaType.ACTIVE:
        return me.active[0].id if me.active and me.active[0] else None
    if (opt.area == AreaType.BENCH and opt.index is not None
            and me.bench and opt.index < len(me.bench)
            and me.bench[opt.index] is not None):
        return me.bench[opt.index].id
    return None


def _opp_board_ids(op) -> frozenset:
    """Card ids of every Pokémon visible on a player's board (active + bench).
    Public obs facts — the O12 archetype trigger reads the OPPONENT'S board."""
    ids = set()
    for p in (op.active or []):
        if p is not None:
            ids.add(p.id)
    for p in (op.bench or []):
        if p is not None:
            ids.add(p.id)
    return frozenset(ids)


def _own_board_ids(me) -> frozenset:
    """The same public-board card-id fact as `_opp_board_ids`, read on OUR
    side. M39 P2b needs it for racemode4's "once the board is built" gate."""
    return _opp_board_ids(me)


def _racemode_engaged(st, me) -> bool:
    """The m37 `racemode2` trigger, extracted so the M39 P2b rules cannot
    drift away from it: a WALL-family Pokemon on the opponent's board fires
    blanket (those lists apply no prize pressure — the m37 law), while a
    PRESSURE-stall Pokemon fires only when we are losing the deck race by
    more than _RACEMODE_MARGIN inside the setup window (blanket there starves
    our own setup: measured −12/−18pp)."""
    op = st.players[1 - st.yourIndex]
    opp_ids = _opp_board_ids(op)
    if opp_ids & _RACEMODE_WALL_IDS:
        return True
    return bool(opp_ids & _RACEMODE_PRESSURE_IDS
                and me.deckCount < op.deckCount - _RACEMODE_MARGIN
                and _RACEMODE_DECK_LO < me.deckCount <= _RACEMODE_DECK_HI)


def apply_play_overrides(obs, ranked: list, fixes: frozenset) -> list:
    """Reorder MAIN-select preference per the M30/M31 economy arms.

    Runs AFTER apply_attach_overrides. Precedence note (m31 O9, docstring
    corrected): a PROMOTE rule moves a PLAY to the front and therefore CAN
    displace an O1-promoted ATTACH. This is intended and benign — a deck
    refill (ash), board dig (poffinfloor) or hand refill (drawfloor) outranks
    the energy attach at its trigger state, and the turn's manual attach is
    still unused, so the next MAIN re-offers the ATTACH and O1 re-promotes it.
    Only O3 tempo is additionally gated on `ranked[0]` being END, so tempo
    alone cannot override O1. Promote order (first match wins): ash >
    benchfloor > poffinfloor > drawfloor > tempo. The DEMOTE rules (deckguard,
    conserve) only reorder when the top pick is the targeted ABILITY.
    - O5 `ash`: own deck <= 10 and a Sacred Ash PLAY is legal (engine
      legality implies Pokémon in the discard) -> play it now.
    - O10 `benchfloor` (m35): bench-alive <= 1 and a basic-Pokemon PLAY is
      legal -> play the highest-ranked such basic (bench what you HOLD). No
      deck floor (benching costs 0 deck). Ranked above poffinfloor so we bench
      a held basic before digging for one — which also fixes the M31 bench-0
      missed-basic regression. Targets the ~39% basic-play-at-bench<=1 defect.
    - O7 `poffinfloor`: bench-alive <= 1 and own deck >= 10 -> play
      Buddy-Buddy Poffin (benches up to 2 basics from deck, ~1.27 cards/use
      — m31 probe) to dig for board when thin. Deck-gated above the low-deck
      economy regime so it never fights ash/deckguard/conserve.
    - O8 `drawfloor`: hand-size <= 4 and own deck >= 10 and Hilda is legal
      -> play it. Targets the supporter under-play only where a draw helps
      (thin hand, not full-hand hoarding); deck floor keeps it clear of the
      deck-out zone O6 conserve protects.
    - O3 `tempo`: the model is about to END with a Poffin/Poké Pad PLAY on
      the menu -> play the highest-ranked such item; END re-offers next MAIN.
    - O4 `deckguard`: own deck <= 6 and the top pick is a Dudunsparce
      ABILITY (draw 3, net deck -1) -> demote every such ability below the
      rest of the model's order.
    - O6 `conserve`: same demotion for the Fezandipiti ex draw ability
      (measured ~2.8 deck cards/use; the only optional draw that still
      fires at low deck once deckguard is on — m30 deck_drain probe).
    - O11 `gustveto` (m36): opponent needs <= 1 prize and the top pick is a
      Boss's Orders PLAY -> demote every such PLAY below the rest. Burning
      the turn's supporter on a gust that may not convert while the opponent
      is at match point fired 5/5 in M35 live losses. Blanket demote (no
      KO-gate): _attack_damage models Powerful Hand as 0, so a KO-gate
      would be blind to our main attacker (M36 W2 design note).
    - O12 `racemode` / `racemoder` (m37): a stall/grim-family Pokémon is
      visible on the OPPONENT'S board -> extend the deckguard/conserve
      demotes (Dudunsparce + Fezandipiti draws) beyond the deck <= 6 floor:
      the 600-band stall lines win by deck-out, so the objective flips from
      tempo to card-economy racing (we win deck-out races when the economy
      rules engage — 4 live M36 wins by opp-deck-0). `racemode` adds the
      m36 raceconserve margin gate (behind by > _RACEMODE_MARGIN on the
      deck race, 6 < deck <= 25) so early setup digs stay untouched;
      `racemoder` is the blanket variant (trigger only) — the m37 battery
      picks between them (bar B2).
    - O12c `racemode2` (m37, the battery synthesis): the wall family
      (crustle/tusk — no prize pressure) gets the BLANKET demote (+14.8pp
      z+5.2 measured); the pressure-stall families (hop/garchomp/grim —
      they attack while stalling) get only the MARGIN-gated demote (the
      blanket regressed them −12/−18pp by starving setup).
    - O12d `racemode3` (m37 final): the wall-family blanket demote ONLY —
      the margin-gated pressure branch measured neutral-to-negative on
      hop (−1.6pp) and garchomp (5-seed z −2.13, pre-registered kill), so
      the rule keeps just the measured win: +15.7pp on the wall bed,
      provably inert against every deck with no wall-family Pokémon.
    - O13 `racemode4` (m39 P2b): same trigger as racemode2, applied to the
      burn sources the M37 audit actually measured rather than to the two
      draw abilities. Demote ATTACH Enriching Energy (4.0 deck cards/attach)
      and PLAY Poké Pad always while the race is on; demote PLAY Dawn/Hilda
      as well ONCE ALAKAZAM IS ON OUR BOARD, i.e. once the draw is surplus.
      Rare Candy and the evolution digs are excluded by design — they are
      the win condition, and trading them for the deck race is the failure
      the m37 blanket variant already produced against pressure decks.
    - O13b `raceash` (m39 P2b, opposite sign, separate name): in a race,
      raise the `ash` promote floor from deck <= 10 to deck <= 20. Sacred
      Ash was played at deck 0 and deck 2 in two M37 losses — recycle value
      sat on until the last possible moment. Independent of `ash`: with
      neither name present the Sacred Ash promote is off entirely.
    """
    if (not fixes or obs.select is None or obs.current is None
            or obs.select.context != SelectContext.MAIN):
        return ranked
    st = obs.current
    me = st.players[st.yourIndex]
    hand = me.hand or []
    opts = obs.select.option
    pick = None
    ash_at = _ASH_AT if PLAY_FIX_ASH in fixes else None
    if PLAY_FIX_RACEASH in fixes and _racemode_engaged(st, me):
        ash_at = max(ash_at or 0, _RACEASH_AT)
    if ash_at is not None and me.deckCount <= ash_at:
        ash = [i for i in ranked
               if opts[i].type == OptionType.PLAY
               and _hand_card_id(opts[i], hand) == SACRED_ASH_ID]
        if ash:
            pick = ash[0]
    if (pick is None and PLAY_FIX_BENCHFLOOR in fixes
            and sum(p is not None for p in (me.bench or []))
            <= _BENCHFLOOR_BENCH_AT):
        basics = [i for i in ranked
                  if opts[i].type == OptionType.PLAY
                  and _hand_card_id(opts[i], hand) in _IS_BASIC_POKEMON]
        if basics:
            pick = basics[0]
    if (pick is None and PLAY_FIX_POFFINFLOOR in fixes
            and me.deckCount >= _POFFINFLOOR_DECK_AT
            and sum(p is not None for p in (me.bench or []))
            <= _POFFINFLOOR_BENCH_AT):
        poffin = [i for i in ranked
                  if opts[i].type == OptionType.PLAY
                  and _hand_card_id(opts[i], hand) == POFFIN_ID]
        if poffin:
            pick = poffin[0]
    if (pick is None and PLAY_FIX_DRAWFLOOR in fixes
            and me.deckCount >= _DRAWFLOOR_DECK_AT
            and len(hand) <= _DRAWFLOOR_HAND_AT):
        hilda = [i for i in ranked
                 if opts[i].type == OptionType.PLAY
                 and _hand_card_id(opts[i], hand) == HILDA_ID]
        if hilda:
            pick = hilda[0]
    if (pick is None and PLAY_FIX_TEMPO in fixes
            and opts[ranked[0]].type == OptionType.END):
        tempo = [i for i in ranked
                 if opts[i].type == OptionType.PLAY
                 and _hand_card_id(opts[i], hand) in _TEMPO_ITEM_IDS]
        if tempo:
            pick = tempo[0]
    if pick is not None and pick != ranked[0]:
        return [pick] + [i for i in ranked if i != pick]
    demote_ids = frozenset()
    if PLAY_FIX_DECKGUARD in fixes and me.deckCount <= _DECKGUARD_AT:
        demote_ids |= DUDUNSPARCE_IDS
    if PLAY_FIX_CONSERVE in fixes and me.deckCount <= _CONSERVE_AT:
        demote_ids |= _CONSERVE_ABILITY_IDS
    if PLAY_FIX_RACEMODE in fixes or PLAY_FIX_RACEMODER in fixes:
        op = st.players[1 - st.yourIndex]
        if _opp_board_ids(op) & _RACEMODE_OPP_IDS and (
                PLAY_FIX_RACEMODER in fixes
                or (me.deckCount < op.deckCount - _RACEMODE_MARGIN
                    and _RACEMODE_DECK_LO < me.deckCount <= _RACEMODE_DECK_HI)):
            demote_ids |= DUDUNSPARCE_IDS | _CONSERVE_ABILITY_IDS
    if PLAY_FIX_RACEMODE2 in fixes and _racemode_engaged(st, me):
        demote_ids |= DUDUNSPARCE_IDS | _CONSERVE_ABILITY_IDS
    if (PLAY_FIX_RACEMODE3 in fixes
            and _opp_board_ids(st.players[1 - st.yourIndex])
            & _RACEMODE_WALL_IDS):
        demote_ids |= DUDUNSPARCE_IDS | _CONSERVE_ABILITY_IDS
    demote_hand_ids = frozenset()
    if PLAY_FIX_RACEMODE4 in fixes and _racemode_engaged(st, me):
        demote_hand_ids = _RACEMODE4_ALWAYS_IDS
        if _own_board_ids(me) & _RACEMODE4_SETUP_POKEMON:
            demote_hand_ids |= _RACEMODE4_SETUP_IDS

    def _demoted(i) -> bool:
        opt = opts[i]
        if opt.type == OptionType.ABILITY:
            return _board_pokemon_id(opt, me) in demote_ids
        if opt.type in (OptionType.PLAY, OptionType.ATTACH):
            return _hand_card_id(opt, hand) in demote_hand_ids
        return False

    if (demote_ids or demote_hand_ids) and _demoted(ranked[0]):
        keep = [i for i in ranked if not _demoted(i)]
        if keep:
            return keep + [i for i in ranked if i not in keep]
    if (PLAY_FIX_GUSTVETO in fixes
            and opts[ranked[0]].type == OptionType.PLAY
            and _hand_card_id(opts[ranked[0]], hand) in GUST_IDS
            and len(st.players[1 - st.yourIndex].prize or ())
            <= _GUSTVETO_OPP_PRIZES_AT):
        keep = [i for i in ranked
                if not (opts[i].type == OptionType.PLAY
                        and _hand_card_id(opts[i], hand) in GUST_IDS)]
        if keep:
            return keep + [i for i in ranked if i not in keep]
    return ranked
