"""Observation / option encoders: game objects -> numpy vectors.

numpy-only on purpose — the Kaggle submission ships these encoders plus .npz
weights and never imports torch (see ARCHITECTURE.md §7.1).

Feature source: data/cards_features.parquet, produced by deck_analysis.ipynb.

Stage B enrichment (docs/M1-plan.md §B1 + the aliasing diagnosis from Stage A):
- options now encode the action's TARGET (which Pokémon an ATTACH/EVOLVE acts on) —
  in Stage A, "attach to Mega Lucario" and "attach to a benchwarmer" were identical
  vectors (337/500 menus had aliased options)
- state now sees: both discard piles, own discarded Basic Fighting Energy count
  (Mega Lucario's attack scales on it), per-slot energy TYPE counts, attached
  tools, status conditions, and the stadium
"""
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Card ids are 1..1267; row 0 of the matrix is an all-zeros "no card" padding row.
_PARQUET = DATA_DIR / "cards_features.parquet"
if _PARQUET.exists():
    import polars as pl

    _cards = pl.read_parquet(_PARQUET)
    _numeric = _cards.drop(["card_id", "name"]).cast(pl.Float32)
    FEAT_DIM = _numeric.width
    FEAT = np.zeros((_cards["card_id"].max() + 1, FEAT_DIM), dtype=np.float32)
    FEAT[_cards["card_id"].to_numpy()] = _numeric.to_numpy()
else:
    # Submission bundle (M7.5): Kaggle has neither data/ nor polars — the v2
    # neural bundle ships this ACTUAL module (no hand-copy) plus the exported
    # feature matrix next to it (written by tcg.shipping export).
    FEAT = np.load(Path(__file__).with_name("card_features.npy"))
    FEAT_DIM = FEAT.shape[1]

N_BENCH = 5          # max bench slots we encode
N_CONTEXTS = 64      # SelectContext one-hot size (49 defined today; head-room for new ones)
N_OPTION_TYPES = 17  # OptionType one-hot size
N_ENERGY = 12        # EnergyType enum size (Colorless..Team_Rocket)
N_STATUS = 5         # poisoned/burned/asleep/paralyzed/confused

BASIC_FIGHTING_ENERGY = 6  # card id; teacher's Mega Brave scales on discarded copies

# NOTE: a per-card-id "revealed opponent cards" feature was tried and reverted — a probe
# showed the opponent archetype is ALREADY ~98% identifiable from the pooled features below,
# so it was redundant and only added overfitting. See docs/DECISIONS.md 2026-07-08.

# --- Combat-lookahead tables (from the engine; available in training AND submission) ---
# The rule experts reason with hidden-state combat math (damage / KO / prize race) that BC
# couldn't imitate from raw board features. We compute the SAME quantities and expose them.
# The core lives in rl.combat (pure-Python, polars-free) so the rule submission can ship it;
# re-exported here so existing `from rl.encoders import _CARD, _best_damage` imports still work.
from rl.combat import (COLORLESS, _ATK, _CARD, _attack_available,  # noqa: E402,F401
                       _can_afford, _best_damage, _turns_to_ready, UNREACHABLE)
N_COMBAT = 11        # combat-lookahead features (see _combat_features)

# Per-Pokémon-slot: card features + hp/maxHp/energy-count + energy-type counts + tools pool
SLOT_DIM = FEAT_DIM + 3 + N_ENERGY + FEAT_DIM
# globals + my hand pool + 2 discard pools + my discarded-fighting-energy count
# + status x2 + stadium + combat features + 6 of my slots + 6 opponent slots
STATE_DIM = (7 + FEAT_DIM + 2 * FEAT_DIM + 1 + 2 * N_STATUS + FEAT_DIM + N_COMBAT
             + 2 * (1 + N_BENCH) * SLOT_DIM)
# Where the M3 combat block sits inside encode_state's output — pre-M3
# checkpoints (bc_v1) were trained without it, and slicing this range out
# reconstructs their input encoding (see matchrunner's model loader).
COMBAT_START = 7 + 4 * FEAT_DIM + 1 + 2 * N_STATUS
COMBAT_SLICE = (COMBAT_START, COMBAT_START + N_COMBAT)
# option-type one-hot + acted card features + TARGET card features + target-is-active flag
OPTION_DIM = N_OPTION_TYPES + FEAT_DIM + FEAT_DIM + 1


def _combat_features(state) -> np.ndarray:
    """The tactical quantities the rule experts reason over (damage / KO / prize race)."""
    me = state.players[state.yourIndex]
    op = state.players[1 - state.yourIndex]
    my_act = me.active[0] if me.active else None
    op_act = op.active[0] if op.active else None

    my_dmg = _best_damage(my_act, op_act)
    op_dmg = _best_damage(op_act, my_act)
    my_ko = float(op_act is not None and my_dmg >= op_act.hp)
    op_ko = float(my_act is not None and op_dmg >= my_act.hp)
    one_from = float(op_act is not None and not my_ko
                     and _best_damage(my_act, op_act, extra_energy=1) >= op_act.hp)
    my_can_attack = float(_best_damage(my_act, op_act) > 0 or
                          (my_act is not None and any(
                              _ATK.get(a, (0, ()))[0] > 0 and _can_afford(my_act.energies, _ATK[a][1])
                              for a in _CARD.get(my_act.id, (None, None, 0, [], 1))[3] if a in _ATK)))
    op_prize = _CARD.get(op_act.id, (None, None, 0, [], 1))[4] / 3.0 if op_act else 0.0
    n_ready = sum(1 for p in ([my_act] + list(me.bench))
                  if p is not None and any(
                      a in _ATK and _ATK[a][0] > 0 and _can_afford(p.energies, _ATK[a][1])
                      for a in _CARD.get(p.id, (None, None, 0, [], 1))[3]))

    return np.array([
        my_dmg / 340.0, my_ko, my_can_attack, one_from,
        op_dmg / 340.0, op_ko,
        (len(me.prize) - len(op.prize)) / 6.0,   # prize race (negative = I'm ahead)
        op_prize,
        n_ready / 6.0,
        (my_act.hp / max(1, my_act.maxHp)) if my_act else 0.0,
        (op_act.hp / max(1, op_act.maxHp)) if op_act else 0.0,
    ], dtype=np.float32)


def _pool(cards, scale: float = 1.0) -> np.ndarray:
    """Sum of card features over a list of Card objects (zeros when empty)."""
    if not cards:
        return np.zeros(FEAT_DIM, dtype=np.float32)
    return FEAT[[c.id for c in cards if c is not None]].sum(axis=0) * scale


def _poke_vec(p) -> np.ndarray:
    """Pokemon | None -> SLOT_DIM vector (zeros for an empty/facedown slot)."""
    v = np.zeros(SLOT_DIM, dtype=np.float32)
    if p is None:
        return v
    v[:FEAT_DIM] = FEAT[p.id]
    base = FEAT_DIM
    v[base:base + 3] = (p.hp / 340.0, p.maxHp / 340.0, len(p.energies) / 5.0)
    base += 3
    for e in p.energies:                       # energy TYPE counts, not just the total
        v[base + int(e)] += 1.0 / 3.0
    base += N_ENERGY
    v[base:base + FEAT_DIM] = _pool(p.tools)   # attached tools (Hero Cape etc.)
    return v


def encode_state(state) -> np.ndarray:
    """cg.api.State -> STATE_DIM float32 vector."""
    me = state.players[state.yourIndex]
    op = state.players[1 - state.yourIndex]

    glob = np.array([
        state.turn / 30.0,
        len(me.prize) / 6.0,
        len(op.prize) / 6.0,
        me.deckCount / 60.0,
        op.handCount / 15.0,
        float(state.energyAttached),
        float(state.supporterPlayed),
    ], dtype=np.float32)

    hand = _pool(me.hand)
    my_discard = _pool(me.discard, scale=0.1)
    op_discard = _pool(op.discard, scale=0.1)
    fighting_in_discard = np.array(
        [sum(1 for c in me.discard if c.id == BASIC_FIGHTING_ENERGY) / 10.0],
        dtype=np.float32)

    def status(ps):
        return np.array([ps.poisoned, ps.burned, ps.asleep, ps.paralyzed, ps.confused],
                        dtype=np.float32)

    stadium = _pool(state.stadium)

    def slots(ps):
        active = ps.active[0] if ps.active else None
        bench = list(ps.bench)[:N_BENCH]
        bench += [None] * (N_BENCH - len(bench))
        return [active] + bench

    parts = [glob, hand, my_discard, op_discard, fighting_in_discard,
             status(me), status(op), stadium, _combat_features(state)]
    parts += [_poke_vec(p) for p in slots(me)]
    parts += [_poke_vec(p) for p in slots(op)]
    return np.concatenate(parts)


def _card_id_at(obs, area, index, player_index) -> int | None:
    """Card id at (area, index) for the given player, or None."""
    state = obs.current
    ps = state.players[player_index]
    zone = {  # AreaType values, see cg/api.py
        1: obs.select.deck, 2: ps.hand, 3: ps.discard, 4: ps.active,
        5: ps.bench, 6: ps.prize, 7: state.stadium, 12: state.looking,
    }.get(int(area))
    try:
        card = zone[index]
        return card.id if card is not None else None
    except (TypeError, IndexError):
        return None


def encode_option(opt, obs) -> np.ndarray:
    """cg.api.Option -> OPTION_DIM float32 vector.

    [OptionType one-hot | features of the acted card | features of the TARGET
    Pokémon (for ATTACH/EVOLVE: what it's attached to / evolves onto) | target-is-active]
    """
    v = np.zeros(OPTION_DIM, dtype=np.float32)
    v[int(opt.type)] = 1.0
    your_index = obs.current.yourIndex

    card_id = opt.cardId
    if card_id is None and opt.index is not None and opt.area is not None:
        player = opt.playerIndex if opt.playerIndex is not None else your_index
        card_id = _card_id_at(obs, opt.area, opt.index, player)
    if card_id:
        v[N_OPTION_TYPES:N_OPTION_TYPES + FEAT_DIM] = FEAT[card_id]

    # The action's target: which Pokémon this ATTACH/EVOLVE acts on.
    if opt.inPlayArea is not None and opt.inPlayIndex is not None:
        target_id = _card_id_at(obs, opt.inPlayArea, opt.inPlayIndex, your_index)
        if target_id:
            base = N_OPTION_TYPES + FEAT_DIM
            v[base:base + FEAT_DIM] = FEAT[target_id]
        v[-1] = float(int(opt.inPlayArea) == 4)  # AreaType.ACTIVE
    return v


def encode_context(context) -> np.ndarray:
    """SelectContext -> N_CONTEXTS one-hot."""
    v = np.zeros(N_CONTEXTS, dtype=np.float32)
    v[int(context)] = 1.0
    return v


# --- Encoders v2 (M7.3) — side-by-side with v1; v1 stays byte-identical -----
# v2 returns (numeric, ids): the numeric vector extends v1 with the k-turn race
# block (deferred from M7.2b) and two deck-context pools, while the id vector
# carries card ids at fixed sites for LEARNABLE embeddings — the 36 features
# are nearly blind for trainers, embeddings let a multi-deck pilot learn
# per-card behavior from data (M7-plan §3.1b). Id 0 = "no card" padding,
# aligned with FEAT row 0.
EMBED_DIM = 16
N_CARD_IDS = FEAT.shape[0]          # 1268: ids 1..1267 + padding row 0
N_STATE_IDS = 2 * (1 + N_BENCH)     # my/opp active + bench card ids
N_OPTION_IDS = 2                    # acted card + target card
N_RACE = 8                          # see _race_features
RACE_TURN_CAP = 10.0                # race turns normalized /10, capped
STATE_V2_DIM = STATE_DIM + N_RACE + 2 * FEAT_DIM
OPTION_V2_DIM = OPTION_DIM
# --- M16 option-identity block (appended; the v1 slice [:OPTION_DIM] is
# byte-identical). Pre-M16, a PLAY option carried only a hand index — every
# trainer in hand encoded to the SAME vector (the M1 Stage-A aliasing disease
# on the other half of the menu: 431/1353 live decision states offered >=2
# indistinguishable trainers) — and an ATTACK option never encoded its
# attackId, so same-Pokémon attacks aliased too. encode_option_v2 now
# resolves the PLAY card into the acted-card FEAT block + embedding id and
# appends this block; encode_option_v2_legacy keeps the exact pre-M16
# encoding for pinned checkpoints.
N_OPTION_EXTRA = 4   # [atk dmg/300, atk cost/5, atk eff-dmg vs opp active/300, number/10]
OPTION_V3_DIM = OPTION_DIM + N_OPTION_EXTRA
_OT_PLAY = 7         # OptionType.PLAY
_OT_ATTACH = 8       # OptionType.ATTACH
_OT_RETREAT = 12     # OptionType.RETREAT
_OT_ATTACK = 13      # OptionType.ATTACK
_OT_END = 14         # OptionType.END
_OT_ABILITY = 10     # OptionType.ABILITY
_AREA_HAND = 2       # AreaType.HAND

# --- M27 play-precondition block (appended; the [:OPTION_V3_DIM] slice stays
# byte-identical, so pinned OPTION_V3_DIM checkpoints consume the same numbers
# after truncation — the M16 append-and-slice pattern, third use).
#
# Why: docs/M27.md Probe 3 — on IDENTICAL menus the clone plays supporters at
# 0.51x and stadiums at 0.06x the teacher's rate while matching items. The
# 0.505 sample agent (docs/M27-sample-agent-diff.md) expresses these as binary
# PRECONDITIONS, not preferences: Boss's Orders iff the plan wants a bench
# target, Gravity Mountain iff no stadium is in play. Our net had to infer them
# through a late-fusion MLP from 166 supporter / 18 stadium positive rows, with
# `supporterPlayed` buried as 1 of 1708 state scalars and NO option-level
# interaction term.
#
# These are INPUTS, not overrides — the override law's five kills are all
# play-time rule consumers (ARCHITECTURE.md). Note the gust term is computed
# from the observation, NOT from plan.needs_gust: Probe 4 showed the plan block
# is inert in the replay-BC lineage (the corpus carries no plans, so the plan
# is identically zero on every training row and no gradient can teach the net
# to use it).
N_OPTION_PLAY_PRE = 3
OPTION_M27_DIM = OPTION_V3_DIM + N_OPTION_PLAY_PRE

# --- M28 phase-interaction block (appended; [:OPTION_M27_DIM] byte-identical).
# docs/M27.md Probe 2: val_acc is 0.738 at deck 30+ but 0.601 at deck 7-15 —
# competence collapses exactly where deck-out is decided. The net's only phase
# signal is `turn/30` and `deckCount/60`, 2 raw scalars among 1708, and the
# two-tower architecture only combines state and option at the score head — so
# an option-type x phase INTERACTION is expensive for it to form and cheap for
# us to hand it. A per-state constant would be useless here (the score head
# already sees the whole state tower); these vary per option AND per state.
N_OPTION_PHASE = 3
OPTION_M28_DIM = OPTION_M27_DIM + N_OPTION_PHASE
DECK_LOW_AT = 15.0   # remaining-deck count at which "late game" reaches full weight

# --- M41 energy-ceiling block (appended; [:OPTION_M28_DIM] byte-identical, the
# fourth use of append-and-slice). Every consumer truncates to its own trained
# width, so the live bundles ignore these columns entirely — this is capability
# for the next collect, NOT a change to any shipped policy.
#
# Why (docs/M41.md, 2026-08-03). The M19 block above is the anti-over-attach
# feature, and two of its three slots are CONSTANTS on our own win condition:
# Alakazam #743's only attack prints 0 damage, so _charged_best drops it,
# _turns_to_ready returns UNREACHABLE at every energy count, and the option
# vector reports "gap 1.0, not saturated" on every energy card forever. The
# shipped clone therefore attaches onto an already-charged Alakazam 33.7% of the
# time while its own BC corpus does it 4.7% and rival ladder pilots 5.4% — a 7x
# amplification of a defect the training labels do not contain. The labels were
# never the problem; the net cannot represent a distinction its inputs hold
# constant.
#
# Slot 0 is the load-bearing one and is deliberately damage-FREE (see
# combat.energy_is_dead — two damage-based drafts were falsified on live
# replays). Slot 1 is the M19 gap recomputed with scaling damage credited, so
# the two sit side by side and the net can learn which to trust.
N_OPTION_ENERGY = 4
OPTION_M41_DIM = OPTION_M28_DIM + N_OPTION_ENERGY


def _race_features(state) -> np.ndarray:
    """k-turn prize-race block on the M7.2b combat primitives — the M3 combat
    features are 1-turn only; these are turns-to-ready / turns-to-first-KO for
    both boards plus the race delta (M7-plan §3b L1)."""
    from rl.combat import _turns_to_first_ko, _turns_to_ready

    me = state.players[state.yourIndex]
    op = state.players[1 - state.yourIndex]

    def board(ps):
        active = ps.active[0] if ps.active else None
        return [p for p in [active] + list(ps.bench) if p is not None], active

    my_board, my_active = board(me)
    op_board, op_active = board(op)

    def norm(turns) -> float:
        return min(float(turns), RACE_TURN_CAP) / RACE_TURN_CAP

    my_ttfk = min((_turns_to_first_ko(p, op_active) for p in my_board),
                  default=RACE_TURN_CAP) if op_active is not None else RACE_TURN_CAP
    op_ttfk = min((_turns_to_first_ko(p, my_active) for p in op_board),
                  default=RACE_TURN_CAP) if my_active is not None else RACE_TURN_CAP
    return np.array([
        norm(_turns_to_ready(my_active, op_active)) if my_active else 1.0,
        norm(_turns_to_first_ko(my_active, op_active)) if my_active and op_active else 1.0,
        norm(my_ttfk),
        sum(1 for p in my_board if _turns_to_ready(p, op_active) == 0) / 6.0,
        norm(_turns_to_ready(op_active, my_active)) if op_active else 1.0,
        norm(_turns_to_first_ko(op_active, my_active)) if op_active and my_active else 1.0,
        norm(op_ttfk),
        (min(float(op_ttfk), RACE_TURN_CAP) - min(float(my_ttfk), RACE_TURN_CAP))
        / RACE_TURN_CAP,                                    # >0: I win the race
    ], dtype=np.float32)


def _deck_pools(state, my_deck: list[int]) -> np.ndarray:
    """Deck-context pools (M7-plan §3.1b): FEAT sums of my FULL 60-card list and
    of my REMAINING deck (list minus hand/board/discard — all observable; the 6
    prized cards stay in "remaining" since which ones is hidden). Pooled, NOT a
    per-id count vector — the reverted-feature lesson. Discard-pool 0.1 scale."""
    from collections import Counter
    me = state.players[state.yourIndex]
    remaining = Counter(my_deck)
    seen = [c.id for c in list(me.hand) + list(me.discard) if c is not None]
    for p in [me.active[0] if me.active else None] + list(me.bench):
        if p is not None:
            seen.append(p.id)
            seen.extend(t.id for t in p.tools if t is not None)
    for cid in seen:
        if remaining[cid] > 0:
            remaining[cid] -= 1
    full = FEAT[list(my_deck)].sum(axis=0) * 0.1
    rest = (sum((FEAT[cid] * n for cid, n in remaining.items() if n > 0),
                np.zeros(FEAT_DIM)) * 0.1)
    return np.concatenate([full, rest]).astype(np.float32)


def _board_ids(state) -> np.ndarray:
    """Card ids of the 12 board slots (my/opp active + bench), 0-padded."""
    ids = []
    for ps in (state.players[state.yourIndex], state.players[1 - state.yourIndex]):
        active = ps.active[0] if ps.active else None
        bench = list(ps.bench)[:N_BENCH]
        bench += [None] * (N_BENCH - len(bench))
        ids += [p.id if p is not None else 0 for p in [active] + bench]
    return np.array(ids, dtype=np.int32)


def encode_state_v2(state, my_deck: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """(numeric STATE_V2_DIM f32, board ids N_STATE_IDS i32). Needs my 60-card
    list — observations don't carry it; the pilot closes over its own deck."""
    numeric = np.concatenate([encode_state(state), _race_features(state),
                              _deck_pools(state, my_deck)])
    return numeric.astype(np.float32), _board_ids(state)


N_HAND_IDS = 8   # M15: hand-card id slots (sorted, 0-padded, overflow capped)
N_STATE_IDS_V3 = N_STATE_IDS + N_HAND_IDS


def encode_state_v3(state, my_deck: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """M15 hand-aware ids: same STATE_V2_DIM numeric, but ids = 12 board +
    up to N_HAND_IDS of MY hand-card ids (sorted for permutation stability,
    0-padded). The M14 encoder audit: the hand was a summed 36-dim pool the
    id-embedding pathway never saw — invisible combos. encode_state_v2 stays
    untouched (legacy checkpoints/bundles keep loading)."""
    numeric, board = encode_state_v2(state, my_deck)
    me = state.players[state.yourIndex]
    hand = sorted(c.id for c in (me.hand or []) if c is not None)[:N_HAND_IDS]
    hand += [0] * (N_HAND_IDS - len(hand))
    return numeric, np.concatenate([board, np.array(hand, dtype=np.int32)])


def _attack_extra(attack_id, obs) -> np.ndarray:
    """[printed dmg/300, cost size/5, effective dmg vs opp active/300].

    Weakness/resistance math and the conditional-attack gate mirror
    rl.combat._charged_best (change BOTH if the rules change). For prompts
    where the attacker isn't my active (e.g. DISABLE_ATTACK) the effective
    term is best-effort; the printed dmg/cost are attack-intrinsic."""
    v = np.zeros(3, dtype=np.float32)
    if attack_id is None or attack_id not in _ATK:
        return v
    dmg, cost = _ATK[attack_id]
    v[0] = dmg / 300.0
    v[1] = len(cost) / 5.0
    state = obs.current
    me = state.players[state.yourIndex]
    opp = state.players[1 - state.yourIndex]
    atk = me.active[0] if me.active and me.active[0] is not None else None
    tgt = opp.active[0] if opp.active and opp.active[0] is not None else None
    if atk is None or tgt is None or dmg <= 0:
        return v
    board = {p.id for p in [atk] + list(me.bench or []) if p is not None}
    if not _attack_available(attack_id, board):
        return v
    t_weak, t_res, _, _, _ = _CARD.get(tgt.id, (None, None, 0, [], 1))
    atk_type = _CARD.get(atk.id, (None, None, 0, [], 1))[2]
    eff = dmg
    if t_weak is not None and int(t_weak) == atk_type:
        eff *= 2
    elif t_res is not None and int(t_res) == atk_type:
        eff = max(0, eff - 30)
    v[2] = eff / 300.0
    return v


def _my_poke_at(obs, in_play_area, in_play_index):
    """The in-play Pokémon OBJECT (with .energies/.hp) at my (area, index)."""
    me = obs.current.players[obs.current.yourIndex]
    zone = {4: me.active, 5: me.bench}.get(int(in_play_area))  # AreaType ACTIVE/BENCH
    if zone is None or in_play_index is None or in_play_index >= len(zone):
        return None
    return zone[in_play_index]


def _attach_extra(opt, obs) -> np.ndarray:
    """[target attached-energy/5, energy gap to charged-best/5, saturated flag].

    M19: ATTACH options previously carried only the target's PRINTED features
    — the net could not see that a 1-cost Solrock was already fed (live
    over-attach defect, forensics R1). Gap mirrors the pilots' charged-best
    semantics via _turns_to_ready incl. the CONDITIONAL_ATTACKS gate."""
    v = np.zeros(3, dtype=np.float32)
    poke = _my_poke_at(obs, opt.inPlayArea, opt.inPlayIndex)
    if poke is None:
        return v
    me = obs.current.players[obs.current.yourIndex]
    opp = obs.current.players[1 - obs.current.yourIndex]
    opp_active = opp.active[0] if opp.active and opp.active[0] is not None else None
    board_ids = {p.id for p in list(me.active or []) + list(me.bench or [])
                 if p is not None}
    gap = _turns_to_ready(poke, opp_active, board_ids)
    v[0] = min(len(poke.energies or ()), 5) / 5.0
    v[1] = min(gap, 5) / 5.0                      # UNREACHABLE clamps to 1.0
    v[2] = float(gap == 0)
    return v


def _attach_energy_ceiling(opt, obs) -> np.ndarray:
    """M41 [dead, scaling-aware gap/5, retreat slack/5, own-energy-scaling].

    0. `dead` — one more energy on this target buys nothing that exists in the
       rules: every attack already affordable, retreat cost already covered, no
       own-energy scaling. The exact predicate, no damage model
       (`combat.energy_is_dead`).
    1. the M19 gap, recomputed with `scaling=True` so an attack that prints 0
       and really does 20 x hand stops reading as UNREACHABLE. On Alakazam the
       M19 slot is 1.0 forever; this one is 0.0 the moment it holds its {P}.
    2. retreat slack: attached - retreatCost, clipped. Negative means energy
       here still buys an escape, which is the ONE legitimate reason to charge
       past the attack cost — so it is a separate signal, not folded into 0.
    3. the target's damage grows with its own attached energy, so it has no
       ceiling at all (Teal Mask Ogerpon ex, Hydrapple ex, 25 attacks pool-wide).
    """
    v = np.zeros(N_OPTION_ENERGY, dtype=np.float32)
    poke = _my_poke_at(obs, opt.inPlayArea, opt.inPlayIndex)
    if poke is None:
        return v
    from rl.combat import _RETREAT, energy_is_dead, scales_on_own_energy
    me = obs.current.players[obs.current.yourIndex]
    opp = obs.current.players[1 - obs.current.yourIndex]
    opp_active = opp.active[0] if opp.active and opp.active[0] is not None else None
    board_ids = {p.id for p in list(me.active or []) + list(me.bench or [])
                 if p is not None}
    attached = len(poke.energies or ())
    v[0] = float(energy_is_dead(poke.id, poke.energies or ()))
    v[1] = min(_turns_to_ready(poke, opp_active, board_ids, scaling=True), 5) / 5.0
    v[2] = max(-5, min(5, attached - _RETREAT.get(poke.id, 0))) / 5.0
    v[3] = float(scales_on_own_energy(poke.id))
    return v


def _retreat_extra(obs) -> np.ndarray:
    """[active damage fraction, active prizes-on-KO/3, bench-ready flag].

    M19: RETREAT options encoded as a bare type one-hot — nothing signalled
    "damaged multi-prize active + attack-ready bench" (the save-the-active
    defect, forensics R1). All three terms come from my own observation."""
    v = np.zeros(3, dtype=np.float32)
    me = obs.current.players[obs.current.yourIndex]
    opp = obs.current.players[1 - obs.current.yourIndex]
    active = me.active[0] if me.active and me.active[0] is not None else None
    if active is None:
        return v
    opp_active = opp.active[0] if opp.active and opp.active[0] is not None else None
    v[0] = 1.0 - active.hp / max(1, active.maxHp)
    v[1] = _CARD.get(active.id, (None, None, 0, [], 1))[4] / 3.0
    v[2] = float(any(_turns_to_ready(b, opp_active) == 0
                     for b in me.bench or [] if b is not None))
    return v


_PLAY_KIND: dict[int, str] = {}


def _play_kind() -> dict[int, str]:
    """{card id -> CardType NAME}, built once. Lazy because the card-data
    import lives further down (the v4 block owns it) and this runs per option.
    Keyed on the name, not the enum int: the fake engine used by the test suite
    numbers CardType differently, and a silent int collision here would encode
    a special energy as a supporter."""
    if not _PLAY_KIND:
        from cg.api import CardType, all_card_data
        # c.cardType is an enum in-repo but a plain int inside the shipped
        # bundle (caught by the isolation gate) — resolve through the enum so
        # both work, and key on the NAME so renumbering cannot collide.
        names = {int(member): member.name for member in CardType}
        _PLAY_KIND.update({c.cardId: names.get(int(c.cardType), "")
                           for c in all_card_data()})
    return _PLAY_KIND


def _phase_interaction(opt, obs) -> np.ndarray:
    """M28 [ending x late, ability x late, supporter x late].

    `late` grades how close WE are to decking out: 0 while the deck is healthy,
    rising to 1 at empty. The three option classes are the ones on the deck-out
    causal chain (docs/M27.md): ending the turn, using the draw engine, and
    playing a supporter.
    """
    v = np.zeros(N_OPTION_PHASE, dtype=np.float32)
    me = obs.current.players[obs.current.yourIndex]
    late = max(0.0, min(1.0, (DECK_LOW_AT - getattr(me, "deckCount", 60))
                        / DECK_LOW_AT))
    if late <= 0.0:
        return v
    t = int(opt.type)
    if t in (_OT_ATTACK, _OT_END):
        v[0] = late
    elif t == _OT_ABILITY:
        v[1] = late
    elif t == _OT_PLAY:
        card_id = opt.cardId
        if card_id is None and opt.index is not None:
            card_id = _card_id_at(obs, opt.area if opt.area is not None else _AREA_HAND,
                                  opt.index, obs.current.yourIndex)
        if card_id and _play_kind().get(int(card_id)) == "SUPPORTER":
            v[2] = late
    return v


def _target_value(attacker, target) -> float:
    """How attractive `target` is to attack — a port of the 0.505 sample
    agent's `pokemon_score` (sample-agent/main.py:97-115) plus its
    `score *= damage / hp` discount for a non-lethal hit. Used only to compare
    an opponent's bench against its active, so the absolute scale is arbitrary."""
    prizes = _CARD.get(target.id, (None, None, 0, [], 1))[4]
    base = (prizes * 1000.0
            + len(target.energies or ()) * 150.0
            + len(target.tools or ()) * 100.0
            + target.hp)
    dmg = _best_damage(attacker, target)
    return base if dmg >= target.hp else base * (dmg / max(1, target.hp))


def _play_precondition(card_id, obs) -> np.ndarray:
    """M27 [supporter-legal, stadium-legal, gust-wanted] for a PLAY option.

    Each slot is the sample agent's own predicate for that card class
    (docs/M27-sample-agent-diff.md), computed from the observation:

    0. is_supporter AND this turn's supporter is unused — the one-per-turn
       constraint, made visible at the OPTION instead of buried in the state.
    1. is_stadium AND no stadium is in play (`Gravity_Mountain: -1 if
       stadium_id != 0 else 10000`).
    2. is_gust AND the best benched opponent Pokemon is a MORE VALUABLE target
       than the active — a port of the sample agent's `pokemon_score` argmax
       (`plan.target >= 1`). A first cut used the narrow "bench is KO-able and
       the active is not" and fired on 15 of 232,059 corpus options (0.01%):
       untrainable. Value-comparison fires ~100x more often and is what the
       0.505 pilot actually computes.
    """
    v = np.zeros(N_OPTION_PLAY_PRE, dtype=np.float32)
    if not card_id:
        return v
    from rl.plan import GUST_IDS          # deferred: plan imports combat, not us
    kind = _play_kind().get(int(card_id))
    state = obs.current
    if kind == "SUPPORTER":
        v[0] = float(not getattr(state, "supporterPlayed", False))
    elif kind == "STADIUM":
        v[1] = float(not (state.stadium or []))
    if int(card_id) in GUST_IDS:
        me = state.players[state.yourIndex]
        opp = state.players[1 - state.yourIndex]
        my_active = me.active[0] if me.active else None
        opp_active = opp.active[0] if opp.active else None
        bench = [b for b in (opp.bench or []) if b is not None]
        if my_active is not None and bench:
            best_bench = max(_target_value(my_active, b) for b in bench)
            active_val = (_target_value(my_active, opp_active)
                          if opp_active is not None else 0.0)
            v[2] = float(best_bench > active_val)
    return v


def encode_option_v2(opt, obs) -> tuple[np.ndarray, np.ndarray]:
    """(numeric OPTION_V3_DIM f32, [acted_id, target_id] i32, 0 = none).

    M16: PLAY options resolve their hand card (FEAT block + embedding id were
    blank pre-M16), and the appended N_OPTION_EXTRA block encodes attack
    identity (ATTACK options) and the count (NUMBER options). M19: the same
    3 extra slots carry energy-sufficiency for ATTACH targets and retreat
    utility for RETREAT (types are mutually exclusive — the type one-hot
    disambiguates). Pinned pre-M16 checkpoints need encode_option_v2_legacy.

    M27: appends N_OPTION_PLAY_PRE play-precondition slots. The output is now
    OPTION_M27_DIM wide and `[:OPTION_V3_DIM]` is byte-identical to the M19
    encoding, so OPTION_V3_DIM checkpoints stay reproducible by truncation —
    every consumer slices to its own net.option_dim.

    M41: appends N_OPTION_ENERGY energy-ceiling slots for ATTACH options; the
    `[:OPTION_M28_DIM]` prefix stays byte-identical, so every live bundle
    truncates the new columns away and serves exactly what it served before."""
    num = np.zeros(OPTION_M41_DIM, dtype=np.float32)
    num[:OPTION_DIM] = encode_option(opt, obs)
    your_index = obs.current.yourIndex
    card_id = opt.cardId
    if card_id is None and opt.index is not None:
        if opt.area is not None:
            player = opt.playerIndex if opt.playerIndex is not None else your_index
            card_id = _card_id_at(obs, opt.area, opt.index, player)
        elif int(opt.type) == _OT_PLAY:
            # PLAY carries only a hand index; v1 stays blank by design.
            card_id = _card_id_at(obs, _AREA_HAND, opt.index, your_index)
            if card_id:
                num[N_OPTION_TYPES:N_OPTION_TYPES + FEAT_DIM] = FEAT[card_id]
    target_id = None
    if opt.inPlayArea is not None and opt.inPlayIndex is not None:
        target_id = _card_id_at(obs, opt.inPlayArea, opt.inPlayIndex, your_index)
    if int(opt.type) == _OT_ATTACK:
        num[OPTION_DIM:OPTION_DIM + 3] = _attack_extra(opt.attackId, obs)
    elif int(opt.type) == _OT_ATTACH and opt.inPlayArea is not None:
        num[OPTION_DIM:OPTION_DIM + 3] = _attach_extra(opt, obs)
        num[OPTION_M28_DIM:OPTION_M41_DIM] = _attach_energy_ceiling(opt, obs)
    elif int(opt.type) == _OT_RETREAT:
        num[OPTION_DIM:OPTION_DIM + 3] = _retreat_extra(obs)
    elif getattr(opt, "number", None) is not None:
        num[OPTION_DIM + 3] = min(float(opt.number), 10.0) / 10.0
    if int(opt.type) == _OT_PLAY:
        num[OPTION_V3_DIM:OPTION_M27_DIM] = _play_precondition(card_id, obs)
    # bounded, not open-ended: the M41 energy block now sits after this one
    num[OPTION_M27_DIM:OPTION_M28_DIM] = _phase_interaction(opt, obs)
    ids = np.array([card_id or 0, target_id or 0], dtype=np.int32)
    return num, ids


def encode_option_v2_legacy(opt, obs) -> tuple[np.ndarray, np.ndarray]:
    """Pre-M16 v2 encoding: (OPTION_V2_DIM f32, ids) with NO option-identity
    resolution — byte-identical to what pinned checkpoints (osv2_*,
    osv3_plan0c, osv3h_plan1) trained on; matchrunner selects it by sniffed
    option width so their measured baselines stay reproducible."""
    your_index = obs.current.yourIndex
    card_id = opt.cardId
    if card_id is None and opt.index is not None and opt.area is not None:
        player = opt.playerIndex if opt.playerIndex is not None else your_index
        card_id = _card_id_at(obs, opt.area, opt.index, player)
    target_id = None
    if opt.inPlayArea is not None and opt.inPlayIndex is not None:
        target_id = _card_id_at(obs, opt.inPlayArea, opt.inPlayIndex, your_index)
    ids = np.array([card_id or 0, target_id or 0], dtype=np.int32)
    return encode_option(opt, obs), ids


# --- Encoder v4 (M21) — full observable state + opponent memory -------------
# Appended AFTER the context one-hot: state_ctx becomes
#   [STATE_V2_DIM numeric | N_CONTEXTS one-hot | V4_EXTRA_DIM block]
# and ids become [12 board | 8 hand | N_MEM_IDS opponent-memory]. The v3
# prefix stays byte-identical, so migrate_v3_to_v4 (rl/plan_iter.py) is a pure
# column shuffle with zero-init v4 columns (the M11/M15/M16 warm-start
# invariant). The memory half lives in rl/memory.py (OppMemory); the constants
# live HERE as the single source of truth (memory imports them — never the
# reverse, to keep the import graph acyclic).
from cg.api import CardType as _CardType, all_card_data as _v4_card_data  # noqa: E402

N_MEM_IDS = 5        # last-4 opp played card ids + last opp attacker id
N_MEM_ATTACH_HOT = 7  # opp's last energy-attach target: none + active + bench 0..4
# [last-attack dmg/cost/eff 3 | attach-target one-hot 7 | attacked/passed/total 3
#  | known-hand count 1 | extra-draw intensity 1 | known-hand FEAT pool]
N_MEM = 3 + N_MEM_ATTACH_HOT + 3 + 1 + 1 + FEAT_DIM

N_SLOT_EXTRA = 5     # appearThisTurn, evo-stack, special-energy, ready, best-dmg
N_GLOBAL_EXTRA = 8
N_SELECT_EXTRA = 4 + 2 * FEAT_DIM   # min/max/damage/energy counts + effect + contextCard
V4_EXTRA_DIM = (N_MEM + 2 * (1 + N_BENCH) * N_SLOT_EXTRA + 2 * FEAT_DIM
                + 2 * (FEAT_DIM + 1) + N_GLOBAL_EXTRA + N_SELECT_EXTRA)
N_STATE_IDS_V4 = N_STATE_IDS_V3 + N_MEM_IDS

_SPECIAL_ENERGY_IDS = frozenset(
    c.cardId for c in _v4_card_data() if c.cardType == _CardType.SPECIAL_ENERGY)


def _v4_slots(ps):
    active = ps.active[0] if ps.active else None
    bench = list(ps.bench)[:N_BENCH]
    bench += [None] * (N_BENCH - len(bench))
    return [active] + bench


def _slot_extras(state) -> np.ndarray:
    """12 slots x N_SLOT_EXTRA: [appearThisTurn, len(preEvolution)/2,
    n special energies/3, attack-ready NOW, best affordable dmg/300].

    The last two make bench strength/readiness salient per-slot — the M21
    forensic retreat/promote defect (23/25 misses) was a policy failure, but
    v3 only exposed readiness pooled across the board (n_ready, race block)."""
    me = state.players[state.yourIndex]
    op = state.players[1 - state.yourIndex]
    out = np.zeros(2 * (1 + N_BENCH) * N_SLOT_EXTRA, dtype=np.float32)
    i = 0
    for ps, other in ((me, op), (op, me)):
        target = other.active[0] if other.active else None
        board = {p.id for p in _v4_slots(ps) if p is not None}
        for p in _v4_slots(ps):
            if p is not None:
                out[i] = float(getattr(p, "appearThisTurn", False))
                out[i + 1] = len(getattr(p, "preEvolution", None) or ()) / 2.0
                out[i + 2] = sum(1 for c in getattr(p, "energyCards", None) or ()
                                 if c is not None and c.id in _SPECIAL_ENERGY_IDS) / 3.0
                dmg = _best_damage(p, target, board_ids=board)
                out[i + 3] = float(dmg > 0)     # can attack NOW (affordable)
                out[i + 4] = dmg / 300.0
            i += N_SLOT_EXTRA
    return out


def _special_energy_pools(state) -> np.ndarray:
    """Pooled FEAT of attached SPECIAL energies per side — v3 collapses
    attachments to plain EnergyType counts, erasing special-energy identity."""
    pools = []
    for ps in (state.players[state.yourIndex], state.players[1 - state.yourIndex]):
        cards = [c for p in _v4_slots(ps) if p is not None
                 for c in getattr(p, "energyCards", None) or ()
                 if c is not None and c.id in _SPECIAL_ENERGY_IDS]
        pools.append(_pool(cards))
    return np.concatenate(pools)


def _prize_pools(state) -> np.ndarray:
    """Per side: pooled FEAT of face-up prize cards + face-up count/6 —
    v3 only sees prize COUNTS."""
    parts = []
    for ps in (state.players[state.yourIndex], state.players[1 - state.yourIndex]):
        face_up = [c for c in (ps.prize or []) if c is not None]
        parts.append(np.concatenate([_pool(face_up),
                                     [np.float32(len(face_up) / 6.0)]]))
    return np.concatenate(parts).astype(np.float32)


def _global_extras(state) -> np.ndarray:
    me = state.players[state.yourIndex]
    op = state.players[1 - state.yourIndex]
    return np.array([
        op.deckCount / 60.0,
        getattr(me, "benchMax", 5) / 5.0,
        getattr(op, "benchMax", 5) / 5.0,
        float(getattr(state, "stadiumPlayed", False)),
        float(getattr(state, "retreated", False)),
        min(getattr(state, "turnActionCount", 0) or 0, 10) / 10.0,
        float(getattr(state, "firstPlayer", 0) == state.yourIndex),
        me.handCount / 15.0,
    ], dtype=np.float32)


def _select_extras(select) -> np.ndarray:
    """Selection constraints + the cards driving the current sub-prompt —
    v3 submenu decisions couldn't see WHOSE effect was resolving or how many
    picks were required."""
    v = np.zeros(N_SELECT_EXTRA, dtype=np.float32)
    v[0] = min(getattr(select, "minCount", 0) or 0, 10) / 10.0
    v[1] = min(getattr(select, "maxCount", 0) or 0, 10) / 10.0
    v[2] = min(getattr(select, "remainDamageCounter", 0) or 0, 10) / 10.0
    v[3] = min(getattr(select, "remainEnergyCost", 0) or 0, 5) / 5.0
    effect = getattr(select, "effect", None)
    if effect is not None:
        v[4:4 + FEAT_DIM] = FEAT[effect.id]
    context_card = getattr(select, "contextCard", None)
    if context_card is not None:
        v[4 + FEAT_DIM:4 + 2 * FEAT_DIM] = FEAT[context_card.id]
    return v


def encode_v4_block(obs, memory) -> np.ndarray:
    """The V4_EXTRA_DIM appended block. `memory` is an rl.memory.OppMemory
    (already observe()d for this prompt); duck-typed to avoid an import cycle."""
    state = obs.current
    return np.concatenate([
        memory.features(state),
        _slot_extras(state),
        _special_energy_pools(state),
        _prize_pools(state),
        _global_extras(state),
        _select_extras(obs.select),
    ]).astype(np.float32)


def encode_ctx_v4(obs, my_deck: list[int], memory) -> tuple[np.ndarray, np.ndarray]:
    """One call for the full v4 network input: (state_ctx, state_ids).

    state_ctx = [STATE_V2_DIM | N_CONTEXTS | V4_EXTRA_DIM]  (f32)
    state_ids = [12 board | 8 hand | N_MEM_IDS]             (i32)

    Callers must OppMemory.observe(obs) exactly once per own prompt BEFORE
    encoding (the logs contract, rl/memory.py docstring)."""
    numeric, ids = encode_state_v3(obs.current, my_deck)
    state_ctx = np.concatenate([numeric, encode_context(obs.select.context),
                                encode_v4_block(obs, memory)])
    return state_ctx, np.concatenate([ids, memory.ids()])
