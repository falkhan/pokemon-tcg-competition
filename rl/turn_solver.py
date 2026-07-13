"""Within-turn combo solver — depth-limited search over MY OWN turn (M7.4a, L2).

The generic pilot is a greedy one-action argmax: a multi-prize lethal that
needs ``item -> attach -> attack`` is never assembled, because each enabling
action is scored on its own (the item can score below an unrelated develop
move). This module answers "is there an action SEQUENCE that takes prizes or
wins right now?" with the engine forward model before the pilot settles for
greedy setup — the "trainer + items into a multi-prize KO turn" case the
experts' documented edge covers (docs/M7-plan.md §3b L2).

Why this is cheap and sound: within my own turn the opponent never acts, so
the search needs NO opponent determinization — only my own draw order is
hidden, and the engine re-prompts after every action, so the solver simply
recomputes at each prompt from ground truth (a draw that breaks the combo
just makes the next solve return None and greedy play resumes). There is no
cached plan to invalidate.

Bundle purity: pure Python on the bundled ``cg`` API + ``rl.combat`` +
``rl.generic_pilot`` only (no numpy / polars / torch) — pinned by
tests/test_imports.py so the M7.5 submission flip is a two-line change.
``make_solver_pilot`` deliberately WRAPS ``make_generic_pilot`` instead of
hooking its scorers: the pilot twins (rl/generic_pilot.py / tcg/pilot.py)
stay byte-identical and parity-pinned (docs/DECISIONS.md).
"""
import random
from collections import namedtuple
from itertools import combinations
from time import perf_counter

from cg.api import (CardType, OptionType, all_card_data, search_begin,
                    search_end, search_step, to_observation_class)
from rl.combat import (_CARD, UNREACHABLE, _best_damage, _hits_to_ko,
                       _turns_to_first_ko, _turns_to_ready)
from rl.generic_pilot import (_IS_BASIC, _IS_POKEMON, make_generic_pilot,
                              score_option)

# --- Search budget (risk 7: multi-select blowup; G6: no external timeout) ---
MAX_DEPTH = 8            # decisions per line, incl. submenu picks
MAX_NODES = 800          # global search_step budget per solve
TOP_K = 4                # non-attack children kept per prompt (attacks always kept)
MAX_MULTI_COMBOS = 8     # maxCount>1 combination cap (the MCTS cap is 64)
SOLVE_DEADLINE_S = 0.4   # perf_counter safety valve per solve call

# --- Trigger ---
LETHAL_MARGIN = 70       # T2: a boost trainer might close damage gaps up to this

# --- Development tier (M8.1): search SETUP lines, not just lethal ones -------
# The M8.0 taxonomy (docs/M8.md 2026-07-12): energy routed off the best racer
# is the dominant setup mistake (55/100 games), then unused trainers. The dev
# tier fires on UNDERDEVELOPED boards (T5) and overrides greedy only when a
# line beats the stand-pat leaf by a margin — weights sized strictly below
# W_THREAT so a real lethal-next-turn setup always outranks generic development.
DEV_DEADLINE_S = 0.2     # T5 fires far more often than T1-T4: half the budget
DEV_TTFK_SLOW = 3        # T5: my fastest KO is >= this many turns away = slow
RACE_CAP_TURNS = 10.0    # ttfk capped here for dev math (encoders' RACE cap)
W_DEV_RACE = 900         # per turn shaved off my board's turns-to-first-KO...
DEV_RACE_CAP = 2         # ... capped: 2 turns = 1800 < W_THREAT's minimum 2000
W_DEV_READY = 500        # a NEW attack-ready damaging attacker appeared
W_DEV_EVO = 400          # per new evolution in play (cap 2)
W_DEV_BENCH = 300        # per new bench member (cap 2)
W_DEV_HAND = 40          # per net card drawn (cap 5; W_DECK_LOW still bites)
DEV_OVERRIDE_MARGIN = 900  # dev line must beat stand-pat by this to override.
                           # Tuning pass 1 (docs/M8.md 2026-07-13): at 500,
                           # bench/evo-only deltas (300-440) overrode greedy and
                           # cost tempo vs the expert (0.314 < 0.362) — at 900
                           # only race-improving lines (the taxonomy's actual #1
                           # mistake) or real multi-delta combos clear the bar.

# --- Leaf scoring: prizes taken NOW dominate everything but the game result ---
W_WIN, W_LOSS, W_DRAW = 1e9, -1e9, -5e8
W_PRIZE = 100_000        # per prize I take this turn
W_MY_PRIZE = -150_000    # per prize I concede (self-KO effects)
W_THREAT = 2_000         # lethal-next-turn setup, x target's prize value
W_COUNTER = -1_000       # opp active can return-KO my active, x its prize value
W_BENCHLESS_KO = -5e8    # ... and my bench is EMPTY: that return-KO ends the GAME,
                         # not a prize — dominates any prize haul (< -W_PRIZE * 6)
W_DECK_LOW = -5_000      # per card drawn while my deckCount <= 6 (anti-mill)
W_DAMAGE = 1.0           # per hp of chip on the opp active (tiebreak)
W_RACE = -10             # per turn of my best turns-to-first-KO (tiebreak)
DECK_LOW_AT = 6          # deckCount at or below which draws start costing
MIN_OVERRIDE_SCORE = W_PRIZE - 1  # override greedy only for >=1 prize or a win

# Opponent hidden-zone fillers (rl/mcts.py recipe): the opponent never acts
# within my turn, so placeholders are exact, not an approximation.
FILLER_POKEMON = 1072    # Snorlax (a Basic, legal as a hidden active)
FILLER_ENERGY = 1

# Trainer = anything that is neither a Pokémon nor an energy (items,
# supporters, tools, stadiums) — the hand cards that might enable a combo.
_IS_TRAINER = {c.cardId for c in all_card_data()
               if c.cardType not in (CardType.POKEMON, CardType.BASIC_ENERGY,
                                     CardType.SPECIAL_ENERGY)}

# root_* fields (M8.1): development facts the dev leaf scores DELTAS against —
# without them "do nothing" ties "develop". Defaults keep pre-M8.1 call sites
# (and pinned tests) valid; they only matter when score_leaf(dev=True).
_Snap = namedtuple("_Snap", "me my_prizes op_prizes op_active_hp my_deck_count "
                            "root_race root_ready root_bench root_evos root_hand",
                   defaults=(RACE_CAP_TURNS, 0, 0, 0, 0))


def _my_board(player):
    """Active + bench, skipping empty slots."""
    board = [p for p in player.active if p is not None]
    return board + [p for p in player.bench if p is not None]


def should_solve(obs) -> bool:
    """Cheap trigger: fire the (expensive) turn search only when a combo could
    plausibly pay off. Pure rl.combat dict math, O(board)."""
    if obs.select is None or getattr(obs, "search_begin_input", None) is None:
        return False
    if len(obs.select.option) < 2:
        return False
    st = obs.current
    me, op = st.players[st.yourIndex], st.players[1 - st.yourIndex]
    op_active = op.active[0] if op.active and op.active[0] is not None else None
    if op_active is None or op_active.id not in _CARD:
        return False
    board = _my_board(me)
    if not board:
        return False
    best_now = max(_best_damage(p, op_active) for p in board)
    best_plus = max(_best_damage(p, op_active, extra_energy=1) for p in board)
    if best_plus >= op_active.hp:                                  # T1: KO <=1 attach away
        return True
    if (best_now >= op_active.hp - LETHAL_MARGIN                   # T2: a boost trainer
            and any(c.id in _IS_TRAINER for c in me.hand)):        #     might close the gap
        return True
    if _CARD[op_active.id][4] >= 2 and any(                        # T3: multi-prize in reach
            _hits_to_ko(p, op_active) == 1 and _turns_to_ready(p, op_active) <= 1
            for p in board):
        return True
    if len(op.prize) <= 2 and best_plus > 0:                       # T4: game-closing range
        return True
    return False


def _dev_facts(me, op_active) -> tuple[float, int, int, int, int]:
    """(race, ready, bench, evos, hand) development facts for one side — the
    quantities the dev leaf scores as deltas vs the root. race = my board's
    best turns-to-first-KO (capped); ready = damaging attackers that can pay
    their best attack NOW. Pure rl.combat dict math, O(board)."""
    board = _my_board(me)
    if op_active is not None and op_active.id in _CARD and board:
        race = min(float(min(_turns_to_first_ko(p, op_active) for p in board)),
                   RACE_CAP_TURNS)
        ready = sum(1 for p in board if _turns_to_ready(p, op_active) == 0
                    and _best_damage(p, op_active) > 0)
    else:
        race, ready = RACE_CAP_TURNS, 0
    bench = sum(1 for p in me.bench if p is not None)
    evos = sum(1 for p in board if p.id not in _IS_BASIC)
    hand = len([c for c in me.hand if c is not None])
    return race, ready, bench, evos, hand


def should_solve_dev(obs) -> bool:
    """T5 (M8.1): the DEVELOPMENT trigger — underdeveloped board + sequencing
    material in hand. The pilot checks the lethal triggers FIRST; this fires
    on the slow positions they ignore: nobody attack-ready, or my fastest KO
    still >= DEV_TTFK_SLOW turns away, while the hand holds >= 2 trainers or
    a Pokémon (the M8.0 sequencing cases: energy routing, bench building,
    trainer chains)."""
    if obs.select is None or getattr(obs, "search_begin_input", None) is None:
        return False
    if len(obs.select.option) < 2:
        return False
    st = obs.current
    me, op = st.players[st.yourIndex], st.players[1 - st.yourIndex]
    op_active = op.active[0] if op.active and op.active[0] is not None else None
    if op_active is None or op_active.id not in _CARD:
        return False
    if not _my_board(me):
        return False
    hand = [c for c in me.hand if c is not None]
    n_trainers = sum(1 for c in hand if c.id in _IS_TRAINER)
    n_pokemon = sum(1 for c in hand if c.id in _IS_POKEMON)
    if n_trainers < 2 and n_pokemon < 1:
        return False                       # nothing to sequence: greedy is fine
    race, ready, _, _, _ = _dev_facts(me, op_active)
    return ready == 0 or race >= DEV_TTFK_SLOW


def _open_search(obs, deck):
    """Open a concrete search game for MY turn (rl/mcts.py determinize recipe,
    duplicated locally: rl.mcts imports numpy/torch and would break bundle
    purity). Own hidden zones sampled from my deck list; opponent zones are
    fillers — the opponent never moves inside my own turn."""
    st = obs.current
    mine, opp = st.players[st.yourIndex], st.players[1 - st.yourIndex]
    need_active = len(opp.active) > 0 and opp.active[0] is None
    return search_begin(
        obs,
        your_deck=random.sample(deck, mine.deckCount),
        your_prize=random.sample(deck, len(mine.prize)),
        opponent_deck=[FILLER_POKEMON] * opp.deckCount,
        opponent_prize=[FILLER_ENERGY] * len(opp.prize),
        opponent_hand=[FILLER_ENERGY] * opp.handCount,
        opponent_active=[FILLER_POKEMON] if need_active else [],  # must be a Basic
    )


def _root_snapshot(obs) -> _Snap:
    """Pre-search facts the leaf scorer diffs against. Prize semantics: the
    engine drains the OPPONENT's prize list as I take prizes (a KO wins when
    len(op.prize) <= its prize value — sample-agent/main.py)."""
    st = obs.current
    me, op = st.players[st.yourIndex], st.players[1 - st.yourIndex]
    op_active = op.active[0] if op.active and op.active[0] is not None else None
    race, ready, bench, evos, hand = _dev_facts(me, op_active)
    return _Snap(st.yourIndex, len(me.prize), len(op.prize),
                 op_active.hp if op_active is not None else 0, me.deckCount,
                 race, ready, bench, evos, hand)


def _candidate_actions(obs) -> list[list[int]]:
    """Beam over one prompt's option menu, lethal-first. Single-pick prompts
    keep every ATTACK + the TOP_K best others by the greedy pilot's own
    score_option (+ END, so every prompt has a turn-terminating child).
    Multi-select prompts keep the MAX_MULTI_COMBOS best exact-maxCount
    combinations by summed member score (the house action semantics —
    tcg/search.py enumerate_actions)."""
    sel = obs.select
    n = len(sel.option)
    scores = [score_option(o, obs) for o in sel.option]
    if sel.maxCount == 1:
        attacks = [i for i in range(n) if sel.option[i].type == OptionType.ATTACK]
        others = sorted((i for i in range(n) if i not in set(attacks)),
                        key=lambda i: scores[i], reverse=True)
        keep = attacks + others[:TOP_K]
        end = next((i for i in range(n) if sel.option[i].type == OptionType.END), None)
        if end is not None and end not in keep:
            keep.append(end)
        return [[i] for i in keep]
    combos = sorted(combinations(range(n), sel.maxCount),
                    key=lambda c: sum(scores[i] for i in c), reverse=True)
    return [list(c) for c in combos[:MAX_MULTI_COMBOS]]


def _dev_bonus(snap: _Snap, me_p, op_active) -> float:
    """The M8.1 development block: deltas vs the root snapshot, all capped,
    everything strictly below W_THREAT's minimum (2000) so a real
    lethal-next-turn setup always outranks generic development. Taxonomy
    weighting (docs/M8.md): race/energy-routing >> ready attacker > evolution
    > bench > card advantage."""
    race, ready, bench, evos, hand = _dev_facts(me_p, op_active)
    bonus = W_DEV_RACE * max(0.0, min(float(DEV_RACE_CAP), snap.root_race - race))
    bonus += W_DEV_READY * max(0, min(1, ready - snap.root_ready))
    bonus += W_DEV_EVO * max(0, min(2, evos - snap.root_evos))
    bonus += W_DEV_BENCH * max(0, min(2, bench - snap.root_bench))
    bonus += W_DEV_HAND * max(0, min(5, hand - snap.root_hand))
    return bonus


def score_leaf(snap: _Snap, obs, dev: bool = False) -> float:
    """End-of-line value from MY perspective: game result, then prizes taken
    this turn, then lethal-next-turn setup / exposure / chip tiebreaks.
    dev=True (M8.1) adds the development block — deltas vs the root snapshot,
    so stand-pat scores 0 development and only real setup gains rank."""
    cur = obs.current
    if cur.result >= 0:
        if cur.result == 2:
            return W_DRAW
        return W_WIN if cur.result == snap.me else W_LOSS
    me_p, op_p = cur.players[snap.me], cur.players[1 - snap.me]
    score = W_PRIZE * max(0, snap.op_prizes - len(op_p.prize))      # prizes I took
    score += W_MY_PRIZE * max(0, snap.my_prizes - len(me_p.prize))  # prizes I conceded
    my_active = me_p.active[0] if me_p.active and me_p.active[0] is not None else None
    op_active = op_p.active[0] if op_p.active and op_p.active[0] is not None else None
    board = _my_board(me_p)
    if op_active is not None and op_active.id in _CARD and board:
        if any(_best_damage(p, op_active, extra_energy=1) >= op_active.hp
               for p in board):
            score += W_THREAT * _CARD[op_active.id][4]              # lethal next turn
        score += W_DAMAGE * max(0, snap.op_active_hp - op_active.hp)
        if my_active is not None and _best_damage(op_active, my_active) >= my_active.hp:
            score += W_COUNTER * _CARD.get(my_active.id, (0, 0, 0, [], 1))[4]
            if len(board) == 1:          # active only, bench EMPTY: game over, not a prize
                score += W_BENCHLESS_KO
        race = min(_turns_to_first_ko(p, op_active) for p in board)
        if race < UNREACHABLE:
            score += W_RACE * race
    if me_p.deckCount <= DECK_LOW_AT:
        score += W_DECK_LOW * max(0, snap.my_deck_count - me_p.deckCount)
    if dev:
        score += _dev_bonus(snap, me_p, op_active)
    return score


def _dfs(state, snap: _Snap, depth: int, deadline: float, budget: dict,
         dev: bool = False):
    """Depth-first search over MY remaining turn. Returns (score, line) where
    line is the action list-of-lists from `state` to the best leaf. Leaves:
    game over, turn passed to the opponent, or depth cap. The stand-pat floor
    means prizes already taken along the way are never given back by a worse
    continuation."""
    obs = state.observation
    if obs.current.result >= 0 or obs.current.yourIndex != snap.me \
            or depth >= MAX_DEPTH:
        return score_leaf(snap, obs, dev), []
    best_score, best_line = score_leaf(snap, obs, dev), []   # stand-pat floor
    for action in _candidate_actions(obs):
        if budget["nodes"] >= MAX_NODES or perf_counter() >= deadline:
            break
        budget["nodes"] += 1
        child = search_step(state.searchId, action)
        score, line = _dfs(child, snap, depth + 1, deadline, budget, dev)
        if score > best_score:
            best_score, best_line = score, [action] + line
        if best_score >= W_WIN:                          # win short-circuit
            break
    return best_score, best_line


def solve_turn(obs, deck: list[int], deadline_s: float | None = None,
               dev: bool = False) -> list[int] | None:
    """Search my remaining turn; return the FIRST action of the best line iff
    it clears the tier's override bar, else None (defer to greedy). The
    caller re-invokes on the next prompt — recompute-per-prompt absorbs own
    draw reveals, so no plan is cached.

    Tiers (M8.1): lethal (default) overrides only for >=1 prize or a win
    (MIN_OVERRIDE_SCORE); dev overrides only when the best line beats the
    stand-pat leaf by DEV_OVERRIDE_MARGIN — a real development gain, not
    line-vs-line noise — under the shorter DEV_DEADLINE_S."""
    if deadline_s is None:
        deadline_s = DEV_DEADLINE_S if dev else SOLVE_DEADLINE_S
    snap = _root_snapshot(obs)
    deadline = perf_counter() + deadline_s
    budget = {"nodes": 0}
    root = _open_search(obs, deck)
    try:
        best_score, best_line = _dfs(root, snap, 0, deadline, budget, dev)
    finally:
        search_end()
    if not best_line:
        return None
    if dev:
        if best_score >= score_leaf(snap, obs, dev=True) + DEV_OVERRIDE_MARGIN:
            return [int(i) for i in best_line[0]]
        return None
    if best_score >= MIN_OVERRIDE_SCORE:
        return [int(i) for i in best_line[0]]
    return None


def make_solver_pilot(deck: list[int], instance: str = "ts", dev: bool = False):
    """Generic pilot + within-turn combo solver. `instance` is accepted for
    the matchrunner uniqueness contract (unused: no module-level state).
    dev=True (M8.1) additionally runs the DEVELOPMENT tier on underdeveloped
    boards the lethal triggers ignore (`solver-dev:` matchrunner spec).

    Solves at ANY prompt the trigger fires on — including submenu prompts
    mid-combo (unlike rl/hybrid.py's MAIN-only guard), otherwise the line
    found at the MAIN prompt would derail one action later. Any solver error
    falls back to the greedy pick: the wrapper must never cost the G1 crash
    gate."""
    inner = make_generic_pilot(deck)

    def agent(obs_dict):
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return deck
        if should_solve(obs):
            try:
                pick = solve_turn(obs, deck)
            except Exception:
                pick = None
            if pick is not None:
                return pick
        elif dev and should_solve_dev(obs):
            try:
                pick = solve_turn(obs, deck, dev=True)
            except Exception:
                pick = None
            if pick is not None:
                return pick
        return inner(obs_dict)

    return agent
