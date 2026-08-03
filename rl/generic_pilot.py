from cg.api import to_observation_class, OptionType, SelectContext, AreaType, CardType, all_card_data
from rl.combat import (_CARD, _ATK, _can_afford, _best_damage,  # pure-Python combat core (ships in submission)
                       _turns_to_first_ko, _turns_to_ready, UNREACHABLE)

# Rough priority so it develops, attacks, and doesn't just pass

_PRIORITY = {
    OptionType.ATTACK: 100, OptionType.ABILITY: 90, OptionType.EVOLVE: 80,
    OptionType.PLAY: 70, OptionType.ATTACH: 60, OptionType.CARD: 50,
    OptionType.YES: 40, OptionType.NUMBER: 30, OptionType.RETREAT: 20,
    OptionType.NO: 10, OptionType.END: 0,
}

_IS_POKEMON = {c.cardId for c in all_card_data() if c.cardType == CardType.POKEMON}
_IS_ENERGY = {c.cardId for c in all_card_data()
              if c.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY)}

# Evolution-line tables (M7.5 guards). getattr: the fake_cg test stub predates
# these card fields. Constants mirrored from tcg/constants.py — change BOTH.
_NAME = {c.cardId: getattr(c, "name", None) for c in all_card_data()}
_IS_BASIC = {c.cardId for c in all_card_data() if getattr(c, "basic", True)}
_EVOLVES_FROM = {c.cardId: getattr(c, "evolvesFrom", None) for c in all_card_data()}
_HAND_DISCARD_TRAINER_NAMES = ("Carmine",)   # HAND_DISCARD_TRAINER_NAMES
_HAND_DISCARD_IDS = {cid for cid, n in _NAME.items()
                     if n in _HAND_DISCARD_TRAINER_NAMES}
_GUST_TRAINER_NAMES = ("Boss’s Orders",)  # GUST_TRAINER_NAMES — U+2019 in card data, not ASCII '
_GUST_IDS = {cid for cid, n in _NAME.items() if n in _GUST_TRAINER_NAMES}

# --- card-selection context categories (for score_card) ---
_PROMOTE_CTX = {SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.SETUP_BENCH_POKEMON,
                SelectContext.TO_ACTIVE, SelectContext.SWITCH, SelectContext.TO_FIELD,
                SelectContext.TO_BENCH}   # pick MY best Pokemon
_KEEP_CTX    = {SelectContext.TO_HAND, SelectContext.LOOK, SelectContext.NOT_MOVE}  # fetch useful
_DISCARD_CTX = {SelectContext.DISCARD, SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM,
                SelectContext.DISCARD_CARD_OR_ATTACHED_CARD}          # lose the worst
_TARGET_CTX  = {SelectContext.DAMAGE, SelectContext.DAMAGE_COUNTER,
                SelectContext.DAMAGE_COUNTER_ANY, SelectContext.EFFECT_TARGET}  # hit opp


def _attacker_quality(cid):
    """Best raw attack damage a card can deal (0 if not an attacker)."""
    return max((_ATK[a][0] for a in _CARD[cid][3] if a in _ATK), default=0)


def _card_usefulness(cid):
    """How valuable a card is to KEEP / fetch: attackers > energy > other."""
    if cid in _IS_POKEMON:
        return 300 + min(_attacker_quality(cid), 300)
    if cid in _IS_ENERGY:
        return 250
    return 120


def _pokemon_in_hand(me):
    return [c for c in (me.hand or []) if c is not None and c.id in _IS_POKEMON]


def _my_pokemon_names(me, hand_pokemon):
    """Names of my Pokémon in play and in hand — the evolution-basis pool."""
    in_play = [p for p in (list(me.active or []) + list(me.bench or []))
               if p is not None]
    return ({_NAME.get(p.id) for p in in_play}
            | {_NAME.get(c.id) for c in hand_pokemon}) - {None}


def _hand_holds_keepers(me):
    """Any hand Pokémon worth protecting from a hand-discard trainer: a basic,
    an evolution whose basis is in play or hand, or a hard hitter even while
    momentarily dead (the deck's win-condition class — kaggle ep 85467275
    lost to Carmine discarding Mega Lucario ex twice)."""
    hand_pokemon = _pokemon_in_hand(me)
    basis_names = _my_pokemon_names(me, hand_pokemon)
    for c in hand_pokemon:
        if c.id in _IS_BASIC:
            return True
        basis = _EVOLVES_FROM.get(c.id)
        if basis is not None and basis in basis_names:
            return True
        if _attacker_quality(c.id) >= 100:   # HAND_DISCARD_PROTECT_QUALITY
            return True
    return False


def _fetch_value(card, me):
    """KEEP/fetch value with evolution-line awareness (M7.5): dead evolutions
    sink, basics rise while the bench is empty or their evolution waits in
    hand (kaggle ep 85469339: Poké Pad fetched a basis-less Hariyama twice
    over live basics; the benched-out loss followed)."""
    base = _card_usefulness(card.id)
    if card.id not in _IS_POKEMON:
        return base
    hand_pokemon = _pokemon_in_hand(me)
    basis = _EVOLVES_FROM.get(card.id)
    if basis is not None:                                 # evolution card
        if basis not in _my_pokemon_names(me, hand_pokemon):
            return 60                                     # FETCH_DEAD_EVOLUTION
        return base
    bonus = 0                                             # basic Pokémon
    if not any(p is not None for p in me.bench):
        bonus += 400                       # FETCH_EMPTY_BENCH_BASIC_BONUS
    my_name = _NAME.get(card.id)
    if my_name is not None and any(_EVOLVES_FROM.get(c.id) == my_name
                                   for c in hand_pokemon):
        bonus += 300                       # FETCH_ENABLES_EVOLUTION_BONUS
    return base + bonus


def _attach_recipient_value(card, me, op_active):
    """ATTACH_FROM: which of MY (usually benched) Pokémon receives an energy.

    Marginal value — the opposite of the promote ladder: a Pokémon whose best
    attack is already paid gains nothing from another energy (kaggle ep
    85607769: 5 energies on a 1-cost Solrock). A basic whose evolution waits
    in hand is charged FOR the evolution — attached energy survives evolving,
    so Riolu carrying 2 is a pre-charged Mega Brave.
    Constants mirrored from tcg/constants.py — change BOTH."""
    from types import SimpleNamespace
    my_name = _NAME.get(card.id)
    evolution = next(
        (c for c in _pokemon_in_hand(me)
         if my_name is not None and _EVOLVES_FROM.get(c.id) == my_name),
        None)
    profile = (SimpleNamespace(id=evolution.id,
                               energies=list(getattr(card, "energies", ()) or ()))
               if evolution is not None else card)
    # M13 0a: pass own board ids so CONDITIONAL_ATTACKS gate the charged-best
    # (a Solrock without Lunatone in play is not the attacker its table row
    # claims — the live ep-85607769/86469359 overfeeding at the source).
    board_ids = {p.id for p in (list(me.active or []) + list(me.bench or []))
                 if p is not None}
    energy_gap = _turns_to_ready(profile, op_active, board_ids)
    if energy_gap == 0:
        return 50                          # ATTACH_RECIPIENT_CHARGED
    quality = min(_attacker_quality(profile.id), 300)   # ATTACKER_QUALITY_CAP
    return 300 + quality - 50 * min(energy_gap - 1, 4)  # BASE - PENALTY*min(gap-1, CAP)


def make_generic_pilot(deck, fixes: frozenset = frozenset()):
    def agent(obs_dict):
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return deck
        scores = [score_option(o, obs, fixes) for o in obs.select.option]
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [int(i) for i in order[:obs.select.maxCount]]
    return agent

def _card_at(obs, area, index, player):
    ps = obs.current.players[player]
    zone = {2: ps.hand, 3: ps.discard, 4: ps.active, 5: ps.bench,
            1: obs.select.deck, 7: obs.current.stadium}.get(int(area)) if area is not None else None
    try: return zone[index]
    except (TypeError, IndexError): return None

def score_option(o, obs, fixes: frozenset = frozenset()):
    st = obs.current; me = st.players[st.yourIndex]; op = st.players[1 - st.yourIndex]
    my_active = me.active[0] if me.active else None
    op_active = op.active[0] if op.active else None

    t = o.type
    if t == OptionType.ATTACK:
        # M41 `scaling`: opt-in effective damage for attacks whose printed
        # number is only their base (Alakazam's Powerful Hand prints 0). OFF
        # unless the spec asked for it — the shipped features must not move.
        scaling = "scaling" in fixes
        hand_n = len(me.hand or []) if scaling else 0
        mine = [p for p in ([my_active] + list(me.bench or [])) if p]
        bench_n = len([b for b in (me.bench or []) if b]) if scaling else 0
        team_nrg = sum(len(p.energies or []) for p in mine) if scaling else 0
        return score_attack(o, my_active, op_active, op.bench, scaling=scaling,
                            hand_size=hand_n, bench_size=bench_n,
                            team_energy=team_nrg)
    if t == OptionType.ATTACH: return score_attach(o, obs, me)
    if t == OptionType.ABILITY: return 3000
    if t == OptionType.EVOLVE: return 2800
    if t == OptionType.PLAY: return score_play(o, obs, fixes)
    if t == OptionType.RETREAT: return score_retreat(obs)
    if t == OptionType.CARD: return score_card(o, obs, fixes)
    return _PRIORITY.get(t, 0)


def score_card(o, obs, fixes: frozenset = frozenset()):
    """Card-selection contexts (SETUP/SWITCH/TO_HAND/DISCARD/...) — no more coin flips."""
    st = obs.current
    op = st.players[1 - st.yourIndex]
    op_active = op.active[0] if op.active else None
    ctx = obs.select.context
    player = o.playerIndex if o.playerIndex is not None else st.yourIndex
    card = _card_at(obs, o.area, o.index, player)
    if card is None:
        return 0

    if ctx == SelectContext.ATTACH_FROM:          # energy recipient: marginal value
        return _attach_recipient_value(card, st.players[st.yourIndex], op_active)
    if ctx in _PROMOTE_CTX:                       # start / promote / bench MY best attacker
        q = _attacker_quality(card.id)
        ready = 0                                 # bonus if it can attack right now
        if op_active is not None and hasattr(card, "energies") and _best_damage(card, op_active) > 0:
            ready = 500
        # Race term (M7.2b): -50 per attach still needed (capped) — a charged
        # attacker beats an equal-damage uncharged one, non-attackers sink.
        return q + ready - 50 * min(_turns_to_ready(card, op_active), 4)
    if ctx in _KEEP_CTX:                          # fetch/keep the most useful card
        return _fetch_value(card, st.players[st.yourIndex])
    if ctx in _DISCARD_CTX:                       # discard the LEAST useful
        return -_card_usefulness(card.id)
    if ctx in _TARGET_CTX:                        # damage the highest-prize opponent Pokémon
        prize_score = 100 * _CARD.get(card.id, (0, 0, 0, [], 1))[4]
        if ("gust" in fixes and ctx == SelectContext.EFFECT_TARGET
                and player != st.yourIndex
                and getattr(card, "hp", None) is not None):
            # Fix B2 (M9): the gust PLAY fired because a faster-KO target
            # exists — pick by fastest KO (race math), prize tie-break.
            # Unknown/unKOable targets keep the plain prize ranking.
            me = st.players[st.yourIndex]
            board = [p for p in (list(me.active or []) + list(me.bench or []))
                     if p is not None]
            ttk = min((_turns_to_first_ko(p, card) for p in board),
                      default=UNREACHABLE)
            return 1000 * max(0, 12 - min(ttk, 12)) + prize_score
        return prize_score
    return 50                                     # unknown card context: neutral


def _gust_has_better_target(me, op):
    """M9 Fix B predicate: the opponent bench holds a target my board KOs
    strictly faster than their active. Unknown/unKOable targets never qualify
    (_turns_to_first_ko returns UNREACHABLE for them)."""
    op_active = op.active[0] if op.active and op.active[0] is not None else None
    bench = [p for p in (op.bench or []) if p is not None]
    board = [p for p in (list(me.active or []) + list(me.bench or [])) if p is not None]
    if op_active is None or not bench or not board:
        return False
    active_ttk = min(_turns_to_first_ko(p, op_active) for p in board)
    return any(min(_turns_to_first_ko(p, t) for p in board) < min(active_ttk, UNREACHABLE)
               for t in bench)


def _op_board_harmless(op_active, op_bench=()):
    """CLOSE MODE predicate (M7.2b): every opponent board Pokémon has a KNOWN
    card id and zero printed attack damage — they can never take a prize by KO.
    Unknown ids count as threats (conservative: the floor-test case only)."""
    board = ([op_active] if op_active is not None else [])
    board += [p for p in op_bench if p is not None]
    return all(p.id in _CARD and _attacker_quality(p.id) == 0 for p in board)


def score_attack(o, my_active, op_active, op_bench=(), scaling: bool = False,
                 hand_size: int = 0, bench_size: int = 0, team_energy: int = 0):
    if op_active is None:
        return 1000

    damage, cost_tuple = _ATK.get(o.attackId, (0, ()))
    if scaling:
        from rl.scaling import effective_damage
        damage = effective_damage(o.attackId, my_active, op_active,
                                  hand_size=hand_size, bench_size=bench_size,
                                  team_energy=team_energy)
    attack_type = _CARD[my_active.id][2]
    op_weakness = _CARD[op_active.id][0]
    op_resistance = _CARD[op_active.id][1]

    if op_weakness is not None and int(op_weakness) == attack_type:
        damage *= 2
    elif op_resistance is not None and int(op_resistance) == attack_type:
        damage -= 30

    if damage >= op_active.hp:          ## KO takes a prize NOW — prioritise closing the game
        prize = _CARD[op_active.id][4]
        # Slots just BELOW free, non-milling setup (ability 3000 / evolve 2800 / attach-to-active
        # 2600) so we still do those first (they don't end the turn), but ABOVE extra benching /
        # bench-attach / draw (<=2400) so we take the prize and end the turn instead of
        # over-developing — which draws cards and races us to deck-out (the self-deck bug).
        return 2500 + 50 * prize
    if _op_board_harmless(op_active, op_bench):
        ## CLOSE MODE (M7.2b): the race is won — attack every turn instead of milling
        ## (the M6 floor-test self-deck fix; chip jumps above trainers <=2200).
        return 2300 + damage / 10
    else:                               ## no KO — chip damage stays low; develop first, attack last
        return 1000 + damage / 10


def score_retreat(obs):
    st = obs.current
    me = st.players[st.yourIndex]; op = st.players[1 - st.yourIndex]
    my_active = me.active[0] if me.active else None
    op_active = op.active[0] if op.active else None
    bench = [p for p in me.bench if p is not None]
    if my_active is None or not bench:
        return -1

    def best_attack(p):
        return max((_ATK[a][0] for a in _CARD[p.id][3] if a in _ATK), default=0)
    bench_best = max(best_attack(b) for b in bench)


    # 1) Promote a lethal attacker from the bench if the active can't KO
    if op_active is not None and _best_damage(my_active, op_active) < op_active.hp:
        if any(_best_damage(b, op_active) >= op_active.hp for b in bench):
            return 2900


    # 2) Escape a KO
    in_danger = op_active is not None and _best_damage(op_active, my_active) >= my_active.hp
    if in_danger and bench_best > 0:
        return 1500

    # 2b) Save a valuable damaged active (M19): a multi-prize Mega/ex at low
    #     HP rotates out into an attack-READY bench member BEFORE the lethal
    #     is on board — live prize-race losses ended with the opponent taking
    #     3 prizes off our chipped-down Mega while an energized bench watched.
    #     Constants mirrored from tcg/constants.py — change BOTH.
    hp_frac = my_active.hp / max(1, my_active.maxHp)
    if (hp_frac <= 0.4                                    # SAVE_ACTIVE_HP_FRACTION
            and _CARD.get(my_active.id, (None, None, 0, [], 1))[4] >= 2
            and any(_turns_to_ready(b, op_active) == 0 for b in bench)):
        return 1450                                       # SCORE_RETREAT_SAVE_VALUABLE

    # 3) Otherwise low; ~0 when healthy, a bit higher if a strong bench attacker wants in
    if hp_frac > 0.75:
        return -1
    return 100 + bench_best // 20

def score_play(o, obs, fixes: frozenset = frozenset()):
    my_index = obs.current.yourIndex
    me = obs.current.players[my_index]
    # Engine quirk (found 2026-07-12, kaggle ep 85467275 + local repro): PLAY
    # options carry NO `area` field — the index is always a hand index. Without
    # this default every play resolved to None -> flat 2000, so the whole play
    # tier below (Pokémon vs trainer, taper, deck-out guard) never fired in a
    # real game — only in tests, whose synthetic options set area explicitly.
    area = o.area if o.area is not None else AreaType.HAND
    card = _card_at(obs, area, o.index, my_index)

    if card is None:
        return 2000
    if card.id in _IS_POKEMON:
        if not any(p is not None for p in me.bench):
            return 2700                   # EMPTY bench: above the whole KO tier (max
                                          # 2650) — benching never ends the turn, but
                                          # attacking does; KO-first with no bench let
                                          # one return-KO end the game (basics in hand)
        return 2400                       # developing the board is always good

    if card.id in _HAND_DISCARD_IDS and _hand_holds_keepers(me):
        if "handdiscard" in fixes:
            return -100                   # SCORE_HAND_DISCARD_REFUSED: below END — pass instead
        return 150                        # SCORE_HAND_DISCARD_BLOCKED: below near-deckout

    if "gust" in fixes and card.id in _GUST_IDS:
        op = obs.current.players[1 - my_index]
        if _gust_has_better_target(me, op):
            return 2450                   # SCORE_GUST_KILLSHOT: above taper/develop, below KO tier

    has_board = (bool(me.active) and me.active[0] is not None) or any(p for p in me.bench)
    if not has_board:
        return 300                        # no Pokémon yet: trainers can't help

    # Trainer: value on NEED, not a flat 2200 — the anti-deck-out fix.
    # (Crude: can't yet tell a draw supporter from a gust/Switch — no effect-text
    #  parsing — so it suppresses all trainers when the hand is full. Tech debt.)
    hand = me.handCount or 0
    deck = me.deckCount or 0
    if deck <= 6:                         # near decking: stop thinning our own deck
        return 200
    base = 2200 - 200 * max(0, hand - 4)  # taper as the hand grows past ~4
    return max(400, base)

def score_attach(o, obs, me):
    my_index = obs.current.yourIndex

    opponent = obs.current.players[1 - my_index]
    opponent_active_card = opponent.active[0] if opponent.active else None


    target_pokemon = _card_at(obs, o.inPlayArea, o.inPlayIndex, my_index)

    if target_pokemon is None:
        return 500

    is_active = o.inPlayArea == AreaType.ACTIVE
    # M14 replay finding (51-63% of live attaches went to saturated
    # recipients; Solrock fed without Lunatone up to 1.8x/game): this path
    # never learned the CONDITIONAL_ATTACKS card facts — thread board_ids so
    # a conditional attacker without its requirement stops counting.
    from rl.combat import _attack_available
    my_board_ids = {p.id for p in (list(me.active or []) + list(me.bench or []))
                    if p is not None}

    # 1) Lookahead: does this attach UNBLOCK a KO on the opponent's active?

    if opponent_active_card is not None:
        now = _best_damage(target_pokemon, opponent_active_card, extra_energy=0,
                           board_ids=my_board_ids)
        after = _best_damage(target_pokemon, opponent_active_card, extra_energy=1,
                             board_ids=my_board_ids)
        if now < opponent_active_card.hp <= after:
            return  4000 if is_active else 2900    # active can cash it this turn -> top priority

    # 2) Otherwise is target a real attacker that still needs energy?

    damaging = [(_ATK[a][0], len(_ATK[a][1])) for a in _CARD[target_pokemon.id][3]
                if a in _ATK and _ATK[a][0] > 0
                and _attack_available(a, my_board_ids)]

    if not damaging:
        return 400

    best_dmg = max(d for d, _ in damaging)
    bonus = min(best_dmg, 300) // 100

    if _turns_to_ready(target_pokemon, opponent_active_card,
                       board_ids=my_board_ids) == 0:
        # The BEST attack is charged (M7.2b — was the cheapest, which stopped
        # charging a 2-cost 270 attacker after its 1-cost 130 was paid).
        # M19: penalize per SURPLUS energy so the least-fed charged target
        # wins the tier and heavy surplus drops below NON_ATTACKER — the flat
        # 600 kept feeding a 1-cost Solrock 3+ energies live. Constants
        # mirrored from tcg/constants.py — change BOTH.
        best_cost = min(c for d, c in damaging if d == best_dmg)
        surplus = len(getattr(target_pokemon, "energies", ()) or ()) - best_cost
        return 600 + bonus - 150 * min(max(surplus, 0), 3)

    # 3) Race math (M7.2b): charge THE ONE attacker that closes first. The
    #    target's own turns-to-first-KO must match the board minimum; ties
    #    bonus every tied target and self-commit after the first attach
    #    (the winner's energy gap drops, making it strictly unique).
    #    The BENCH tier additionally requires an attack-ready ACTIVE — banking
    #    on a benched closer while the active can neither attack nor pay its
    #    retreat starves the whole board and mills the deck (the measured
    #    floor-test failure, docs/M7.md 2026-07-09).
    if opponent_active_card is not None:
        my_active = me.active[0] if me.active and me.active[0] is not None else None
        board = ([my_active] if my_active is not None else [])
        board += [p for p in me.bench if p is not None]
        mine = _turns_to_first_ko(target_pokemon, opponent_active_card)
        if mine < UNREACHABLE and board and \
                mine <= min(_turns_to_first_ko(p, opponent_active_card) for p in board):
            if is_active:
                return 2750 + bonus
            if my_active is not None and _turns_to_ready(my_active, opponent_active_card) == 0:
                return 2680 + bonus

    return (2600 if is_active else 2400) + bonus


