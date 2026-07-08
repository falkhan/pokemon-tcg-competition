from cg.api import to_observation_class, OptionType, SelectContext, AreaType, CardType, all_card_data
from rl.combat import _CARD, _ATK, _can_afford, _best_damage  # pure-Python combat core (ships in submission)

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

# --- card-selection context categories (for score_card) ---
_PROMOTE_CTX = {SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.SETUP_BENCH_POKEMON,
                SelectContext.TO_ACTIVE, SelectContext.SWITCH, SelectContext.TO_FIELD,
                SelectContext.TO_BENCH, SelectContext.ATTACH_FROM}   # pick MY best Pokemon
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

def make_generic_pilot(deck):
    def agent(obs_dict):
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return deck
        scores = [score_option(o, obs) for o in obs.select.option]
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [int(i) for i in order[:obs.select.maxCount]]
    return agent

def _card_at(obs, area, index, player):
    ps = obs.current.players[player]
    zone = {2: ps.hand, 3: ps.discard, 4: ps.active, 5: ps.bench,
            1: obs.select.deck, 7: obs.current.stadium}.get(int(area)) if area is not None else None
    try: return zone[index]
    except (TypeError, IndexError): return None

def score_option(o, obs):
    st = obs.current; me = st.players[st.yourIndex]; op = st.players[1 - st.yourIndex]
    my_active = me.active[0] if me.active else None
    op_active = op.active[0] if op.active else None

    t = o.type
    if t == OptionType.ATTACK: return score_attack(o, my_active, op_active)
    if t == OptionType.ATTACH: return score_attach(o, obs, me)
    if t == OptionType.ABILITY: return 3000
    if t == OptionType.EVOLVE: return 2800
    if t == OptionType.PLAY: return score_play(o, obs)
    if t == OptionType.RETREAT: return score_retreat(obs)
    if t == OptionType.CARD: return score_card(o, obs)
    return _PRIORITY.get(t, 0)


def score_card(o, obs):
    """Card-selection contexts (SETUP/SWITCH/TO_HAND/DISCARD/...) — no more coin flips."""
    st = obs.current
    op = st.players[1 - st.yourIndex]
    op_active = op.active[0] if op.active else None
    ctx = obs.select.context
    player = o.playerIndex if o.playerIndex is not None else st.yourIndex
    card = _card_at(obs, o.area, o.index, player)
    if card is None:
        return 0

    if ctx in _PROMOTE_CTX:                       # start / promote / bench MY best attacker
        q = _attacker_quality(card.id)
        ready = 0                                 # bonus if it can attack right now
        if op_active is not None and hasattr(card, "energies") and _best_damage(card, op_active) > 0:
            ready = 500
        return q + ready
    if ctx in _KEEP_CTX:                          # fetch/keep the most useful card
        return _card_usefulness(card.id)
    if ctx in _DISCARD_CTX:                       # discard the LEAST useful
        return -_card_usefulness(card.id)
    if ctx in _TARGET_CTX:                        # damage the highest-prize opponent Pokémon
        return 100 * _CARD.get(card.id, (0, 0, 0, [], 1))[4]
    return 50                                     # unknown card context: neutral


def score_attack(o, my_active, op_active):
    if op_active is None:
        return 1000

    damage, cost_tuple = _ATK.get(o.attackId, (0, ()))
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

    # 3) Otherwise low; ~0 when healthy, a bit higher if a strong bench attacker wants in
    hp_frac = my_active.hp / max(1, my_active.maxHp)
    if hp_frac > 0.75:
        return -1
    return 100 + bench_best // 20

def score_play(o, obs):
    my_index = obs.current.yourIndex
    me = obs.current.players[my_index]
    card = _card_at(obs, o.area, o.index, my_index)

    if card is None:
        return 2000
    if card.id in _IS_POKEMON:
        return 2400                       # developing the board is always good

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

    # 1) Lookahead: does this attach UNBLOCK a KO on the opponent's active?

    if opponent_active_card is not None:
        now = _best_damage(target_pokemon, opponent_active_card, extra_energy=0)
        after = _best_damage(target_pokemon, opponent_active_card, extra_energy=1)
        if now < opponent_active_card.hp <= after:
            return  4000 if is_active else 2900    # active can cash it this turn -> top priority

    # 2) Otherwise is target a real attacker that still needs energy?

    damaging = [(_ATK[a][0], len(_ATK[a][1])) for a in _CARD[target_pokemon.id][3]
                if a in _ATK and _ATK[a][0] > 0]

    if not damaging:
        return 400

    best_dmg = max(d for d, _ in damaging)
    cheapest = min(c for _, c in damaging)

    if len(target_pokemon.energies) >= cheapest:
        return 600

    return (2600 if is_active else 2400) + min(best_dmg, 300) // 100


