# Pilot API Cheatsheet

Quick reference for building `rl/generic_pilot.py`. How to read the observation, the combat
tables from `rl/encoders.py` (M3), and how to resolve an option to the card/Pokémon it refers to.
(The positional tuples below are the tech-debt we'll dataclass later — for now, the indices.)

---

## Observation shape (`obs = to_observation_class(obs_dict)`)

```
obs.select                 # None at deck-return; else the current decision
obs.select.context         # SelectContext (MAIN, SETUP_ACTIVE_POKEMON, TO_HAND, ...)
obs.select.option          # list[Option] — the choices you score
obs.select.maxCount        # how many indices to return
obs.current                # the State
obs.search_begin_input     # present only in real games (needed for MCTS; ignore for the pilot)
```

### `State` (`obs.current`)
```
.turn .turnActionCount .yourIndex .firstPlayer
.supporterPlayed .stadiumPlayed .energyAttached .retreated
.result           # -1 ongoing; 0/1 winner seat; 2 draw
.stadium          # list[Card] (0 or 1)
.players          # [PlayerState, PlayerState]
me = obs.current.players[obs.current.yourIndex]
op = obs.current.players[1 - obs.current.yourIndex]
```

### `PlayerState`
```
.active           # list[Pokemon | None], size 0 or 1   -> active = ps.active[0] if ps.active else None
.bench            # list[Pokemon]
.discard          # list[Card]     (visible for BOTH players)
.prize            # list[Card|None] (face-down = None; count = len)
.hand             # list[Card] for ME, None for opponent
.handCount .deckCount .benchMax
.poisoned .burned .asleep .paralyzed .confused   # bools (active's status)
```

### `Pokemon`  (an in-play Pokémon)
```
.id               # CardData id  -> use with _CARD / _ATK / FEAT
.hp .maxHp        # current / max HP
.energies         # list[EnergyType-int] currently attached  -> len() = energy count
.energyCards      # list[Card] (the actual energy cards)
.tools            # list[Card] (attached tools)
.appearThisTurn   # bool
```

### `Card`
```
.id .serial .playerIndex
```

### `Option` (fields depend on `.type`)
```
.type             # OptionType (see below)
.attackId         # ATTACK -> key into _ATK
.area .index      # the card this option is about (area = AreaType); resolve with _card_at()
.playerIndex      # whose card
.inPlayArea .inPlayIndex   # ATTACH/EVOLVE: the target Pokémon in play
.number .count .toolIndex .energyIndex .specialConditionType .cardId
```

---

## Enums (int values)

```
EnergyType : COLORLESS 0, GRASS 1, FIRE 2, WATER 3, LIGHTNING 4, PSYCHIC 5,
             FIGHTING 6, DARKNESS 7, METAL 8, DRAGON 9, RAINBOW 10, TEAM_ROCKET 11
AreaType   : DECK 1, HAND 2, DISCARD 3, ACTIVE 4, BENCH 5, PRIZE 6, STADIUM 7, ... LOOKING 12
OptionType : NUMBER 0, YES 1, NO 2, CARD 3, TOOL_CARD 4, ENERGY_CARD 5, ENERGY 6,
             PLAY 7, ATTACH 8, EVOLVE 9, ABILITY 10, DISCARD 11, RETREAT 12, ATTACK 13, END 14
SelectContext: MAIN 0, SETUP_ACTIVE_POKEMON 1, SETUP_BENCH_POKEMON 2, SWITCH 3, TO_ACTIVE 4,
               TO_HAND 7, DISCARD 8, ATTACH_FROM 21, ...
```
Import them: `from cg.api import EnergyType, AreaType, OptionType, SelectContext`.

---

## Combat tables & helpers (`from rl.encoders import _CARD, _ATK, _can_afford, _best_damage`)

```python
_CARD[card_id]   # tuple: (weakness, resistance, energy_type, attacks, prize)
#   [0] weakness    : EnergyType | None   (defender takes x2 if == attacker type)
#   [1] resistance  : EnergyType | None   (defender takes -30 if == attacker type)
#   [2] energy_type : int                 (the Pokémon's OWN type; use for weakness math)
#   [3] attacks     : list[int]           (attackIds this card can use)
#   [4] prize       : int                 (prizes given on KO: 1/2/3)

_ATK[attack_id]  # tuple: (damage, cost)
#   [0] damage : int             (0 for variable/effect attacks — a known gap)
#   [1] cost   : tuple[int]      (EnergyType ints; 0 = colorless/any)

_can_afford(pokemon.energies, cost)      # -> bool: can this Pokémon pay that attack cost?
_best_damage(attacker, target, extra_energy=0)  # -> int: max AFFORDABLE damage after
                                                 # weakness/resistance; extra_energy simulates
                                                 # attaching +N of the attacker's type
```

Also available: `FEAT` (numpy card-feature matrix) — but for the rule pilot the two tables above
are usually enough.

---

## Resolve an option → the card/Pokémon it refers to

```python
def _card_at(obs, area, index, player):
    ps = obs.current.players[player]
    zone = {2: ps.hand, 3: ps.discard, 4: ps.active, 5: ps.bench,
            1: obs.select.deck, 7: obs.current.stadium}.get(int(area)) if area is not None else None
    try:
        return zone[index]
    except (TypeError, IndexError):
        return None
```

---

## Recipes for the scorers you're writing

**score_attack(o, my_active, op_active)**
```python
dmg, cost = _ATK[o.attackId]
atk_type = _CARD[my_active.id][2]
weak = _CARD[op_active.id][0]
if weak is not None and int(weak) == atk_type: dmg *= 2
# ... resistance -30 ... ; KO if dmg >= op_active.hp -> big + 100*_CARD[op_active.id][4]
```

**score_attach(o, obs, me)**  ← what you're on
```python
my_index = obs.current.yourIndex
target = _card_at(obs, o.inPlayArea, o.inPlayIndex, my_index)   # the Pokémon getting energy
if target is None:
    return 500
# is it a real attacker? (has a damaging attack)  and does it still NEED energy?
cheapest = min((len(_ATK[a][1]) for a in _CARD[target.id][3]
                if a in _ATK and _ATK[a][0] > 0), default=99)
needs_energy = len(target.energies) < cheapest
best_dmg = max((_ATK[a][0] for a in _CARD[target.id][3] if a in _ATK), default=0)
is_active = (int(o.inPlayArea) == 4)     # AreaType.ACTIVE
# high score if a strong attacker that still needs energy (prefer the active); low otherwise
```
Idea: `2600` when `needs_energy and best_dmg` is high (esp. active), taper down; `~600` if it's
already loaded or not an attacker. Don't waste energy on benchwarmers.
