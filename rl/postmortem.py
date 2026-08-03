"""Post-mortem forensics for one Kaggle episode replay — the loss-anatomy tool.

Answers, from a cached replay, the questions every leaderboard loss raises:
what were the decks, how did the game unfold turn by turn, what was in OUR
hand at each decision, which options did the engine offer and which did we
take, and WHY did we lose (prizes / deck-out / benched-out / agent error)?
Also writes the watchable replay page via rl/replay.py.

The trajectory and hand views come from our seat's own observations, so
everything printed is information our agent actually had — the audit never
peeks at hidden zones (the opponent's hand/deck stay hidden in the replay
too; prize cards are counted, not named).

Usage:
  uv run python -m rl.postmortem 85467275                # fetch/cached + full report
  uv run python -m rl.postmortem 85467275 --decisions    # + decoded option menus
  uv run python -m rl.postmortem 85467275 --seat 0       # override seat autodetect
  uv run python -m rl.postmortem 85467275 --no-html      # skip replay page
  uv run python -m rl.postmortem --batch replays/m8_taxonomy   # M8.0 aggregate:
      audit_flags + classify_end over a directory of env.toJSON() files
      (rl/eval.py play_games json_prefix output) -> flag-frequency table

Seat autodetect: the seat whose decklist hash matches a deck we ship or have
league-enrolled (decks/*.csv, data/league/decks/*.csv). --seat overrides.
In --batch mode the default seat is the `_a<slot>` filename suffix play_games
writes (audit agent_a's play); --batch-seat {0,1,auto} overrides.
"""
import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from cg.api import AreaType, CardType, OptionType, SelectContext, all_card_data

from rl.kaggle_ingest import (RAW_DIR, deck_hash, fetch_agent_logs,
                              fetch_episode, parse_episode)

ROOT = Path(__file__).resolve().parent.parent

_CARDS = {c.cardId: c for c in all_card_data()}
_CARDS_BY_NAME = {getattr(c, "name", f"card{c.cardId}"): c for c in all_card_data()}

# Trainers that DISCARD the hand as a cost (vs. shuffle-back, which is safe).
# Flagged in the decision audit when key cards are thrown away with them.
HAND_DISCARD_TRAINERS = {"Carmine"}


def card_name(cid) -> str:
    c = _CARDS.get(cid)
    return getattr(c, "name", f"id{cid}") if c else f"id{cid}"


def _load(episode_id: int) -> dict:
    gz = RAW_DIR / f"episode_{episode_id}.json.gz"
    if gz.exists():
        return json.loads(gzip.open(gz, "rt").read())
    return fetch_episode(episode_id)          # [NET] on cache miss


def _known_deck_hashes() -> set[str]:
    hashes = set()
    for pat in ("decks/*.csv", "decks/gen/*.csv", "data/league/decks/*.csv"):
        for csv in ROOT.glob(pat):
            try:
                ids = [int(x) for x in csv.read_text().split() if x.strip()]
            except ValueError:
                continue
            if len(ids) == 60:
                hashes.add(deck_hash(ids))
    return hashes


def detect_our_seat(parsed) -> int | None:
    """Seat whose deck hash matches a repo deck; None if neither/both match."""
    known = _known_deck_hashes()
    matches = [s for s in (0, 1)
               if parsed.decks[s] and deck_hash(parsed.decks[s]) in known]
    return matches[0] if len(matches) == 1 else None


# ---------------------------------------------------------------------------
# state extraction (schema: obs.current.players[i], verified on ep 85467275)
# ---------------------------------------------------------------------------
def _cur(step_agent: dict) -> dict | None:
    cur = (step_agent.get("observation") or {}).get("current")
    return cur if cur and cur.get("players") else None


def _active(player: dict) -> dict:
    act = player.get("active") or [{}]
    return act[0] if act and act[0] else {}


def side_brief(player: dict) -> dict:
    a = _active(player)
    return {
        "prizes_left": len(player.get("prize") or []),
        "deck": player.get("deckCount"),
        "active": card_name(a["id"]) if a.get("id") else None,
        "hp": a.get("hp"),
        "energy": len(a.get("energies") or []),
        "bench": [card_name(b.get("id")) for b in (player.get("bench") or [])],
        "hand_n": player.get("handCount"),
    }


def hand_names(player: dict) -> list[str]:
    return [card_name(c.get("id")) for c in (player.get("hand") or [])
            if isinstance(c, dict)]


def _fmt_side(b: dict) -> str:
    act = f"{b['active']}({b['hp']}hp,{b['energy']}e)" if b["active"] else "-"
    return (f"prizes_left={b['prizes_left']} deck={b['deck']} "
            f"active={act} bench={b['bench']}")


# ---------------------------------------------------------------------------
# decision decoding (best-effort: forensics, not a rules engine)
# ---------------------------------------------------------------------------
def describe_option(opt: dict, cur: dict, us: int) -> str:
    """Resolve one select option to '<TYPE>:<card>' where the index is mappable."""
    try:
        t = OptionType(opt.get("type")).name
    except ValueError:
        t = f"type{opt.get('type')}"
    idx = opt.get("index")
    target = ""
    players = cur.get("players") or []
    if "area" in opt and opt.get("playerIndex") is not None:
        try:
            area = AreaType(opt["area"])
            p = players[opt["playerIndex"]]
            zone = {AreaType.HAND: p.get("hand"), AreaType.BENCH: p.get("bench"),
                    AreaType.ACTIVE: p.get("active"),
                    AreaType.DISCARD: p.get("discard")}.get(area)
            who = "us" if opt["playerIndex"] == us else "opp"
            if zone and idx is not None and idx < len(zone) and isinstance(zone[idx], dict):
                target = f":{who}.{area.name}[{idx}]={card_name(zone[idx].get('id'))}"
            else:
                target = f":{who}.{area.name}[{idx}]"
        except (ValueError, IndexError):
            target = f":area{opt.get('area')}[{idx}]"
    elif idx is not None:
        hand = players[us].get("hand") if len(players) > us else None
        if t in ("PLAY", "EVOLVE", "ATTACH", "DISCARD") and hand and idx < len(hand) \
                and isinstance(hand[idx], dict):
            target = f":hand[{idx}]={card_name(hand[idx].get('id'))}"
        else:
            target = f"[{idx}]"
    return t + target


def parse_net_log(payload: list) -> dict[int, dict]:
    """Parse `NN|{json}` net-internals lines (submission/main.py, M19) out of a
    Kaggle agent-logs payload into {obs step -> decision record}.

    Records carry: s=obs step, t=turn, c=SelectContext, a=chosen indices,
    sc=per-option logits; on the turn's plan commit also p=plan index and
    psc=plan logits. `NN|ERR|` lines are counted, printed once, and skipped
    (the shipped logger never hides its own failures)."""
    recs, errs = {}, []
    for entry in payload or []:
        for cell in entry if isinstance(entry, list) else [entry]:
            for line in (cell.get("stderr") or "").splitlines():
                if not line.startswith("NN|"):
                    continue
                if line.startswith("NN|ERR|"):
                    errs.append(line)
                    continue
                try:
                    rec = json.loads(line[3:])
                except ValueError:
                    errs.append(line)
                    continue
                if isinstance(rec, dict) and rec.get("s") is not None:
                    recs[int(rec["s"])] = rec
    if errs:
        print(f"WARNING: {len(errs)} unparseable NN| lines, first: {errs[0][:120]}")
    return recs


def load_net_log(episode_id: int, us: int) -> dict[int, dict] | None:
    """Cached-first agent-log load; [NET] fetch on miss (like `_load`), but a
    fetch failure degrades to None — the rest of the report still works."""
    try:
        return parse_net_log(fetch_agent_logs(episode_id, us))
    except RuntimeError as e:
        print(f"(no agent log: {e})")
        return None


def _action_for(steps: list, i: int, us: int):
    """The action answering step i's select: recorded on the NEXT step.

    Replay convention (env.toJSON() and the Kaggle cache alike, verified
    2026-07-12 against the live pilot's scores on mirror_g000): steps[i] holds
    the observation GIVEN to the agent, and the agent's response is recorded
    at steps[i+1].action — pairing select and action at the same index reads
    every decision one prompt late (the M8.0 instrument bug)."""
    if i + 1 < len(steps):
        return steps[i + 1][us].get("action")
    return None


def iter_decisions(steps: list, us: int):
    """(step_idx, turn, context_name, [option strs], chosen_action, cur) per prompt."""
    for i, step in enumerate(steps):
        st = step[us]
        sel = (st.get("observation") or {}).get("select")
        cur = _cur(st)
        if not sel or not cur:
            continue
        try:
            ctx = SelectContext(sel.get("context")).name
        except ValueError:
            ctx = f"context{sel.get('context')}"
        opts = [describe_option(o, cur, us) for o in (sel.get("option") or [])]
        yield i, cur.get("turn"), ctx, opts, _action_for(steps, i, us), cur


# ---------------------------------------------------------------------------
# loss classification + agent-error flags
# ---------------------------------------------------------------------------
def classify_end(last_cur: dict, us: int, our_reward) -> str:
    """The last observation precedes the killing blow, so classify from the
    state we were in when we made our final decision."""
    b_us = side_brief(last_cur["players"][us])
    b_op = side_brief(last_cur["players"][1 - us])
    if our_reward is not None and our_reward > 0:
        return "WE WON"
    if not b_us["bench"]:
        return ("loss: BENCHED-OUT — final KO with an empty bench "
                f"(opponent still had {b_op['prizes_left']} prizes to take)")
    if (b_us["deck"] or 0) == 0:
        return "loss: DECK-OUT — we could not draw"
    if b_op["prizes_left"] <= 1:
        return "loss: PRIZES — opponent took their prizes"
    return "loss: unclassified (timeout/error? check statuses)"


def audit_flags(steps: list, us: int) -> list[str]:
    """Mechanical checks for known pilot failure modes. Facts, not judgments.
    Every flag starts with a stable "[kind]" tag — the M8.0 batch aggregator
    counts by it.

    Per-TURN checks (empty-bench, evolve-left) look at the state of the LAST
    main prompt of each turn: fixing it later in the same turn is fine; ending
    the turn in the bad state is the failure.
    """
    flags = []
    turn_bench_miss: dict[int, list[str]] = {}   # turn -> declined basics, latest prompt
    turn_bench_state: dict[int, bool] = {}       # turn -> bench empty at last MAIN prompt
    for i, turn, ctx, opts, action, cur in iter_decisions(steps, us):
        if ctx != "MAIN":
            continue
        players = cur["players"]
        hand = hand_names(players[us])
        bench = (players[us].get("bench") or [])
        chosen = set(action) if isinstance(action, list) else {action}

        # (1) hand-discard trainer played while holding evolution cards
        for j, o in enumerate(opts):
            if j in chosen and o.startswith("PLAY") and any(
                    t in o for t in HAND_DISCARD_TRAINERS):
                evos = [h for h in hand if _CARDS_BY_NAME.get(h)
                        and getattr(_CARDS_BY_NAME[h], "evolvesFrom", None)]
                if evos:
                    flags.append(f"[hand-discard] s{i} t{turn}: hand-discard "
                                 f"trainer ({o}) threw away evolution cards {evos}")

        # (2) empty bench + benchable basic in hand: remember the state at the
        # LAST main prompt of each turn; flag only turns that stayed that way.
        benchable = [o for j, o in enumerate(opts)
                     if o.startswith("PLAY:hand") and _is_basic_pokemon(o)
                     and j not in chosen]
        turn_bench_state[turn] = not bench
        turn_bench_miss[turn] = benchable if not bench else []
    for turn, declined in sorted(turn_bench_miss.items()):
        if turn_bench_state.get(turn) and declined:
            flags.append(f"[empty-bench] t{turn}: turn ENDED with an empty bench "
                         f"while a benchable basic sat in hand: {declined}")

    flags += _setup_taxonomy_flags(steps, us)    # M8.0 checks (3)-(6)
    return flags


def _is_basic_pokemon(opt_str: str) -> bool:
    name = opt_str.split("=", 1)[-1]
    c = _CARDS_BY_NAME.get(name)
    return bool(c and getattr(c, "basic", False))


# ---------------------------------------------------------------------------
# M8.0 setup taxonomy — checks (3)-(6). Same facts-not-judgments contract.
# ---------------------------------------------------------------------------
_KEEP_CONTEXTS = {int(SelectContext.TO_HAND), int(SelectContext.LOOK),
                  int(SelectContext.NOT_MOVE)}
HOARD_MIN_TURNS = 4   # a playable trainer ignored across this many turns = hoarded


def _iter_selects(steps: list, us: int):
    """Raw (step_idx, turn, context_int, sel, chosen_set, cur) per prompt —
    the dict-level twin of iter_decisions for checks that need option fields
    (inPlayArea/inPlayIndex, deck zone) the description strings drop."""
    for i, step in enumerate(steps):
        st = step[us]
        sel = (st.get("observation") or {}).get("select")
        cur = _cur(st)
        if not sel or not cur:
            continue
        action = _action_for(steps, i, us)
        chosen = set(action) if isinstance(action, list) else {action}
        yield i, cur.get("turn"), sel.get("context"), sel, chosen, cur


def _option_card_id(opt: dict, sel: dict, cur: dict, us: int):
    """Card id an option refers to, or None. Unlike describe_option this also
    resolves the DECK zone (fetch prompts) and defaults the area to HAND (the
    play-tier lesson: real PLAY options carry no `area`)."""
    idx = opt.get("index")
    if idx is None:
        return None
    pi = opt.get("playerIndex")
    player = cur["players"][pi if pi is not None else us]
    area = opt.get("area")
    zone = {int(AreaType.DECK): sel.get("deck"), int(AreaType.HAND): player.get("hand"),
            int(AreaType.DISCARD): player.get("discard"),
            int(AreaType.ACTIVE): player.get("active"),
            int(AreaType.BENCH): player.get("bench"),
            }.get(int(area)) if area is not None else player.get("hand")
    try:
        card = zone[idx]
        return card.get("id") if isinstance(card, dict) else None
    except (TypeError, IndexError):
        return None


def _poke_shim(d: dict):
    """Replay-JSON Pokémon dict -> the .id/.hp/.energies object rl.combat expects."""
    return SimpleNamespace(id=d.get("id"), hp=d.get("hp") or 0,
                           maxHp=d.get("maxHp") or d.get("hp") or 0,
                           energies=list(d.get("energies") or []), tools=[])


def _is_trainer_name(name: str) -> bool:
    c = _CARDS_BY_NAME.get(name)
    if c is None:
        return False
    card_type = getattr(c, "cardType", None)
    if card_type is not None:
        return card_type not in (CardType.POKEMON, CardType.BASIC_ENERGY,
                                 CardType.SPECIAL_ENERGY)
    # synthetic test pools carry no cardType: non-Pokémon heuristic
    return not getattr(c, "basic", False) and getattr(c, "evolvesFrom", None) is None


def _my_board_and_hand_names(cur: dict, us: int) -> set:
    me = cur["players"][us]
    names = set(hand_names(me))
    for d in [_active(me)] + list(me.get("bench") or []):
        if isinstance(d, dict) and d.get("id"):
            names.add(card_name(d["id"]))
    return names


def _attach_off_racer(opt: dict, cur: dict, us: int, i: int, turn) -> str | None:
    """(4) energy went to a Pokémon that is NOT the board's fastest racer while
    that racer still needed energy — the M7.2b starvation failure, per move."""
    from rl.combat import UNREACHABLE, _turns_to_first_ko, _turns_to_ready

    opponent = cur["players"][1 - us]
    op_active = _active(opponent)
    if not op_active.get("id"):
        return None
    op_shim = _poke_shim(op_active)
    me = cur["players"][us]
    board = []                                     # (area, index, shim)
    active = _active(me)
    if active.get("id"):
        board.append((int(AreaType.ACTIVE), 0, _poke_shim(active)))
    for bi, b in enumerate(me.get("bench") or []):
        if isinstance(b, dict) and b.get("id"):
            board.append((int(AreaType.BENCH), bi, _poke_shim(b)))
    if len(board) < 2:
        return None                                # one target: nothing to misroute
    racer = min(board, key=lambda t: _turns_to_first_ko(t[2], op_shim))
    if _turns_to_first_ko(racer[2], op_shim) >= UNREACHABLE:
        return None                                # nobody can ever KO: no race
    if _turns_to_ready(racer[2], op_shim) == 0:
        return None                                # racer loaded: attach anywhere
    target = (opt.get("inPlayArea"), opt.get("inPlayIndex"))
    if target[0] is None or (int(target[0]), target[1]) == (racer[0], racer[1]):
        return None
    return (f"[attach-off-racer] s{i} t{turn}: energy attached to "
            f"area{target[0]}[{target[1]}] while the best racer "
            f"{card_name(racer[2].id)} was still unloaded")


def _attach_saturated(opt: dict, cur: dict, us: int, i: int, turn) -> str | None:
    """M19: energy attached to a target whose charged-best attack is ALREADY
    paid (surplus attach — the live 3-energies-on-a-1-cost-Solrock defect).
    Complements [attach-off-racer], which only catches misrouting while the
    racer is unloaded, not overfeeding a charged one.

    M41: SKIPPED for attackers whose damage scales with their OWN attached
    energy. Teal Mask Ogerpon ex's `Myriad Leaf Shower` is paid at 3 energy and
    then gains +30 damage for every further attachment (180 -> 300 between 3 and
    7), so "already charged" is not "saturated" and the flag was reporting
    correct play as a defect. Cost-satisfaction and damage-satisfaction are the
    same thing only for flat attackers, which is what M19 was written against.

    M41 (2026-08-03): the FALSE-NEGATIVE half of the same confusion. The test
    was `_turns_to_ready == 0`, which routes through `_charged_best` and its
    `if dmg <= 0: continue` — so on any attacker whose printed damage is 0 the
    gap is UNREACHABLE and this flag could never fire AT ALL. On our own
    Alakazam #743 (`Powerful Hand`, prints 0, really 20 x hand) it reported 1.5%
    over-attach and blamed Kadabra and Abra while the true rate was 33.7%. Four
    milestones of post-mortems missed the defect because the forensics shared
    the agent's blind spot. `energy_is_dead` asks about affordability and
    retreat cost instead of damage, so no printed number can blind it; it also
    subsumes the own-energy-scaling skip above, which is kept as the explicit
    early return because it documents the other half of the story.
    """
    from rl.combat import energy_is_dead
    from rl.scaling import SCALING_ATTACKS

    # modes where more energy on the ATTACKER means more damage
    _SELF_SCALING = {"my_nrg", "both_nrg", "team_nrg"}

    me = cur["players"][us]
    area, idx = opt.get("inPlayArea"), opt.get("inPlayIndex")
    if area is None:
        return None
    zone = {int(AreaType.ACTIVE): [_active(me)],
            int(AreaType.BENCH): me.get("bench") or []}.get(int(area))
    if zone is None or idx is None or idx >= len(zone):
        return None
    target = zone[idx]
    if not isinstance(target, dict) or not target.get("id"):
        return None
    shim = _poke_shim(target)
    from rl.combat import _CARD
    if any(SCALING_ATTACKS.get(a, ("", 0, 0))[0] in _SELF_SCALING
           for a in _CARD.get(shim.id, (None, None, 0, [], 1))[3]):
        return None
    if energy_is_dead(shim.id, shim.energies):
        return (f"[over-attach] s{i} t{turn}: energy attached to "
                f"{card_name(shim.id)} which could already pay for every attack "
                f"it has and its retreat ({len(shim.energies)} energy attached)")
    return None


def _setup_taxonomy_flags(steps: list, us: int) -> list[str]:
    flags = []
    turn_evolve_left: dict = {}                    # turn -> declined EVOLVE names
    trainer_offer_turns: dict[str, set] = {}       # trainer name -> turns playable
    trainers_played: set = set()

    for i, turn, ctx, sel, chosen, cur in _iter_selects(steps, us):
        options = sel.get("option") or []
        if ctx == int(SelectContext.MAIN):
            # (3) evolutions offered but left in hand at turn end (per turn,
            # latest-prompt state — same pattern as the empty-bench check)
            declined = [card_name(cid) for j, o in enumerate(options)
                        if o.get("type") == int(OptionType.EVOLVE) and j not in chosen
                        and (cid := _option_card_id(o, sel, cur, us))]
            turn_evolve_left[turn] = declined

            for j, o in enumerate(options):
                # (4) attach off the racer / onto a saturated target
                # (chosen ATTACH options only)
                if j in chosen and o.get("type") == int(OptionType.ATTACH):
                    for check in (_attach_off_racer, _attach_saturated):
                        try:
                            flag = check(o, cur, us, i, turn)
                        except Exception:          # forensics, not a rules engine
                            flag = None
                        if flag:
                            flags.append(flag)
                # (6) trainer-hoarding bookkeeping (effect-text-free proxy for
                # "discard-retrieval item unused": a PLAY-able trainer ignored
                # for HOARD_MIN_TURNS distinct turns and never played)
                if o.get("type") == int(OptionType.PLAY):
                    cid = _option_card_id(o, sel, cur, us)
                    name = card_name(cid) if cid else None
                    if name and _is_trainer_name(name):
                        trainer_offer_turns.setdefault(name, set()).add(turn)
                        if j in chosen:
                            trainers_played.add(name)
        elif ctx in _KEEP_CONTEXTS:
            # (5) fetch/keep chose an evolution whose basis is nowhere
            available = _my_board_and_hand_names(cur, us)
            for j in chosen:
                if not isinstance(j, int) or j >= len(options):
                    continue
                cid = _option_card_id(options[j], sel, cur, us)
                card = _CARDS.get(cid)
                basis = getattr(card, "evolvesFrom", None) if card else None
                if basis and basis not in available:
                    flags.append(f"[fetch-dead-evolution] s{i} t{turn}: fetched "
                                 f"{card_name(cid)} whose basis {basis} is "
                                 f"nowhere in play or hand")

    for turn, declined in sorted(turn_evolve_left.items()):
        if declined:
            flags.append(f"[evolve-left] t{turn}: turn ENDED with evolution(s) "
                         f"offered but not played: {declined}")
    for name, turns in sorted(trainer_offer_turns.items()):
        if len(turns) >= HOARD_MIN_TURNS and name not in trainers_played:
            flags.append(f"[trainer-hoarded] {name} was playable across "
                         f"{len(turns)} turns and never played")
    return flags


# ---------------------------------------------------------------------------
# batch aggregation (M8.0): a directory of env.toJSON() files -> taxonomy table
# ---------------------------------------------------------------------------
def _seat_for(path: Path, raw: dict, seat: str) -> int | None:
    if seat in ("0", "1"):
        return int(seat)
    if seat == "a":
        m = re.search(r"_a([01])\.json(\.gz)?$", path.name)
        if m:
            return int(m.group(1))
    parsed = parse_episode(raw, episode_id=0)      # fall back to deck-hash detect
    return detect_our_seat(parsed)


def batch(dir_path: str, seat: str = "a") -> None:
    """Aggregate classify_end + audit_flags over every *.json[.gz] in dir_path
    (rl/eval.py play_games `json_prefix` output, or the gzipped Kaggle episode
    cache in data/kaggle/raw/). Prints ending counts and a flag-kind frequency
    table with one example each — the M8.0 setup-mistake taxonomy."""
    files = sorted(Path(dir_path).glob("*.json")) + sorted(Path(dir_path).glob("*.json.gz"))
    end_counts, kind_counts = Counter(), Counter()
    example: dict[str, str] = {}
    n_games = 0
    for f in files:
        if f.suffix == ".gz":
            with gzip.open(f, "rt") as fh:
                raw = json.load(fh)
        else:
            raw = json.loads(f.read_text())
        steps = raw.get("steps") or []
        if not steps:
            continue
        us = _seat_for(f, raw, seat)
        if us is None:
            print(f"  (skipped {f.name}: seat undetectable)")
            continue
        last_cur = None
        for step in steps:
            cur = _cur(step[us])
            if cur:
                last_cur = cur
        if last_cur is None:
            continue
        n_games += 1
        rewards = raw.get("rewards") or [None, None]
        verdict = classify_end(last_cur, us, rewards[us])
        end_counts[verdict.split(" — ")[0].split(" (")[0]] += 1
        for flag in audit_flags(steps, us):
            kind = flag.split("]", 1)[0].lstrip("[")
            kind_counts[kind] += 1
            example.setdefault(kind, f"{f.name}: {flag}")

    print(f"=== batch post-mortem: {n_games} games from {dir_path} ===")
    print("\n-- endings --")
    for name, count in end_counts.most_common():
        print(f"  {count:>4}  {name}")
    print("\n-- agent-error flags (kind: count across games) --")
    for name, count in kind_counts.most_common():
        print(f"  {count:>4}  {name}")
        print(f"        e.g. {example[name]}")
    if not kind_counts:
        print("  (none)")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def report(episode_id: int, seat: int | None = None, decisions: bool = False,
           html: bool = True) -> None:
    raw = _load(episode_id)
    parsed = parse_episode(raw, episode_id=episode_id)
    us = seat if seat is not None else detect_our_seat(parsed)
    if us is None:
        raise SystemExit("could not autodetect our seat (no deck-hash match) — pass --seat")
    steps = raw["steps"]

    print(f"=== episode {episode_id} ===")
    for s in (0, 1):
        tag = "US " if s == us else "OPP"
        h = deck_hash(parsed.decks[s])[:12] if parsed.decks[s] else "?"
        print(f"{tag} seat {s}: team={parsed.team_names[s]} reward={parsed.rewards[s]} "
              f"deck={h} status={parsed.statuses[s]}")

    print("\n=== DECKS ===")
    for s in (0, 1):
        label = "US" if s == us else "OPPONENT"
        print(f"-- {label} --")
        if parsed.decks[s]:
            for name, n in sorted(Counter(card_name(c) for c in parsed.decks[s]).items(),
                                  key=lambda kv: -kv[1]):
                print(f"   {n}x {name}")

    print("\n=== TRAJECTORY (our observations; hand is ours) ===")
    prev, last_cur = None, None
    for i, step in enumerate(steps):
        cur = _cur(step[us])
        if not cur:
            continue
        last_cur = cur
        b_us, b_op = side_brief(cur["players"][us]), side_brief(cur["players"][1 - us])
        hand = hand_names(cur["players"][us])
        key = (cur.get("turn"), b_us["prizes_left"], b_us["deck"], b_us["active"],
               b_op["prizes_left"], b_op["active"], tuple(hand))
        if key != prev:
            prev = key
            print(f"t{cur.get('turn'):>2} s{i:>3} | US  {_fmt_side(b_us)}")
            print(f"          |     hand={hand}")
            print(f"          | OPP {_fmt_side(b_op)}")

    net = load_net_log(episode_id, us)
    if net is not None:
        commits = sum(1 for r in net.values() if "p" in r)
        print(f"\nnet log: {len(net)} decisions with net internals, "
              f"{commits} plan commits"
              + ("" if net else " — empty (pre-M19 submission?)"))

    if decisions:
        print("\n=== DECISIONS (context, options, chosen) ===")
        for i, turn, ctx, opts, action, _ in iter_decisions(steps, us):
            rec = (net or {}).get(i)
            print(f"s{i:>3} t{turn:>2} {ctx:<12} chose={action}")
            if rec and "p" in rec:
                psc = rec.get("psc") or []
                print(f"     plan committed: #{rec['p']} of {len(psc)} "
                      f"(logit {psc[rec['p']]:+.2f}, "
                      f"runner-up {max((x for k, x in enumerate(psc) if k != rec['p']), default=float('nan')):+.2f})")
            logits = rec.get("sc") if rec else None
            for j, o in enumerate(opts):
                mark = " <== " if (isinstance(action, list) and j in action) else "     "
                net_note = ""
                if logits is not None and j < len(logits):
                    net_note = f"  logit={logits[j]:+.2f}"
                print(f"    {mark}[{j}] {o}{net_note}")

    print("\n=== VERDICT ===")
    if last_cur:
        print(classify_end(last_cur, us, parsed.rewards[us]))
        b = side_brief(last_cur["players"][us])
        print(f"end: turn={last_cur.get('turn')} US {_fmt_side(b)} "
              f"hand={hand_names(last_cur['players'][us])}")
    flags = audit_flags(steps, us)
    if flags:
        print("\n=== AGENT-ERROR FLAGS ===")
        for f in flags:
            print(" !", f)
    else:
        print("(no mechanical agent-error flags)")

    if html and parsed.has_visualize:
        from rl.replay import save_replay

        class _Shim:                                  # adapt cached JSON to save_replay
            pass
        env = _Shim()
        env.steps = steps
        env.state = [_Shim(), _Shim()]
        env.state[0].reward, env.state[1].reward = parsed.rewards
        who = [parsed.team_names[s] or f"seat{s}" for s in (0, 1)]
        who[us] += "(us)"
        name = f"kaggle_ep{episode_id}"
        if save_replay(env, name, (who[0], who[1])):
            print(f"\nreplay page: replays/{name}.html")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("episode", type=int, nargs="?")
    p.add_argument("--seat", type=int, default=None, choices=(0, 1))
    p.add_argument("--decisions", action="store_true",
                   help="print every decoded option menu + choice")
    p.add_argument("--no-html", dest="html", action="store_false")
    p.add_argument("--batch", metavar="DIR", default=None,
                   help="aggregate audit_flags/classify_end over a directory "
                        "of env.toJSON() replays (M8.0 taxonomy)")
    p.add_argument("--batch-seat", default="a", choices=("a", "0", "1", "auto"),
                   help="--batch seat: 'a' = the _a<slot> filename suffix "
                        "(default), fixed 0/1, or deck-hash 'auto'")
    a = p.parse_args()
    if a.batch:
        batch(a.batch, a.batch_seat)
    elif a.episode is None:
        p.error("an episode id is required unless --batch is given")
    else:
        report(a.episode, seat=a.seat, decisions=a.decisions, html=a.html)
