"""M46 C3 — mine card-fact tables from the raw replay corpus.

The engine is a native library and NO card carries ability/effect text in our
features, so generic guard rules (docs/M46-plan.md Track C) build on facts
mined EMPIRICALLY from the ~7.5k cached raw episodes — both seats, teacher
seats included. Four tables -> data/rule_facts.json (committed via
`git add -f`; regenerated on meta refresh):

  1. self_removal_abilities — ABILITY uses followed by the user's own card
     LEAVING our board within the same turn (Run-Away-Draw-class effects,
     discovered meta-wide, not just Dudunsparce). State-diff mining between
     consecutive own MAIN prompts — during our own turn nothing else removes
     our Pokémon.
  2. zero_damage_pairs — (attacker card, defender card) where every observed
     ATTACK dealt 0 (Crustle-class immunities without effect text). Mined
     from the log windows: an ATTACK log event with no defender HP_CHANGE
     before the next attack/turn boundary counts as 0. ALL-ZERO semantics:
     one positive-damage observation VETOES the pair — conditional
     protections and coin-flip whiffs must not poison the table.
  3. deck_cost — WORST-CASE observed own-deck drop attributable to each
     optional PLAY/ABILITY (generalizes the m30/m39 hand-measured burn
     table; the m37_pm_burn attribution law: drop between consecutive
     same-turn MAIN prompts belongs to the action chosen between them,
     sub-prompt resolutions included). Worst-case, not mean — the consumer
     is a deck-out veto.
  4. gust_cards — PLAYs followed by the OPPONENT's active changing within
     our own turn (forced switch; generalizes GUST_IDS beyond Boss's
     Orders). Nothing else changes their active during our turn.

Alignment law: `rl.replay_bc.iter_replay_decisions` (the ACTIVE-seat
boundary — the interpreter refreshes `observation.logs` only for the ACTIVE
seat, so each seat is mined from its own prompts). Log-window attack
pairing anchors both actives from the last prompt's STATE and re-tracks
through SWITCH/CHANGE events; an ATTACK whose cardId matches neither
tracked active is skipped, not guessed.

    uv run python scripts/build_rule_facts.py [--workers 8] [--limit N]
"""
import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cg.api import AreaType, LogType, OptionType, SelectContext  # noqa: E402
from rl.replay_bc import iter_replay_decisions                   # noqa: E402

RAW_DIR = ROOT / "data" / "kaggle" / "raw"
OUT = ROOT / "data" / "rule_facts.json"

# Fact-admission floors: below these an observation is an anecdote, not a
# fact. Zero-damage additionally requires that NO positive observation exists.
SELF_REMOVAL_MIN = 2
ZERO_DAMAGE_MIN = 3
GUST_MIN = 2

MAIN = int(SelectContext.MAIN)
_TURN_BOUNDARY = {int(LogType.TURN_START), int(LogType.TURN_END)}


def _board_ids(player: dict) -> list[int]:
    ids = []
    for zone in (player.get("active") or []), (player.get("bench") or []):
        for p in zone:
            if isinstance(p, dict) and p.get("id") is not None:
                ids.append(int(p["id"]))
    return ids


def _active_id(player: dict):
    act = player.get("active") or []
    return int(act[0]["id"]) if act and isinstance(act[0], dict) \
        and act[0].get("id") is not None else None


def _chosen(sel: dict, action: list, hand: list, me: dict):
    """(option_type, card_id) for the chosen MAIN option — the m37_pm_burn
    resolution: PLAY/ATTACH/EVOLVE via the hand index, ABILITY via the
    board area+index (abilities never carry cardId — M30 P0)."""
    opts = sel.get("option") or []
    if not action or action[0] >= len(opts):
        return None, None
    opt = opts[action[0]]
    ot = int(opt.get("type", -1))
    idx = opt.get("index")
    if ot in (int(OptionType.PLAY), int(OptionType.ATTACH),
              int(OptionType.EVOLVE)):
        if opt.get("area") in (int(AreaType.HAND), None) and idx is not None \
                and idx < len(hand) and isinstance(hand[idx], dict):
            return ot, int(hand[idx]["id"])
        return ot, None
    if ot == int(OptionType.ABILITY):
        area = opt.get("area")
        if area == int(AreaType.ACTIVE):
            return ot, _active_id(me)
        if area == int(AreaType.BENCH) and idx is not None:
            bench = me.get("bench") or []
            if idx < len(bench) and isinstance(bench[idx], dict):
                return ot, int(bench[idx]["id"])
    return ot, None


def _mine_logs(logs: list, our_active, opp_active, tables):
    """Zero-damage pairs from one log window. Actives are tracked through
    SWITCH (cardIdActive in, cardIdBench out is the BENCH view — the pair of
    serials is ambiguous per side, so both directions are tried) and CHANGE
    (cardIdBefore -> cardIdAfter); an ATTACK matching neither tracked active
    is skipped rather than guessed."""
    pending = None                      # (attacker_id, defender_id)

    def flush(damaged: bool):
        nonlocal pending
        if pending is not None:
            atk, dfn = pending
            tables["dmg_obs"][(atk, dfn)][1 if damaged else 0] += 1
            pending = None

    for ev in logs or []:
        if not isinstance(ev, dict):
            continue
        et = ev.get("type")
        if et in _TURN_BOUNDARY or et == int(LogType.ATTACK):
            flush(False)
        if et == int(LogType.ATTACK):
            cid = ev.get("cardId")
            if cid == our_active and opp_active is not None:
                pending = (int(cid), int(opp_active))
            elif cid == opp_active and our_active is not None:
                pending = (int(cid), int(our_active))
        elif et == int(LogType.HP_CHANGE):
            if pending is not None and ev.get("cardId") == pending[1] \
                    and (ev.get("value") or 0) < 0:
                flush(True)
        elif et == int(LogType.SWITCH):
            a, b = ev.get("cardIdActive"), ev.get("cardIdBench")
            if our_active in (a, b):
                our_active = a if our_active == b else b
            elif opp_active in (a, b):
                opp_active = a if opp_active == b else b
        elif et == int(LogType.CHANGE):
            before, after = ev.get("cardIdBefore"), ev.get("cardIdAfter")
            if before == our_active:
                our_active = after
            elif before == opp_active:
                opp_active = after
    flush(False)
    return our_active, opp_active


def mine_episode(steps: list, episode_id: int = 0) -> dict:
    """All four tables from one raw episode, both seats. Pure function of
    the steps list — the unit under test."""
    tables = {
        "dmg_obs": defaultdict(lambda: [0, 0]),   # (atk,dfn) -> [zero, dmg]
        "self_removal": Counter(),                # ability card id -> n
        "self_removal_ep": {},                    # card id -> example ep
        "deck_cost": defaultdict(lambda: [0, 0]),  # card id -> [worst, n]
        "gust": Counter(),                        # play card id -> n
    }
    for seat in (0, 1):
        drops = Counter()
        prev = None       # (turn, deck, board_ids, opp_active, otype, cid)
        our_active = opp_active = None
        prev_window: list = []
        for _, obs, action in iter_replay_decisions(steps, seat, drops):
            sel, cur = obs.get("select"), obs.get("current")
            # sub-prompt chains RE-DELIVER the window as a prefix (the
            # OppMemory prefix-dedupe case, rl/replay_bc.py M40 S5 note) —
            # process only the new suffix or attacks double-count.
            window = obs.get("logs") or []
            fresh = (window[len(prev_window):]
                     if window[:len(prev_window)] == prev_window else window)
            prev_window = window
            our_active, opp_active = _mine_logs(
                fresh, our_active, opp_active, tables)
            me = cur["players"][cur["yourIndex"]]
            op = cur["players"][1 - cur["yourIndex"]]
            board, opp_act = _board_ids(me), _active_id(op)
            our_active = _active_id(me)          # re-anchor from state
            opp_active = opp_act
            if int(sel.get("context", -1)) != MAIN:
                continue
            turn, deck = cur.get("turn"), me.get("deckCount")
            if prev is not None and prev[0] == turn:
                p_turn, p_deck, p_board, p_opp_act, p_ot, p_cid = prev
                if p_cid is not None:
                    if p_ot in (int(OptionType.PLAY), int(OptionType.ABILITY),
                                int(OptionType.ATTACH)):
                        drop = max((p_deck or 0) - (deck or 0), 0)
                        worst_n = tables["deck_cost"][p_cid]
                        worst_n[0] = max(worst_n[0], drop)
                        worst_n[1] += 1
                    if (p_ot == int(OptionType.ABILITY)
                            and p_cid not in board
                            and len(board) < len(p_board)):
                        tables["self_removal"][p_cid] += 1
                        tables["self_removal_ep"].setdefault(
                            p_cid, episode_id)
                    if (p_ot == int(OptionType.PLAY)
                            and p_opp_act is not None and opp_act is not None
                            and opp_act != p_opp_act):
                        tables["gust"][p_cid] += 1
            ot, cid = _chosen(sel, action, me.get("hand") or [], me)
            prev = (turn, deck, board, opp_act, ot, cid)
    return tables


def _merge(parts: list[dict]) -> dict:
    out = {
        "dmg_obs": defaultdict(lambda: [0, 0]),
        "self_removal": Counter(), "self_removal_ep": {},
        "deck_cost": defaultdict(lambda: [0, 0]), "gust": Counter(),
    }
    for t in parts:
        for k, (z, d) in t["dmg_obs"].items():
            out["dmg_obs"][k][0] += z
            out["dmg_obs"][k][1] += d
        out["self_removal"].update(t["self_removal"])
        for cid, ep in t["self_removal_ep"].items():
            out["self_removal_ep"].setdefault(cid, ep)
        for cid, (w, n) in t["deck_cost"].items():
            out["deck_cost"][cid][0] = max(out["deck_cost"][cid][0], w)
            out["deck_cost"][cid][1] += n
        out["gust"].update(t["gust"])
    return out


def finalize(merged: dict, n_episodes: int) -> dict:
    """Admission floors + ALL-ZERO veto -> the committed artifact."""
    from rl.postmortem import card_name
    zero_pairs = sorted(
        [[atk, dfn, z] for (atk, dfn), (z, d) in merged["dmg_obs"].items()
         if d == 0 and z >= ZERO_DAMAGE_MIN])
    return {
        "_doc": (f"mined by scripts/build_rule_facts.py over {n_episodes} "
                 "raw episodes, both seats (docs/M46-plan.md C3). "
                 "zero_damage_pairs are ALL-ZERO (one damaging observation "
                 "vetoes); deck_cost is WORST-CASE observed."),
        "n_episodes": n_episodes,
        "self_removal_abilities": {
            str(cid): {"name": card_name(cid), "n": n,
                       "example_ep": merged["self_removal_ep"].get(cid)}
            for cid, n in sorted(merged["self_removal"].items())
            if n >= SELF_REMOVAL_MIN},
        "zero_damage_pairs": [
            {"attacker": atk, "attacker_name": card_name(atk),
             "defender": dfn, "defender_name": card_name(dfn), "n": z}
            for atk, dfn, z in zero_pairs],
        "deck_cost": {
            str(cid): {"name": card_name(cid), "worst": w, "n": n}
            for cid, (w, n) in sorted(merged["deck_cost"].items())
            if w > 0},
        "gust_cards": {
            str(cid): {"name": card_name(cid), "n": n}
            for cid, n in sorted(merged["gust"].items()) if n >= GUST_MIN},
    }


def _mine_path(path: Path) -> dict | None:
    try:
        raw = json.loads(gzip.open(path, "rt").read())
        t = mine_episode(raw["steps"],
                         int(path.name.split("_")[1].split(".")[0]))
        # plain containers only — lambda-factory defaultdicts do not pickle
        # across the pool boundary
        return {"dmg_obs": dict(t["dmg_obs"]),
                "self_removal": t["self_removal"],
                "self_removal_ep": t["self_removal_ep"],
                "deck_cost": dict(t["deck_cost"]), "gust": t["gust"]}
    except Exception as e:                     # one bad gz must not kill 7.5k
        print(f"  skip {path.name}: {e}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="mine only the first N episodes (smoke runs)")
    a = ap.parse_args()
    if a.workers > 8:
        raise SystemExit("--workers 8 is the proven-stable ceiling")
    paths = sorted(RAW_DIR.glob("episode_*.json.gz"))
    if a.limit:
        paths = paths[:a.limit]
    print(f"mining {len(paths)} raw episodes with {a.workers} workers")
    with Pool(a.workers) as pool:
        parts = [t for t in pool.imap_unordered(_mine_path, paths, 32)
                 if t is not None]
    facts = finalize(_merge(parts), len(parts))
    OUT.write_text(json.dumps(facts, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: "
          f"{len(facts['self_removal_abilities'])} self-removal, "
          f"{len(facts['zero_damage_pairs'])} zero-damage pairs, "
          f"{len(facts['deck_cost'])} deck costs, "
          f"{len(facts['gust_cards'])} gust cards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
