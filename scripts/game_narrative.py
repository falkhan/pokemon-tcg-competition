"""Full-game narrative from a cached replay (M46 C3 side task).

The m44 live post-mortem's decision-level findings (Run-Away-Draw
self-benchout, the 17-turn Crustle tunnel, mirror tempo passes) came out of
an uncommitted scratchpad tool; this is that tool rebuilt on rl/postmortem's
primitives and committed (the standing save-down rule). One episode ->
turn-by-turn: both boards, every MAIN decision as '<TYPE>:<card>', attack /
damage events from the log windows, the end classification and the
mechanical audit flags.

    uv run python scripts/game_narrative.py 92684808 [--seat auto|0|1]
    uv run python scripts/game_narrative.py 92684808 --full   # sub-prompts too

Reads data/kaggle/raw/episode_<id>.json.gz (fetches on cache miss via
rl.postmortem._load). `--seat auto` (default) resolves our seat from
episodes.parquet, falling back to repo-deck detection, falling back to 0.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cg.api import LogType  # noqa: E402
from rl.postmortem import (_load, audit_flags, card_name, classify_end,  # noqa: E402
                           iter_decisions, side_brief, _fmt_side)

_NARRATED = {int(LogType.ATTACK): "ATTACK", int(LogType.HP_CHANGE): "HP",
             int(LogType.SWITCH): "SWITCH", int(LogType.CHANGE): "CHANGE"}


def _resolve_seat(episode_id: int, raw: dict, arg: str) -> int:
    if arg in ("0", "1"):
        return int(arg)
    try:
        import polars as pl
        from rl.kaggle_ingest import EPISODES_PQ
        row = (pl.read_parquet(EPISODES_PQ)
               .filter(pl.col("episode_id") == episode_id))
        if len(row) and row["our_seat"][0] is not None:
            return int(row["our_seat"][0])
    except Exception:
        pass
    try:
        from rl.postmortem import detect_our_seat, parse_episode
        seat = detect_our_seat(parse_episode(raw))
        if seat is not None:
            return seat
    except Exception:
        pass
    return 0


def _fmt_event(ev: dict) -> str | None:
    kind = _NARRATED.get(ev.get("type"))
    if kind == "ATTACK":
        return (f"ATTACK {card_name(ev.get('cardId'))}"
                + (f" #{ev['attackId']}" if ev.get("attackId") is not None
                   else ""))
    if kind == "HP":
        return f"HP {card_name(ev.get('cardId'))} {ev.get('value'):+d}"
    if kind == "SWITCH":
        return (f"SWITCH {card_name(ev.get('cardIdBench'))} <-> "
                f"{card_name(ev.get('cardIdActive'))}")
    if kind == "CHANGE":
        return (f"CHANGE {card_name(ev.get('cardIdBefore'))} -> "
                f"{card_name(ev.get('cardIdAfter'))}")
    return None


def narrate(episode_id: int, seat_arg: str = "auto",
            full: bool = False) -> int:
    raw = _load(episode_id)
    steps = raw["steps"]
    us = _resolve_seat(episode_id, raw, seat_arg)
    rewards = raw.get("rewards") or [None, None]
    print(f"=== episode {episode_id} — narrating seat {us} "
          f"(rewards {rewards}) ===")

    last_turn = None
    last_cur = None
    prev_window: list = []
    for i, turn, ctx, opts, action, cur in iter_decisions(steps, us):
        st = steps[i][us]
        if st.get("status") != "ACTIVE":
            continue      # INACTIVE steps repeat a stale select and stale
        last_cur = cur    # logs (the rl/replay_bc ACTIVE-seat law)
        window = (st.get("observation") or {}).get("logs") or []
        fresh = (window[len(prev_window):]
                 if window[:len(prev_window)] == prev_window else window)
        prev_window = window
        events = [e for e in (_fmt_event(ev) for ev in fresh
                              if isinstance(ev, dict)) if e]
        if turn != last_turn:
            last_turn = turn
            b_us = side_brief(cur["players"][us])
            b_op = side_brief(cur["players"][1 - us])
            print(f"\n--- turn {turn} ---")
            print(f"  us : {_fmt_side(b_us)} hand={b_us['hand_n']}")
            print(f"  opp: {_fmt_side(b_op)} hand={b_op['hand_n']}")
        for e in events:
            print(f"      ~ {e}")
        if ctx != "MAIN" and not full:
            continue
        chosen = (set(action) if isinstance(action, list)
                  else {action} if action is not None else set())
        picks = [o for j, o in enumerate(opts) if j in chosen]
        declined = len(opts) - len(picks)
        print(f"  s{i:<4}{ctx:<6}-> {', '.join(picks) or '(no action)'}"
              f"   [{declined} declined]")

    if last_cur is not None:
        print(f"\n=== end: {classify_end(last_cur, us, rewards[us])} ===")
    flags = audit_flags(steps, us)
    if flags:
        print("\naudit flags:")
        for f in flags:
            print(f"  {f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("episode", type=int)
    ap.add_argument("--seat", default="auto", choices=("auto", "0", "1"))
    ap.add_argument("--full", action="store_true",
                    help="narrate sub-prompt decisions too, not just MAIN")
    a = ap.parse_args()
    return narrate(a.episode, a.seat, a.full)


if __name__ == "__main__":
    raise SystemExit(main())
