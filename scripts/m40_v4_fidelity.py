"""M40 S5 FIDELITY PROBE — the pre-registered kill for the v4 re-encode.

The lane's premise is that we can reconstruct `OppMemory` offline, from a
recorded replay, and get the SAME state the live agent had. If that is false
the re-encode produces an encoder that lies, which is worse than one that is
blind (docs/M40-plan.md §2 S5), and the lane dies here rather than after a
fine-tune.

DESIGN. Two halves, and the comparison has to be live-vs-offline — a
walker-against-itself check would pass identically whether or not the replay's
log window matches the live one, which is exactly the failure being tested.

  LIVE     an agent records the observation dict it was handed, in call order,
           to a JSONL side-channel. That IS the ground truth: live sees every
           own prompt, which is the property the offline walker must reproduce.
  OFFLINE  the same games' env.toJSON() (rl.eval.play_games --json_prefix, the
           same shape as a cached Kaggle episode) is replayed through
           rl.replay_bc.iter_replay_prompts with the deck-prompt reset rule.

Both streams are fed to a fresh OppMemory and the FULL INTERNAL STATE is
compared after every prompt — not `features()`, which saturates (min(x,10)/10)
and would hide a real divergence in e.g. _attacks_total 11->12.

Also reported, non-gating: SELF-CONSISTENCY — how much the naive
iter_replay_decisions path (which skips six prompt classes) actually corrupts.
That number is the walker's ROI and belongs in the diary either way.

Pre-registered threshold: 100.0% exact per-prompt equality of the internal-state
sequence, plus exact prompt-count equality. Not 99% — this is a deterministic
accumulator over a recorded stream, so a divergence is a defect, never noise.

Usage:
    uv run python scripts/m40_v4_fidelity.py [-n 6] [--self-consistency 200]
"""
import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RECORDER = '''
import json, os, sys
sys.path.insert(0, {root!r})
_TRACE = os.environ["PKM_MEM_TRACE"]
_N = [0]

def agent(obs, config=None):
    o = obs if isinstance(obs, dict) else dict(obs)
    with open(_TRACE, "a") as f:
        f.write(json.dumps({{"call": _N[0],
                            "you": (o.get("current") or {{}}).get("yourIndex"),
                            "step": o.get("step"),
                            "logs": o.get("logs") or []}}) + "\\n")
    _N[0] += 1
    sel = o.get("select")
    if sel is None:
        import csv
        with open({deck!r}) as fh:
            return [int(x[0]) for x in csv.reader(fh) if x and x[0].strip()]
    return list(range(sel.get("maxCount", 1)))
'''


def _fingerprint(m) -> tuple:
    """The FULL accumulator state. features() is a lossy projection — it
    saturates at 10 and one-hots the attach target — so comparing it would
    let real divergences through."""
    return (tuple(m._prev_fps), tuple(sorted(m._known_hand.items())),
            tuple(m._played), m._last_attacker_id, m._last_attack_id,
            m._attacks_total, m._attacked_this_turn, m._attacked_last_turn,
            m._passed_last_turn, m._last_attach_target, m._draw_reverse,
            m._opp_turns)


def live_sequences(trace: Path) -> dict[int, list[tuple]]:
    """Replay the recorded windows in call order, per seat."""
    from rl.memory import OppMemory
    seqs: dict[int, list[tuple]] = {}
    mems: dict[int, OppMemory] = {}
    for line in trace.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        seat = rec["you"]
        if seat is None:                 # deck prompt: live resets, no observe
            continue
        m = mems.setdefault(seat, OppMemory())
        m.observe({"current": {"yourIndex": seat}, "logs": rec["logs"]})
        seqs.setdefault(seat, []).append(_fingerprint(m))
    return seqs


def offline_sequences(episode_json: Path) -> dict[int, list[tuple]]:
    """Reconstruct from the recorded episode, exactly as encode_decisions_v4
    does: observe every own ACTIVE prompt, reset (never observe) on the deck
    prompt."""
    from rl.memory import OppMemory
    from rl.replay_bc import iter_replay_prompts
    steps = json.loads(episode_json.read_text())["steps"]
    seqs: dict[int, list[tuple]] = {}
    for seat in (0, 1):
        mem, out = OppMemory(), []
        for _i, obs, _a, _r in iter_replay_prompts(steps, seat, Counter()):
            sel = obs.get("select") if isinstance(obs, dict) else None
            cur = obs.get("current") if isinstance(obs, dict) else None
            if sel is None:
                mem.reset()
                continue
            if not isinstance(cur, dict) or "yourIndex" not in cur:
                continue
            mem.observe(obs)
            out.append(_fingerprint(mem))
        if out:
            seqs[seat] = out
    return seqs


def self_consistency(n_episodes: int) -> tuple[int, int, int, int]:
    """Non-gating: how many rows does the NAIVE (labelled-prompts-only) path
    get wrong? Returns (rows, wrong_rows, episodes, episodes_affected)."""
    from rl.memory import OppMemory
    from rl.replay_bc import _cached_episodes, _load_raw, iter_replay_prompts
    rows = wrong = eps = eps_bad = 0
    for k, (_eid, path) in enumerate(_cached_episodes()):
        if k >= n_episodes:
            break
        try:
            steps = _load_raw(path)["steps"]
        except Exception:
            continue
        eps += 1
        bad_here = 0
        for seat in (0, 1):
            full, naive = OppMemory(), OppMemory()
            for _i, obs, _a, reason in iter_replay_prompts(steps, seat,
                                                           Counter()):
                sel = obs.get("select") if isinstance(obs, dict) else None
                cur = obs.get("current") if isinstance(obs, dict) else None
                if sel is None:
                    full.reset()
                    naive.reset()
                    continue
                if not isinstance(cur, dict) or "yourIndex" not in cur:
                    continue
                full.observe(obs)
                if not reason:
                    naive.observe(obs)      # the naive path never sees drops
                    rows += 1
                    if _fingerprint(full) != _fingerprint(naive):
                        wrong += 1
                        bad_here += 1
        eps_bad += bool(bad_here)
    return rows, wrong, eps, eps_bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--games", type=int, default=6)
    ap.add_argument("--deck", default="alakazam_v2_h4")
    ap.add_argument("--self-consistency", type=int, default=200,
                    help="episodes for the non-gating corruption census")
    a = ap.parse_args()

    from rl.eval import play_games

    tmp = Path(tempfile.mkdtemp(prefix="m40_fid_"))
    deck_csv = ROOT / "decks" / f"{a.deck}.csv"
    rec = tmp / "recorder.py"
    rec.write_text(RECORDER.format(root=str(ROOT), deck=str(deck_csv)))

    # ONE GAME PER play_games CALL, with its own trace file. Both agents in a
    # game append to the same trace, and a multi-game run would interleave four
    # streams into one file with no way to attribute them — the first version
    # of this probe did exactly that and reported off=102 vs live=390, which
    # reads as a catastrophic lane failure and is actually a harness bug.
    total = matched = 0
    mismatches = []
    print(f"playing {a.games} games (recorder vs recorder) -> {tmp}")
    for g in range(a.games):
        trace = tmp / f"trace_{g:03d}.jsonl"
        os.environ["PKM_MEM_TRACE"] = str(trace)
        play_games(str(rec), str(rec), 1, json_prefix=str(tmp / f"ep{g:03d}"))
        live = live_sequences(trace)
        for ep in sorted(tmp.glob(f"ep{g:03d}_g*.json")):
            off = offline_sequences(ep)
            for seat, off_seq in off.items():
                live_seq = live.get(seat, [])
                total += len(off_seq)
                for i in range(min(len(off_seq), len(live_seq))):
                    if off_seq[i] == live_seq[i]:
                        matched += 1
                    elif len(mismatches) < 5:
                        mismatches.append((ep.name, seat, f"prompt {i}"))
                if len(off_seq) != len(live_seq):
                    mismatches.append(
                        (ep.name, seat,
                         f"COUNT off={len(off_seq)} live={len(live_seq)}"))

    print("\n" + "=" * 70)
    print("FIDELITY (gating): offline reconstruction vs the live stream")
    print("=" * 70)
    print(f"  prompts compared : {total}")
    print(f"  exact matches    : {matched}")
    rate = matched / total if total else 0.0
    print(f"  agreement        : {rate:.4%}")
    for m in mismatches[:5]:
        print(f"    first divergence: {m}")

    rows, wrong, eps, eps_bad = self_consistency(a.self_consistency)
    print("\n" + "=" * 70)
    print("SELF-CONSISTENCY (non-gating): what the NAIVE path gets wrong")
    print("=" * 70)
    print(f"  episodes scanned      : {eps}   affected: {eps_bad} "
          f"({eps_bad / max(eps, 1):.2%})")
    print(f"  labelled rows         : {rows}")
    print(f"  rows with WRONG memory: {wrong} ({wrong / max(rows, 1):.4%})")
    print("  ^ this is the walker's ROI: a dropped window poisons the rest of")
    print("    that seat's game, so contamination >> the drop rate.")

    ok = total > 0 and matched == total
    print("\n" + ("PASS — 100% exact agreement; the re-encode may proceed."
                  if ok else
                  "FAIL — pre-registered kill: the replay log window differs "
                  "from live. Do NOT re-encode."))
    print("\nWhat this does NOT prove: that HARVESTED Kaggle episodes carry the "
          "same log semantics as locally generated ones. It validates our "
          "replay path, and assumes transfer.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
