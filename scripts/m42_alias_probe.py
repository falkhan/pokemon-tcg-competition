"""Encoder ALIASING probe: how often can the net not tell two options apart?

M16 found the original defect this way ("337/500 menus had aliased options")
and fixed it by adding option identity. This makes the measurement repeatable,
and — the part M16's headline number lacked — it separates aliasing that
MATTERS from aliasing that does not.

Two options encoding identically is fine when they really are interchangeable:
two copies of the same Basic Energy in hand are the same move. It is a defect
only when the two options address game objects that DIFFER in something
observable. So every aliased group is checked against a fingerprint of what it
actually addresses (acted card + in-play target, each by id / hp / attached
energies / tools), and only groups spanning more than one fingerprint count.

Run against local replay JSON (`rl/eval.py --json_prefix` output, which
scripts/watch_games.py and the QC battery write):

    uv run python scripts/m42_alias_probe.py

Columns 0..OPTION_M28_DIM plus the two embedding ids are what a LIVE bundle
sees, so that is the DEFAULT key width — appended blocks are sliced away in
service and must not flatter this number. `--width N` keys on a different
prefix: `--width 143` (OPTION_M41B_DIM) is the M41b Phase 3 kill, whose
pre-registered bar is real::* == 0 with harmless_true_duplicates unchanged
(docs/M41b-plan.md Phase 3.1).
"""
import argparse, glob, json, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from cg.api import AreaType, OptionType, to_observation_class
from rl.encoders import encode_option_v2, OPTION_M28_DIM

def obj_at(obs, area, index, player):
    if area is None or index is None:
        return None
    me = obs.current.players[player]
    zone = {1: obs.select.deck, 2: me.hand, 3: me.discard,
            4: me.active, 5: me.bench}.get(int(area))
    try:
        return zone[index]
    except (TypeError, IndexError):
        return None

def fingerprint(obs, o):
    """What the option ACTUALLY addresses, in game terms."""
    st = obs.current
    you = st.yourIndex
    acted = obj_at(obs, o.area if o.area is not None else 2, o.index, you)
    tgt = obj_at(obs, o.inPlayArea, o.inPlayIndex, you)
    def sig(p):
        if p is None:
            return None
        return (getattr(p, "id", None), getattr(p, "hp", None),
                tuple(sorted(int(e) for e in (getattr(p, "energies", ()) or ()))),
                tuple(sorted(getattr(c, "id", 0) for c in (getattr(p, "tools", ()) or ()))))
    return (int(o.type), sig(acted), sig(tgt))

def iter_replay_observations(paths):
    """Converted observations that carry a select block, one per prompt,
    in file/step/seat order (the ordering Counter tie-breaks depend on)."""
    for path in paths:
        try:
            raw = json.loads(Path(path).read_text())
        except ValueError:
            continue
        if not isinstance(raw, dict) or not raw.get("steps"):
            continue
        for step in raw["steps"]:
            for seat in (0, 1):
                od = (step[seat] if seat < len(step) else {}).get("observation")
                if not od or not od.get("select"):
                    continue
                try:
                    yield to_observation_class(od)
                except Exception:
                    continue

def probe_aliasing(observations, key_width):
    """The measurement core: bucket each menu's options by (numeric-prefix,
    embedding ids) key and split every aliased group into real (spans more
    than one game-object fingerprint) vs harmless true duplicates.
    Returns (counters, examples: option-type -> fingerprint pairs)."""
    c = Counter()
    examples = defaultdict(list)
    for obs in observations:
        opts = list(obs.select.option or ())
        if len(opts) < 2:
            continue
        c["menus"] += 1
        buckets = defaultdict(list)
        for i, o in enumerate(opts):
            try:
                num, ids = encode_option_v2(o, obs)
            except Exception:
                continue
            key = (num[:key_width].tobytes(), int(ids[0]), int(ids[1]))
            buckets[key].append(i)
        real = 0
        for key, idxs in buckets.items():
            if len(idxs) < 2:
                continue
            fps = {fingerprint(obs, opts[i]) for i in idxs}
            if len(fps) > 1:          # same encoding, DIFFERENT game object
                real += len(idxs) - 1
                t = OptionType(opts[idxs[0]].type).name
                c[f"real::{t}"] += len(idxs) - 1
                if len(examples[t]) < 2:
                    examples[t].append(sorted(fps, key=repr)[:2])
            else:
                c["harmless_true_duplicates"] += len(idxs) - 1
        if real:
            c["menus_with_REAL_aliasing"] += 1
    return c, examples

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=OPTION_M28_DIM,
                    help="encoding prefix the alias key covers "
                         f"(default {OPTION_M28_DIM}, the live-bundle width)")
    args = ap.parse_args(argv)
    c, examples = probe_aliasing(
        iter_replay_observations(sorted(glob.glob("replays/**/*.json",
                                                  recursive=True))),
        args.width)
    for k, v in c.most_common():
        print(f"{v:8d}  {k}")
    print()
    for t, ex in examples.items():
        print(f"--- {t} ---")
        for pair in ex[:1]:
            for fp in pair:
                print("   ", fp)

if __name__ == "__main__":
    main()
