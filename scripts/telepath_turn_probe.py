"""Per-TURN telepath compliance probe (M36 P0.3 — the M31 artifact law).

The per-PROMPT telepath rate (live_postmortem.py) dilutes: only one manual
attach exists per turn, so every extra MAIN prompt *before* the attach (e.g.
O10 benchfloor promoting a basic PLAY first) adds a declined "offer" without
any behavior change. This probe groups our MAIN prompts by (episode, turn):

  offered turn   — >=1 prompt where Telepath is an ATTACH option from hand
  attached       — Telepath attached at some prompt that turn
  other-energy   — a different energy attached instead (real contested decline)
  no-attach      — turn ended with no manual energy attach (true failure)

Usage: uv run python scripts/telepath_turn_probe.py <sub_id> [<sub_id> ...]
(defaults: 55011605 54997669 54935640 — M35 vs the M33/M31 priors)
"""
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/home/falkhan/Documents/python_projects/pokemon-tcg-competition")

import polars as pl
from cg.api import AreaType, OptionType, SelectContext, to_observation_class
from rl.plan import TELEPATH_ID, _IS_ENERGY
from rl.replay_bc import iter_replay_decisions

ROOT = Path("/home/falkhan/Documents/python_projects/pokemon-tcg-competition")
SUBS = [int(a) for a in sys.argv[1:]] or [55011605, 54997669, 54935640]


def probe(sub: int) -> None:
    df = pl.read_parquet(ROOT / "data/kaggle/episodes.parquet").filter(
        (pl.col("submission_id_0") == sub) | (pl.col("submission_id_1") == sub)
    ).sort("episode_id")
    # (ep, turn) -> {"offered": bool, "tele": bool, "other": bool}
    turns: dict[tuple[int, int], dict] = defaultdict(
        lambda: {"offered": False, "tele": False, "other": False})
    prompt_opps = prompt_att = games = 0

    for r in df.iter_rows(named=True):
        ep = int(r["episode_id"])
        seat = 0 if r["submission_id_0"] == sub else 1
        path = ROOT / f"data/kaggle/raw/episode_{ep}.json.gz"
        if not path.exists():
            continue
        games += 1
        steps = json.load(gzip.open(path))["steps"]
        drops = Counter()
        for _, obs_dict, action in iter_replay_decisions(steps, seat, drops):
            obs = to_observation_class(obs_dict)
            st = obs.current
            if st is None or obs.select is None \
                    or obs.select.context != SelectContext.MAIN:
                continue
            me = st.players[st.yourIndex]
            hand = me.hand or []
            opts = obs.select.option
            key = (ep, st.turn)
            offered_here = any(
                OptionType(o.type) == OptionType.ATTACH and o.index is not None
                and o.area in (AreaType.HAND, None) and o.index < len(hand)
                and hand[o.index].id == TELEPATH_ID
                for o in opts)
            if offered_here:
                turns[key]["offered"] = True
                prompt_opps += 1
            chosen = opts[action[0]]
            if (OptionType(chosen.type) == OptionType.ATTACH
                    and chosen.index is not None
                    and chosen.area in (AreaType.HAND, None)
                    and chosen.index < len(hand)):
                cid = hand[chosen.index].id
                if cid == TELEPATH_ID:
                    turns[key]["tele"] = True
                    if offered_here:
                        prompt_att += 1
                elif cid in _IS_ENERGY:
                    turns[key]["other"] = True

    offered = [t for t in turns.values() if t["offered"]]
    n = len(offered)
    tele = sum(t["tele"] for t in offered)
    other = sum(t["other"] and not t["tele"] for t in offered)
    none = n - tele - other
    print(f"\nsub {sub} ({games} games):")
    print(f"  per-PROMPT: {prompt_att}/{prompt_opps} "
          f"({prompt_att / max(prompt_opps, 1):.1%})   (live_postmortem metric)")
    print(f"  per-TURN offered={n}: telepath {tele} ({tele / max(n, 1):.1%}) | "
          f"other-energy {other} ({other / max(n, 1):.1%}) | "
          f"NO attach {none} ({none / max(n, 1):.1%})")


for sub in SUBS:
    probe(sub)
