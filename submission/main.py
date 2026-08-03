"""Kaggle entry point for the neural submission: an OptionScorer checkpoint
exported to .npz by tcg/shipping.py and replayed here in pure numpy — Kaggle
provides numpy but neither torch nor polars.

One main.py replays every export tier, auto-sniffed from the weights:
- v2 (M7.3): OptionScorerV2 — card-id embeddings + deck-conditioned encoders.
- v3 (M11): OptionScorerV3 — plan-conditioned. Plans are scored once per turn
  at the first own MAIN prompt (plan_head) and held for that turn's submenus;
  M15 hand-aware exports (20 state ids) are sniffed from the trunk width.
- v4 (M21): encoder v4 — full observable state + per-game opponent memory
  (rl/memory.py OppMemory), declared by the `enc_ver` buffer. v4 implies v3.
Current ships (M22: B2/B3 and the PPO-trained M22c-RL, all lucario) are
v3+v4 exports; PPO checkpoints carry value_head keys in the npz which are
simply unused at play time (logits only here).

Ships the ACTUAL encoder modules (rl/encoders.py + rl/plan.py + rl/memory.py
+ rl/combat.py, no drift-prone hand-copy — the M6 bundle doctrine):
rl/encoders.py falls back to the exported rl/card_features.npy when the
training parquet is absent, so the bundle never imports polars. Greedy argmax
and the per-turn plan/memory state machine match evaluation exactly
(rl/matchrunner.py model pilot fn4, greedy=True).

Built by `./build_submission.sh --checkpoint <ckpt> --deck <deck>` (--deck is
MANDATORY — the M18.1/M22c fossil-deck accidents), a thin wrapper over
`python -m tcg.shipping export` + `gate`; the gate replays this file against
the torch net (weight parity, plan-head parity, live-game encoder parity, v4
memory-drift parity) before packaging.

Per-decision net-internals log (M19): one compact `NN|{json}` line per agent
call on stderr — Kaggle stores agent stderr per step (episode agent logs,
`rl.kaggle_ingest agent-logs`), and the replay JSON itself carries the option
menu but none of the net's scores. Joined back to the replay by the obs
`step` field (`rl/postmortem.py`). Disable with PKM_AGENT_LOG=0.
"""
import json
import os
import sys

import numpy as np

# Make the bundled cg/ and rl/ importable whether run locally or on Kaggle
# (Kaggle exec's this file from /kaggle_simulations/agent/).
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else None
for _d in [_HERE, "/kaggle_simulations/agent", "submission", "."]:
    if _d and os.path.exists(os.path.join(_d, "policy_weights.npz")):
        _BASE = _d
        break
else:
    raise FileNotFoundError("neural-agent artifacts not found (policy_weights.npz)")

if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from cg.api import to_observation_class
from rl.encoders import (EMBED_DIM, N_CARD_IDS, N_CONTEXTS, N_OPTION_IDS,
                         N_STATE_IDS, OPTION_V2_DIM, STATE_V2_DIM,
                         encode_context, encode_option_v2, encode_state_v2)

WEIGHTS = np.load(os.path.join(_BASE, "policy_weights.npz"))
if "embedding.weight" not in WEIGHTS:
    raise ValueError(
        "policy_weights.npz is a v1 (OptionScorer) export; this main.py replays "
        "OptionScorerV2 — re-export with `python -m tcg.shipping export`")
DECK = [int(x) for x in open(os.path.join(_BASE, "deck.csv")) if x.strip()]

_EMBED = WEIGHTS["embedding.weight"]
_IS_V3 = "plan_enc.0.weight" in WEIGHTS          # M11 plan-conditioned export
_IS_V4 = "enc_ver" in WEIGHTS                    # M21 encoder-v4 export
# M27: this bundle's option width, sniffed from option_enc.0 (torch stores
# (out, in); the trailing 2*EMBED columns are the option-id embeddings).
_OPTION_DIM = WEIGHTS["option_enc.0.weight"].shape[1] - 2 * _EMBED.shape[1]
if _IS_V3:
    from cg.api import SelectContext
    from rl.plan import (PLAN_DIM, SERVE_FIX_PLANZERO, apply_attach_overrides,
                         apply_play_overrides, encode_plan, enumerate_plans)
    # M15: hand-aware exports carry 20 state ids — sniff from the weights
    # and pick the matching encoder. M21: v4 exports declare themselves via
    # the enc_ver buffer (width sniffing is ambiguous with the v4 block in).
    _V4_EXTRA = 0
    if _IS_V4:
        from rl.encoders import V4_EXTRA_DIM as _V4_EXTRA
        from rl.encoders import encode_ctx_v4
        from rl.memory import OppMemory
        # Per-game opponent memory (the _PSTATE precedent): observed once
        # per agent call, reset on deck-return and turn-counter drop.
        _MEM = OppMemory()
    _N_IDS = (WEIGHTS["state_enc.0.weight"].shape[1]
              - STATE_V2_DIM - N_CONTEXTS - _V4_EXTRA - PLAN_DIM) // EMBED_DIM
    if _N_IDS > N_STATE_IDS:
        from rl.encoders import encode_state_v3 as _encode_state
    else:
        _encode_state = encode_state_v2
    # Turn-scoped plan state (buddy-precedent module globals): plan once at
    # each turn's FIRST own MAIN prompt, hold for the turn. Reset on the
    # deck-return call and on a turn-counter drop (new game, reused process).
    _PSTATE = {"key": None, "vec": None, "last_turn": -1}


_LOG_NET = os.environ.get("PKM_AGENT_LOG", "1") != "0"
# M26/M30/M31/M35/M37 override arms (rl/plan.apply_attach_overrides + apply_play_overrides):
# comma-separated fix names ("telepath", "backstop", "tempo", "deckguard",
# "ash", "conserve", "poffinfloor", "drawfloor", "benchfloor", "racemode3").
#
# M39 SHIP B: "conserve" + the anti-deck-out package "racemode2,racemode4".
#   Previous default (Ship A, sub 55172160): "conserve"
#   Before that (M38, sub 55146658): "telepath,deckguard,ash,conserve,
#   benchfloor,racemode3" (gacfr3 = the M35 gacf arm + O12d racemode3).
#
# Why (docs/M39.md, P2). Single-lane rules change on the SHIPPED net, gated on
# the G-13 panel roster (>=3 independently trained clones per family, so a cell
# is a panel mean rather than one draw of a training lottery):
#     conserve,racemode2,racemode4 vs conserve : +2.37pp weighted  z=+5.03
#     conserve,racemode4           vs conserve : +1.48pp weighted  z=+3.15
#     conserve,racemode2           vs conserve : +0.24pp weighted  z=+0.51
# The wall PANEL moves +15.7 / +11.1 / +6.7pp across its three draws (mean
# +11.2pp) on our worst matchup, and no cell is significantly harmed.
#
# The two rules are SYNERGISTIC and neither ships alone: racemode2 closes the
# draw-ABILITY exits (Fezandipiti/Dudunsparce) and racemode4 the burn
# PLAY/ATTACH exits (Enriching Energy 4.0 cards/attach, Poke Pad, surplus
# Dawn/Hilda once Alakazam is on board). Close one and the policy leaves
# through the other -- which is why racemode2 ALONE is negative on wall
# (-1.3pp). That is why the M37 racemode lane failed for four milestones: it
# only ever closed half the door.
#
# Mechanism (G-11, scripts/m39_race_probe.py, n=200/leg): against the wall bed
# the package takes the OPPONENT's deck-out rate from 3% to 31% and their
# lowest deck count from 23.1 to 7.9, while the mirror bed -- where the rules
# provably never fire -- moves by <=5.5pp. We do not burn less; we make the
# game last long enough that the wall's own burn kills it first.
#
# Why `conserve` is still in the string (docs/M39.md, P0.7 + P1-inv — the
# Ship A rationale, unchanged). The M38 G5 matrix already showed the full
# stack was actively harmful on the wall bed. M39 ran the leave-one-out
# ablation G5 never had, then a 3-arm gate over the 9-bed live-mix-weighted
# roster at n=2400/cell:
#     conserve-only vs gacfr3 : +1.74pp  z=+3.42   <- this ship
#     conserve-only vs plain  : +1.03pp  z=+2.03
#     plain         vs gacfr3 : +0.71pp  z=+1.39   (no detectable difference)
# Positive on 6 of 8 beds at ~+2-3pp, including every big-share one.
#
# `conserve` survived alone because a MECHANISM probe (scripts/conserve_probe.py)
# showed it actually acts. Fire rate over 60-90 games/bed across seeds
# (fires / states where our own deck is in the demote zone):
#     top (900+ mirror)  103/956 = 11%      grim   39/292 = 13%
#     mirror              47/744 =  6%      wall    1/473 =  0.2%
# So it is active in mirror-family and grim deck races and effectively inert
# vs wall — which matches the gate, where conserve-only's gains were largest
# on grim (+2.9pp) and absent where it never fires. A significance threshold
# had discarded this rule; the mechanism probe rescued it.
#
# An approved ship changes this default string, never the predicate.
#
# `planzero` added 2026-08-02 (M40 S6) on the PRE-REGISTERED NULL branch, not
# on a win. The battery measured +0.43pp (z=0.91) on cont3 and +0.80pp (z=1.72)
# on retain_b — same sign, neither resolving against a 1.3pp MDE. The plan
# decided in advance (§6 decision 2) that a null adopts it ANYWAY, because it
# is not a bet: every training row since M24 carries plans=zeros, so zeroing at
# serve makes the served distribution match the trained one exactly. It removes
# a mismatch rather than adding a mechanism, and removals are the class of
# change that has actually transferred live. Bundled with the next net ship
# rather than spending a slot of its own.
# M40b round 1 (2026-08-03): the racemode package is DROPPED — live evidence
# ranks conserve-only above it (Ship A 773.6 > Ship B 726.3 > floor 659.7)
# and it produced no live deck-out reduction (5/27 vs 7/24, 7/30 losses).
# `ash`+`ashguard` is the deep-dive Sacred Ash timing window (play at deck
# 4-11 like the 1000+ pilots of this list; never at deck >12 — our live
# pilots burned it at 12-36). Battery: +0.41pp z=+1.05 vs cz, mechanism
# probe: ash plays moved from deck {8,13,24,42} to {2,3,8,9,10,12}.
_ATTACH_FIXES = frozenset(
    f for f in os.environ.get(
        "PKM_ATTACH_FIXES",
        "conserve,planzero,ash,ashguard").split(",") if f)
# M40 S6: `planzero` is a SERVE fix, not a reranker — it changes what the trunk
# is FED, so it is read out here rather than passed to apply_*_overrides (which
# ignore it harmlessly). See rl/plan.py O14 for why the plan head is
# out-of-distribution by construction on every corpus we have trained since M24.
_PLAN_ZERO = "planzero" in _ATTACH_FIXES


def _log_net(rec: dict) -> None:
    """Emit one `NN|{json}` line on stderr. Must never kill a live game, but
    failures are not swallowed silently either (M18 lesson) — they surface as
    an NN|ERR| line in the same log."""
    try:
        sys.stderr.write("NN|" + json.dumps(rec, separators=(",", ":")) + "\n")
    except Exception as exc:
        sys.stderr.write(f"NN|ERR|{exc!r}\n")


def _r3(arr) -> list[float]:
    return [round(float(x), 3) for x in arr]


def _linear(x, name):    # torch Linear stores weight as (out, in) -> transpose!
    return x @ WEIGHTS[f"{name}.weight"].T + WEIGHTS[f"{name}.bias"]


def _relu(x):
    return np.maximum(0, x)


def score_options_v2(state_ctx, state_ids, options, option_ids):
    """Numpy twin of tcg.network.OptionScorerV2.forward, logits only.

    state_ctx: (STATE_V2_DIM+N_CONTEXTS,); state_ids: (N_STATE_IDS,) int;
    options: (N, OPTION_V2_DIM); option_ids: (N, N_OPTION_IDS) int.
    M27 width shim: truncate to this bundle's trained option width.
    """
    options = options[:, :_OPTION_DIM]
    se = _EMBED[state_ids].reshape(-1)                                # (IDS*E,)
    s = _relu(_linear(_relu(_linear(np.concatenate([state_ctx, se]),
                                    "state_enc.0")), "state_enc.2"))  # (H,)
    oe = _EMBED[option_ids].reshape(len(options), -1)                 # (N, 2E)
    o = _relu(_linear(np.concatenate([options, oe], axis=1), "option_enc.0"))
    so = np.concatenate([np.broadcast_to(s, (len(o), s.size)), o], axis=1)
    return _linear(_relu(_linear(so, "score_head.0")), "score_head.2").ravel()


def _trunk_v3(state_ctx, plan, state_ids):
    se = _EMBED[state_ids].reshape(-1)
    x = np.concatenate([state_ctx, plan, se])
    return _relu(_linear(_relu(_linear(x, "state_enc.0")), "state_enc.2"))


def score_options_v3(state_ctx, plan, state_ids, options, option_ids):
    """Numpy twin of tcg.network.OptionScorerV3.forward, logits only.

    M27 width shim (twin of the torch one): encode_option_v2 appends new blocks
    and the leading slice stays byte-identical, so truncate to whatever width
    THIS bundle's weights were trained on."""
    options = options[:, :_OPTION_DIM]
    s = _trunk_v3(state_ctx, plan, state_ids)
    oe = _EMBED[option_ids].reshape(len(options), -1)
    o = _relu(_linear(np.concatenate([options, oe], axis=1), "option_enc.0"))
    so = np.concatenate([np.broadcast_to(s, (len(o), s.size)), o], axis=1)
    return _linear(_relu(_linear(so, "score_head.0")), "score_head.2").ravel()


def score_plans(state_ctx, state_ids, plan_cands):
    """Numpy twin of tcg.network.OptionScorerV3.plan_logits (plan=zeros trunk)."""
    zeros = np.zeros(plan_cands.shape[1], dtype=np.float32)
    s = _trunk_v3(state_ctx, zeros, state_ids)
    p = _relu(_linear(plan_cands, "plan_enc.0"))
    sp = np.concatenate([np.broadcast_to(s, (len(p), s.size)), p], axis=1)
    return _linear(_relu(_linear(sp, "plan_head.0")), "plan_head.2").ravel()


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:  # game start: return the deck list
        if _IS_V3:
            _PSTATE.update(key=None, vec=None, last_turn=-1)
        if _IS_V4:
            _MEM.reset()
        if _LOG_NET:
            _log_net({"ev": "start", "v3": _IS_V3, "v4": _IS_V4,
                      "pz": bool(_PLAN_ZERO)})
        return DECK
    if _IS_V4:
        if obs.current.turn < _PSTATE["last_turn"]:   # new game, reused process
            _MEM.reset()
        _MEM.observe(obs)                # once per call, BEFORE encoding
        state_ctx, state_ids = encode_ctx_v4(obs, DECK, _MEM)
    else:
        numeric, state_ids = (_encode_state if _IS_V3
                              else encode_state_v2)(obs.current, DECK)
        state_ctx = np.concatenate([numeric,
                                    encode_context(obs.select.context)]).astype(np.float32)
    pairs = [encode_option_v2(o, obs) for o in obs.select.option]
    options = np.stack([n for n, _ in pairs]).astype(np.float32)
    option_ids = np.stack([i for _, i in pairs])
    plan_rec = None
    if not _IS_V3:
        scores = score_options_v2(state_ctx, state_ids, options, option_ids)
    else:
        t = obs.current.turn
        if t < _PSTATE["last_turn"]:             # new game in a reused process
            _PSTATE.update(key=None, vec=None)
        _PSTATE["last_turn"] = t
        key = (t, obs.current.yourIndex)
        if _PLAN_ZERO:
            # M40 S6: the plan head never runs, so the trunk sees the zero
            # vector every training row carries. No plan_rec in the NN| log is
            # the live tell that the token is acting.
            plan = np.zeros(PLAN_DIM, dtype=np.float32)
        else:
            if obs.select.context == SelectContext.MAIN and _PSTATE["key"] != key:
                cands = enumerate_plans(obs)
                mat = np.stack([encode_plan(c) for c in cands]).astype(np.float32)
                plan_scores = score_plans(state_ctx, state_ids, mat)
                idx = int(np.argmax(plan_scores))
                _PSTATE.update(key=key, vec=mat[idx].copy())
                plan_rec = {"p": idx, "psc": _r3(plan_scores)}
            plan = (_PSTATE["vec"] if _PSTATE["key"] == key and _PSTATE["vec"] is not None
                    else np.zeros(PLAN_DIM, dtype=np.float32))
        scores = score_options_v3(state_ctx, plan, state_ids, options, option_ids)
    order = [int(i) for i in np.argsort(scores)[::-1]]
    if _IS_V3 and _ATTACH_FIXES:
        order = apply_attach_overrides(obs, order, _ATTACH_FIXES)
        order = apply_play_overrides(obs, order, _ATTACH_FIXES)
    acts = order[:obs.select.maxCount]
    if _LOG_NET:
        rec = {"s": obs_dict.get("step"), "t": obs.current.turn,
               "c": int(obs.select.context), "a": acts, "sc": _r3(scores)}
        if plan_rec:
            rec.update(plan_rec)
        elif _PLAN_ZERO:
            # The live tell that the token is acting: a v3 bundle emitting no
            # `p`/`psc` on a MAIN prompt would otherwise be indistinguishable
            # from a submenu. Greppable out of `rl.kaggle_ingest agent-logs`,
            # so a Kaggle episode can PROVE what the ship actually served —
            # the class of evidence that was missing from M24 to M40.
            rec["pz"] = 1
        _log_net(rec)
    return acts
