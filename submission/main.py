"""Kaggle entry point for the NEURAL v2 submission: OptionScorerV2 (M7.3 id
embeddings + deck-conditioned encoders v2) exported to .npz and replayed here
in pure numpy — Kaggle provides numpy but neither torch nor polars.

Ships the ACTUAL encoder modules (rl/encoders.py + rl/combat.py, no
drift-prone hand-copy — the M6 bundle doctrine): rl/encoders.py falls back to
the exported rl/card_features.npy when the training parquet is absent, so the
bundle never imports polars. Greedy argmax matches evaluation
(rl/matchrunner.py model pilot, greedy=True).

Built by `python -m tcg.shipping export --checkpoint <ckpt> --deck <deck>`.

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
if _IS_V3:
    from cg.api import SelectContext
    from rl.plan import PLAN_DIM, encode_plan, enumerate_plans
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
    """
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
    """Numpy twin of tcg.network.OptionScorerV3.forward, logits only."""
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
            _log_net({"ev": "start", "v3": _IS_V3, "v4": _IS_V4})
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
    order = np.argsort(scores)[::-1]
    acts = [int(i) for i in order[:obs.select.maxCount]]
    if _LOG_NET:
        rec = {"s": obs_dict.get("step"), "t": obs.current.turn,
               "c": int(obs.select.context), "a": acts, "sc": _r3(scores)}
        if plan_rec:
            rec.update(plan_rec)
        _log_net(rec)
    return acts
