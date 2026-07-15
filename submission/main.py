"""Kaggle entry point for the NEURAL v2 submission: OptionScorerV2 (M7.3 id
embeddings + deck-conditioned encoders v2) exported to .npz and replayed here
in pure numpy — Kaggle provides numpy but neither torch nor polars.

Ships the ACTUAL encoder modules (rl/encoders.py + rl/combat.py, no
drift-prone hand-copy — the M6 bundle doctrine): rl/encoders.py falls back to
the exported rl/card_features.npy when the training parquet is absent, so the
bundle never imports polars. Greedy argmax matches evaluation
(rl/matchrunner.py model pilot, greedy=True).

Built by `python -m tcg.shipping export --checkpoint <ckpt> --deck <deck>`.
"""
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
from rl.encoders import (N_CARD_IDS, N_CONTEXTS, N_OPTION_IDS, N_STATE_IDS,
                         OPTION_V2_DIM, STATE_V2_DIM, encode_context,
                         encode_option_v2, encode_state_v2)

WEIGHTS = np.load(os.path.join(_BASE, "policy_weights.npz"))
if "embedding.weight" not in WEIGHTS:
    raise ValueError(
        "policy_weights.npz is a v1 (OptionScorer) export; this main.py replays "
        "OptionScorerV2 — re-export with `python -m tcg.shipping export`")
DECK = [int(x) for x in open(os.path.join(_BASE, "deck.csv")) if x.strip()]

_EMBED = WEIGHTS["embedding.weight"]


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


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:  # game start: return the deck list
        return DECK
    numeric, state_ids = encode_state_v2(obs.current, DECK)
    state_ctx = np.concatenate([numeric,
                                encode_context(obs.select.context)]).astype(np.float32)
    pairs = [encode_option_v2(o, obs) for o in obs.select.option]
    options = np.stack([n for n, _ in pairs]).astype(np.float32)
    option_ids = np.stack([i for _, i in pairs])
    scores = score_options_v2(state_ctx, state_ids, options, option_ids)
    order = np.argsort(scores)[::-1]
    return [int(i) for i in order[:obs.select.maxCount]]
