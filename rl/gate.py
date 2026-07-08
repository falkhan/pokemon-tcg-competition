"""Pre-submission gates: everything that must be true before a package ships.

Run all gates:  python -m rl.gate
Each check is importable on its own (notebook, tests, future CI).
"""
import importlib.util

import numpy as np

SUBMISSION_MAIN = "submission/main.py"


def load_submission_module():
    """Import submission/main.py as a module (defines __file__, unlike Kaggle's exec)."""
    spec = importlib.util.spec_from_file_location("sub_main", SUBMISSION_MAIN)
    sub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub)
    return sub


def parity_check() -> float:
    """Shipped npz through the submission's numpy forward pass vs torch. Returns max diff."""
    import torch
    from rl.policy import OptionScorer

    sub = load_submission_module()

    # Reference model rebuilt from the *shipped* weights, not a fresh seed.
    model = OptionScorer()
    model.load_state_dict({k: torch.from_numpy(v)
                           for k, v in np.load("submission/policy_weights.npz").items()})

    rng = np.random.default_rng(0)
    state_ctx = rng.random(sub.STATE_DIM + sub.N_CONTEXTS, dtype=np.float32)
    options = rng.random((9, sub.OPTION_DIM), dtype=np.float32)

    ours = sub.score_options(state_ctx, options)
    with torch.no_grad():
        ref, _ = model(torch.from_numpy(state_ctx).unsqueeze(0),
                       torch.from_numpy(options).unsqueeze(0))
    ref = ref.squeeze(0).numpy()

    assert np.allclose(ours, ref, rtol=1e-4, atol=1e-4), \
        f"parity MISMATCH\nnumpy: {ours}\ntorch: {ref}"
    return float(np.abs(ours - ref).max())


def encoder_parity_check(steps: int = 40) -> int:
    """Guard against train/serve ENCODER drift (M0 lesson #5): walk a real game and
    assert rl.encoders and the submission module encode every observation and
    option identically. The weight-parity test can't catch this — it feeds random
    vectors, never real observations. Returns the number of decisions compared."""
    from cg.api import to_observation_class
    from cg.game import battle_start, battle_select, battle_finish
    from rl import encoders as enc

    sub = load_submission_module()
    assert (enc.STATE_DIM, enc.OPTION_DIM) == (sub.STATE_DIM, sub.OPTION_DIM), \
        f"dim mismatch: rl {(enc.STATE_DIM, enc.OPTION_DIM)} vs sub {(sub.STATE_DIM, sub.OPTION_DIM)}"

    deck = [int(x) for x in open("submission/deck.csv") if x.strip()]
    obs_dict, _ = battle_start(deck, deck)
    compared = 0
    try:
        for _ in range(steps):
            if obs_dict["current"]["result"] >= 0:
                break
            obs = to_observation_class(obs_dict)
            a = np.concatenate([enc.encode_state(obs.current), enc.encode_context(obs.select.context)])
            b = np.concatenate([sub.encode_state(obs.current), sub.encode_context(obs.select.context)])
            assert np.allclose(a, b), "encode_state/context drift between rl.encoders and submission"
            for o in obs.select.option:
                assert np.allclose(enc.encode_option(o, obs), sub.encode_option(o, obs)), \
                    "encode_option drift between rl.encoders and submission"
            compared += 1
            obs_dict = battle_select(list(range(obs_dict["select"]["maxCount"])))
    finally:
        battle_finish()
    return compared


def deck_check() -> int:
    """Shipped deck.csv is a legal deck. Returns deck size."""
    from rl.deck_search import validate_deck

    deck = [int(x) for x in open("submission/deck.csv") if x.strip()]
    legal, reasons = validate_deck(deck)
    assert legal, f"illegal deck: {reasons}"
    return len(deck)


def gate_game() -> dict:
    """Full self-play game with the agent loaded Kaggle-style (file path -> exec)."""
    from kaggle_environments import make

    env = make("cabt")
    env.run([SUBMISSION_MAIN, SUBMISSION_MAIN])
    statuses = [s.status for s in env.state]
    assert statuses == ["DONE", "DONE"], f"gate game failed: {statuses}"
    return {"rewards": [s.reward for s in env.state], "decisions": len(env.steps)}


def main() -> None:
    diff = parity_check()
    print(f"parity OK (max diff {diff:.2e})")
    n = encoder_parity_check()
    print(f"encoder parity OK ({n} decisions compared)")
    n = deck_check()
    print(f"deck legal ({n} cards)")
    g = gate_game()
    print(f"gate game OK: rewards {g['rewards']}, {g['decisions']} decisions")


if __name__ == "__main__":
    main()
