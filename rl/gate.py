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
    n = deck_check()
    print(f"deck legal ({n} cards)")
    g = gate_game()
    print(f"gate game OK: rewards {g['rewards']}, {g['decisions']} decisions")


if __name__ == "__main__":
    main()
