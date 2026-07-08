"""Pre-submission gates: everything that must be true before a package ships.

Run neural gates:  python -m rl.gate
Run rule gates:    python -m rl.gate --agent rules
Each check is importable on its own (notebook, tests, future CI).
"""
import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np

SUBMISSION_MAIN = "submission/main.py"
SUBMISSION_RULES_MAIN = "submission_rules/main.py"


def load_submission_module(main_path: str = SUBMISSION_MAIN):
    """Import a submission main.py as a module (defines __file__, unlike Kaggle's exec)."""
    spec = importlib.util.spec_from_file_location("sub_main", main_path)
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


def deck_check(base: str = "submission") -> int:
    """Shipped deck.csv is a legal deck. Returns deck size."""
    from rl.deck_search import validate_deck

    deck = [int(x) for x in open(f"{base}/deck.csv") if x.strip()]
    legal, reasons = validate_deck(deck)
    assert legal, f"illegal deck: {reasons}"
    return len(deck)


def gate_game(main_path: str = SUBMISSION_MAIN) -> dict:
    """Full self-play game with the agent loaded Kaggle-style (file path -> exec)."""
    from kaggle_environments import make

    env = make("cabt")
    env.run([main_path, main_path])
    statuses = [s.status for s in env.state]
    assert statuses == ["DONE", "DONE"], f"gate game failed: {statuses}"
    return {"rewards": [s.reward for s in env.state], "decisions": len(env.steps)}


def bundle_isolation_check(base: str = "submission_rules") -> int:
    """Prove the bundle is self-contained: run main.py + a full game in a subprocess whose
    sys.path is ONLY the bundle dir (no project root), and assert it never imports polars/
    torch. This is what the in-process gate can't catch — the real Kaggle isolation. Rule
    bundle only (the neural bundle ships flat numpy files, not a package)."""
    bdir = Path(base).resolve()
    here = bdir.as_posix()            # forward slashes: safe inside a source string on all OSes
    root = bdir.parent.as_posix()
    driver = (
        "import sys,os\n"
        "nc=lambda p: os.path.normcase(os.path.abspath(p))\n"
        f"here=nc('{here}');root=nc('{root}')\n"
        "sys.path[:]=[p for p in sys.path if nc(p or os.getcwd())!=root]\n"
        f"sys.path.insert(0,'{here}')\n"  # bundled cg/rl take precedence; stdlib+site kept
        "import main\n"
        "from cg.game import battle_start,battle_select,battle_finish\n"
        "od,_=battle_start(main.DECK,main.DECK);n=0\n"
        "while od['current']['result']<0 and n<5000:\n"
        "    od=battle_select([int(i) for i in main.agent(od)]);n+=1\n"
        "battle_finish()\n"
        "assert od['current']['result'] in (0,1,2),'game did not finish'\n"
        "assert 'polars' not in sys.modules and 'torch' not in sys.modules,'bundle pulled a heavy dep'\n"
        "rlf=nc(sys.modules['rl'].__file__)\n"
        "assert rlf.startswith(here),'imported project rl (%s), not bundled rl'%rlf\n"
        "print('ISO_OK',n)"
    )
    env = {k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", driver], cwd=str(bdir),
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "ISO_OK" in r.stdout, \
        f"bundle isolation FAILED:\n{r.stdout}\n{r.stderr}"
    return int(r.stdout.strip().split("ISO_OK")[-1])


def main_neural() -> None:
    diff = parity_check()
    print(f"parity OK (max diff {diff:.2e})")
    n = encoder_parity_check()
    print(f"encoder parity OK ({n} decisions compared)")
    n = deck_check()
    print(f"deck legal ({n} cards)")
    g = gate_game()
    print(f"gate game OK: rewards {g['rewards']}, {g['decisions']} decisions")


def main_rules() -> None:
    n = deck_check("submission_rules")
    print(f"deck legal ({n} cards)")
    d = bundle_isolation_check("submission_rules")
    print(f"bundle isolation OK (self-contained; {d} decisions, no polars/torch)")
    g = gate_game(SUBMISSION_RULES_MAIN)
    print(f"gate game OK: rewards {g['rewards']}, {g['decisions']} decisions")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=["neural", "rules"], default="neural")
    args = p.parse_args()
    (main_rules if args.agent == "rules" else main_neural)()
