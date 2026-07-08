import shutil
from pathlib import Path

import torch
import numpy as np

from rl.encoders import FEAT
from rl.policy import OptionScorer, save_npz

SUBMISSION_PATH = Path(__file__).parent.parent / "submission"
ROOT_PATH = Path(__file__).parent.parent

torch.manual_seed(0)
model = OptionScorer()

# Ship the newest trained checkpoint when one exists; fall back to the seeded
# random init otherwise (M0 behavior).
checkpoints = sorted((ROOT_PATH / "checkpoints").glob("*.pt"),
                     key=lambda p: p.stat().st_mtime)
if checkpoints:
    model.load_state_dict(torch.load(checkpoints[-1], map_location="cpu"))
    print(f"exporting trained checkpoint: {checkpoints[-1].name}")
else:
    print("no checkpoint found -- exporting seeded random weights")

save_npz(model,str(SUBMISSION_PATH / "policy_weights.npz"))
np.save(str(SUBMISSION_PATH / "card_features.npy"), FEAT)
shutil.copy(str(ROOT_PATH / "deck.csv"), str(SUBMISSION_PATH / "deck.csv"))

# The Kaggle agent runtime provides numpy but NOT the cg engine bindings -- the
# agent only gets what's inside its own bundle, so cg/ must ship with it.
shutil.copytree(ROOT_PATH / "cg", SUBMISSION_PATH / "cg",
                ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)