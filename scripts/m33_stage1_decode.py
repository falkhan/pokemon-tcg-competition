"""M33 Stage 1 gate decode: evaluate BOTH setup-value checkpoints (baseline vs
late+coarse) on a COMMON val-pair set, pooled by turn regime with CIs.

The A/B trains evaluate each recipe on its OWN pairs (coarse changes the
pairing), so their per-bucket numbers are not comparable. Here we build one
common, late-heavy val set (coarse pairs on the deterministic seed-0 val games)
and score both nets on it. Decode: does t32+ accuracy clear the ~0.80 early bar,
and does the late recipe beat baseline there?

Usage: uv run python scripts/m33_stage1_decode.py [data_dir]
"""
import math
import sys
import random

import numpy as np
import torch

import rl.setup_value as sv
from rl.policy import OptionScorerV3, option_dim_of
from rl.plan import PLAN_DIM

DATA = sys.argv[1] if len(sys.argv) > 1 else "data/setupval_m33"
BASE_CKPT = "checkpoints/m33_sv_base.pt"
LATE_CKPT = "checkpoints/m33_sv_late.pt"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def values(sd_path, rows, idx, extra_dim, base_w):
    sd = torch.load(sd_path, map_location="cpu")
    net = OptionScorerV3(n_state_ids=rows["state_ids"].shape[1],
                         option_dim=option_dim_of(sd), extra_dim=extra_dim)
    net.load_state_dict(sd)
    net.eval()
    w = base_w + extra_dim
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), 512):
            sel = idx[i:i + 512]
            sc = torch.from_numpy(rows["states"][sel, :w])
            sids = torch.from_numpy(rows["state_ids"][sel].astype(np.int64))
            se = net.embedding(sids).flatten(-2)
            s = net.state_enc(torch.cat(
                [sc, torch.zeros(len(sel), PLAN_DIM), se], dim=-1))
            out.append(net.value_head(s).squeeze(-1))
    return torch.cat(out).numpy()


def main():
    rows = sv._load_rows([DATA])
    print(f"corpus {len(rows['states'])} states, width {rows['states'].shape[1]}")
    # deterministic seed-0 val games (matches train())
    rng = np.random.default_rng(0)
    uniq = np.unique(rows["game_ids"])
    val_games = rng.choice(uniq, max(1, int(0.1 * len(uniq))), replace=False)
    ok_train = np.setdiff1d(uniq, val_games)
    # one COMMON late-heavy val set (coarse pairs on the val games)
    val_pairs, _ = sv.build_pairs(rows, random.Random(0),
                                  exclude_games=ok_train,
                                  coarse_late=True, late_cap=1200)
    w_idx = np.array([p[0] for p in val_pairs])
    l_idx = np.array([p[1] for p in val_pairs])
    turns = rows["turn"][w_idx]              # pair turn = the win-state turn bucket
    print(f"common val pairs: {len(val_pairs)}")

    vb_w = values(BASE_CKPT, rows, w_idx, 0, sv.BASE_STATE_W)
    vb_l = values(BASE_CKPT, rows, l_idx, 0, sv.BASE_STATE_W)
    vn_w = values(LATE_CKPT, rows, w_idx, sv.LATE_DIM, sv.BASE_STATE_W)
    vn_l = values(LATE_CKPT, rows, l_idx, sv.LATE_DIM, sv.BASE_STATE_W)
    hit_b = vb_w > vb_l
    hit_n = vn_w > vn_l

    regimes = [("t<16", turns < 16), ("t16-31", (turns >= 16) & (turns < 32)),
               ("t32-51", (turns >= 32) & (turns < 52)), ("t52+", turns >= 52),
               ("t32+ (all late)", turns >= 32)]
    print(f"\n{'regime':16s} {'n':>5s}  {'baseline':>22s}  {'late+coarse':>22s}"
          f"  {'z(new-base)':>11s}")
    for name, mask in regimes:
        n = int(mask.sum())
        if n == 0:
            print(f"{name:16s} {n:5d}  (none)")
            continue
        kb, kn = int(hit_b[mask].sum()), int(hit_n[mask].sum())
        pb, pn = kb / n, kn / n
        lob, hib = wilson(kb, n)
        lon, hin = wilson(kn, n)
        # paired-ish two-proportion z (same n, treat independent — conservative)
        se = math.sqrt(pb * (1 - pb) / n + pn * (1 - pn) / n) or 1e-9
        z = (pn - pb) / se
        print(f"{name:16s} {n:5d}  {pb:.3f} [{lob:.2f},{hib:.2f}]"
              f"  {pn:.3f} [{lon:.2f},{hin:.2f}]  {z:+11.2f}")
    print("\nGATE: does t32+ clear ~0.80 AND does late+coarse beat baseline "
          "there? (both near-chance would KILL; both already ~0.8 => premise "
          "was corpus-specific / VS_MAX_TURN over-conservative.)")


if __name__ == "__main__":
    main()
