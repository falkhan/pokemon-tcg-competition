"""Supervised value-head training on self-play outcomes (docs/DECISIONS.md 2026-07-08).

Why this exists: PPO could not train a useful critic on this sparse-reward, mirror-heavy
problem (3 attempts stalled or degraded the policy). But the *information* to predict the
winner is in the encoder (opponent archetype is observable). Training the value head
DIRECTLY — supervised regression to game outcomes, policy frozen — works: bc_v1's original
value head scored 0.62 sign-accuracy (≈ chance); a supervised head hits ~0.87 on the same
frozen features. This decouples "get a good critic" from "don't wreck the policy," and it's
the value-target step of an AlphaGo-Zero-lite loop (self-play -> outcomes -> value head ->
MCTS).

  python -m rl.value_train collect --checkpoint checkpoints/bc_v1.pt --games 600
  python -m rl.value_train train   --checkpoint checkpoints/bc_v1.pt --out bc_v1_value.pt
"""
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "value_train.npz"
DECK_DIR = ROOT / "decks"


def _deck(name: str) -> list[int]:
    return [int(x) for x in (DECK_DIR / f"{name}.csv").read_text().split() if x.strip()]


def collect(checkpoint: str, learn_deck: str = "kyogre", n_games: int = 600,
            out: Path = DATA) -> None:
    """Play the policy (greedy) vs the opponent pool; record (state_ctx, final outcome)
    for the learner seat. Outcome is +1 win / -1 loss / 0 draw from the learner's view."""
    import torch
    from cg.api import to_observation_class
    from cg.game import battle_start, battle_select, battle_finish
    from rl.encoders import encode_state, encode_context, encode_option
    from rl.policy import OptionScorer
    from rl.teacher import load_teacher
    import random

    model = OptionScorer(); model.load_state_dict(torch.load(checkpoint, map_location="cpu")); model.eval()
    ld = _deck(learn_deck)
    opps = [(None, ld),
            (load_teacher("vpl", agent="lucario", deck="lucario"), _deck("lucario")),
            (load_teacher("vpi", agent="iono", deck="iono"), _deck("iono"))]

    def pick(obs):
        sc = np.concatenate([encode_state(obs.current), encode_context(obs.select.context)]).astype(np.float32)
        opts = np.stack([encode_option(o, obs) for o in obs.select.option]).astype(np.float32)
        return sc, [int(i) for i in model.act(sc, opts, k=obs.select.maxCount, greedy=True)]

    states, outcomes = [], []
    for g in range(n_games):
        oa, od = opps[g % 3]
        learn_seat = g % 2
        d0, d1 = (ld, od) if learn_seat == 0 else (od, ld)
        obs_dict, _ = battle_start(d0, d1)
        pend = []
        while obs_dict["current"]["result"] < 0:
            seat = obs_dict["current"]["yourIndex"]
            obs = to_observation_class(obs_dict)
            if seat == learn_seat:
                sc, picks = pick(obs); pend.append(sc)
                obs_dict = battle_select(picks)
            else:
                p = random.sample(range(len(obs_dict["select"]["option"])), obs_dict["select"]["maxCount"]) \
                    if oa is None else oa(obs_dict)
                obs_dict = battle_select([int(i) for i in p])
        res = obs_dict["current"]["result"]; battle_finish()
        y = 0.0 if res == 2 else (1.0 if res == learn_seat else -1.0)
        states += pend; outcomes += [y] * len(pend)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, states=np.stack(states).astype(np.float32),
                        outcomes=np.array(outcomes, dtype=np.float32))
    print(f"collected {len(states)} states -> {out}")


def train(checkpoint: str, out: str = "bc_v1_value.pt", data: Path = DATA,
          epochs: int = 40, lr: float = 1e-3) -> None:
    """Freeze policy+body; train ONLY the value head to regress outcomes. Saves a
    checkpoint = original policy/body + the new value head."""
    import torch
    import torch.nn.functional as F
    from rl.policy import OptionScorer

    d = np.load(data)
    S = torch.from_numpy(d["states"]); Y = torch.from_numpy(d["outcomes"])
    n = len(S); perm = np.random.default_rng(0).permutation(n)
    tr, va = perm[:int(0.85 * n)], perm[int(0.85 * n):]

    model = OptionScorer(); model.load_state_dict(torch.load(ROOT / "checkpoints" / checkpoint, map_location="cpu"))
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("value_head")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    def sign_acc(idx):
        with torch.no_grad():
            v = model.value_head(model.state_enc(S[idx])).squeeze(-1)
        return ((v > 0).float() == (Y[idx] > 0).float()).float().mean().item()

    base = sign_acc(va)
    for _ in range(epochs):
        model.train()
        for i in range(0, len(tr), 512):
            b = tr[i:i + 512]
            with torch.no_grad():
                feats = model.state_enc(S[b])
            loss = F.mse_loss(model.value_head(feats).squeeze(-1), Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    torch.save(model.state_dict(), ROOT / "checkpoints" / out)
    print(f"value-head sign-acc {base:.2f} -> {sign_acc(va):.2f}  saved checkpoints/{out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("--checkpoint", default="checkpoints/bc_v1.pt")
    c.add_argument("--deck", default="kyogre"); c.add_argument("--games", type=int, default=600)
    t = sub.add_parser("train"); t.add_argument("--checkpoint", default="bc_v1.pt")
    t.add_argument("--out", default="bc_v1_value.pt")
    a = p.parse_args()
    if a.cmd == "collect":
        collect(a.checkpoint, a.deck, a.games)
    else:
        train(a.checkpoint, a.out)
