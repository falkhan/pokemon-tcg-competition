"""Self-play PPO from the bc_v1 warm start (docs/M2 plan, Phase 1).

Division of labor (same as M1):
  Claude scaffold (done):  shard loading, per-(game,seat) trajectory slicing,
                           padded/masked batching, the iteration loop, opponent-pool
                           evaluation + promotion gate, TensorBoard, checkpoints.
  PIOTR (the core):        compute_gae() and ppo_update() — marked with
                           NotImplementedError below. This is the RL heart of M2.

Run one iteration end-to-end (will stop at your NotImplementedError):
  python -m rl.ppo --iterations 1
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rl.collector import collect
from rl.policy import OptionScorer, OptionScorerV2

ROOT = Path(__file__).resolve().parent.parent
PPO_DIR = ROOT / "data" / "ppo"
CKPT_DIR = ROOT / "checkpoints"

GAMMA = 0.99          # discount: how much future reward is worth now
LAM = 0.95            # GAE lambda: bias/variance dial for advantage estimates
CLIP_EPS = 0.2        # PPO ratio clip
ENTROPY_COEF = 0.001  # was 0.01 -- too high: the 25-iter run showed entropy RISING
                      # (0.82 -> 1.0), i.e. the bonus overpowered the weak advantage signal
                      # and diffused the policy toward random. Lowered 10x so the policy
                      # gradient can actually sharpen the policy.
VALUE_COEF = 0.5


# ---------------------------------------------------------------------------
# Shard loading + trajectory slicing (infra, done)
# ---------------------------------------------------------------------------

def load_shards(ppo_dir: Path = PPO_DIR) -> dict[str, np.ndarray]:
    """Concatenate all ppo_shard_*.npz into flat arrays (same trick as BCDataset:
    ragged options stay flat, `starts` gives each decision's slice offset).
    game_ids are remapped to be globally unique across shards."""
    cols: dict[str, list] = {}
    game_base = 0
    option_base = 0
    for path in sorted(ppo_dir.glob("ppo_shard_*.npz")):
        z = np.load(path)
        starts = np.cumsum(z["n_options"]) - z["n_options"]
        cols.setdefault("starts", []).append(starts + option_base)
        cols.setdefault("game_ids", []).append(z["game_ids"] + game_base)
        for k in ("states", "options", "n_options", "actions", "logprobs",
                  "values", "rewards", "players"):
            cols.setdefault(k, []).append(z[k])
        for k in ("state_ids", "option_ids", "deck_idx"):   # encoders-v2 shards (M7.4b)
            if k in z:
                cols.setdefault(k, []).append(z[k])
        option_base += len(z["options"])
        game_base += int(z["game_ids"].max()) + 1
    return {k: np.concatenate(v) for k, v in cols.items()}


def trajectory_slices(data: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Row indices of each (game, seat) trajectory, in decision order.

    GAE runs over one agent's consecutive decisions; the shards interleave both
    seats chronologically, so this de-interleaves them. Row order within a shard
    is already chronological."""
    key = data["game_ids"].astype(np.int64) * 2 + data["players"]
    order = np.argsort(key, kind="stable")           # stable keeps chronology
    slices, prev, start = [], None, 0
    for pos, idx in enumerate(order):
        if key[idx] != prev:
            if prev is not None:
                slices.append(order[start:pos])
            prev, start = key[idx], pos
    slices.append(order[start:])
    return slices


def collate_ppo(data: dict[str, np.ndarray], rows: np.ndarray,
                advantages: np.ndarray, returns: np.ndarray):
    """Build one padded/masked minibatch from row indices (same padding idea as
    rl/bc.py collate). Returns tensors ready for ppo_update; the two id slots
    are None for v1 shards and padded long tensors for encoders-v2 shards."""
    B = len(rows)
    n_opts = data["n_options"][rows]
    maxN = int(n_opts.max())
    opt_dim = data["options"].shape[1]
    has_ids = "state_ids" in data

    states = torch.from_numpy(data["states"][rows])
    options = torch.zeros(B, maxN, opt_dim)
    valid = torch.zeros(B, maxN, dtype=torch.bool)
    state_ids = (torch.from_numpy(data["state_ids"][rows].astype(np.int64))
                 if has_ids else None)
    option_ids = (torch.zeros(B, maxN, data["option_ids"].shape[1], dtype=torch.long)
                  if has_ids else None)
    for i, r in enumerate(rows):
        s, n = data["starts"][r], n_opts[i]
        options[i, :n] = torch.from_numpy(data["options"][s:s + n])
        if has_ids:
            option_ids[i, :n] = torch.from_numpy(
                data["option_ids"][s:s + n].astype(np.int64))
        valid[i, :n] = True

    return (states, state_ids, options, option_ids, valid,
            torch.from_numpy(data["actions"][rows]).long(),
            torch.from_numpy(data["logprobs"][rows]).float(),
            torch.from_numpy(advantages[rows]).float(),
            torch.from_numpy(returns[rows]).float())


# ---------------------------------------------------------------------------
# THE CORE — Piotr's part
# ---------------------------------------------------------------------------

def compute_gae(rewards: np.ndarray, values: np.ndarray,
                gamma: float = GAMMA, lam: float = LAM) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation over ONE trajectory (one game, one seat).

    Inputs are that trajectory's per-decision rewards and value predictions, in
    order. The episode ends after the last decision (terminal value = 0).

    Returns (advantages, returns) for the trajectory, where
      delta_t   = r_t + gamma * V(s_{t+1}) - V(s_t)        # TD error; V(s_T) = 0
      A_t       = delta_t + gamma * lam * A_{t+1}          # computed BACKWARD
      return_t  = A_t + V(s_t)                             # value-head target

    Hints: walk the arrays back-to-front with a running accumulator (exactly like
    the reversed() loop in the official MCTS sample's value labeling). Loss ~2.0
    moment here: with lam=1, gamma=1 this reduces to "sum of future rewards minus
    V" — sanity-check your implementation against that by hand on a 3-step toy.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def ppo_update(model: OptionScorer, opt: torch.optim.Optimizer,
               data: dict[str, np.ndarray], advantages: np.ndarray,
               returns: np.ndarray, epochs: int = 4, batch_size: int = 256,
               entropy_coef: float = ENTROPY_COEF) -> dict:
    """The clipped PPO update over all collected decisions.

    Per minibatch (use collate_ppo above; shuffle rows each epoch):
      1. logits, value = model(states, options); mask invalid logits to -1e9.
      2. new_logprob   = log_softmax(logits)[action]       # gather along dim 1
      3. ratio         = exp(new_logprob - old_logprob)    # how far policy moved
      4. policy_loss   = -min(ratio * A, clamp(ratio, 1-eps, 1+eps) * A).mean()
      5. entropy       = -(softmax * log_softmax).sum(dim over REAL options).mean()
      6. value_loss    = F.huber_loss(value, returns_batch)
      7. loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
      8. zero_grad / backward / clip_grad_norm_(model.parameters(), 0.5) / step

    Normalize advantages ONCE globally before training: (A - A.mean()) / (A.std()+1e-8).
    Return a dict of mean losses/entropy for TensorBoard.
    Read FIRST: the "37 implementation details of PPO" post (ARCHITECTURE.md §9).
    """
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    n = len(data["actions"])
    logs = {"policy_loss": [], "value_loss": [], "entropy": [], "ratio": []}

    for _ in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, batch_size):
            rows = perm[i:i + batch_size]
            states, state_ids, options, option_ids, valid, actions, old_lp, adv_b, ret_b = \
                collate_ppo(data, rows, adv, returns)
            if state_ids is None:
                logits, value = model(states, options)
            else:                       # encoders-v2 shards -> OptionScorerV2
                logits, value = model(states, state_ids, options, option_ids)
            logits = logits.masked_fill(~valid, -1e9)
            logp = torch.log_softmax(logits, dim=1)
            new_lp = logp.gather(1, actions.unsqueeze(1)).squeeze(1)

            ratio = torch.exp(new_lp - old_lp)
            unclipped = ratio * adv_b
            clipped = torch.clamp(ratio, 1- CLIP_EPS, 1 + CLIP_EPS) * adv_b
            policy_loss = -torch.min(unclipped, clipped).mean()

            probs = logp.exp()
            entropy = -(probs * logp).masked_fill(~valid, 0.0).sum(dim=1).mean()
            value_loss = F.huber_loss(value, ret_b)

            # M7.4b fix: this was `policy_loss * VALUE_COEF * value_loss` for
            # every stalled PPO run (M2-M4) — a multiplicative loss whose policy
            # gradient scales with the value error (and flips sign with it)
            # instead of the documented additive PPO objective.
            loss = policy_loss + VALUE_COEF * value_loss - entropy_coef * entropy

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

            logs["policy_loss"].append(policy_loss.item())
            logs["value_loss"].append(value_loss.item())
            logs["entropy"].append(entropy.item())
            logs["ratio"].append(ratio.mean().item())
    return {k: float(np.mean(v)) for k, v in logs.items()}

# ---------------------------------------------------------------------------
# Iteration loop (infra, done)
# ---------------------------------------------------------------------------

def warm_start_value(model, value_ckpt: Path) -> list[str]:
    """Load ONLY the value head from a rl/value_train.py checkpoint — the
    supervised critic is the one proven training success (MSE 1.0->0.35,
    sign-acc 0.87); starting PPO from it beats trusting PPO to learn a critic
    from sparse ±1 returns (M7-plan §3.1e). value_head keys are shape-identical
    across OptionScorer and OptionScorerV2 (both operate on the hidden-dim body
    output); for v2 bodies the fit is approximate but a far better prior than
    random. Returns the loaded key names."""
    sd = torch.load(value_ckpt, map_location="cpu")
    value_keys = {k: v for k, v in sd.items() if k.startswith("value_head.")}
    model.load_state_dict(value_keys, strict=False)
    return sorted(value_keys)


def _load_model(ckpt_path: Path):
    """Build the right architecture for a checkpoint: OptionScorerV2 when the
    state dict carries the id-embedding table (M7.3 checkpoints), else v1."""
    sd = torch.load(ckpt_path, map_location="cpu")
    model = OptionScorerV2() if "embedding.weight" in sd else OptionScorer()
    model.load_state_dict(sd)
    return model


def _as_kaggle_agent(ckpt_path: Path, deck, instance: str):
    """A kaggle_environments-callable agent from a checkpoint: matchrunner's
    v2/pre-M3-aware model pilot plus the deck-return first call (the direct
    engine loop never asks for the deck; env.run does)."""
    from rl.matchrunner import make_pilot
    fn, deck_ids = make_pilot(("model", str(ckpt_path), deck), instance)

    def agent(obs_dict: dict) -> list[int]:
        if obs_dict.get("select") is None:
            return deck_ids
        return [int(i) for i in fn(obs_dict)]
    return agent


def _solver_opponent(deck: str, instance: str):
    """The FIXED eval opponent (M7.5 attempt 2): the live solver ship agent.
    Attempt 1 promoted on the mirror (vs the drifting champion) and the two
    metrics decorrelated — vs_best held ~47% while G3 collapsed 0.44->0.145.
    Promotion now tracks the actual goal: beat the rules agent we ship."""
    from rl.matchrunner import make_pilot
    fn, deck_ids = make_pilot(("solver", deck), instance)

    def agent(obs_dict: dict) -> list[int]:
        if obs_dict.get("select") is None:
            return deck_ids
        return [int(i) for i in fn(obs_dict)]
    return agent


def train(iterations: int, games_per_iter: int = 400, workers: int = 4,
          lr: float = 1e-4, eval_every: int = 5, start: str = "bc_v1.pt",
          value_ckpt: str | None = None, decks_file: str | None = None,
          eval_deck: str = "kyogre", entropy_coef: float = ENTROPY_COEF,
          race_shaping: float = 0.0, shaping: str = "race"):
    from torch.utils.tensorboard import SummaryWriter
    from rl.eval import play_games
    from rl.teacher import load_teacher

    writer = SummaryWriter(str(ROOT / "runs" / "ppo"))
    best_path = CKPT_DIR / "ppo_best.pt"
    if not best_path.exists():
        shutil.copy(CKPT_DIR / start, best_path)       # bc_v1 is the initial champion

    model = _load_model(CKPT_DIR / start)
    if value_ckpt:
        loaded = warm_start_value(model, Path(value_ckpt))
        print(f"critic warm-start from {value_ckpt}: {loaded}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # M7.5 attempt 2: promotion tracks the FIXED goal opponent (the solver ship
    # agent), not the drifting mirror champion. Bar = the current best's own
    # measured rate, so a fresh warm start must genuinely improve to promote.
    best_wr, _ = play_games(_as_kaggle_agent(best_path, eval_deck, "ppo_bp_base"),
                            _solver_opponent(eval_deck, "ppo_sv_base"), 200,
                            names=("best", "solver"))
    print(f"baseline: current best vs_solver {best_wr:.1%}", flush=True)

    for it in range(iterations):
        # 1. fresh self-play data with the CURRENT policy
        for old in PPO_DIR.glob("ppo_shard_*.npz"):
            old.unlink()                                # on-policy: stale data is poison
        work = CKPT_DIR / "ppo_current.pt"
        torch.save(model.state_dict(), work)
        collect(games_per_iter, str(work), workers, decks_file=decks_file,
                race_shaping=race_shaping, shaping=shaping)
        data = load_shards()

        # 2. advantages per (game, seat) trajectory  [Piotr's compute_gae]
        advantages = np.zeros(len(data["actions"]), dtype=np.float32)
        returns = np.zeros_like(advantages)
        for rows in trajectory_slices(data):
            adv, ret = compute_gae(data["rewards"][rows], data["values"][rows])
            advantages[rows], returns[rows] = adv, ret

        # 3. clipped update  [Piotr's ppo_update]
        stats = ppo_update(model, opt, data, advantages, returns,
                           entropy_coef=entropy_coef)
        for k, v in stats.items():
            writer.add_scalar(f"train/{k}", v, it)
        print(f"iter {it}: {stats}")

        # 4. periodic evaluation + promotion vs the fixed goal opponent
        if (it + 1) % eval_every == 0:
            challenger = _as_kaggle_agent(work, eval_deck, f"ppo_ch{it}")
            wr, _ = play_games(challenger, _solver_opponent(eval_deck, f"ppo_sv{it}"),
                               200, names=("challenger", "solver"))
            wr_teacher, _ = play_games(challenger, load_teacher(f"ev{it}"), 100,
                                       names=("challenger", "teacher"))
            writer.add_scalar("eval/vs_solver", wr, it)
            writer.add_scalar("eval/vs_teacher", wr_teacher, it)
            print(f"iter {it}: vs_solver {wr:.1%}  vs_teacher {wr_teacher:.1%}")
            if wr > best_wr:
                best_wr = wr
                torch.save(model.state_dict(), best_path)
                torch.save(model.state_dict(), CKPT_DIR / f"ppo_it{it:04d}.pt")
                print(f"iter {it}: PROMOTED (best vs_solver {wr:.1%})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--games-per-iter", type=int, default=400)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--start", type=str, default="bc_v1.pt")
    p.add_argument("--value-ckpt", type=str, default=None,
                   help="critic warm-start from a rl.value_train checkpoint")
    p.add_argument("--decks", type=str, default=None,
                   help="deck population for multi-deck self-play (v2 checkpoints)")
    p.add_argument("--eval-deck", type=str, default="kyogre")
    p.add_argument("--entropy-coef", type=float, default=ENTROPY_COEF,
                   help="entropy bonus; cut it when entropy RISES over a run "
                        "(the M2 diffusion signature, seen again in attempt 1)")
    p.add_argument("--race-shaping", type=float, default=0.0,
                   help="potential-based setup-shaping coef for the collector")
    p.add_argument("--shaping", choices=["race", "dev"], default="race",
                   help="shaping potential: race delta (M7.4b) or the M8.3 "
                        "dev potential (race+ready+evo+bench)")
    p.add_argument("--eval-every", type=int, default=5,
                   help="iterations between evals (M8.3 decay probes use 2)")
    args = p.parse_args()
    train(args.iterations, args.games_per_iter, args.workers, lr=args.lr,
          start=args.start, value_ckpt=args.value_ckpt, decks_file=args.decks,
          eval_deck=args.eval_deck, entropy_coef=args.entropy_coef,
          race_shaping=args.race_shaping, shaping=args.shaping,
          eval_every=args.eval_every)
