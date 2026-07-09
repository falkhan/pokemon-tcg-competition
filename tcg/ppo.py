"""Self-play PPO from the bc_v1 warm start (docs/M2 plan, Phase 1).

Same training loop as the old ``rl/ppo.py``: shard loading, per-(game, seat)
trajectory slicing, padded/masked batching, GAE, the clipped update, and the
iteration loop with its opponent-pool evaluation + promotion gate.

Run one iteration end-to-end:
  python -m tcg.ppo --iterations 1
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tcg.decks import ROOT, ROOT_DECK_PATH, load_deck_file
from tcg.network import MASKED_LOGIT, OptionScorer
from tcg.selfplay import collect

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
PROMOTION_WIN_RATE = 0.55   # challenger must beat the champion by this to be promoted
GRAD_CLIP_NORM = 0.5        # PPO detail: clip the global grad norm every step
ADVANTAGE_NORM_EPS = 1e-8   # keeps the global advantage normalization finite


# ---------------------------------------------------------------------------
# Shard loading + trajectory slicing
# ---------------------------------------------------------------------------

def load_shards(ppo_dir: Path = PPO_DIR) -> dict[str, np.ndarray]:
    """Concatenate all ppo_shard_*.npz into flat arrays (same trick as BCDataset:
    ragged options stay flat, `starts` gives each decision's slice offset).
    game_ids are remapped to be globally unique across shards."""
    cols: dict[str, list] = {}
    game_base = 0
    option_base = 0
    for path in sorted(ppo_dir.glob("ppo_shard_*.npz")):
        shard = np.load(path)
        starts = np.cumsum(shard["n_options"]) - shard["n_options"]
        cols.setdefault("starts", []).append(starts + option_base)
        cols.setdefault("game_ids", []).append(shard["game_ids"] + game_base)
        for name in ("states", "options", "n_options", "actions", "logprobs",
                     "values", "rewards", "players"):
            cols.setdefault(name, []).append(shard[name])
        option_base += len(shard["options"])
        game_base += int(shard["game_ids"].max()) + 1
    return {name: np.concatenate(values) for name, values in cols.items()}


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
    behavior_cloning.collate). Returns tensors ready for ppo_update."""
    batch_size = len(rows)
    n_options = data["n_options"][rows]
    max_options = int(n_options.max())
    option_dim = data["options"].shape[1]

    states = torch.from_numpy(data["states"][rows])
    options = torch.zeros(batch_size, max_options, option_dim)
    valid = torch.zeros(batch_size, max_options, dtype=torch.bool)
    for i, row in enumerate(rows):
        start, n = data["starts"][row], n_options[i]
        options[i, :n] = torch.from_numpy(data["options"][start:start + n])
        valid[i, :n] = True

    return (states, options, valid,
            torch.from_numpy(data["actions"][rows]).long(),
            torch.from_numpy(data["logprobs"][rows]).float(),
            torch.from_numpy(advantages[rows]).float(),
            torch.from_numpy(returns[rows]).float())


# ---------------------------------------------------------------------------
# GAE + the clipped update
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

    Sanity anchor: with lam=1, gamma=1 this reduces to "sum of future rewards
    minus V" — check a 3-step toy by hand against that.
    """
    n_steps = len(rewards)
    advantages = np.zeros(n_steps, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(n_steps)):
        next_value = values[t + 1] if t + 1 < n_steps else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def ppo_update(model: OptionScorer, opt: torch.optim.Optimizer,
               data: dict[str, np.ndarray], advantages: np.ndarray,
               returns: np.ndarray, epochs: int = 4, batch_size: int = 256) -> dict:
    """The clipped PPO update over all collected decisions.

    Per minibatch (use collate_ppo above; shuffle rows each epoch):
      1. logits, value = model(states, options); mask invalid logits to MASKED_LOGIT.
      2. new_logprob   = log_softmax(logits)[action]       # gather along dim 1
      3. ratio         = exp(new_logprob - old_logprob)    # how far policy moved
      4. policy_loss   = -min(ratio * A, clamp(ratio, 1-eps, 1+eps) * A).mean()
      5. entropy       = -(softmax * log_softmax).sum(dim over REAL options).mean()
      6. value_loss    = F.huber_loss(value, returns_batch)
      7. loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
      8. zero_grad / backward / clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM) / step

    Normalize advantages ONCE globally before training:
    (A - A.mean()) / (A.std() + ADVANTAGE_NORM_EPS).
    Return a dict of mean losses/entropy for TensorBoard.
    Read FIRST: the "37 implementation details of PPO" post (ARCHITECTURE.md §9).
    """
    normalized_advantages = ((advantages - advantages.mean())
                             / (advantages.std() + ADVANTAGE_NORM_EPS))
    n_rows = len(data["actions"])
    logs = {"policy_loss": [], "value_loss": [], "entropy": [], "ratio": []}

    for _ in range(epochs):
        perm = np.random.permutation(n_rows)
        for i in range(0, n_rows, batch_size):
            rows = perm[i:i + batch_size]
            states, options, valid, actions, old_logprob, batch_advantages, batch_returns = \
                collate_ppo(data, rows, normalized_advantages, returns)
            logits, value = model(states, options)
            logits = logits.masked_fill(~valid, MASKED_LOGIT)
            logprobs = torch.log_softmax(logits, dim=1)
            new_logprob = logprobs.gather(1, actions.unsqueeze(1)).squeeze(1)

            ratio = torch.exp(new_logprob - old_logprob)
            unclipped = ratio * batch_advantages
            clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * batch_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()

            probs = logprobs.exp()
            entropy = -(probs * logprobs).masked_fill(~valid, 0.0).sum(dim=1).mean()
            value_loss = F.huber_loss(value, batch_returns)

            # SUSPECTED BUG (preserved verbatim — this refactor is behavior-preserving):
            # step 7 of the docstring says `policy_loss + VALUE_COEF * value_loss`, but
            # this MULTIPLIES policy_loss by (VALUE_COEF * value_loss) instead of adding
            # the value term. checkpoints/ppo_run1_entropy_bug/ corroborates a bad run.
            # Fix in a separate, behavior-changing PR so its training effect can be
            # measured in isolation (tests/test_ppo.py pins the current behavior).
            loss = policy_loss * VALUE_COEF * value_loss - ENTROPY_COEF * entropy

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()

            logs["policy_loss"].append(policy_loss.item())
            logs["value_loss"].append(value_loss.item())
            logs["entropy"].append(entropy.item())
            logs["ratio"].append(ratio.mean().item())
    return {name: float(np.mean(values)) for name, values in logs.items()}


# ---------------------------------------------------------------------------
# Iteration loop
# ---------------------------------------------------------------------------

def train(iterations: int, games_per_iter: int = 400, workers: int = 4,
          lr: float = 1e-4, eval_every: int = 5, start: str = "bc_v1.pt"):
    from torch.utils.tensorboard import SummaryWriter
    from tcg.evaluation import play_games
    from tcg.teachers import load_teacher

    writer = SummaryWriter(str(ROOT / "runs" / "ppo"))
    best_path = CKPT_DIR / "ppo_best.pt"
    if not best_path.exists():
        shutil.copy(CKPT_DIR / start, best_path)       # bc_v1 is the initial champion

    model = OptionScorer()
    model.load_state_dict(torch.load(CKPT_DIR / start, map_location="cpu"))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for iteration in range(iterations):
        # 1. fresh self-play data with the CURRENT policy
        for stale in PPO_DIR.glob("ppo_shard_*.npz"):
            stale.unlink()                              # on-policy: stale data is poison
        work = CKPT_DIR / "ppo_current.pt"
        torch.save(model.state_dict(), work)
        collect(games_per_iter, str(work), workers)
        data = load_shards()

        # 2. advantages per (game, seat) trajectory
        advantages = np.zeros(len(data["actions"]), dtype=np.float32)
        returns = np.zeros_like(advantages)
        for rows in trajectory_slices(data):
            trajectory_adv, trajectory_ret = compute_gae(data["rewards"][rows],
                                                         data["values"][rows])
            advantages[rows], returns[rows] = trajectory_adv, trajectory_ret

        # 3. clipped update
        stats = ppo_update(model, opt, data, advantages, returns)
        for name, value in stats.items():
            writer.add_scalar(f"train/{name}", value, iteration)
        print(f"iter {iteration}: {stats}")

        # 4. periodic evaluation + promotion (slot-fair)
        if (iteration + 1) % eval_every == 0:
            challenger = make_greedy_agent(model)
            champion = agent_from_checkpoint(best_path)
            win_rate, _ = play_games(challenger, champion, 200,
                                     names=("challenger", "best"))
            win_rate_teacher, _ = play_games(challenger, load_teacher(f"ev{iteration}"),
                                             100, names=("challenger", "teacher"))
            writer.add_scalar("eval/vs_best", win_rate, iteration)
            writer.add_scalar("eval/vs_teacher", win_rate_teacher, iteration)
            print(f"iter {iteration}: vs_best {win_rate:.1%}  "
                  f"vs_teacher {win_rate_teacher:.1%}")
            if win_rate >= PROMOTION_WIN_RATE:
                torch.save(model.state_dict(), best_path)
                torch.save(model.state_dict(), CKPT_DIR / f"ppo_it{iteration:04d}.pt")
                print(f"iter {iteration}: PROMOTED")


def make_greedy_agent(model: OptionScorer):
    """Greedy agent closure over a live model (for evaluation games)."""
    from cg.api import to_observation_class
    from tcg.encoders import encode_context, encode_option, encode_state

    deck = load_deck_file(ROOT_DECK_PATH)

    def agent(obs_dict: dict) -> list[int]:
        observation = to_observation_class(obs_dict)
        if observation.select is None:
            return deck
        state_ctx = np.concatenate([encode_state(observation.current),
                                    encode_context(observation.select.context)])
        option_vectors = np.stack([encode_option(option, observation)
                                   for option in observation.select.option])
        picks = model.act(state_ctx.astype(np.float32),
                          option_vectors.astype(np.float32),
                          k=observation.select.maxCount, greedy=True)
        return [int(i) for i in picks]
    return agent


def agent_from_checkpoint(path: Path):
    model = OptionScorer()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return make_greedy_agent(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--games-per-iter", type=int, default=400)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--start", type=str, default="bc_v1.pt")
    args = parser.parse_args()
    train(args.iterations, args.games_per_iter, args.workers, lr=args.lr,
          start=args.start)
