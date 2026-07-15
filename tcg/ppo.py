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

from tcg.decks import ROOT
from tcg.network import MASKED_LOGIT, OptionScorer, OptionScorerV2
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
# M7.5 attempt 2: promotion is now vs the FIXED solver ship agent (beat your
# own best measured rate), not a 0.55 bar vs the drifting mirror champion —
# attempt 1 showed those metrics decorrelate (G3 0.44->0.145 while vs_best ~47%).
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
        for name in ("state_ids", "option_ids", "deck_idx"):   # encoders-v2 shards
            if name in shard:
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
    behavior_cloning.collate). Returns tensors ready for ppo_update; the two id
    slots are None for v1 shards and padded long tensors for encoders-v2 shards."""
    batch_size = len(rows)
    n_options = data["n_options"][rows]
    max_options = int(n_options.max())
    option_dim = data["options"].shape[1]
    has_ids = "state_ids" in data

    states = torch.from_numpy(data["states"][rows])
    options = torch.zeros(batch_size, max_options, option_dim)
    valid = torch.zeros(batch_size, max_options, dtype=torch.bool)
    state_ids = (torch.from_numpy(data["state_ids"][rows].astype(np.int64))
                 if has_ids else None)
    option_ids = (torch.zeros(batch_size, max_options, data["option_ids"].shape[1],
                              dtype=torch.long)
                  if has_ids else None)
    for i, row in enumerate(rows):
        start, n = data["starts"][row], n_options[i]
        options[i, :n] = torch.from_numpy(data["options"][start:start + n])
        if has_ids:
            option_ids[i, :n] = torch.from_numpy(
                data["option_ids"][start:start + n].astype(np.int64))
        valid[i, :n] = True

    return (states, state_ids, options, option_ids, valid,
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
               returns: np.ndarray, epochs: int = 4, batch_size: int = 256,
               entropy_coef: float = ENTROPY_COEF) -> dict:
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
            (states, state_ids, options, option_ids, valid, actions, old_logprob,
             batch_advantages, batch_returns) = \
                collate_ppo(data, rows, normalized_advantages, returns)
            if state_ids is None:
                logits, value = model(states, options)
            else:                       # encoders-v2 shards -> OptionScorerV2
                logits, value = model(states, state_ids, options, option_ids)
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

            # M7.4b fix: this was `policy_loss * VALUE_COEF * value_loss` for
            # every stalled PPO run (M2-M4) — a multiplicative loss whose policy
            # gradient scales with the value error (and flips sign with it)
            # instead of the documented additive PPO objective.
            loss = policy_loss + VALUE_COEF * value_loss - entropy_coef * entropy

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

def warm_start_value(model, value_ckpt: Path) -> list[str]:
    """Load ONLY the value head from a value-training checkpoint — the
    supervised critic is the one proven training success (MSE 1.0->0.35,
    sign-acc 0.87); starting PPO from it beats trusting PPO to learn a critic
    from sparse ±1 returns (M7-plan §3.1e). value_head keys are shape-identical
    across OptionScorer and OptionScorerV2 (both operate on the hidden-dim body
    output); for v2 bodies the fit is approximate but a far better prior than
    random. Returns the loaded key names."""
    state_dict = torch.load(value_ckpt, map_location="cpu")
    value_keys = {k: v for k, v in state_dict.items() if k.startswith("value_head.")}
    model.load_state_dict(value_keys, strict=False)
    return sorted(value_keys)


def load_model(ckpt_path: Path):
    """Build the right architecture for a checkpoint: OptionScorerV2 when the
    state dict carries the id-embedding table (M7.3 checkpoints), else v1."""
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model = OptionScorerV2() if "embedding.weight" in state_dict else OptionScorer()
    model.load_state_dict(state_dict)
    return model


def make_kaggle_agent(ckpt_path: Path, deck, instance: str):
    """A kaggle_environments-callable agent from a checkpoint: matchrunner's
    v2/pre-M3-aware model pilot plus the deck-return first call (the direct
    engine loop never asks for the deck; env.run does)."""
    from rl.matchrunner import make_pilot
    act, deck_ids = make_pilot(("model", str(ckpt_path), deck), instance)

    def agent(obs_dict: dict) -> list[int]:
        if obs_dict.get("select") is None:
            return deck_ids
        return [int(i) for i in act(obs_dict)]
    return agent


def solver_opponent(deck: str, instance: str):
    """The FIXED eval opponent (M7.5 attempt 2): the live solver ship agent.
    Attempt 1 promoted on the mirror (vs the drifting champion) and the two
    metrics decorrelated — vs_best held ~47% while G3 collapsed 0.44->0.145.
    Promotion now tracks the actual goal: beat the rules agent we ship."""
    from rl.matchrunner import make_pilot
    act, deck_ids = make_pilot(("solver", deck), instance)

    def agent(obs_dict: dict) -> list[int]:
        if obs_dict.get("select") is None:
            return deck_ids
        return [int(i) for i in act(obs_dict)]
    return agent


def train(iterations: int, games_per_iter: int = 400, workers: int = 4,
          lr: float = 1e-4, eval_every: int = 5, start: str = "bc_v1.pt",
          value_ckpt: str | None = None, decks_file: str | None = None,
          eval_deck: str = "kyogre", entropy_coef: float = ENTROPY_COEF,
          race_shaping: float = 0.0, shaping: str = "race"):
    from torch.utils.tensorboard import SummaryWriter
    from tcg.evaluation import play_games
    from tcg.teachers import load_teacher

    writer = SummaryWriter(str(ROOT / "runs" / "ppo"))
    best_path = CKPT_DIR / "ppo_best.pt"
    if not best_path.exists():
        shutil.copy(CKPT_DIR / start, best_path)       # bc_v1 is the initial champion

    model = load_model(CKPT_DIR / start)
    if value_ckpt:
        loaded = warm_start_value(model, Path(value_ckpt))
        print(f"critic warm-start from {value_ckpt}: {loaded}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # M7.5 attempt 2: promotion tracks the FIXED goal opponent (the solver ship
    # agent), not the drifting mirror champion. Bar = the current best's own
    # measured rate, so a fresh warm start must genuinely improve to promote.
    best_win_rate, _ = play_games(
        make_kaggle_agent(best_path, eval_deck, "ppo_bp_base"),
        solver_opponent(eval_deck, "ppo_sv_base"), 200, names=("best", "solver"))
    print(f"baseline: current best vs_solver {best_win_rate:.1%}", flush=True)

    for iteration in range(iterations):
        # 1. fresh self-play data with the CURRENT policy
        for stale in PPO_DIR.glob("ppo_shard_*.npz"):
            stale.unlink()                              # on-policy: stale data is poison
        work = CKPT_DIR / "ppo_current.pt"
        torch.save(model.state_dict(), work)
        collect(games_per_iter, str(work), workers, decks_file=decks_file,
                race_shaping=race_shaping, shaping=shaping)
        data = load_shards()

        # 2. advantages per (game, seat) trajectory
        advantages = np.zeros(len(data["actions"]), dtype=np.float32)
        returns = np.zeros_like(advantages)
        for rows in trajectory_slices(data):
            trajectory_adv, trajectory_ret = compute_gae(data["rewards"][rows],
                                                         data["values"][rows])
            advantages[rows], returns[rows] = trajectory_adv, trajectory_ret

        # 3. clipped update
        stats = ppo_update(model, opt, data, advantages, returns,
                           entropy_coef=entropy_coef)
        for name, value in stats.items():
            writer.add_scalar(f"train/{name}", value, iteration)
        print(f"iter {iteration}: {stats}")

        # 4. periodic evaluation + promotion vs the fixed goal opponent
        if (iteration + 1) % eval_every == 0:
            challenger = make_kaggle_agent(work, eval_deck, f"ppo_ch{iteration}")
            win_rate, _ = play_games(challenger,
                                     solver_opponent(eval_deck, f"ppo_sv{iteration}"),
                                     200, names=("challenger", "solver"))
            win_rate_teacher, _ = play_games(challenger, load_teacher(f"ev{iteration}"),
                                             100, names=("challenger", "teacher"))
            writer.add_scalar("eval/vs_solver", win_rate, iteration)
            writer.add_scalar("eval/vs_teacher", win_rate_teacher, iteration)
            print(f"iter {iteration}: vs_solver {win_rate:.1%}  "
                  f"vs_teacher {win_rate_teacher:.1%}")
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                torch.save(model.state_dict(), best_path)
                torch.save(model.state_dict(), CKPT_DIR / f"ppo_it{iteration:04d}.pt")
                print(f"iter {iteration}: PROMOTED (best vs_solver {win_rate:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--games-per-iter", type=int, default=400)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--start", type=str, default="bc_v1.pt")
    parser.add_argument("--value-ckpt", type=str, default=None,
                        help="critic warm-start from a value-training checkpoint")
    parser.add_argument("--decks", type=str, default=None,
                        help="deck population for multi-deck self-play (v2 checkpoints)")
    parser.add_argument("--eval-deck", type=str, default="kyogre")
    parser.add_argument("--entropy-coef", type=float, default=ENTROPY_COEF,
                        help="entropy bonus; cut it when entropy RISES over a run "
                             "(the M2 diffusion signature, seen again in attempt 1)")
    parser.add_argument("--race-shaping", type=float, default=0.0,
                        help="potential-based setup-shaping coef for the collector")
    parser.add_argument("--shaping", choices=["race", "dev"], default="race",
                   help="shaping potential: race delta (M7.4b) or the M8.3 "
                        "dev potential (race+ready+evo+bench)")
    parser.add_argument("--eval-every", type=int, default=5,
                        help="iterations between evals (M8.3 decay probes use 2)")
    args = parser.parse_args()
    train(args.iterations, args.games_per_iter, args.workers, lr=args.lr,
          start=args.start, value_ckpt=args.value_ckpt, decks_file=args.decks,
          eval_deck=args.eval_deck, entropy_coef=args.entropy_coef,
          race_shaping=args.race_shaping, shaping=args.shaping,
          eval_every=args.eval_every)
