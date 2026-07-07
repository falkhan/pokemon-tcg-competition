"""Evaluation: run games, compute win rates, dump replays (ARCHITECTURE.md §4.2, §6).

Note: the cg engine keeps ONE global battle per process (cg.sim.Battle), so
parallel evaluation must use multiprocessing with one environment per worker.
"""
from kaggle_environments import make

from .replay import save_replay


def _agent_name(agent) -> str:
    if isinstance(agent, str):
        return agent.replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".py") or agent
    return getattr(agent, "__name__", type(agent).__name__)


def play_games(agent_a, agent_b, n_games: int, replay_prefix: str | None = None,
               names: tuple[str, str] | None = None) -> tuple[float, list[list[int]]]:
    """Run n_games of agent_a vs agent_b. Returns (win_rate_a, per-game rewards).

    Agents may be callables or file paths / built-in names ("random", "first") —
    anything kaggle_environments' env.run accepts. Rewards per game are
    [reward_a, reward_b] with +1 win / -1 loss / 0 draw.

    If replay_prefix is set, EVERY game is saved to replays/ as a self-contained
    HTML page (embedded visualize JSON + button opening the official ptcgvis
    viewer), and replays/index.html is regenerated with win-rate stats.
    """
    names = names or (_agent_name(agent_a), _agent_name(agent_b))
    results = []
    for g in range(n_games):
        env = make("cabt")
        env.run([agent_a, agent_b])
        results.append([env.state[0].reward, env.state[1].reward])
        if replay_prefix:
            save_replay(env, f"{replay_prefix}_{g:03d}", names)

    wins = sum((r[0] or 0) > (r[1] or 0) for r in results)
    return wins / n_games, results


class RecordingAgent:
    """Wrap an agent to log every (obs, action) decision for BC / PPO training."""

    def __init__(self, inner):
        self.inner = inner
        self.traj: list[dict] = []

    def __call__(self, obs_dict: dict) -> list[int]:
        action = self.inner(obs_dict)
        if obs_dict.get("select") is not None:
            self.traj.append({"obs": obs_dict, "action": action})
        return action


# TODO: dashboard — aggregate win rate vs {random, rule-based, past self},
# mean game length, prizes taken; write scalars to TensorBoard
# (torch.utils.tensorboard.SummaryWriter).
# TODO: openskill ratings for the opponent pool / deck population.
