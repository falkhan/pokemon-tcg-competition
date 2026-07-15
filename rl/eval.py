"""Evaluation: run games, compute win rates, dump replays (ARCHITECTURE.md §4.2, §6).

Note: the cg engine keeps ONE global battle per process (cg.sim.Battle), so
parallel evaluation must use multiprocessing with one environment per worker.
"""
import json
from pathlib import Path

from kaggle_environments import make

from .replay import save_replay


def _agent_name(agent) -> str:
    if isinstance(agent, str):
        return agent.replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".py") or agent
    return getattr(agent, "__name__", type(agent).__name__)


def play_games(agent_a, agent_b, n_games: int, replay_prefix: str | None = None,
               names: tuple[str, str] | None = None,
               swap_slots: bool = True,
               json_prefix: str | None = None) -> tuple[float, list[list[int]]]:
    """Run n_games of agent_a vs agent_b. Returns (win_rate_a, per-game rewards).

    Agents may be callables or file paths / built-in names ("random", "first") —
    anything kaggle_environments' env.run accepts. Rewards per game are
    [reward_a, reward_b] with +1 win / -1 loss / 0 draw, ALWAYS reported from
    agent_a's perspective regardless of which slot it occupied.

    swap_slots (default True): agent_a plays even games in the player-0 slot and
    odd games in player-1. Slot 0 carries a measured ~61% built-in advantage in
    this engine, so unswapped head-to-head numbers are biased by ~+10pp; only
    disable this for slot-bias experiments.

    If replay_prefix is set, EVERY game is saved to replays/ as a self-contained
    HTML page (embedded visualize JSON + button opening the official ptcgvis
    viewer), and replays/index.html is regenerated with win-rate stats.

    If json_prefix is set (M8.0), every game's env.toJSON() is written to
    <json_prefix>_g<NNN>_a<slot>.json — the same shape as a cached Kaggle
    episode, so rl/postmortem.py (incl. `--batch`) reads it directly; the
    `_a<slot>` suffix records which seat agent_a occupied that game.
    """
    names = names or (_agent_name(agent_a), _agent_name(agent_b))
    results = []
    for g in range(n_games):
        a_slot = g % 2 if swap_slots else 0
        env = make("cabt")
        env.run([agent_a, agent_b] if a_slot == 0 else [agent_b, agent_a])
        r = [env.state[0].reward, env.state[1].reward]
        if a_slot == 1:
            r = r[::-1]                       # report from agent_a's perspective
        results.append(r)
        if replay_prefix:
            slot_names = names if a_slot == 0 else (names[1], names[0])
            save_replay(env, f"{replay_prefix}_{g:03d}", slot_names)
        if json_prefix:
            path = Path(f"{json_prefix}_g{g:03d}_a{a_slot}.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(env.toJSON()))

    wins = sum((r[0] or 0) > (r[1] or 0) for r in results)
    return wins / n_games, results


def option_type_report(agent, opponent="random", n_games: int = 3) -> str:
    """Behavioral fingerprint: in MAIN decisions, which OptionTypes does `agent`
    pick vs what's offered? (The M0 diagnostic that explained losing to random:
    ATTACH offered 1,439x, chosen 2x.) Returns a printable table.

    `agent` must be a callable; `opponent` is anything env.run accepts.
    """
    from collections import Counter

    from cg.api import OptionType, SelectContext, to_observation_class

    offered, chosen = Counter(), Counter()

    def spy(obs_dict):
        picks = agent(obs_dict)
        if obs_dict.get("select") is not None:
            obs = to_observation_class(obs_dict)
            if obs.select.context == SelectContext.MAIN:
                for o in obs.select.option:
                    offered[OptionType(o.type).name] += 1
                for i in picks[:obs.select.maxCount]:
                    chosen[OptionType(obs.select.option[i].type).name] += 1
        return picks

    for _ in range(n_games):
        env = make("cabt")
        env.run([spy, opponent])

    lines = [f"{'OptionType':<12} {'offered':>8} {'chosen':>7} {'take%':>6}"]
    for t in sorted(offered, key=offered.get, reverse=True):
        take = chosen.get(t, 0)
        lines.append(f"{t:<12} {offered[t]:>8} {take:>7} {100 * take // max(1, offered[t]):>5}%")
    return "\n".join(lines)


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
