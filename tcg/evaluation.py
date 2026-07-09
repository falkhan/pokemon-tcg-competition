"""Evaluation + replay persistence (ARCHITECTURE.md §4.2, §6).

Merges the old ``rl/eval.py`` and ``rl/replay.py`` — saving a replay is a
step of running evaluation games, so the two lived as one workflow anyway.

Engine note: the cg engine keeps ONE global battle per process (cg.sim.Battle),
so parallel evaluation must use multiprocessing with one environment per worker.

Replay note — why not env.render(mode="html")? The pip-packaged cabt env is
missing its JS visualizer (html_renderer() returns ""), which produces a broken
page (`window.kaggle.renderer = ;`). The supported viewer is the official one
at https://ptcgvis.heroz.jp, which accepts the engine's visualize JSON via form
POST (see visualizer.html in the repo root). So per game we save:
  replays/<name>.html   self-contained page: game metadata + embedded visualize
                        JSON + a button that POSTs it to the official viewer
and maintain:
  replays/manifest.json append-only game metadata (matchup, rewards, steps)
  replays/index.html    regenerated browser: all games + win-rate summary
"""
import datetime
import html
import json
from pathlib import Path

from kaggle_environments import make

REPLAY_DIR = Path(__file__).resolve().parent.parent / "replays"
VIEWER_URL = "https://ptcgvis.heroz.jp/Visualizer/Replay/0"

GAME_PAGE_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
 .meta {{ color: #555; margin: .5rem 0 1.5rem; }}
 button {{ font-size: 1.1rem; padding: .6rem 1.4rem; cursor: pointer; }}
</style></head>
<body>
<h2>{title}</h2>
<div class="meta">{meta}</div>
<button onclick="watch()">▶ Watch replay (opens ptcgvis.heroz.jp)</button>
<script>
const vis = {vis_json};
function watch() {{
  const input = document.createElement("input");
  input.type = "hidden"; input.name = "json";
  input.value = JSON.stringify(vis);
  const form = document.createElement("form");
  form.method = "POST"; form.action = "{viewer}"; form.target = "_blank";
  form.appendChild(input); document.body.appendChild(form); form.submit();
}}
</script>
</body></html>
"""


def agent_name(agent) -> str:
    """A short display name for an agent (path, callable, or built-in name)."""
    if isinstance(agent, str):
        return agent.replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".py") or agent
    return getattr(agent, "__name__", type(agent).__name__)


def play_games(agent_a, agent_b, n_games: int, replay_prefix: str | None = None,
               names: tuple[str, str] | None = None,
               swap_slots: bool = True) -> tuple[float, list[list[int]]]:
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
    """
    names = names or (agent_name(agent_a), agent_name(agent_b))
    results = []
    for game in range(n_games):
        a_slot = game % 2 if swap_slots else 0
        env = make("cabt")
        env.run([agent_a, agent_b] if a_slot == 0 else [agent_b, agent_a])
        rewards = [env.state[0].reward, env.state[1].reward]
        if a_slot == 1:
            rewards = rewards[::-1]           # report from agent_a's perspective
        results.append(rewards)
        if replay_prefix:
            slot_names = names if a_slot == 0 else (names[1], names[0])
            save_replay(env, f"{replay_prefix}_{game:03d}", slot_names)

    wins = sum((rewards[0] or 0) > (rewards[1] or 0) for rewards in results)
    return wins / n_games, results


def option_type_report(agent, opponent="random", n_games: int = 3) -> str:
    """Behavioral fingerprint: in MAIN decisions, which OptionTypes does ``agent``
    pick vs what's offered? (The M0 diagnostic that explained losing to random:
    ATTACH offered 1,439x, chosen 2x.) Returns a printable table.

    ``agent`` must be a callable; ``opponent`` is anything env.run accepts.
    """
    from collections import Counter

    from cg.api import OptionType, SelectContext, to_observation_class

    offered, chosen = Counter(), Counter()

    def spy(obs_dict):
        picks = agent(obs_dict)
        if obs_dict.get("select") is not None:
            observation = to_observation_class(obs_dict)
            if observation.select.context == SelectContext.MAIN:
                for option in observation.select.option:
                    offered[OptionType(option.type).name] += 1
                for i in picks[:observation.select.maxCount]:
                    chosen[OptionType(observation.select.option[i].type).name] += 1
        return picks

    for _ in range(n_games):
        env = make("cabt")
        env.run([spy, opponent])

    lines = [f"{'OptionType':<12} {'offered':>8} {'chosen':>7} {'take%':>6}"]
    for type_name in sorted(offered, key=offered.get, reverse=True):
        take = chosen.get(type_name, 0)
        lines.append(f"{type_name:<12} {offered[type_name]:>8} {take:>7} "
                     f"{100 * take // max(1, offered[type_name]):>5}%")
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


def save_replay(env, name: str, agents: tuple[str, str],
                out_dir: Path = REPLAY_DIR) -> dict | None:
    """Persist one finished game. Returns the manifest entry (None if no vis data)."""
    vis = env.steps[0][0].get("visualize")
    if vis is None:
        return None

    out_dir.mkdir(exist_ok=True)
    rewards = [env.state[0].reward, env.state[1].reward]
    entry = {
        "name": name,
        "agents": list(agents),
        "rewards": rewards,
        "result": ("draw" if rewards[0] == rewards[1]
                   else agents[0] if (rewards[0] or 0) > (rewards[1] or 0) else agents[1]),
        "decisions": len(env.steps),
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    meta = (f"{agents[0]} vs {agents[1]} &nbsp;|&nbsp; winner: <b>{entry['result']}</b>"
            f" &nbsp;|&nbsp; {entry['decisions']} decisions &nbsp;|&nbsp; {entry['saved_at']}")
    page = GAME_PAGE_TEMPLATE.format(
        title=html.escape(name),
        meta=meta,
        # "</" would terminate the <script> block if a card text ever contains it
        vis_json=json.dumps(vis).replace("</", "<\\/"),
        viewer=VIEWER_URL,
    )
    (out_dir / f"{name}.html").write_text(page, encoding="utf-8")

    manifest_path = out_dir / "manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else [])
    manifest = [e for e in manifest if e["name"] != name] + [entry]
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    rebuild_index(out_dir)
    return entry


def rebuild_index(out_dir: Path = REPLAY_DIR) -> None:
    """Regenerate replays/index.html from manifest.json."""
    manifest_path = out_dir / "manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else [])

    # Win-rate summary per matchup (agent-a perspective, order-sensitive).
    stats: dict[tuple, list] = {}
    for entry in manifest:
        key = tuple(entry["agents"])
        tally = stats.setdefault(key, [0, 0, 0])      # wins_a, wins_b, draws
        reward_a = entry["rewards"][0] or 0
        reward_b = entry["rewards"][1] or 0
        tally[0 if reward_a > reward_b else 1 if reward_b > reward_a else 2] += 1

    summary_rows = "".join(
        f"<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td>"
        f"<td>{wins_a}–{wins_b}–{draws}</td>"
        f"<td>{wins_a / max(1, wins_a + wins_b + draws):.0%}</td></tr>"
        for (a, b), (wins_a, wins_b, draws) in sorted(stats.items())
    )
    game_rows = "".join(
        f"<tr><td><a href='{html.escape(entry['name'])}.html'>"
        f"{html.escape(entry['name'])}</a></td>"
        f"<td>{html.escape(entry['agents'][0])}</td>"
        f"<td>{html.escape(entry['agents'][1])}</td>"
        f"<td>{html.escape(str(entry['result']))}</td><td>{entry['decisions']}</td>"
        f"<td>{entry['saved_at']}</td></tr>"
        for entry in reversed(manifest)                # newest first
    )
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Replays</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 70rem; }}
 table {{ border-collapse: collapse; margin: 1rem 0 2rem; width: 100%; }}
 th, td {{ border: 1px solid #ccc; padding: .4rem .8rem; text-align: left; }}
 th {{ background: #f0f0f0; }}
 tr:nth-child(even) {{ background: #fafafa; }}
</style></head>
<body>
<h1>Replays</h1>
<h2>Win rates (row = first agent)</h2>
<table><tr><th>Agent A</th><th>Agent B</th><th>W–L–D (A)</th><th>Win% (A)</th></tr>{summary_rows}</table>
<h2>Games ({len(manifest)})</h2>
<table><tr><th>Replay</th><th>Agent A</th><th>Agent B</th><th>Winner</th><th>Decisions</th><th>Saved</th></tr>{game_rows}</table>
</body></html>
"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")


# TODO: dashboard — aggregate win rate vs {random, rule-based, past self},
# mean game length, prizes taken; write scalars to TensorBoard
# (torch.utils.tensorboard.SummaryWriter).
# TODO: openskill ratings for the opponent pool / deck population.
