"""Replay persistence + a local browser UI for watching games.

Why not env.render(mode="html")? The pip-packaged cabt env is missing its JS
visualizer (html_renderer() returns ""), which produces a broken page
(`window.kaggle.renderer = ;`). The supported viewer is the official one at
https://ptcgvis.heroz.jp, which accepts the engine's visualize JSON via form
POST (see visualizer.html in the repo root).

This module therefore saves, per game:
  replays/<name>.html   self-contained page: game metadata + embedded visualize
                        JSON + a button that POSTs it to the official viewer
and maintains:
  replays/manifest.json append-only game metadata (matchup, rewards, steps)
  replays/index.html    regenerated browser: all games + win-rate summary
"""
import datetime
import html
import json
from pathlib import Path

REPLAY_DIR = Path(__file__).resolve().parent.parent / "replays"
VIEWER_URL = "https://ptcgvis.heroz.jp/Visualizer/Replay/0"

_GAME_TEMPLATE = """<!DOCTYPE html>
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


def save_replay(env, name: str, agents: tuple[str, str], out_dir: Path = REPLAY_DIR) -> dict | None:
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
    page = _GAME_TEMPLATE.format(
        title=html.escape(name),
        meta=meta,
        # "</" would terminate the <script> block if a card text ever contains it
        vis_json=json.dumps(vis).replace("</", "<\\/"),
        viewer=VIEWER_URL,
    )
    (out_dir / f"{name}.html").write_text(page, encoding="utf-8")

    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    manifest = [e for e in manifest if e["name"] != name] + [entry]
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    rebuild_index(out_dir)
    return entry


def rebuild_index(out_dir: Path = REPLAY_DIR) -> None:
    """Regenerate replays/index.html from manifest.json."""
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []

    # Win-rate summary per matchup (agent-a perspective, order-sensitive).
    stats: dict[tuple, list] = {}
    for e in manifest:
        key = tuple(e["agents"])
        s = stats.setdefault(key, [0, 0, 0])          # wins_a, wins_b, draws
        ra, rb = e["rewards"][0] or 0, e["rewards"][1] or 0
        s[0 if ra > rb else 1 if rb > ra else 2] += 1

    summary_rows = "".join(
        f"<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td>"
        f"<td>{w}–{l}–{d}</td><td>{w / max(1, w + l + d):.0%}</td></tr>"
        for (a, b), (w, l, d) in sorted(stats.items())
    )
    game_rows = "".join(
        f"<tr><td><a href='{html.escape(e['name'])}.html'>{html.escape(e['name'])}</a></td>"
        f"<td>{html.escape(e['agents'][0])}</td><td>{html.escape(e['agents'][1])}</td>"
        f"<td>{html.escape(str(e['result']))}</td><td>{e['decisions']}</td>"
        f"<td>{e['saved_at']}</td></tr>"
        for e in reversed(manifest)                    # newest first
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
