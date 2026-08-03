"""Generates notebooks/leaderboard_decks.ipynb. Edit here, re-run, re-open the notebook.

Kept as a .py because hand-editing .ipynb JSON is how cells silently break
(the lesson recorded in build_m22_ab_monitor.py).
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "leaderboard_decks.ipynb"


def md(*src):
    return {"cell_type": "markdown", "metadata": {}, "source": list(src)}


def code(*src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": list(src)}


def _with_ids(cells):
    for i, c in enumerate(cells):
        c["id"] = f"lbd-{i:02d}"
    return cells


CELLS = [
    md("""# 🏆 Leaderboard Deck Census — what is the top of the ladder playing?

We weight gate beds against `data/m39_live_mix.json`: the mix our agent has
actually **played**. That is not the same population as the **top of the ladder**,
and until now nobody had measured the difference.

This notebook labels every team in the leaderboard top-N with the deck it is
piloting, identified from the cards themselves — energy type, owner theme
(*Marnie's*, *Team Rocket's*), and the Mega/ex Pokémon doing the attacking.

**How to use:** run cell 1 (setup + load). To pull a fresh leaderboard, run the
CLI and re-run cell 1:

```
uv run python scripts/leaderboard_decks.py --top 250
```

## How a deck gets identified

    leaderboard(top)        ->  team_id, score          (the only network call)
    episodes.parquet        ->  team_id -> submission_id (latest by end_time)
    opp_decks.parquet       ->  submission_id -> the 60-card decklist
    cards_features.parquet  ->  decklist -> energy / theme / main attacker
    data/kaggle/raw/*.gz    ->  what the pilot ACTUALLY attacked with

## What this cannot see

Kaggle retired the `ListEpisodes` teamId filter (probed 2026-07-15), so a team we
have **never played** is unreachable by any targeted call — only the snowball
harvest finds them. This notebook is offline past the leaderboard call: gaps are
reported as `never_seen` / `no_deck`, never quietly filled or dropped. Read the
coverage tiles before quoting any share."""),

    code("""# 1 — setup + load the census. Run once (re-run after refreshing the CLI).
import json
import subprocess
import sys
from pathlib import Path

# repo-root bootstrap: the kernel's cwd is notebooks/
_ROOT = Path.cwd()
while not (_ROOT / "rl").exists() and _ROOT != _ROOT.parent:
    _ROOT = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl

CENSUS_PQ = _ROOT / "data/kaggle/leaderboard_decks.parquet"
if not CENSUS_PQ.exists():
    raise SystemExit(
        f"{CENSUS_PQ} missing — build it first:\\n"
        "  uv run python scripts/leaderboard_decks.py --top 250")

# pyarrow is not installed in this venv: pl.to_pandas() raises. Go via dicts,
# the same route model_monitor.ipynb uses.
census = pd.DataFrame(pl.read_parquet(CENSUS_PQ).to_dicts())
known = census[census["label"].notna()].copy()

# --- house style (matches notebooks/model_monitor.ipynb) --------------------
INK, INK2, GRID, SURFACE = "#333333", "#666666", "#e5e5e5", "#fcfcfb"

# Validated with the dataviz skill validator against surface #fcfcfb.
# 6-slot categorical (rank-band stack): ALL CHECKS PASS, worst adjacent CVD
# dE 9.1 (protan). Contrast WARN on 3 slots -> discharged by direct labels +
# the table view under every chart.
PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834"]
# Emphasis form (one bar is the story, the rest are context): 1 hue + gray,
# CVD dE 22.2, contrast PASS.
ACCENT, MUTED = "#2a78d6", "#8a8a8a"
# Two populations compared per family (dumbbell): blue/orange, CVD dE 24.7.
LADDER_C, OURS_C = "#2a78d6", "#eb6834"

pio.templates["pokedex"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=13, color=INK),
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    title=dict(font=dict(size=17, color=INK), x=0, xanchor="left"),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
               tickfont=dict(color=INK2)),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
               tickfont=dict(color=INK2)),
    legend=dict(font=dict(color=INK2), bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=70, r=40, t=70, b=50), hoverlabel=dict(font_size=12),
))
pio.templates.default = "pokedex"

TOP_N = len(census)
print(f"census: {TOP_N} teams, {len(known)} labeled "
      f"({len(known) / max(TOP_N, 1):.0%})")"""),

    md("""## 1. Coverage — read this before quoting any share

A share is only as good as the denominator. `never_seen` teams are the honest
ceiling on this survey: we have no episode against them at all, and no targeted
call can reach them."""),

    code("""# 2 — coverage tiles + table
from IPython.display import HTML, display

cov = census["coverage"].value_counts().to_dict()
labeled = len(known)
stale = int(known["last_seen"].fillna("").lt("2026-07-30").sum())

def _tile(accent, label, value, sub):
    return (f'<div style="flex:1;min-width:150px;background:{SURFACE};'
            f'border:1px solid {GRID};border-left:5px solid {accent};'
            f'border-radius:7px;padding:11px 15px;margin:4px">'
            f'<div style="color:{INK2};font-size:11px;text-transform:uppercase;'
            f'letter-spacing:.6px">{label}</div>'
            f'<div style="color:{INK};font-size:25px;font-weight:650;'
            f'line-height:1.25">{value}</div>'
            f'<div style="color:{INK2};font-size:11px">{sub}</div></div>')

display(HTML(
    '<div style="display:flex;flex-wrap:wrap;font-family:Inter,Segoe UI,sans-serif">'
    + _tile(ACCENT, "teams surveyed", TOP_N,
            f'score {census["score"].min():.0f} – {census["score"].max():.0f}')
    + _tile("#008300", "decks identified", f"{labeled / max(TOP_N, 1):.0%}",
            f"{labeled} of {TOP_N} teams")
    + _tile("#eda100", "never played", cov.get("never_seen", 0),
            "no episode on record — unreachable")
    + _tile(MUTED, "seen, no decklist", cov.get("no_deck", 0),
            f"{stale} labeled from a stale replay")
    + "</div>"))

display(pd.DataFrame([
    {"coverage": k, "teams": v, "share": f"{v / max(TOP_N, 1):.1%}",
     "meaning": {
         "current": "decklist from their newest submission",
         "older_submission": "newest submission unseen; labeled from an older one",
         "no_deck": "played them, but no cached replay carries a decklist",
         "never_seen": "no episode against them at all",
     }[k]}
    for k, v in sorted(cov.items(), key=lambda kv: -kv[1])
]).style.hide(axis="index").set_caption("Coverage of the leaderboard top-N"))"""),

    md("""## 2. The deck distribution

Sorted by how many teams pilot it. The leader is drawn in the accent hue and the
rest in gray — the story here is the concentration, not fifteen separate
identities. Energy and attacker tier are carried in the table below, not by
color: energy type has ten values, and ten hues cannot be told apart."""),

    code("""# 3 — deck distribution across the surveyed leaderboard
dist = (known.groupby("label")
             .agg(teams=("label", "size"), mean_score=("score", "mean"),
                  best_rank=("rank", "min"), energy=("primary_energy", "first"),
                  tier=("attacker_tier", "first"), theme=("theme", "first"))
             .sort_values("teams", ascending=False).reset_index())
dist["share"] = dist["teams"] / len(known)

top = dist.head(14).iloc[::-1]  # plotly draws bottom-up
colors = [ACCENT if v == dist["teams"].max() else MUTED for v in top["teams"]]

fig = go.Figure(go.Bar(
    x=top["teams"], y=top["label"], orientation="h",
    marker=dict(color=colors, line=dict(color=SURFACE, width=2)),
    text=[f"{n}  ({s:.0%})" for n, s in zip(top["teams"], top["share"])],
    textposition="outside", textfont=dict(color=INK2, size=11),
    customdata=top[["mean_score", "energy", "tier", "best_rank"]].values,
    hovertemplate=("<b>%{y}</b><br>%{x} teams<br>mean score %{customdata[0]:.0f}"
                   "<br>energy %{customdata[1]}<br>tier %{customdata[2]}"
                   "<br>best rank #%{customdata[3]}<extra></extra>"),
))
fig.update_layout(
    title=f"Decks piloted across the leaderboard top {TOP_N}"
          f"<br><sub>{len(known)} identified teams · one bar per deck</sub>",
    xaxis_title="teams piloting it", yaxis_title=None,
    height=460, bargap=0.28,
    # automargin: deck names run to ~26 chars and would clip the 70px template margin
    xaxis=dict(range=[0, dist["teams"].max() * 1.22]),
    yaxis=dict(automargin=True),
)
fig.show()

display(dist.assign(share=lambda d: d["share"].map("{:.1%}".format))
            .rename(columns={"label": "deck (main attacker)"})
            .style.hide(axis="index")
            .format({"mean_score": "{:.0f}"})
            .bar(subset=["teams"], color="#cfe0f5")
            .set_caption("Every identified deck — the table view of the chart above"))"""),

    md("""## 3. Does the popular deck actually win?

Popularity and performance are different questions. This is the score
distribution per deck (families with at least 3 teams), sorted by median."""),

    code("""# 4 — score distribution per deck
big = (known.groupby("label").filter(lambda g: len(g) >= 3))
order = big.groupby("label")["score"].median().sort_values().index.tolist()

fig = go.Figure()
for name in order:
    s = big[big["label"] == name]
    fig.add_trace(go.Box(
        x=s["score"], name=name, orientation="h", boxpoints="all",
        jitter=0.45, pointpos=0, marker=dict(color=ACCENT, size=6, opacity=0.55),
        line=dict(color=INK2, width=1.5), fillcolor="rgba(42,120,214,0.12)",
        hovertemplate="<b>%{y}</b><br>score %{x:.0f}<extra></extra>",
    ))
fig.update_layout(
    title="Leaderboard score by deck"
          "<br><sub>decks with >=3 teams · every team shown as a point</sub>",
    xaxis_title="leaderboard score", showlegend=False,
    height=90 + 52 * len(order),
    yaxis=dict(automargin=True),
)
fig.show()

display(big.groupby("label")["score"]
           .agg(teams="size", median="median", best="max", worst="min")
           .sort_values("median", ascending=False)
           .style.format({"median": "{:.0f}", "best": "{:.0f}", "worst": "{:.0f}"})
           .set_caption("Score by deck (>=3 teams)"))"""),

    md("""## 4. Where in the ranking does each family sit?

Share of each rank band, using the canonical gate families
(`scripts/m39_live_mix.py`). The tail folds into **other** — a generated 7th hue
would be indistinguishable from an existing one under colorblind vision."""),

    code("""# 5 — family share by rank band
BANDS = 5
band_size = max(1, TOP_N // BANDS)
k = known.copy()
k["band"] = ((k["rank"] - 1) // band_size).clip(upper=BANDS - 1)
k["band_label"] = k["band"].map(
    lambda b: f"#{b * band_size + 1}–{min((b + 1) * band_size, TOP_N)}")

top_fams = known["family"].value_counts().head(5).index.tolist()
k["fam"] = k["family"].where(k["family"].isin(top_fams), "other")
FAM_COLOR = dict(zip(top_fams, PALETTE))
FAM_COLOR["other"] = MUTED

pivot = (k.pivot_table(index="band_label", columns="fam", values="rank",
                       aggfunc="size", fill_value=0)
          .reindex([f"#{b * band_size + 1}–{min((b + 1) * band_size, TOP_N)}"
                    for b in range(BANDS)]))
share = pivot.div(pivot.sum(axis=1), axis=0)

fig = go.Figure()
for fam in top_fams + ["other"]:
    if fam not in share.columns:
        continue
    fig.add_trace(go.Bar(
        y=share.index, x=share[fam], name=fam, orientation="h",
        marker=dict(color=FAM_COLOR[fam], line=dict(color=SURFACE, width=2)),
        text=[f"{v:.0%}" if v >= 0.09 else "" for v in share[fam]],
        textposition="inside", insidetextanchor="middle",
        textfont=dict(color="#ffffff", size=11),
        customdata=pivot[fam],
        hovertemplate=("<b>%{fullData.name}</b><br>%{y}<br>"
                       "%{customdata} teams (%{x:.0%})<extra></extra>"),
    ))
fig.update_layout(
    barmode="stack",
    title="Family share by leaderboard rank band"
          "<br><sub>identified teams only · bands of "
          f"{band_size} · tail folded into <i>other</i></sub>",
    xaxis=dict(title="share of identified teams in the band", tickformat=".0%"),
    yaxis=dict(title=None, autorange="reversed", automargin=True),
    height=380, bargap=0.3, legend=dict(orientation="h", y=-0.18),
)
fig.show()

display(pivot.style.set_caption("Teams per family per rank band (counts)"))"""),

    md("""## 5. Energy, theme, and what tier of attacker

The three card attributes the deck identity is built from. Each is a magnitude
comparison, so each gets one hue and direct labels."""),

    code("""# 6 — energy / owner theme / attacker tier
from plotly.subplots import make_subplots

def _counts(col, fillna=None):
    s = known[col].fillna(fillna) if fillna else known[col].dropna()
    return s.value_counts().sort_values()

panels = [("primary_energy", "Energy type", "(no basic energy)"),
          ("theme", "Owner theme", "(no theme)"),
          ("attacker_tier", "Attacker tier", None)]

fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.13,
                    subplot_titles=[p[1] for p in panels])
for i, (col, _title, fill) in enumerate(panels, start=1):
    c = _counts(col, fill)
    fig.add_trace(go.Bar(
        x=c.values, y=c.index, orientation="h", showlegend=False,
        marker=dict(color=ACCENT, line=dict(color=SURFACE, width=2)),
        text=c.values, textposition="outside",
        textfont=dict(color=INK2, size=11),
        hovertemplate="<b>%{y}</b><br>%{x} teams<extra></extra>",
    ), row=1, col=i)
    fig.update_xaxes(range=[0, c.max() * 1.25], row=1, col=i)
    fig.update_yaxes(automargin=True, row=1, col=i)

fig.update_layout(
    title=f"Card attributes across {len(known)} identified decks",
    height=430, bargap=0.3, margin=dict(t=110),
)
fig.for_each_annotation(lambda a: a.update(font=dict(size=13, color=INK)))
fig.show()

display(pd.concat({p[1]: _counts(p[0], p[2]) for p in panels}, axis=0)
          .rename("teams").to_frame()
          .style.set_caption("Card attributes (table view)"))"""),

    md("""## 6. Top of the ladder vs the mix we actually play

**The actionable chart.** Left dot = share of the surveyed leaderboard; right dot
= share of our own live games (`data/m39_live_mix.json`). A long connector is a
family our gate beds weight very differently from how the top of the ladder is
built."""),

    code("""# 7 — ladder mix vs our played mix
MIX_PATH = _ROOT / "data/m39_live_mix.json"
mix = json.loads(MIX_PATH.read_text(encoding="utf-8"))
ours = {fam: d["share"] for fam, d in mix["families"].items()}
print(f"our mix: {mix['n_games']} games over subs {mix['source_subs']}")

ladder = (known["family"].value_counts() / len(known)).to_dict()
fams = sorted(set(ladder) | set(ours), key=lambda f: -ladder.get(f, 0))
rows = [{"family": f, "ladder": ladder.get(f, 0.0), "ours": ours.get(f, 0.0)}
        for f in fams]
cmp_df = pd.DataFrame(rows).iloc[::-1]

fig = go.Figure()
for _, r in cmp_df.iterrows():
    fig.add_trace(go.Scatter(
        x=[r["ladder"], r["ours"]], y=[r["family"], r["family"]],
        mode="lines", line=dict(color=GRID, width=3),
        showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(
    x=cmp_df["ladder"], y=cmp_df["family"], mode="markers",
    name=f"leaderboard top {TOP_N}",
    marker=dict(color=LADDER_C, size=13, line=dict(color=SURFACE, width=2)),
    hovertemplate="<b>%{y}</b><br>ladder %{x:.1%}<extra></extra>"))
fig.add_trace(go.Scatter(
    x=cmp_df["ours"], y=cmp_df["family"], mode="markers",
    name="our live games",
    marker=dict(color=OURS_C, size=13, symbol="diamond",
                line=dict(color=SURFACE, width=2)),
    hovertemplate="<b>%{y}</b><br>ours %{x:.1%}<extra></extra>"))
fig.update_layout(
    title="What the top plays vs what we face"
          "<br><sub>share of teams (leaderboard) vs share of games "
          "(m39_live_mix.json)</sub>",
    xaxis=dict(title="share", tickformat=".0%"),
    yaxis=dict(title=None, automargin=True),
    height=90 + 34 * len(cmp_df),
    legend=dict(orientation="h", y=-0.14),
)
fig.show()

out = cmp_df.iloc[::-1].copy()
out["gap (ladder - ours)"] = out["ladder"] - out["ours"]
display(out.sort_values("gap (ladder - ours)", key=abs, ascending=False)
           .style.hide(axis="index")
           .format({"ladder": "{:.1%}", "ours": "{:.1%}",
                    "gap (ladder - ours)": "{:+.1%}"})
           .set_caption("Where the two populations disagree most"))"""),

    md("""## 7. Does the decklist label match what they actually do?

A decklist says what a pilot *could* do. This walks the cached replays and counts
what each pilot actually **attacked** with (discounting pre-evolutions, which chip
early and then evolve). Disagreements are the interesting rows — a deck whose
listed centrepiece never does the attacking is a deck we would mis-model in a
gate bed."""),

    code("""# 8 — decklist heuristic vs what actually attacked
ev = census[census["evidence_agrees"].notna()].copy()
if ev.empty:
    print("no replay evidence — re-run the CLI without --no-evidence")
else:
    # nullable column arrives as object dtype; ~ would bitwise-not the ints
    ev["evidence_agrees"] = ev["evidence_agrees"].astype(bool)
    agree = int(ev["evidence_agrees"].sum())
    print(f"replay evidence: {agree}/{len(ev)} listed attackers confirmed "
          f"({agree / len(ev):.0%})")
    bad = (ev[~ev["evidence_agrees"]]
           .groupby(["attacker", "played_attacker"])
           .agg(teams=("rank", "size"), best_rank=("rank", "min"),
                family=("family", "first"))
           .sort_values("teams", ascending=False).reset_index())
    display(bad.rename(columns={"attacker": "listed as main attacker",
                                "played_attacker": "actually attacked with"})
               .style.hide(axis="index")
               .set_caption("Disagreements — the decklist and the replays "
                            "name different attackers"))"""),

    md("""### Reading the disagreements

Two kinds show up, and they mean opposite things:

- **The listed centrepiece never reaches the board.** *Crustle → Mega Kangaskhan
  ex* is the documented "Kanga-only wall blindspot" (`scripts/m39_live_mix.py`):
  wall lists whose Crustle sits in the deck while Kangaskhan does the work. Here
  the replays, not the decklist, are right.
- **A splashed line does the attacking in one build.** *Team Rocket's Spidops →
  Team Rocket's Mewtwo ex* is the reverse of the case that forced energy-matching
  into the attacker heuristic — in this particular list the Psychic tech really is
  the plan.

Both are reasons the census reports the two labels side by side instead of
collapsing them into one confident answer."""),
]

OUT.write_text(json.dumps({
    "cells": _with_ids(CELLS),
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.8"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}, indent=1), encoding="utf-8")
print(f"wrote {OUT} ({len(CELLS)} cells)")
