"""Deck Lab — browse the card pool, build a deck, smoke-test it.

    uv run --with streamlit streamlit run app/deck_lab.py

Built for M41: the deck question (switch to grim / engineer an anti-grim deck /
keep teaching ours) needs a surface for actually making decks, and the census
gave us a target to build against.

This file holds NO logic — everything computable lives in tcg/cardpool.py and
tcg/decklab.py, which are streamlit-free and tested. Two rules that keep it that
way:

* **Games run in a subprocess.** The cg engine keeps one global mutable Battle
  per process; driving a battle inside Streamlit's rerun model would corrupt it.
  `cg.api.all_card_data()` is exempt — a pure table read, no battle.
* **All state mutation happens in on_click / on_change callbacks**, never in
  `if st.button(): mutate`. Callbacks run before widgets render, which is what
  keeps the Deck tab's count inputs consistent when a card is added from Browse.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go        # noqa: E402
import plotly.io as pio                  # noqa: E402
import polars as pl                      # noqa: E402
import streamlit as st                   # noqa: E402

from tcg import cardpool as cp           # noqa: E402
from tcg import decklab as dl            # noqa: E402

st.set_page_config(page_title="Deck Lab", page_icon="🃏", layout="wide")

# House style, copied from notebooks/build_leaderboard_decks.py (validated with
# the dataviz palette validator against surface #fcfcfb).
INK, INK2, GRID, SURFACE = "#333333", "#666666", "#e5e5e5", "#fcfcfb"
PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834"]
ACCENT, MUTED = "#2a78d6", "#8a8a8a"

if "pokedex" not in pio.templates:
    pio.templates["pokedex"] = go.layout.Template(layout=dict(
        font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=13, color=INK),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left"),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
                   tickfont=dict(color=INK2)),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
                   tickfont=dict(color=INK2)),
        legend=dict(font=dict(color=INK2), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=30, t=50, b=40),
    ))
pio.templates.default = "pokedex"

MODE_NEW, MODE_LOAD = "✨ New deck", "📂 From existing"

# Opponents for the smoke test: one per live family, plus the shipped list.
SMOKE_OPPONENTS = [
    ("grim (Marnie's Grimmsnarl) — 49.5% of the top 250", "decks/grim_live.csv"),
    ("mirror (our Alakazam list)", "decks/alakazam_v2_h4.csv"),
    ("wall (Great Tusk / Crustle)", "decks/greattusk_wall.csv"),
    ("archaludon", "decks/archaludon.csv"),
    ("dragapult", "decks/dragapult.csv"),
    ("garchomp (Cynthia's)", "decks/cynthia_garchomp.csv"),
    ("lucario", "decks/lucario.csv"),
    ("iono", "decks/iono.csv"),
    ("stall (Hop's)", "decks/hops_stall.csv"),
    ("kyogre / abomasnow", "decks/kyogre.csv"),
]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("deck", {})
    # The NAME lives in the widget key. A keyed widget owns its value after the
    # first render, so `value=` is ignored from then on — Load must assign
    # `w_deck_name` itself (legal from a callback, which runs before the
    # widgets re-render) or the save destination silently keeps the old name.
    ss.setdefault("w_deck_name", "untitled")
    ss.setdefault("deck_source", None)
    ss.setdefault("saved_path", None)
    ss.setdefault("active_card", None)
    ss.setdefault("smoke_history", [])
    # Legality is reported only when the user asks to save — an empty deck is a
    # deck you have not finished, not a deck with five errors in it.
    ss.setdefault("save_errors", [])
    ss.setdefault("save_ok", None)
    ss.setdefault("overwrite_target", None)   # set when Save hits an existing file
    ss.setdefault("fill_note", None)


def _set_count(cid: int, n: int) -> None:
    """Write the deck AND the Deck tab's number_input for that card.

    A keyed widget owns its value after first render, so mutating `deck` alone
    leaves the input showing the old count — add a 14th copy from Browse and the
    Deck tab still reads 13, then pushes 13 back the next time it is touched.
    Assigning the widget key is legal here because callbacks run before widgets
    re-render.
    """
    ss = st.session_state
    if n > 0:
        ss.deck[cid] = n
    else:
        ss.deck.pop(cid, None)
    key = f"cnt_{cid}"
    if key in ss:
        ss[key] = n


def _forget_counts() -> None:
    """Drop every count widget — the next deck's cards are different cards."""
    for key in [k for k in st.session_state if k.startswith("cnt_")]:
        del st.session_state[key]


def _add(cid: int, n: int = 1) -> None:
    ss = st.session_state
    cap = 60 if cp.cards()[cid]["is_basic_energy"] else dl.MAX_COPIES
    _set_count(cid, min(cap, ss.deck.get(cid, 0) + n))
    ss.active_card = cid


def _remove(cid: int, n: int = 1) -> None:
    _set_count(cid, st.session_state.deck.get(cid, 0) - n)


def _fill_energy() -> None:
    """Top the deck up to 60 with basic energy of its dominant type."""
    ss = st.session_state
    filled, etype, n = dl.fill_with_energy(ss.deck)
    if not n:
        return
    cid = dl.basic_energy_ids()[etype]
    _set_count(cid, filled[cid])
    ss.fill_note = f"added {n}x Basic {etype} Energy"


def _sync_count(cid: int) -> None:
    n = int(st.session_state.get(f"cnt_{cid}", 0))
    if n > 0:
        st.session_state.deck[cid] = n
    else:
        st.session_state.deck.pop(cid, None)


def _reset_save_state() -> None:
    st.session_state.save_errors = []
    st.session_state.save_ok = None
    st.session_state.overwrite_target = None


def _load() -> None:
    ss = st.session_state
    label = ss.get("w_load_src")
    path = dict(dl.list_deck_files()).get(label)
    if path is None:
        return
    _forget_counts()
    ss.deck = dl.load_counts(path)
    ss.deck_source = label
    ss.saved_path = None
    stem = Path(label).stem.replace(".", "_")[:48] or "untitled"
    # A copy of decks/lucario.csv is not decks/lucario.csv — suggest a free
    # name so the first Save cannot land on something that already exists.
    ss.w_deck_name = dl.suggest_name(stem)
    ss.active_card = None
    _reset_save_state()


def _new_deck() -> None:
    """Start an empty deck. Nothing touches disk until Save is pressed."""
    ss = st.session_state
    _forget_counts()
    ss.deck = {}
    ss.deck_source = None
    ss.saved_path = None
    ss.active_card = None
    ss.w_deck_name = dl.suggest_name("untitled")
    _reset_save_state()


def _on_mode_change() -> None:
    """Switching to New starts a blank deck; switching to Load leaves the
    current one alone until the user picks a source and presses Load."""
    if st.session_state.get("w_mode") == MODE_NEW:
        _new_deck()
    else:
        _reset_save_state()


def _clear() -> None:
    _forget_counts()
    st.session_state.deck = {}
    st.session_state.deck_source = None
    st.session_state.saved_path = None
    _reset_save_state()


def _save(overwrite: bool = False) -> Path | None:
    """Validate, then write. This is the ONLY place legality is enforced —
    an unfinished deck should not be shouting errors while you build it."""
    ss = st.session_state
    name = (ss.get("w_deck_name") or "").strip()
    _reset_save_state()

    errors = []
    if not dl.NAME_RE.match(name):
        errors.append(f"invalid deck name {name!r}: letters, digits, '_' and "
                      "'-' only, starting alphanumeric, max 48 chars")
    ok, reasons = dl.legality(ss.deck)
    if not ok:
        errors.extend(reasons)
    if errors:
        ss.save_errors = errors
        return None

    try:
        path = dl.save_deck(name, ss.deck, overwrite=overwrite)
    except FileExistsError:
        ss.overwrite_target = name
        return None
    ss.saved_path = path
    ss.save_ok = path.relative_to(ROOT).as_posix()
    return path


def _save_overwrite() -> None:
    _save(overwrite=True)


_init_state()
ss = st.session_state
counts = ss.deck


# ---------------------------------------------------------------------------
# Sidebar — the deck slot
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🃏 Deck")
    st.segmented_control(
        "Mode", [MODE_NEW, MODE_LOAD], key="w_mode", default=MODE_NEW,
        on_change=_on_mode_change, width="stretch", label_visibility="collapsed",
        help="Saving always writes a NEW file under decks/custom/ — the curated "
             "decks in decks/ and the harvested lists in data/kaggle/ are never "
             "touched.")
    mode = ss.get("w_mode") or MODE_NEW

    if mode == MODE_LOAD:
        labels = [lbl for lbl, _ in dl.list_deck_files()]
        st.selectbox("Start from", labels, key="w_load_src",
                     index=labels.index("decks/lucario.csv")
                     if "decks/lucario.csv" in labels else 0)
        c1, c2 = st.columns(2)
        c1.button("📂 Load", on_click=_load, width="stretch")
        c2.button("Clear", on_click=_clear, width="stretch")
        if ss.deck_source:
            st.caption(f"editing a copy of `{ss.deck_source}`")
    else:
        st.caption("Building from scratch. Nothing is written until you Save.")

    st.text_input("Name", key="w_deck_name")
    try:
        dest = dl.custom_deck_path((ss.get("w_deck_name") or "").strip())
        rel = dest.relative_to(ROOT).as_posix()
        st.caption(("⚠️ saves to `{}` — **already exists**" if dest.exists()
                    else "saves to `{}`").format(rel))
    except ValueError:
        st.caption("saves to `decks/custom/<name>.csv`")

    st.divider()

    # Progress, not judgement. Legality is only asserted on Save.
    n = dl.deck_size(counts)
    st.progress(min(n / dl.DECK_SIZE, 1.0), text=f"{n} / {dl.DECK_SIZE} cards")
    if counts:
        s = dl.summarize(counts)
        st.caption(
            f"🃏 {s.kind_counts.get('Pokemon', 0)} Pokemon · "
            f"🎒 {s.kind_counts.get('Trainer', 0)} Trainer · "
            f"🔋 {s.kind_counts.get('Energy', 0)} Energy")
        types = [t for t, _ in sorted(s.pokemon_energy.items(),
                                      key=lambda kv: -kv[1])[:4]]
        if types:
            st.caption(" ".join(f"{cp.energy_emoji(t)} {t}" for t in types))

    short = dl.DECK_SIZE - n
    fill_type = dl.dominant_energy(counts) if short > 0 else None
    if fill_type:
        st.button(f"🔋 Fill {short} slot{'s' if short != 1 else ''} with "
                  f"{cp.energy_emoji(fill_type)} {fill_type}",
                  on_click=_fill_energy, width="stretch",
                  help="Basic energy of the type this deck's attacks most "
                       "demand. Basic energy is exempt from the 4-copy rule.")
    elif short > 0 and counts:
        st.caption(f"{short} slots free — no basic energy matches this deck's "
                   "attack costs yet")
    if ss.fill_note:
        st.caption(f"✅ {ss.fill_note}")
        ss.fill_note = None

    st.button("💾 Save", on_click=_save, width="stretch", type="primary")
    for msg in ss.save_errors:
        st.error(msg)
    if ss.overwrite_target:
        st.error(f"`decks/custom/{ss.overwrite_target}.csv` already exists — "
                 "rename above, or overwrite it deliberately.")
        st.button(f"⚠️ Overwrite {ss.overwrite_target}.csv",
                  on_click=_save_overwrite, width="stretch")
    if ss.save_ok:
        st.success(f"wrote {ss.save_ok}")

    if counts:
        st.caption(f"deck_hash `{dl.deck_hash(dl.deck_ids(counts))[:12]}`")
    if ss.deck_source:
        st.caption(f"from {ss.deck_source}")


tab_browse, tab_deck, tab_stats, tab_test, tab_meta = st.tabs(
    ["🔍 Browse", "📋 Deck", "📊 Stats", "⚔️ Test", "🏆 Meta"])


# ---------------------------------------------------------------------------
# Browse
# ---------------------------------------------------------------------------
with tab_browse:
    a, b, c = st.columns(3)
    with a:
        types = st.multiselect("Card type", list(cp.CARD_TYPES),
                               format_func=lambda i: cp.CARD_TYPES[i])
        energies = st.multiselect("Energy type", list(cp.ENERGY_TYPES),
                                  format_func=lambda i: cp.ENERGY_TYPES[i],
                                  help="Pokemon and energy cards only — every "
                                       "Trainer carries Colorless internally")
    with b:
        owner = st.selectbox("Owner theme", ["any", *cp.owners()])
        stages = st.multiselect("Stage", list(cp.STAGES))
    with c:
        tiers = st.multiselect("Tier", list(cp.TIERS),
                               help="'ex' and 'Mega ex' are disjoint in this pool")
        text = st.text_input("Name contains", "")

    with st.expander("More filters"):
        d, e, f = st.columns(3)
        hp = d.slider("HP", 0, 400, (0, 400), step=10)
        retreat = d.slider("Retreat cost", 0, 4, (0, 4))
        cost = e.slider("Cheapest attack cost", 0, 5, (0, 5))
        dmg = e.slider("Base damage", 0, 350, (0, 350), step=10)
        keep_var = f.checkbox("Keep scaling attackers", True,
                             help="Cards like Alakazam (10) and Team Rocket's "
                                  "Spidops (0) record BASE damage; a damage "
                                  "floor would hide two real ladder decks")
        ace_only = f.checkbox("ACE SPEC only", False)

    filt = cp.CardFilter(
        text=text, card_types=tuple(types), energy_types=tuple(energies),
        owner=None if owner == "any" else owner, hp=tuple(hp),
        stages=tuple(stages), tiers=tuple(tiers), ace_spec_only=ace_only,
        retreat=tuple(retreat), attack_cost=tuple(cost), damage=tuple(dmg),
        keep_variable=keep_var)
    found = cp.filter_cards(filt)

    st.caption(f"{found.height} of {cp.cards_frame().height} cards · "
               "click a row to inspect it")

    show = found.select(
        "icon", "name", "badge", "type_name", "type_label", "stage",
        "hp", "atk_max_damage", "atk_min_cost", "retreat_cost", "weakness",
        "owner", "card_id")
    event = st.dataframe(
        show, height=330, width="stretch", hide_index=True,
        on_select="rerun", selection_mode="single-row", key="w_card_table",
        column_config={
            "icon": st.column_config.TextColumn("", width="small"),
            "name": st.column_config.TextColumn("Card", width="medium"),
            "badge": st.column_config.TextColumn("", width="small",
                                                 help="🌟 Mega ex · ✨ ex · "
                                                      "💎 Tera · 🅰 ACE SPEC"),
            "type_name": st.column_config.TextColumn("Type", width="small"),
            "type_label": st.column_config.TextColumn("Kind", width="small"),
            "stage": st.column_config.TextColumn("Stage", width="small"),
            "hp": st.column_config.ProgressColumn(
                "HP", min_value=0, max_value=400, format="%d"),
            "atk_max_damage": st.column_config.ProgressColumn(
                "Damage", min_value=0, max_value=350, format="%d",
                help="PRINTED base damage — scaling attacks record their base, "
                     "so Alakazam reads 0 and really does 20 per card in hand"),
            "atk_min_cost": st.column_config.NumberColumn("Cost", width="small"),
            "retreat_cost": st.column_config.NumberColumn("Retreat", width="small"),
            "weakness": st.column_config.TextColumn("Weak to", width="small"),
            "owner": st.column_config.TextColumn("Theme", width="small"),
            "card_id": st.column_config.NumberColumn("#", width="small"),
        })

    rows = event.selection.rows if event and event.selection else []
    if rows and rows[0] < found.height:
        ss.active_card = int(found["card_id"][rows[0]])
    pick = ss.active_card if ss.active_card in cp.cards() else None

    if pick is None:
        st.info("Select a row above to see the card's attacks and abilities.")
    else:
        r = cp.cards()[pick]
        colour = cp.energy_color(r["type_name"])
        badges = []
        if r["tier"] and r["tier"] != "regular":
            badges.append(r["tier"])
        if r["is_ace_spec"]:
            badges.append("ACE SPEC")
        st.markdown(
            f'<div style="border-left:6px solid {colour};background:#fbfbfa;'
            f'border-radius:6px;padding:10px 14px;margin:6px 0 2px">'
            f'<span style="font-size:20px;font-weight:650;color:#333">'
            f'{cp.card_icon(r)} {r["name_norm"]}</span>'
            + (f'<span style="color:{colour};font-weight:600;font-size:13px">'
               f'  {" · ".join(badges)}</span>' if badges else "")
            + f'<div style="color:#666;font-size:12px;margin-top:3px">'
              f'#{pick} · {cp.CARD_TYPES.get(r["card_type"], "?")}'
            + (f' · {r["type_name"]} · {r["hp"]} HP · {r["stage"]} · '
               f'retreat {r["retreat_cost"]} · weak to '
               f'{cp.ENERGY_TYPES.get(r["weakness_id"], "—")} · '
               f'{r["prizes_on_ko"]} prize(s)' if r["is_pokemon"] else "")
            + '</div></div>', unsafe_allow_html=True)

        g1, g2, g3 = st.columns([1, 1, 6])
        g1.button("＋1", on_click=_add, args=(pick, 1), width="stretch")
        g2.button("＋4", on_click=_add, args=(pick, 4), width="stretch")
        g3.caption(f"in deck: {counts.get(pick, 0)}")

        for name, txt in cp.card_abilities().get(pick, []):
            st.markdown(f"**🔮 {name}** — {txt}")
        texts = cp.attack_texts()
        for atk in cp.attacks_by_card().get(pick, []):
            t = texts.get(atk["attackId"], {})
            cost_s = "".join(
                cp.ENERGY_EMOJI.get(cp.ENERGY_TYPES[tid], "?") * int(atk.get(col) or 0)
                for col, tid in cp.COST_COLS.items()) or "—"
            dmg = f"**{atk['damage']}**"
            if atk["is_variable"]:
                dmg = f"**{atk['damage']}+** _(scales)_"
            st.markdown(f"⚔️ {cost_s} **{t.get('name') or 'attack'}** — {dmg}")
            if t.get("text"):
                st.caption(t["text"])
        if not cp.engine_available():
            st.info("card text unavailable (engine not importable) — "
                    "the stats above come from the parquets")


# ---------------------------------------------------------------------------
# Deck
# ---------------------------------------------------------------------------
with tab_deck:
    if not counts:
        st.info("Empty. Press **✨ New deck** to start from scratch, or load one "
                "from the sidebar, then add cards in Browse.")
    else:
        ft = cp.cards()
        by_name = dl.copies_by_name(counts)
        pre = cp.pre_evo_names()

        notes = dl.deck_notes(counts)
        if notes:
            with st.expander(f"⚠️ {len(notes)} note(s) on this deck",
                             expanded=False):
                st.caption("Advisory — these do not block saving. Legality is "
                           "checked when you press Save.")
                for note in notes:
                    {"error": st.error, "warn": st.warning}.get(
                        note.level, st.info)(note.message)

        for kind in ("Pokemon", "Trainer", "Energy"):
            group = sorted((cid for cid in counts if ft[cid]["kind"] == kind),
                           key=lambda c: (-counts[c], ft[c]["name_norm"]))
            if not group:
                continue
            total = sum(counts[c] for c in group)
            icon = {"Pokemon": "🃏", "Trainer": "🎒", "Energy": "🔋"}[kind]
            st.subheader(f"{icon} {kind} · {total}")
            for cid in group:
                r = ft[cid]
                c1, c2, c3 = st.columns([5, 2, 1])
                dup = len(cp.printings_by_name()[r["name_norm"]]) > 1
                badge = cp.TIER_EMOJI.get(r["tier"] or "", "")
                label = (f"{cp.card_icon(r)} {r['name_norm']} {badge}"
                         + (f"  `#{cid}`" if dup else ""))
                c1.markdown(label)
                # The key IS the value: seed it once, then `_set_count` keeps it
                # in step. Passing `value=` as well makes Streamlit warn that it
                # will be ignored, which it would be.
                ss.setdefault(f"cnt_{cid}", counts[cid])
                c2.number_input(
                    "copies", 0, 60 if r["is_basic_energy"] else dl.MAX_COPIES,
                    key=f"cnt_{cid}", label_visibility="collapsed",
                    on_change=_sync_count, args=(cid,))
                c3.button("✕", key=f"del_{cid}", on_click=_remove, args=(cid, 99))
                if dup:
                    c1.caption(f"{by_name[r['name_norm']]}x {r['name_norm']} "
                               "across printings (the 4-copy limit is per name)")

        st.divider()
        st.subheader("Evolution lines")
        tops = [n for n in by_name if n in pre and
                not any(pre.get(o) == n for o in by_name)]
        if not tops:
            st.caption("no evolution lines in this deck")
        for top in sorted(tops):
            chain, cur = [], top
            while cur:
                chain.append(cur)
                cur = pre.get(cur)
            parts = []
            for name in reversed(chain):
                have = by_name.get(name, 0)
                exempt = name in {ft[i]["name_norm"] for i in cp.self_starting_ids()
                                  if i in ft}
                mark = "✓" if have or exempt else "✗"
                parts.append(f"{name}({have}){mark}")
            st.write(" → ".join(parts))


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def _bar(mapping: dict, title: str, color=ACCENT) -> go.Figure:
    items = sorted(mapping.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(
        x=[v for _, v in items], y=[str(k) for k, _ in items], orientation="h",
        marker=dict(color=color, line=dict(color=SURFACE, width=2)),
        text=[v for _, v in items], textposition="outside",
        textfont=dict(color=INK2, size=11),
        hovertemplate="<b>%{y}</b>: %{x}<extra></extra>"))
    top = max([v for _, v in items], default=1)
    fig.update_layout(title=title, height=60 + 34 * max(len(items), 1),
                      xaxis=dict(range=[0, top * 1.25]),
                      yaxis=dict(automargin=True), bargap=0.3, showlegend=False)
    return fig


with tab_stats:
    if not counts:
        st.info("Nothing to summarise yet.")
    else:
        s = dl.summarize(counts)
        m = st.columns(4)
        m[0].metric("Cards", s.n_cards)
        m[1].metric("Pokemon / Trainer / Energy",
                    f"{s.kind_counts.get('Pokemon', 0)}/"
                    f"{s.kind_counts.get('Trainer', 0)}/"
                    f"{s.kind_counts.get('Energy', 0)}")
        m[2].metric("Prizes on board", s.prize_liability,
                    help="total prizes the opponent takes for KOing every "
                         "Pokemon in the list — ex and Mega ex give 2 and 3")
        m[3].metric("ACE SPEC", s.ace_spec or "—")

        left, right = st.columns(2)
        with left:
            st.plotly_chart(_bar(s.kind_counts, "Card kinds"),
                            width="stretch")
            st.plotly_chart(_bar(s.stage_counts, "Pokemon stages"),
                            width="stretch")
            st.plotly_chart(_bar({str(k): v for k, v in s.retreat_curve.items()},
                                 "Retreat cost"), width="stretch")
        with right:
            # The insight chart: what the attacks ask for vs what the deck packs.
            types = sorted(set(s.energy_demand) | set(s.energy_cards)
                           - {"Special"})
            if types:
                fig = go.Figure()
                fig.add_bar(x=types, y=[s.energy_demand.get(t, 0) for t in types],
                            name="symbols demanded", marker_color=PALETTE[0])
                fig.add_bar(x=types, y=[s.energy_cards.get(t, 0) for t in types],
                            name="energy cards", marker_color=PALETTE[3])
                fig.update_layout(title="Energy: demanded vs supplied",
                                  barmode="group", height=320,
                                  legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig, width="stretch")
                if s.energy_cards.get("Special"):
                    st.caption(f"plus {s.energy_cards['Special']} special energy, "
                               "which can pay costs of any type")
            st.plotly_chart(
                _bar({str(k): v for k, v in s.attack_cost_curve.items()},
                     "Attack cost curve"), width="stretch")

        st.subheader("Top attackers")
        st.dataframe(pl.DataFrame(
            [{"card": n, "damage": d, "cost": c} for n, d, c in s.top_attackers]),
            hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
with tab_test:
    st.caption(
        "Both sides are piloted by the **deck-agnostic** rule pilot "
        "(`rl/generic_pilot.py`), so the win rate reflects the DECK, not a "
        "brain tuned to particular cards. Games run in a subprocess — the "
        "engine keeps one global battle per process.")

    o1, o2, o3 = st.columns([4, 2, 2])
    opp_label = o1.selectbox("Opponent", [lbl for lbl, _ in SMOKE_OPPONENTS])
    games = o2.slider("Games", 10, 200, 40, step=10)
    kind = o3.selectbox("Pilot", ["generic", "solver"],
                        help="solver adds within-turn combo search — stronger, "
                             "slower, same for both sides")

    if st.button("▶ Save & smoke-test", type="primary", disabled=not counts):
        saved = _save()
        if not saved:
            for msg in ss.save_errors:
                st.error(msg)
        else:
            opp = dict(SMOKE_OPPONENTS)[opp_label]
            a = dl.deck_spec(saved, kind)
            b = dl.deck_spec(ROOT / opp, kind)
            with st.spinner(f"{games} games vs {opp_label}…"):
                got = dl.run_smoke(a, b, games=games)
            if got["ok"]:
                r = got["result"]
                st.success(f"**{r['wr']:.1%}** — {r['w']}W {r['l']}L {r['d']}D "
                           f"over {r['n']}")
                ss.smoke_history.insert(0, {
                    "deck": ss.w_deck_name, "opponent": opp_label,
                    "pilot": kind, "n": r["n"], "wr": r["wr"],
                    "W": r["w"], "L": r["l"], "D": r["d"]})
                st.caption(
                    f"n={r['n']} resolves roughly ±{1.96 * (0.25 / r['n']) ** 0.5:.0%} "
                    "— a smoke test, not a measurement. Use scripts/deck_probe.py "
                    "for anything you intend to act on.")
            else:
                st.error(got.get("error", "run failed"))
                with st.expander("stderr"):
                    st.code(got.get("stderr", "") or "(empty)")
            with st.expander("command"):
                st.code(" ".join(got["cmd"]))

    if ss.smoke_history:
        st.subheader("This session")
        st.dataframe(pl.DataFrame(ss.smoke_history), hide_index=True,
                     width="stretch")


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------
with tab_meta:
    census = ROOT / "data/kaggle/leaderboard_decks.parquet"
    if not census.exists():
        st.info("No leaderboard census on disk. Build it with:\n\n"
                "`uv run python scripts/leaderboard_decks.py --top 250`")
    else:
        df = pl.read_parquet(census)
        known = df.filter(pl.col("label").is_not_null())
        st.caption(f"{known.height} of {df.height} leaderboard teams identified")

        dist = (known.group_by("label")
                     .agg(pl.len().alias("teams"),
                          pl.mean("score").alias("mean_score"),
                          pl.first("primary_energy"), pl.first("attacker_tier"))
                     .sort("teams", descending=True))
        st.dataframe(dist, hide_index=True, width="stretch", height=330)

        if counts:
            h = dl.deck_hash(dl.deck_ids(counts))
            hit = df.filter(pl.col("deck_hash") == h)
            if hit.height:
                st.success(f"This exact list is played by {hit.height} team(s) "
                           f"in the top {df.height} — best rank #{hit['rank'].min()}")
            else:
                st.caption("This exact list does not appear in the surveyed "
                           "leaderboard.")
