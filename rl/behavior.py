"""M22b: behavioral counters that price the alternative line.

THE RULE THIS MODULE EXISTS TO ENFORCE
--------------------------------------
Every counter's denominator is the set of states in which the alternative line
is **strictly worse**. A denominator of "the action was legal" measures option
availability, not policy defect.

M21 shipped a whole leg against a gust "defect" of 0/27, then 0/217. The
post-hoc audit found 138 of 142 gust targets were 1-prize benchwarmers and in 78
of them the opponent's active was ALSO KO-able that turn — worth more 38x, equal
38x, and gust better only 2x. The agent was usually right to decline. A second,
independent defect compounded it: the prototypes tested KO-ability with
``_charged_best``, which the docstring at rl/combat.py:60 says explicitly
"assumes FULL charge — this skips affordability". So the denominator also
counted KOs we could not pay for, and the retreat denominator called
zero-energy bench Pokemon "strictly better attackers".

Both are fixed here: ``_best_damage`` everywhere (affordability-gated), and
tiered verdicts that never collapse into a bare conversion rate.

WHY THIS DOES NOT REUSE ``rl.plan.enumerate_plans``
---------------------------------------------------
It computes the same prize arithmetic and is tempting. But it is the AGENT's
candidate generator: it caps at ``MAX_PLAN_CANDS = 48`` with a hard return
(rl/plan.py:163), and its target list puts the active first and bench targets
after — so truncation preferentially drops exactly the gust lines this module
must be able to see. **An instrument must not inherit its subject's blind
spots.** Pure game math (``_best_damage``, ``_CARD``) is reused; candidate
generation is deliberately independent and uncapped.

CLI
    uv run python -m rl.behavior replays --sub 54849475 [--limit N]
    uv run python -m rl.behavior series --a model:checkpoints/X.pt:lucario \\
        --b solver:lucario -n 200 --seed 1
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from cg.api import AreaType, OptionType, SelectContext, all_card_data, to_observation_class

from rl.combat import _best_damage, _CARD
from rl.kaggle_ingest import EPISODES_PQ, RAW_DIR
from rl.plan import GUST_IDS

# Retreat cost lives ONLY here: `retreatCost` is declared at cg/api.py:468 but had
# zero readers repo-wide, and rl/combat.py is copied into the Kaggle bundle by
# build_submission.sh:44,52 — so extending _CARD there would ship a measurement
# concern AND break 29 positional `(None, None, 0, [], 1)` fallbacks.
_RETREAT: dict[int, int] = {}


def _retreat_cost(card_id) -> int:
    if not _RETREAT:
        for c in all_card_data():
            _RETREAT[c.cardId] = int(getattr(c, "retreatCost", 0) or 0)
    return _RETREAT.get(card_id, 0)


def prizes(card_id) -> int:
    """Prizes the opponent takes for this KO: 3 mega-ex / 2 ex / 1 (rl/combat.py:17)."""
    return _CARD.get(card_id, (None, None, 0, [], 1))[4]


@dataclass(frozen=True)
class Verdict:
    kind: str          # "gust" | "retreat"
    tier: str          # "strict" | "tempo" | "neutral" | "undecidable"
    gain: float        # counterfactual prize gain of the line, prize-capped
    taken: bool
    detail: dict = field(default_factory=dict)


def _board(player) -> list[tuple[int, object]]:
    """[(slot, pokemon)] — 0 = active, 1..5 = bench+1; None holes skipped."""
    out = []
    act = player.active[0] if getattr(player, "active", None) and player.active[0] is not None else None
    if act is not None:
        out.append((0, act))
    for i, p in enumerate(getattr(player, "bench", None) or []):
        if p is not None:
            out.append((i + 1, p))
    return out


def _option_card_id(opt, obs):
    """Card id an option refers to, or None.

    Real PLAY options carry no `area` and default to HAND — the trap that makes a
    naive "did we play Boss's Orders" counter silently read zero
    (rl/postmortem.py:318 documents it).
    """
    cid = getattr(opt, "cardId", None)
    if cid is not None:
        return cid
    idx = getattr(opt, "index", None)
    if idx is None:
        return None
    st = obs.current
    pi = getattr(opt, "playerIndex", None)
    player = st.players[pi if pi is not None else st.yourIndex]
    area = getattr(opt, "area", None)
    zone = ({int(AreaType.HAND): getattr(player, "hand", None),
             int(AreaType.DISCARD): getattr(player, "discard", None),
             int(AreaType.ACTIVE): getattr(player, "active", None),
             int(AreaType.BENCH): getattr(player, "bench", None),
             }.get(int(area)) if area is not None else getattr(player, "hand", None))
    try:
        card = zone[idx]
    except (TypeError, IndexError, KeyError):
        return None
    return getattr(card, "id", None)


def _val(card_id, prizes_left: int) -> int:
    """Prize value of KOing this card, CAPPED by prizes we still need.

    The cap is what makes "gust to close the game" score correctly: a 3-prize
    Mega is worth only 1 when 1 prize wins.
    """
    return min(prizes(card_id), max(prizes_left, 0))


def judge_gust(obs, chosen) -> Verdict | None:
    """The ONLY definition of a gust opportunity. None when not a candidate state."""
    st = obs.current
    me, op = st.players[st.yourIndex], st.players[1 - st.yourIndex]
    if getattr(st, "supporterPlayed", False):
        return None
    boss_opt = next((o for o in obs.select.option
                     if _option_card_id(o, obs) in GUST_IDS), None)
    if boss_opt is None:
        return None                       # cannot gust — not an opportunity
    active = me.active[0] if me.active and me.active[0] is not None else None
    op_active = op.active[0] if op.active and op.active[0] is not None else None
    if active is None or op_active is None:
        return Verdict("gust", "undecidable", 0.0, False, {"why": "no active"})

    my_ids = {p.id for _, p in _board(me)}
    left = len(getattr(me, "prize", []) or [])
    taken = _option_card_id(chosen, obs) in GUST_IDS

    # Best AFFORDABLE bench KO — every bench slot, not the first (the prototypes
    # broke after slot 0, letting bench ordering decide what got measured).
    gust_best, tgt = 0, None
    for p in (op.bench or []):
        if p is None:
            continue
        if _best_damage(active, p, board_ids=my_ids) >= (p.hp or 0):
            v = _val(p.id, left)
            if v > gust_best:
                gust_best, tgt = v, p
    stay_best = (_val(op_active.id, left)
                 if _best_damage(active, op_active, board_ids=my_ids) >= (op_active.hp or 0)
                 else 0)
    gain = gust_best - stay_best
    detail = {"gust_best": gust_best, "stay_best": stay_best,
              "target_id": getattr(tgt, "id", None), "prizes_left": left}

    if gust_best == 0:
        return Verdict("gust", "neutral", 0.0, taken, detail | {"why": "no affordable bench KO"})
    if gain > 0:
        return Verdict("gust", "strict", float(gain), taken, detail)
    if stay_best == 0 and tgt is not None and (
            len(getattr(tgt, "energies", ()) or []) >= 1 or prizes(tgt.id) >= 2):
        # Active not KO-able and the bench target is a developing threat:
        # defensible tempo, but a judgement call — never merged into `strict`.
        return Verdict("gust", "tempo", 0.0, taken, detail)
    return Verdict("gust", "neutral", float(gain), taken, detail)


def judge_retreat(obs, chosen) -> Verdict | None:
    """The ONLY definition of a retreat opportunity.

    Prices what the old metric ignored: the energy discarded to retreat, the
    forfeited attack, and whether the active actually DIES (not merely that it
    is scuffed — the old `hp/maxHp <= 0.5` test measured cosmetics).
    """
    st = obs.current
    me, op = st.players[st.yourIndex], st.players[1 - st.yourIndex]
    if not any(o.type == OptionType.RETREAT for o in obs.select.option):
        return None
    active = me.active[0] if me.active and me.active[0] is not None else None
    op_active = op.active[0] if op.active and op.active[0] is not None else None
    if active is None or op_active is None:
        return Verdict("retreat", "undecidable", 0.0, False, {"why": "no active"})

    my_ids = {p.id for _, p in _board(me)}
    op_ids = {p.id for _, p in _board(op)}
    left, opp_left = (len(getattr(me, "prize", []) or []),
                      len(getattr(op, "prize", []) or []))
    taken = getattr(chosen, "type", None) == OptionType.RETREAT

    dies = _best_damage(op_active, active, board_ids=op_ids) >= (active.hp or 0)
    stay_gain = (_val(op_active.id, left)
                 if _best_damage(active, op_active, board_ids=my_ids) >= (op_active.hp or 0)
                 else 0)
    stay_cost = _val(active.id, opp_left) if dies else 0

    cost = _retreat_cost(active.id)
    best, promoted = None, None
    for slot, b in _board(me):
        if slot == 0:
            continue
        # The promoted Pokemon attacks with ITS OWN energy; the discard is paid
        # from the RETREATING active, so b's affordability is unchanged. What we
        # must not do is credit b with energy it never had.
        pg = (_val(op_active.id, left)
              if _best_damage(b, op_active, board_ids=my_ids) >= (op_active.hp or 0)
              else 0)
        rc = (_val(b.id, opp_left)
              if _best_damage(op_active, b, board_ids=op_ids) >= (b.hp or 0) else 0)
        v = pg - rc
        if best is None or v > best:
            best, promoted = v, b
    if best is None:
        return Verdict("retreat", "neutral", 0.0, taken, {"why": "no bench to promote"})
    if cost > len(getattr(active, "energies", ()) or []):
        return Verdict("retreat", "neutral", 0.0, taken,
                       {"why": "cannot pay retreat cost", "retreat_cost": cost})

    gain = best - (stay_gain - stay_cost)
    detail = {"retreat_best": best, "stay_gain": stay_gain, "stay_cost": stay_cost,
              "retreat_cost": cost, "dies": dies,
              "promote_id": getattr(promoted, "id", None)}
    if gain > 0:
        return Verdict("retreat", "strict", float(gain), taken, detail)
    if gain == 0 and dies and stay_gain == 0:
        return Verdict("retreat", "tempo", 0.0, taken, detail)
    return Verdict("retreat", "neutral", float(gain), taken, detail)


def judge_state(obs, chosen) -> list[Verdict]:
    """Every verdict for one MAIN decision. The single definition site."""
    if obs.select is None or int(obs.select.context) != int(SelectContext.MAIN):
        return []
    return [v for v in (judge_gust(obs, chosen), judge_retreat(obs, chosen)) if v]


# --------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------
def _accumulate(acc: Counter, verdicts: list[Verdict]) -> None:
    for v in verdicts:
        acc[f"{v.kind}_{v.tier}"] += 1
        if v.taken:
            acc[f"{v.kind}_{v.tier}_taken"] += 1


def from_replays(sub_id: int, limit: int | None = None) -> Counter:
    """Counters over cached Kaggle replays for one submission."""
    import polars as pl

    from rl.replay_bc import iter_replay_decisions

    df = pl.read_parquet(EPISODES_PQ).filter(
        (pl.col("submission_id_0") == sub_id) | (pl.col("submission_id_1") == sub_id))
    acc, drops = Counter(), Counter()
    for n, row in enumerate(df.iter_rows(named=True)):
        if limit and n >= limit:
            break
        path = RAW_DIR / f"episode_{row['episode_id']}.json.gz"   # underscore, not hyphen
        if not path.exists():
            drops["missing_raw"] += 1
            continue
        raw = json.loads(gzip.decompress(path.read_bytes()))
        steps = raw.get("steps") or raw
        seat = row["our_seat"]
        acc["games"] += 1
        # iter_replay_decisions owns BOTH replay traps: the M8.0 off-by-one
        # (the action answering step i is at steps[i+1]) and the stale `select`
        # on the INACTIVE seat (only status=="ACTIVE" steps are decisions).
        for _, obs_dict, action in iter_replay_decisions(steps, seat, drops):
            obs = to_observation_class(obs_dict)
            if obs.select is None or not obs.select.option:
                continue
            try:
                chosen = obs.select.option[action[0]]
            except (IndexError, TypeError):
                drops["bad_label"] += 1
                continue
            acc["main_states"] += int(int(obs.select.context) == int(SelectContext.MAIN))
            _accumulate(acc, judge_state(obs, chosen))
    acc.update({f"drop_{k}": v for k, v in drops.items()})
    return acc


def from_series(a: str, b: str, n: int = 100, seed: int = 1,
                workers: int = 8) -> Counter:
    """Counters over a LOCAL match series — same judge_state as the replay path.

    Instruments via a custom `game_fn` (the seam at rl/matchrunner.py:545) rather
    than `on_game`, which is per-GAME only. The engine loop hands the pilot one
    obs per decision, so unlike replays there is no off-by-one and no stale
    INACTIVE select to filter.
    """
    from rl.matchrunner import make_pilot, parse_spec, play_series

    acc = Counter()

    def game_fn(fn0, fn1, deck0, deck1, stats):
        from cg.game import battle_finish, battle_select, battle_start
        obs_dict, start = battle_start(list(deck0), list(deck1))
        if start.errorPlayer >= 0:
            battle_finish()
            return 2
        try:
            steps = 0
            while obs_dict["current"]["result"] < 0 and steps < 600:
                seat = obs_dict["current"]["yourIndex"]
                fn = fn0 if seat == 0 else fn1
                picks = [int(i) for i in fn(obs_dict)]
                if seat == 0:                       # instrument side A only
                    obs = to_observation_class(obs_dict)
                    if obs.select is not None and obs.select.option:
                        try:
                            _accumulate(acc, judge_state(obs, obs.select.option[picks[0]]))
                        except IndexError:
                            pass
                obs_dict = battle_select(picks)
                steps += 1
            return int(obs_dict["current"]["result"])
        finally:
            battle_finish()

    play_series(parse_spec(a), parse_spec(b), n, seed=seed, game_fn=game_fn)
    acc["games"] = n
    return acc


def _fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Exact test on the 2x2 [[a,b],[c,d]]. No scipy in this env."""
    from math import comb
    n, r1, r2, c1 = a + b + c + d, a + b, c + d, a + c
    if not n or not r1 or not r2:
        return 1.0

    def p(x):
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)

    p0 = p(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 + 1e-12))


def flag_contrast(sub_id: int, limit: int | None = None) -> dict:
    """Per-flag WIN-vs-LOSS enrichment for one submission.

    `postmortem --batch` aggregates flags over losses only, which cannot
    distinguish a defect from a habit: a flag appearing at the same rate in wins
    carries zero information about why we lost. This computes the contrast, so
    the denominator is "games", not "games we lost".

    Same discipline as the gust metric: nothing is called a defect unless it is
    enriched in losses at a significance that survives correcting for the number
    of flag kinds tested.

    WHAT THIS CAN AND CANNOT SHOW. Flags are counted over a whole game, so they
    are confounded with game state: a game you are winning has more turns, more
    energy and a wider board, which moves flag rates for reasons that have
    nothing to do with causation. So a flag that is MORE common in wins is not
    thereby "good play" — it may simply be what winning boards look like.

    This is a FILTER, not a cause-finder. It can refute "flag X explains our
    losses"; it cannot establish that X helps or hurts. Use it to kill phantom
    defects before they motivate a leg, then investigate survivors properly.
    """
    import polars as pl

    from rl.postmortem import _cur, audit_flags

    df = pl.read_parquet(EPISODES_PQ).filter(
        (pl.col("submission_id_0") == sub_id) | (pl.col("submission_id_1") == sub_id))
    win_flags: Counter = Counter()      # kind -> games won that carried it
    loss_flags: Counter = Counter()
    n_win = n_loss = 0
    for n, row in enumerate(df.iter_rows(named=True)):
        if limit and n >= limit:
            break
        path = RAW_DIR / f"episode_{row['episode_id']}.json.gz"
        if not path.exists():
            continue
        seat = row["our_seat"]
        mine, theirs = row[f"reward_{seat}"], row[f"reward_{1 - seat}"]
        if mine is None or theirs is None or mine == theirs:
            continue                     # draws carry no win/loss contrast
        raw = json.loads(gzip.decompress(path.read_bytes()))
        steps = raw.get("steps") or raw
        if not any(_cur(s[seat]) for s in steps if isinstance(s[seat], dict)):
            continue
        won = mine > theirs
        kinds = {f.split("]", 1)[0].lstrip("[") for f in audit_flags(steps, seat)}
        if won:
            n_win += 1
            for k in kinds:
                win_flags[k] += 1
        else:
            n_loss += 1
            for k in kinds:
                loss_flags[k] += 1

    kinds = sorted(set(win_flags) | set(loss_flags))
    rows = []
    for k in kinds:
        w, l = win_flags[k], loss_flags[k]
        rows.append({
            "kind": k,
            "win_rate": w / n_win if n_win else 0.0,
            "loss_rate": l / n_loss if n_loss else 0.0,
            "w": w, "l": l,
            "p": _fisher_two_sided(l, n_loss - l, w, n_win - w),
        })
    rows.sort(key=lambda r: r["loss_rate"] - r["win_rate"], reverse=True)
    return {"n_win": n_win, "n_loss": n_loss, "n_kinds": len(kinds), "rows": rows}


def report_flags(res: dict) -> str:
    """Enrichment table. Bonferroni-corrected, because every flag kind is a test."""
    n_win, n_loss, k = res["n_win"], res["n_loss"], max(res["n_kinds"], 1)
    out = [f"flag contrast — {n_win} wins vs {n_loss} losses "
           f"({k} kinds tested, Bonferroni alpha = {0.05 / k:.4f})",
           f"{'flag':<24}{'in wins':>9}{'in losses':>11}{'delta':>9}{'p':>9}  verdict"]
    for r in res["rows"]:
        delta = r["loss_rate"] - r["win_rate"]
        verdict = ("ENRICHED in losses" if r["p"] < 0.05 / k
                   else "not established")
        out.append(f"{r['kind']:<24}{r['win_rate']:>9.2f}{r['loss_rate']:>11.2f}"
                   f"{delta:>+9.2f}{r['p']:>9.3f}  {verdict}")
    if not any(r["p"] < 0.05 / k for r in res["rows"]):
        out.append("")
        out.append("NO flag is enriched in losses after correction. Loss-only counts "
                   "would have shown all of these as 'the top failure modes'.")
    out.append("")
    out.append("A negative delta does NOT mean the behaviour helps: flags are counted "
               "over whole games, so they are confounded with game state (winning "
               "boards have more turns and energy). This filter refutes 'X explains "
               "our losses'; it cannot show that X helps or hurts.")
    return "\n".join(out)


def report(acc: Counter) -> str:
    """Headline is taken/strict WITH n_neutral adjacent.

    M21 reported gust 0/27 where 26 of the 27 were neutral. Printing the neutral
    count beside the headline is what makes that unmissable.
    """
    lines = [f"games {acc.get('games', 0)}  ·  MAIN states {acc.get('main_states', 0)}"]
    for kind in ("gust", "retreat"):
        strict, tempo = acc.get(f"{kind}_strict", 0), acc.get(f"{kind}_tempo", 0)
        neutral, undec = acc.get(f"{kind}_neutral", 0), acc.get(f"{kind}_undecidable", 0)
        took = acc.get(f"{kind}_strict_taken", 0)
        total = strict + tempo + neutral + undec
        rate = f"{took}/{strict}" + (f" = {took / strict:.3f}" if strict else "")
        lines.append(
            f"  {kind:<8} STRICT {rate}   "
            f"[tempo {tempo} · neutral {neutral} · undecidable {undec}]")
        if total and undec / total > 0.20:
            lines.append(f"    ⚠ {undec}/{total} undecidable (>20%) — "
                         f"metric is VOID for this run")
        if strict == 0 and total:
            lines.append(f"    (no strict {kind} opportunities — a 0/0 headline is "
                         f"not a defect)")
    for k in sorted(acc):
        if k.startswith("drop_") and acc[k]:
            lines.append(f"  drop {k[5:]}: {acc[k]}")
    return "\n".join(lines)


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replays")
    r.add_argument("--sub", type=int, required=True)
    r.add_argument("--limit", type=int, default=None)
    s = sub.add_parser("series")
    s.add_argument("--a", required=True)
    s.add_argument("--b", default="solver:lucario")
    s.add_argument("-n", type=int, default=100)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--workers", type=int, default=8)   # 8 is a HARD cap (CLAUDE.md)
    f = sub.add_parser("flags")
    f.add_argument("--sub", type=int, required=True)
    f.add_argument("--limit", type=int, default=None)
    a = p.parse_args()

    if a.cmd == "flags":
        print(report_flags(flag_contrast(a.sub, a.limit)))
        return
    acc = (from_replays(a.sub, a.limit) if a.cmd == "replays"
           else from_series(a.a, a.b, a.n, a.seed, min(a.workers, 8)))
    print(report(acc))


if __name__ == "__main__":
    _main()
