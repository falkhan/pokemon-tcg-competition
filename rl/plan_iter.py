"""M11 expert iteration — plan-conditioned collection + training.

The loop: the WIDENED turn solver (rl/turn_solver.solve_turn_line at 5x
deadline / 5x nodes, override gates bypassed) is the improvement operator;
training is purely supervised (rl/ppo.py untouched). Two collection modes:

  expert  teacher self-play: the solver's line is executed AND labeled —
          bootstrap data for osv3_plan0 (Rung 0).
  ei      the STUDENT advances the game (plan sampled at temperature tau,
          actions Gumbel-sampled conditioned on it — plan-level exploration,
          annealed over rounds); the teacher labels every visited decision.

THE ANTI-ALIASING INVARIANT (docs/M11-plan.md): a row's plan features always
come from the same solve that produced that row's label. The student's
tau-sampled plan shapes execution only — it is never a training input. This
is what separates M11 from the buddy-DAgger dead end (M9: 0.198).

Collection is multiprocess (spawn pool over game chunks, one live battle per
process) — the M9 single-process collector's ~5 games/min was the bottleneck.

CLI:
  python -m rl.plan_iter collect --mode expert --games 600 --out data/plan_ei0
  python -m rl.plan_iter train --data data/plan_ei0 data/bc_v2b \
      --init-v2 checkpoints/osv2_bc2.pt --name osv3_plan0
"""
import multiprocessing as mp
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cg.api import (CardType, OptionType, SelectContext, all_card_data,
                    to_observation_class)
from cg.game import battle_finish, battle_select, battle_start
from rl.turn_solver import MIN_OVERRIDE_SCORE
from rl.bc import load_population
from rl.encoders import (N_CONTEXTS, N_OPTION_TYPES, N_STATE_IDS_V3,
                         STATE_V2_DIM, encode_context, encode_option_v2,
                         encode_state_v2, encode_state_v3)
from rl.plan import (PLAN_DIM, derive_plan, encode_plan, enumerate_plans,
                     match_candidate)
from rl.policy import OptionScorerV3

ROOT = Path(__file__).resolve().parent.parent

WIDE_NODES = 4000          # widened data-gen budget (inference default: 800)
WIDE_DEPTH = 10            # (inference default: 8)

# card id -> CardType int, for --card-kind-weight (M27). Same source of truth as
# rl/plan.py's _IS_ENERGY, so a card-data change moves both at once.
_CARD_KIND = {c.cardId: int(c.cardType) for c in all_card_data()}

# M14 setup-plan teacher: the M13 value consumed at DATA-GEN only (the
# override law bans live consumption). Constants carry their provenance:
# margin 200 = calibrated 0.80-accuracy gap; turn cap 32 = the value's
# validated per-bucket range; LAMBDA 3000 = the score_leaf tail band.
VS_LAMBDA = 3000.0
VS_MARGIN = 200.0
VS_MAX_TURN = 32
# M18a: ei-mode rows where the student's executed action departs from the
# teacher's label carry this weight (disagreement-weighted DAgger).
EI_DISAGREE_WEIGHT = 10.0
# Runaway-game guard (found 2026-07-17: a buddy/rule matchup can stall
# forever; solver mirrors never do, so M11/M13 never hit it). Games past the
# cap are scored as draws. Normal games run well under 300 prompts.
MAX_GAME_DECISIONS = 600


def _value_solve_factory(vnet, cell, stats, vs_margin=VS_MARGIN,
                         vs_max_turn=VS_MAX_TURN):
    """Per-worker value-guided solver for SETUP-plan commits. Returns
    value_solve(obs, deck) -> (line, trail) when a line beats stand-pat by
    the calibrated margin (else (None, None))."""
    from rl.turn_solver import _root_snapshot, score_leaf, solve_turn_line
    ctx_main = encode_context(SelectContext.MAIN)
    # M18 fix: the net's id width picks the state encoder. M17 fed 12-id v2
    # states to the 20-id hand-aware net -> RuntimeError on every call,
    # silently swallowed -> SETUP 0 across the whole 10k collection.
    enc_state = (encode_state_v3
                 if getattr(vnet, "n_state_ids", None) == N_STATE_IDS_V3
                 else encode_state_v2)

    def leaf_value_for(deck):
        def leaf_value(obs):
            num, sids = enc_state(obs.current, deck)
            sc = np.concatenate([num, ctx_main]).astype(np.float32)
            with torch.no_grad():
                se = vnet.embedding(
                    torch.from_numpy(sids).long().unsqueeze(0)).flatten(-2)
                s = vnet.state_enc(torch.cat(
                    [torch.from_numpy(sc).unsqueeze(0),
                     torch.zeros(1, PLAN_DIM), se], dim=-1))
                v = float(vnet.value_head(s).squeeze())
            sign = 1.0 if obs.current.yourIndex == cell["me"] else -1.0
            return VS_LAMBDA * sign * v
        return leaf_value

    def value_solve(obs, deck):
        if obs.current.turn >= vs_max_turn \
                or getattr(obs, "search_begin_input", None) is None:
            return None, None
        cell["me"] = obs.current.yourIndex
        lv = leaf_value_for(deck)
        stats["vs_calls"] += 1
        try:
            score, line, trail = solve_turn_line(
                obs, deck, deadline_s=0.5, dev=True, leaf_value=lv)
            snap = _root_snapshot(obs)
            if line:
                margin = score - score_leaf(snap, obs, dev=True,
                                            leaf_value=lv)
                if len(stats["vs_margins"]) < 500:
                    stats["vs_margins"].append(float(margin))
                if margin >= vs_margin:
                    return line, trail
        except Exception:
            # Worker crash-discipline (as in _solve) — but COUNTED, and the
            # first failure prints: M17's silent swallow hid the encoder bug.
            stats["vs_errors"] += 1
            if stats["vs_errors"] == 1:
                import traceback
                print("[value-solve] first error:", flush=True)
                traceback.print_exc()
        return None, None
    return value_solve


# ---------------------------------------------------------------- collection

def _solve(obs, deck, deadline_s):
    """Widened solve_turn_line with the solver-pilot's crash discipline:
    any exception -> no line (fallback pilot takes over)."""
    from rl.turn_solver import solve_turn_line
    if getattr(obs, "search_begin_input", None) is None:
        return None, [], []
    try:
        return solve_turn_line(obs, deck, deadline_s=deadline_s,
                               max_depth=WIDE_DEPTH, max_nodes=WIDE_NODES)
    except Exception:
        return None, [], []


def _action_valid(action, obs) -> bool:
    n = len(obs.select.option)
    k = obs.select.maxCount
    if not action or any(not (0 <= int(i) < n) for i in action):
        return False
    return len(action) == (1 if k == 1 else min(k, n))


def _fresh_seat_state():
    return {"key": None, "vec": np.zeros(PLAN_DIM, np.float32),
            "line": None, "step": 0, "on": False}


def _cleared(score) -> bool:
    """Rung 0'' (2026-07-17): honor the shipping solver's override bar. Lines
    below MIN_OVERRIDE_SCORE are the M8.1 dev-tier trap — trusting them was
    measured at 0.314 < 0.362, and cloning them produced Rung 0's 0.33 wall."""
    return score is not None and score >= MIN_OVERRIDE_SCORE


def _teacher_step(st, obs, obs_dict, deck, fallback, key, is_main,
                  deadline, label_deadline, stats, value_solve=None):
    """Teacher's labeled action for this prompt + (cands, plan_label) when a
    fresh plan row was created. Teacher semantics = the SHIPPING solver pilot:
    a solver line is executed/labeled only when it clears the override bar,
    else the greedy pilot labels. Invariant: a row carries a non-zero plan IFF
    its label comes from a bar-clearing line, and the plan derives from that
    very line (anti-aliasing)."""
    plan_row = None
    if is_main and st["key"] != key:
        score, line, trail = _solve(obs, deck, deadline)
        committed = bool(line) and _cleared(score)
        if committed:
            stats["kill_plans"] += 1
        elif value_solve is not None:
            vline, vtrail = value_solve(obs, deck)   # M14: SETUP plans
            if vline:
                line, trail, committed = vline, vtrail, True
                stats["setup_plans"] += 1
                if obs.current.turn >= 32:      # M33: late (turn>=32) commits
                    stats["setup_plans_late"] += 1
        plan = derive_plan(line, trail, obs) if committed else None
        cands = enumerate_plans(obs)
        idx = match_candidate(plan, cands)
        plan_row = (cands, idx)
        stats["plan_rows"] += 1
        if idx < 0:
            stats["plan_missed"] += 1
        elif idx == 0:
            stats["plan_null"] += 1
        elif cands[idx].needs_gust:
            stats["plan_gust"] += 1
        st.update(key=key, vec=encode_plan(plan),
                  line=line if committed else None, step=0, on=committed)

    action = None
    if st["key"] == key and st["on"] and st["line"] and \
            st["step"] < len(st["line"]):
        cand = [int(i) for i in st["line"][st["step"]]]
        if _action_valid(cand, obs):
            action = cand
            st["step"] += 1
        else:
            st["on"] = False
            stats["derails"] += 1
    if action is None and len(obs.select.option) >= 2:
        score2, line2, trail2 = _solve(obs, deck, label_deadline)
        if line2 and _cleared(score2) \
                and _action_valid([int(i) for i in line2[0]], obs):
            action = [int(i) for i in line2[0]]
            plan2 = derive_plan(line2, trail2, obs)
            st.update(key=key, vec=encode_plan(plan2), line=line2, step=1,
                      on=True)
        elif st["key"] == key and st["on"] is False and st["vec"].any():
            # committed plan died with the derail: abandon it so later rows
            # this turn stay label-consistent (greedy label <-> zero plan)
            st["vec"] = np.zeros(PLAN_DIM, np.float32)
    if action is None:
        picks = fallback(obs_dict)
        action = [int(i) for i in picks[:obs.select.maxCount]]
    return action, plan_row


def _collect_chunk(args):
    """One worker: play [lo, hi) games, write its own shards. Spawn-safe —
    everything it needs is re-imported/rebuilt here."""
    (mode, lo, hi, decks_file, out_dir, checkpoint, tau, dirichlet,
     deadline, label_deadline, shard_size, seed, worker, opponents,
     value_ckpt, vs_margin, vs_max_turn) = args
    import random

    from rl.generic_pilot import make_generic_pilot

    out = Path(out_dir)
    population = load_population(decks_file)
    rng = random.Random(seed * 10007 + worker)
    np_rng = np.random.default_rng(seed * 10007 + worker)
    torch.manual_seed(seed * 10007 + worker)

    from rl.policy import option_dim_of

    student = None
    if mode == "ei":
        ckpt = Path(checkpoint)
        if not ckpt.is_absolute() and not ckpt.exists():
            ckpt = ROOT / ckpt
        sdict = torch.load(ckpt, map_location="cpu")
        if "state_enc.0.weight" in sdict:
            from rl.encoders import V4_EXTRA_DIM
            skw = dict(n_state_ids=_n_ids_of(sdict),
                       option_dim=option_dim_of(sdict),
                       extra_dim=V4_EXTRA_DIM if "enc_ver" in sdict else 0)
        else:
            skw = {}                                       # {} = test stubs
        student = OptionScorerV3(**skw)
        student.load_state_dict(sdict)
        student.eval()

    stats = {"games": 0, "decisions": 0, "plan_rows": 0, "plan_missed": 0,
             "plan_null": 0, "plan_gust": 0, "derails": 0, "wins": [0, 0, 0],
             "kill_plans": 0, "setup_plans": 0, "setup_plans_late": 0,
             "timeouts": 0, "opp": {},
             "vs_calls": 0, "vs_errors": 0, "vs_margins": []}

    value_solve = None
    if value_ckpt:
        vp = Path(value_ckpt)
        if not vp.is_absolute() and not vp.exists():
            vp = ROOT / vp
        vdict = torch.load(vp, map_location="cpu")
        vkw = (dict(n_state_ids=_n_ids_of(vdict),
                    option_dim=option_dim_of(vdict))
               if "state_enc.0.weight" in vdict else {})   # {} = test stubs
        vnet = OptionScorerV3(**vkw)
        vnet.load_state_dict(vdict)
        vnet.eval()
        value_solve = _value_solve_factory(vnet, {"me": 0}, stats, vs_margin,
                                           vs_max_turn=vs_max_turn)

    opp_pilots = {}          # spec string -> (fn, deck_ids), built lazily
    def _opponent(spec_str, instance):
        if spec_str not in opp_pilots:
            from rl.matchrunner import make_pilot, parse_spec
            opp_pilots[spec_str] = make_pilot(parse_spec(spec_str), instance)
        return opp_pilots[spec_str]

    columns = ("states", "plans", "state_ids", "options", "option_ids",
               "n_options", "labels", "game_ids", "results", "deck_idx",
               "plan_cands", "n_plan_cands", "plan_labels", "weights")
    shard = {k: [] for k in columns}
    shard_idx = sum(1 for _ in out.glob(f"shard_w{worker:02d}_*.npz"))
    t0 = perf_counter()

    def flush():
        nonlocal shard_idx
        if not shard["labels"]:
            return
        np.savez_compressed(
            out / f"shard_w{worker:02d}_{shard_idx:04d}.npz",
            states=np.stack(shard["states"]),
            plans=np.stack(shard["plans"]),
            state_ids=np.stack(shard["state_ids"]),
            options=np.concatenate(shard["options"]),
            option_ids=np.concatenate(shard["option_ids"]),
            n_options=np.array(shard["n_options"], dtype=np.int32),
            labels=np.array(shard["labels"], dtype=np.int32),
            game_ids=np.array(shard["game_ids"], dtype=np.int32),
            results=np.array(shard["results"], dtype=np.float32),
            deck_idx=np.array(shard["deck_idx"], dtype=np.int32),
            plan_cands=(np.concatenate(shard["plan_cands"])
                        if shard["plan_cands"] else
                        np.zeros((0, PLAN_DIM), np.float32)),
            n_plan_cands=np.array(shard["n_plan_cands"], dtype=np.int32),
            plan_labels=np.array(shard["plan_labels"], dtype=np.int32),
            weights=np.array(shard["weights"], dtype=np.float32),
        )
        shard_idx += 1
        for v in shard.values():
            v.clear()

    # M21: expert mode always records the newest (v4) encoding — the M15
    # precedent; ei mode follows the student's own architecture.
    use_v4 = mode == "expert" or getattr(student, "extra_dim", 0) > 0
    if use_v4:
        from rl.encoders import encode_ctx_v4
        from rl.memory import OppMemory

    for game in range(lo, hi):
        picks_idx = [rng.randrange(len(population)) for _ in range(2)]
        decks = [population[picks_idx[0]], population[picks_idx[1]]]
        # M21: one opponent-memory per RECORDED seat, observed only at that
        # seat's own prompts (per-seat logs contract, rl/memory.py docstring)
        memories = [OppMemory(), OppMemory()] if use_v4 else None
        # M14 mixed opponents: one seat may be an external pilot (buddy /
        # rule expert / plain solver spec). Only the TEACHER seat is
        # recorded then (M10 ban: never imitate third parties).
        teacher_seat = game % 2
        opp_spec = None
        opp_fn = None
        if opponents:
            opp_spec = opponents[rng.randrange(len(opponents))]
            if opp_spec != "self":
                opp_fn, opp_deck = _opponent(opp_spec, f"op{worker}")
                decks[1 - teacher_seat] = list(opp_deck)
                picks_idx[1 - teacher_seat] = 0
        fallbacks = [make_generic_pilot(d) for d in decks]
        seat_state = [_fresh_seat_state(), _fresh_seat_state()]
        student_plan = [np.zeros(PLAN_DIM, np.float32),
                        np.zeros(PLAN_DIM, np.float32)]
        student_key = [None, None]

        obs_dict, start_data = battle_start(decks[0], decks[1])
        if start_data.errorPlayer >= 0:
            raise ValueError(f"battle_start rejected a deck "
                             f"(errorType={start_data.errorType})")

        game_rows: list[tuple] = []
        n_prompts = 0
        while obs_dict["current"]["result"] < 0:
            n_prompts += 1
            if n_prompts > MAX_GAME_DECISIONS:
                stats["timeouts"] += 1
                break
            player = obs_dict["current"]["yourIndex"]
            obs = to_observation_class(obs_dict)

            if opp_fn is not None and player != teacher_seat:
                # external opponent seat: it plays, nothing is recorded
                picks = opp_fn(obs_dict)
                obs_dict = battle_select(
                    [int(i) for i in picks[:obs.select.maxCount]])
                continue

            key = (obs.current.turn, player)
            is_main = obs.select.context == SelectContext.MAIN

            label_action, plan_row = _teacher_step(
                seat_state[player], obs, obs_dict, decks[player],
                fallbacks[player], key, is_main, deadline, label_deadline,
                stats, value_solve=value_solve)

            # M15: new data carries hand-aware ids (expert mode always; ei
            # mode follows the student's own id width). M21: v4 states carry
            # the appended encoder-v4 block + memory ids.
            if use_v4:
                memories[player].observe(obs)
                state_ctx, state_ids = encode_ctx_v4(
                    obs, decks[player], memories[player])
            else:
                enc_state = (encode_state_v3 if mode == "expert"
                             or getattr(student, "n_state_ids",
                                        None) == N_STATE_IDS_V3
                             else encode_state_v2)
                state_num, state_ids = enc_state(obs.current, decks[player])
                state_ctx = np.concatenate(
                    [state_num, encode_context(obs.select.context)]
                ).astype(np.float32)
            pairs = [encode_option_v2(o, obs) for o in obs.select.option]
            opts = np.stack([num for num, _ in pairs]).astype(np.float32)
            opt_ids = np.stack([ids for _, ids in pairs])

            cands_mat = np.zeros((0, PLAN_DIM), np.float32)
            plan_label = -1
            if plan_row is not None:
                cands, plan_label = plan_row
                cands_mat = np.stack([encode_plan(c) for c in cands]
                                     ).astype(np.float32)

            row_weight = 1.0
            if mode == "ei":
                if plan_row is not None:      # student picks ITS OWN plan here
                    s_idx = student.act_plan(state_ctx, state_ids, cands_mat,
                                             tau=tau, dirichlet_eps=dirichlet,
                                             rng=np_rng)
                    student_plan[player] = cands_mat[s_idx].copy()
                    student_key[player] = key
                s_vec = (student_plan[player] if student_key[player] == key
                         else np.zeros(PLAN_DIM, np.float32))
                # Actions are GREEDY given the sampled plan (fix 2026-07-17,
                # EI round 1 measured 0.314): exploration lives at the PLAN
                # level only — per-prompt Gumbel noise makes turns incoherent,
                # the exact failure mode plan conditioning exists to prevent.
                # M16: a pre-option-identity student consumes the legacy
                # option slice (recorded rows keep the full new encoding).
                s_dim = getattr(student, "option_dim", opts.shape[1])
                s_opts = opts[:, :s_dim] if opts.shape[1] > s_dim else opts
                exec_action = student.act(state_ctx, s_vec, state_ids, s_opts,
                                          opt_ids, obs.select.maxCount,
                                          greedy=True)
                exec_action = [int(i) for i in exec_action]
                if exec_action != label_action:   # off the teacher's line now
                    seat_state[player]["on"] = False
                    row_weight = EI_DISAGREE_WEIGHT   # M18a: DAgger weighting
            else:
                exec_action = label_action

            game_rows.append((state_ctx, seat_state[player]["vec"].copy()
                              if seat_state[player]["key"] == key
                              else np.zeros(PLAN_DIM, np.float32),
                              state_ids, opts, opt_ids, label_action[0],
                              player, cands_mat, plan_label, row_weight))
            stats["decisions"] += 1
            obs_dict = battle_select(exec_action)

        result = obs_dict["current"]["result"]
        if result < 0:
            result = 2                       # capped runaway -> scored a draw
        battle_finish()
        stats["wins"][result] += 1
        stats["games"] += 1
        if opp_spec is not None:
            o = stats["opp"].setdefault(opp_spec.split(":")[0], [0, 0])
            o[0] += result == teacher_seat        # teacher-seat wins
            o[1] += 1

        for (state_ctx, plan_vec, state_ids, opts, opt_ids, label, player,
             cands_mat, plan_label, row_weight) in game_rows:
            shard["states"].append(state_ctx)
            shard["plans"].append(plan_vec)
            shard["state_ids"].append(state_ids)
            shard["options"].append(opts)
            shard["option_ids"].append(opt_ids)
            shard["n_options"].append(len(opts))
            shard["labels"].append(label)
            shard["game_ids"].append(game)
            shard["results"].append(
                0.0 if result == 2 else (1.0 if result == player else -1.0))
            shard["deck_idx"].append(picks_idx[player])
            shard["plan_cands"].append(cands_mat)
            shard["n_plan_cands"].append(len(cands_mat))
            shard["plan_labels"].append(plan_label)
            shard["weights"].append(row_weight)

        if (game - lo + 1) % shard_size == 0:
            flush()
        if game - lo + 1 == 25:
            per_game = (perf_counter() - t0) / 25
            total_min = per_game * (hi - lo) / 60
            print(f"[w{worker}] game 25: {per_game:.1f}s/game -> "
                  f"projected {total_min:.0f} min for this worker's "
                  f"{hi - lo} games", flush=True)

    flush()
    return stats


def collect(mode: str, n_games: int, decks_file, out_dir: Path,
            checkpoint: str | None = None, tau: float = 1.0,
            dirichlet: float = 0.0, deadline: float = 2.0,
            label_deadline: float = 0.5, workers: int = 8,
            shard_size: int = 200, seed: int = 0,
            opponents: list | None = None,
            value_ckpt: str | None = None,
            vs_margin: float = VS_MARGIN,
            vs_max_turn: int = VS_MAX_TURN) -> dict:
    assert mode in ("expert", "ei")
    assert mode != "ei" or checkpoint, "--mode ei needs --checkpoint"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk = -(-n_games // workers)                    # ceil
    jobs = []
    lo = 0
    for w in range(workers):
        hi = min(lo + chunk, n_games)
        if lo >= hi:
            break
        jobs.append((mode, lo, hi, str(decks_file), str(out_dir), checkpoint,
                     tau, dirichlet, deadline, label_deadline, shard_size,
                     seed, w, opponents, value_ckpt, vs_margin, vs_max_turn))
        lo = hi

    if len(jobs) == 1:                                # tests / smoke path
        results = [_collect_chunk(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            results = pool.map(_collect_chunk, jobs)

    agg = {k: sum(r[k] for r in results)
           for k in ("games", "decisions", "plan_rows", "plan_missed",
                     "plan_null", "plan_gust", "derails", "kill_plans",
                     "setup_plans", "setup_plans_late", "timeouts",
                     "vs_calls", "vs_errors")}
    agg["wins"] = [sum(r["wins"][i] for r in results) for i in range(3)]
    agg["opp"] = {}
    for r in results:
        for k, (w_, n_) in r["opp"].items():
            o = agg["opp"].setdefault(k, [0, 0])
            o[0] += w_
            o[1] += n_
    rows = max(1, agg["plan_rows"])
    agg["coverage"] = 1.0 - agg["plan_missed"] / rows
    print(f"done: {agg['games']} games, {agg['decisions']} decisions, "
          f"{agg['plan_rows']} plan rows -> {out_dir}", flush=True)
    print(f"plan coverage {agg['coverage']:.3f}  "
          f"(missed {agg['plan_missed']})  null {agg['plan_null'] / rows:.2f}  "
          f"kill {agg['kill_plans']}  SETUP {agg['setup_plans']}  "
          f"gust {agg['plan_gust'] / rows:.2f}  derails {agg['derails']}",
          flush=True)
    if value_ckpt:
        margins = np.array([m for r in results for m in r["vs_margins"]],
                           dtype=np.float32)
        agg["vs_margins"] = margins.tolist()
        pct = ("p10/p50/p90/p99 = " + "/".join(
                   f"{np.percentile(margins, q):.0f}" for q in (10, 50, 90, 99))
               if len(margins) else "n/a (no solved lines)")
        print(f"value-solve: calls {agg['vs_calls']}  "
              f"errors {agg['vs_errors']}  commits {agg['setup_plans']}  "
              f"(late turn>=32: {agg['setup_plans_late']})  "
              f"margin {pct}  (threshold {vs_margin:.0f})", flush=True)
        if agg["setup_plans"] == 0:
            print("WARNING: SETUP=0 — a value ckpt was supplied but no setup "
                  "plan was ever committed (M17 failure signature; check "
                  "vs_errors and the margin distribution)", flush=True)
    for k, (w_, n_) in sorted(agg["opp"].items()):
        print(f"  vs {k}: teacher seat {w_}/{n_} ({w_ / max(1, n_):.2f})",
              flush=True)
    return agg


# ------------------------------------------------------------------ training

class BCDatasetV3(torch.utils.data.Dataset):
    """v2 shard rows + plan columns. Old plan-less shards (bc_v2b, ...) load
    with shaped defaults: plans=zeros (== "no plan"), no plan rows — so the
    diverse v2 data keeps regularizing the plan=0 fallback policy."""

    def __init__(self, data_dir: Path | list):
        cols = {k: [] for k in ("states", "state_ids", "options", "option_ids",
                                "n_options", "labels", "game_ids", "results",
                                "deck_idx", "weights")}
        plan_cols = {"plans": [], "plan_cands": [], "n_plan_cands": [],
                     "plan_labels": []}
        starts_list, cstarts_list = [], []
        option_base = game_base = cand_base = 0
        dirs = data_dir if isinstance(data_dir, (list, tuple)) else [data_dir]
        for d in dirs:
            for path in sorted(Path(d).glob("*.npz")):
                print(f"loading {path.name}", flush=True)
                shard = np.load(path)
                n = len(shard["labels"])
                starts = np.cumsum(shard["n_options"]) - shard["n_options"]
                starts_list.append(starts + option_base)
                cols["game_ids"].append(shard["game_ids"] + game_base)
                # M18a shim: weightless shards (all pre-M18 data) load as
                # uniformly weighted — same spirit as the plan-column shim.
                cols["weights"].append(
                    shard["weights"].astype(np.float32)
                    if "weights" in shard.files
                    else np.ones(n, np.float32))
                for k in cols:
                    if k not in ("game_ids", "weights"):
                        cols[k].append(shard[k])
                if "plans" in shard.files:
                    plan_cols["plans"].append(shard["plans"])
                    plan_cols["plan_cands"].append(shard["plan_cands"])
                    ncands = shard["n_plan_cands"]
                    plan_cols["plan_labels"].append(shard["plan_labels"])
                else:
                    plan_cols["plans"].append(np.zeros((n, PLAN_DIM), np.float32))
                    plan_cols["plan_cands"].append(
                        np.zeros((0, PLAN_DIM), np.float32))
                    ncands = np.zeros(n, dtype=np.int32)
                    plan_cols["plan_labels"].append(
                        np.full(n, -1, dtype=np.int32))
                plan_cols["n_plan_cands"].append(ncands)
                cstarts_list.append(np.cumsum(ncands) - ncands + cand_base)
                cand_base += int(ncands.sum())
                option_base += len(shard["options"])
                game_base += int(shard["game_ids"].max()) + 1
        # M15 pad shim: mixed id widths (legacy 12 vs hand-aware 20) — pad
        # every shard's state_ids to the widest with zeros ("no hand info",
        # the same semantics as the plan-column shim).
        widths = {a.shape[1] for a in cols["state_ids"]}
        if len(widths) > 1:
            wmax = max(widths)
            cols["state_ids"] = [
                np.pad(a, ((0, 0), (0, wmax - a.shape[1])))
                for a in cols["state_ids"]]
        # M16 pad shim: mixed option widths (legacy vs option-identity) — pad
        # legacy shards' options with zeros ("no identity info"; their PLAY
        # FEAT blocks are already blank, the same semantics).
        owidths = {a.shape[1] for a in cols["options"]}
        if len(owidths) > 1:
            owmax = max(owidths)
            cols["options"] = [
                np.pad(a, ((0, 0), (0, owmax - a.shape[1])))
                for a in cols["options"]]
        # M21 pad shim: mixed STATE widths (v3 vs encoder-v4) — pad v3 shards'
        # states with zeros ("no v4 info"): the v4 block is appended at the
        # tail, so zero-padding is exactly the migrated net's zero-init
        # semantics. state_ids mixing (20 vs 25) is the M15 shim above.
        swidths = {a.shape[1] for a in cols["states"]}
        if len(swidths) > 1:
            swmax = max(swidths)
            cols["states"] = [
                np.pad(a, ((0, 0), (0, swmax - a.shape[1])))
                for a in cols["states"]]
        for k, v in cols.items():
            setattr(self, k, np.concatenate(v))
        for k, v in plan_cols.items():
            setattr(self, k, np.concatenate(v))
        self.starts = np.concatenate(starts_list)
        self.cand_starts = np.concatenate(cstarts_list)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        s, n = self.starts[i], self.n_options[i]
        cs, cn = self.cand_starts[i], self.n_plan_cands[i]
        return (self.states[i], self.plans[i], self.state_ids[i],
                self.options[s:s + n], self.option_ids[s:s + n],
                self.labels[i], self.results[i], self.deck_idx[i],
                self.plan_cands[cs:cs + cn], self.plan_labels[i],
                self.weights[i])


def collate_v3(batch):
    B = len(batch)
    maxN = max(row[3].shape[0] for row in batch)
    maxM = max(1, max(row[8].shape[0] for row in batch))

    # state width read from the batch (M21: v3 = STATE_V2_DIM+N_CONTEXTS,
    # v4 adds V4_EXTRA_DIM; the dataset pad shim makes every row equal-width)
    states = torch.zeros(B, batch[0][0].shape[0])
    plans = torch.zeros(B, PLAN_DIM)
    state_ids = torch.zeros(B, batch[0][2].shape[0], dtype=torch.long)
    options = torch.zeros(B, maxN, batch[0][3].shape[1])
    option_ids = torch.zeros(B, maxN, 2, dtype=torch.long)
    valid = torch.zeros(B, maxN, dtype=torch.bool)
    labels = torch.zeros(B, dtype=torch.long)
    results = torch.zeros(B, dtype=torch.float32)
    deck_idx = torch.zeros(B, dtype=torch.long)
    plan_cands = torch.zeros(B, maxM, PLAN_DIM)
    plan_valid = torch.zeros(B, maxM, dtype=torch.bool)
    plan_labels = torch.full((B,), -1, dtype=torch.long)
    weights = torch.ones(B, dtype=torch.float32)

    for i, (state, plan, sids, menu, oids, label, result, didx,
            cands, plabel, weight) in enumerate(batch):
        n = menu.shape[0]
        states[i] = torch.from_numpy(state)
        plans[i] = torch.from_numpy(plan)
        state_ids[i] = torch.from_numpy(sids.astype(np.int64))
        options[i, :n] = torch.from_numpy(menu)
        option_ids[i, :n] = torch.from_numpy(oids.astype(np.int64))
        valid[i, :n] = True
        labels[i] = int(label)
        results[i] = float(result)
        deck_idx[i] = int(didx)
        m = cands.shape[0]
        if m:
            plan_cands[i, :m] = torch.from_numpy(cands)
            plan_valid[i, :m] = True
        plan_labels[i] = int(plabel)
        weights[i] = float(weight)

    return (states, plans, state_ids, options, option_ids, valid, labels,
            results, deck_idx, plan_cands, plan_valid, plan_labels, weights)


def load_v3_into_v3h(v3_sd: dict, plan_dim: int = PLAN_DIM) -> OptionScorerV3:
    """M15 warm start: legacy 12-id v3 weights into a hand-aware (20-id) net.
    All shared weights copy verbatim; the 8 new hand-embedding column blocks
    of state_enc.0 are ZERO-initialized, so hand-aware(zero-hand-ids) ==
    legacy net exactly (the warm-start invariant, third use). option_dim is
    INFERRED from the source checkpoint (M25 fix — was hardcoded to legacy
    OPTION_V2_DIM, which silently broke on the option-identity (OPTION_V3_DIM)
    checkpoints that have been the default since M16; same inference `_n_ids_of`
    /`load_v3h_into_v3o` already use for their own dims)."""
    from rl.encoders import EMBED_DIM, N_OPTION_IDS, N_STATE_IDS_V3
    option_dim = v3_sd["option_enc.0.weight"].shape[1] - N_OPTION_IDS * EMBED_DIM
    model = OptionScorerV3(plan_dim=plan_dim, n_state_ids=N_STATE_IDS_V3,
                           option_dim=option_dim)
    sd = model.state_dict()
    for k, v in v3_sd.items():
        if k == "state_enc.0.weight":
            new = torch.zeros_like(sd[k])
            new[:, :v.shape[1]] = v               # ids sit at the END: old
            sd[k] = new                           # block maps 1:1, rest zero
        else:
            sd[k] = v
    model.load_state_dict(sd)
    return model


def load_v2_into_v3(v2_sd: dict, plan_dim: int = PLAN_DIM) -> OptionScorerV3:
    """Warm-start a V3 from a V2 state dict: shared weights copied verbatim,
    the plan_dim new state_enc.0 columns ZERO-initialized (V3(plan=0) == V2 —
    the warm-start invariant), plan_enc/plan_head keep their fresh init."""
    model = OptionScorerV3(plan_dim=plan_dim)
    sd = model.state_dict()
    base = STATE_V2_DIM + N_CONTEXTS
    for k, v in v2_sd.items():
        if k == "state_enc.0.weight":
            new = torch.zeros_like(sd[k])
            new[:, :base] = v[:, :base]
            new[:, base + plan_dim:] = v[:, base:]
            sd[k] = new
        else:
            sd[k] = v
    model.load_state_dict(sd)
    return model


def load_v3h_into_v3o(v3h_sd: dict, plan_dim: int = PLAN_DIM) -> OptionScorerV3:
    """M16 warm start: pre-option-identity weights into an OPTION_V3_DIM net.
    option_enc.0's new numeric input columns are ZERO-init and its embedding
    column block shifts right, so v3o(legacy-encoded options padded with
    zeros, PLAY ids 0) == old net exactly (warm-start invariant, fourth use).
    NB the invariant holds for MASKED identity, not live encodings: real PLAY
    ids light up already-trained embedding rows through existing weights —
    intended (that pathway is the warm start's head start), just not identity."""
    from rl.encoders import EMBED_DIM, N_OPTION_IDS, OPTION_V3_DIM
    old_opt = v3h_sd["option_enc.0.weight"].shape[1] - N_OPTION_IDS * EMBED_DIM
    model = OptionScorerV3(plan_dim=plan_dim, n_state_ids=_n_ids_of(v3h_sd),
                           option_dim=OPTION_V3_DIM)
    sd = model.state_dict()
    for k, v in v3h_sd.items():
        if k == "option_enc.0.weight":
            new = torch.zeros_like(sd[k])
            new[:, :old_opt] = v[:, :old_opt]
            new[:, OPTION_V3_DIM:] = v[:, old_opt:]
            sd[k] = new
        else:
            sd[k] = v
    model.load_state_dict(sd)
    return model


def load_v3o_into_v3m(v3o_sd: dict, plan_dim: int = PLAN_DIM) -> OptionScorerV3:
    """M27 warm start: an OPTION_V3_DIM checkpoint into an OPTION_M27_DIM net.

    Same shape as load_v3h_into_v3o (sixth use of the warm-start invariant):
    option_enc.0's N_OPTION_PLAY_PRE new numeric columns are ZERO-init and the
    embedding column block shifts right, so at init the new net reproduces the
    old one EXACTLY on any input — the M27 play-precondition slots start with
    no influence and have to earn it from the gradient.
    """
    from rl.encoders import (EMBED_DIM, N_OPTION_IDS, OPTION_M28_DIM,
                             V4_EXTRA_DIM)
    OPTION_M27_DIM = OPTION_M28_DIM          # widen to the CURRENT option width
    old_opt = v3o_sd["option_enc.0.weight"].shape[1] - N_OPTION_IDS * EMBED_DIM
    model = OptionScorerV3(
        plan_dim=plan_dim, n_state_ids=_n_ids_of(v3o_sd),
        option_dim=OPTION_M28_DIM,
        extra_dim=V4_EXTRA_DIM if "enc_ver" in v3o_sd else 0)
    sd = model.state_dict()
    for k, v in v3o_sd.items():
        if k == "option_enc.0.weight":
            new = torch.zeros_like(sd[k])
            new[:, :old_opt] = v[:, :old_opt]           # numeric block verbatim
            new[:, OPTION_M27_DIM:] = v[:, old_opt:]    # id embeddings shift right
            sd[k] = new
        else:
            sd[k] = v
    model.load_state_dict(sd)
    return model


def migrate_v3_to_v4(v3_sd: dict, plan_dim: int = PLAN_DIM) -> OptionScorerV3:
    """M21 warm start: a v3 checkpoint into an encoder-v4 net
    (extra_dim=V4_EXTRA_DIM, N_STATE_IDS_V4 ids). state_enc.0 columns:
    [S|C] verbatim → V4 block ZERO → [plan | old id embeds] shifted right →
    new memory-id embed columns ZERO. Fifth use of the warm-start invariant:
    v4(any v4 block, any mem ids) == v3 exactly at init, since every new
    input column is zero — parity-tested with garbage in the new inputs."""
    from rl.encoders import (EMBED_DIM, N_OPTION_IDS, N_STATE_IDS_V4,
                             V4_EXTRA_DIM)
    base = STATE_V2_DIM + N_CONTEXTS
    old_ids = _n_ids_of(v3_sd)
    old_tail = plan_dim + old_ids * EMBED_DIM
    option_dim = v3_sd["option_enc.0.weight"].shape[1] - N_OPTION_IDS * EMBED_DIM
    model = OptionScorerV3(plan_dim=plan_dim, n_state_ids=N_STATE_IDS_V4,
                           option_dim=option_dim, extra_dim=V4_EXTRA_DIM)
    sd = model.state_dict()
    for k, v in v3_sd.items():
        if k == "state_enc.0.weight":
            new = torch.zeros_like(sd[k])
            new[:, :base] = v[:, :base]
            new[:, base + V4_EXTRA_DIM:base + V4_EXTRA_DIM + old_tail] = v[:, base:]
            sd[k] = new
        else:
            sd[k] = v
    model.load_state_dict(sd)   # enc_ver buffer keeps its 4.0 init
    return model


def _n_ids_of(sd: dict) -> int:
    """Infer a v3/v4 checkpoint's state-id count from its first Linear width
    (M21: v4 checkpoints declare themselves via the enc_ver buffer)."""
    from rl.encoders import EMBED_DIM, V4_EXTRA_DIM
    width = sd["state_enc.0.weight"].shape[1]
    if "enc_ver" in sd:
        width -= V4_EXTRA_DIM
    return (width - STATE_V2_DIM - N_CONTEXTS - PLAN_DIM) // EMBED_DIM


def apply_class_weights(ds, class_weights: dict[int, float]) -> None:
    """M26: per-OptionType loss weighting on the teacher's chosen action —
    policy-CE only (the same channel as the M18a weights); evaluate() stays
    unweighted so val_acc remains comparable across milestones. Dose law
    (M19/M21): cost tracks the reweighted-row fraction — keep the touched
    share small (ATTACH is 5.9% of the M25 corpus)."""
    chosen_type = ds.options[ds.starts + ds.labels, :N_OPTION_TYPES].argmax(1)
    for opt_type, factor in class_weights.items():
        mask = chosen_type == opt_type
        ds.weights[mask] *= factor
        print(f"class-weight {OptionType(opt_type).name} x{factor}: "
              f"{int(mask.sum())} rows ({mask.mean():.1%})")


def apply_card_kind_weights(ds, kind_weights: dict[int, float]) -> None:
    """M27: per-CardType loss weighting on the teacher's ACTED CARD.

    `--class-weight PLAY:k` is too blunt for the M27 defect: the clone plays
    supporters at 0.51x and stadiums at 0.06x the teacher's rate while MATCHING
    items (13.4% vs 13.8%, docs/M27.md Probe 3), and PLAY lumps all three
    together — lifting the class would push the already-over-played items
    further over. This keys on the acted card's CardType instead, so supporters
    and stadiums can be lifted on their own.

    Same weight channel and same policy-CE-only contract as
    apply_class_weights; evaluate() stays unweighted.
    """
    acted = ds.option_ids[ds.starts + ds.labels, 0].astype(int)
    kinds = np.array([_CARD_KIND.get(int(cid), -1) for cid in acted])
    for kind, factor in kind_weights.items():
        mask = kinds == kind
        ds.weights[mask] *= factor
        print(f"card-kind-weight {CardType(kind).name} x{factor}: "
              f"{int(mask.sum())} rows ({mask.mean():.1%})")


def apply_phase_weights(ds, factor: float, deck_at: int = 15) -> None:
    """M28 Track G1: upweight LATE-GAME rows in the policy CE.

    docs/M27.md Probe 2: val_acc runs 0.738 at deck 30+ but 0.601 at deck 7-15
    and 0.644 at deck <=6 — competence collapses in exactly the region where
    deck-out is decided, and only 13.2% of the corpus lives there. This gives
    those rows more gradient. `deck_at` is the remaining-deck threshold below
    which a row counts as late game.

    Reads deckCount from the state vector's global block (encode_state index 3,
    normalized /60). Same weight channel and policy-CE-only contract as
    apply_class_weights; evaluate() stays unweighted.
    """
    deck = ds.states[:, 3] * 60.0
    mask = deck <= deck_at
    ds.weights[mask] *= factor
    print(f"phase-weight deck<={deck_at} x{factor}: "
          f"{int(mask.sum())} rows ({mask.mean():.1%})")


def apply_outcome_weights(ds, alpha: float) -> None:
    """M37 AWR pre-work: down-weight rows from games the imitated seat did
    NOT win (advantage-weighted BC, the soft form of --winners-only).

    M28's winners-only filter — the campaign's one resolved-positive weights
    change — throws away the loss half of the corpus; this keeps it at
    weight `alpha` (< 1) so the trainer still sees what losing play looks
    like without cloning it at full strength. Reads the per-row `results`
    column (+1 win / -1 loss / 0 draw of the imitated seat,
    rl/replay_bc.py); rows with results < 1 are scaled by alpha. Same
    weight channel and policy-CE-only contract as apply_class_weights;
    evaluate() stays unweighted. The M38 arm sweeps alpha; alpha=0
    reproduces winners-only exactly (up to the shard split).
    """
    mask = ds.results < 1.0
    ds.weights[mask] *= alpha
    print(f"outcome-weight non-win x{alpha}: "
          f"{int(mask.sum())} rows ({mask.mean():.1%})")


def train(data_dirs: list, name: str, init: str | None = None,
          init_v2: str | None = None, init_v3h: str | None = None,
          init_v3o: str | None = None, init_v3m: str | None = None,
          epochs: int = 8, lr: float = 3e-4,
          batch_size: int = 256, plan_weight: float = 1.0,
          uniform_weights: bool = False,
          class_weights: dict[int, float] | None = None,
          card_kind_weights: dict[int, float] | None = None,
          phase_weight: float | None = None,
          phase_deck_at: int = 15,
          outcome_weight: float | None = None) -> None:
    """Supervised: CE(policy) + 0.5*Huber(value) + plan_weight*CE(plan head)
    over rows with plan_labels >= 0. Best-val-acc checkpointing (train_v2's
    ritual); reports policy AND plan-head validation accuracy."""
    ds = BCDatasetV3([Path(d) for d in data_dirs])
    if uniform_weights:
        # M21 Gate-A kill lesson: 36% of the 4k-game EI rows carried the 10x
        # disagreement weight (~85% effective loss share) and dragged the
        # policy off its PPO gains (osv4_ei1 0.225 vs seed ~0.487 — the M18
        # plan4 / M19 reweighting law at maximum dose). This neutralizes the
        # recorded weights at load time.
        n_rw = int((ds.weights > 1.0).sum())
        ds.weights = np.ones_like(ds.weights)
        print(f"uniform-weights: neutralized {n_rw} reweighted rows")
    if class_weights:
        apply_class_weights(ds, class_weights)
    if card_kind_weights:
        apply_card_kind_weights(ds, card_kind_weights)
    if phase_weight:
        apply_phase_weights(ds, phase_weight, phase_deck_at)
    if outcome_weight is not None:
        apply_outcome_weights(ds, outcome_weight)

    rng = np.random.default_rng(0)
    unique_games = np.unique(ds.game_ids)
    val_games = set(rng.choice(unique_games,
                               max(1, int(0.1 * len(unique_games))),
                               replace=False).tolist())
    is_val = np.isin(ds.game_ids, list(val_games))
    train_ds = torch.utils.data.Subset(ds, np.nonzero(~is_val)[0])
    val_ds = torch.utils.data.Subset(ds, np.nonzero(is_val)[0])
    print(f"train {len(train_ds)}, val {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          collate_fn=collate_v3)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_v3)

    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() or p.exists() else ROOT / p

    if init is not None:
        from rl.encoders import V4_EXTRA_DIM
        from rl.policy import option_dim_of
        sdict = torch.load(_resolve(init), map_location="cpu")
        extra = V4_EXTRA_DIM if "enc_ver" in sdict else 0   # M21 v4 sniff
        model = OptionScorerV3(n_state_ids=_n_ids_of(sdict),
                               option_dim=option_dim_of(sdict),
                               extra_dim=extra)
        model.load_state_dict(sdict)
        print(f"warm-start from {'v4' if extra else 'v3'} {init} "
              f"(n_state_ids={model.n_state_ids}, "
              f"option_dim={model.option_dim}, extra_dim={extra})")
    elif init_v3m is not None:
        model = load_v3o_into_v3m(torch.load(_resolve(init_v3m),
                                             map_location="cpu"))
        print(f"warm-start from pre-M27 v3 {init_v3m} "
              "(play-precondition columns zero-init)")
    elif init_v3o is not None:
        model = load_v3h_into_v3o(torch.load(_resolve(init_v3o),
                                             map_location="cpu"))
        print(f"warm-start from pre-option-identity v3 {init_v3o} "
              "(option-identity columns zero-init)")
    elif init_v3h is not None:
        model = load_v3_into_v3h(torch.load(_resolve(init_v3h),
                                            map_location="cpu"))
        print(f"warm-start from legacy v3 {init_v3h} "
              "(hand-embedding columns zero-init)")
    elif init_v2 is not None:
        model = load_v2_into_v3(torch.load(_resolve(init_v2),
                                           map_location="cpu"))
        print(f"warm-start from v2 {init_v2} (plan columns zero-init)")
    else:
        # M23 audit E1: width-driven from the data, like the bc.py v2 path —
        # replay-clone shards carry legacy 12-wide state_ids, and a net built
        # at the hand-aware width (20) cannot consume them.
        model = OptionScorerV3(n_state_ids=ds.state_ids.shape[1],
                               option_dim=ds.options.shape[1])
        print(f"fresh V3 (n_state_ids={model.n_state_ids}, "
              f"option_dim={model.option_dim})")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_acc = 0.0

    def evaluate():
        model.eval()
        correct = total = pcorrect = ptotal = 0
        with torch.no_grad():
            for batch in val_dl:
                (states, plans, sids, options, oids, valid, labels, _, _,
                 cands, cvalid, plabels, _) = batch
                logits, _ = model(states, plans, sids, options, oids)
                logits = logits.masked_fill(~valid, -1e9)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += len(labels)
                mask = plabels >= 0
                if mask.any():
                    pl = model.plan_logits(states[mask], sids[mask],
                                           cands[mask])
                    pl = pl.masked_fill(~cvalid[mask], -1e9)
                    pcorrect += (pl.argmax(dim=1) == plabels[mask]).sum().item()
                    ptotal += int(mask.sum())
        return correct / max(1, total), pcorrect / max(1, ptotal)

    if any(p is not None for p in (init, init_v2, init_v3h, init_v3o,
                                   init_v3m)):
        acc0, pacc0 = evaluate()
        print(f"init val_acc {acc0:.3f}  plan_acc {pacc0:.3f}")

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch in train_dl:
            (states, plans, sids, options, oids, valid, labels, results, _,
             cands, cvalid, plabels, weights) = batch
            opt.zero_grad()
            logits, value = model(states, plans, sids, options, oids)
            logits = logits.masked_fill(~valid, -1e9)
            # M18a: per-row disagreement weights on the policy CE only —
            # value/plan targets aren't better on disagreement rows, and an
            # unweighted evaluate() keeps val_acc comparable across milestones.
            ce = F.cross_entropy(logits, labels, reduction="none")
            loss = (ce * weights).sum() / weights.sum() \
                + 0.5 * F.huber_loss(value, results)
            mask = plabels >= 0
            if mask.any():
                pl = model.plan_logits(states[mask], sids[mask], cands[mask])
                pl = pl.masked_fill(~cvalid[mask], -1e9)
                loss = loss + plan_weight * F.cross_entropy(pl, plabels[mask])
            loss.backward()
            opt.step()
            running += loss.item()
        acc, pacc = evaluate()
        print(f"epoch {epoch}: train_loss {running / len(train_dl):.3f}  "
              f"val_acc {acc:.3f}  plan_acc {pacc:.3f}", flush=True)
        if acc > best_acc:
            best_acc = acc
            (ROOT / "checkpoints").mkdir(exist_ok=True)
            torch.save(model.state_dict(),
                       ROOT / "checkpoints" / f"{name}.pt")


# ------------------------------------------------------------- M18a relabel

def _student_agreement(model, shard, batch_size: int = 512) -> np.ndarray:
    """Per-row bool mask: does the model's greedy argmax match the stored
    teacher label? Batched forward over the ragged option slices — the same
    masked-argmax math as act()/evaluate(). Narrow legacy columns are
    zero-padded for inference only (the BCDatasetV3 shim semantics)."""
    n = len(shard["labels"])
    n_opts = shard["n_options"].astype(np.int64)
    starts = np.cumsum(n_opts) - n_opts
    states = shard["states"].astype(np.float32)
    plans = (shard["plans"] if "plans" in shard
             else np.zeros((n, PLAN_DIM))).astype(np.float32)
    sids = shard["state_ids"].astype(np.int64)
    if sids.shape[1] < model.n_state_ids:
        sids = np.pad(sids, ((0, 0), (0, model.n_state_ids - sids.shape[1])))
    opts = shard["options"].astype(np.float32)
    if opts.shape[1] < model.option_dim:
        opts = np.pad(opts, ((0, 0), (0, model.option_dim - opts.shape[1])))
    oids = shard["option_ids"].astype(np.int64)

    agree = np.zeros(n, dtype=bool)
    with torch.no_grad():
        for lo in range(0, n, batch_size):
            hi = min(lo + batch_size, n)
            maxN = int(n_opts[lo:hi].max())
            menu = torch.zeros(hi - lo, maxN, opts.shape[1])
            menu_ids = torch.zeros(hi - lo, maxN, oids.shape[1],
                                   dtype=torch.long)
            valid = torch.zeros(hi - lo, maxN, dtype=torch.bool)
            for i in range(hi - lo):
                s, m = starts[lo + i], n_opts[lo + i]
                menu[i, :m] = torch.from_numpy(opts[s:s + m])
                menu_ids[i, :m] = torch.from_numpy(oids[s:s + m])
                valid[i, :m] = True
            logits, _ = model(torch.from_numpy(states[lo:hi]),
                              torch.from_numpy(plans[lo:hi]),
                              torch.from_numpy(sids[lo:hi]),
                              menu, menu_ids)
            logits = logits.masked_fill(~valid, -1e9)
            agree[lo:hi] = (logits.argmax(dim=1).numpy()
                            == shard["labels"][lo:hi])
    return agree


def relabel(data_dirs: list, ckpt: str, weight: float = EI_DISAGREE_WEIGHT,
            suffix: str = "_w", batch_size: int = 512) -> None:
    """M18a offline disagreement weighting: run the student's greedy argmax
    over every stored decision and write sibling '<dir><suffix>' shard dirs —
    all original columns unchanged, plus a per-row `weights` column (`weight`
    where the student disagrees with the stored teacher label, else 1)."""
    from rl.policy import option_dim_of

    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() or p.exists() else ROOT / p

    sdict = torch.load(_resolve(ckpt), map_location="cpu")
    model = OptionScorerV3(n_state_ids=_n_ids_of(sdict),
                           option_dim=option_dim_of(sdict))
    model.load_state_dict(sdict)
    model.eval()

    for d in data_dirs:
        src = _resolve(d)
        out = Path(str(src).rstrip("/") + suffix)
        out.mkdir(parents=True, exist_ok=True)
        n_dis = n_rows = 0
        for path in sorted(src.glob("*.npz")):
            shard = dict(np.load(path))
            agree = _student_agreement(model, shard, batch_size)
            shard["weights"] = np.where(agree, 1.0, weight).astype(np.float32)
            np.savez_compressed(out / path.name, **shard)
            n_dis += int((~agree).sum())
            n_rows += len(agree)
            print(f"{path.name}: disagreement {int((~agree).sum())}"
                  f"/{len(agree)}", flush=True)
        print(f"{d}: disagreement {n_dis}/{n_rows} = "
              f"{n_dis / max(1, n_rows):.3f} -> {out}", flush=True)


# ----------------------------------------------------------------------- CLI

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="expert/EI plan-conditioned collection")
    c.add_argument("--mode", choices=["expert", "ei"], required=True)
    c.add_argument("--games", type=int, default=600)
    c.add_argument("--decks", type=str, default="data/league/population.json")
    c.add_argument("--out", type=str, required=True)
    c.add_argument("--checkpoint", type=str, default=None,
                   help="student checkpoint (ei mode)")
    c.add_argument("--tau", type=float, default=1.0)
    c.add_argument("--dirichlet", type=float, default=0.0)
    c.add_argument("--deadline", type=float, default=2.0)
    c.add_argument("--label-deadline", type=float, default=0.5)
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--shard-size", type=int, default=200)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--opponents", type=str, nargs="+", default=None,
                   help="opponent specs rotated per game ('self' = teacher "
                        "mirror); only the teacher seat is recorded vs "
                        "external opponents (M14)")
    c.add_argument("--value-ckpt", type=str, default=None,
                   help="setup-value checkpoint enabling SETUP-plan commits "
                        "(M14; margin 200, turn<32)")
    c.add_argument("--vs-margin", type=float, default=VS_MARGIN,
                   help="SETUP commit margin over stand-pat (M18: "
                        "recalibrate from the printed margin distribution)")
    c.add_argument("--vs-max-turn", type=int, default=VS_MAX_TURN,
                   help="M33: turn cap above which value_solve is off (default "
                        "32; raise once the value ranks late states — Stage 1 "
                        "found t32+ acc ~0.76-0.84 on a fresh net)")
    t = sub.add_parser("train", help="train OptionScorerV3 on plan shards")
    t.add_argument("--data", type=str, nargs="+", required=True)
    t.add_argument("--name", type=str, required=True)
    t.add_argument("--init", type=str, default=None,
                   help="warm-start from a v3 checkpoint")
    t.add_argument("--init-v2", type=str, default=None,
                   help="warm-start from a v2 checkpoint (zero-init plan cols)")
    t.add_argument("--init-v3h", type=str, default=None,
                   help="warm-start a HAND-AWARE net from a legacy v3 "
                        "checkpoint (M15; hand columns zero-init)")
    t.add_argument("--init-v3m", type=str, default=None,
                   help="M27 warm start: an OPTION_V3_DIM checkpoint into an "
                        "OPTION_M27_DIM net (play-precondition columns "
                        "zero-init, so init is exactly the old net)")
    t.add_argument("--init-v3o", type=str, default=None,
                   help="warm-start an OPTION-IDENTITY net from a pre-M16 "
                        "v3/v3h checkpoint (option columns zero-init)")
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--plan-weight", type=float, default=1.0)
    t.add_argument("--uniform-weights", action="store_true",
                   help="M21: neutralize per-row disagreement weights at load "
                        "(the Gate-A kill fix — see docs/M21.md)")
    t.add_argument("--class-weight", action="append", default=[],
                   metavar="TYPE:K",
                   help="M26: multiply the policy-CE weight of rows whose "
                        "teacher action is OptionType TYPE by K "
                        "(e.g. ATTACH:3; repeatable)")
    t.add_argument("--card-kind-weight", action="append", default=[],
                   metavar="KIND:K",
                   help="M27: multiply the policy-CE weight of rows whose "
                        "teacher ACTED CARD is CardType KIND by K "
                        "(e.g. SUPPORTER:5 STADIUM:5; repeatable). Finer than "
                        "--class-weight PLAY:k, which also lifts items")
    t.add_argument("--phase-weight", type=float, default=None,
                   metavar="K",
                   help="M28: multiply the policy-CE weight of LATE-GAME rows "
                        "(remaining deck <= --phase-deck-at) by K — the region "
                        "where val_acc collapses (docs/M27.md Probe 2)")
    t.add_argument("--phase-deck-at", type=int, default=15,
                   help="remaining-deck threshold for --phase-weight")
    t.add_argument("--outcome-weight", type=float, default=None,
                   metavar="ALPHA",
                   help="M37 AWR: multiply the policy-CE weight of rows from "
                        "games the imitated seat did NOT win by ALPHA — the "
                        "soft form of replay_bc --winners-only (alpha=0 "
                        "reproduces it; docs/M37-plan.md W4)")
    r = sub.add_parser("relabel", help="M18a: write disagreement-weighted "
                                       "sibling shard dirs (<dir><suffix>)")
    r.add_argument("--data", type=str, nargs="+", required=True)
    r.add_argument("--ckpt", type=str, required=True,
                   help="student checkpoint whose argmax defines agreement")
    r.add_argument("--weight", type=float, default=EI_DISAGREE_WEIGHT)
    r.add_argument("--suffix", type=str, default="_w")
    r.add_argument("--batch-size", type=int, default=512)
    args = p.parse_args()

    if args.cmd == "collect":
        collect(args.mode, args.games, args.decks, Path(args.out),
                checkpoint=args.checkpoint, tau=args.tau,
                dirichlet=args.dirichlet, deadline=args.deadline,
                label_deadline=args.label_deadline, workers=args.workers,
                shard_size=args.shard_size, seed=args.seed,
                opponents=args.opponents, value_ckpt=args.value_ckpt,
                vs_margin=args.vs_margin, vs_max_turn=args.vs_max_turn)
    elif args.cmd == "relabel":
        relabel(args.data, args.ckpt, weight=args.weight,
                suffix=args.suffix, batch_size=args.batch_size)
    else:
        class_weights = {}
        for spec in args.class_weight:
            type_name, _, factor = spec.partition(":")
            class_weights[int(OptionType[type_name.upper()])] = float(factor)
        card_kind_weights = {}
        for spec in args.card_kind_weight:
            kind_name, _, factor = spec.partition(":")
            card_kind_weights[int(CardType[kind_name.upper()])] = float(factor)
        train(args.data, args.name, init=args.init, init_v2=args.init_v2,
              init_v3h=args.init_v3h, init_v3o=args.init_v3o,
              init_v3m=args.init_v3m,
              epochs=args.epochs, lr=args.lr,
              batch_size=args.batch_size, plan_weight=args.plan_weight,
              uniform_weights=args.uniform_weights,
              class_weights=class_weights or None,
              card_kind_weights=card_kind_weights or None,
              phase_weight=args.phase_weight,
              phase_deck_at=args.phase_deck_at,
              outcome_weight=args.outcome_weight)
