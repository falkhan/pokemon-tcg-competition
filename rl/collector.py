"""Parallel self-play collection for PPO with an OPPONENT POOL (M2 Path A).

Why a pool: pure mirror self-play only teaches the agent to beat copies of itself, but
the leaderboard is full of other decks/strategies. So the LEARNING policy plays one seat
(sampled actions, recorded), and the OTHER seat is sampled per game from a pool of
opponents: the current policy (mirror), the rule experts, the GENERIC pilot on the
known decks (M7.4b), past promoted selves, and random. Only the learning seat's
decisions are recorded — PPO is on-policy, so we can only update on actions taken by
the current policy with known logprobs.

The cg engine holds ONE battle per process (cg/sim.py global) -> multiprocessing, one
engine per worker. Everything heavy is imported/loaded inside the worker; only picklable
config (paths, deck names/id-lists, spec tuples) crosses the process boundary.

Opponents are built through rl/matchrunner.make_pilot (the canonical OpponentSpec
home), so every spec kind it supports works here: ("model", ckpt, deck) incl. v2 and
pre-M3 checkpoints, ("rule", agent, deck), ("generic", deck), ("random", deck).

M7.4b additions:
- per-game LEARNER deck sampling from a population file (multi-deck self-play — the
  field-diversity fix; REQUIRED for encoders-v2 checkpoints, whose deck-context
  features need the deck list);
- encoders-v2 checkpoints (embedding.weight in the state dict) are auto-detected:
  decisions are encoded with encode_state_v2/encode_option_v2 and shards gain
  state_ids / option_ids / deck_idx columns (rl/ppo.py trains OptionScorerV2 on them);
- optional potential-based race shaping (docs/M7-plan.md §3b): reward +=
  coef * (phi_t - phi_{t-1}) with phi = the race-delta feature, so SETUP actions get
  credit without corrupting the optimal policy (the shaping telescopes out of returns).

Shard layout = BC fields + PPO fields (actions/logprobs/values/rewards/players
[+ state_ids/option_ids/deck_idx for v2]); see rl/ppo.py load_shards. `players` is
always the learning seat for that game.

Usage:  python -m rl.collector --games 400 --workers 4 --checkpoint checkpoints/bc_v1.pt
"""
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "ppo"
DECK_DIR = ROOT / "decks"
PRIZE_SHAPING = 0.1
LEARN_DECK = "kyogre"           # the deck the learning policy pilots (our champion base)


def _deck(name: str) -> list[int]:
    return [int(x) for x in (DECK_DIR / f"{name}.csv").read_text().split() if x.strip()]


def default_pool(checkpoint: str, learn_deck: str = LEARN_DECK):
    """Build (specs, weights): current self (mirror) + rule experts + the generic
    pilot on the known decks (M7.4b field diversity) + the LIVE solver ship agent
    + random + recent selves.

    M7.5 attempt-2 rebalance: attempt 1 spent 50% of games on mirror+random,
    optimized the mirror, and REGRESSED vs every rules pilot (G3 0.44->0.145 —
    docs/M7.md 2026-07-12). You become what you train against: the pool now
    majority-weights the rule-based opponents, with the ship agent heaviest."""
    specs = [("model", str(checkpoint), learn_deck),   # mirror vs current policy
             ("rule", "lucario", "lucario"),           # strong fighting deck
             ("rule", "iono", "iono"),                 # lightning deck
             ("generic", "lucario"),                   # deck-agnostic pilot, strong deck
             ("generic", "iono"),                      # ... and the lightning deck
             ("solver", "lucario"),                    # the LIVE ship agent — the bar
             ("random", "kyogre")]                     # weak/varied baseline
    weights = [0.2, 0.15, 0.1, 0.15, 0.1, 0.25, 0.05]
    past = sorted((ROOT / "checkpoints").glob("ppo_it*.pt"))[-3:]   # recent promoted selves
    for p in past:
        specs.append(("model", str(p), learn_deck))
        weights.append(0.1 / len(past))
    w = np.array(weights, dtype=np.float64)
    return specs, (w / w.sum()).tolist()


def _dev_potential(obs) -> float:
    """M8.3 leg-B shaping potential: the M8.1 dev-tier terms as a bounded
    state potential (the killed tier's leaf math migrates here per plan —
    docs/M8.md 2026-07-13). phi in ~[0, 1.65]; taxonomy weights: race progress
    dominates, then attack-readiness, evolution, bench insurance."""
    from rl.turn_solver import _dev_facts
    st = obs.current
    me = st.players[st.yourIndex]
    op = st.players[1 - st.yourIndex]
    op_active = op.active[0] if op.active and op.active[0] is not None else None
    race, ready, bench, evos, _hand = _dev_facts(me, op_active)
    return ((10.0 - race) / 10.0
            + 0.3 * min(ready, 1)
            + 0.2 * min(evos, 2) / 2.0
            + 0.15 * min(bench, 3) / 3.0)


def _is_over_attach(obs, option) -> bool:
    """M20: the sampled action attaches energy to a Pokémon whose charged-best
    attack is ALREADY paid (rl/combat._turns_to_ready == 0 — same primitive as
    the [over-attach] postmortem flag, re-expressed against live obs objects).
    This is the defect the per-step penalty docks."""
    from cg.api import AreaType, OptionType
    from rl.combat import _turns_to_ready

    if int(option.type) != int(OptionType.ATTACH) or option.inPlayArea is None:
        return False
    me = obs.current.players[obs.current.yourIndex]
    zone = {int(AreaType.ACTIVE): me.active,
            int(AreaType.BENCH): me.bench}.get(int(option.inPlayArea))
    idx = option.inPlayIndex
    if zone is None or idx is None or idx >= len(zone) or zone[idx] is None:
        return False
    target = zone[idx]
    opp = obs.current.players[1 - obs.current.yourIndex]
    opp_active = opp.active[0] if opp.active and opp.active[0] is not None else None
    board_ids = {p.id for p in list(me.active or []) + list(me.bench or [])
                 if p is not None}
    return _turns_to_ready(target, opp_active, board_ids) == 0


def parse_pool(items: list[str], checkpoint: str,
               learn_deck: str = LEARN_DECK) -> tuple[list, list]:
    """M21 curriculum: parse "spec=weight" strings into (specs, weights).

    Spec syntax is rl/matchrunner.parse_spec, plus two tokens:
      mirror=w  -> the CURRENT checkpoint piloting learn_deck (re-resolve each
                   iteration so 'mirror' tracks the live policy)
      past=w    -> the recent promoted selves (default_pool tail), weight split
    Weights are normalized; deck csv paths work anywhere a deck name does
    (rl/matchrunner.resolve_deck)."""
    from rl.matchrunner import parse_spec
    specs, weights = [], []
    for item in items:
        spec_s, _, w_s = item.partition("=")
        w = float(w_s) if w_s else 1.0
        if spec_s == "mirror":
            specs.append(("model", str(checkpoint), learn_deck))
            weights.append(w)
        elif spec_s == "past":
            past = sorted(Path(ROOT / "checkpoints").glob("ppo_*it*.pt"))[-3:]
            if not past:
                continue                     # no promoted selves yet — drop
            for p in past:
                specs.append(("model", str(p), learn_deck))
                weights.append(w / len(past))
        else:
            specs.append(parse_spec(spec_s))
            weights.append(w)
    specs, weights = _dedupe_pool(specs, weights)
    total = sum(weights)
    return specs, [w / total for w in weights]


def _pool_identity(spec):
    """A spec's identity as the sampler actually experiences it: (pilot, RESOLVED deck).

    Deck-slot position mirrors rl.matchrunner.spec_deck so the two cannot drift.
    """
    from rl.matchrunner import resolve_deck
    i = 2 if spec[0] in ("rule", "model", "ext", "rank", "vsolver") else 1
    if i >= len(spec):
        return spec
    try:
        deck_key = tuple(sorted(resolve_deck(spec[i])))
    except (OSError, ValueError, TypeError):
        deck_key = ("unresolved", str(spec[i]))
    return tuple(spec[:i]) + (deck_key,) + tuple(spec[i + 1:])


def _dedupe_pool(specs: list, weights: list) -> tuple[list, list]:
    """Merge pool entries that resolve to the SAME (pilot, deck), summing weights.

    M22b: B3's mixture declared `solver:.../deck_20dcd3130bc0.csv=0.30` and
    `solver:lucario=0.15` as two opponents. They are one — that csv is
    byte-identical to decks/lucario.csv (md5 aef8da62…). So 45% of training was a
    single opponent while the config read 0.30 and 0.15, and the mirror gate
    measures exactly that opponent.

    Merging does NOT change the sampling distribution (two entries at 0.30 and
    0.15 already put 0.45 of the mass there). What it changes is legibility: the
    warning below is the only thing that would have surfaced the collapse.
    Spec STRINGS differ while resolved decks match, so string-keyed dedupe
    cannot catch this.
    """
    order: list = []
    merged: dict = {}
    for spec, w in zip(specs, weights):
        key = _pool_identity(spec)
        if key in merged:
            merged[key] = (merged[key][0], merged[key][1] + w, merged[key][2] + [spec])
        else:
            merged[key] = (spec, w, [spec])
            order.append(key)
    out_specs, out_weights = [], []
    for key in order:
        spec, w, dupes = merged[key]
        if len(dupes) > 1:
            names = ", ".join(str(d) for d in dupes)
            print(f"WARNING: opponent pool declared {len(dupes)} entries that resolve "
                  f"to the SAME (pilot, deck) — merged to weight {w:.3f}: {names}",
                  flush=True)
        out_specs.append(spec)
        out_weights.append(w)
    return out_specs, out_weights


_PLAN_GUST_IDX = 18   # encode_plan: v[18] = needs_gust (rl/plan.py)
_BOSS_ID = 1182       # Boss's Orders (rl/plan.GUST_IDS)


def _play_worker(args: tuple) -> tuple:
    (worker_id, n_games, checkpoint, learn_deck_name, learn_decks, specs, weights,
     out_dir, seed, race_shaping, shaping, defect_penalty,
     plan_tau, plan_dirichlet, plan_ppo, gust_boost) = args

    import random
    import torch
    from cg.api import SelectContext, to_observation_class
    from cg.game import battle_start, battle_select, battle_finish
    from rl.encoders import (_race_features, encode_context, encode_option,
                             encode_option_v2, encode_state, encode_state_v2)
    from rl.matchrunner import make_pilot
    from rl.policy import OptionScorer, OptionScorerV2

    rng = random.Random(seed)
    torch.manual_seed(seed)

    sd = torch.load(checkpoint, map_location="cpu")
    is_v4 = "enc_ver" in sd                 # M21: explicit version buffer
    is_v3 = is_v4 or "plan_enc.0.weight" in sd
    is_v2 = not is_v3 and "embedding.weight" in sd
    if is_v2 and not learn_decks:
        raise ValueError("encoders-v2 checkpoints need a deck population "
                         "(collect(..., decks_file=...)): the deck-context "
                         "features require the learner's deck list")
    if is_v3:
        # M20: plan-conditioned v3 learner — arch sniff + encoders exactly as
        # rl/matchrunner's model pilot (the serve-time twin). M21: v4 nets add
        # the encoder-v4 block + a per-game OppMemory.
        from rl.encoders import (EMBED_DIM, N_CONTEXTS, OPTION_V3_DIM,
                                 STATE_V2_DIM, V4_EXTRA_DIM, encode_ctx_v4,
                                 encode_option_v2_legacy, encode_state_v3)
        from rl.plan import PLAN_DIM, encode_plan, enumerate_plans
        from rl.policy import OptionScorerV3, option_dim_of
        extra = V4_EXTRA_DIM if is_v4 else 0
        n_ids = (sd["state_enc.0.weight"].shape[1] - STATE_V2_DIM - N_CONTEXTS
                 - extra - PLAN_DIM) // EMBED_DIM
        opt_dim = option_dim_of(sd)
        learner = OptionScorerV3(n_state_ids=n_ids, option_dim=opt_dim,
                                 extra_dim=extra)
        enc_state = encode_state_v3 if n_ids > 12 else encode_state_v2
        enc_opt = (encode_option_v2 if opt_dim >= OPTION_V3_DIM
                   else encode_option_v2_legacy)
    else:
        learner = OptionScorerV2() if is_v2 else OptionScorer()
    learner.load_state_dict(sd)
    learner.eval()
    fixed_deck = _deck(learn_deck_name)
    pstate = {"key": None, "vec": None}     # v3 turn-scoped plan (reset per game)
    memory = None
    if is_v4:
        from rl.memory import OppMemory
        memory = OppMemory()                # learner-seat opponent memory

    np_rng = np.random.default_rng(seed)
    nonlocal_counts = {"boosts": 0}    # gust-boosted option samples (M21 B3)

    def run_learner(obs, deck):
        """Sampled decision: (picks, action, logprob, value, sc, sids, opts,
        oids, plan, plan_rec). plan_rec (M21 plan-PPO) is set only at a turn's
        first MAIN when plan_ppo is on: dict(cands, action, logprob) — the
        logprob is under log_softmax(logits) (the tau=1 policy), NOT the
        tau/dirichlet behavior distribution, so the surrogate ratio starts at
        1 and clipping absorbs the modest exploration off-policy-ness
        (standard PPO-with-temperature practice)."""
        plan = None
        plan_rec = None
        if is_v3:
            if is_v4:
                memory.observe(obs)          # once per own prompt, pre-encode
                sc, sids = encode_ctx_v4(obs, deck, memory)
            else:
                num, sids = enc_state(obs.current, deck)
                sc = np.concatenate([num, encode_context(obs.select.context)]).astype(np.float32)
            key = (obs.current.turn, obs.current.yourIndex)
            if obs.select.context == SelectContext.MAIN and pstate["key"] != key:
                # Plan once at the turn's first MAIN, hold for submenus.
                # M21: --plan-tau samples the plan (the aligned exploration
                # axis — per-prompt option noise makes turns incoherent, the
                # EI lesson); --plan-ppo additionally records the decision so
                # ppo_update can train the plan head.
                cands = enumerate_plans(obs)
                mat = np.stack([encode_plan(c) for c in cands]).astype(np.float32)
                if plan_tau > 0 or plan_ppo:
                    with torch.no_grad():
                        pl = learner.plan_logits(
                            torch.from_numpy(sc).unsqueeze(0),
                            torch.from_numpy(sids.astype(np.int64)).unsqueeze(0),
                            torch.from_numpy(mat).unsqueeze(0)).squeeze(0)
                    plogp = torch.log_softmax(pl, dim=-1)
                    if plan_tau > 0:
                        probs = torch.softmax(pl / plan_tau, dim=-1)
                        if plan_dirichlet > 0 and len(mat) > 1:
                            noise = np_rng.dirichlet(
                                [0.5] * len(mat)).astype(np.float32)
                            probs = ((1 - plan_dirichlet) * probs
                                     + plan_dirichlet * torch.from_numpy(noise))
                        idx = int(torch.multinomial(probs, 1).item())
                    else:
                        idx = int(torch.argmax(pl).item())
                    if plan_ppo:
                        plan_rec = dict(cands=mat, action=idx,
                                        logprob=float(plogp[idx]))
                else:
                    idx = learner.act_plan(sc, sids, mat)
                pstate.update(key=key, vec=mat[idx].copy())
            plan = (pstate["vec"] if pstate["key"] == key and pstate["vec"] is not None
                    else np.zeros(PLAN_DIM, dtype=np.float32))
            pairs = [enc_opt(o, obs) for o in obs.select.option]
            opts = np.stack([p[0] for p in pairs]).astype(np.float32)
            oids = np.stack([p[1] for p in pairs])
            with torch.no_grad():
                logits, value = learner(
                    torch.from_numpy(sc).unsqueeze(0),
                    torch.from_numpy(plan).unsqueeze(0),
                    torch.from_numpy(sids.astype(np.int64)).unsqueeze(0),
                    torch.from_numpy(opts).unsqueeze(0),
                    torch.from_numpy(oids.astype(np.int64)).unsqueeze(0))
        elif is_v2:
            num, sids = encode_state_v2(obs.current, deck)
            sc = np.concatenate([num, encode_context(obs.select.context)]).astype(np.float32)
            pairs = [encode_option_v2(o, obs) for o in obs.select.option]
            opts = np.stack([p[0] for p in pairs]).astype(np.float32)
            oids = np.stack([p[1] for p in pairs])
            with torch.no_grad():
                logits, value = learner(
                    torch.from_numpy(sc).unsqueeze(0),
                    torch.from_numpy(sids.astype(np.int64)).unsqueeze(0),
                    torch.from_numpy(opts).unsqueeze(0),
                    torch.from_numpy(oids.astype(np.int64)).unsqueeze(0))
        else:
            sids = oids = None
            sc = np.concatenate([encode_state(obs.current),
                                 encode_context(obs.select.context)]).astype(np.float32)
            opts = np.stack([encode_option(o, obs) for o in obs.select.option]).astype(np.float32)
            with torch.no_grad():
                logits, value = learner(torch.from_numpy(sc).unsqueeze(0),
                                        torch.from_numpy(opts).unsqueeze(0))
        logits = logits.squeeze(0)
        probs = torch.softmax(logits, dim=0)
        sample_probs = probs
        if (gust_boost > 0 and plan is not None
                and plan[_PLAN_GUST_IDX] > 0.5 and oids is not None):
            # M21 B3 deadlock-breaker: the committed plan needs a gust but the
            # option head has ~zero Boss-play mass (measured 0/793) — no
            # reward signal can ever reach either head. Guided exploration:
            # mix a fixed boost onto the Boss PLAY option. old_lp stays the
            # POLICY logprob (the plan-tau convention): ratio starts at 1 and
            # the clip bounds each update; the boost only changes WHICH
            # decisions land in the batch, letting advantage decide if gust
            # turns are worth keeping.
            boss = next((i for i, oid in enumerate(oids)
                         if int(oid[0]) == _BOSS_ID), None)
            if boss is not None:
                sample_probs = (1 - gust_boost) * probs
                sample_probs[boss] += gust_boost
                nonlocal_counts["boosts"] += 1
        action = int(torch.multinomial(sample_probs, 1))
        logprob = float(torch.log(probs[action] + 1e-12))
        order = torch.argsort(logits, descending=True).tolist()
        picks = ([action] + [i for i in order if i != action])[:obs.select.maxCount]
        return (picks, action, logprob, float(value), sc, sids, opts, oids,
                plan, plan_rec)

    # Build each distinct opponent once (matchrunner handles every spec kind,
    # incl. generic pilots and v2 / pre-M3 model checkpoints).
    opp_cache: dict = {}

    def get_opponent(spec):
        if spec not in opp_cache:
            opp_cache[spec] = make_pilot(spec, instance=f"cw{worker_id}_{len(opp_cache)}")
        return opp_cache[spec]

    col_names = ["states", "options", "n_options", "actions",
                 "logprobs", "values", "rewards", "game_ids", "players"]
    if is_v2 or is_v3:
        col_names += ["state_ids", "option_ids"]
    if is_v3:
        col_names += ["plans"]
    if plan_ppo:
        col_names += ["plan_cands", "n_plan_cands", "plan_actions",
                      "plan_logprobs"]
    if learn_decks:
        col_names += ["deck_idx"]
    cols: dict[str, list] = {k: [] for k in col_names}
    n_defects = 0                                   # M20: over-attach actions taken

    for game in range(n_games):
        pstate.update(key=None, vec=None)           # v3: fresh plan state per game
        if memory is not None:
            memory.reset()                          # v4: fresh opponent memory
        if learn_decks:
            deck_idx = rng.randrange(len(learn_decks))
            learn_deck = learn_decks[deck_idx]
        else:
            deck_idx, learn_deck = -1, fixed_deck
        spec = specs[rng.choices(range(len(specs)), weights=weights)[0]]
        opp_fn, opp_deck = get_opponent(spec)
        learn_seat = game % 2                          # alternate seats -> slot-fair data
        d0, d1 = (learn_deck, opp_deck) if learn_seat == 0 else (opp_deck, learn_deck)

        obs_dict, start = battle_start(d0, d1)
        if start.errorPlayer >= 0:
            raise ValueError(f"battle_start rejected deck (errorType={start.errorType})")

        rows, prev_delta, prev_phi = [], 0.0, None
        while obs_dict["current"]["result"] < 0:
            seat = obs_dict["current"]["yourIndex"]
            if seat == learn_seat:
                obs = to_observation_class(obs_dict)
                (picks, action, logprob, value, sc, sids, opts, oids, plan,
                 plan_rec) = run_learner(obs, learn_deck)
                me = obs.current.players[seat]; op = obs.current.players[1 - seat]
                delta = (6 - len(op.prize)) - (6 - len(me.prize))     # prizes I took - they took
                reward = PRIZE_SHAPING * (delta - prev_delta)
                prev_delta = delta
                if defect_penalty and _is_over_attach(obs, obs.select.option[action]):
                    # M20: dock the decision that wastes the attach — a dense,
                    # NON-potential term that deliberately redefines optimal
                    # play (unlike the race/dev shaping below).
                    reward -= defect_penalty
                    n_defects += 1
                if race_shaping:
                    # Potential-based setup shaping (M7-plan §3b): phi = the race
                    # delta; F = coef*(phi' - phi) telescopes out of the return,
                    # so it credits board development without changing the
                    # optimal policy.
                    phi = (_dev_potential(obs) if shaping == "dev"
                           else float(_race_features(obs.current)[7]))
                    if prev_phi is not None:
                        reward += race_shaping * (phi - prev_phi)
                    prev_phi = phi
                rows.append(dict(state=sc, state_ids=sids, opts=opts, option_ids=oids,
                                 action=action, logprob=logprob, value=value,
                                 reward=reward, plan=plan, plan_rec=plan_rec))
                obs_dict = battle_select([int(i) for i in picks])
            else:
                obs_dict = battle_select([int(i) for i in opp_fn(obs_dict)])

        result = obs_dict["current"]["result"]         # 0/1 winner, 2 draw
        battle_finish()
        if rows:                                        # terminal reward on learner's last row
            rows[-1]["reward"] += 0.0 if result == 2 else (1.0 if result == learn_seat else -1.0)

        for r in rows:
            cols["states"].append(r["state"]); cols["options"].append(r["opts"])
            cols["n_options"].append(len(r["opts"])); cols["actions"].append(r["action"])
            cols["logprobs"].append(r["logprob"]); cols["values"].append(r["value"])
            cols["rewards"].append(r["reward"]); cols["game_ids"].append(game)
            cols["players"].append(learn_seat)
            if is_v2 or is_v3:
                cols["state_ids"].append(r["state_ids"])
                cols["option_ids"].append(r["option_ids"])
            if is_v3:
                cols["plans"].append(r["plan"])
            if plan_ppo:
                # M21 plan-PPO: the plan decision rides on its first-MAIN
                # option row (same state, no reward in between) — rows without
                # a plan decision carry 0 candidates.
                rec = r["plan_rec"]
                if rec is not None:
                    cols["plan_cands"].append(rec["cands"])
                    cols["n_plan_cands"].append(len(rec["cands"]))
                    cols["plan_actions"].append(rec["action"])
                    cols["plan_logprobs"].append(rec["logprob"])
                else:
                    cols["plan_cands"].append(
                        np.zeros((0, len(r["plan"])), np.float32))
                    cols["n_plan_cands"].append(0)
                    cols["plan_actions"].append(-1)
                    cols["plan_logprobs"].append(0.0)
            if learn_decks:
                cols["deck_idx"].append(deck_idx)

    arrays = dict(
        states=np.stack(cols["states"]),
        options=np.concatenate(cols["options"]),
        n_options=np.array(cols["n_options"], dtype=np.int32),
        actions=np.array(cols["actions"], dtype=np.int32),
        logprobs=np.array(cols["logprobs"], dtype=np.float32),
        values=np.array(cols["values"], dtype=np.float32),
        rewards=np.array(cols["rewards"], dtype=np.float32),
        game_ids=np.array(cols["game_ids"], dtype=np.int32),
        players=np.array(cols["players"], dtype=np.int32),
    )
    if is_v2 or is_v3:
        arrays["state_ids"] = np.stack(cols["state_ids"])
        arrays["option_ids"] = np.concatenate(cols["option_ids"])
    if is_v3:
        arrays["plans"] = np.stack(cols["plans"])
    if plan_ppo:
        arrays["plan_cands"] = (np.concatenate(cols["plan_cands"])
                                if cols["plan_cands"] else
                                np.zeros((0, 27), np.float32))
        arrays["n_plan_cands"] = np.array(cols["n_plan_cands"], dtype=np.int32)
        arrays["plan_actions"] = np.array(cols["plan_actions"], dtype=np.int32)
        arrays["plan_logprobs"] = np.array(cols["plan_logprobs"], dtype=np.float32)
    if learn_decks:
        arrays["deck_idx"] = np.array(cols["deck_idx"], dtype=np.int32)
    out = Path(out_dir) / f"ppo_shard_w{worker_id:02d}.npz"
    np.savez_compressed(out, **arrays)
    return str(out), n_defects, n_games, nonlocal_counts["boosts"]


def _load_population(decks_file) -> list[list[int]]:
    import json
    from rl.matchrunner import resolve_deck
    return [resolve_deck(d) for d in json.loads(Path(decks_file).read_text())["decks"]]


def collect(n_games: int, checkpoint: str, n_workers: int = 4,
            learn_deck: str = LEARN_DECK, pool=None, out_dir: Path = OUT_DIR,
            decks_file=None, race_shaping: float = 0.0,
            shaping: str = "race", defect_penalty: float = 0.0,
            plan_tau: float = 0.0, plan_dirichlet: float = 0.0,
            plan_ppo: bool = False,
            gust_boost: float = 0.0) -> tuple[list[str], float]:
    """Collect n_games across n_workers, learning policy vs an opponent pool.

    decks_file: population.json — the learner samples a deck per game from it
    (multi-deck self-play; required for encoders-v2 checkpoints). race_shaping:
    coefficient of the potential-based setup-shaping term (0 = off).
    defect_penalty (M20): per-step reward dock for over-attach actions (0 = off;
    the defect is still counted). plan_tau/plan_dirichlet (M21): plan-level
    exploration (softmax temperature + AZ-style Dirichlet mix at the turn's
    first MAIN). plan_ppo (M21): record plan decisions (candidates, sampled
    index, tau=1 logprob) so rl/ppo.py can train the plan head.
    Returns (shard paths, defect rate per game)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    specs, weights = pool if pool is not None else default_pool(checkpoint, learn_deck)
    learn_decks = _load_population(decks_file) if decks_file else None
    per = [n_games // n_workers + (1 if i < n_games % n_workers else 0) for i in range(n_workers)]
    jobs = [(i, per[i], checkpoint, learn_deck, learn_decks, specs, weights,
             str(out_dir), 1000 + i, race_shaping, shaping, defect_penalty,
             plan_tau, plan_dirichlet, plan_ppo, gust_boost)
            for i in range(n_workers) if per[i] > 0]

    ctx = mp.get_context("spawn")
    with ctx.Pool(len(jobs)) as pool_:
        results = pool_.map(_play_worker, jobs)
    shards = [r[0] for r in results]
    total_defects = sum(r[1] for r in results)
    total_games = sum(r[2] for r in results)
    total_boosts = sum(r[3] for r in results)
    defect_rate = total_defects / max(1, total_games)
    print(f"collect: over-attach actions {total_defects} in {total_games} games "
          f"= {defect_rate:.2f}/game"
          + (f"; gust-boosted samples {total_boosts} "
             f"= {total_boosts / max(1, total_games):.2f}/game"
             if gust_boost > 0 else ""), flush=True)
    return shards, defect_rate


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--checkpoint", type=str, default=str(ROOT / "checkpoints" / "bc_v1.pt"))
    p.add_argument("--decks", type=str, default=None,
                   help="population.json for per-game learner deck sampling")
    p.add_argument("--race-shaping", type=float, default=0.0)
    p.add_argument("--learn-deck", type=str, default=LEARN_DECK,
                   help="learner's fixed deck (M20 probe: lucario)")
    p.add_argument("--defect-penalty", type=float, default=0.0,
                   help="M20: per-step reward dock for over-attach actions")
    p.add_argument("--opponents", type=str, nargs="*", default=None,
                   help="M21 curriculum: 'spec=weight' items (+ mirror=w, "
                        "past=w tokens); default = default_pool")
    p.add_argument("--plan-tau", type=float, default=0.0,
                   help="M21: plan softmax temperature at first MAIN (0 = argmax)")
    p.add_argument("--plan-dirichlet", type=float, default=0.0,
                   help="M21: Dirichlet(0.5) mix into the plan sampling probs")
    p.add_argument("--plan-ppo", action="store_true",
                   help="M21: record plan decisions for plan-head PPO training")
    p.add_argument("--gust-boost", type=float, default=0.0,
                   help="M21 B3: guided Boss-play exploration under committed "
                        "gust plans (mix probability; 0 = off)")
    args = p.parse_args()

    pool_arg = (parse_pool(args.opponents, args.checkpoint, args.learn_deck)
                if args.opponents else None)
    import time
    t0 = time.time()
    shards, _rate = collect(args.games, args.checkpoint, args.workers,
                            learn_deck=args.learn_deck, pool=pool_arg,
                            decks_file=args.decks,
                            race_shaping=args.race_shaping,
                            defect_penalty=args.defect_penalty,
                            plan_tau=args.plan_tau,
                            plan_dirichlet=args.plan_dirichlet,
                            plan_ppo=args.plan_ppo,
                            gust_boost=args.gust_boost)
    dt = time.time() - t0
    print(f"{args.games} games in {dt:.0f}s ({3600 * args.games / dt:.0f} games/hr) -> {shards}")
