"""M40 S2 / E0 — does the value net out-rank the outcome proxy?

This is S2's first step and its gate: docs/M40-plan.md §2 S2.1 pre-registers
"it must out-rank the outcome proxy on the loss families, or the branch dies on
day two." Everything downstream — advantage-filtered BC, ROIDA loser replays —
is blocked on it.

E0 IS AN EVALUATION JOB, NOT A TRAINING ONE. The plan reads as though a value
net has to be built; it does not. `rl/plan_iter.py` adds
`0.5 * huber_loss(value, results)` to EVERY BC train, unweighted and including
loser rows (apply_outcome_weights never touches the value target), so every net
in the campaign already ships a value head trained exactly the way BACKLOG
sweep #6 asks for. What has never existed is a measurement of whether that head
is any good. Hence this.

WHAT "OUT-RANK THE OUTCOME PROXY" HAS TO MEAN. The outcome proxy assigns one
number to a whole game: every row of a won game gets +1, every row of a lost
game -1 (rl/plan_iter.py apply_outcome_weights). So it is CONSTANT WITHIN A
GAME by construction. A value net can only beat it by carrying information the
proxy cannot represent — that is, by varying WITHIN a game in a way that
tracks who is actually winning. Three measurements, in increasing strictness:

  1. AUC of V(s) for the eventual game outcome, split BY GAME so no game
     contributes to both sides. Tests whether V has any signal at all.
  2. Matched-pair accuracy in the rl/setup_value.py sense: a state from a won
     game against one from a lost game at the SAME turn. This is the
     campaign's existing instrument with its existing bars (GATE_V0 = 0.62,
     kill < 0.58) and is the number the pre-registration is written against.
  3. WITHIN-GAME discrimination: the share of V's variance that is within
     games rather than between them. This is the one that actually decides
     E0, because between-game variance is precisely what the outcome proxy
     already has. A value head that separates won from lost games perfectly
     but is flat inside each game adds NOTHING to advantage estimation.

Usage:
    uv run python scripts/m40_e0_value.py --checkpoint checkpoints/m39_retain_b.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The loss families the pre-registration names, plus the champion harvest as a
# reference. Family is known BY DIRECTORY here (each was collected against one
# bed), which is the only place in the repo where per-row opponent family is
# recoverable without re-deriving it from replays.
CORPORA = {
    "wall  (loss family)": "data/bc_m39_br_wall",
    "grim  (loss family)": "data/bc_m39_br_grim",
    "arch  (loss family)": "data/bc_m39_br_archaludon",
    "champion harvest": "data/bc_m38_w9294",
}
GATE_V0 = 0.62      # rl/setup_value.py's pre-registered bar
KILL_V0 = 0.58      # ...and its kill


def load_corpus(d: Path):
    st, ids, gid, res = [], [], [], []
    for f in sorted(d.glob("*.npz")):
        z = np.load(f)
        st.append(z["states"])
        ids.append(z["state_ids"])
        gid.append(z["game_ids"])
        res.append(z["results"])
    if not st:
        return None
    w = max(a.shape[1] for a in st)
    st = [a if a.shape[1] == w else np.pad(a, ((0, 0), (0, w - a.shape[1])))
          for a in st]
    iw = max(a.shape[1] for a in ids)
    ids = [a if a.shape[1] == iw else np.pad(a, ((0, 0), (0, iw - a.shape[1])))
           for a in ids]
    return (np.concatenate(st), np.concatenate(ids),
            np.concatenate(gid), np.concatenate(res))


def values(net, states, state_ids, n_ids, batch=512):
    """V(s) for every row.

    The state_ids MUST be the corpus's own: the trunk concatenates their
    embeddings onto state_ctx (rl/policy.py _trunk), so feeding zeros would
    evaluate the head on an input no encoder can emit — the same class of
    defect G-14's 6d check exists to catch, and the first version of this
    script did exactly that.
    """
    import torch
    out = []
    zeros_plan = torch.zeros(1, net.plan_dim)
    ids_all = state_ids[:, :n_ids] if state_ids.shape[1] >= n_ids else np.pad(
        state_ids, ((0, 0), (0, n_ids - state_ids.shape[1])))
    for i in range(0, len(states), batch):
        s = torch.from_numpy(states[i:i + batch].astype(np.float32))
        n = s.shape[0]
        ids = torch.from_numpy(ids_all[i:i + batch].astype(np.int64))
        with torch.no_grad():
            v = net.value_head(net._trunk(s, zeros_plan.expand(n, -1), ids))
        out.append(v.squeeze(-1).numpy())
    return np.concatenate(out)


def auc(score, label) -> float:
    """Rank AUC. label in {0,1}."""
    pos, neg = score[label == 1], score[label == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def matched_pairs(v, gid, res, rng, n_pairs=20000):
    """setup_value's instrument: a state from a WON game against one from a
    LOST game, matched on within-game position so the pair is comparable."""
    won = res > 0
    lost = res < 0
    if not won.any() or not lost.any():
        return float("nan"), 0
    # within-game position, so we pair like with like rather than an opening
    # against an endgame
    pos = np.zeros(len(gid), np.float32)
    for g in np.unique(gid):
        m = gid == g
        k = m.sum()
        pos[m] = np.arange(k) / max(k - 1, 1)
    wi, li = np.where(won)[0], np.where(lost)[0]
    hits = tot = 0
    for _ in range(n_pairs):
        a = wi[rng.integers(len(wi))]
        # nearest-position loser from a bounded random probe (cheap match)
        cand = li[rng.integers(len(li), size=8)]
        b = cand[np.argmin(np.abs(pos[cand] - pos[a]))]
        if v[a] == v[b]:
            continue
        hits += int(v[a] > v[b])
        tot += 1
    return (hits / tot if tot else float("nan")), tot


def variance_split(v, gid):
    """Share of V's variance that lives WITHIN games. The outcome proxy has
    zero within-game variance by construction, so this is the fraction of V's
    signal that is even capable of beating it."""
    tot = float(np.var(v))
    if tot <= 0:
        return 0.0, 0.0
    within = 0.0
    n = 0
    for g in np.unique(gid):
        m = gid == g
        if m.sum() < 2:
            continue
        within += float(np.var(v[m])) * m.sum()
        n += m.sum()
    within /= max(n, 1)
    return within / tot, tot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/m39_retain_b.pt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    from rl.policy import OptionScorerV3, option_dim_of
    sd = torch.load(ROOT / a.checkpoint, map_location="cpu")
    n_ids = 25 if "enc_ver" in sd else 20
    net = OptionScorerV3(n_state_ids=n_ids, option_dim=option_dim_of(sd))
    net.load_state_dict(sd)
    net.eval()
    rng = np.random.default_rng(a.seed)

    print(f"E0 — value head of {a.checkpoint}")
    print("outcome proxy = the game result, CONSTANT within a game. A value")
    print("net beats it only via WITHIN-game variation that tracks the game.\n")
    print(f"  {'corpus':<22}{'rows':>8}{'games':>7}{'AUC':>7}"
          f"{'pair-acc':>10}{'within-var':>12}")
    rows = {}
    for label, rel in CORPORA.items():
        d = ROOT / rel
        c = load_corpus(d)
        if c is None:
            print(f"  {label:<22}{'(missing)':>8}")
            continue
        states, sids, gid, res = c
        v = values(net, states, sids, n_ids)
        a_auc = auc(v, (res > 0).astype(int))
        pa, npairs = matched_pairs(v, gid, res, rng)
        wshare, _ = variance_split(v, gid)
        rows[label] = (a_auc, pa, wshare)
        print(f"  {label:<22}{len(res):>8}{len(np.unique(gid)):>7}"
              f"{a_auc:>7.3f}{pa:>10.3f}{wshare:>12.3f}")

    loss_fams = [k for k in rows if "loss family" in k]
    if not loss_fams:
        print("\nno loss-family corpora found — E0 cannot be read")
        return 2
    pa = float(np.mean([rows[k][1] for k in loss_fams]))
    wv = float(np.mean([rows[k][2] for k in loss_fams]))
    print(f"\n  LOSS-FAMILY MEAN   matched-pair {pa:.3f}   "
          f"within-game variance share {wv:.3f}")
    print(f"  bars (rl/setup_value.py): GATE_V0 {GATE_V0}   kill {KILL_V0}")

    print("\n  VERDICT")
    if pa < KILL_V0:
        print(f"  * matched-pair {pa:.3f} < kill {KILL_V0} — the value head does"
              f" not rank states at all.")
    elif pa < GATE_V0:
        print(f"  * matched-pair {pa:.3f} is between the kill and GATE_V0"
              f" {GATE_V0} — signal, but below the bar.")
    else:
        print(f"  * matched-pair {pa:.3f} CLEARS GATE_V0 {GATE_V0}.")
    if wv < 0.10:
        print(f"  * within-game variance is {wv:.3f} of the total: the head is"
              f" essentially an OUTCOME")
        print("    CLASSIFIER, near-constant inside a game. That is exactly what"
              " the outcome proxy")
        print("    already is, so it cannot supply per-decision advantages.")
    else:
        print(f"  * within-game variance is {wv:.3f} of the total — there IS"
              f" per-decision signal to")
        print("    weight with.")
    passes = pa >= GATE_V0 and wv >= 0.10
    print(f"\n  E0 {'PASSES' if passes else 'FAILS'} — the advantage-weighted"
          f" branch {'proceeds' if passes else 'DIES'}"
          f"{'' if passes else '; S2 falls back to exploration+retention filtered self-imitation (plan §4)'}.")
    return 0 if passes else 3


if __name__ == "__main__":
    raise SystemExit(main())
