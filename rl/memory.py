"""Per-game opponent memory distilled from obs.logs (M21 encoder v4).

numpy-only on purpose — ships in the Kaggle bundle next to rl/encoders.py
(ARCHITECTURE.md §7.1). The v3 encoders drop obs.logs entirely; this module
turns that stream into a fixed-width feature block + embedding ids so the net
can see what the opponent has been doing (their charging target is their
announced next attacker) and which of their hand cards are known.

Contract (locked by the M21 logs probe, docs/M21.md 2026-07-19):
- obs.logs is PER-SEAT: every event since this seat's previous prompt,
  including 100% of the opponent's public actions (ATTACK/PLAY/ATTACH/EVOLVE).
- Sub-prompt chains re-deliver the previous window verbatim as a prefix with
  new events appended → observe() dedupes by exact prefix match against the
  previous prompt's fingerprint list.
- observe(obs) must be called EXACTLY ONCE per own prompt, before encoding;
  reset() at game start (the per-game-state precedent: _PSTATE in
  submission/main.py).
"""
from collections import deque

import numpy as np

from rl.combat import _ATK, _CARD, _attack_available
# N_MEM / N_MEM_IDS / N_MEM_ATTACH_HOT live in rl.encoders (single source of
# truth for all encoding widths); this import direction keeps the graph acyclic.
from rl.encoders import FEAT, FEAT_DIM, N_MEM, N_MEM_ATTACH_HOT, N_MEM_IDS

# LogType values (cg/api.py; ints so raw-dict logs work without the enum)
_LOG_TURN_START = 2
_LOG_TURN_END = 3
_LOG_DRAW_REVERSE = 5
_LOG_MOVE_CARD = 6
_LOG_PLAY = 10
_LOG_ATTACH = 11
_LOG_EVOLVE = 12
_LOG_ATTACK = 15
_AREA_HAND = 2  # AreaType.HAND

_N_ATTACH_HOT = N_MEM_ATTACH_HOT


def _g(ev, key):
    """Field access for both raw-dict logs (training loop) and cg.api.Log."""
    return ev.get(key) if isinstance(ev, dict) else getattr(ev, key, None)


def _fingerprint(ev) -> tuple:
    return (_g(ev, "type"), _g(ev, "playerIndex"), _g(ev, "serial"),
            _g(ev, "serialTarget"), _g(ev, "cardId"), _g(ev, "attackId"),
            _g(ev, "value"), _g(ev, "head"))


class OppMemory:
    """Accumulates opponent-visible history for one seat of one game."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._prev_fps: list[tuple] = []
        self._known_hand: dict[int, int] = {}     # serial -> card id
        self._played: deque[int] = deque(maxlen=N_MEM_IDS - 1)  # most recent LAST
        self._last_attacker_id = 0
        self._last_attack_id: int | None = None
        self._attacks_total = 0
        self._attacked_this_turn = False
        self._attacked_last_turn = False
        self._passed_last_turn = False
        self._last_attach_target: int | None = None  # serialTarget
        self._draw_reverse = 0
        self._opp_turns = 0

    # -- ingestion ----------------------------------------------------------
    def observe(self, obs) -> None:
        """Consume obs.logs once per own prompt (prefix-dedupe, see module doc)."""
        current = obs["current"] if isinstance(obs, dict) else obs.current
        your_index = (current["yourIndex"] if isinstance(current, dict)
                      else current.yourIndex)
        logs = (obs.get("logs") if isinstance(obs, dict)
                else getattr(obs, "logs", None)) or []
        fps = [_fingerprint(e) for e in logs]
        if self._prev_fps and fps[:len(self._prev_fps)] == self._prev_fps:
            new = logs[len(self._prev_fps):]
        else:
            new = logs
        self._prev_fps = fps
        opp = 1 - your_index
        for ev in new:
            self._consume(ev, opp)

    def _consume(self, ev, opp: int) -> None:
        etype, player = _g(ev, "type"), _g(ev, "playerIndex")
        if etype == _LOG_MOVE_CARD and player == opp:
            serial, card_id = _g(ev, "serial"), _g(ev, "cardId")
            if _g(ev, "toArea") == _AREA_HAND and card_id:
                self._known_hand[serial] = card_id     # revealed into opp hand
            elif serial in self._known_hand:
                self._known_hand.pop(serial)           # left the hand face-up
            return
        if player != opp:
            return
        if etype == _LOG_DRAW_REVERSE:
            self._draw_reverse += 1
        elif etype == _LOG_TURN_START:
            self._opp_turns += 1
            self._attacked_this_turn = False
        elif etype == _LOG_TURN_END:
            self._attacked_last_turn = self._attacked_this_turn
            self._passed_last_turn = not self._attacked_this_turn
        elif etype == _LOG_ATTACK:
            self._attacks_total += 1
            self._attacked_this_turn = True
            self._last_attacker_id = _g(ev, "cardId") or 0
            self._last_attack_id = _g(ev, "attackId")
        elif etype in (_LOG_PLAY, _LOG_ATTACH, _LOG_EVOLVE):
            card_id = _g(ev, "cardId")
            if card_id:
                self._played.append(card_id)
            self._known_hand.pop(_g(ev, "serial"), None)
            if etype == _LOG_ATTACH:
                self._last_attach_target = _g(ev, "serialTarget")

    # -- output -------------------------------------------------------------
    def features(self, state) -> np.ndarray:
        """(N_MEM,) f32: [last-attack 3 | attach-target one-hot 7 | flags 3 |
        known-hand count 1 | extra-draw intensity 1 | known-hand FEAT pool 36]."""
        v = np.zeros(N_MEM, dtype=np.float32)
        me = state.players[state.yourIndex]
        op = state.players[1 - state.yourIndex]

        # last opp attack: printed dmg / cost, effective dmg vs MY active —
        # weakness math mirrors rl.encoders._attack_extra with roles flipped.
        aid = self._last_attack_id
        if aid is not None and aid in _ATK:
            dmg, cost = _ATK[aid]
            v[0] = dmg / 300.0
            v[1] = len(cost) / 5.0
            my_act = me.active[0] if me.active and me.active[0] is not None else None
            if my_act is not None and dmg > 0:
                op_board = {p.id for p in list(op.active or []) + list(op.bench or [])
                            if p is not None}
                if _attack_available(aid, op_board):
                    t_weak, t_res, _, _, _ = _CARD.get(my_act.id, (None, None, 0, [], 1))
                    atk_type = _CARD.get(self._last_attacker_id, (None, None, 0, [], 1))[2]
                    eff = dmg
                    if t_weak is not None and int(t_weak) == atk_type:
                        eff *= 2
                    elif t_res is not None and int(t_res) == atk_type:
                        eff = max(0, eff - 30)
                    v[2] = eff / 300.0

        # where the opponent last attached: their announced next attacker
        base = 3
        hot = 0                                    # none / unresolved
        if self._last_attach_target is not None:
            op_act = op.active[0] if op.active and op.active[0] is not None else None
            if op_act is not None and op_act.serial == self._last_attach_target:
                hot = 1
            else:
                for i, b in enumerate(list(op.bench or [])[:5]):
                    if b is not None and b.serial == self._last_attach_target:
                        hot = 2 + i
                        break
        v[base + hot] = 1.0
        base += _N_ATTACH_HOT

        v[base] = float(self._attacked_last_turn)
        v[base + 1] = float(self._passed_last_turn)
        v[base + 2] = min(self._attacks_total, 10) / 10.0
        base += 3
        v[base] = min(len(self._known_hand), 10) / 10.0
        v[base + 1] = max(0, self._draw_reverse - self._opp_turns) / 10.0
        base += 2
        if self._known_hand:
            v[base:base + FEAT_DIM] = FEAT[list(self._known_hand.values())].sum(axis=0)
        return v

    def ids(self) -> np.ndarray:
        """(N_MEM_IDS,) i32: last-4 opp played card ids (oldest→newest,
        0-padded at the front) + last opp attacker card id."""
        played = list(self._played)
        played = [0] * (N_MEM_IDS - 1 - len(played)) + played
        return np.array(played + [self._last_attacker_id], dtype=np.int32)
