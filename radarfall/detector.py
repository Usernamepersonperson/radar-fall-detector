"""Streaming fall detector: ML score and/or rule, confirmation window, cancel.

States: IDLE -> CONFIRMING (after a candidate) -> ALARMED (latched until the person
is upright again or leaves) -> IDLE.  Events returned from ``push`` / ``cancel``:
``fall_suspected`` (local prompt, "press to cancel"), ``fall_alarm`` (publish),
``fall_cleared`` (recovered before the alarm), ``fall_cancelled`` (user cancel).
"""

from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np

from .config import DETECTOR, RADAR, DetectorConfig
from .dsp import TI
from .features import MED, N_FEAT, speed_env, window_features

IDLE, CONFIRMING, ALARMED = "idle", "confirming", "alarmed"


class FallDetector:
    def __init__(self, scorer: Callable[[np.ndarray], np.ndarray] | None = None,
                 cfg: DetectorConfig = DETECTOR, mode: str = "ml"):
        if mode not in ("ml", "rule", "both"):
            raise ValueError("mode must be ml, rule or both")
        if mode != "rule" and scorer is None:
            raise ValueError("ml/both mode needs a scorer")
        self.scorer, self.cfg, self.mode = scorer, cfg, mode
        fps = RADAR.fps
        self.fps = fps
        self.win = int(cfg.window_s * fps)
        self.hop = max(1, int(cfg.hop_s * fps))
        self.cap = self.win + int(2 * fps)
        self.tr: deque = deque(maxlen=self.cap)
        self.dp: deque = deque(maxlen=self.cap)
        self.zs: deque = deque(maxlen=self.cap)
        self._zlast = 0.9
        self._zraw: deque = deque(maxlen=MED)
        self.i = 0
        self.state = IDLE
        self.t_cand = 0.0
        self.source = ""
        self.score = 0.0
        self.t_alarm = -1e9
        self.up_frames = 0
        self.last_score = 0.0

    @property
    def t(self) -> float:
        return self.i / self.fps

    # --- public API -------------------------------------------------------
    def push(self, track_row: np.ndarray, dop_row: np.ndarray) -> list[dict]:
        ev: list[dict] = []
        z = track_row[TI["z"]]
        if track_row[TI["present"]] > 0.5 and np.isfinite(z):
            self._zlast = float(z)
        self._zraw.append(self._zlast)
        zs = float(np.median(self._zraw)) if len(self._zraw) == MED else self._zlast
        self.tr.append(track_row)
        self.dp.append(dop_row)
        self.zs.append(zs)
        self.i += 1

        up = zs >= self.cfg.z_recover or track_row[TI["present"]] < 0.5
        self.up_frames = self.up_frames + 1 if up else 0

        if self.state == IDLE:
            if self.i % self.hop == 0 and len(self.tr) >= self.win:
                src = self._candidate()
                if src:
                    self.state, self.t_cand, self.source = CONFIRMING, self.t, src
                    ev.append(self._event("fall_suspected"))
        elif self.state == CONFIRMING:
            if self.up_frames >= self.fps:
                self.state = IDLE
                ev.append(self._event("fall_cleared"))
            elif self.t - self.t_cand >= self.cfg.confirm_s:
                if self._confirm():
                    self.state, self.t_alarm = ALARMED, self.t
                    ev.append(self._event("fall_alarm"))
                else:
                    self.state = IDLE
                    ev.append(self._event("fall_cleared"))
        elif self.state == ALARMED and self.up_frames >= 2 * self.fps:
            self.state = IDLE
        return ev

    def cancel(self) -> list[dict]:
        """User/carer cancel: valid while confirming or within the grace period."""
        if self.state == CONFIRMING:
            ev = self._event("fall_cancelled")
            self.state = IDLE
            return [ev]
        if self.state == ALARMED and self.t - self.t_alarm <= self.cfg.cancel_grace_s:
            return [self._event("fall_cancelled")]  # stays latched: no re-alarm
        return []

    # --- internals ----------------------------------------------------------
    def _arrays(self):
        tr = np.asarray(self.tr)
        return tr, np.asarray(self.dp), np.asarray(self.zs, np.float32)

    def _candidate(self) -> str:
        tr, dp, zs = self._arrays()
        src = ""
        if self.mode in ("rule", "both") and self._rule(tr, zs):
            src = "rule"
        if not src and self.mode in ("ml", "both"):
            x = window_features(tr, dp, zs, len(tr), self.win)
            assert x.shape[0] == N_FEAT
            self.last_score = self.score = float(self.scorer(x)[0])
            if self.score >= self.cfg.ml_threshold:
                src = "ml"
        return src

    def _rule(self, tr, zs) -> bool:
        """Fast descent (height proxy + Doppler speed), ending low."""
        c, fps = self.cfg, self.fps
        n = int(c.rule_span_s * fps)
        z = zs[-n:]
        drop = z.max() - z[-1]
        peak = speed_env(tr[-n:]).max()
        return bool(drop >= c.rule_drop_m and z[-1] < c.z_low + 0.1
                    and peak >= c.rule_speed_ms)

    def _confirm(self) -> bool:
        c, fps = self.cfg, self.fps
        tr, _, zs = self._arrays()
        n = int(3 * fps)
        low = float(np.median(zs[-n:])) < c.z_low
        mdb = tr[-n:, TI["m_db"]]
        still = float((mdb < 2.0).mean())
        self._still = still
        if self.source == "rule":
            return low and still >= c.rule_still_frac
        return low

    def _event(self, name: str) -> dict:
        tr, _, zs = self._arrays()
        n = int(2 * self.fps)
        return {
            "event": name,
            "t": round(self.t, 2),
            "source": self.source,
            "score": round(self.score, 3),
            "z_end_m": round(float(np.median(zs[-n:])), 2),
            "peak_speed_ms": round(float(speed_env(tr[-int(8 * self.fps):]).max()), 2),
            "still_fraction": round(float((tr[-n:, TI["m_db"]] < 2.0).mean()), 2),
        }


def run_stream(det: FallDetector, track: np.ndarray, dop: np.ndarray) -> list[dict]:
    """Feed a whole recording through a detector; returns all events."""
    out = []
    for r, d in zip(track, dop):
        out.extend(det.push(r, d))
    return out


__all__ = ["FallDetector", "run_stream", "IDLE", "CONFIRMING", "ALARMED"]
