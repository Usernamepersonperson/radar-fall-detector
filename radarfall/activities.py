"""Activities (daily living, fall-like non-falls, falls) built from keyframes.

Coordinates: radar on the wall at the origin looking along +x, y to the left.
``Actor`` tracks posture ("stand" | "sit" | "lie" | "floor") so only valid
transitions are produced. Every method logs ground-truth events.
"""

from __future__ import annotations

import numpy as np

from .motion import Timeline

X_RANGE = (1.6, 4.2)
MAX_GROUND_R = 4.3  # radar range is 6.4 m; mounted 1.8 m up
BED_H, SEAT_H = 0.55, 0.50

FALL_KINDS = ("forward", "backward", "side_left", "side_right", "walking_trip",
              "from_chair", "slow_slide")
ADL_FALL_LIKE = ("sit_normal", "sit_fast", "bed_lie", "bed_flop", "bend_pick",
                 "squat_pick", "floor_sit", "floor_lie", "stumble_recover")


def in_room(x: float, y: float) -> tuple[float, float]:
    x = float(np.clip(x, *X_RANGE))
    lim = x * np.tan(np.radians(50))
    y = float(np.clip(y, -lim, lim))
    r = float(np.hypot(x, y))
    if r > MAX_GROUND_R:  # keep the person (and a fall) inside the unambiguous range
        x, y = x * MAX_GROUND_R / r, y * MAX_GROUND_R / r
    return x, y


class Actor:
    def __init__(self, rng: np.random.Generator, x: float = 3.0, y: float = 0.0,
                 psi: float = 0.0):
        self.rng = rng
        self.tl = Timeline()
        self.tl.p[0][[0, 1, 3]] = [x, y, psi]
        self.posture = "stand"

    # --- helpers -------------------------------------------------------
    def _ev(self, kind, t0, t1, fall=False, t_impact=None):
        self.tl.events.append(dict(kind=kind, fall=fall, t_onset=float(t0),
                                   t_end=float(t1),
                                   t_impact=float(t_impact if t_impact else t1)))

    def _pose(self):
        return self.tl.last()

    def idle(self, dur: float, amp: float = 0.2):
        t0 = self.tl.now
        self.tl.fidget.append((t0, t0 + dur, amp if self.posture != "lie" else 0.0))
        self.tl.hold(dur)

    def absent(self, dur: float):
        t0 = self.tl.now
        self.tl.absent.append((t0, t0 + dur))
        self.tl.hold(dur)

    def turn(self, psi: float, dur: float = 0.7):
        self.tl.add(dur, psi=psi)

    def walk_to(self, tx: float, ty: float, speed: float):
        tl, p = self.tl, self._pose()
        tx, ty = in_room(tx, ty)
        dx, dy = tx - p["x"], ty - p["y"]
        dist = float(np.hypot(dx, dy))
        if dist < 0.3:
            return
        head = float(np.arctan2(dy, dx))
        # unwrap the heading so the turn is the short way round
        head = p["psi"] + (head - p["psi"] + np.pi) % (2 * np.pi) - np.pi
        self.turn(head, 0.6)
        dur = dist / speed
        t0 = tl.now
        tl.gait.append((t0, t0 + dur, 0.8 + 0.55 * speed, 1.0))
        tl.add(dur, "linear", x=tx, y=ty)

    def _back(self, d: float):
        p = self._pose()
        return p["x"] - d * np.cos(p["psi"]), p["y"] - d * np.sin(p["psi"])

    # --- fall-like, non-fall activities --------------------------------
    def sit_down(self, fast: bool = False):
        assert self.posture == "stand"
        tl = self.tl
        dur = self.rng.uniform(0.55, 0.9) if fast else self.rng.uniform(1.6, 2.6)
        t0 = tl.now
        bx, by = self._back(0.12)
        if fast:
            tl.add(dur * 0.3, Lt=0.45, zp=0.88)
            tl.add(dur * 0.7, "accel", Lt=0.08, zp=SEAT_H, ath=1.45, ak=1.45,
                   x=bx, y=by, arm=0.5)
        else:
            tl.add(dur * 0.4, Lt=0.45, zp=0.85, ath=0.5, ak=0.5)
            tl.add(dur * 0.6, "smooth", Lt=0.08, zp=SEAT_H, ath=1.45, ak=1.45,
                   x=bx, y=by, arm=0.2)
        self._ev("sit_fast" if fast else "sit_normal", t0, tl.now)
        self.posture = "sit"

    def stand_up(self):
        assert self.posture == "sit"
        tl = self.tl
        dur = self.rng.uniform(1.4, 2.4)
        px, py = self._back(-0.12)
        tl.add(dur * 0.4, Lt=0.5, zp=0.62, arm=0.3)
        tl.add(dur * 0.6, "smooth", Lt=0.0, zp=0.95, ath=0.0, ak=0.0, x=px, y=py,
               arm=0.05)
        self.posture = "stand"

    def lie_on_bed(self, flop: bool = False):
        assert self.posture == "sit"
        tl = self.tl
        dur = self.rng.uniform(0.9, 1.5) if flop else self.rng.uniform(3.0, 5.0)
        p = self._pose()
        t0 = tl.now
        tl.add(0.4, zp=BED_H + 0.05)
        tl.add(dur, "accel" if flop else "smooth", Lb=1.42, D=p["psi"] + np.pi,
               ath=0.15, ak=0.15, zp=BED_H + 0.08, arm=0.25)
        self._ev("bed_flop" if flop else "bed_lie", t0, tl.now)
        self.posture = "lie"

    def sit_up_from_bed(self):
        assert self.posture == "lie"
        tl = self.tl
        dur = self.rng.uniform(2.5, 4.0)
        tl.add(dur, "smooth", Lb=0.0, ath=1.45, ak=1.45, zp=SEAT_H, arm=0.2)
        self.posture = "sit"

    def bend_pick(self):
        assert self.posture == "stand"
        tl = self.tl
        t0 = tl.now
        tl.add(self.rng.uniform(0.9, 1.4), Lt=1.25, zp=0.82, ath=0.25, ak=0.25,
               arm=0.2)
        tl.add(self.rng.uniform(0.6, 1.5), "linear")
        tl.add(self.rng.uniform(1.0, 1.5), Lt=0.0, zp=0.95, ath=0.0, ak=0.0)
        self._ev("bend_pick", t0, tl.now)

    def squat_pick(self):
        assert self.posture == "stand"
        tl = self.tl
        t0 = tl.now
        tl.add(self.rng.uniform(1.0, 1.6), Lt=0.7, zp=0.48, ath=1.7, ak=2.3)
        tl.add(self.rng.uniform(0.8, 1.6), "linear")
        tl.add(self.rng.uniform(1.2, 1.8), Lt=0.0, zp=0.95, ath=0.0, ak=0.0)
        self._ev("squat_pick", t0, tl.now)

    def sit_on_floor(self):
        assert self.posture == "stand"
        tl = self.tl
        t0 = tl.now
        tl.add(self.rng.uniform(1.0, 1.5), Lt=0.6, zp=0.5, ath=1.7, ak=2.3)
        tl.add(self.rng.uniform(1.5, 2.5), zp=0.14, ath=1.45, ak=0.25, Lt=0.15)
        self._ev("floor_sit", t0, tl.now)
        self.posture = "floor"

    def lie_on_floor(self):
        """Deliberate lowering to lie on the floor (exercise / rest)."""
        assert self.posture == "floor"
        tl = self.tl
        p = self._pose()
        t0 = tl.now
        tl.add(self.rng.uniform(3.0, 6.0), "smooth", Lb=1.42, D=p["psi"] + np.pi,
               ath=0.1, ak=0.1, zp=0.16)
        self._ev("floor_lie", t0, tl.now)
        self.posture = "floor_lie"

    def get_up_from_floor(self):
        tl = self.tl
        if self.posture == "floor_lie":
            tl.add(self.rng.uniform(2.5, 4.0), Lb=0.0, ath=1.45, ak=0.25, zp=0.14)
        tl.add(self.rng.uniform(2.0, 3.5), Lt=0.6, zp=0.5, ath=1.7, ak=2.3)
        tl.add(self.rng.uniform(1.5, 2.2), Lt=0.0, zp=0.95, ath=0.0, ak=0.0)
        self.posture = "stand"

    def stumble_recover(self):
        assert self.posture == "stand"
        tl = self.tl
        p = self._pose()
        t0 = tl.now
        tl.add(0.01, pivot=1.0, D=p["psi"] + self.rng.choice([0, np.pi]))
        tl.add(self.rng.uniform(0.35, 0.5), "accel", Lb=self.rng.uniform(0.35, 0.6),
               arm=1.3, ath=0.3)
        tl.add(self.rng.uniform(0.6, 0.9), "smooth", Lb=0.0, arm=0.1, ath=0.0)
        tl.add(0.01, pivot=0.0)
        self._ev("stumble_recover", t0, tl.now)

    # --- falls ---------------------------------------------------------
    def fall(self, kind: str):
        """Perform a fall of the given kind; ends lying on the floor."""
        tl, rng = self.tl, self.rng
        if kind == "from_chair":
            assert self.posture == "sit"
            return self._fall_from_chair()
        if kind == "slow_slide":
            assert self.posture == "stand"
            return self._slow_slide()
        assert self.posture == "stand"
        p = self._pose()
        dirs = {"forward": 0.0, "backward": np.pi, "side_left": np.pi / 2,
                "side_right": -np.pi / 2, "walking_trip": 0.0}
        dur = rng.uniform(0.45, 0.85)
        t_start = tl.now
        tl.add(0.01, pivot=1.0, D=p["psi"] + dirs[kind])
        if kind == "walking_trip":
            sp = rng.uniform(0.7, 1.2)
            t0 = tl.now
            tl.gait.append((t0, t0 + 0.5, 1.2, 1.0))
            tl.add(0.5, "linear", Lb=0.1, x=p["x"] + 0.5 * sp * np.cos(p["psi"]),
                   y=p["y"] + 0.5 * sp * np.sin(p["psi"]))
            q = self._pose()
            tl.add(dur, "accel", Lb=1.5, arm=1.4, ak=0.3,
                   x=q["x"] + 0.4 * sp * np.cos(p["psi"]),
                   y=q["y"] + 0.4 * sp * np.sin(p["psi"]))
            t_on = tl.now - dur
        else:
            tl.add(rng.uniform(0.25, 0.5), Lb=0.12, arm=0.4)
            t_on = tl.now
            tl.add(dur, "accel", Lb=1.5, arm=1.4, ak=0.3)
        t_imp = tl.now
        tl.add(0.15, "decel", Lb=1.55)
        self._ev(f"fall_{kind}", t_on, tl.now, fall=True, t_impact=t_imp)
        self.posture = "fallen"
        del t_start

    def _fall_from_chair(self):
        tl, rng = self.tl, self.rng
        p = self._pose()
        t_on = tl.now
        d = rng.uniform(0.5, 0.9)
        tl.add(d, "accel", Lb=1.2, D=p["psi"] + rng.choice([np.pi, np.pi / 2]),
               zp=0.14, ath=0.9, ak=0.5, arm=1.2)
        t_imp = tl.now
        tl.add(0.4, "decel", Lb=1.4, zp=0.12)
        self._ev("fall_from_chair", t_on, tl.now, fall=True, t_impact=t_imp)
        self.posture = "fallen"

    def _slow_slide(self):
        tl, rng = self.tl, self.rng
        p = self._pose()
        t_on = tl.now
        tl.add(rng.uniform(1.0, 1.8), "smooth", zp=0.50, ath=1.2, ak=2.0, Lt=0.25,
               arm=0.6)
        tl.add(0.01, pivot=0.0)
        tl.add(rng.uniform(1.0, 1.8), "smooth", Lb=1.3, D=p["psi"] + np.pi, zp=0.14,
               ath=0.6, ak=0.3, Lt=0.0)
        t_imp = tl.now
        tl.add(0.3, "decel", Lb=1.45)
        self._ev("fall_slow_slide", t_on, tl.now, fall=True, t_impact=t_imp)
        self.posture = "fallen"

    def lie_still(self, dur: float, moves: bool = False):
        t0 = self.tl.now
        if moves:
            self.tl.fidget.append((t0, t0 + dur, 0.12))
        self.tl.hold(dur)
