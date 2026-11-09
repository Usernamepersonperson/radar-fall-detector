"""Point-scatterer human motion model.

The body is 12 scatterers (head, chest, abdomen, pelvis, 2 upper arms, 2 forearms,
2 thighs, 2 shanks). A pose is 12 numbers; a ``Timeline`` interpolates keyframes
(with easing) and adds gait / fidget / breathing overlays. Activities are built
from keyframes, so every motion is smooth and has a known ground-truth timing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

POSE = ("x", "y", "zp", "psi", "Lb", "D", "Lt", "ath", "ak", "arm", "elb", "pivot")
IDX = {n: i for i, n in enumerate(POSE)}
STAND = dict(x=0.0, y=0.0, zp=0.95, psi=0.0, Lb=0.0, D=0.0, Lt=0.0, ath=0.0,
             ak=0.0, arm=0.05, elb=0.1, pivot=0.0)

H_PELVIS = 0.95  # standing pelvis height, m
THIGH, SHANK, UARM, FARM = 0.42, 0.42, 0.28, 0.27
BODY_RCS = np.array([0.08, 0.8, 0.6, 0.5, 0.15, 0.15, 0.12, 0.12, 0.3, 0.3, 0.2, 0.2])
N_BODY = 12

EASE = {
    "linear": lambda u: u,
    "smooth": lambda u: u * u * (3 - 2 * u),
    "accel": lambda u: u**2.2,  # gravity-like acceleration into the floor
    "decel": lambda u: 1 - (1 - u) ** 2,
}


@dataclass
class Timeline:
    """Keyframed pose trajectory plus overlays and ground-truth annotations."""

    pose0: dict = field(default_factory=lambda: dict(STAND))
    t: list = field(default_factory=lambda: [0.0])
    p: list = field(default_factory=list)
    ease: list = field(default_factory=lambda: ["linear"])
    gait: list = field(default_factory=list)  # (t0, t1, freq_hz, amp)
    fidget: list = field(default_factory=list)  # (t0, t1, amp)
    absent: list = field(default_factory=list)  # (t0, t1)
    events: list = field(default_factory=list)  # dicts with kind/t_onset/t_impact

    def __post_init__(self):
        if not self.p:
            self.p = [np.array([self.pose0[n] for n in POSE], float)]

    @property
    def now(self) -> float:
        return self.t[-1]

    def last(self) -> dict:
        return {n: float(self.p[-1][i]) for n, i in IDX.items()}

    def add(self, dt: float, ease: str = "smooth", **ch) -> float:
        new = self.p[-1].copy()
        for k, v in ch.items():
            new[IDX[k]] = v
        self.t.append(self.t[-1] + max(dt, 1e-3))
        self.p.append(new)
        self.ease.append(ease)
        return self.now

    def hold(self, dt: float) -> float:
        return self.add(dt, "linear")

    def render(self, t: np.ndarray) -> dict:
        """Evaluate the pose (and overlay signals) at times ``t`` (seconds)."""
        kt = np.asarray(self.t)
        kp = np.asarray(self.p)
        t = np.asarray(t, float)
        if len(kt) == 1:  # a single keyframe is a constant pose
            kt, kp = np.r_[kt, kt + 1.0], np.vstack([kp, kp])
            ease = ["linear", "linear"]
        else:
            ease = self.ease
        seg = np.clip(np.searchsorted(kt, t, side="right") - 1, 0, len(kt) - 2)
        u = np.clip((t - kt[seg]) / (kt[seg + 1] - kt[seg]), 0, 1)
        s = np.empty_like(u)
        for name in set(ease[1:]):
            m = np.array([ease[i + 1] == name for i in seg])
            s[m] = EASE[name](u[m])
        pose = kp[seg] + (kp[seg + 1] - kp[seg]) * s[:, None]
        out = {n: pose[:, i] for n, i in IDX.items()}
        out["gait"] = _segments(t, self.gait, 3)
        out["fidget"] = _segments(t, self.fidget, 2)
        out["present"] = np.ones_like(t)
        for t0, t1 in self.absent:
            out["present"][(t >= t0) & (t < t1)] = 0.0
        out["t"] = t
        return out


def _segments(t: np.ndarray, segs: list, ncol: int) -> np.ndarray:
    """Per-time overlay: columns (amp, phase[, ...]) with 0.3 s fades."""
    res = np.zeros((len(t), 2))
    for sg in segs:
        t0, t1 = sg[0], sg[1]
        m = (t >= t0) & (t < t1)
        if not m.any():
            continue
        fade = np.clip(np.minimum(t - t0, t1 - t) / 0.3, 0, 1)
        if ncol == 3:  # gait: freq, amp
            freq, amp = sg[2], sg[3]
            res[m, 0] = (amp * fade)[m]
            res[m, 1] = 2 * np.pi * freq * (t[m] - t0)
        else:  # fidget: amp
            res[m, 0] = (sg[2] * fade)[m]
            res[m, 1] = 2 * np.pi * 0.7 * (t[m] - t0)
    return res


def _rodrigues(v: np.ndarray, k: np.ndarray, ang: np.ndarray) -> np.ndarray:
    c = np.cos(ang)[:, None]
    s = np.sin(ang)[:, None]
    kv = np.cross(k, v)
    return v * c + kv * s + k * (k * v).sum(1, keepdims=True) * (1 - c)


def body_points(r: dict) -> np.ndarray:
    """Pose arrays -> scatterer positions (T, 12, 3) in room coordinates."""
    t = r["t"]
    n = len(t)
    psi, lb, dd = r["psi"], r["Lb"], r["D"]
    f = np.stack([np.cos(psi), np.sin(psi), np.zeros(n)], 1)
    z = np.tile([0.0, 0.0, 1.0], (n, 1))
    s0 = np.cross(z, f)
    dh = np.stack([np.cos(dd), np.sin(dd), np.zeros(n)], 1)
    k = np.cross(z, dh)
    gb, sb, ub = (_rodrigues(v, k, lb) for v in (f, s0, z))

    piv = r["pivot"]
    sl = np.sin(lb)
    zlie = np.maximum(0.12, H_PELVIS * np.cos(lb) + 0.12 * sl)
    zp = (1 - piv) * r["zp"] + piv * zlie
    px = r["x"] + piv * H_PELVIS * sl * np.cos(dd)
    py = r["y"] + piv * H_PELVIS * sl * np.sin(dd)
    pel = np.stack([px, py, zp], 1)

    # overlays: gait (opposite legs/arms), fidget, breathing
    g_amp, g_ph = r["gait"][:, 0], r["gait"][:, 1]
    f_amp, f_ph = r["fidget"][:, 0], r["fidget"][:, 1]
    sw = np.sin(g_ph)
    ath_l = r["ath"] + g_amp * 0.42 * sw
    ath_r = r["ath"] - g_amp * 0.42 * sw
    ak_l = r["ak"] + g_amp * 0.55 * np.maximum(0, -sw + 0.3)
    ak_r = r["ak"] + g_amp * 0.55 * np.maximum(0, sw + 0.3)
    arm_l = r["arm"] - g_amp * 0.4 * sw + f_amp * 0.9 * np.sin(f_ph)
    arm_r = r["arm"] + g_amp * 0.4 * sw + f_amp * 0.9 * np.sin(f_ph * 1.3 + 1)
    elb_l = r["elb"] + 0.3 * g_amp + f_amp * 0.5 * (1 + np.sin(f_ph * 0.8))
    elb_r = r["elb"] + 0.3 * g_amp + f_amp * 0.5 * (1 + np.sin(f_ph * 1.1))
    breath = 0.004 * np.sin(2 * np.pi * 0.27 * t)
    bob = 0.015 * g_amp * np.abs(sw)

    lt = r["Lt"]
    ut = np.stack([np.sin(lt), np.zeros(n), np.cos(lt)], 1)  # (g, s, u) coords
    gt = np.stack([np.cos(lt), np.zeros(n), -np.sin(lt)], 1)
    sdir = np.tile([0.0, 1.0, 0.0], (n, 1))

    def limb(start, ang, length, side_up=False):
        # direction measured from -u_t toward g_t
        d = -np.cos(ang)[:, None] * ut + np.sin(ang)[:, None] * gt
        return start + d * length, start + d * length / 2

    rel = np.zeros((n, N_BODY, 3))
    head = ut * 0.72 + gt * 0.03
    chest = ut * 0.40 + gt * breath[:, None]
    abdo = ut * 0.15 + gt * breath[:, None] * 0.6
    rel[:, 0], rel[:, 1], rel[:, 2], rel[:, 3] = head, chest, abdo, 0.0
    for j, (sd, a, e) in enumerate(((1, arm_l, elb_l), (-1, arm_r, elb_r))):
        sh = ut * 0.50 + sd * 0.18 * sdir
        elbow, mid1 = limb(sh, a, UARM)
        _, mid2 = limb(elbow, a + e, FARM)
        rel[:, 4 + j], rel[:, 6 + j] = mid1, mid2
    for j, (sd, a, kf) in enumerate(((1, ath_l, ak_l), (-1, ath_r, ak_r))):
        hip = sd * 0.09 * sdir
        d1 = np.stack([np.sin(a), np.zeros(n), -np.cos(a)], 1)
        knee = hip + d1 * THIGH
        a2 = a - kf
        d2 = np.stack([np.sin(a2), np.zeros(n), -np.cos(a2)], 1)
        rel[:, 8 + j] = hip + d1 * THIGH / 2
        rel[:, 10 + j] = knee + d2 * SHANK / 2

    pts = (
        pel[:, None, :]
        + rel[:, :, 0:1] * gb[:, None, :]
        + rel[:, :, 1:2] * sb[:, None, :]
        + rel[:, :, 2:3] * ub[:, None, :]
    )
    pts[:, :, 2] += bob[:, None]
    pts[:, :, 2] = np.maximum(pts[:, :, 2], 0.03)
    return pts
