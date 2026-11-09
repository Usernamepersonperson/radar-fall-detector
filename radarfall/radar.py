"""FMCW radar signal simulator (beat-signal level).

For every frame the IF signal of K point scatterers is synthesised for 2 RX
channels (vertical pair, for an elevation / height proxy):

    s[rx, m, n] = sum_k a_k * exp(j*(4*pi*(r_k + v_k*m*Tc)/lambda + elev_k*rx))
                          * exp(j*2*pi*(r_k/dr)*n/Ns)

``m`` = chirp, ``n`` = fast-time sample. Simplifications (documented in README):
constant radial velocity within a frame burst, no range migration, ideal antenna
patterns (cosine in azimuth), floor-bounce and back-wall images as the only
multipath, static clutter plus oscillating clutter (fan / curtain), LO phase jitter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import RADAR, RadarConfig
from .motion import BODY_RCS, N_BODY

WALL_X = 5.2  # back wall for the multipath image, m
FLOOR_GAIN, WALL_GAIN = 0.30, 0.15
VEL_DT = 0.004  # finite-difference step for radial velocity, s


@dataclass
class Room:
    """Static clutter + oscillating clutter (fan, curtain), fixed by a seed."""

    seed: int = 0
    n_static: int = 16
    n_fans: int = 2
    static: np.ndarray = field(init=False)  # (n, 4): x y z rcs
    fans: np.ndarray = field(init=False)  # (n, 7): x y z rcs amp_m f_hz phase

    def __post_init__(self):
        g = np.random.default_rng(self.seed + 7919)
        n = self.n_static
        self.static = np.column_stack([
            g.uniform(0.8, 5.0, n), g.uniform(-2.2, 2.2, n), g.uniform(0.1, 2.4, n),
            10 ** g.uniform(0.5, 2.0, n)])
        m = self.n_fans
        self.fans = np.column_stack([
            g.uniform(1.5, 4.5, m), g.uniform(-1.8, 1.8, m), g.uniform(1.8, 2.4, m),
            g.uniform(0.3, 1.5, m), g.uniform(0.002, 0.006, m), g.uniform(1.0, 3.0, m),
            g.uniform(0, 6.28, m)])


def _amp(rcs, r, az, cfg: RadarConfig):
    """Two-way amplitude: sqrt(rcs)/r^2 normalised to 1.0 for rcs=1 at 3 m."""
    pat = np.clip(np.cos(az / np.radians(cfg.az_fov_deg) * np.pi / 2), 0, 1)
    return np.sqrt(rcs) * 9.0 / np.maximum(r, 0.3) ** 2 * pat


def _geom(p: np.ndarray, cfg: RadarConfig):
    d = p - np.array([0.0, 0.0, cfg.mount_height])
    r = np.linalg.norm(d, axis=-1)
    az = np.arctan2(d[..., 1], d[..., 0])
    sin_el = d[..., 2] / np.maximum(r, 1e-6)
    return r, az, sin_el


class RadarSim:
    """Generates raw IF cubes (T, 2, Nc, Ns) for a batch of frames."""

    def __init__(self, room: Room, snr_db: float = 30.0, cfg: RadarConfig = RADAR,
                 seed: int = 0, phase_jitter: float = 0.004):
        self.cfg, self.room = cfg, room
        self.rng = np.random.default_rng(seed)
        self.phase_jitter = phase_jitter
        nc, ns = cfg.n_chirps, cfg.n_samples
        # noise sigma so a rcs=1 scatterer at 3 m has post-FFT SNR = snr_db
        self.sigma = float(np.sqrt(nc * ns / 10 ** (snr_db / 10) / 2))
        self.m = np.arange(nc)
        self.n = np.arange(ns)
        self.static_cube = self._static()

    def _fast(self, r):
        c = self.cfg
        return np.exp(2j * np.pi * (r[..., None] / c.range_res) * self.n / c.n_samples)

    def _static(self) -> np.ndarray:
        c, s = self.cfg, self.room.static
        r, az, se = _geom(s[:, :3], c)
        a = _amp(s[:, 3], r, az, c)
        ph = np.exp(1j * 4 * np.pi * r / c.wavelength)
        cube = np.zeros((2, c.n_samples), complex)
        for rx in range(2):
            w = a * ph * np.exp(1j * np.pi * se * rx)
            cube[rx] = (w[:, None] * self._fast(r)).sum(0)
        return cube[:, None, :]  # same in every chirp

    def _scatter(self, t, body, present):
        """Dynamic scatterers: body + floor/wall images + fans."""
        room = self.room
        T = len(t)
        fan_p = np.zeros((2, T, len(room.fans), 3))
        for j, tt in enumerate((t, t + VEL_DT)):
            osc = room.fans[None, :, 4] * np.sin(
                2 * np.pi * room.fans[None, :, 5] * tt[:, None] + room.fans[None, :, 6])
            fan_p[j, :, :, 0] = room.fans[None, :, 0] + osc
            fan_p[j, :, :, 1] = room.fans[None, :, 1]
            fan_p[j, :, :, 2] = room.fans[None, :, 2]
        return fan_p

    def frames(self, t: np.ndarray, body0: np.ndarray, body1: np.ndarray,
               present: np.ndarray) -> np.ndarray:
        """``body0``/``body1``: scatterer positions at t and t+VEL_DT, (T,12,3)."""
        c = self.cfg
        T = len(t)
        fan = self._scatter(t, body0, present)
        # build full point sets: body, floor image, wall image, fans
        def images(b):
            fl = b * [1, 1, -1]
            wl = b.copy()
            wl[..., 0] = 2 * WALL_X - wl[..., 0]
            return np.concatenate([b, fl, wl], axis=1)
        p0 = np.concatenate([images(body0), fan[0]], axis=1)
        p1 = np.concatenate([images(body1), fan[1]], axis=1)
        rcs = np.concatenate([
            np.tile(BODY_RCS, 3), np.full(len(self.room.fans), 0.0)])[None, :]
        rcs = np.broadcast_to(rcs, (T, rcs.shape[1])).copy()
        rcs[:, : 3 * N_BODY] *= present[:, None]
        rcs[:, 3 * N_BODY:] = self.room.fans[None, :, 3]
        gain = np.concatenate([
            np.ones(N_BODY), np.full(N_BODY, FLOOR_GAIN), np.full(N_BODY, WALL_GAIN),
            np.ones(len(self.room.fans))])[None, :]
        r0, az, se = _geom(p0, c)
        r1, _, _ = _geom(p1, c)
        v = (r1 - r0) / VEL_DT
        a = _amp(rcs, r0, az, c) * gain
        a = a * (1 + 0.12 * self.rng.standard_normal(a.shape))  # scintillation
        base = a * np.exp(1j * 4 * np.pi * r0 / c.wavelength)  # (T,K)
        dop = np.exp(1j * 4 * np.pi * v[:, :, None] * self.m[None, None, :]
                     * c.t_chirp / c.wavelength)  # (T,K,Nc)
        fast = self._fast(r0)  # (T,K,Ns)
        cube = np.empty((T, 2, c.n_chirps, c.n_samples), np.complex64)
        for rx in range(2):
            w = base * np.exp(1j * np.pi * se * rx)
            slow = w[:, :, None] * dop  # (T,K,Nc)
            cube[:, rx] = np.matmul(slow.transpose(0, 2, 1), fast)
        cube += self.static_cube[None] * np.exp(
            1j * self.phase_jitter * self.rng.standard_normal((T, 1, 1, 1)))
        noise = self.rng.standard_normal((2,) + cube.shape).astype(np.float32)
        cube += self.sigma * (noise[0] + 1j * noise[1])
        return cube

    def empty_frames(self, n: int) -> np.ndarray:
        t = np.arange(n) * self.cfg.frame_s
        z = np.zeros((n, N_BODY, 3)) + [3.0, 0.0, 1.0]
        return self.frames(t, z, z, np.zeros(n))
