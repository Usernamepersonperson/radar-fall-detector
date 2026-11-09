"""Signal chain: range FFT -> Doppler FFT -> gating -> per-frame track features.

Per frame (20 Hz) the output is a 10-value track row and a noise-normalised
Doppler profile (one STFT column of the micro-Doppler spectrogram).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RADAR, RadarConfig

TRACK = ("m_db", "v_mean", "v_std", "v_hi", "v_lo", "p_db", "rng", "z", "present",
         "n_gate")
TI = {n: i for i, n in enumerate(TRACK)}
MOTION_K = 2.5  # motion gate: bin power > K * median bin power
PRES_K = 12.0  # presence gate (vs. empty-room reference)
MIN_PRES_BINS = 2


@dataclass
class Background:
    """Empty-room reference measured at install (complex range profile)."""

    ref: np.ndarray  # (2, Ns) complex
    mask: np.ndarray  # (Ns,) bool, True = usable bin
    noise: float  # mean |S - ref|^2 per bin & RX in quiet bins


def range_doppler(cube: np.ndarray, cfg: RadarConfig = RADAR):
    """cube (T,2,Nc,Ns) -> S (T,2,Ns) static profile, P (T,Nc,Ns) RD power."""
    nc, ns = cfg.n_chirps, cfg.n_samples
    wr = np.hanning(ns + 2)[1:-1]
    wd = np.hanning(nc + 2)[1:-1]
    x = np.fft.fft(cube * wr, axis=-1)  # range FFT
    s = x.mean(axis=2)
    xd = (x - s[:, :, None, :]) * wd[None, None, :, None]
    rd = np.fft.fftshift(np.fft.fft(xd, axis=2), axes=2)  # Doppler FFT
    p = (np.abs(rd) ** 2).sum(axis=1)
    return s.astype(np.complex64), p.astype(np.float32)


def calibrate(cube_empty: np.ndarray, cfg: RadarConfig = RADAR) -> Background:
    s, _ = range_doppler(cube_empty, cfg)
    ref = s.mean(axis=0)
    var = (np.abs(s - ref) ** 2).sum(axis=1).mean(axis=0)  # (Ns,)
    floor = np.median(var)
    mask = var < 6 * floor
    return Background(ref=ref, mask=mask, noise=float(floor))


def track_features(s: np.ndarray, p: np.ndarray, bg: Background,
                   cfg: RadarConfig = RADAR):
    """-> track (T, len(TRACK)) float32, D (T, Nc) float32 (noise-normalised)."""
    T, nc, ns = p.shape
    v = (np.arange(nc) - nc // 2) * cfg.vel_res
    mask = bg.mask[None, :]

    pm = p.sum(axis=1)  # (T,Ns) motion power per range bin
    noise_m = np.median(pm[:, bg.mask], axis=1) + 1e-9
    gate_m = (pm > MOTION_K * noise_m[:, None]) & mask
    g = gate_m.sum(axis=1)
    d = np.einsum("tvn,tn->tv", p, gate_m.astype(np.float32))  # (T,Nc)
    d_noise = (g * noise_m / nc)[:, None]  # expected noise per Doppler bin
    dn = d / noise_m[:, None]
    m_e = np.maximum(d.sum(1) - g * noise_m, 0) / noise_m
    m_db = 10 * np.log10(1 + m_e)
    dsum = d.sum(1) + 1e-12
    v_mean = (d * v).sum(1) / dsum
    v_std = np.sqrt(np.maximum((d * v**2).sum(1) / dsum - v_mean**2, 0))
    sig = d > 6 * d_noise + 1e-12
    sig &= (g > 0)[:, None]
    v_hi = np.where(sig.any(1), np.where(sig, v, -np.inf).max(1), 0.0)
    v_lo = np.where(sig.any(1), np.where(sig, v, np.inf).min(1), 0.0)
    v_mean = np.where(g > 0, v_mean, 0.0)
    v_std = np.where(g > 0, v_std, 0.0)

    ds = s - bg.ref[None]
    pp = (np.abs(ds) ** 2).sum(axis=1)  # (T,Ns)
    noise_p = np.median(pp[:, bg.mask], axis=1) / 0.8 + 1e-9
    gate_p = (pp > PRES_K * noise_p[:, None]) & mask
    present = gate_p.sum(axis=1) >= MIN_PRES_BINS
    gate_u = (gate_p | gate_m) & mask
    w = np.where(gate_u, pp, 0.0)
    wsum = w.sum(1) + 1e-12
    rbins = np.arange(ns) * cfg.range_res
    rng = (w * rbins).sum(1) / wsum
    cross = (ds[:, 1] * np.conj(ds[:, 0]) * gate_u).sum(1)
    sin_el = np.clip(np.angle(cross) / np.pi, -1, 1)
    z = cfg.mount_height + rng * sin_el
    p_db = 10 * np.log10(1 + (pp * gate_p).sum(1) / noise_p)

    pres_f = present.astype(np.float32)
    z = np.where(present, z, np.nan)
    rng = np.where(present, rng, np.nan)
    out = np.stack([m_db, v_mean, v_std, v_hi, v_lo, p_db, rng, z, pres_f,
                    gate_u.sum(1)], 1).astype(np.float32)
    return out, dn.astype(np.float32)


def process(cube: np.ndarray, bg: Background, cfg: RadarConfig = RADAR):
    s, p = range_doppler(cube, cfg)
    return track_features(s, p, bg, cfg)
