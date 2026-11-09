"""Window features from the per-frame track + micro-Doppler spectrogram.

Everything is causal (uses only past frames), so the batch code used for training
and the streaming code used on the device compute identical numbers.
"""

from __future__ import annotations

import numpy as np

from .config import RADAR
from .dsp import TI

FPS = RADAR.fps
MED = 5  # frames of causal median smoothing for z
BANDS = (0.0, 0.3, 0.8, 1.6, 3.0, 99.0)  # |v| band edges, m/s
N_SEG = 4

NAMES = (
    ["peak_speed", "top5_speed", "t_peak", "fast_s", "z_hi", "z_end", "z_min",
     "z_drop", "desc_rate", "t_after", "still_mean", "still_max", "still_frac",
     "pres_end", "m_mean"]
    + [f"band{b}_seg{s}" for b in range(len(BANDS) - 1) for s in range(N_SEG)]
)
N_FEAT = len(NAMES)


def smooth_z(z: np.ndarray, present: np.ndarray, init: float = 0.9) -> np.ndarray:
    """Forward-fill missing heights, then causal median over the last MED frames."""
    z = np.where(present > 0.5, z, np.nan)
    idx = np.where(np.isnan(z), 0, np.arange(len(z)))
    np.maximum.accumulate(idx, out=idx)
    zf = z[idx]
    zf = np.where(np.isnan(zf), init, zf)
    pad = np.concatenate([np.full(MED - 1, zf[0]), zf])
    win = np.lib.stride_tricks.sliding_window_view(pad, MED)
    return np.median(win, axis=1).astype(np.float32)


def speed_env(track: np.ndarray) -> np.ndarray:
    return np.maximum(np.abs(track[:, TI["v_hi"]]), np.abs(track[:, TI["v_lo"]]))


def window_features(track: np.ndarray, dop: np.ndarray, zs: np.ndarray,
                    end: int, win: int) -> np.ndarray:
    """Feature vector for frames [end-win, end). ``zs`` = smooth_z(track)."""
    lo = end - win
    tr, d, z = track[lo:end], dop[lo:end], zs[lo:end]
    sp = speed_env(tr)
    ts = np.sort(sp)[::-1]
    i_pk = int(np.argmax(sp))
    last3 = slice(-int(3 * FPS), None)
    mdb = tr[:, TI["m_db"]]
    half = int(0.5 * FPS)
    dz = z[half:] - z[:-half]
    i_d = int(np.argmin(dz))

    v = (np.arange(d.shape[1]) - d.shape[1] // 2) * RADAR.vel_res
    av = np.abs(v)
    seg = np.array_split(np.arange(win), N_SEG)
    bands = []
    for b in range(len(BANDS) - 1):
        sel = (av >= BANDS[b]) & (av < BANDS[b + 1])
        e = d[:, sel].sum(1)
        for s in seg:
            bands.append(np.log10(1 + e[s].mean()))

    f = [
        sp.max(), ts[:5].mean(), i_pk / win, (sp > 1.2).sum() / FPS,
        np.percentile(z, 90), np.median(z[-int(FPS):]), z.min(),
        np.percentile(z, 90) - np.median(z[-int(FPS):]),
        dz[i_d] / 0.5, (win - 1 - (i_d + half)) / FPS,
        mdb[last3].mean(), mdb[last3].max(), (mdb[last3] < 2.0).mean(),
        tr[-int(2 * FPS):, TI["present"]].mean(), mdb.mean(),
    ] + bands
    return np.asarray(f, np.float32)


def all_windows(track, dop, win_s=8.0, hop_s=0.5):
    """Features for every hop; returns (ends, X)."""
    win, hop = int(win_s * FPS), int(hop_s * FPS)
    zs = smooth_z(track[:, TI["z"]], track[:, TI["present"]])
    ends = np.arange(win, len(track) + 1, hop)
    x = np.stack([window_features(track, dop, zs, e, win) for e in ends]) \
        if len(ends) else np.zeros((0, N_FEAT), np.float32)
    return ends, x
