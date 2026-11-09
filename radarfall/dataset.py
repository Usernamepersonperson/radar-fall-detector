"""Dataset generation (parallel, cached) and window labelling."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .config import RADAR
from .features import all_windows
from .scenarios import FALL_KINDS, Recording, render_adl, render_fall

SIM_VERSION = 5  # bump when the simulator / DSP changes (invalidates the cache)
CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"


def say(msg: str):
    try:
        print(msg, flush=True)
    except OSError:  # a full disk must not kill a long run
        pass


def _run(spec: dict) -> Recording:
    if spec["type"] == "fall":
        rec = render_fall(spec["kind"], spec["seed"], spec.get("snr", 30.0),
                          spec.get("az"), spec.get("rel"))
    else:
        rec = render_adl(spec["dur"], spec["seed"], spec.get("snr", 30.0))
    # store as float16 (halves the cache); round here so cached and fresh runs match
    rec.track = rec.track.astype(np.float16).astype(np.float32)
    rec.doppler = rec.doppler.astype(np.float16).astype(np.float32)
    return rec


def generate(specs: list[dict], workers: int | None = None, name: str | None = None,
             verbose: bool = True) -> list[Recording]:
    """Render recordings for ``specs``; cached on disk when ``name`` is given."""
    key = hashlib.sha1(json.dumps([SIM_VERSION, specs], sort_keys=True).encode()
                       ).hexdigest()[:10]
    path = CACHE / f"{name}-{key}.npz" if name else None
    if path and path.exists():
        try:
            return _load(path)
        except (KeyError, OSError, ValueError, EOFError, zipfile.BadZipFile):
            path.unlink(missing_ok=True)  # truncated cache (e.g. disk full): rebuild
    workers = workers or max(1, (os.cpu_count() or 2) - 0)
    if workers == 1:
        recs = [_run(s) for s in specs]
    else:
        with ProcessPoolExecutor(workers) as ex:
            recs = []
            for i, r in enumerate(ex.map(_run, specs, chunksize=1)):
                recs.append(r)
                if verbose and (i + 1) % max(1, len(specs) // 10) == 0:
                    say(f"  {name or 'set'}: {i + 1}/{len(specs)}")
    for r, s in zip(recs, specs):
        r.meta["spec"] = s
    if path:
        try:
            _save(path, recs)
        except OSError as e:  # e.g. disk full: keep going without a cache
            say(f"  warning: cache not written ({e})")
            path.with_name(path.stem + ".tmp.npz").unlink(missing_ok=True)
    return recs


def _save(path: Path, recs: list[Recording]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        n=len(recs),
        meta=json.dumps([[r.events, r.meta] for r in recs]),
        **{f"t{i}": r.track.astype(np.float16) for i, r in enumerate(recs)},
        **{f"d{i}": r.doppler.astype(np.float16) for i, r in enumerate(recs)},
    )
    os.replace(tmp, path)  # atomic: a half-written file is never picked up


def _load(path: Path) -> list[Recording]:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        return [Recording(z[f"t{i}"].astype(np.float32), z[f"d{i}"].astype(np.float32), ev, mt)
                for i, (ev, mt) in enumerate(meta)]


def fall_specs(per_kind: int, seed0: int, snr: float = 30.0, az=None, rel=None,
               kinds=FALL_KINDS) -> list[dict]:
    return [dict(type="fall", kind=k, seed=seed0 + 1000 * j + i, snr=snr, az=az,
                 rel=rel) for j, k in enumerate(kinds) for i in range(per_kind)]


def adl_specs(total_s: float, seed0: int, chunk_s: float = 1800.0, snr: float = 30.0):
    n = int(np.ceil(total_s / chunk_s))
    return [dict(type="adl", dur=min(chunk_s, total_s - i * chunk_s), seed=seed0 + i,
                 snr=snr) for i in range(n)]


# --- labelling ------------------------------------------------------------------
def label_windows(ends: np.ndarray, events: list[dict]) -> np.ndarray:
    """1 = fall decided window, 0 = negative, -1 = ambiguous (excluded in training).

    Positive: window end 1.5-6 s after impact (descent + early stillness inside).
    Ambiguous: during the fall, or 6-14 s after impact (already on the floor).
    """
    te = ends / RADAR.fps
    y = np.zeros(len(ends), np.int8)
    for ev in events:
        if not ev["fall"]:
            continue
        dt = te - ev["t_impact"]
        y[(dt >= 1.5) & (dt <= 6.0)] = 1
        amb = ((te >= ev["t_onset"]) & (dt < 1.5)) | ((dt > 6.0) & (dt <= 14.0))
        y[amb & (y == 0)] = -1
    return y


def windows_of(recs: list[Recording], win_s=8.0, hop_s=0.5):
    """Features + labels + bookkeeping for a list of recordings."""
    xs, ys, rid, tend = [], [], [], []
    for i, r in enumerate(recs):
        ends, x = all_windows(r.track, r.doppler, win_s, hop_s)
        if len(ends) == 0:
            continue
        xs.append(x)
        ys.append(label_windows(ends, r.events))
        rid.append(np.full(len(ends), i))
        tend.append(ends / RADAR.fps)
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(rid),
            np.concatenate(tend))
