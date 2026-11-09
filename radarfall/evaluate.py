"""Event-level evaluation of the streaming detector on simulated recordings."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import numpy as np

from .activities import ADL_FALL_LIKE
from .config import DETECTOR, RADAR, DetectorConfig
from .detector import FallDetector, run_stream
from .scenarios import Recording

MATCH_AFTER_S = 30.0  # an alarm up to 30 s after impact still counts for that fall
ATTRIB_AFTER_S = 25.0  # alarm attribution window for fall-like activities


def run_recording(rec: Recording, scorer, mode: str,
                  cfg: DetectorConfig = DETECTOR) -> list[dict]:
    det = FallDetector(scorer, cfg, mode)
    return run_stream(det, rec.track, rec.doppler)


def eval_falls(recs: list[Recording], scorer, mode: str,
               cfg: DetectorConfig = DETECTOR) -> dict:
    """Sensitivity and latency (seconds after impact) per fall kind."""
    res = defaultdict(lambda: dict(n=0, hit=0, suspected=0, lat_alarm=[], lat_susp=[],
                                   extra=0))
    for rec in recs:
        ev = run_recording(rec, scorer, mode, cfg)
        fall = next(e for e in rec.events if e["fall"])
        r = res[fall["kind"]]
        r["n"] += 1
        lo, hi = fall["t_onset"], fall["t_impact"] + MATCH_AFTER_S
        alarms = [e for e in ev if e["event"] == "fall_alarm"]
        susp = [e for e in ev if e["event"] == "fall_suspected" and lo <= e["t"] <= hi]
        good = [e for e in alarms if lo <= e["t"] <= hi]
        r["extra"] += len(alarms) - len(good)
        if susp:
            r["suspected"] += 1
            r["lat_susp"].append(susp[0]["t"] - fall["t_impact"])
        if good:
            r["hit"] += 1
            r["lat_alarm"].append(good[0]["t"] - fall["t_impact"])
    return dict(res)


def eval_adl(recs: list[Recording], scorer, mode: str,
             cfg: DetectorConfig = DETECTOR) -> dict:
    """False alarms/day and per-activity confusion on fall-free streams."""
    secs = sum(len(r.track) for r in recs) / RADAR.fps
    kinds = {k: dict(n=0, suspected=0, alarmed=0) for k in ADL_FALL_LIKE}
    alarms = suspects = 0
    cause: dict[str, int] = defaultdict(int)
    for rec in recs:
        ev = run_recording(rec, scorer, mode, cfg)
        a_t = [e["t"] for e in ev if e["event"] == "fall_alarm"]
        s_t = [e["t"] for e in ev if e["event"] == "fall_suspected"]
        alarms += len(a_t)
        suspects += len(s_t)
        for t in a_t:  # what was the person doing when the false alarm fired?
            hit = [e["kind"] for e in rec.events
                   if e["t_onset"] <= t <= e["t_end"] + ATTRIB_AFTER_S]
            cause[hit[0] if hit else "no_event"] += 1
        for e in rec.events:
            k = kinds[e["kind"]]
            k["n"] += 1
            lo, hi = e["t_onset"], e["t_end"] + ATTRIB_AFTER_S
            k["suspected"] += any(lo <= t <= hi for t in s_t)
            k["alarmed"] += any(lo <= t <= hi for t in a_t)
    hours = secs / 3600
    return dict(hours=hours, alarms=alarms, suspected=suspects,
                alarms_per_day=alarms / hours * 24 if hours else 0.0,
                suspected_per_day=suspects / hours * 24 if hours else 0.0,
                alarm_cause=dict(cause), kinds=kinds)


def summarize_falls(res: dict) -> dict:
    n = sum(r["n"] for r in res.values())
    hit = sum(r["hit"] for r in res.values())
    lat = [x for r in res.values() for x in r["lat_alarm"]]
    sus = [x for r in res.values() for x in r["lat_susp"]]
    return dict(
        n=n, hit=hit, sensitivity=hit / n if n else 0.0,
        latency_alarm_median_s=float(np.median(lat)) if lat else None,
        latency_alarm_p90_s=float(np.percentile(lat, 90)) if lat else None,
        latency_suspected_median_s=float(np.median(sus)) if sus else None,
        per_kind={k: dict(n=r["n"], hit=r["hit"],
                          sensitivity=r["hit"] / r["n"] if r["n"] else 0.0)
                  for k, r in res.items()})


def with_threshold(thr: float) -> DetectorConfig:
    return replace(DETECTOR, ml_threshold=thr)
