"""Scenario builders (fall scenarios, activity-of-daily-living streams) + renderer."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .activities import Actor, FALL_KINDS, in_room
from .config import RADAR
from .dsp import Background, calibrate, process
from .motion import Timeline, body_points
from .radar import VEL_DT, RadarSim, Room

CHUNK = 160


@dataclass
class Recording:
    track: np.ndarray  # (T, 10)
    doppler: np.ndarray  # (T, Nc)
    events: list
    meta: dict = field(default_factory=dict)


def make_background(room: Room, snr_db: float, seed: int) -> tuple[RadarSim, Background]:
    sim = RadarSim(room, snr_db=snr_db, seed=seed)
    return sim, calibrate(sim.empty_frames(120))


def render(tl: Timeline, duration: float, sim: RadarSim, bg: Background) -> Recording:
    n = int(duration * RADAR.fps)
    t_all = np.arange(n) * RADAR.frame_s
    tracks, dops = [], []
    for i in range(0, n, CHUNK):
        t = t_all[i: i + CHUNK]
        r0 = tl.render(t)
        r1 = tl.render(t + VEL_DT)
        cube = sim.frames(t, body_points(r0), body_points(r1), r0["present"])
        tr, dn = process(cube, bg)
        tracks.append(tr)
        dops.append(dn)
    return Recording(np.concatenate(tracks), np.concatenate(dops), list(tl.events))


def start_position(rng, az_deg: float | None = None) -> tuple[float, float]:
    """Polar start position; ground radius <= 3.0 m so a fall stays in range."""
    rg = rng.uniform(1.9, 3.0)
    az = np.radians(az_deg if az_deg is not None else rng.uniform(-40, 40))
    return in_room(rg * np.cos(az), rg * np.sin(az))


def fall_scenario(kind: str, rng: np.random.Generator, az_deg: float | None = None,
                  rel_dir_deg: float | None = None, lead_s: tuple = (6, 12),
                  after_s: float = 32.0) -> Timeline:
    """Lead-in activity, one fall of ``kind``, then lying on the floor."""
    x, y = start_position(rng, az_deg)
    psi = (np.radians(rel_dir_deg) + np.arctan2(y, x) + np.pi
           if rel_dir_deg is not None else rng.uniform(-np.pi, np.pi))
    a = Actor(rng, x, y, psi)
    if kind == "from_chair":
        a.tl.p[0][2] = 0.5
        a.tl.p[0][7:9] = 1.45
        a.posture = "sit"
        a.idle(rng.uniform(*lead_s))
    elif kind == "walking_trip":
        a.walk_to(x + rng.uniform(-0.3, 0.3), y + rng.uniform(-0.3, 0.3), 0.9)
        a.idle(rng.uniform(*lead_s))
        p = a._pose()
        a.walk_to(p["x"] + np.cos(p["psi"]) * 0.2, p["y"] + np.sin(p["psi"]) * 0.2, 0.9)
    else:
        a.idle(rng.uniform(*lead_s))
    a.fall(kind)
    a.lie_still(after_s, moves=bool(rng.random() < 0.3))
    return a.tl


# --- daily living -----------------------------------------------------------
BED_POS = (3.3, 0.9)
SEATS = ((3.0, -1.0), (3.3, 1.3), (2.6, 0.6), BED_POS)


def adl_timeline(duration: float, rng: np.random.Generator,
                 hard_negatives: bool = True) -> Timeline:
    """A random day-in-the-life stream with NO falls. Contains fall-like activities."""
    a = Actor(rng, 3.0, 0.0, 0.0)
    guard = 0
    while a.tl.now < duration and guard < 5000:
        guard += 1
        p = a._pose()
        u = rng.random()
        if a.posture == "stand":
            if u < 0.34:
                x, y = start_position(rng)
                a.walk_to(x, y, rng.uniform(0.45, 1.2))
            elif u < 0.50:
                a.idle(rng.uniform(5, 25))
            elif u < 0.60:
                a.absent(rng.uniform(30, 240))
            elif u < 0.78:
                # go to a seat or the bed
                sx, sy = SEATS[rng.integers(len(SEATS))]
                a.walk_to(sx - 0.6, sy, rng.uniform(0.5, 1.0))
                a.turn(0.0 if rng.random() < 0.5 else 0.5, 0.5)
                a.sit_down(fast=bool(rng.random() < 0.25))
                if rng.random() < 0.4:
                    a.idle(rng.uniform(3, 8))
                    a.lie_on_bed(flop=bool(rng.random() < 0.3))
                    a.idle(rng.uniform(60, 420))
                    a.sit_up_from_bed()
                    a.stand_up()
                else:
                    a.idle(rng.uniform(40, 360))
                    a.stand_up()
            elif u < 0.87 and hard_negatives:
                a.bend_pick()
            elif u < 0.92 and hard_negatives:
                a.squat_pick()
            elif u < 0.95 and hard_negatives:
                a.stumble_recover()
            elif u < 0.975 and hard_negatives:
                a.sit_on_floor()
                a.idle(rng.uniform(10, 40))
                if rng.random() < 0.4:
                    a.lie_on_floor()
                    a.lie_still(rng.uniform(10, 40))
                a.get_up_from_floor()
            else:
                a.idle(rng.uniform(3, 10))
        else:  # safety net: should not happen
            a.idle(5)
        del p
    return a.tl


def render_fall(kind: str, seed: int, snr_db: float = 30.0, az_deg=None,
                rel_dir_deg=None) -> Recording:
    rng = np.random.default_rng(seed)
    room = Room(seed=int(rng.integers(1_000_000)))
    sim, bg = make_background(room, snr_db, seed + 1)
    tl = fall_scenario(kind, rng, az_deg, rel_dir_deg)
    rec = render(tl, tl.now, sim, bg)
    rec.meta = dict(kind=kind, seed=seed, snr_db=snr_db)
    return rec


def render_adl(duration: float, seed: int, snr_db: float = 30.0) -> Recording:
    rng = np.random.default_rng(seed)
    room = Room(seed=int(rng.integers(1_000_000)))
    sim, bg = make_background(room, snr_db, seed + 1)
    tl = adl_timeline(duration, rng)
    rec = render(tl, min(duration, tl.now), sim, bg)
    rec.meta = dict(kind="adl", seed=seed, snr_db=snr_db)
    return rec


__all__ = ["FALL_KINDS", "Recording", "render_fall", "render_adl", "fall_scenario",
           "adl_timeline"]


def sit_scenario(rng: np.random.Generator, fast: bool = True, az_deg=None,
                 rel_dir_deg=None, lead_s=(6, 8), after_s=32.0) -> Timeline:
    """Stand, sit down (fast = 'plop'), sit still: the classic fall confuser."""
    x, y = start_position(rng, az_deg)
    psi = (np.radians(rel_dir_deg) + np.arctan2(y, x) + np.pi
           if rel_dir_deg is not None else rng.uniform(-np.pi, np.pi))
    a = Actor(rng, x, y, psi)
    a.idle(rng.uniform(*lead_s))
    a.sit_down(fast=fast)
    a.idle(after_s)
    return a.tl


def render_sit(seed: int, fast: bool = True, snr_db: float = 30.0, az_deg=None,
               rel_dir_deg=None) -> Recording:
    rng = np.random.default_rng(seed)
    room = Room(seed=int(rng.integers(1_000_000)))
    sim, bg = make_background(room, snr_db, seed + 1)
    tl = sit_scenario(rng, fast, az_deg, rel_dir_deg)
    rec = render(tl, tl.now, sim, bg)
    rec.meta = dict(kind="sit_fast" if fast else "sit_normal", seed=seed)
    return rec
