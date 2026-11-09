"""Radar geometry, simulator and DSP checks against known values."""
import numpy as np
import pytest

from radarfall.config import C, RADAR
from radarfall.dsp import TI, calibrate, process, range_doppler
from radarfall.motion import N_BODY
from radarfall.radar import VEL_DT, RadarSim, Room


def point_cloud(x, y, z, vr=0.0, n=1):
    """All 12 body scatterers at one point, moving radially at vr (m/s)."""
    p0 = np.zeros((n, N_BODY, 3)) + [x, y, z]
    d = np.array([x, y, z - RADAR.mount_height])
    p1 = p0 + d / np.linalg.norm(d) * vr * VEL_DT
    return p0, p1


def test_derived_parameters():
    assert RADAR.wavelength == pytest.approx(C / 60e9)
    assert RADAR.range_res == pytest.approx(0.1, rel=1e-2)
    assert RADAR.max_range == pytest.approx(6.4, rel=1e-2)
    assert RADAR.vel_max == pytest.approx(RADAR.wavelength / (4 * 250e-6))
    assert RADAR.vel_res == pytest.approx(0.3123, rel=1e-2)
    assert RADAR.fps == pytest.approx(20.0)


def test_static_target_lands_in_expected_range_bin():
    sim = RadarSim(Room(seed=3), snr_db=45, seed=1)
    bg = calibrate(sim.empty_frames(60))
    p0, p1 = point_cloud(3.0, 0.0, 1.8)  # same height as radar: r = 3.0 m
    cube = sim.frames(np.zeros(1), p0, p1, np.ones(1))
    s, _ = range_doppler(cube)
    ds = np.abs(s[0] - bg.ref).sum(0) * bg.mask
    assert abs(int(np.argmax(ds)) - 3.0 / RADAR.range_res) <= 1


@pytest.mark.parametrize("vr", [-2.0, -0.9, 0.9, 1.9])
def test_doppler_sign_and_magnitude(vr):
    sim = RadarSim(Room(seed=4), snr_db=45, seed=2)
    bg = calibrate(sim.empty_frames(60))
    p0, p1 = point_cloud(3.0, 0.0, 1.8, vr)
    cube = sim.frames(np.zeros(1), p0, p1, np.ones(1))
    tr, dn = process(cube, bg)
    v = (np.arange(RADAR.n_chirps) - RADAR.n_chirps // 2) * RADAR.vel_res
    v_peak = v[int(np.argmax(dn[0]))]
    assert abs(v_peak - vr) <= RADAR.vel_res
    assert np.sign(tr[0, TI["v_mean"]]) == np.sign(vr)  # positive = moving away


def test_empty_room_has_no_presence_or_motion():
    sim = RadarSim(Room(seed=5), snr_db=30, seed=3)
    bg = calibrate(sim.empty_frames(80))
    tr, _ = process(sim.empty_frames(120), bg)
    assert tr[:, TI["present"]].mean() < 0.05
    assert tr[:, TI["m_db"]].mean() < 1.5


def test_still_person_is_detected_by_presence_not_motion():
    sim = RadarSim(Room(seed=6), snr_db=30, seed=4)
    bg = calibrate(sim.empty_frames(80))
    p0, p1 = point_cloud(2.5, 0.0, 0.3, n=40)
    cube = sim.frames(np.arange(40) * 0.05, p0, p1, np.ones(40))
    tr, _ = process(cube, bg)
    assert tr[:, TI["present"]].mean() > 0.9
    assert tr[:, TI["m_db"]].mean() < 3.0


def test_height_proxy_tracks_true_height():
    sim = RadarSim(Room(seed=7), snr_db=40, seed=5)
    bg = calibrate(sim.empty_frames(80))
    zs = []
    for z in (0.3, 1.0, 1.6):
        p0, p1 = point_cloud(2.5, 0.0, z, n=30)
        cube = sim.frames(np.arange(30) * 0.05, p0, p1, np.ones(30))
        tr, _ = process(cube, bg)
        zs.append(np.nanmedian(tr[:, TI["z"]]))
    assert zs[0] < zs[1] < zs[2]
    assert zs[0] == pytest.approx(0.3, abs=0.35)
    assert zs[2] == pytest.approx(1.6, abs=0.35)


def test_noise_level_follows_snr():
    lo = RadarSim(Room(seed=1), snr_db=10, seed=1)
    hi = RadarSim(Room(seed=1), snr_db=40, seed=1)
    assert lo.sigma / hi.sigma == pytest.approx(10 ** (30 / 20), rel=1e-6)


def test_room_is_deterministic_per_seed():
    a, b, c = Room(seed=1), Room(seed=1), Room(seed=2)
    assert np.array_equal(a.static, b.static)
    assert not np.array_equal(a.static, c.static)


def test_calibration_masks_oscillating_clutter_bins():
    room = Room(seed=8, n_fans=0)
    room.fans = np.array([[2.0, 0.0, 1.8, 1.0, 0.03, 2.0, 0.0]])  # strong sway
    sim = RadarSim(room, snr_db=35, seed=6)
    t = np.arange(120) * 0.05
    z = np.zeros((120, N_BODY, 3)) + [3.0, 0, 1]
    cube = sim.frames(t, z, z, np.zeros(120))
    bg = calibrate(cube)
    assert (~bg.mask).any()
    assert not bg.mask[int(2.0 / RADAR.range_res)]
