import numpy as np
import pytest

from radarfall.config import RADAR
from radarfall.dataset import adl_specs, fall_specs, label_windows
from radarfall.dsp import TI, TRACK
from radarfall.features import N_FEAT, NAMES, all_windows, smooth_z, window_features


def fake_track(n, z=1.0):
    t = np.zeros((n, len(TRACK)), np.float32)
    t[:, TI["z"]] = z
    t[:, TI["present"]] = 1
    return t, np.zeros((n, RADAR.n_chirps), np.float32)


def test_names_match_feature_count():
    assert len(NAMES) == N_FEAT == len(set(NAMES))


def test_smooth_z_forward_fills_and_is_causal():
    z = np.array([1, 1, np.nan, np.nan, 1, 1, 1, 0.2, 0.2, 0.2, 0.2, 0.2], np.float32)
    pres = np.isfinite(z).astype(np.float32)
    s = smooth_z(z, pres)
    assert np.all(np.isfinite(s))
    assert s[3] == pytest.approx(1.0)
    # median of 5 reacts only once 3 of the last 5 frames are low
    assert s[7] > 0.9 and s[9] == pytest.approx(0.2)
    s2 = smooth_z(z[:8], pres[:8])
    assert np.array_equal(s[:8], s2)  # causality: the future does not change the past


def test_smooth_z_all_missing_uses_default():
    s = smooth_z(np.full(10, np.nan, np.float32), np.zeros(10, np.float32))
    assert np.all(s == pytest.approx(0.9))


def test_window_features_shape_and_finite():
    tr, dp = fake_track(300)
    zs = smooth_z(tr[:, TI["z"]], tr[:, TI["present"]])
    x = window_features(tr, dp, zs, 300, 160)
    assert x.shape == (N_FEAT,) and np.isfinite(x).all()


def test_fast_descent_features():
    tr, dp = fake_track(300)
    tr[200:, TI["z"]] = 0.15
    tr[195:205, TI["v_lo"]] = -3.0
    zs = smooth_z(tr[:, TI["z"]], tr[:, TI["present"]])
    x = dict(zip(NAMES, window_features(tr, dp, zs, 300, 160)))
    assert x["peak_speed"] == pytest.approx(3.0)
    assert x["z_drop"] > 0.7 and x["z_end"] == pytest.approx(0.15)
    assert x["desc_rate"] < -1.0
    assert x["still_frac"] == 1.0


def test_all_windows_hop_and_count():
    tr, dp = fake_track(400)
    ends, x = all_windows(tr, dp, 8.0, 0.5)
    assert ends[0] == 160 and np.all(np.diff(ends) == 10) and len(x) == len(ends)
    ends, x = all_windows(*fake_track(100))
    assert len(ends) == 0 and x.shape == (0, N_FEAT)


def test_labels_positive_ambiguous_negative():
    ends = np.arange(160, 1200, 10)
    ev = [dict(fall=True, t_onset=20.0, t_impact=21.0, t_end=21.5, kind="x")]
    y = label_windows(ends, ev)
    te = ends / 20
    assert np.all(y[(te >= 22.5) & (te <= 27)] == 1)
    assert np.all(y[(te >= 20) & (te < 22.5)] == -1)
    assert np.all(y[(te > 27) & (te <= 35)] == -1)
    assert np.all(y[te < 20] == 0) and np.all(y[te > 35] == 0)


def test_labels_ignore_non_falls():
    y = label_windows(np.arange(160, 400, 10),
                      [dict(fall=False, t_onset=10, t_impact=11, t_end=11, kind="y")])
    assert (y == 0).all()


def test_specs_are_disjoint_and_sized():
    f = fall_specs(3, 100)
    assert len(f) == 21 and len({s["seed"] for s in f}) == 21
    a = adl_specs(4000, 7, 1800)
    assert [s["dur"] for s in a] == [1800, 1800, 400]


def test_cache_roundtrip(tmp_path):
    from radarfall import dataset
    from radarfall.scenarios import Recording

    rec = Recording(np.ones((5, len(TRACK)), np.float32),
                    np.ones((5, 32), np.float32), [dict(kind="k")], dict(a=1))
    p = tmp_path / "x.npz"
    dataset._save(p, [rec])
    out = dataset._load(p)[0]
    assert out.events == rec.events and np.allclose(out.track, rec.track)
