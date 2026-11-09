import numpy as np
import pytest

from radarfall.activities import ADL_FALL_LIKE, FALL_KINDS, Actor, in_room
from radarfall.motion import POSE, STAND, Timeline, body_points


def render_pose(**kw):
    tl = Timeline()
    for k, v in kw.items():
        tl.p[0][POSE.index(k)] = v
    r = tl.render(np.array([0.0]))
    return body_points(r)[0]


def test_standing_geometry():
    p = render_pose(x=3.0, y=0.0)
    assert 1.5 < p[0, 2] < 1.8  # head
    assert p[10:, 2].max() < 0.6  # shanks low
    assert p[:, 2].min() > 0.0
    assert set(STAND) == set(POSE)


def test_timeline_interpolates_and_eases():
    tl = Timeline()
    tl.add(2.0, "linear", x=4.0)
    tl.add(1.0, "smooth", x=0.0)
    r = tl.render(np.array([0.0, 1.0, 2.0, 2.5, 3.0, 9.0]))
    assert r["x"] == pytest.approx([0, 2, 4, 2, 0, 0])


def test_accel_ease_starts_slow_ends_fast():
    tl = Timeline()
    tl.add(1.0, "accel", Lb=1.5)
    r = tl.render(np.array([0.25, 0.5, 0.75, 1.0]))
    d = np.diff(np.r_[0, r["Lb"]])
    assert np.all(np.diff(d) > 0)


def test_lying_pose_is_low_and_extended():
    p = render_pose(Lb=1.5, pivot=1.0, D=0.0, x=3.0)
    assert p[:, 2].max() < 0.35
    assert p[0, 0] - p[10, 0] > 1.0  # head far from the feet along the lean


def test_pivot_switch_is_continuous_at_zero_tilt():
    a = render_pose(pivot=0.0)
    b = render_pose(pivot=1.0)
    assert np.allclose(a, b, atol=1e-6)


def test_gait_overlay_fades_and_swings_legs():
    tl = Timeline()
    tl.gait.append((1.0, 5.0, 1.0, 1.0))
    tl.add(6.0, "linear")
    r = tl.render(np.array([0.5, 1.0, 3.25, 5.5]))
    assert r["gait"][0, 0] == 0 and r["gait"][1, 0] == 0 and r["gait"][2, 0] == 1
    p = body_points(tl.render(np.linspace(1.5, 3.0, 30)))
    assert np.ptp(p[:, 8, 0] - p[:, 9, 0]) > 0.1  # thighs swing apart


def test_absent_flag():
    tl = Timeline()
    tl.absent.append((1.0, 2.0))
    tl.add(3.0, "linear")
    assert tl.render(np.array([0.5, 1.5, 2.5]))["present"].tolist() == [1, 0, 1]


def test_in_room_clamps():
    x, y = in_room(10, 10)
    assert x <= 4.2 and abs(y) < 10


@pytest.mark.parametrize("kind", FALL_KINDS)
def test_every_fall_ends_on_the_floor_with_annotations(kind):
    rng = np.random.default_rng(3)
    a = Actor(rng, 3.0, 0.0, 0.3)
    if kind == "from_chair":
        a.tl.p[0][2], a.tl.p[0][7:9], a.posture = 0.5, 1.45, "sit"
    a.idle(2.0)
    a.fall(kind)
    a.lie_still(3.0)
    ev = [e for e in a.tl.events if e["fall"]]
    assert len(ev) == 1 and ev[0]["t_onset"] < ev[0]["t_impact"] <= ev[0]["t_end"]
    end = body_points(a.tl.render(np.array([a.tl.now - 0.1])))[0]
    assert end[[0, 1, 2, 3, 8, 9], 2].max() < 0.45  # head, trunk, pelvis, thighs


def test_fall_speed_ordering_slow_slide_is_slower():
    rng = np.random.default_rng(1)
    durs = {}
    for kind in ("forward", "slow_slide"):
        a = Actor(rng, 3.0, 0.0, 0.0)
        a.idle(1.0)
        a.fall(kind)
        e = a.tl.events[0]
        durs[kind] = e["t_impact"] - e["t_onset"]
    assert durs["slow_slide"] > 1.5 * durs["forward"]


def test_invalid_transition_raises():
    a = Actor(np.random.default_rng(0))
    with pytest.raises(AssertionError):
        a.stand_up()  # not sitting


def test_sit_down_ends_at_seat_height_and_fast_is_faster():
    d = {}
    for fast in (False, True):
        a = Actor(np.random.default_rng(2))
        a.idle(1)
        a.sit_down(fast)
        d[fast] = a.tl.events[0]["t_end"] - a.tl.events[0]["t_onset"]
        assert a.tl.last()["zp"] == pytest.approx(0.5)
    assert d[True] < d[False]


def test_stumble_recovers_upright():
    a = Actor(np.random.default_rng(5))
    a.stumble_recover()
    last = a.tl.last()
    assert last["Lb"] == 0 and last["pivot"] == 0
    assert all(not e["fall"] for e in a.tl.events)


def test_adl_stream_has_no_falls_and_covers_fall_like_kinds():
    from radarfall.scenarios import adl_timeline

    kinds = set()
    for seed in range(6):
        tl = adl_timeline(3600, np.random.default_rng(seed))
        assert tl.now >= 3600
        assert all(not e["fall"] for e in tl.events)
        kinds |= {e["kind"] for e in tl.events}
    assert kinds <= set(ADL_FALL_LIKE)
    assert {"sit_fast", "bend_pick", "bed_lie"} <= kinds
