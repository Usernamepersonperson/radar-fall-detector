"""Detector state machine on scripted tracks, ONNX export, MQTT payloads."""
import json
from dataclasses import replace

import numpy as np
import onnx
import pytest

from radarfall.config import DETECTOR, RADAR
from radarfall.detector import ALARMED, CONFIRMING, IDLE, FallDetector, run_stream
from radarfall.dsp import TI, TRACK
from radarfall.edge import (NOTICE, SCHEMA, EdgeRunner, MqttSink, StdoutSink,
                            event_message, topic)
from radarfall.features import N_FEAT
from radarfall.model import OnnxScorer, save_model, to_onnx, train_mlp

FPS = int(RADAR.fps)


def script(segments):
    """segments: (seconds, z, speed, m_db). -> track, doppler."""
    rows = []
    for sec, z, sp, m in segments:
        for _ in range(int(sec * FPS)):
            r = np.zeros(len(TRACK), np.float32)
            r[TI["z"]], r[TI["present"]], r[TI["m_db"]] = z, 1, m
            r[TI["v_lo"]] = -sp
            rows.append(r)
    tr = np.array(rows)
    return tr, np.zeros((len(tr), RADAR.n_chirps), np.float32)


FALL = [(10, 1.0, 0.3, 4.0), (0.6, 0.55, 3.0, 20.0), (0.4, 0.15, 1.0, 8.0),
        (25, 0.12, 0.0, 0.0)]
SIT = [(10, 1.0, 0.3, 4.0), (1.0, 0.7, 1.5, 12.0), (25, 0.65, 0.0, 0.0)]
STAND_UP_AGAIN = [(10, 1.0, 0.3, 4.0), (0.6, 0.55, 3.0, 20.0), (0.4, 0.15, 1.0, 8.0),
                  (3, 0.12, 0.0, 0.0), (1.0, 0.5, 0.8, 9.0), (20, 1.0, 0.2, 3.0)]


def names(ev, name):
    return [e for e in ev if e["event"] == name]


def test_rule_alarms_after_confirmation_window():
    det = FallDetector(None, mode="rule")
    ev = run_stream(det, *script(FALL))
    alarm = names(ev, "fall_alarm")
    assert len(alarm) == 1 and alarm[0]["source"] == "rule"
    sus = names(ev, "fall_suspected")[0]
    assert alarm[0]["t"] - sus["t"] == pytest.approx(DETECTOR.confirm_s, abs=0.1)
    assert 10 <= sus["t"] <= 12.5  # suspected within ~2 s of the impact


def test_rule_ignores_sitting_down():
    det = FallDetector(None, mode="rule")
    assert names(run_stream(det, *script(SIT)), "fall_alarm") == []


def test_recovery_before_confirmation_clears():
    det = FallDetector(None, mode="rule")
    ev = run_stream(det, *script(STAND_UP_AGAIN))
    assert names(ev, "fall_suspected") and not names(ev, "fall_alarm")
    assert names(ev, "fall_cleared")
    assert det.state == IDLE


def test_cancel_while_confirming_prevents_alarm():
    det = FallDetector(None, mode="rule")
    tr, dp = script(FALL)
    out = []
    for i, (r, d) in enumerate(zip(tr, dp)):  # noqa: B007
        out += det.push(r, d)
        if det.state == CONFIRMING and det.t - det.t_cand > 2:
            out += det.cancel()
            break
    out += run_stream(det, tr[i + 1:], dp[i + 1:])
    assert names(out, "fall_cancelled") and not names(out, "fall_alarm")


def test_cancel_after_alarm_within_grace_and_latched():
    det = FallDetector(None, mode="rule")
    tr, dp = script(FALL)
    out = []
    for i, (r, d) in enumerate(zip(tr, dp)):  # noqa: B007
        out += det.push(r, d)
        if det.state == ALARMED:
            break
    assert det.state == ALARMED
    assert names(det.cancel(), "fall_cancelled")
    out += run_stream(det, tr[i + 1:], dp[i + 1:])
    assert len(names(out, "fall_alarm")) == 1  # latched: no second alarm while down


def test_cancel_after_grace_is_ignored():
    det = FallDetector(None, mode="rule")
    ev = run_stream(det, *script(FALL))
    assert names(ev, "fall_alarm") and det.state == ALARMED
    assert det.t - det.t_alarm > DETECTOR.cancel_grace_s
    assert det.cancel() == []


def test_cancel_when_idle_does_nothing():
    assert FallDetector(None, mode="rule").cancel() == []


def test_no_alarm_when_person_leaves():
    tr, dp = script(FALL)
    tr[int(14 * FPS):, TI["present"]] = 0
    tr[int(14 * FPS):, TI["z"]] = np.nan
    det = FallDetector(None, mode="rule")
    assert names(run_stream(det, tr, dp), "fall_alarm") == []


def test_ml_mode_uses_scorer_and_threshold():
    calls = []

    def scorer(x):
        calls.append(x.shape)
        return np.array([0.9 if len(calls) > 24 else 0.1])

    det = FallDetector(scorer, mode="ml")
    ev = run_stream(det, *script(FALL))
    assert calls and calls[0] == (N_FEAT,)
    assert names(ev, "fall_alarm")[0]["source"] == "ml"
    det2 = FallDetector(lambda x: np.array([0.4]), mode="ml")
    assert names(run_stream(det2, *script(FALL)), "fall_alarm") == []


def test_invalid_mode_and_missing_scorer():
    with pytest.raises(ValueError):
        FallDetector(None, mode="magic")
    with pytest.raises(ValueError):
        FallDetector(None, mode="ml")


def test_refractory_config_and_confirm_time_respected():
    cfg = replace(DETECTOR, confirm_s=4.0)
    ev = run_stream(FallDetector(None, cfg, "rule"), *script(FALL))
    sus, alarm = names(ev, "fall_suspected")[0], names(ev, "fall_alarm")[0]
    assert alarm["t"] - sus["t"] == pytest.approx(4.0, abs=0.1)


def test_event_fields():
    ev = run_stream(FallDetector(None, mode="rule"), *script(FALL))
    a = names(ev, "fall_alarm")[0]
    assert a["z_end_m"] < 0.5 and a["peak_speed_ms"] >= 2.5 and a["still_fraction"] > 0.8


# --- ONNX -------------------------------------------------------------------
@pytest.fixture(scope="module")
def trained():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, N_FEAT)).astype(np.float32)
    y = (x[:, 0] + x[:, 3] > 0.5).astype(int)
    clf, mu, sd = train_mlp(x, y)
    return x, clf, mu, sd


def test_onnx_matches_sklearn(trained, tmp_path):
    x, clf, mu, sd = trained
    m = to_onnx(clf, mu, sd)
    onnx.checker.check_model(m)
    path = tmp_path / "m.onnx"
    onnx.save(m, str(path))
    got = OnnxScorer(path)(x[:50])
    ref = clf.predict_proba((x[:50] - mu) / sd)[:, 1]
    assert np.abs(got - ref).max() < 1e-5


def test_onnx_is_small_and_batchable(trained, tmp_path):
    x, clf, mu, sd = trained
    path = tmp_path / "m.onnx"
    onnx.save(to_onnx(clf, mu, sd), str(path))
    assert path.stat().st_size < 20_000
    s = OnnxScorer(path)
    assert s(x[0]).shape == (1,) and s(x[:7]).shape == (7,)


def test_save_model_writes_meta(trained, tmp_path, monkeypatch):
    from radarfall import model

    monkeypatch.setattr(model, "META_PATH", tmp_path / "meta.json")
    save_model(to_onnx(*[trained[i] for i in (1, 2, 3)]), {"a": 1}, tmp_path / "m.onnx")
    assert json.loads((tmp_path / "meta.json").read_text()) == {"a": 1}


# --- MQTT / edge ------------------------------------------------------------
class FakeClient:
    def __init__(self):
        self.pub, self.subs, self.on_message = [], [], None

    def publish(self, t, p, qos=0):
        self.pub.append((t, p, qos))

    def subscribe(self, t):
        self.subs.append(t)


def test_event_message_is_derived_features_only():
    ev = run_stream(FallDetector(None, mode="rule"), *script(FALL))
    msg = event_message(names(ev, "fall_alarm")[0], "radar-01", ts=0)
    assert msg["schema"] == SCHEMA and msg["notice"] == NOTICE
    assert set(msg["features"]) == {"z_end_m", "peak_speed_ms", "still_fraction"}
    assert len(json.dumps(msg)) < 500
    banned = ("spectrogram", "doppler", "raw", "range_profile", "frames", "image")
    assert not any(b in json.dumps(msg).lower() for b in banned)


def test_runner_publishes_json_to_topic():
    c = FakeClient()
    sink = MqttSink("x", device_id="radar-01", client=c)
    runner = EdgeRunner(FallDetector(None, mode="rule"), sink, "radar-01")
    for r, d in zip(*script(FALL)):
        runner.frame(r, d)
    kinds = [json.loads(p)["event"] for _, p, _ in c.pub]
    assert kinds == ["fall_suspected", "fall_alarm"]
    assert all(t == topic("radar-01") == "radar-fall/radar-01/event" for t, _, _ in c.pub)
    assert all(q == 1 for _, _, q in c.pub)


def test_cancel_topic_wiring():
    c = FakeClient()
    runner = EdgeRunner(FallDetector(None, mode="rule"), None, "radar-01")
    runner.sink = MqttSink("x", device_id="radar-01", client=c, on_cancel=runner.cancel)
    assert c.subs == ["radar-fall/radar-01/cancel"]
    tr, dp = script(FALL)
    for r, d in zip(tr, dp):
        runner.frame(r, d)
        if runner.det.state == CONFIRMING:
            break
    c.on_message(None, None, None)  # a message arrives on the cancel topic
    assert json.loads(c.pub[-1][1])["event"] == "fall_cancelled"


def test_stdout_sink_prints(capsys):
    StdoutSink().publish("t", "{}")
    assert capsys.readouterr().out.strip() == "t {}"
