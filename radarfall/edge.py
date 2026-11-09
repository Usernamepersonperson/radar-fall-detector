"""Edge runtime: detector + MQTT alarm JSON.

Privacy by design: raw ADC data, range-Doppler maps and spectrograms are processed
in memory on the device and never stored or sent. Only the event JSON below
(a handful of derived numbers) leaves the device.
"""

from __future__ import annotations

import json
import time
from typing import Protocol

from .detector import FallDetector

SCHEMA = "radar-fall/1"
NOTICE = "research demo, not a medical device"


def event_message(ev: dict, device_id: str, ts: float | None = None) -> dict:
    """Event dict from the detector -> the JSON document that is published."""
    return {
        "schema": SCHEMA,
        "device_id": device_id,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts or time.time())),
        "event": ev["event"],
        "source": ev["source"],
        "score": ev["score"],
        "t_stream_s": ev["t"],
        "features": {k: ev[k] for k in ("z_end_m", "peak_speed_ms", "still_fraction")},
        "notice": NOTICE,
    }


def topic(device_id: str, leaf: str = "event") -> str:
    return f"radar-fall/{device_id}/{leaf}"


class Sink(Protocol):
    def publish(self, topic: str, payload: str) -> None: ...


class StdoutSink:
    def publish(self, topic: str, payload: str) -> None:
        print(f"{topic} {payload}", flush=True)


class MqttSink:
    """paho-mqtt publisher; ``client`` can be injected for tests."""

    def __init__(self, host: str, port: int = 1883, device_id: str = "radar-01",
                 client=None, on_cancel=None):
        self.device_id = device_id
        if client is None:
            import paho.mqtt.client as mqtt

            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            client.connect(host, port, keepalive=30)
            client.loop_start()
        self.client = client
        if on_cancel is not None:
            client.subscribe(topic(device_id, "cancel"))
            client.on_message = lambda c, u, m: on_cancel()

    def publish(self, topic_: str, payload: str) -> None:
        self.client.publish(topic_, payload, qos=1)


class EdgeRunner:
    """Feeds frames to the detector and forwards events to a sink."""

    def __init__(self, detector: FallDetector, sink: Sink, device_id: str = "radar-01"):
        self.det, self.sink, self.device_id = detector, sink, device_id
        self.sent: list[dict] = []

    def _emit(self, events):
        for ev in events:
            msg = event_message(ev, self.device_id)
            self.sent.append(msg)
            self.sink.publish(topic(self.device_id), json.dumps(msg))

    def frame(self, track_row, dop_row):
        self._emit(self.det.push(track_row, dop_row))

    def cancel(self):
        """Wired to the MQTT cancel topic / a button."""
        self._emit(self.det.cancel())


def simulate_cli(kind: str, seed: int, mqtt: str | None):
    from .model import ONNX_PATH, OnnxScorer
    from .scenarios import render_fall

    rec = render_fall(kind, seed)
    scorer = OnnxScorer() if ONNX_PATH.exists() else None
    det = FallDetector(scorer, mode="both" if scorer else "rule")
    if mqtt:
        host, _, port = mqtt.partition(":")
        sink = MqttSink(host, int(port or 1883))
    else:
        sink = StdoutSink()
    runner = EdgeRunner(det, sink)
    for r, d in zip(rec.track, rec.doppler):
        runner.frame(r, d)
    ev = rec.events[0]
    print(f"# ground truth: {ev['kind']} impact at t={ev['t_impact']:.2f} s")
