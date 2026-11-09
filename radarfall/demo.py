"""One-command demo: simulate, train, export, evaluate, draw the figures."""

from __future__ import annotations

import json
from pathlib import Path

from .config import DETECTOR
from .evaluate import run_recording, with_threshold
from .figures import make_main_figure, make_pair_figures
from .model import OnnxScorer, paths_for
from .pipeline import out_dir_for, run
from .scenarios import render_fall, render_sit
from .sets import Scale

DOCS = Path(__file__).resolve().parent.parent / "docs"


def pick_pair(scorer, cfg, seed0: int = 7000):
    """First simulated radial forward fall that alarms + a fast sit-down that does not."""
    fall = fall_ev = None
    for s in range(seed0, seed0 + 20):
        rec = render_fall("forward", s, rel_dir_deg=0, az_deg=0)
        ev = run_recording(rec, scorer, "both", cfg)
        if any(e["event"] == "fall_alarm" for e in ev):
            fall, fall_ev = rec, ev
            break
    for s in range(seed0, seed0 + 20):
        rec = render_sit(s, True, rel_dir_deg=0, az_deg=0)
        ev = run_recording(rec, scorer, "both", cfg)
        if not any(e["event"] == "fall_alarm" for e in ev):
            return fall, fall_ev, rec, ev
    return fall, fall_ev, rec, ev


def build_linkedin(res: dict) -> dict:
    m = res["modes"]["both"]
    f, a = m["falls"], m["adl"]
    sens = 100 * f["sensitivity"]
    lat = f["latency_alarm_median_s"]
    kb = res["edge"]["onnx_bytes"] / 1024
    n_fl = sum(v["n"] for v in a["kinds"].values())
    stats = [[f"{sens:.0f} %", f"of {f['n']} simulated falls alarmed", "accent"],
             [f"{a['alarms_per_day']:.0f}/day", f"false alarms, from {a['alarms']} in "
              f"{a['hours']:.0f} h simulated", "red"],
             [f"{lat:.0f} s", "median fall to alarm (incl. check)", "amber"]]
    post = (
        "Falls are a leading reason older people lose their independence, and the "
        "usual answers are a camera (privacy problem) or a wearable (often not worn).\n\n"
        "I built an open-source research demo of a third option: a 60 GHz FMCW radar "
        "that sees motion, not faces. It is a simulation of the whole chain: a moving "
        "point-scatterer body, radar signal synthesis with room clutter and multipath, "
        f"range and Doppler FFTs, micro-Doppler features, a {kb:.1f} KB ONNX classifier plus a "
        "rule-based check, a confirmation window and a cancel option. Only a few "
        "derived numbers leave the device as MQTT JSON, which fits the Swiss nFADP "
        "privacy idea.\n\n"
        f"Measured on simulated data: {f['hit']} of {f['n']} falls alarmed "
        f"({sens:.0f} %), {a['alarms']} false alarms in {a['hours']:.0f} hours of "
        f"daily living ({a['alarms_per_day']:.1f} per day extrapolated), median "
        f"{lat:.0f} s from impact to alarm including the confirmation wait. Sitting "
        f"down fast and lying on a bed were part of the {n_fl} fall-like events. "
        "That false-alarm rate is far too high for real use.\n\n"
        "What is not verified: everything here is simulated. No real radar, no real "
        "people, no care-home trial. It is a research demo, not a medical device.\n\n"
        "What would you check first before trusting a system like this near a "
        "person who lives alone?\n\n"
        "Code: https://github.com/Usernamepersonperson/radar-fall-detector")
    return {
        "claim": f"A simulated radar caught {sens:.0f} % of falls but raised "
                 f"{a['alarms_per_day']:.0f} false alarms a day, far too many."[:95],
        "data_label": "simulated data: 60 GHz FMCW radar model, seeded",
        "stats": stats,
        "limit": "Simulation only, and the false-alarm rate is unusable; no real radar or "
                 "people; not a medical device.",
        "alt": "Two spectrogram panels with speed on the vertical axis and time on the "
               "horizontal axis. In the top panel a short bright burst at a fast "
               "downward speed is followed by stillness and a red alarm line; the "
               "height line drops to the floor. In the bottom panel sitting down gives "
               "a similar burst but the height stays at seat level and there is no "
               "alarm.",
        "hashtags": ["#IoT", "#Radar", "#AgeTech", "#Datenschutz", "#nFADP"],
        "post": post}


def demo(scale: Scale, workers: int | None = None):
    res = run(scale, workers)
    onnx_path, meta_path = paths_for(scale.name)
    scorer = OnnxScorer(onnx_path)
    meta = json.loads(meta_path.read_text())
    cfg = with_threshold(meta.get("ml_threshold", DETECTOR.ml_threshold))
    out = DOCS if scale.name == "full" else out_dir_for(scale.name) / "docs"
    fall, fall_ev, sit, sit_ev = pick_pair(scorer, cfg)
    geo = make_pair_figures(fall, sit, fall_ev, sit_ev, out)
    make_main_figure(res, fall, sit, fall_ev, geo, out)
    li = build_linkedin(res)
    (out / "linkedin.json").write_text(json.dumps(li, indent=2, ensure_ascii=False))
    (out_dir_for(scale.name) / "results.json").write_text(json.dumps(res, indent=2))
    m = res["modes"]
    print("\n=== radar-fall-detector demo summary (simulated data) ===")
    for mode in ("rule", "ml", "both"):
        f, a = m[mode]["falls"], m[mode]["adl"]
        print(f"{mode:5s} sensitivity {100 * f['sensitivity']:.1f} % "
              f"({f['hit']}/{f['n']})  false alarms {a['alarms']} in "
              f"{a['hours']:.1f} h = {a['alarms_per_day']:.2f}/day  median alarm "
              f"latency {f['latency_alarm_median_s']} s")
    print(f"edge: {res['edge']}")
    print("figures: docs/figure.png docs/post-figure.png docs/linkedin.json")
    return res
