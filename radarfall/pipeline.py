"""End-to-end: simulate -> train -> export ONNX -> evaluate -> results.json."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime  # noqa: F401  (fail early if missing)
from sklearn.metrics import roc_auc_score

from .config import DETECTOR, RADAR
from .dataset import say, windows_of
from .dsp import calibrate, process
from .evaluate import (eval_adl, eval_falls, summarize_falls, with_threshold)
from .features import N_FEAT, all_windows
from .model import (ONNX_PATH, OnnxScorer, paths_for, save_model, to_onnx,
                    train_gb, train_mlp)
from .radar import RadarSim, Room
from .sets import AZ_SWEEP, REL_SWEEP, SNR_SWEEP, Scale, build

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _log(msg: str):
    say(f"[{time.strftime('%H:%M:%S')}] {msg}")


def choose_threshold(scorer, va_falls, va_adl) -> tuple[float, list]:
    """Pick the lowest ML threshold with no false alarm on the validation ADL set."""
    rows = []
    for thr in (0.5, 0.7, 0.85, 0.95, 0.99):
        cfg = with_threshold(thr)
        adl = eval_adl(va_adl, scorer, "ml", cfg)
        fl = summarize_falls(eval_falls(va_falls, scorer, "ml", cfg))
        rows.append(dict(thr=thr, alarms=adl["alarms"], hours=adl["hours"],
                         sens=fl["sensitivity"]))
        _log(f"  thr {thr}: val alarms {adl['alarms']} / {adl['hours']:.2f} h, "
             f"val sensitivity {fl['sensitivity']:.2f}")
    ok = [r for r in rows if r["alarms"] == 0]
    best = (max(ok, key=lambda r: (r["sens"], -r["thr"])) if ok
            else min(rows, key=lambda r: (r["alarms"], -r["sens"])))
    return best["thr"], rows


def edge_benchmark(scorer: OnnxScorer, onnx_path: Path = ONNX_PATH,
                   n: int = 300) -> dict:
    """Per-frame DSP, per-hop features + ONNX latency, and model size on this CPU."""
    room = Room(seed=1)
    sim = RadarSim(room, seed=1)
    bg = calibrate(sim.empty_frames(60))
    cubes = sim.empty_frames(n)
    t_dsp = []
    for i in range(n):
        t0 = time.perf_counter()
        process(cubes[i: i + 1], bg)
        t_dsp.append(time.perf_counter() - t0)
    tr = np.random.default_rng(0).random((200, 10)).astype(np.float32)
    dp = np.random.default_rng(1).random((200, RADAR.n_chirps)).astype(np.float32)
    from .features import smooth_z, window_features
    zs = smooth_z(tr[:, 7], tr[:, 8])
    t_feat, t_onx = [], []
    x = window_features(tr, dp, zs, 200, 160)
    for _ in range(n):
        t0 = time.perf_counter()
        x = window_features(tr, dp, zs, 200, 160)
        t1 = time.perf_counter()
        scorer(x)
        t2 = time.perf_counter()
        t_feat.append(t1 - t0)
        t_onx.append(t2 - t1)
    ms = lambda a: float(np.median(a) * 1e3)  # noqa: E731
    return dict(
        cpu="laptop CPU, 1 thread (2013 dual-core, 1.7 GHz)",
        dsp_per_frame_ms=ms(t_dsp), dsp_per_frame_p95_ms=float(
            np.percentile(t_dsp, 95) * 1e3),
        features_per_window_ms=ms(t_feat), onnx_per_window_ms=ms(t_onx),
        frame_budget_ms=RADAR.frame_s * 1e3,
        onnx_bytes=Path(onnx_path).stat().st_size, n_features=N_FEAT)


def out_dir_for(scale_name: str) -> Path:
    return RESULTS if scale_name == "full" else RESULTS.parent / "data" / scale_name


def run(scale: Scale, workers: int | None = None, out_dir: Path | None = None) -> dict:
    t_start = time.time()
    out_dir = out_dir or out_dir_for(scale.name)
    _log(f"building sets ({scale.name})")
    sets = build(scale, workers)

    _log("windowing + training")
    xt, yt, _, _ = windows_of(sets["tr_fall"] + sets["tr_adl"])
    keep = yt >= 0
    mlp, mu, sd = train_mlp(xt[keep], yt[keep])
    gb = train_gb(xt[keep], yt[keep])
    onnx_model = to_onnx(mlp, mu, sd)
    onnx_path, meta_path = paths_for(scale.name)
    save_model(onnx_model, dict(n_features=N_FEAT, scale=scale.name), onnx_path,
               meta_path)
    scorer = OnnxScorer(onnx_path)

    xe, ye, _, _ = windows_of(sets["te_fall"] + sets["te_adl"])
    k = ye >= 0
    p_mlp = scorer(xe[k])
    p_gb = gb.predict_proba(xe[k])[:, 1]
    # parity of the exported graph with sklearn
    ref = mlp.predict_proba((xe[k][:200] - mu) / sd)[:, 1]
    parity = float(np.abs(ref - p_mlp[:200]).max())
    window_metrics = dict(auc_mlp=float(roc_auc_score(ye[k], p_mlp)),
                          auc_gb=float(roc_auc_score(ye[k], p_gb)),
                          onnx_parity_max_abs_diff=parity,
                          n_windows=int(k.sum()), n_pos=int(ye[k].sum()))
    _log(f"window AUC mlp {window_metrics['auc_mlp']:.4f} gb "
         f"{window_metrics['auc_gb']:.4f} parity {parity:.2e}")

    _log("threshold selection on validation")
    thr, thr_rows = choose_threshold(scorer, sets["va_fall"], sets["va_adl"])
    meta = dict(n_features=N_FEAT, scale=scale.name, ml_threshold=thr,
                threshold_sweep=thr_rows)
    meta_path.write_text(json.dumps(meta, indent=2))
    cfg = with_threshold(thr)

    results = dict(scale=scale.name, ml_threshold=thr, window=window_metrics,
                   modes={}, robustness={}, edge=edge_benchmark(scorer, onnx_path))
    for mode in ("rule", "ml", "both"):
        _log(f"test mode {mode}")
        fl = summarize_falls(eval_falls(sets["te_fall"], scorer, mode, cfg))
        ad = eval_adl(sets["te_adl"], scorer, mode, cfg)
        results["modes"][mode] = dict(falls=fl, adl=ad)
        _log(f"  sens {fl['sensitivity']:.3f} FA/day {ad['alarms_per_day']:.2f} "
             f"({ad['alarms']} in {ad['hours']:.1f} h)")

    _log("robustness")
    rob = results["robustness"]
    for key, name in ([(f"rob_snr{s}", f"snr_{s}") for s in SNR_SWEEP]
                      + [(f"rob_az{a}", f"az_{a}") for a in AZ_SWEEP]
                      + [(f"rob_rel{r}", f"rel_{r}") for r in REL_SWEEP]):
        fl = summarize_falls(eval_falls(sets[key], scorer, "both", cfg))
        rob[name] = dict(sensitivity=fl["sensitivity"], n=fl["n"],
                         latency_alarm_median_s=fl["latency_alarm_median_s"])
    for snr in (15, 5):
        ad = eval_adl(sets[f"rob_adl_snr{snr}"], scorer, "both", cfg)
        rob[f"adl_snr_{snr}"] = dict(alarms=ad["alarms"], hours=ad["hours"],
                                     alarms_per_day=ad["alarms_per_day"])
    results["wall_clock_min"] = (time.time() - t_start) / 60
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"results_{scale.name}.json").write_text(json.dumps(results, indent=2))
    _log(f"done in {results['wall_clock_min']:.1f} min")
    return results


__all__ = ["run", "choose_threshold", "edge_benchmark", "all_windows", "DETECTOR"]
