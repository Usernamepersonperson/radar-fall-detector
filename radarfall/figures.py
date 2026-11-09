"""Figures: micro-Doppler spectrograms (fall vs sitting down) and result panels."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .config import RADAR  # noqa: E402
from .dsp import TI  # noqa: E402
from .features import smooth_z  # noqa: E402

BG, GRID, TEXT, MUTED = "#16213A", "#2A3750", "#F8FAFC", "#A9B6CB"
ACCENT, GREEN, AMBER, RED, GREY, VIOLET = ("#38BDF8", "#34D399", "#FBBF24", "#F87171",
                                           "#64748B", "#A78BFA")
CMAP = LinearSegmentedColormap.from_list(
    "radar", [BG, "#1D3E78", ACCENT, "#BDE9FB", "#FDE68A", TEXT])
DOCS = Path(__file__).resolve().parent.parent / "docs"
V_LIM = 4.0


def style():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "text.color": TEXT,
        "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": GRID, "axes.facecolor": BG, "figure.facecolor": BG,
        "savefig.facecolor": BG, "axes.grid": False, "font.size": 11})


def _style_ax(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED)


def spectrogram_panel(ax, rec, t0, t1, alarm=None, suspected=None, impact=None,
                      vmax=None, label_marks=True, ylabel=True):
    """Micro-Doppler spectrogram + height proxy line + event markers."""
    i0, i1 = int(t0 * RADAR.fps), int(t1 * RADAR.fps)
    d = 10 * np.log10(1 + rec.doppler[i0:i1])
    vmax = vmax or float(np.percentile(d, 99.7))
    ax.imshow(d.T, origin="lower", aspect="auto", cmap=CMAP, vmin=0, vmax=vmax,
              extent=(t0, t1, -(RADAR.n_chirps // 2) * RADAR.vel_res,
                      (RADAR.n_chirps // 2) * RADAR.vel_res), interpolation="nearest")
    ax.set_ylim(-V_LIM, V_LIM)
    ax.set_xlim(t0, t1)
    if ylabel:
        ax.set_ylabel("radial speed (m/s)", fontsize=11)
    ax.axhline(0, color=GRID, lw=0.8)
    z = smooth_z(rec.track[:, TI["z"]], rec.track[:, TI["present"]])[i0:i1]
    tz = np.arange(i0, i1) / RADAR.fps
    ax2 = ax.twinx()
    ax2.plot(tz, z, color=GREEN, lw=2.2)
    ax2.set_ylim(0, 2.0)
    ax2.set_yticks([0, 0.5, 1.0, 1.5])
    ax2.tick_params(colors=GREEN, labelsize=10)
    ax2.set_ylabel("height proxy (m)", color=GREEN, fontsize=11)
    for s in ax2.spines.values():
        s.set_color(GRID)
    top = V_LIM * 0.86
    if impact is not None:
        ax.axvline(impact, color=MUTED, ls=":", lw=1.6)
    if suspected is not None:
        ax.axvline(suspected, color=AMBER, ls="--", lw=1.8)
    if alarm is not None:
        ax.axvline(alarm, color=RED, lw=2.6)
        if label_marks:
            ax.text(alarm - 0.15, top, "ALARM", color=RED, ha="right", fontsize=13,
                    fontweight="bold")
    ax.set_xlabel("time (s)", fontsize=11)
    _style_ax(ax)
    return ax2


def make_pair_figures(fall_rec, sit_rec, fall_ev, sit_ev, out_dir: Path = DOCS):
    """Hook image (post-figure.png, 960x700) and the left half of figure.png."""
    style()
    ev = fall_rec.events[0]
    tf0 = max(0.0, ev["t_impact"] - 3.0)
    tf1 = tf0 + 15.0
    es = sit_rec.events[0]
    ts0 = max(0.0, es["t_onset"] - 3.0)
    ts1 = ts0 + 15.0
    alarm = next((e["t"] for e in fall_ev if e["event"] == "fall_alarm"), None)
    susp = next((e["t"] for e in fall_ev if e["event"] == "fall_suspected"), None)
    sit_alarm = any(e["event"] == "fall_alarm" for e in sit_ev)
    seg = fall_rec.doppler[int(tf0 * 20):int(tf1 * 20)]
    vmax = float(np.percentile(10 * np.log10(1 + seg), 99.7))

    fig, axs = plt.subplots(2, 1, figsize=(9.6, 7.0), dpi=100, facecolor=BG)
    fig.subplots_adjust(left=0.085, right=0.915, top=0.745, bottom=0.095, hspace=0.62)
    spectrogram_panel(axs[0], fall_rec, tf0, tf1, alarm=alarm, suspected=susp,
                      impact=ev["t_impact"], vmax=vmax)
    spectrogram_panel(axs[1], sit_rec, ts0, ts1, alarm=None, vmax=vmax,
                      impact=es["t_end"])
    axs[0].set_title("Fall: fast descent, then lying still", loc="left", fontsize=13,
                     color=TEXT, pad=8)
    axs[1].set_title("Sitting down fast: same speed burst, no alarm" if not sit_alarm
                     else "Sitting down fast: FALSE ALARM", loc="left", fontsize=13,
                     color=TEXT if not sit_alarm else RED, pad=8)
    fig.text(0.04, 0.945, "Radar tells a fall from sitting down - without a camera",
             fontsize=14, fontweight="bold", ha="left", va="center")
    fig.text(0.04, 0.895, "Simulated 60 GHz FMCW radar, micro-Doppler spectrogram. "
             "Research demo, not a medical device.", fontsize=11, color=MUTED,
             ha="left", va="center")
    handles = [Line2D([], [], color=GREEN, lw=2.2, label="height proxy"),
               Line2D([], [], color=MUTED, ls=":", lw=1.6, label="impact"),
               Line2D([], [], color=AMBER, ls="--", lw=1.8, label="suspected"),
               Line2D([], [], color=RED, lw=2.6, label="alarm")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.04, 0.862),
               ncol=4, frameon=False, fontsize=11, labelcolor=MUTED,
               handlelength=1.8, columnspacing=1.6)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "post-figure.png", dpi=100)
    plt.close(fig)
    return dict(tf0=tf0, tf1=tf1, ts0=ts0, ts1=ts1, alarm=alarm, susp=susp,
                impact=ev["t_impact"], vmax=vmax)


def make_main_figure(results: dict, fall_rec, sit_rec, fall_ev, geo: dict,
                     out_dir: Path = DOCS):
    """docs/figure.png (1600x900): spectrograms + measured results."""
    style()
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(2, 3, left=0.055, right=0.965, top=0.83, bottom=0.085,
                          hspace=0.55, wspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    spectrogram_panel(ax, fall_rec, geo["tf0"], geo["tf1"], alarm=geo["alarm"],
                      suspected=geo["susp"], impact=geo["impact"], vmax=geo["vmax"])
    ax.set_title("Fall (simulated)", loc="left", fontsize=13, pad=6)
    es = sit_rec.events[0]
    ax = fig.add_subplot(gs[0, 1])
    spectrogram_panel(ax, sit_rec, geo["ts0"], geo["ts1"], vmax=geo["vmax"],
                      impact=es["t_end"])
    ax.set_title("Sitting down fast: no alarm", loc="left", fontsize=13, pad=6)

    both = results["modes"]["both"]
    # fall-like activities
    ax = fig.add_subplot(gs[0, 2])
    kinds = [(k, v) for k, v in both["adl"]["kinds"].items() if v["n"] > 0]
    labels = [k.replace("_", " ") for k, _ in kinds]
    y = np.arange(len(kinds))
    ax.barh(y + 0.2, [100 * v["suspected"] / v["n"] for _, v in kinds], height=0.38,
            color=AMBER, label="suspected")
    ax.barh(y - 0.2, [100 * v["alarmed"] / v["n"] for _, v in kinds], height=0.38,
            color=RED, label="alarm")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{lb} (n={v['n']})" for lb, (_, v) in zip(labels, kinds)],
                       fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of events triggering", fontsize=11)
    ax.set_title("Fall-like activities", loc="left", fontsize=13, pad=6)
    ax.legend(frameon=False, fontsize=10, labelcolor=MUTED, loc="lower right",
              bbox_to_anchor=(1.0, 1.09), ncol=2)
    ax.grid(axis="x", color=GRID, lw=0.6)
    _style_ax(ax)

    # sensitivity per fall kind
    ax = fig.add_subplot(gs[1, 0])
    pk = both["falls"]["per_kind"]
    ks = list(pk)
    ax.bar(range(len(ks)), [100 * pk[k]["sensitivity"] for k in ks], color=ACCENT)
    ax.set_xticks(range(len(ks)))
    short = {"forward": "fwd", "backward": "back", "side_left": "left",
             "side_right": "right", "walking_trip": "trip", "from_chair": "chair",
             "slow_slide": "slow"}
    ax.set_xticklabels([short.get(k.replace("fall_", ""), k) for k in ks], fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% alarmed", fontsize=11)
    ax.set_title(f"Sensitivity by fall type (n={both['falls']['n']})", loc="left",
                 fontsize=13, pad=6)
    ax.grid(axis="y", color=GRID, lw=0.6)
    _style_ax(ax)

    rob = results["robustness"]
    ax = fig.add_subplot(gs[1, 1])
    snrs = sorted(int(k.split("_")[1]) for k in rob if k.startswith("snr_"))
    ax.plot(snrs, [100 * rob[f"snr_{s}"]["sensitivity"] for s in snrs], "-o",
            color=GREEN, lw=2.4, ms=7)
    ax.set_xlabel("SNR of a reference torso at 3 m (dB)", fontsize=11)
    ax.set_ylabel("% alarmed", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_title("Robustness to noise", loc="left", fontsize=13, pad=6)
    ax.grid(color=GRID, lw=0.6)
    _style_ax(ax)

    ax = fig.add_subplot(gs[1, 2])
    azs = sorted(int(k.split("_")[1]) for k in rob if k.startswith("az_"))
    ax.plot(azs, [100 * rob[f"az_{a}"]["sensitivity"] for a in azs], "-o",
            color=VIOLET, lw=2.4, ms=7)
    ax.set_xlabel("person azimuth off boresight (deg)", fontsize=11)
    ax.set_ylabel("% alarmed", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_title("Robustness to position", loc="left", fontsize=13, pad=6)
    ax.grid(color=GRID, lw=0.6)
    _style_ax(ax)

    a = both["adl"]
    fig.text(0.02, 0.955, "radar-fall-detector: fall detection with no camera",
             fontsize=19, fontweight="bold", ha="left")
    fig.text(0.02, 0.915,
             f"Simulated 60 GHz FMCW radar. {both['falls']['hit']}/{both['falls']['n']}"
             f" simulated falls alarmed; {a['alarms']} false alarms in {a['hours']:.0f}"
             " h of simulated daily living. Research demo, not a medical device.",
             fontsize=12, color=MUTED, ha="left")
    fig.text(0.02, 0.872, "Green line: height proxy from the 2-RX elevation phase. "
             "Dotted: impact. Dashed amber: suspected. Red: alarm.", fontsize=11,
             color=MUTED, ha="left")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figure.png", dpi=100)
    plt.close(fig)
