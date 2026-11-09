"""Radar and detector parameters. All numbers here are documented in the README."""

from __future__ import annotations

from dataclasses import dataclass

C = 299_792_458.0


@dataclass(frozen=True)
class RadarConfig:
    """60 GHz FMCW radar, one TX, two RX stacked vertically (lambda/2 apart).

    A frame is a burst of ``n_chirps`` chirps; one frame every ``frame_s``.
    Derived values (range/velocity resolution, limits) are properties.
    """

    fc: float = 60.0e9  # carrier, Hz
    bandwidth: float = 1.5e9  # swept bandwidth, Hz
    t_ramp: float = 40e-6  # chirp ramp time, s
    t_chirp: float = 250e-6  # chirp repetition interval (PRI), s
    n_samples: int = 64  # ADC samples per chirp (complex IQ)
    n_chirps: int = 32  # chirps per frame
    frame_s: float = 0.05  # frame period (20 Hz)
    mount_height: float = 1.8  # wall mount height, m
    az_fov_deg: float = 65.0  # one-sided azimuth field of view

    @property
    def wavelength(self) -> float:
        return C / self.fc

    @property
    def slope(self) -> float:
        return self.bandwidth / self.t_ramp

    @property
    def fs(self) -> float:
        return self.n_samples / self.t_ramp

    @property
    def range_res(self) -> float:
        return C / (2 * self.bandwidth)

    @property
    def max_range(self) -> float:
        return self.range_res * self.n_samples

    @property
    def vel_res(self) -> float:
        return self.wavelength / (2 * self.n_chirps * self.t_chirp)

    @property
    def vel_max(self) -> float:
        return self.wavelength / (4 * self.t_chirp)

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_s


@dataclass(frozen=True)
class DetectorConfig:
    """Streaming detector settings (times in seconds, heights in metres)."""

    window_s: float = 8.0  # classifier window
    hop_s: float = 0.5  # classifier evaluation interval
    ml_threshold: float = 0.5  # set by training (see models/meta.json)
    confirm_s: float = 8.0  # must stay low this long before the alarm
    z_low: float = 0.50  # "on the floor" height proxy
    z_recover: float = 0.85  # upright again -> clear
    cancel_grace_s: float = 5.0  # cancel still accepted after the alarm
    refractory_s: float = 60.0  # no second alarm for this long
    rule_drop_m: float = 0.40  # rule: height drop within rule_span_s
    rule_span_s: float = 1.5
    rule_speed_ms: float = 1.0  # rule: peak Doppler speed during the drop
    rule_still_frac: float = 0.5  # rule: fraction of "still" frames after


RADAR = RadarConfig()
DETECTOR = DetectorConfig()
