"""Dataset definitions (seeds are disjoint between train / val / test)."""

from __future__ import annotations

from dataclasses import dataclass

from .activities import FALL_KINDS
from .dataset import adl_specs, fall_specs, generate

SNR_SWEEP = (40, 30, 20, 15, 10, 5)
AZ_SWEEP = (-50, -25, 0, 25, 50)
REL_SWEEP = (0, 45, 90)


@dataclass(frozen=True)
class Scale:
    name: str
    tr_falls: int  # per kind
    tr_adl_h: float
    va_falls: int
    va_adl_h: float
    te_falls: int
    te_adl_h: float
    rob_falls: int  # per kind per condition
    rob_adl_h: float  # per low-SNR condition
    rob_kinds: tuple = FALL_KINDS


FULL = Scale("full", 36, 2.5, 8, 1.5, 30, 8.0, 6, 1.0)
QUICK = Scale("quick", 3, 0.2, 2, 0.1, 2, 0.2, 1, 0.05, ("forward", "backward", "slow_slide"))


TRAIN_SNR = (30, 30, 25, 20, 15)  # training noise augmentation (dB)


def _vary_snr(specs: list[dict]) -> list[dict]:
    for i, sp in enumerate(specs):
        sp["snr"] = TRAIN_SNR[i % len(TRAIN_SNR)]
    return specs


def build(scale: Scale, workers: int | None = None) -> dict:
    """Generate (or load from cache) every set. Returns dict name -> recordings."""
    sd = {"tr": 10_000, "va": 20_000, "te": 30_000, "rob": 40_000}
    h = 3600.0
    out = {}
    out["tr_fall"] = generate(_vary_snr(fall_specs(scale.tr_falls, sd["tr"])), workers,
                              f"{scale.name}-tr-fall")
    out["tr_adl"] = generate(_vary_snr(adl_specs(scale.tr_adl_h * h, sd["tr"] + 500,
                                                 chunk_s=900.0)), workers,
                             f"{scale.name}-tr-adl")
    out["va_fall"] = generate(fall_specs(scale.va_falls, sd["va"]), workers,
                              f"{scale.name}-va-fall")
    out["va_adl"] = generate(adl_specs(scale.va_adl_h * h, sd["va"] + 500), workers,
                             f"{scale.name}-va-adl")
    out["te_fall"] = generate(fall_specs(scale.te_falls, sd["te"]), workers,
                              f"{scale.name}-te-fall")
    out["te_adl"] = generate(adl_specs(scale.te_adl_h * h, sd["te"] + 500), workers,
                             f"{scale.name}-te-adl")
    for snr in SNR_SWEEP:
        out[f"rob_snr{snr}"] = generate(
            fall_specs(scale.rob_falls, sd["rob"], snr=snr, kinds=scale.rob_kinds), workers,
            f"{scale.name}-rob-snr{snr}")
    for az in AZ_SWEEP:
        out[f"rob_az{az}"] = generate(
            fall_specs(scale.rob_falls, sd["rob"], az=az, kinds=scale.rob_kinds), workers,
            f"{scale.name}-rob-az{az}")
    for rel in REL_SWEEP:
        out[f"rob_rel{rel}"] = generate(
            fall_specs(scale.rob_falls, sd["rob"], rel=rel, kinds=scale.rob_kinds), workers,
            f"{scale.name}-rob-rel{rel}")
    for snr in (15, 5):
        out[f"rob_adl_snr{snr}"] = generate(
            adl_specs(scale.rob_adl_h * h, sd["rob"] + 800, snr=snr), workers,
            f"{scale.name}-rob-adl-snr{snr}")
    return out
