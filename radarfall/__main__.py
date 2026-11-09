"""CLI: ``python -m radarfall <command>``.

gen    generate + cache the simulated datasets
run    gen + train + export ONNX + evaluate -> results/results_<scale>.json
demo   run + figures + a printed alarm walk-through (``--quick`` for a smoke run)
sim    simulate one scenario through the detector and print MQTT-style JSON
"""

from __future__ import annotations

import argparse


def main(argv=None):
    ap = argparse.ArgumentParser(prog="radarfall", description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gen", "run", "demo"):
        p = sub.add_parser(name)
        p.add_argument("--quick", action="store_true", help="small smoke-test scale")
        p.add_argument("--workers", type=int, default=None)
    p = sub.add_parser("sim")
    p.add_argument("--kind", default="forward")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--mqtt", default=None, help="host[:port] of a broker (optional)")
    args = ap.parse_args(argv)

    from .sets import FULL, QUICK
    scale = QUICK if getattr(args, "quick", False) else FULL
    if args.cmd == "gen":
        from .sets import build
        build(scale, args.workers)
    elif args.cmd == "run":
        from .pipeline import run
        run(scale, args.workers)
    elif args.cmd == "demo":
        from .demo import demo
        demo(scale, args.workers)
    elif args.cmd == "sim":
        from .edge import simulate_cli
        simulate_cli(args.kind, args.seed, args.mqtt)


if __name__ == "__main__":
    main()
