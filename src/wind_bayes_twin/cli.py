"""Command line interface.

Usage:
    python -m wind_bayes_twin prepare            # read the archives, reference, thermal models, daily statistics
    python -m wind_bayes_twin calibrate          # hierarchical noise model (NUTS), dispersion, chart thresholds
    python -m wind_bayes_twin validate --workers 6 [--engine block|particle [--unguided]]
    python -m wind_bayes_twin cases              # status-log events vs the twin on unmodified data
    python -m wind_bayes_twin report             # tables, figures, dashboard
    python -m wind_bayes_twin all

The Kelmarsh archives are read from ``$KELMARSH_DATA_DIR/raw`` or ``<data-dir>/raw``; caches go to
``<data-dir>/processed``.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from .data import raw_dir_default

log = logging.getLogger("wind_bayes_twin")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="wind_bayes_twin", description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["prepare", "calibrate", "validate", "cases", "report", "all"])
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--docs-dir", type=Path, default=Path("docs"))
    ap.add_argument("--engine", choices=["block", "particle"], default="block")
    ap.add_argument("--particles", type=int, default=2000)
    ap.add_argument("--unguided", action="store_true", help="particle engine without change-point guidance")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--refresh", action="store_true", help="recompute cached stages")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from . import pipeline

    raw, cache = raw_dir_default(args.data_dir), args.data_dir / "processed"
    cache.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    cmds = ["prepare", "calibrate", "validate", "cases", "report"] if args.command == "all" else [args.command]
    for cmd in cmds:
        if cmd == "prepare":
            pipeline.prepare(raw, cache, refresh=args.refresh)
        elif cmd == "calibrate":
            pipeline.calibrate(pipeline.prepare(raw, cache), cache, args.results_dir, refresh=args.refresh)
        elif cmd == "validate":
            from .experiments import validate

            validate(cache, args.results_dir, engine=args.engine, workers=args.workers, n_particles=args.particles,
                     guided=not args.unguided)
        elif cmd == "cases":
            from .experiments import cases

            cases(cache, args.results_dir)
        elif cmd == "report":
            from .report import report

            report(cache, args.results_dir, args.docs_dir)
        log.info("%s done (%.0f s so far)", cmd, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
