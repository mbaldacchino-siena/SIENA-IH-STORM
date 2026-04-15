# -*- coding: utf-8 -*-
"""
SIENA-IH-STORM parallel generation entry point.

Parallelizes across (basin, phase, loop_index) combinations using
multiprocessing. Each worker is fully independent: reads shared
coefficient files (read-only) and writes its own output file.

Usage examples:
    # 10,000 years for NA La Nina, using 10 parallel loops of 1000 years each
    python MASTER_storm.py --phase LN --basins NA --years 1000 --loop 10 --workers 10

    # All 3 phases for NA, 10k years each, 50 workers
    python MASTER_storm.py --phase ALL --basins NA --years 1000 --loop 10 --workers 50

    # Full run: 3 phases x 2 basins x 10 loops = 60 jobs across 50 workers
    python MASTER_storm.py --phase ALL --basins NA WP --years 1000 --loop 10 --workers 50
"""

import argparse
import numpy as np
import os
import time
import multiprocessing as mp
from functools import partial


__location__ = os.path.realpath(os.getcwd())
dir_path = __location__

def _run_single_job(job, years_per_loop, use_yearly=True, forecast_config_path=None):
    """
    Worker function: generates one loop of synthetic storms.
    Called in a separate process -- must import everything locally
    to avoid pickling issues.
    """
    basin, phase, loop_idx = job

    # ---- Each worker gets a unique random seed ----
    seed_base = hash((basin, phase, loop_idx)) % (2**31)
    np.random.seed(seed_base)
    import random

    random.seed(seed_base)

    # ---- Local imports (required for multiprocessing) ----
    from CODE.SELECT_BASIN import Basins_WMO, Basins_WMO_forecast
    from CODE.SAMPLE_STARTING_POINT import Startingpoint
    from CODE.SAMPLE_TC_MOVEMENT import TC_movement
    from CODE.SAMPLE_TC_PRESSURE import TC_pressure
    from CODE.siena_utils import load_env_pool, draw_env_years_for_season

    # ---- Load year pool (tiny JSON, fast) ----
    env_pool = None
    if use_yearly:
        env_pool = load_env_pool(__location__)
        if not env_pool:
            env_pool = None

    # ── Deterministic active-season months per basin ──
    # Must NOT come from Basins_WMO (which draws a random Poisson count and
    # random genesis months — incomplete and non-reproducible).
    # Read from input.dat to stay in sync with preprocessing.
    import CODE.import_data as import_data
    _BASIN_NAMES = ["EP", "NA", "NI", "SI", "SP", "WP"]
    _input = import_data.input_data("input.dat")
    _months_all = _input[4]  # months is the 5th return value
    _basin_idx = _BASIN_NAMES.index(basin)
    active_months = _months_all[_basin_idx]

    # ---- Forecast mode setup ----
    forecast_cfg = None
    month_phases = None
    poisson_phase_rate = None
    genesis_months_phase = None

    if forecast_config_path is not None:
        from FORECAST.forecast_config import (
            load_forecast_config,
            build_forecast_env_years,
            get_month_phases,
        )

        forecast_cfg = load_forecast_config(os.path.join(__location__,forecast_config_path))
        month_phases = get_month_phases(forecast_cfg, active_months)

        # Load phase-specific genesis parameters
        poisson_phase_rate = np.load(
            os.path.join(__location__, "POISSON_GENESIS_PARAMETERS_PHASE.npy"),
            allow_pickle=True,
        ).item()
        genesis_months_phase = np.load(
            os.path.join(__location__, "GENESIS_MONTHS_PHASE.npy"),
            allow_pickle=True,
        ).item()



    pid = os.getpid()
    print(
        f"[PID {pid}] Starting: basin={basin} phase={phase} loop={loop_idx} ({years_per_loop} years)"
        f"{' [year resampling]' if env_pool else ''}"
    )
    t0 = time.time()

    TC_data = []
    for year in range(years_per_loop):

        if forecast_cfg is not None:
            # ── FORECAST MODE ──
            # Blended genesis: single Poisson + multinomial across months
            storms_per_year, genesis_month, lat0, lat1, lon0, lon1 = (
                Basins_WMO_forecast(
                    basin,
                    month_phases,
                    poisson_phase_rate=poisson_phase_rate,
                    genesis_months_phase=genesis_months_phase,
                    active_months=active_months,
                )
            )

            # Build env_years from forecast config
            # "historical" months get a fresh resample each synthetic year
            env_years = build_forecast_env_years(
                forecast_cfg, env_pool or {}, active_months
            )

        else:
            # ── ORIGINAL MODE ──
            storms_per_year, genesis_month, lat0, lat1, lon0, lon1 = Basins_WMO(
                basin, phase=phase
            )

            # ── Draw historical years per month for this simulated year ──
            env_years = None
            if env_pool is not None:
                env_years = draw_env_years_for_season(env_pool, phase, active_months)

        if storms_per_year > 0:
            lon_genesis_list, lat_genesis_list = Startingpoint(
                storms_per_year, genesis_month, basin, phase=phase
            )
            latlist, lonlist, landfalllist = TC_movement(
                lon_genesis_list,
                lat_genesis_list,
                basin,
                phase=phase,
                monthlist=genesis_month,
                env_years=env_years,
            )
            TC_data = TC_pressure(
                basin,
                latlist,
                lonlist,
                landfalllist,
                year,
                storms_per_year,
                genesis_month,
                TC_data,
                phase=phase,
                env_years=env_years,
            )

    TC_data = np.array(TC_data)
    # ── Output filename includes forecast tag ──
    if forecast_cfg is not None:
        member = forecast_cfg.get("ensemble_member", 0)
        out = (
            f"STORM_DATA_IBTRACS_{basin}_FORECAST_m{member}"
            f"_{years_per_loop}_YEARS_{loop_idx}.txt"
        )
    else:
        out = f"STORM_DATA_IBTRACS_{basin}_{phase}_{years_per_loop}_YEARS_{loop_idx}.txt"
    outpath = os.path.join(__location__, out)
    if len(TC_data) > 0:
        TC_data = np.nan_to_num(TC_data, nan=0.0)
        np.savetxt(outpath, TC_data, fmt="%5s", delimiter=",")

    elapsed = time.time() - t0
    n_storms = len(np.unique(TC_data[:, 2])) if len(TC_data) > 0 else 0
    print(
        f"[PID {pid}] Done: {basin}/{phase}/loop{loop_idx} -> "
        f"{n_storms} storms in {elapsed:.0f}s -> {out}"
    )
    return outpath


def main():
    parser = argparse.ArgumentParser(
        description="SIENA-IH-STORM parallel synthetic TC generation"
    )
    parser.add_argument(
        "--phase",
        default="ALL",
        choices=["LN", "NEU", "EN", "ALL"],
        help="ENSO phase (ALL runs all three phases)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=1000,
        help="Years per loop (each loop produces one output file)",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=10,
        help="Number of loops (output files) per basin/phase",
    )
    parser.add_argument(
        "--basins", nargs="*", default=["NA"], help="Basins to generate"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )
    parser.add_argument(
        "--no-yearly",
        action="store_true",
        help="Disable year resampling (use phase-mean fields)",
    )

    parser.add_argument(
        "--forecast",
        type=str,
        default=None,
        help="Path to forecast_config.json. Enables seasonal forecast mode. "
        "When set, --phase is ignored (phases come from the config).",
    )

    args = parser.parse_args()


    # ── Forecast mode: override phase handling ──
    if args.forecast is not None:
        # In forecast mode, phase is not used for genesis — month_phases
        # from the config drive everything. We still run ONE phase label
        # for output naming. Use "FCST" as a pseudo-phase.
        phases = ["FCST"]
    elif args.phase == "ALL":
        phases = ["LN", "NEU", "EN"]
    else:
        phases = [args.phase]

    # Build job list: all (basin, phase, loop_idx) combinations
    jobs = []
    for basin in args.basins:
        for phase in phases:
            for loop_idx in range(args.loop):
                jobs.append((basin, phase, loop_idx))

    n_workers = args.workers or min(len(jobs), mp.cpu_count())
    total_years = args.years * args.loop * len(phases) * len(args.basins)

    print("=" * 70)
    print("SIENA-IH-STORM Parallel Generation")
    print("=" * 70)
    print(f"Basins:      {args.basins}")
    print(f"Phases:      {phases}")
    print(f"Years/loop:  {args.years}")
    print(f"Loops:       {args.loop}")
    print(f"Total jobs:  {len(jobs)}")
    print(f"Total years: {total_years:,}")
    print(f"Workers:     {n_workers}")
    print(f"Year resamp: {'disabled' if args.no_yearly else 'enabled'}")
    print("=" * 70)

    start_time = time.time()

    # ---- Run in parallel ----
    worker_fn = partial(
        _run_single_job,
        years_per_loop=args.years,
        use_yearly=not args.no_yearly,
        forecast_config_path=args.forecast,
    )

    if n_workers == 1:
        # Sequential mode (useful for debugging)
        results = [worker_fn(job) for job in jobs]
    else:
        # imap_unordered yields results as each job finishes, so you see
        # progress immediately instead of waiting for ALL jobs to complete.
        # pool.map blocked on the slowest worker — if one storm entered an
        # infinite loop, no results were returned and no output was visible.
        results = []
        with mp.Pool(processes=n_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(worker_fn, jobs)):
                results.append(result)
                elapsed_so_far = time.time() - start_time
                print(
                    f"  [{i + 1}/{len(jobs)} done] "
                    f"{os.path.basename(result) if result else '(empty)'} "
                    f"  ({elapsed_so_far / 60:.1f} min elapsed)"
                )

    elapsed = time.time() - start_time
    print()
    print("=" * 70)
    print(
        f"All done. {len(results)} files generated in {elapsed:.0f}s "
        f"({elapsed / 60:.1f} min)"
    )
    print(f"Throughput: {total_years / elapsed:.0f} years/second")
    print("=" * 70)

    # List output files
    for r in sorted(results):
        if r and os.path.exists(r):
            size_mb = os.path.getsize(r) / 1e6
            print(f"  {os.path.basename(r):60s} {size_mb:.1f} MB")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
