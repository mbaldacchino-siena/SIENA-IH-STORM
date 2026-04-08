"""
run_evaluation.py  —  Full evaluation + visualization pipeline
===============================================================

Pipeline steps:
  0. Prepare IBTrACS reference  (auto-convert if needed)
  1. Compute IBTrACS metrics    (per phase + ALL → appears in summary CSV)
  2. Per synthetic candidate    (metrics + CSV exports + all plots)
  3. Cross-candidate summary    (lifetime overlay, summary CSV)

Edit the CONFIG section, then:

    python run_evaluation.py

Requires: evaluation.py and ibtracs_to_storm.py in the same directory.
          CLIMADA for return periods (Holland 2008 wind model).
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

from evaluation import (
    load_catalog,
    compute_all_metrics,
    assemble_all_catalog,
    compute_phase_fractions,
    compute_effective_years,
    return_periods_at_cities,
    return_periods_all_catalog,
    export_all_densities,
    lifetime_distribution,
    print_summary,
    run_all_plots,
    plot_lifetime_distribution,
    plot_return_period_curves,
    DEFAULT_CITIES,
)

# ═══════════════════════════════════════════════════════════════════════
# CONFIG — edit these to match your filesystem
# ═══════════════════════════════════════════════════════════════════════

BASINS = ["NA"]

DATA_ROOT = Path("/home/mbaldacchino/data")

# ── IBTrACS source ──
IBTRACS_SOURCE = str(DATA_ROOT / "IBTrACS.ALL.v04r01.nc")
IBTRACS_FORMAT = "netcdf"
IBTRACS_YEAR_RANGE = (1980, 2021)
IBTRACS_N_YEARS = IBTRACS_YEAR_RANGE[1] - IBTRACS_YEAR_RANGE[0] + 1  # 42

# Where converted IBTrACS STORM-format files live (or will be created)
REF_FOLDER = str(DATA_ROOT / "ibtracs_storm_format")
LAND_MASK_DIR = str(DATA_ROOT / "code" / "SIENA-IH-STORM-VWS-Track")

# ── Synthetic candidates ──
CANDIDATES = {
    "B0": {
        "ALL": {"folder": str(DATA_ROOT / "STORM_baseline"), "phase": None},
    },
    "B1": {
        "EN": {
            "folder": str(
                DATA_ROOT / "STORM_ENSO/IB1980-2021ELNINO/IB1980-2021ELNINO_nanfilled"
            ),
            "phase": None,
        },
        "NEU": {
            "folder": str(
                DATA_ROOT
                / "STORM_ENSO/IB1980-2021ENSO_NEUTRAL/IB1980-2021ENSO_NEUTRAL_nanfilled"
            ),
            "phase": None,
        },
        "LN": {
            "folder": str(
                DATA_ROOT / "STORM_ENSO/IB1980-2021LANINA/IB1980-2021LANINA_nanfilled"
            ),
            "phase": None,
        },
    },
    "S1": {
        "EN": {"folder": str(DATA_ROOT / "SIENA_S1"), "phase": "EN"},
        "NEU": {"folder": str(DATA_ROOT / "SIENA_S1"), "phase": "NEU"},
        "LN": {"folder": str(DATA_ROOT / "SIENA_S1"), "phase": "LN"},
    },
    "S2": {
        "EN": {"folder": str(DATA_ROOT / "SIENA_S2"), "phase": "EN"},
        "NEU": {"folder": str(DATA_ROOT / "SIENA_S2"), "phase": "NEU"},
        "LN": {"folder": str(DATA_ROOT / "SIENA_S2"), "phase": "LN"},
    },
    "S3": {
        "EN": {"folder": str(DATA_ROOT / "SIENA_S3"), "phase": "EN"},
        "NEU": {"folder": str(DATA_ROOT / "SIENA_S3"), "phase": "NEU"},
        "LN": {"folder": str(DATA_ROOT / "SIENA_S3"), "phase": "LN"},
    },
}

N_YEARS_PER_PHASE = 10_000
N_YEARS_ALL = 10_000

# ── Grid / RP settings ──
RES_DENSITY = 2.0
RES_TRACK = 1.0
TARGET_RP = np.array([2, 5, 10, 25, 50, 100, 250, 500, 1000])
COMPUTE_RP = True  # Set False to skip CLIMADA return periods (faster)
RP_MODEL = "H08"  # Holland 2008 parametric wind model

# ── Output ──
OUTDIR = Path("evaluation_output")
OLD_CANDIDATE = "B1"


# ═══════════════════════════════════════════════════════════════════════
# STEP 0: Prepare IBTrACS reference
# ═══════════════════════════════════════════════════════════════════════


def _prepare_ibtracs(basins):
    """Auto-convert IBTrACS if needed. Returns True if reference available."""
    ref_path = Path(REF_FOLDER)
    sample_file = ref_path / f"STORM_DATA_IBTRACS_{basins[0]}_1000_YEARS_0.txt"
    if sample_file.exists():
        print(f"  IBTrACS reference found at {REF_FOLDER}")
        return True

    if not IBTRACS_SOURCE or not os.path.exists(IBTRACS_SOURCE):
        print(f"  WARNING: IBTrACS source not found at {IBTRACS_SOURCE}")
        print(f"  Pipeline will run without IBTrACS reference.\n")
        return False

    print(f"  Converting IBTrACS → STORM format ...")
    from ibtracs_to_storm import convert_ibtracs

    convert_ibtracs(
        input_path=IBTRACS_SOURCE,
        fmt=IBTRACS_FORMAT,
        outdir=REF_FOLDER,
        basins=basins,
        year_range=IBTRACS_YEAR_RANGE,
        land_mask_dir=LAND_MASK_DIR
        if LAND_MASK_DIR and os.path.isdir(LAND_MASK_DIR)
        else None,
        split_phases=True,
        interpolate_3h=True,
    )
    return sample_file.exists()


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Compute IBTrACS metrics
# ═══════════════════════════════════════════════════════════════════════


def _evaluate_ibtracs(basin, outdir, summary_rows, lifetime_dfs):
    """
    Load IBTrACS per phase + ALL, compute metrics, export CSVs.
    Returns dict {"ALL": df, "EN": df, ...} and file paths.
    """
    ibt_outdir = outdir / "IBTrACS"
    ibt_outdir.mkdir(exist_ok=True)

    ibt_cats = {}
    ibt_files = {}

    # Compute effective years per phase for this basin
    # e.g. NA: ALL=42, EN≈9.3, NEU≈23, LN≈9.7
    eff_years = compute_effective_years(basin, total_years=IBTRACS_N_YEARS)
    print(f"  Effective years per phase:")
    for ph, yr in eff_years.items():
        print(f"    {ph}: {yr:.1f}")

    # ALL
    try:
        ibt_all, files_all = load_catalog(REF_FOLDER, basin, phase=None)
        ibt_cats["ALL"] = ibt_all
        ibt_files["ALL"] = files_all
    except FileNotFoundError:
        print(f"  IBTrACS ALL not found for {basin}, skipping.")
        return {}, {}

    # Per-phase
    for ph in ["EN", "NEU", "LN"]:
        try:
            ibt_ph, files_ph = load_catalog(REF_FOLDER, basin, phase=ph)
            ibt_cats[ph] = ibt_ph
            ibt_files[ph] = files_ph
        except FileNotFoundError:
            print(f"  IBTrACS {ph} not found for {basin}.")

    # Compute metrics for each
    cities_for_basin = [c for c in DEFAULT_CITIES if c.get("basin") == basin]
    for phase_label, cat in ibt_cats.items():
        files = ibt_files[phase_label]
        n_years_phase = eff_years[phase_label]
        n_storms = cat["global_storm_uid"].nunique()
        print(
            f"\n  IBTrACS / {phase_label}: {n_storms} storms, "
            f"{n_years_phase:.1f} effective years"
        )

        m = compute_all_metrics(
            cat,
            n_years_phase,
            file_paths=files,
            basin=basin,
            target_rp=TARGET_RP if COMPUTE_RP else None,
            cities=cities_for_basin if COMPUTE_RP else [],
        )
        print_summary(m, label=f"IBTrACS / {phase_label}")

        row = {
            "candidate": "IBTrACS",
            "phase": phase_label,
            "n_eff_years": n_years_phase,
            "genesis_mean": m["genesis_count"]["mean"],
            "genesis_std": m["genesis_count"]["std"],
            "total_storms": m["genesis_count"]["total_storms"],
            "lifetime_hours": m["lifetime"]["mean_hours"],
            "landfall_mean": m["landfall_counts"]["annual_mean"],
            "landfall_std": m["landfall_counts"]["annual_std"],
        }
        summary_rows.append(row)

        label_str = f"IBTrACS_{phase_label}"
        export_all_densities(
            cat,
            n_years_phase,
            str(ibt_outdir),
            label_str,
            basin,
            RES_TRACK,
            RES_DENSITY,
        )
        lt_df = lifetime_distribution(cat)
        lt_df.to_csv(ibt_outdir / f"lifetime_{label_str}.csv", index=False)
        lifetime_dfs[label_str] = lt_df

    return ibt_cats, ibt_files


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Evaluate synthetic candidates
# ═══════════════════════════════════════════════════════════════════════


def _evaluate_candidate(
    cand_name,
    phase_specs,
    basin,
    ref_cat,
    outdir,
    summary_rows,
    lifetime_dfs,
    candidate_phase_cats,
):
    """Load, evaluate, and plot one synthetic candidate."""
    import matplotlib.pyplot as plt

    cand_outdir = outdir / cand_name
    cand_outdir.mkdir(exist_ok=True)

    phase_catalogs = {}
    phase_files = {}

    for phase_label, spec in phase_specs.items():
        cat, files = load_catalog(spec["folder"], basin, phase=spec["phase"])
        n_yrs = N_YEARS_PER_PHASE

        print(
            f"\n  {phase_label}: {cat['global_storm_uid'].nunique()} storms, "
            f"{len(files)} file(s)"
        )

        # Scalar metrics
        cities_for_basin = [c for c in DEFAULT_CITIES if c.get("basin") == basin]
        m = compute_all_metrics(
            cat,
            n_yrs,
            file_paths=files,
            basin=basin,
            reference=ref_cat,
            ref_n_years=IBTRACS_N_YEARS,
            target_rp=TARGET_RP if COMPUTE_RP else None,
            cities=cities_for_basin if COMPUTE_RP else [],
        )
        print_summary(m, label=f"{cand_name} / {phase_label}")

        row = {
            "candidate": cand_name,
            "phase": phase_label,
            "genesis_mean": m["genesis_count"]["mean"],
            "genesis_std": m["genesis_count"]["std"],
            "total_storms": m["genesis_count"]["total_storms"],
            "lifetime_hours": m["lifetime"]["mean_hours"],
            "landfall_mean": m["landfall_counts"]["annual_mean"],
            "landfall_std": m["landfall_counts"]["annual_std"],
        }
        if "ks_pmin" in m["intensity"]:
            row["ks_pmin"] = m["intensity"]["ks_pmin"]
            row["ks_vmax"] = m["intensity"]["ks_vmax"]
        summary_rows.append(row)

        # CSV exports
        label_str = f"{cand_name}_{phase_label}"
        export_all_densities(
            cat, n_yrs, str(cand_outdir), label_str, basin, RES_TRACK, RES_DENSITY
        )
        lt_df = lifetime_distribution(cat)
        lt_df.to_csv(cand_outdir / f"lifetime_{label_str}.csv", index=False)
        lifetime_dfs[label_str] = lt_df

        # RP curve export (per phase)
        if COMPUTE_RP and "return_periods" in m:
            rp_df = m["return_periods"]
            rp_df.to_csv(cand_outdir / f"rp_{label_str}.csv", index=False)

        if phase_label != "ALL":
            phase_catalogs[phase_label] = cat
            phase_files[phase_label] = files

    candidate_phase_cats[cand_name] = phase_catalogs

    # ── Assemble ALL ──
    if set(phase_catalogs.keys()) == {"EN", "NEU", "LN"}:
        print(f"\n  Assembling ALL catalog for {cand_name} ...")
        phase_folders = {ph: phase_specs[ph]["folder"] for ph in ["EN", "NEU", "LN"]}
        all_cat, all_files = assemble_all_catalog(
            phase_folders, basin, total_years=N_YEARS_ALL
        )

        # Compute non-RP metrics on the assembled DataFrame
        # (RP is computed separately with proper year-weighted CLIMADA loading)
        m_all = compute_all_metrics(
            all_cat,
            N_YEARS_ALL,
            file_paths=all_files,
            basin=basin,
            reference=ref_cat,
            ref_n_years=IBTRACS_N_YEARS,
            target_rp=None,
            cities=[],
        )
        print_summary(m_all, label=f"{cand_name} / ALL")

        # RP for ALL via ENSO-weighted year sampling (Holland 2008)
        if COMPUTE_RP:
            print(f"  Computing ALL return periods (ENSO-weighted Holland 2008) ...")
            try:
                rp_all = return_periods_all_catalog(
                    phase_folders,
                    basin,
                    total_years=N_YEARS_ALL,
                    target_rp=TARGET_RP,
                    model=RP_MODEL,
                )
                if len(rp_all) > 0:
                    m_all["return_periods"] = rp_all
                    rp_all.to_csv(cand_outdir / f"rp_{cand_name}_ALL.csv", index=False)
                    print(f"  Saved: {cand_outdir}/rp_{cand_name}_ALL.csv")
            except Exception as e:
                print(f"  WARNING: ALL return period failed: {e}")

        row_all = {
            "candidate": cand_name,
            "phase": "ALL",
            "genesis_mean": m_all["genesis_count"]["mean"],
            "genesis_std": m_all["genesis_count"]["std"],
            "total_storms": m_all["genesis_count"]["total_storms"],
            "lifetime_hours": m_all["lifetime"]["mean_hours"],
            "landfall_mean": m_all["landfall_counts"]["annual_mean"],
            "landfall_std": m_all["landfall_counts"]["annual_std"],
        }
        if "ks_pmin" in m_all["intensity"]:
            row_all["ks_pmin"] = m_all["intensity"]["ks_pmin"]
            row_all["ks_vmax"] = m_all["intensity"]["ks_vmax"]
        summary_rows.append(row_all)

    # ── Spatial plots ──
    if phase_catalogs:
        n_yrs_dict = {ph: N_YEARS_PER_PHASE for ph in phase_catalogs}
        old_cats, old_nyrs = None, None
        if (
            OLD_CANDIDATE
            and OLD_CANDIDATE != cand_name
            and OLD_CANDIDATE in candidate_phase_cats
        ):
            old_cats = candidate_phase_cats[OLD_CANDIDATE]
            old_nyrs = N_YEARS_PER_PHASE

        run_all_plots(
            datasets=phase_catalogs,
            n_years_dict=n_yrs_dict,
            basin=basin,
            outdir=str(cand_outdir),
            resolution_density=RES_DENSITY,
            resolution_track=RES_TRACK,
            dataset_ref=ref_cat,
            n_years_ref=IBTRACS_N_YEARS,
            datasets_old=old_cats,
            n_years_old=old_nyrs,
        )

        # RP curves plot (using all files pooled across phases)
        if COMPUTE_RP:
            all_phase_files = []
            for ph_files in phase_files.values():
                all_phase_files.extend(ph_files)
            if all_phase_files:
                print("  Computing return periods (Holland 2008) ...")
                try:
                    rp_long = return_periods_at_cities(
                        all_phase_files,
                        N_YEARS_PER_PHASE * len(phase_catalogs),
                        target_rp=TARGET_RP,
                        model=RP_MODEL,
                    )
                    if len(rp_long) > 0:
                        rp_long.to_csv(
                            cand_outdir / f"rp_{cand_name}_pooled.csv", index=False
                        )
                        fig, _ = plot_return_period_curves(rp_long)
                        fig.savefig(
                            cand_outdir / f"validation_rp_{basin}.png",
                            dpi=200,
                            bbox_inches="tight",
                        )
                        print(f"  Saved: {cand_outdir}/validation_rp_{basin}.png")
                        plt.close(fig)
                except Exception as e:
                    print(f"  WARNING: Return period computation failed: {e}")

    plt.close("all")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════


def run():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTDIR.mkdir(exist_ok=True)

    for basin in BASINS:
        print(f"\n{'═' * 60}")
        print(f"  BASIN: {basin}")
        print(f"{'═' * 60}")

        basin_outdir = OUTDIR / basin
        basin_outdir.mkdir(exist_ok=True)

        fracs = compute_phase_fractions(basin)
        print(
            f"  Phase fractions: "
            + ", ".join(f"{ph}={f:.3f}" for ph, f in fracs.items())
        )

        all_summary_rows = []
        all_lifetime_dfs = {}
        candidate_phase_cats = {}

        # ── Step 0: Prepare IBTrACS ──
        print(f"\n{'─' * 60}")
        print(f"  Step 0: IBTrACS reference")
        print(f"{'─' * 60}")
        has_ref = _prepare_ibtracs([basin])

        # ── Step 1: Evaluate IBTrACS ──
        ref_cat = None
        if has_ref:
            print(f"\n{'─' * 60}")
            print(f"  Step 1: IBTrACS metrics")
            print(f"{'─' * 60}")
            ibt_cats, ibt_files = _evaluate_ibtracs(
                basin, basin_outdir, all_summary_rows, all_lifetime_dfs
            )
            ref_cat = ibt_cats.get("ALL")
            if ref_cat is not None:
                print(
                    f"\n  IBTrACS reference: "
                    f"{ref_cat['global_storm_uid'].nunique()} storms, "
                    f"{IBTRACS_N_YEARS} years"
                )

        # ── Step 2: Evaluate each synthetic candidate ──
        for cand_name, phase_specs in CANDIDATES.items():
            print(f"\n{'─' * 60}")
            print(f"  Step 2: Candidate {cand_name}")
            print(f"{'─' * 60}")

            _evaluate_candidate(
                cand_name,
                phase_specs,
                basin,
                ref_cat,
                basin_outdir,
                all_summary_rows,
                all_lifetime_dfs,
                candidate_phase_cats,
            )

        # ── Step 3: Cross-candidate summaries ──
        print(f"\n{'─' * 60}")
        print(f"  Step 3: Cross-candidate summaries")
        print(f"{'─' * 60}")

        if all_lifetime_dfs:
            print("  Lifetime overlay ...")
            fig, _ = plot_lifetime_distribution(all_lifetime_dfs)
            fig.savefig(
                basin_outdir / f"validation_lifetime_{basin}.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)

        summary_df = pd.DataFrame(all_summary_rows)
        csv_path = basin_outdir / "summary_metrics.csv"
        summary_df.to_csv(csv_path, index=False, float_format="%.4f")

        print(f"\n  Summary saved to {csv_path}")
        print(summary_df.to_string(index=False))

    print(f"\n{'═' * 60}")
    print(f"  Pipeline complete.  Output: {OUTDIR}/")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    run()
