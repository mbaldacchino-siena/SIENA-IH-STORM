# -*- coding: utf-8 -*-
"""
This module is part of the STORM model

For more information, please see
Bloemendaal, N., Haigh, I.D., de Moel, H. et al.
Generation of a global synthetic tropical cyclone hazard dataset using STORM.
Sci Data 7, 40 (2020). https://doi.org/10.1038/s41597-020-0381-2

Functions described here are part of the data pre-processing.

Copyright (C) 2020 Nadia Bloemendaal. All versions released under GNU General Public License v3.0
"""

import xarray as xr
import numpy as np
from datetime import date, timedelta
from scipy import stats
import os
import sys
from CODE.siena_utils import TS_THRESHOLD_MS


__location__ = os.path.realpath(os.getcwd())  # TEMP FIX?
dir_path = __location__

# Basin indices:
# 0 = EP = Eastern Pacific
# 1 = NA = North Atlantic
# 2 = NI = North Indian
# 3 = SI = South Indian
# 4 = SP = South Pacific
# 5 = WP = Western Pacific


def BOUNDARIES_BASINS(idx):
    if idx == 0:  # Eastern Pacific
        lat0, lat1, lon0, lon1 = 5, 60, 180, 285
    if idx == 1:  # North Atlantic
        lat0, lat1, lon0, lon1 = 5, 60, 255, 360
    if idx == 2:  # North Indian
        lat0, lat1, lon0, lon1 = 5, 60, 30, 100
    if idx == 3:  # South Indian
        lat0, lat1, lon0, lon1 = -60, -5, 10, 135
    if idx == 4:  # South Pacific
        lat0, lat1, lon0, lon1 = -60, -5, 135, 240
    if idx == 5:  # Western Pacific
        lat0, lat1, lon0, lon1 = 5, 60, 100, 180

    return lat0, lat1, lon0, lon1


def find_month(t):
    """
    Find the month corresponding to the timestep in the IBTrACS dataset
    input:
        t: timestep (in days since 17-11-1858)
    output:
        mo: month (1-12)
    """

    start = date(1858, 11, 17)
    days = t
    delta = timedelta(days)
    offset = delta + start
    mo = offset.month

    return mo


def find_basin(name):
    """
    Give a basin index to the basin name
    Input:
        name: name of basin, EP=Eastern Pacific, NA=North Atlantic, NI=North Indian, SI=South Indian, SP=South Pacific, WP= Western Pacific
    Output:
        index: value between 0 and 5 corresponding to the basin names
    """
    for basinname, index in zip(["EP", "NA", "NI", "SI", "SP", "WP"], range(0, 6)):
        if name == basinname:
            return index


def interpolate(dataset):
    """
    Interpolate the data to 3-hourly values
    Input:
        dataset: the respective dataset
    Output:
        dataset: the interpolated dataset

    """

    if (
        np.any(np.isnan(dataset)) == True
        and len([x for x, v in enumerate(dataset) if np.isnan(v) == False]) > 1
    ):
        ind = [
            x for x, v in enumerate(dataset) if np.isnan(v) == False
        ]  # indices with values

        ind1 = [
            x + ind[0]
            for x, v in enumerate(dataset[ind[0] : ind[-1]])
            if np.isnan(v) == True
        ]  # indices with no values

        val = [v for v in dataset if np.isnan(v) == False]

        if len(ind1) > 0:
            interlist = np.interp(
                ind1, ind, val
            )  # interpolate between the known values of dataset

            for ii, jj in zip(ind1, range(len(ind1))):
                dataset[ii] = interlist[jj]

        else:
            dataset = dataset

    return dataset


def check_timelist(tlist):
    """
    Check whether the consecutive time steps are 3 hours apart
    Input:
        tlist: list of time steps
    Output:
        print tlist if the consecutive time steps are not 3 hours apart

    """

    for ii in range(1, len(tlist)):
        if tlist[ii] - tlist[ii - 1] != 0.125:
            print(tlist)


def convert_wind_speed(wind, agency):
    """
    Convert IBTrACS wind speed to 10-min sustained wind speed.
    From the official IBTrACS documentation:
        Maximum sustained wind speed from the WMO agency for the current location.
        NO adjustment is made for differences in wind speed averaging periods.
        hurdat/atcf = North Atlantic - U.S. Miami (NOAA NHC) - 1-minute winds
        tokyo = RSMC Tokyo (JMA) - 10-minute
        newdelhi = RSMC New Delhi (IMD) - 3-minute
        reunion = RSMC La Reunion (MFLR) - 10 minute
        bom = Australian TCWCs (TCWC Perth, Darwin, Brisbane) - 10-minute
        nadi = RSMC Nadi (FMS) - 10 minute
        wellington = TCWC Wellington (NZMS) - 10-minute

    Input:
        wind: wind speed
        agency: name of agency
    Output:
        wind_conv: converted wind
    """

    if (
        agency == "hurdat_epa"
        or agency == "hurdat_atl"
        or agency == "newdelhi"
        or agency == "atcf"
    ):  # 1-minute wind
        wind_conv = 0.88 * wind
    else:
        wind_conv = wind

    return wind_conv


def extract_data(data, final_year):
    """
    Extract different variables from IBTrACS dataset.
    Input:
        *data*: dataset (IBTrACS)
    Output:
        *LATLIST_INTERP.npy*: interpolated values of latitude, where each entry in the dictionary stands for one TC
        *LONLIST_INTERP.npy*: interpolated values of longitude (0-360 deg)
        *WINDLIST_INTERP.npy*: interpolated values of wind (m/s)
        *PRESLIST_INTERP.npy*: interpolated values of pressure (hPa)
        *RMAXLIST_INTERP.npy*: interpolated values of Rmax (km)
        *MONTHLIST_INTERP.npy*: Month of TC genesis
        *BASINLIST_INTERP.npy*: Basin of TC genesis
        *YEARLIST_INTERP.npy*: Year of TC genesis
    """

    basin = data.basin.values
    years = data.season.values
    wind = data.wmo_wind.values
    wind = wind * 0.51444444  # convert from knots to m/s
    pres = data.wmo_pres.values
    time = data.time.values
    latitude = data.lat.values
    longitude = data.lon.values
    rmax = data.usa_rmw.values * 1.85200  # convert from nm to km
    wmo_agency = data.wmo_agency.values
    nature = data.nature.values

    """Create a npy list for each of the items"""
    latlist = {i: [] for i in range(len(years))}
    lonlist = {i: [] for i in range(len(years))}
    timelist = {i: [] for i in range(len(years))}
    windlist = {i: [] for i in range(len(years))}
    preslist = {i: [] for i in range(len(years))}
    monthlist = {i: [] for i in range(len(years))}
    basinlist = {i: [] for i in range(len(years))}
    rmaxlist = {i: [] for i in range(len(years))}
    yearlist = {i: [] for i in range(len(years))}

    for i in range(len(years)):
        if years[i] < (final_year + 1):
            idx = [x for x, v in enumerate(wmo_agency[i]) if len(v) > 1.0]
            if len(idx) > 0:  # there is data on wind speed and associated agency.
                # Note that if the wind list solely consists of 'nan',there would be no associated agency.
                # And that if there is a wind reading, there is also an associated agency.
                wind_conv = convert_wind_speed(
                    wind[i], wmo_agency[i][idx[0]].decode("utf-8")
                )

                if (
                    np.all(np.isnan(wind_conv)) == False
                    and np.nanmax(wind_conv) >= TS_THRESHOLD_MS
                ):
                    """We consider the timesteps between the first and the last moment of maximum wind speed > 18 m/s (equal to a tropical storm)"""
                    ind = [x for x, v in enumerate(wind_conv) if v >= TS_THRESHOLD_MS]
                    nature_list = [x.decode("utf-8") for x in nature[i]]

                    if "ET" in nature_list:
                        et_idx = nature_list.index("ET")

                        if et_idx > ind[0]:
                            end = max(ii for ii in ind if ii < et_idx)
                            ind = ind[: end + 1]
                        else:
                            ind = []

                    if (
                        len(ind) > 0.0 and basin[i][ind[0]].decode("utf-8") != "SA"
                    ):  # exclude the south atlantic
                        j0 = ind[0]  # first location at which storm is tropical storm
                        if len(ind) > 1:  # the storm spans multiple time steps
                            j1 = ind[
                                -1
                            ]  # last location at which storm is tropical storm
                        else:
                            j0 = ind[0]
                            j1 = j0

                        monthlist[i].append(find_month(time[i][ind[0]]))
                        basinlist[i].append(
                            find_basin(basin[i][ind[0]].decode("utf-8"))
                        )
                        yearlist[i].append(years[i])

                        idx = [x for x, v in enumerate(wmo_agency[i]) if len(v) > 1.0]
                        time_idx = [
                            j0 + x
                            for x, v in enumerate(time[i][j0 : j1 + 1])
                            if round(v, 3) % 0.125 == 0.0
                        ]
                        new_list = np.intersect1d(ind, time_idx)

                        if len(new_list) > 1.0:
                            n0 = time_idx.index(new_list[0])
                            n1 = time_idx.index(new_list[-1])

                            new_time = time_idx[n0 : n1 + 1]

                            j_idx = 0
                            while j_idx < len(new_time):
                                j = new_time[j_idx]
                                latlist[i].append(latitude[i][j])

                                if longitude[i][j] < 0.0:
                                    longitude[i][j] += 360.0

                                lonlist[i].append(longitude[i][j])
                                timelist[i].append(round(time[i][j], 3))
                                windlist[i].append(wind_conv[j])
                                preslist[i].append(pres[i][j])
                                rmaxlist[i].append(rmax[i][j])
                                j_idx = j_idx + 1

                            check_timelist(timelist[i])

    """This part is for interpolating the missing values"""
    lat_int = {i: [] for i in range(len(years))}
    lon_int = {i: [] for i in range(len(years))}
    wind_int = {i: [] for i in range(len(years))}
    pres_int = {i: [] for i in range(len(years))}
    rmax_int = {i: [] for i in range(len(years))}

    for i in range(len(latlist)):
        if len(latlist[i]) > 0:
            if np.isnan(windlist[i][-1]) == True:
                lat_int[i] = interpolate(latlist[i][:-1])
                lon_int[i] = interpolate(lonlist[i][:-1])
                wind_int[i] = interpolate(windlist[i][:-1])
                pres_int[i] = interpolate(preslist[i][:-1])
                rmax_int[i] = interpolate(rmaxlist[i][:-1])
            else:
                lat_int[i] = interpolate(latlist[i])
                lon_int[i] = interpolate(lonlist[i])
                wind_int[i] = interpolate(windlist[i])
                pres_int[i] = interpolate(preslist[i])
                rmax_int[i] = interpolate(rmaxlist[i])

    """
    Save the interpolated datasets as .npy files. These files will be used later on 
    and also come in handy when plotting IBTrACS data
    """
    np.save(os.path.join(dir_path, "LATLIST_INTERP.npy"), lat_int)
    np.save(os.path.join(dir_path, "LONLIST_INTERP.npy"), lon_int)
    np.save(os.path.join(dir_path, "TIMELIST_INTERP.npy"), timelist)
    np.save(os.path.join(dir_path, "WINDLIST_INTERP.npy"), wind_int)
    np.save(os.path.join(dir_path, "PRESLIST_INTERP.npy"), pres_int)
    np.save(os.path.join(dir_path, "RMAXLIST_INTERP.npy"), rmax_int)
    np.save(os.path.join(dir_path, "MONTHLIST_INTERP.npy"), monthlist)
    np.save(os.path.join(dir_path, "BASINLIST_INTERP.npy"), basinlist)
    np.save(os.path.join(dir_path, "YEARLIST_INTERP.npy"), yearlist)


def TC_variables(
    nyear,
    monthsall,
    oni_table=None,
    phase_month_counts=None,
    vws_fields=None,
    rh_fields=None,
    latitudes=None,
    longitudes=None,
):
    """
    Extract the important variables.
    SIENA extension: keep all storms pooled, add ENSO phase/year and co-located
    VWS/RH for track and pressure variables.

    C4 FIX: Poisson rates use exposure-based correction.  Each storm is still
    assigned its genesis-month's ENSO phase (per-month ONI), but the rate
    denominator uses active-season month counts per phase rather than year
    counts, preventing double-counting.

        rate_ph = storms_ph × L / M_ph

    where L = season length, M_ph = total active-season months in phase ph.

    Parameters
    ----------
    phase_month_counts : dict {basin_idx: {phase_code: int}}, from
                         count_phase_months().  If None, falls back to the old
                         year-counting approach (which double-counts).
    """

    def _lookup_env(fields, month, phase_name):
        """Pick the best available field: phase-specific > pooled."""
        if fields is None:
            return None
        key = (month, phase_name)
        if key in fields:
            return fields[key]
        key = (month, None)
        if key in fields:
            return fields[key]
        if month in fields:
            return fields[month]
        return None

    try:
        latlist = np.load(
            os.path.join(__location__, "LATLIST_INTERP.npy"), allow_pickle=True
        ).item()
        lonlist = np.load(
            os.path.join(__location__, "LONLIST_INTERP.npy"), allow_pickle=True
        ).item()
        windlist = np.load(
            os.path.join(__location__, "WINDLIST_INTERP.npy"), allow_pickle=True
        ).item()
        preslist = np.load(
            os.path.join(__location__, "PRESLIST_INTERP.npy"), allow_pickle=True
        ).item()
        rmaxlist = np.load(
            os.path.join(__location__, "RMAXLIST_INTERP.npy"), allow_pickle=True
        ).item()
        monthlist = np.load(
            os.path.join(__location__, "MONTHLIST_INTERP.npy"), allow_pickle=True
        ).item()
        basinlist = np.load(
            os.path.join(__location__, "BASINLIST_INTERP.npy"), allow_pickle=True
        ).item()
        yearlist = np.load(
            os.path.join(__location__, "YEARLIST_INTERP.npy"), allow_pickle=True
        ).item()
    except FileNotFoundError:
        print("Files do not exist in " + str(__location__) + ", please check directory")
        return

    from CODE.siena_utils import build_phase_lookup, nearest_env_value, verify_phase_rates

    phase_lookup = build_phase_lookup(oni_table)

    if latitudes is None or longitudes is None:
        try:
            ds = xr.open_dataset(os.path.join(__location__, "Monthly_mean_SST.nc"))
            latitudes = ds.latitude.values
            longitudes = ds.longitude.values
            ds.close()
        except Exception:
            latitudes = np.linspace(90, -90, 721)
            longitudes = np.linspace(0, 359.75, 1440)

    months = {i: [] for i in range(0, 6)}
    genesis_wind = {i: [] for i in range(0, 6)}
    genesis_pressure = {i: [] for i in range(0, 6)}
    genesis_dpres = {i: [] for i in range(0, 6)}
    genesis_pres_var = {i: [] for i in range(0, 6)}
    genesis_loc = {i: [] for i in range(0, 6)}

    # --- C4 FIX: Phase counts use year-level assignment ---
    # storms_per_phase[basin][phase_code] = total storm count
    storms_per_phase = {idx: {0: 0, 1: 0, 2: 0} for idx in range(6)}
    # pooled count (unchanged)
    poisson = {i: [0] for i in range(0, 6)}
    genesis_poisson = []

    track = {i: [] for i in range(0, 10)}
    pressure = {i: [] for i in range(0, 10)}
    for i in range(0, 10):
        track[i] = {j: [] for j in range(0, 6)}
        pressure[i] = {j: [] for j in range(0, 6)}

    genesis_loc_phase = {
        idx: {m: {"LN": [], "NEU": [], "EN": []} for m in monthsall[idx]}
        for idx in range(6)
    }
    genesis_months_phase = {idx: {"LN": [], "NEU": [], "EN": []} for idx in range(6)}

    for idx in range(0, 6):
        genesis_wind[idx] = {i: [] for i in monthsall[idx]}
        genesis_pressure[idx] = {i: [] for i in monthsall[idx]}
        genesis_dpres[idx] = {i: [] for i in monthsall[idx]}
        genesis_pres_var[idx] = {i: [] for i in monthsall[idx]}
        genesis_loc[idx] = {i: [] for i in monthsall[idx]}

    for i in range(len(latlist)):
        if len(latlist[i]) > 0:
            idx = basinlist[i][0]
            month = monthlist[i][0]
            year = int(yearlist[i][0]) if len(yearlist[i]) > 0 else -1

            # Phase from genesis month's ONI value (per-month assignment)
            phase = phase_lookup.get((year, month), 1)
            phase_name = {0: "LN", 1: "NEU", 2: "EN"}.get(phase, "NEU")

            if month in monthsall[idx]:
                months[idx].append(month)
                genesis_wind[idx][month].append(windlist[i][0])
                genesis_dpres[idx][month].append(preslist[i][1] - preslist[i][0])
                genesis_pressure[idx][month].append(preslist[i][0])
                genesis_loc[idx][month].append([latlist[i][0], lonlist[i][0]])
                genesis_loc_phase[idx][month][phase_name].append(
                    [latlist[i][0], lonlist[i][0]]
                )
                genesis_months_phase[idx][phase_name].append(month)
                poisson[idx][0] += 1
                storms_per_phase[idx][phase] += 1

                for j in range(1, len(latlist[i]) - 1):
                    lat_now = latlist[i][j]
                    lon_now = lonlist[i][j]
                    vws_val = np.nan
                    rh_val = np.nan
                    vws_fld = _lookup_env(vws_fields, month, phase_name)
                    if vws_fld is not None:
                        vws_val = nearest_env_value(
                            vws_fld, latitudes, longitudes, lat_now, lon_now
                        )
                    rh_fld = _lookup_env(rh_fields, month, phase_name)
                    if rh_fld is not None:
                        rh_val = nearest_env_value(
                            rh_fld, latitudes, longitudes, lat_now, lon_now
                        )

                    track[0][idx].append(latlist[i][j] - latlist[i][j - 1])
                    track[1][idx].append(latlist[i][j + 1] - latlist[i][j])
                    track[2][idx].append(lonlist[i][j] - lonlist[i][j - 1])
                    track[3][idx].append(lonlist[i][j + 1] - lonlist[i][j])
                    track[4][idx].append(lat_now)
                    track[5][idx].append(lon_now)
                    track[6][idx].append(phase)
                    track[7][idx].append(year)
                    track[8][idx].append(vws_val)
                    track[9][idx].append(rh_val)

                for j in range(1, len(preslist[i]) - 1):
                    if (
                        np.isnan(preslist[i][j - 1]) == False
                        and np.isnan(preslist[i][j]) == False
                        and np.isnan(preslist[i][j + 1]) == False
                    ):
                        lat_now = latlist[i][j]
                        lon_now = lonlist[i][j]
                        vws_val = np.nan
                        rh_val = np.nan
                        vws_fld = _lookup_env(vws_fields, month, phase_name)
                        if vws_fld is not None:
                            vws_val = nearest_env_value(
                                vws_fld,
                                latitudes,
                                longitudes,
                                lat_now,
                                lon_now,
                            )
                        rh_fld = _lookup_env(rh_fields, month, phase_name)
                        if rh_fld is not None:
                            rh_val = nearest_env_value(
                                rh_fld,
                                latitudes,
                                longitudes,
                                lat_now,
                                lon_now,
                            )
                        pressure[0][idx].append(preslist[i][j] - preslist[i][j - 1])
                        pressure[1][idx].append(preslist[i][j + 1] - preslist[i][j])
                        pressure[2][idx].append(preslist[i][j])
                        pressure[3][idx].append(lat_now)
                        pressure[4][idx].append(lon_now)
                        pressure[5][idx].append(month)
                        pressure[6][idx].append(phase)
                        pressure[7][idx].append(year)
                        pressure[8][idx].append(vws_val)
                        pressure[9][idx].append(rh_val)

    # ---- C4 FIX: Exposure-based Poisson rates ----
    # rate_ph = storms_ph × L / M_ph
    # where L = season length, M_ph = total active-season months classified as phase ph
    # This is the MLE for Poisson rate with fractional exposure, and avoids
    # double-counting years that straddle multiple ENSO phases.
    MIN_EXPOSURE_MONTHS = None  # will be set per basin = L (one full season)

    poisson_phase_rate = {idx: {} for idx in range(6)}
    for idx in range(6):
        L = len(monthsall[idx])  # season length for this basin
        MIN_EXPOSURE_MONTHS = L  # require at least 1 full-season-equivalent of exposure
        for ph in [0, 1, 2]:
            M_ph = phase_month_counts[idx][ph] if phase_month_counts is not None else 0
            raw_storms = storms_per_phase[idx][ph]
            if M_ph >= MIN_EXPOSURE_MONTHS:
                rate = round(raw_storms * L / M_ph, 1)
            else:
                # Insufficient exposure: fall back to pooled rate
                rate = round(poisson[idx][0] / nyear[idx], 1)
                ph_name = {0: "LN", 1: "NEU", 2: "EN"}[ph]
                print(
                    f"  WARNING: Basin {idx} phase {ph_name}: only {M_ph} "
                    f"active-season months (min={MIN_EXPOSURE_MONTHS}). "
                    f"Falling back to pooled rate."
                )
            poisson_phase_rate[idx][ph] = rate

        print(
            f"  Basin {idx} phase rates: LN={poisson_phase_rate[idx][0]}, "
            f"NEU={poisson_phase_rate[idx][1]}, EN={poisson_phase_rate[idx][2]} "
            f"(exposure months: LN={phase_month_counts[idx][0]}, "
            f"NEU={phase_month_counts[idx][1]}, EN={phase_month_counts[idx][2]}, "
            f"season_length={L})"
        )

    for idx in range(0, 6):
        pooled_rate = round(poisson[idx][0] / nyear[idx], 1)
        genesis_poisson.append(pooled_rate)

        # C4 FIX: Sanity check — weighted phase rates ≈ pooled rate
        L = len(monthsall[idx])
        if phase_month_counts is not None:
            verify_phase_rates(
                poisson_phase_rate[idx], pooled_rate, phase_month_counts[idx], L, idx
            )

        dp0_neg, dp0_pos = [], []
        for j in range(len(pressure[0][idx])):
            if pressure[0][idx][j] < 0.0:
                dp0_neg.append(pressure[0][idx][j])
            elif pressure[0][idx][j] > 0:
                dp0_pos.append(pressure[0][idx][j])

        pneg = np.percentile(dp0_neg, 1) if len(dp0_neg) > 0 else -50
        ppos = np.percentile(dp0_pos, 99) if len(dp0_pos) > 0 else 50

        for month in monthsall[idx]:
            dplist = [
                v
                for v in genesis_dpres[idx][month]
                if np.isnan(v) == False and v > -1000.0
            ]
            plist = [
                v
                for v in genesis_pressure[idx][month]
                if np.isnan(v) == False and v > 0.0
            ]
            if len(dplist) < 2 or len(plist) < 2:
                mudp0, stddp0, mupres, stdpres = 0.0, 1.0, 1000.0, 10.0
            else:
                mudp0, stddp0 = stats.norm.fit(dplist)
                mupres, stdpres = stats.norm.fit(plist)
            genesis_pres_var[idx][month] = [mupres, stdpres, mudp0, stddp0, pneg, ppos]

    radius = {i: [] for i in range(0, 3)}
    for i in range(len(rmaxlist)):
        if len(rmaxlist[i]) > 0.0:
            for j in range(len(rmaxlist[i])):
                if (
                    np.isnan(rmaxlist[i][j]) == False
                    and np.isnan(preslist[i][j]) == False
                ):
                    if preslist[i][j] <= 920.0:
                        radius[0].append(rmaxlist[i][j])
                    elif preslist[i][j] > 920.0 and preslist[i][j] <= 960.0:
                        radius[1].append(rmaxlist[i][j])
                    elif preslist[i][j] > 960.0:
                        radius[2].append(rmaxlist[i][j])

    print("genesis per basin: ", genesis_poisson)
    np.save(os.path.join(__location__, "RMAX_PRESSURE.npy"), radius)
    np.savetxt(
        os.path.join(__location__, "POISSON_GENESIS_PARAMETERS.txt"), genesis_poisson
    )
    np.save(
        os.path.join(__location__, "POISSON_GENESIS_PARAMETERS_PHASE.npy"),
        poisson_phase_rate,
    )
    np.save(os.path.join(__location__, "TC_TRACK_VARIABLES.npy"), track)
    np.save(os.path.join(__location__, "TC_PRESSURE_VARIABLES.npy"), pressure)
    np.save(os.path.join(__location__, "DP0_PRES_GENESIS.npy"), genesis_pres_var)
    np.save(os.path.join(__location__, "DP_GEN.npy"), genesis_dpres)
    np.save(os.path.join(__location__, "PRES_GEN.npy"), genesis_pressure)
    np.save(os.path.join(__location__, "GEN_LOC.npy"), genesis_loc)
    np.save(os.path.join(__location__, "GEN_LOC_PHASE.npy"), genesis_loc_phase)
    np.save(os.path.join(__location__, "GENESIS_WIND.npy"), genesis_wind)
    np.save(os.path.join(__location__, "GENESIS_MONTHS.npy"), months)
    np.save(
        os.path.join(__location__, "GENESIS_MONTHS_PHASE.npy"), genesis_months_phase
    )
