"""
This module is part of the STORM model

For more information, please see
Bloemendaal, N., Haigh, I.D., de Moel, H. et al.
Generation of a global synthetic tropical cyclone hazard dataset using STORM.
Sci Data 7, 40 (2020). https://doi.org/10.1038/s41597-020-0381-2

Functions described here are part of the data pre-processing and calculate the environmental
conditions + wind-pressure relationship.

Copyright (C) 2020 Nadia Bloemendaal. All versions released under the GNU General Public License v3.0
"""

import xarray as xr
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import curve_fit, least_squares
import math
import CODE.preprocessing as preprocessing
from CODE.siena_utils import solve_ridge
import os
import sys

pd.options.mode.chained_assignment = None  # default='warn'
dir_path = os.path.dirname(os.path.realpath(sys.argv[0]))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def monthly_mean_pressure_STORM(data):
    """
    Create the monthly mean MSLP fields. This function outputs a txt-file of a global field of monthly mean MSLP for every month.

    Parameters
    ----------
    data : dataset with monthly mean MSLP values for 38 years of data (ERA-5)

    """
    mslp = data.msl.values
    lon = data.longitude.values
    lat = data.latitude.values

    for month in range(0, 12):
        mean_matrix = np.zeros((len(lat), len(lon)))

        for t in range(0, int(nyear)):
            # loop over 38 years
            mean_matrix = mean_matrix + mslp[month + t * 12, :, :] / 100.0

        mean_matrix = mean_matrix / nyear
        np.savetxt(
            os.path.join(__location__, "Monthly_mean_MSLP_" + str(month + 1) + ".txt"),
            mean_matrix,
        )


def monthly_mean_pressure(data):
    """
    Create the monthly mean MSLP fields. This do function outputs a txt-file of a global field of monthly mean MSLP for every month.
    This function consider number of months (e.g. january) instead of years. So it can be use as climatalogies like ENSO
    Parameters
    ----------
    data : dataset with monthly mean MSLP values for a number of years of data (ERA-5)

    @itxasoOderiz 2024

    """
    # Group by month and take the sum
    data_sum = data.groupby("valid_time.month").sum(dim="valid_time")

    for nmonth in range(0, 12):
        nmonth1 = nmonth + 1
        print("SLP fields for month:", nmonth1)

        # sum all months equal to nmonth1
        month_sum = data_sum.sel(month=nmonth1)

        data_month = data.sel(valid_time=data["valid_time.month"] == nmonth1)

        # mean considering all number of months, not years
        month_mean = month_sum / data_month.sizes["valid_time"]

        # save to a txt
        mean_matrix = month_mean.msl.values / 100  # convert to milibars
        np.savetxt(
            os.path.join(__location__, "Monthly_mean_MSLP_" + str(nmonth1) + ".txt"),
            mean_matrix,
        )


def monthly_mean_sst_STORM(data, nyear):
    """
    Create the monthly mean SST fields. This function outputs a txt-file of a global field of monthly mean SSTs for every month.

    Parameters
    ----------
    data : dataset with monthly mean SST values for n years of data (ERA-5)

    """
    sst = data.sst.values
    lon = data.longitude.values
    lat = data.latitude.values

    for month in range(0, 12):
        mean_matrix = np.zeros((len(lat), len(lon)))

        for t in range(0, int(nyear)):
            mean_matrix = mean_matrix + sst[month + t * 12, :, :]

        mean_matrix = mean_matrix / nyear
        np.savetxt(
            os.path.join(__location__, "Monthly_mean_SST_" + str(month + 1) + ".txt"),
            mean_matrix,
        )


def monthly_mean_sst(data):
    """
    Create the monthly mean SST fields. This do function outputs a txt-file of a global field of monthly mean MSLP for every month.
    This function consider number of months (e.g. january) instead of years. So it can be use as climatalogies like ENSO
    Parameters
    ----------
    data : dataset with monthly mean MSLP values for a number of years of data (ERA-5)

    @itxasoOderiz 2024

    """
    # Group by month and take the sum
    data_sum = data.groupby("valid_time.month").sum(dim="valid_time")

    for nmonth in range(0, 12):
        nmonth1 = nmonth + 1
        print("SST fields for month:", nmonth1)

        # sum all months equal to nmonth1
        month_sum = data_sum.sel(month=nmonth1)

        data_month = data.sel(valid_time=data["valid_time.month"] == nmonth1)

        # mean considering all number of months, not years
        month_mean = month_sum / data_month.sizes["valid_time"]

        # save to a txt
        mean_matrix = month_mean.sst.values
        np.savetxt(
            os.path.join(__location__, "Monthly_mean_SST_" + str(nmonth1) + ".txt"),
            mean_matrix,
        )


def monthly_mean_vws(data):
    """
    Create the monthly mean VWS fields. This do function outputs a txt-file of a global field of monthly mean VWS for every month.
    This function consider number of months (e.g. january) instead of years. So it can be use as climatalogies like ENSO
    Parameters
    ----------
    data : dataset with monthly mean Wind values for a number of years of data (ERA-5) at 850hPa and 200hPa

    @mbaldacchino 2026

    """
    # Group by month and take the sum
    data_sum = data.groupby("valid_time.month").sum(dim="valid_time")

    for nmonth in range(0, 12):
        nmonth1 = nmonth + 1
        print("VWS fields for month:", nmonth1)

        # sum all months equal to nmonth1
        month_sum = data_sum.sel(month=nmonth1)

        data_month = data.sel(valid_time=data["valid_time.month"] == nmonth1)

        # mean considering all number of months, not years
        month_mean = month_sum / data_month.sizes["valid_time"]

        # save to a txt
        mean_matrix = month_mean.vws.values
        np.savetxt(
            os.path.join(__location__, "Monthly_mean_VWS_" + str(nmonth1) + ".txt"),
            mean_matrix,
        )


def monthly_mean_rh(data):
    """
    Create the monthly mean RH fields. This do function outputs a txt-file of a global field of monthly mean RH for every month.
    This function consider number of months (e.g. january) instead of years. So it can be use as climatalogies like ENSO
    Parameters
    ----------
    data : dataset with monthly mean RH600 values for a number of years of data (ERA-5)

    @mbaldacchino 2026

    """
    # Group by month and take the sum
    data_sum = (
        data.isel(pressure_level=0).groupby("valid_time.month").sum(dim="valid_time")
    )

    for nmonth in range(0, 12):
        nmonth1 = nmonth + 1
        print("SST fields for month:", nmonth1)

        # sum all months equal to nmonth1
        month_sum = data_sum.sel(month=nmonth1)

        data_month = data.sel(valid_time=data["valid_time.month"] == nmonth1)

        # mean considering all number of months, not years
        month_mean = month_sum / data_month.sizes["valid_time"]

        # save to a txt
        mean_matrix = month_mean.r.values
        print(mean_matrix)
        np.savetxt(
            os.path.join(__location__, "Monthly_mean_RH600_" + str(nmonth1) + ".txt"),
            mean_matrix,
        )


def check_season(idx, month):
    """
    Check if TC occurred in TC season.

    Parameters
    ----------
    idx : Basin index (EP=0,NA=1,NI=2,SI=3,SP=4,WP=5)
    month : month in which TC occurred

    Returns
    -------
    check : 0 if TC did not occur in TC season, 1 if TC did occur in TC season.
    """
    check = 0
    if idx == 0 or idx == 1:
        if month > 5 and month < 12:
            check = 1
    elif idx == 2:
        if month > 3 and month < 7:
            check = 1
        elif month > 8 and month < 12:
            check = 1
    elif idx == 3 or idx == 4:
        if month < 5 or month > 10:
            check = 1
    elif idx == 5:
        if month > 4 and month < 12:
            check = 1
    return check


def Vmax_function(DP, A, B):
    """
    This is the wind-pressure relationship. Here, we calculate the values of the coefficients
    A en B for the wind and pressure found in the dataset.
    Parameters
    ----------
    DP : Difference between environmental pressure and central pressure (hPa)
    A,B : Coefficients for wind-pressure relationship.
    """
    return A * (DP) ** B


def wind_pressure_relationship(idx_basin, months):
    """
    This function calculates the coefficients for the wind-pressure relationship.
    The wind-pressure relationship is based on the empirical wind-pressure relationship (for overview, see Harper 2002:
        Tropical Cyclone Parameter Estimation in the Australian Region: Wind-Pressure Relationships and
        Related Issues for Engineering Planning and Design - A Discussion Paper)

    Adapted by e.g. Atkinson and Holliday (1977), Love and Murphy (1985) and Crane (1985)

    This script saves the coefficients list for the wind-pressure relationship, per month as an npy-file.
    """
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
    monthlist = np.load(
        os.path.join(__location__, "MONTHLIST_INTERP.npy"), allow_pickle=True
    ).item()
    basinlist = np.load(
        os.path.join(__location__, "BASINLIST_INTERP.npy"), allow_pickle=True
    ).item()

    data = xr.open_dataset(os.path.join(__location__, "Monthly_mean_SST.nc"))

    lon = data.longitude.values
    lat = data.latitude.values
    data.close()

    pres_basin = {i: [] for i in range(0, 6)}
    wind_basin = {i: [] for i in range(0, 6)}
    month_basin = {i: [] for i in range(0, 6)}

    for i in range(len(latlist)):
        if len(latlist[i]) > 0:
            idx = basinlist[i][0]
            month = monthlist[i][0]
            check = check_season(idx, month)
            # print(idx,month,check)
            if check == 1:
                MSLP = np.loadtxt(
                    os.path.join(
                        __location__, "Monthly_mean_MSLP_" + str(month) + ".txt"
                    )
                )
                for j in range(0, len(latlist[i])):
                    # Wind needs to be greater than 15 kt.
                    latn = np.abs(lat - latlist[i][j]).argmin()
                    lonn = np.abs(lon - lonlist[i][j]).argmin()
                    if (
                        preslist[i][j] > 0
                        and MSLP[latn][lonn] - preslist[i][j] > 0
                        and windlist[i][j] > 15.0 * 0.5144444
                    ):
                        pres_basin[idx].append(MSLP[latn][lonn] - preslist[i][j])
                        wind_basin[idx].append(windlist[i][j])
                        month_basin[idx].append(month)

    coeff_list = {i: [] for i in range(0, 6)}

    # months=[[6,7,8,9,10,11],[6,7,8,9,10,11],[9,10,11],[1,2,3,4,11,12],[1,2,3,4,11,12],[5,6,7,8,9,10,11]]

    # months=[[6,7,8,9,10,11],[6,7,8,9,10,11],[4,5,6,9,10,11],[1,2,3,4,11,12],[1,2,3,4,11,12],[5,6,7,8,9,10,11]]

    for ii in range(len(idx_basin)):
        idx = idx_basin[ii]
        print("analasing: ", idx, "basin in wind relationship")
        coeff_list[idx] = {i: [] for i in months[idx]}

        # ---- FIX: Build full-basin DataFrame once ----
        df = pd.DataFrame(
            {
                "Wind": wind_basin[idx],
                "Pressure": pres_basin[idx],
                "Month": month_basin[idx],
            }
        )

        # ---- FIX: Fit basin-wide WPR first as fallback ----
        basin_coef = []
        if len(df) > 10:
            step = 2.0
            to_bin = lambda x: np.floor(x / step) * step
            df_all = df.copy()
            df_all["windbin"] = df_all["Wind"].map(to_bin)
            minpres_all = df_all.groupby("windbin")["Pressure"].mean()
            maxwind_all = np.array(minpres_all.index)
            minpres_all = minpres_all.values
            # Remove NaN/inf
            valid = (
                np.isfinite(minpres_all) & np.isfinite(maxwind_all) & (minpres_all > 0)
            )
            if valid.sum() >= 3:
                try:
                    opt, _ = curve_fit(
                        Vmax_function,
                        minpres_all[valid],
                        maxwind_all[valid],
                        p0=[0.7, 0.6],
                        bounds=([0.001, 0.01], [50.0, 3.0]),
                        maxfev=10000,
                    )
                    basin_coef = [opt[0], opt[1]]
                except (RuntimeError, TypeError):
                    pass

        for i in range(len(months[idx])):
            m = months[idx][i]
            df1 = df[df["Month"] == m].copy()  # ---- FIX: explicit .copy() ----

            if len(df1) < 5:
                if basin_coef:
                    coeff_list[idx][m] = basin_coef
                    print(
                        f"  Month {m}: too few points ({len(df1)}), using basin-wide fit"
                    )
                else:
                    print(f"  Month {m}: too few points and no basin fallback")
                continue

            step = 2.0  # Group in 2 m/s bins
            to_bin = lambda x: np.floor(x / step) * step
            df1["windbin"] = df1["Wind"].map(to_bin)
            minpres = df1.groupby("windbin")["Pressure"].mean()
            maxwind = np.array(minpres.index)
            minpres = minpres.values

            # ---- FIX: clean data before fitting ----
            valid = np.isfinite(minpres) & np.isfinite(maxwind) & (minpres > 0)
            if valid.sum() < 3:
                if basin_coef:
                    coeff_list[idx][m] = basin_coef
                    print(f"  Month {m}: insufficient valid bins, using basin-wide fit")
                continue

            try:
                # ---- FIX: add p0 and bounds for robust convergence ----
                opt, _ = curve_fit(
                    Vmax_function,
                    minpres[valid],
                    maxwind[valid],
                    p0=[0.7, 0.6],
                    bounds=([0.001, 0.01], [50.0, 3.0]),
                    maxfev=10000,
                )
                [a, b] = opt
                coeff_list[idx][m] = [a, b]

            except RuntimeError:
                print("Optimal parameters not found")
                if basin_coef:
                    coeff_list[idx][m] = basin_coef
                    print(f"  Month {m}: using basin-wide fallback")
            except TypeError:
                print("Too few items")
                if basin_coef:
                    coeff_list[idx][m] = basin_coef

        # ---- FIX: final safety — fill any remaining empty months ----
        for m in months[idx]:
            if not coeff_list[idx][m] and basin_coef:
                coeff_list[idx][m] = basin_coef
                print(f"  Month {m}: filled with basin-wide fallback (final pass)")

    np.save(os.path.join(__location__, "COEFFICIENTS_WPR_PER_MONTH.npy"), coeff_list)


def MPI_function(T, A, B, C):
    """
    Fit the MPI function to the data. This function returns the optimal coefficients.
    Parameters
    ----------
    T : Sea-surface temperature in Celcius.
    A,B,C : coefficients

    """
    # ---- FIX: clamp exponent to prevent overflow ----
    exponent = np.clip(C * (np.asarray(T, dtype=float) - 30.0), -500, 500)
    return A + B * np.exp(exponent)


def Calculate_P(V, Penv, a, b):
    """
    Convert Vmax to Pressure following the empirical wind-pressure relationship (Harper 2002, Atkinson and Holliday 1977)

    Input:
        Vmax: 10-min mean maximum wind speed in m/s
        Penv: environmental pressure (hPa)
        a,b: coefficients. See Atkinson_Holliday_wind_pressure_relationship.py

    Returns:
        Pc: central pressure in the eye

    """

    Pc = Penv - (V / a) ** (1.0 / b)
    return Pc


def calculate_MPI_fields(idx_basin, months, months_for_coef, mpi_bounds):
    """
    Calculate the MPI fields from the pressure drop and environmental conditions.
    """
    # =============================================================================
    # Calculate the MPI and SST - NOTE: THIS PART TAKES VERY LOOONG
    # =============================================================================
    data = xr.open_dataset(os.path.join(__location__, "Monthly_mean_SST.nc"))

    lon = data.longitude.values
    lat = data.latitude.values
    data.close()
    latlist = np.load(
        os.path.join(__location__, "LATLIST_INTERP.npy"), allow_pickle=True
    ).item()
    lonlist = np.load(
        os.path.join(__location__, "LONLIST_INTERP.npy"), allow_pickle=True
    ).item()
    monthlist = np.load(
        os.path.join(__location__, "MONTHLIST_INTERP.npy"), allow_pickle=True
    ).item()
    basinlist = np.load(
        os.path.join(__location__, "BASINLIST_INTERP.npy"), allow_pickle=True
    ).item()
    preslist = np.load(
        os.path.join(__location__, "PRESLIST_INTERP.npy"), allow_pickle=True
    ).item()

    sst_list = {i: [] for i in range(0, 6)}
    month_list = {i: [] for i in range(0, 6)}
    intensity_list = {i: [] for i in range(0, 6)}
    pressure_drop_list = {i: [] for i in range(0, 6)}

    MSLP_field_all = {i: [] for i in range(1, 13)}
    SST_field_all = {i: [] for i in range(1, 13)}

    for month in range(1, 13):
        MSLP_field_all[month] = np.loadtxt(
            os.path.join(__location__, "Monthly_mean_MSLP_" + str(month) + ".txt")
        )
        SST_field_all[month] = np.loadtxt(
            os.path.join(__location__, "Monthly_mean_SST_" + str(month) + ".txt")
        )

    for i in range(len(latlist)):
        if len(preslist[i]) > 0:
            idx = basinlist[i][0]
            month = monthlist[i][0]

            SST_field = SST_field_all[month]
            MSLP_field = MSLP_field_all[month]

            for j in range(len(preslist[i])):
                lat_index = np.abs(lat - latlist[i][j]).argmin()
                lon_index = np.abs(lon - lonlist[i][j]).argmin()

                if (
                    SST_field[lat_index, lon_index] > 288.15 and preslist[i][j] > 0
                ):  # only use SST>15C for the fit.
                    sst_list[idx].append(SST_field[lat_index, lon_index] - 273.15)
                    intensity_list[idx].append(preslist[i][j])
                    pressure_drop_list[idx].append(
                        MSLP_field[lat_index, lon_index] - preslist[i][j]
                    )
                    month_list[idx].append(month)

    # =============================================================================
    # Calculate the MPI coefficients (see DeMaria & Kaplan 1994)
    # =============================================================================
    basins = ["EP", "NA", "NI", "SI", "SP", "WP"]
    coeflist = {i: [] for i in range(0, 6)}
    # Only consider those in the hurricane season
    # months          =[[6,7,8,9,10,11],[6,7,8,9,10,11],[9,10,11],[1,2,3,4,11,12],[1,2,3,4,11,12],[5,6,7,8,9,10,11]]
    # months_for_coef =[[6,7,8,9,10,10],[6,7,8,9,10,11],[10,10,11],[1,2,3,4,11,12],[1,2,3,4,11,12],[5,6,7,8,9,10,11]]

    # months=[[6,7,8,9,10,11],[6,7,8,9,10,11],[4,5,6,9,10,11],[1,2,3,4,11,12],[1,2,3,4,11,12],[5,6,7,8,9,10,11]]
    # months_for_coef=[[6,7,8,9,10,10],[6,7,8,9,10,11],[6,6,6,10,10,11],[1,2,3,4,11,12],[1,2,3,4,11,12],[5,6,7,8,9,10,11]]
    for ii in range(len(idx_basin)):
        idx = idx_basin[ii]

        coeflist[idx] = {i: [] for i in months[idx]}

        df = pd.DataFrame(
            {
                "Drop": pressure_drop_list[idx],
                "SST": sst_list[idx],
                "Month": month_list[idx],
            }
        )

        df = df[df["Drop"] > -99999.0]

        for i in range(len(months[idx])):
            m = months_for_coef[idx][i]
            mc = months[idx][i]
            print(idx, mc)
            if idx == 2 and m < 7.0:
                df1 = df[(df["Month"] == 4) | (df["Month"] == 5) | (df["Month"] == 6)]

            elif idx == 2 and m > 7.0:
                df1 = df[(df["Month"] == 9) | (df["Month"] == 10) | (df["Month"] == 11)]

            elif m > 10 and idx == 3 or idx == 4:
                df1 = df[(df["Month"] == 11) | (df["Month"] == 12)]

            elif m < 5 and idx == 3 or idx == 4:
                df1 = df[
                    (df["Month"] == 1)
                    | (df["Month"] == 2)
                    | (df["Month"] == 3)
                    | (df["Month"] == 4)
                ]

            else:
                df1 = df[(df["Month"] == m)]

            df1 = df1[(df1["SST"] < 30.0)]

            step = 1.0
            to_bin = lambda x: np.floor(x / step) * step
            df1["sstbin"] = df1.SST.map(to_bin)

            droplist = df1.groupby(["sstbin"]).agg({"Drop": "max"})["Drop"]
            sstlist = df1.groupby(["sstbin"]).agg({"SST": "mean"})["SST"]

            try:
                opt, l = curve_fit(MPI_function, sstlist, droplist, maxfev=5000)
                [a, b, c] = opt
                coeflist[idx][mc] = [a, b, c]
            except RuntimeError:
                print("Optimal parameters not found for " + str(basins[idx]))
            except TypeError:
                print("Too few items")

    np.save(
        os.path.join(__location__, "COEFFICIENTS_MPI_PRESSURE_DROP_MONTH.npy"), coeflist
    )
    # =============================================================================
    #  Calculate the new MPI in hPa
    # =============================================================================

    # original
    # months=[[6,7,8,9,10,11],[6,7,8,9,10,11],[4,5,6,9,10,11],[1,2,3,4,11,12],[1,2,3,4,11,12],[5,6,7,8,9,10,11]]

    # these are the lowest mpi values per basin and serve as the lower bound, derived from Bister & Emanuel 2002
    # mpi_bounds=[[860,880,900,900,880,860],[920,900,900,900,880,880],[840,860,880,900,880,860],[840,880,860,860,840,860],[840,840,860,860,840,840],[860,860,860,870,870,860,860]]

    # for idx in range(0,6):

    for ii in range(len(idx_basin)):
        idx = idx_basin[ii]
        for m, midx in zip(months[idx], range(len(months[idx]))):
            print(idx, m, midx)

            [A, B, C] = coeflist[idx][m]

            SST = SST_field_all[m]
            MSLP = MSLP_field_all[m]

            lat0, lat1, lon0, lon1 = preprocessing.BOUNDARIES_BASINS(idx)

            lat_0 = np.abs(lat - lat1).argmin()
            lat_1 = np.abs(lat - lat0).argmin()
            lon_0 = np.abs(lon - lon0).argmin()
            lon_1 = np.abs(lon - lon1).argmin()

            SST_field = SST[lat_0:lat_1, lon_0:lon_1]
            MSLP_field = MSLP[lat_0:lat_1, lon_0:lon_1]
            PC_MATRIX = np.zeros((SST_field.shape))
            PC_MATRIX[:] = np.nan

            PRESDROP = MPI_function(SST_field - 273.15, A, B, C)  # Vmax is given in m/s
            PC_MATRIX = MSLP_field - PRESDROP
            boundary = mpi_bounds[idx][midx]

            PC_MATRIX[PC_MATRIX < boundary] = boundary

            np.savetxt(
                os.path.join(__location__, "MPI_FIELDS_" + str(idx) + str(m) + ".txt"),
                PC_MATRIX,
            )


def PRESFUNCTION(X, a, b, c, d):
    """
    Fit the data to the pressure function.
    Parameters
    ----------
    X : array of change in pressure and difference between pressure and mpi ([dp0,p-mpi])
    a,b,c,d : Coefficients

    """
    dp, presmpi = X
    return a + b * dp + c * np.exp(-d * presmpi)


def PRESEXPECTED(dp, presmpi, a, b, c, d):
    """
    Calculate the forward change in pressure (dp1, p[i+1]-p[i])

    Parameters
    ----------
    dp : backward change in pressure (dp0, p[i]-p[i-1])
    presmpi : difference between pressure and mpi (p-mpi).
    a,b,c,d : coefficients

    Returns
    -------
    dp1_list : array of forward change in pressure (dp1, p[i+1]-p[i])

    """
    dp1_list = []
    for k in range(len(dp)):
        # ---- FIX: clamp exponent to prevent overflow ----
        exponent = -d * max(0.0, presmpi[k])
        dp1_list.append(a + b * dp[k] + c * np.exp(exponent))
    return dp1_list


def PRESEXPECTED_SIENA(
    dp0,
    presmpi,
    a,
    b,
    c,
    d,
    c_vws=0.0,
    c_rh=0.0,
    c_en=0.0,
    c_ln=0.0,
    vws=None,
    rh=None,
    i_en=None,
    i_ln=None,
):
    dp0 = np.asarray(dp0, dtype=float)
    presmpi = np.asarray(presmpi, dtype=float)
    base = a + b * dp0 + c * np.exp(-d * np.maximum(0.0, presmpi))
    if vws is not None:
        base = base + c_vws * np.nan_to_num(np.asarray(vws, dtype=float))
    if rh is not None:
        base = base + c_rh * np.nan_to_num(np.asarray(rh, dtype=float))
    if i_en is not None:
        base = base + c_en * np.asarray(i_en, dtype=float)
    if i_ln is not None:
        base = base + c_ln * np.asarray(i_ln, dtype=float)
    return base


def _fit_pressure_model_siena(dp0, presmpi, vws, rh, i_en, i_ln, dp1, lambda_phase=5.0):
    dp0 = np.asarray(dp0, dtype=float)
    presmpi = np.asarray(presmpi, dtype=float)
    vws = np.nan_to_num(np.asarray(vws, dtype=float))
    rh = np.nan_to_num(np.asarray(rh, dtype=float))
    i_en = np.asarray(i_en, dtype=float)
    i_ln = np.asarray(i_ln, dtype=float)
    dp1 = np.asarray(dp1, dtype=float)

    def residuals(theta):
        a, b, c, d, c_vws, c_rh, c_en, c_ln = theta
        pred = PRESEXPECTED_SIENA(
            dp0, presmpi, a, b, c, d, c_vws, c_rh, c_en, c_ln, vws, rh, i_en, i_ln
        )
        res = dp1 - pred
        if lambda_phase and lambda_phase > 0:
            res = np.concatenate([res, np.sqrt(lambda_phase) * np.array([c_en, c_ln])])
        return res

    # Stable initialization close to original STORM structure
    a0 = 0.0
    b0 = 0.5
    c0 = max(1.0, np.nanstd(dp1) if len(dp1) else 1.0)
    d0 = 0.05
    x0 = np.array([a0, b0, c0, d0, 0.0, 0.0, 0.0, 0.0], dtype=float)

    lower = np.array([-50.0, -5.0, 0.0, 0.0, -20.0, -20.0, -20.0, -20.0])
    upper = np.array([50.0, 5.0, 100.0, 5.0, 20.0, 20.0, 20.0, 20.0])

    result = least_squares(
        residuals, x0=x0, bounds=(lower, upper), method="trf", max_nfev=5000
    )
    theta = result.x
    pred = PRESEXPECTED_SIENA(
        dp0,
        presmpi,
        *theta[:4],
        c_vws=theta[4],
        c_rh=theta[5],
        c_en=theta[6],
        c_ln=theta[7],
        vws=vws,
        rh=rh,
        i_en=i_en,
        i_ln=i_ln,
    )
    resid = dp1 - pred
    mu = float(np.mean(resid))
    centered = resid - mu
    neg = centered[centered < 0]
    pos = centered[centered >= 0]
    # Two-piece normal: separate sigma for intensification (neg dp) vs weakening (pos dp)
    # This captures the asymmetric tails of TC pressure change residuals, where
    # rapid intensification events produce a heavier left tail than the symmetric
    # normal assumed in Bloemendaal et al. (2020).
    # (John 1982, Commun. Stat. Theory Methods 11(8), 879-885)

    std_floor = 1e-6
    std_sym = float(np.std(resid))
    if not np.isfinite(std_sym) or std_sym < std_floor:
        std_sym = std_floor
    std_neg = float(np.sqrt(np.mean(neg**2))) if len(neg) > 1 else std_sym
    std_pos = float(np.sqrt(np.mean(pos**2))) if len(pos) > 1 else std_sym
    std_neg = max(std_neg, std_floor)
    std_pos = max(std_pos, std_floor)
    return theta, pred, resid, mu, std_neg, std_pos


# =========================================================================
# H1 FIX: Two-stage CV for pressure model lambda selection
# =========================================================================

PRESSURE_LAMBDA_GRID = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]


def _select_lambda_pressure_cv(df_basin, lambda_grid=None, min_fold_size=20):
    """
    Two-stage leave-one-year-out CV for pressure model ridge penalty.

    Stage 1: Fit base model (a,b,c,d,c_vws,c_rh) WITHOUT ENSO terms
             (using very high lambda to zero them out) on full basin data.
    Stage 2: Compute residuals from Stage 1. These residuals contain the
             ENSO signal. Run linear CV on residual ~ c_en*I_EN + c_ln*I_LN
             to select optimal lambda for the ENSO terms.

    This two-stage approach is standard for partially linear models and
    avoids running 320+ nonlinear least_squares fits during CV.

    Returns
    -------
    best_lambda : float
    cv_results : list of (lambda, mean_mse)
    """
    if lambda_grid is None:
        lambda_grid = PRESSURE_LAMBDA_GRID

    dp0 = df_basin["DP0"].values
    dp1 = df_basin["DP1"].values
    presmpi = np.maximum(0.0, df_basin["Pressure"].values - df_basin["MPI"].values)
    vws = np.nan_to_num(df_basin["VWS"].values.astype(float))
    rh = np.nan_to_num(df_basin["RH600"].values.astype(float))
    i_en = df_basin["I_EN"].values.astype(float)
    i_ln = df_basin["I_LN"].values.astype(float)
    years = df_basin["Year"].values.astype(float)

    if len(dp0) < 100:
        return 5.0, []

    # Stage 1: Fit base model with ENSO terms effectively zeroed
    # (lambda_phase=1e6 forces c_en ≈ 0, c_ln ≈ 0)
    try:
        theta_base, _, _, _, _, _ = _fit_pressure_model_siena(
            dp0, presmpi, vws, rh, i_en, i_ln, dp1, lambda_phase=1e6
        )
    except Exception:
        return 5.0, []

    # Base prediction (ENSO terms are ~0)
    base_pred = PRESEXPECTED_SIENA(
        dp0,
        presmpi,
        *theta_base[:4],
        c_vws=theta_base[4],
        c_rh=theta_base[5],
        c_en=0.0,
        c_ln=0.0,
        vws=vws,
        rh=rh,
        i_en=i_en,
        i_ln=i_ln,
    )
    residuals_base = dp1 - base_pred

    # Stage 2: Linear CV on residual ~ c_en*I_EN + c_ln*I_LN
    X_enso = np.column_stack([i_en, i_ln])
    # After intercept prepend: col 0=intercept, 1=I_EN, 2=I_LN
    from CODE.siena_utils import select_lambda_cv

    best_lambda, best_mse, cv_results = select_lambda_cv(
        X_enso,
        residuals_base,
        years,
        penalty_cols=[1, 2],
        lambda_grid=lambda_grid,
        add_intercept=True,
        min_fold_size=min_fold_size,
    )
    return best_lambda, cv_results


def pressure_coefficients(idx_basin, months, months_for_coef, lambda_phase=None):
    """
    Calculate pressure coefficients using pooled fitting while preserving the original
    nonlinear James-Mason / MPI mean structure.

    H1 FIX: If lambda_phase is None (default), select lambda per basin via
    two-stage leave-one-year-out CV. If a float is provided, use that value
    for all basins (legacy behavior).

    Fitted mean function per local 5x5 degree cell:
        dp1 = a + b*dp0 + c*exp(-d*(p-MPI)+) + c_vws*VWS + c_rh*RH + c_en*I_EN + c_ln*I_LN + eps

    Only the ENSO terms (c_en, c_ln) are shrinkage-penalized.
    Stored coefficient order per cell is:
        [a, b, c, d, mu, std_neg, std_pos, mpi, c_vws, c_rh, c_en, c_ln]
    where mu/std_neg/std_pos are residual moments of eps = dp1 - E[dp1 | X].
    """
    data = xr.open_dataset(os.path.join(__location__, "Monthly_mean_SST.nc"))
    lon = data.longitude.values
    lat = data.latitude.values
    data.close()
    step = 5
    pres_variables = np.load(
        os.path.join(__location__, "TC_PRESSURE_VARIABLES.npy"), allow_pickle=True
    ).item()

    coeflist = {i: [] for i in range(0, 6)}
    basin_names = ["EP", "NA", "NI", "SI", "SP", "WP"]
    lambda_report = {}

    for ii in range(len(idx_basin)):
        idx = idx_basin[ii]
        coeflist[idx] = {i: [] for i in months_for_coef[idx]}

        lat0, lat1, lon0, lon11 = preprocessing.BOUNDARIES_BASINS(idx)
        lat_0 = np.abs(lat - lat1).argmin()
        lon_0 = np.abs(lon - lon0).argmin()

        # --- H1 FIX: Basin-level lambda selection via CV ---
        # Pool all months for this basin to select lambda once
        if lambda_phase is None:
            df_all_months = pd.DataFrame(
                {
                    "Latitude": pres_variables[3][idx],
                    "Longitude": pres_variables[4][idx],
                    "Pressure": pres_variables[2][idx],
                    "DP0": pres_variables[0][idx],
                    "DP1": pres_variables[1][idx],
                    "Month": pres_variables[5][idx],
                    "Phase": pres_variables[6][idx]
                    if 6 in pres_variables
                    else np.ones(len(pres_variables[0][idx])),
                    "Year": pres_variables[7][idx]
                    if 7 in pres_variables
                    else np.zeros(len(pres_variables[0][idx])),
                    "VWS": pres_variables[8][idx]
                    if 8 in pres_variables
                    else np.nan * np.ones(len(pres_variables[0][idx])),
                    "RH600": pres_variables[9][idx]
                    if 9 in pres_variables
                    else np.nan * np.ones(len(pres_variables[0][idx])),
                }
            )
            df_all_months = df_all_months[
                (df_all_months["Pressure"] > 0.0)
                & (df_all_months["DP0"] > -10000.0)
                & (df_all_months["DP1"] > -10000.0)
            ].copy()
            df_all_months["I_EN"] = (df_all_months["Phase"] == 2).astype(float)
            df_all_months["I_LN"] = (df_all_months["Phase"] == 0).astype(float)
            # Use median MPI across all cells as a representative value
            # (cell-specific MPI is used in the actual fit, this is only for CV)
            df_all_months["MPI"] = df_all_months["Pressure"].median() - 30.0

            basin_lambda, cv_results = _select_lambda_pressure_cv(
                df_all_months, lambda_grid=PRESSURE_LAMBDA_GRID
            )
            print(f"  {basin_names[idx]}: CV-selected pressure λ={basin_lambda:.1f}")
            if cv_results:
                for lam_val, mse_val in cv_results:
                    print(f"    λ={lam_val:.1f} -> MSE={mse_val:.4f}")
        else:
            basin_lambda = float(lambda_phase)
            print(f"  {basin_names[idx]}: Fixed pressure λ={basin_lambda}")
        lambda_report[idx] = basin_lambda

        for i in range(len(months[idx])):
            m = months[idx][i]
            m_coef = months_for_coef[idx][i]
            print(idx, m)
            # Fix 2: Prefer thermodynamic PI fields over empirical MPI
            pi_path = os.path.join(__location__, f"Monthly_mean_PI_{m}.txt")
            mpi_path = os.path.join(
                __location__, "MPI_FIELDS_" + str(idx) + str(m) + ".txt"
            )
            if os.path.exists(pi_path):
                PI_GLOBAL = np.loadtxt(pi_path)
                lat_0_pi = np.abs(lat - lat1).argmin()
                lat_1_pi = np.abs(lat - lat0).argmin()
                lon_0_pi = np.abs(lon - lon0).argmin()
                lon_1_pi = np.abs(lon - lon11).argmin()
                MPI_MATRIX = PI_GLOBAL[lat_0_pi:lat_1_pi, lon_0_pi:lon_1_pi]
                print(f"  Using thermodynamic PI field for month {m}")
            elif os.path.exists(mpi_path):
                MPI_MATRIX = np.loadtxt(mpi_path)
                print(f"  Falling back to empirical MPI field for month {m}")
            else:
                print(f"  WARNING: No PI or MPI field found for idx={idx}, month={m}")
                continue

            # ── Load phase-specific PI fields for point-level assignment ──
            # Each observation will get PI from its ENSO phase's field at
            # its exact (lat, lon). This makes fitting consistent with runtime,
            # where SAMPLE_TC_PRESSURE loads the phase-specific PI at 0.25°.
            # The pooled MPI_MATRIX above is kept for the 5° cell fallback
            # stored in row[7] of COEFFICIENTS_JM_PRESSURE.
            _pi_fields_month = {}
            _pi_fields_month[None] = PI_GLOBAL if os.path.exists(pi_path) else None
            for _ph in ["LN", "NEU", "EN"]:
                _ph_path = os.path.join(__location__, f"Monthly_mean_PI_{m}_{_ph}.txt")
                if os.path.exists(_ph_path):
                    _pi_fields_month[_ph] = np.loadtxt(_ph_path)
                else:
                    _pi_fields_month[_ph] = _pi_fields_month[None]
            _n_phase_pi = sum(
                1
                for _ph in ["LN", "NEU", "EN"]
                if _pi_fields_month[_ph] is not _pi_fields_month[None]
            )
            print(
                f"  Point-level PI: {_n_phase_pi}/3 phase-specific fields loaded"
                f" (fallback to pooled for the rest)"
            )

            lat_df, lon_df, mpi_df = [], [], []
            for i0 in range(len(MPI_MATRIX[:, 0])):
                for j0 in range(len(MPI_MATRIX[0, :])):
                    lat_df.append(lat[i0 + lat_0])
                    lon_df.append(lon[j0 + lon_0])
                    mpi_df.append(MPI_MATRIX[i0, j0])

            df = pd.DataFrame({"Latitude": lat_df, "Longitude": lon_df, "MPI": mpi_df})
            to_bin = lambda x: np.floor(x / step) * step
            df["latbin"] = df.Latitude.map(to_bin)
            df["lonbin"] = df.Longitude.map(to_bin)
            MPI = df.groupby(["latbin", "lonbin"])["MPI"].apply(list)

            latbins1 = np.linspace(lat0, lat1 - 5, (lat1 - 5 - lat0) // step + 1)
            lonbins1 = np.linspace(lon0, lon11 - 5, (lon11 - 5 - lon0) // step + 1)
            matrix_mpi = np.nan * np.ones(
                (int((lat1 - lat0) / 5), int((lon11 - lon0) / 5))
            )
            for latidx in latbins1:
                for lonidx in lonbins1:
                    i_ind = int((latidx - lat0) / 5.0)
                    j_ind = int((lonidx - lon0) / 5.0)
                    try:
                        matrix_mpi[i_ind, j_ind] = np.nanmedian(MPI[latidx][lonidx])
                    except Exception:
                        matrix_mpi[i_ind, j_ind] = np.nan

            df_data = pd.DataFrame(
                {
                    "Latitude": pres_variables[3][idx],
                    "Longitude": pres_variables[4][idx],
                    "Pressure": pres_variables[2][idx],
                    "DP0": pres_variables[0][idx],
                    "DP1": pres_variables[1][idx],
                    "Month": pres_variables[5][idx],
                    "Phase": pres_variables[6][idx]
                    if 6 in pres_variables
                    else np.ones(len(pres_variables[0][idx])),
                    "Year": pres_variables[7][idx]
                    if 7 in pres_variables
                    else np.zeros(len(pres_variables[0][idx])),
                    "VWS": pres_variables[8][idx]
                    if 8 in pres_variables
                    else np.nan * np.ones(len(pres_variables[0][idx])),
                    "RH600": pres_variables[9][idx]
                    if 9 in pres_variables
                    else np.nan * np.ones(len(pres_variables[0][idx])),
                }
            )
            df_data = df_data[
                (df_data["Pressure"] > 0.0)
                & (df_data["DP0"] > -10000.0)
                & (df_data["DP1"] > -10000.0)
                & (df_data["Longitude"] >= lon0)
                & (df_data["Longitude"] < lon11)
                & (df_data["Latitude"] >= lat0)
                & (df_data["Latitude"] < lat1)
            ]
            df_data1 = df_data[df_data["Month"] == m].copy()
            if len(df_data1) == 0:
                continue
            df_data1["latbin"] = df_data1.Latitude.map(to_bin)
            df_data1["lonbin"] = df_data1.Longitude.map(to_bin)
            df_data1["I_EN"] = (df_data1["Phase"] == 2).astype(float)
            df_data1["I_LN"] = (df_data1["Phase"] == 0).astype(float)

            latbins = np.unique(df_data1["latbin"])
            lonbins = df_data1.groupby("latbin")["lonbin"].apply(list)
            lon1 = lon11

            matrices = {
                name: -100 * np.ones((int((lat1 - lat0) / 5), int((lon1 - lon0) / 5)))
                for name in [
                    "mean",
                    "std_neg",
                    "std_pos",
                    "c0",
                    "c1",
                    "c2",
                    "c3",
                    "cvws",
                    "crh",
                    "cen",
                    "cln",
                ]
            }
            lijst = []
            for latidx in latbins:
                for lonidx in np.unique(lonbins[latidx]):
                    lijst.append((latidx, lonidx))

            failed = 0
            for latidx in latbins:
                for lonidx in np.unique(lonbins[latidx]):
                    i_ind = int((latidx - lat0) / 5.0)
                    j_ind = int((lonidx - lon0) / 5.0)
                    subset = []
                    _phase_int_to_str = {0: "LN", 1: "NEU", 2: "EN"}
                    for lat_sur in [-5, 0, 5]:
                        for lon_sur in [-5, 0, 5]:
                            key = (int(latidx + lat_sur), int(lonidx + lon_sur))
                            if key in lijst:
                                # Get cell-level median as fallback
                                try:
                                    cell_median = np.nanmedian(
                                        MPI[latidx + lat_sur][lonidx + lon_sur]
                                    )
                                except Exception:
                                    cell_median = np.nan

                                chunk = df_data1[
                                    (df_data1["latbin"] == latidx + lat_sur)
                                    & (df_data1["lonbin"] == lonidx + lon_sur)
                                ].copy()
                                if len(chunk) > 0:
                                    # Point-level PI: each obs gets PI from
                                    # its phase's field at its (lat, lon)
                                    pi_vals = np.full(len(chunk), np.nan)
                                    for _k in range(len(chunk)):
                                        _olat = float(chunk.iloc[_k]["Latitude"])
                                        _olon = float(chunk.iloc[_k]["Longitude"])
                                        _oph = (
                                            int(chunk.iloc[_k]["Phase"])
                                            if "Phase" in chunk.columns
                                            else 1
                                        )
                                        _ph_str = _phase_int_to_str.get(_oph)
                                        _pi_fld = _pi_fields_month.get(
                                            _ph_str, _pi_fields_month.get(None)
                                        )
                                        if _pi_fld is not None:
                                            _li = int(round((90.0 - _olat) / 0.25))
                                            _lo = (
                                                int(round((_olon % 360.0) / 0.25))
                                                % 1440
                                            )
                                            if (
                                                0 <= _li < _pi_fld.shape[0]
                                                and 0 <= _lo < _pi_fld.shape[1]
                                            ):
                                                _v = float(_pi_fld[_li, _lo])
                                                if np.isfinite(_v) and _v > 0:
                                                    pi_vals[_k] = _v
                                    # Fill remaining NaNs with cell median
                                    if np.isfinite(cell_median) and cell_median > 0:
                                        _nan_mask = ~np.isfinite(pi_vals) | (
                                            pi_vals <= 0
                                        )
                                        pi_vals[_nan_mask] = cell_median
                                    chunk["MPI"] = pi_vals
                                    chunk = chunk[
                                        chunk["MPI"].notna() & (chunk["MPI"] > 0)
                                    ]
                                    if len(chunk) > 0:
                                        subset.append(chunk)
                    if len(subset) == 0:
                        continue
                    sub = pd.concat(subset, ignore_index=True)
                    if len(sub) > 9:
                        presmpi = np.maximum(
                            0.0, sub["Pressure"].values - sub["MPI"].values
                        )
                        try:
                            # H1 FIX: use basin_lambda from CV (or fixed value)
                            theta, pred, resid, mu, std_neg, std_pos = (
                                _fit_pressure_model_siena(
                                    sub["DP0"].values,
                                    presmpi,
                                    sub["VWS"].values,
                                    sub["RH600"].values,
                                    sub["I_EN"].values,
                                    sub["I_LN"].values,
                                    sub["DP1"].values,
                                    lambda_phase=basin_lambda,
                                )
                            )
                            a, b, c, d, c_vws, c_rh, c_en, c_ln = theta
                            if abs(mu) < 2 and c > 0 and d >= 0:
                                matrices["mean"][i_ind, j_ind] = mu
                                matrices["std_neg"][i_ind, j_ind] = std_neg
                                matrices["std_pos"][i_ind, j_ind] = std_pos
                                matrices["c0"][i_ind, j_ind] = a
                                matrices["c1"][i_ind, j_ind] = b
                                matrices["c2"][i_ind, j_ind] = c
                                matrices["c3"][i_ind, j_ind] = d
                                matrices["cvws"][i_ind, j_ind] = c_vws
                                matrices["crh"][i_ind, j_ind] = c_rh
                                matrices["cen"][i_ind, j_ind] = c_en
                                matrices["cln"][i_ind, j_ind] = c_ln
                        except RuntimeError:
                            failed += 1
            print(str(failed) + " fields could not be fit directly")
            print("Filling succeeded")

            Xdim, Ydim = matrices["mean"].shape
            neighbors = lambda x, y: [
                (x2, y2)
                for (x2, y2) in [
                    (x, y - 1),
                    (x, y + 1),
                    (x + 1, y),
                    (x - 1, y),
                    (x - 1, y - 1),
                    (x - 1, y + 1),
                    (x + 1, y - 1),
                    (x + 1, y + 1),
                ]
                if (
                    -1 < x < Xdim
                    and -1 < y < Ydim
                    and (x != x2 or y != y2)
                    and (0 <= x2 < Xdim)
                    and (0 <= y2 < Ydim)
                )
            ]

            for name in [
                "mean",
                "std_neg",
                "std_pos",
                "c0",
                "c1",
                "c2",
                "c3",
                "cvws",
                "crh",
                "cen",
                "cln",
            ]:
                matrix = matrices[name]
                var = 100
                while var != 0:
                    shadowmatrix = np.zeros((Xdim, Ydim))
                    zeroeslist = [
                        [i1, j1]
                        for i1, x in enumerate(matrix)
                        for j1, y in enumerate(x)
                        if y == -100
                    ]
                    var = len(zeroeslist)
                    for [i1, j1] in zeroeslist:
                        for i0, j0 in neighbors(i1, j1):
                            if matrix[i0, j0] != -100 and shadowmatrix[i0, j0] == 0:
                                matrix[i1, j1] = matrix[i0, j0]
                                shadowmatrix[i1, j1] = 1
                                break
                matrices[name] = matrix

            Xmpi, Ympi = matrix_mpi.shape
            var = 100
            while var != 0:
                shadowmatrix = np.zeros((Xmpi, Ympi))
                zeroeslist = [
                    [i1, j1]
                    for i1, x in enumerate(matrix_mpi)
                    for j1, y in enumerate(x)
                    if not np.isfinite(y)
                ]
                var = len(zeroeslist)
                for [i1, j1] in zeroeslist:
                    neigh = [
                        (x2, y2)
                        for (x2, y2) in [
                            (i1, j1 - 1),
                            (i1, j1 + 1),
                            (i1 + 1, j1),
                            (i1 - 1, j1),
                            (i1 - 1, j1 - 1),
                            (i1 - 1, j1 + 1),
                            (i1 + 1, j1 - 1),
                            (i1 + 1, j1 + 1),
                        ]
                        if (0 <= x2 < Xmpi and 0 <= y2 < Ympi)
                    ]
                    for i0, j0 in neigh:
                        if (
                            np.isfinite(matrix_mpi[i0, j0])
                            and shadowmatrix[i0, j0] == 0
                        ):
                            matrix_mpi[i1, j1] = matrix_mpi[i0, j0]
                            shadowmatrix[i1, j1] = 1
                            break

            for i0 in range(0, matrix_mpi.shape[0]):
                for j0 in range(0, matrix_mpi.shape[1]):
                    coeflist[idx][m_coef].append(
                        [
                            matrices["c0"][i0, j0],
                            matrices["c1"][i0, j0],
                            matrices["c2"][i0, j0],
                            matrices["c3"][i0, j0],
                            matrices["mean"][i0, j0],
                            matrices["std_neg"][i0, j0],
                            matrices["std_pos"][i0, j0],
                            matrix_mpi[i0, j0],
                            matrices["cvws"][i0, j0],
                            matrices["crh"][i0, j0],
                            matrices["cen"][i0, j0],
                            matrices["cln"][i0, j0],
                        ]
                    )

        np.save(os.path.join(__location__, "COEFFICIENTS_JM_PRESSURE.npy"), coeflist)

    # Save lambda report for paper
    np.save(os.path.join(__location__, "PRESSURE_LAMBDA_REPORT.npy"), lambda_report)
    print("Pressure lambda report saved to PRESSURE_LAMBDA_REPORT.npy")
