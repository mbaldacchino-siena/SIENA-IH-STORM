# -*- coding: utf-8 -*-
"""
Track coefficient fitting for SIENA-IH-STORM.

CHANGE LOG (VWS track update):
  - Replaced ENSO phase dummies (I_EN, I_LN) with VWS as a continuous
    physical covariate in both lat and lon track models.
  - No ridge penalty needed: VWS is a continuous predictor, not a sparse
    dummy. Plain OLS via np.linalg.lstsq.
  - Coefficient row layout (17 elements, same length as before):
      [0]  a_lat0     intercept (latitude)
      [1]  a_lat1     dlat0 coefficient
      [2]  a_lat2     latitude coefficient
      [3]  b_vws_lat  VWS coefficient (latitude)   ← was g_en
      [4]  0.0        unused                        ← was g_ln
      [5]  b_lon0     intercept (longitude)
      [6]  b_lon1     dlon0 coefficient
      [7]  b_vws_lon  VWS coefficient (longitude)   ← was d_en
      [8]  0.0        unused                        ← was d_ln
      [9]  Elatmu     latitude residual mean
      [10] Elatstd    latitude residual std
      [11] Elonmu     longitude residual mean
      [12] Elonstd    longitude residual std
      [13] Dlat0mu    initial dlat0 mean
      [14] Dlat0std   initial dlat0 std
      [15] Dlon0mu    initial dlon0 mean
      [16] Dlon0std   initial dlon0 std

  REQUIRES: re-run MASTER_preprocessing.py to regenerate TRACK_COEFFICIENTS.npy
"""

import numpy as np
from scipy.stats import norm
import pandas as pd
import CODE.preprocessing as preprocessing
import os
import sys

dir_path = os.path.dirname(os.path.realpath(sys.argv[0]))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def track_coefficients():
    """
    Calculate pooled track coefficients with VWS as environmental covariate.

    Models:
        dlat1 = a + b·dlat0 + c·lat + β_vws·VWS + ε_lat
        dlon1 = a + b·dlon0 + γ_vws·VWS + ε_lon

    VWS carries the ENSO signal implicitly: at runtime, phase-specific
    VWS climatology fields are loaded, so El Niño storms experience
    different VWS than La Niña storms at the same location.
    """
    step = 5.0
    data = np.load(
        os.path.join(__location__, "TC_TRACK_VARIABLES.npy"), allow_pickle=True
    ).item()
    coefficients_list = {i: [] for i in range(0, 6)}

    for idx in range(0, 6):
        # Build DataFrame including VWS
        vws_raw = data[8][idx] if 8 in data else [np.nan] * len(data[0][idx])

        df = pd.DataFrame(
            {
                "Latitude": data[4][idx],
                "Longitude": data[5][idx],
                "Dlat0": data[0][idx],
                "Dlat1": data[1][idx],
                "Dlon0": data[2][idx],
                "Dlon1": data[3][idx],
                "VWS": vws_raw,
            }
        )

        # Replace NaN VWS with basin mean (neutral effect)
        vws_mean = df["VWS"].median()
        if not np.isfinite(vws_mean):
            vws_mean = 10.0  # fallback
        df["VWS"] = df["VWS"].fillna(vws_mean)

        lat0, lat1, lon0, lon1 = preprocessing.BOUNDARIES_BASINS(idx)
        df = df[
            (df["Latitude"] <= lat1)
            & (df["Latitude"] >= lat0)
            & (df["Longitude"] <= lon1)
            & (df["Longitude"] >= lon0)
        ]
        latspace = np.linspace(lat0, lat1 - 5.0, int(abs(lat0 - lat1 + 5) / step) + 1)

        to_bin = lambda x: np.floor(x / step) * step
        df["latbin"] = df.Latitude.map(to_bin)
        coeff_array = [[0]] * len(latspace)
        count = 0

        for latbin in np.unique(df["latbin"]):
            i_ind = int((latbin - lat0) / step)
            sub = df[df["latbin"] == latbin].copy()

            if len(sub) > 50:
                try:
                    # ── Latitude model: dlat1 ~ 1 + dlat0 + lat + VWS ──
                    X_lat = np.column_stack(
                        [
                            np.ones(len(sub)),
                            sub["Dlat0"].values,
                            sub["Latitude"].values,
                            sub["VWS"].values,
                        ]
                    )
                    y_lat = sub["Dlat1"].values
                    beta_lat, *_ = np.linalg.lstsq(X_lat, y_lat, rcond=None)
                    # beta_lat = [intercept, b_dlat0, b_lat, b_vws_lat]

                    pred_lat = X_lat @ beta_lat
                    e_lat = y_lat - pred_lat
                    Elatmu, Elatstd = norm.fit(e_lat)

                    # ── Longitude model: dlon1 ~ 1 + dlon0 + VWS ──
                    X_lon = np.column_stack(
                        [
                            np.ones(len(sub)),
                            sub["Dlon0"].values,
                            sub["VWS"].values,
                        ]
                    )
                    y_lon = sub["Dlon1"].values
                    beta_lon, *_ = np.linalg.lstsq(X_lon, y_lon, rcond=None)
                    # beta_lon = [intercept, b_dlon0, b_vws_lon]

                    pred_lon = X_lon @ beta_lon
                    e_lon = y_lon - pred_lon
                    Elonmu, Elonstd = norm.fit(e_lon)

                    Dlat0mu, Dlat0std = norm.fit(sub["Dlat0"].values)
                    Dlon0mu, Dlon0std = norm.fit(sub["Dlon0"].values)

                    if abs(Elatmu) < 1 and abs(Elonmu) < 1:
                        coeff_array[i_ind] = [
                            beta_lat[0],  # [0] a_lat0: intercept
                            beta_lat[1],  # [1] a_lat1: dlat0 coeff
                            beta_lat[2],  # [2] a_lat2: latitude coeff
                            beta_lat[3],  # [3] b_vws_lat: VWS coeff (was g_en)
                            0.0,  # [4] unused (was g_ln)
                            beta_lon[0],  # [5] b_lon0: intercept
                            beta_lon[1],  # [6] b_lon1: dlon0 coeff
                            beta_lon[2],  # [7] b_vws_lon: VWS coeff (was d_en)
                            0.0,  # [8] unused (was d_ln)
                            Elatmu,  # [9]
                            Elatstd,  # [10]
                            Elonmu,  # [11]
                            Elonstd,  # [12]
                            Dlat0mu,  # [13]
                            Dlat0std,  # [14]
                            Dlon0mu,  # [15]
                            Dlon0std,  # [16]
                        ]
                        count += 1
                except Exception as exc:
                    print(f"No fit found for basin {idx}, latbin {latbin}: {exc}")

        # ── Gap-fill unfitted latitude bins (same logic as original) ──
        if idx in (3, 4):  # SH basins: fill from high lat toward equator
            while count < len(latspace):
                for i in reversed(range(len(latspace))):
                    if len(coeff_array[i]) == 1:
                        if i < (len(latspace) - 1) and len(coeff_array[i + 1]) > 1:
                            coeff_array[i] = coeff_array[i + 1]
                            count += 1
                        elif i > 0 and len(coeff_array[i - 1]) > 1:
                            coeff_array[i] = coeff_array[i - 1]
                            count += 1
        else:  # NH basins: fill from equator toward pole
            while count < len(latspace):
                for i in range(len(latspace)):
                    if len(coeff_array[i]) == 1:
                        if i > 0 and len(coeff_array[i - 1]) > 1:
                            coeff_array[i] = coeff_array[i - 1]
                            count += 1
                        elif i < len(latspace) - 1 and len(coeff_array[i + 1]) > 1:
                            coeff_array[i] = coeff_array[i + 1]
                            count += 1

        coefficients_list[idx] = coeff_array

    np.save(os.path.join(__location__, "TRACK_COEFFICIENTS.npy"), coefficients_list)
