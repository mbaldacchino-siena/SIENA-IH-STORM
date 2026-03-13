# -*- coding: utf-8 -*-
"""
This module is part of the STORM model

For more information, please see
Bloemendaal, N., Haigh, I.D., de Moel, H. et al.
Generation of a global synthetic tropical cyclone hazard dataset using STORM.
Sci Data 7, 40 (2020). https://doi.org/10.1038/s41597-020-0381-2

Functions described here are part of the data pre-processing and derive the coefficients
of the regression formulas.
"""

import numpy as np
from scipy.stats import norm
import pandas as pd
import preprocessing
import os
import sys
from siena_utils import solve_ridge

dir_path=os.path.dirname(os.path.realpath(sys.argv[0]))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def LATEXPECTED(dlat, lat, beta):
    return beta[0] + beta[1] * np.asarray(dlat) + beta[2] * np.asarray(lat) + beta[3] * np.asarray(beta[-2:]).sum() * 0


def track_coefficients(lambda_phase=5.0):
    """Calculate pooled track coefficients with ENSO phase dummies."""
    step=5.
    data=np.load(os.path.join(__location__,'TC_TRACK_VARIABLES.npy'),allow_pickle=True).item()
    coefficients_list={i:[] for i in range(0,6)}

    for idx in range(0,6):
        df=pd.DataFrame({
            'Latitude':data[4][idx],
            'Longitude':data[5][idx],
            'Dlat0':data[0][idx],
            'Dlat1':data[1][idx],
            'Dlon0':data[2][idx],
            'Dlon1':data[3][idx],
            'Phase':data[6][idx] if 6 in data else np.ones(len(data[0][idx])),
        })
        lat0,lat1,lon0,lon1=preprocessing.BOUNDARIES_BASINS(idx)
        df=df[(df['Latitude']<=lat1) & (df['Latitude']>=lat0) & (df['Longitude']<=lon1) & (df['Longitude']>=lon0)]
        latspace=np.linspace(lat0,lat1-5.,int(abs(lat0-lat1+5)/step)+1)

        to_bin=lambda x:np.floor(x/step)*step
        df['latbin']=df.Latitude.map(to_bin)
        coeff_array=[[0]]*len(latspace)
        count=0
        for latbin in np.unique(df['latbin']):
            i_ind=int((latbin-lat0)/step)
            sub = df[df['latbin'] == latbin].copy()
            if len(sub) > 50:
                sub['I_EN'] = (sub['Phase'] == 2).astype(float)
                sub['I_LN'] = (sub['Phase'] == 0).astype(float)
                try:
                    # latitude model: dlat1 ~ 1 + dlat0 + latitude + phase dummies
                    X_lat = np.column_stack([sub['Dlat0'].values, sub['Latitude'].values, sub['I_EN'].values, sub['I_LN'].values])
                    y_lat = sub['Dlat1'].values
                    beta_lat = solve_ridge(X_lat, y_lat, penalty_cols=[3,4], alpha=lambda_phase, add_intercept=True)
                    pred_lat = beta_lat[0] + X_lat @ beta_lat[1:]
                    e_lat = y_lat - pred_lat
                    Elatmu, Elatstd = norm.fit(e_lat)

                    # longitude model: dlon1 ~ 1 + dlon0 + phase dummies
                    X_lon = np.column_stack([sub['Dlon0'].values, sub['I_EN'].values, sub['I_LN'].values])
                    y_lon = sub['Dlon1'].values
                    beta_lon = solve_ridge(X_lon, y_lon, penalty_cols=[2,3], alpha=lambda_phase, add_intercept=True)
                    pred_lon = beta_lon[0] + X_lon @ beta_lon[1:]
                    e_lon = y_lon - pred_lon
                    Elonmu, Elonstd = norm.fit(e_lon)

                    Dlat0mu,Dlat0std=norm.fit(sub['Dlat0'].values)
                    Dlon0mu,Dlon0std=norm.fit(sub['Dlon0'].values)
                    if abs(Elatmu)<1 and abs(Elonmu)<1:
                        coeff_array[i_ind]=[
                            beta_lat[0], beta_lat[1], beta_lat[2], beta_lat[3], beta_lat[4],
                            beta_lon[0], beta_lon[1], beta_lon[2], beta_lon[3],
                            Elatmu, Elatstd, Elonmu, Elonstd,
                            Dlat0mu, Dlat0std, Dlon0mu, Dlon0std
                        ]
                        count += 1
                except Exception as exc:
                    print(f'No fit found for basin {idx}, latbin {latbin}: {exc}')

        if idx in (3,4):
            while count<len(latspace):
                for i in reversed(range(len(latspace))):
                    if len(coeff_array[i])==1:
                        if i<(len(latspace)-1) and len(coeff_array[i+1])>1:
                            coeff_array[i]=coeff_array[i+1]
                            count += 1
                        elif i>0 and len(coeff_array[i-1])>1:
                            coeff_array[i]=coeff_array[i-1]
                            count += 1
        else:
            while count<len(latspace):
                for i in range(len(latspace)):
                    if len(coeff_array[i])==1:
                        if i>0 and len(coeff_array[i-1])>1:
                            coeff_array[i]=coeff_array[i-1]
                            count += 1
                        elif i < len(latspace)-1 and len(coeff_array[i+1])>1:
                            coeff_array[i]=coeff_array[i+1]
                            count += 1

        coefficients_list[idx]=coeff_array

    np.save(os.path.join(__location__,'TRACK_COEFFICIENTS.npy'),coefficients_list)
