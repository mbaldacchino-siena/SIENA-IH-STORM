# -*- coding: utf-8 -*-
"""
STORM module for simulation of genesis month, frequency, and basin boundaries.
SIENA extension: optional ENSO-phase-aware sampling with pooled fallback.
"""
import numpy as np
import random
import os
import sys
from CODE.siena_utils import normalize_phase, phase_code


__location__ = os.path.realpath(os.getcwd())
dir_path = __location__
def Genesis_month(idx, storms, phase=None):
    phase = normalize_phase(phase)
    if phase is not None and os.path.exists(os.path.join(__location__, 'GENESIS_MONTHS_PHASE.npy')):
        monthlist = np.load(os.path.join(__location__, 'GENESIS_MONTHS_PHASE.npy'), allow_pickle=True).item()
        choices = monthlist[idx].get(phase, [])
        if len(choices) > 0:
            return [int(np.random.choice(choices)) for _ in range(storms)]
    monthlist=np.load(os.path.join(__location__,'GENESIS_MONTHS.npy'),allow_pickle=True).item()
    return [int(np.random.choice(monthlist[idx])) for _ in range(storms)]


def Storms(idx, phase=None):
    phase = normalize_phase(phase)
    if phase is not None and os.path.exists(os.path.join(__location__, 'POISSON_GENESIS_PARAMETERS_PHASE.npy')):
        mu_dict = np.load(os.path.join(__location__, 'POISSON_GENESIS_PARAMETERS_PHASE.npy'), allow_pickle=True).item()
        mu = mu_dict[idx].get(phase_code(phase), 0)
        if mu > 0:
            poisson=np.random.poisson(mu,10000)
            return int(random.choice(poisson))
    mu_list=np.loadtxt(os.path.join(__location__,'POISSON_GENESIS_PARAMETERS.txt'))
    mu=float(mu_list[idx])
    poisson=np.random.poisson(mu,10000)
    return int(random.choice(poisson))


def Basins_WMO(basin, phase=None):
    basins=['EP','NA','NI','SI','SP','WP']
    basin_name = dict(zip(basins,[0,1,2,3,4,5]))
    idx=basin_name[basin]
    s=Storms(idx, phase=phase)
    month=Genesis_month(idx,s, phase=phase)

    if idx==0:
        lat0,lat1,lon0,lon1=5,60,180,285
    if idx==1:
        lat0,lat1,lon0,lon1=5,60,255,360
    if idx==2:
        lat0,lat1,lon0,lon1=5,60,30,100
    if idx==3:
        lat0,lat1,lon0,lon1=-60,-5,10,135
    if idx==4:
        lat0,lat1,lon0,lon1=-60,-5,135,240
    if idx==5:
        lat0,lat1,lon0,lon1=5,60,100,180
    return s,month,lat0,lat1,lon0,lon1



# =========================================================================
# Forecast mode: Adjusted monthly-based WMO
# =========================================================================


def Basins_WMO_forecast(
    basin,
    month_phases,
    poisson_phase_rate=None,
    genesis_months_phase=None,
    active_months=None,
):
    """
    Forecast-mode genesis: blended Poisson + multinomial month distribution.

    Parameters
    ----------
    basin : str
    month_phases : dict {month: "LN"|"NEU"|"EN"}
    poisson_phase_rate : dict from POISSON_GENESIS_PARAMETERS_PHASE.npy
    genesis_months_phase : dict from GENESIS_MONTHS_PHASE.npy
    active_months : list of int

    Returns
    -------
    Same as Basins_WMO: (storms, month_list, lat0, lat1, lon0, lon1)
    """
    from siena_utils import blended_genesis

    basins = ["EP", "NA", "NI", "SI", "SP", "WP"]
    idx = basins.index(basin)

    storms, month_list = blended_genesis(
        poisson_phase_rate,
        genesis_months_phase,
        idx,
        active_months,
        month_phases,
    )

    # Basin bounds (same as Basins_WMO)
    bounds = {
        0: (5, 60, 180, 285),
        1: (5, 60, 255, 360),
        2: (5, 60, 30, 100),
        3: (-60, -5, 10, 135),
        4: (-60, -5, 135, 240),
        5: (5, 60, 100, 180),
    }
    lat0, lat1, lon0, lon1 = bounds[idx]
    return storms, month_list, lat0, lat1, lon0, lon1