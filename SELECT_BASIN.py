# -*- coding: utf-8 -*-
"""
STORM module for simulation of genesis month, frequency, and basin boundaries.
SIENA extension: optional ENSO-phase-aware sampling with pooled fallback.
"""
import numpy as np
import random
import os
import sys
from siena_utils import normalize_phase, phase_code

dir_path=os.path.dirname(os.path.realpath(sys.argv[0]))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


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
        lat0,lat1,lon0,lon1=5,60,255,359
    if idx==2:
        lat0,lat1,lon0,lon1=5,60,30,100
    if idx==3:
        lat0,lat1,lon0,lon1=-60,-5,10,135
    if idx==4:
        lat0,lat1,lon0,lon1=-60,-5,135,240
    if idx==5:
        lat0,lat1,lon0,lon1=5,60,100,180
    return s,month,lat0,lat1,lon0,lon1
