# -*- coding: utf-8 -*-
"""STORM genesis-location sampler with optional ENSO-phase-specific grids."""
import numpy as np
import random
import os
import sys
from SELECT_BASIN import Basins_WMO
from siena_utils import normalize_phase

dir_path=os.path.dirname(os.path.realpath(sys.argv[0]))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def Check_EP_formation(lat,lon):
    return not (lon>276 and lat>20)


def Check_NA_formation(lat,lon):
    return not (lon<276 and lat<20)


def Check_if_landfall(lat,lon,basin,land_mask):
    s,monthdummy,lat0_WMO,lat1_WMO,lon0_WMO,lon1_WMO=Basins_WMO(basin)
    x=int(10*(lon-lon0_WMO))
    y=int(10*(lat1_WMO-lat))
    return land_mask[y,x]


def Startingpoint(no_storms,monthlist,basin,phase=None):
    phase = normalize_phase(phase)
    basins=['EP','NA','NI','SI','SP','WP']
    basin_name = dict(zip(basins,[0,1,2,3,4,5]))
    idx=basin_name[basin]
    lon_coordinates=[]
    lat_coordinates=[]
    s,monthdummy,lat0,lat1,lon0,lon1=Basins_WMO(basin, phase=phase)
    land_mask=np.loadtxt(os.path.join(dir_path,'Land_ocean_mask_'+str(basin)+'.txt'))

    for month in monthlist:
        if phase is not None and os.path.exists(os.path.join(dir_path, f'GRID_GENESIS_MATRIX_{idx}_{month}_{phase}.txt')):
            grid_path = os.path.join(dir_path, f'GRID_GENESIS_MATRIX_{idx}_{month}_{phase}.txt')
        else:
            grid_path = os.path.join(dir_path, f'GRID_GENESIS_MATRIX_{idx}_{month}.txt')
        grid_copy=np.loadtxt(grid_path)
        grid_copy=np.array(grid_copy)
        grid_copy=np.round(grid_copy,1)
        weighted_list_index=[]
        for i in range(0,len(grid_copy[:,0])):
            for j in range(0,len(grid_copy[0,:])):
                value=max(int(10*grid_copy[i,j]), 0)
                if value>0:
                    weighted_list_index.extend([i*(len(grid_copy[0,:])-1)+j]*value)
        var=0
        while var==0:
            idx0=random.choice(weighted_list_index)
            row=int(np.floor(idx0/(len(grid_copy[0,:])-1)))
            col=int(idx0%(len(grid_copy[0,:])-1))
            lat_pert=random.uniform(0,0.94)
            lon_pert=random.uniform(0,0.94)
            lon=lon0+round(col+lon_pert,1)
            lat=lat1-round(row+lat_pert,1)
            if lon<lon1 and lat<lat1:
                check=Check_if_landfall(lat,lon,basin,land_mask)
                if basin=='EP':
                    check = check or Check_EP_formation(lat,lon)
                if basin=='NA':
                    check = check or Check_NA_formation(lat,lon)
                if check==0:
                    var=1
                    lon_coordinates.append(lon)
                    lat_coordinates.append(lat)
    return lon_coordinates,lat_coordinates
