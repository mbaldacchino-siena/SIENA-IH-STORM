# -*- coding: utf-8 -*-
"""SIENA-IH-STORM generation entry point with phase argument."""
import argparse
import numpy as np
import os
import time
from SELECT_BASIN import Basins_WMO
from SAMPLE_STARTING_POINT import Startingpoint
from SAMPLE_TC_MOVEMENT import TC_movement
from SAMPLE_TC_PRESSURE import TC_pressure
import import_data

dir_path=os.path.dirname(os.path.realpath(__file__))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def run_for_basin(basin, total_years, loop, phase):
    for nloop in range(loop):
        print('basin:', basin, 'phase:', phase, nloop)
        TC_data=[]
        for year in range(total_years):
            storms_per_year,genesis_month,lat0,lat1,lon0,lon1=Basins_WMO(basin, phase=phase)
            if storms_per_year>0:
                lon_genesis_list,lat_genesis_list=Startingpoint(storms_per_year,genesis_month,basin, phase=phase)
                latlist,lonlist,landfalllist=TC_movement(lon_genesis_list,lat_genesis_list,basin, phase=phase)
                TC_data=TC_pressure(basin,latlist,lonlist,landfalllist,year,storms_per_year,genesis_month,TC_data, phase=phase)
        TC_data=np.array(TC_data)
        out = f'STORM_DATA_IBTRACS_{basin}_{phase}_{total_years}_YEARS_{nloop}.txt'
        np.savetxt(os.path.join(__location__, out), TC_data, fmt='%5s', delimiter=',')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default=None, choices=['LN','NEU','EN', None], help='ENSO phase to generate')
    parser.add_argument('--years', type=int, default=1000)
    parser.add_argument('--loop', type=int, default=1)
    parser.add_argument('--basins', nargs='*', default=['EP','NA','NI','SI','SP','WP'])
    args = parser.parse_args()
    *_, generation_phase = import_data.input_data('input.dat')
    phase = args.phase or generation_phase
    start_time=time.time()
    for basin in args.basins:
        run_for_basin(basin, args.years, args.loop, phase)
    print('Elapsed:', time.time()-start_time)

if __name__ == '__main__':
    main()
