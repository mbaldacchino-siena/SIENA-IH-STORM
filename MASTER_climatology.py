# -*- coding: utf-8 -*-
"""
This module is part of the IH-STORM model

Functions described here are part of the data pre-processing.

Copyright (C) 2024 Itxaso Odériz.
"""

import os
import sys
import import_data
import climatology


dir_path = os.path.dirname(os.path.realpath(sys.argv[0]))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

(
    period,
    climate_index,
    threshold,
    idx_basin,
    months,
    mpi_bounds,
    months_for_coef_MPI,
    months_for_coef_PRESS,
    generation_phase,
) = import_data.input_data("input.dat")

print("period:", period)
print("climate_index:", climate_index)
print("threshold:", threshold)
print("idx_basin:", idx_basin)
print("months:", months)
print("mpi_bounds:", mpi_bounds)

# Always prepare the all-years inputs first.
climatology.climatology_data(period)

# Then derive pooled + phase-aware climatologies from the selected index.
if climate_index.lower() != "none":
    climatology.build_pooled_and_phase_climatologies(
        period, climate_index=climate_index, threshold=threshold
    )
