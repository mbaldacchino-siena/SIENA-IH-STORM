# -*- coding: utf-8 -*-
"""
This module is part of the IH-STORM model

Functions described here are part of the data pre-processing.

Copyright (C) 2024 Itxaso Odériz.
"""

import ast
import numpy as np


def _parse_value(value):
    value = value.strip()
    try:
        return ast.literal_eval(value)
    except Exception:
        return value.strip("'\"")


def input_data(file):
    variables = {}
    with open(file, 'r', encoding='utf-8') as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw or raw.startswith('#') or '=' not in raw:
                continue
            key, value = raw.split('=', 1)
            variables[key.strip()] = _parse_value(value)

    period = list(map(int, variables.get('period', [1980, 2021])))
    climate_index = str(variables.get('climate_index', 'ONI')).strip("'")
    threshold = float(variables.get('threshold', 0.5))
    idx_basin = list(map(int, variables.get('idx_basin', [0, 1, 2, 3, 4, 5])))
    months = [list(map(int, m)) for m in variables.get('months', [])]
    mpi_bounds = [list(map(int, b)) for b in variables.get('mpi_bounds', [])]
    months_for_coef_MPI = [list(map(int, b)) for b in variables.get('months_for_coef_MPI', mpi_bounds)]
    months_for_coef_PRESS = [list(map(int, b)) for b in variables.get('months_for_coef_PRESS', mpi_bounds)]
    generation_phase = variables.get('generation_phase', 'NEU')

    return period, climate_index, threshold, idx_basin, months, mpi_bounds, months_for_coef_MPI, months_for_coef_PRESS, generation_phase
