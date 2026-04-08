import numpy as np
from climada.hazard import TCTracks
from pathlib import Path


def add_p_env(path_to_file : str | Path, tracks : TCTracks) -> TCTracks:

    # Read raw file to grab column 14 (0-indexed: col 13)
    raw = np.loadtxt(path_to_file)  # STORM uses whitespace-delimited

    # The tracks in tracks.data are built sequentially from the file,
    # one per (year, storm_num) pair, in file order.
    row_idx = 0
    for track in tracks.data:
        n_steps = track.sizes["time"]
        penv = raw[row_idx : row_idx + n_steps, 13]
        track["environmental_pressure"] = ("time", penv)
        row_idx += n_steps

    assert row_idx == raw.shape[0], "Row count mismatch — check for filtering in reader"

    return tracks
