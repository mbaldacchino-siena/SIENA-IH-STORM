import numpy as np
from climada.hazard import TCTracks
from pathlib import Path
import datetime as dt
import xarray as xr
import pandas as pd
from climada.util import ureg


STORM_1MIN_WIND_FACTOR = 0.88
# Scaling factor used in Bloemendaal et al. (2020) to convert 1-minute sustained wind speeds to
# 10-minute sustained wind speeds.


def add_p_env(path_to_file: str | Path, tracks: TCTracks) -> TCTracks:

    # Read raw file to grab column 14 (0-indexed: col 13)
    raw = np.loadtxt(path_to_file, delimiter=",")  # STORM uses whitespace-delimited

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


def from_simulations_storm_mod(path, years=None) -> TCTracks:
    """Create new TCTracks object from STORM simulations

        Bloemendaal et al. (2020): Generation of a global synthetic tropical cyclone hazard
        dataset using STORM. Scientific Data 7(1): 40.

    Track data available for download from

        https://doi.org/10.4121/uuid:82c1dc0d-5485-43d8-901a-ce7f26cda35d

    Wind speeds are converted to 1-minute sustained winds through division by 0.88 (this value
    is taken from Bloemendaal et al. (2020), cited above).

    Parameters
    ----------
    path : str
        Full path to a txt-file as contained in the `data.zip` archive from the official source
        linked above.
    years : list of int, optional
        If given, only read the specified "years" from the txt-File. Note that a "year" refers
        to one ensemble of tracks in the data set that represents one sample year.

    Returns
    -------
    tracks : TCTracks
        TCTracks with data from the STORM simulations.

    Notes
    -----
    All tracks are set in the year 1980. The id of the year (starting from 0) is saved in the
    attribute 'id_no'. To obtain the year of each track use

    >>> years = [int(tr.attrs['id_no'] / 1000) for tr in tc_tracks.data]
    >>> # or, alternatively,
    >>> years = [int(tr.attrs['sid'].split("-")[-2]) for tr in tc_tracks.data]

    If a windfield is generated from these tracks using the method ``TropCylcone.from_tracks()``,
    the following should be considered:

    1. The frequencies will be set to ``1`` for each storm. Thus, in order to compute annual
       values, the frequencies of the TropCylone should be changed to ``1/number of years``.
    2. The storm year and the storm id are stored in the ``TropCyclone.event_name`` attribute.
    """
    basins = ["EP", "NA", "NI", "SI", "SP", "WP"]
    tracks_df = pd.read_csv(
        path,
        names=[
            "year",
            "time_start",
            "tc_num",
            "time_delta",
            "basin",
            "lat",
            "lon",
            "pres",
            "wind",
            "rmw",
            "category",
            "landfall",
            "dist_to_land",
            "pressure_env",
            "ref_year",
        ],
        converters={
            "time_start": lambda d: dt.datetime(1980, int(float(d)), 1, 0),
            "time_delta": lambda d: dt.timedelta(hours=3 * float(d)),
            "basin": lambda d: basins[int(float(d))],
        },
        dtype={
            "year": int,
            "tc_num": int,
            "category": int,
        },
    )

    # filter specified years
    if years is not None:
        tracks_df = tracks_df[np.isin(tracks_df["year"], years)]

    # a bug in the data causes some storm tracks to be double-listed:
    tracks_df = tracks_df.drop_duplicates(subset=["year", "tc_num", "time_delta"])

    # conversion of units
    tracks_df["rmw"] *= (1 * ureg.kilometer).to(ureg.nautical_mile).magnitude
    tracks_df["wind"] *= (1 * ureg.meter / ureg.second).to(ureg.knot).magnitude

    # convert from 10-minute to 1-minute sustained winds, see Bloemendaal et al. (2020)
    tracks_df["wind"] /= STORM_1MIN_WIND_FACTOR

    # conversion to absolute times
    tracks_df["time"] = tracks_df["time_start"] + tracks_df["time_delta"]

    tracks_df = tracks_df.drop(
        labels=["time_start", "time_delta", "landfall", "dist_to_land"], axis=1
    )

    # add tracks one by one
    last_perc = 0
    fname = Path(path).name
    groups = tracks_df.groupby(by=["year", "tc_num"])
    data = []
    for idx, group in groups:
        track_name = f"{fname}-{idx[0]}-{idx[1]}"

        data.append(
            xr.Dataset(
                {
                    "time_step": ("time", np.full(group["time"].shape, 3)),
                    "max_sustained_wind": ("time", group["wind"].values),
                    "central_pressure": ("time", group["pres"].values),
                    "radius_max_wind": ("time", group["rmw"].values),
                    "environmental_pressure": ("time", group["pressure_env"].values),
                    "basin": ("time", group["basin"].values.astype("<U2")),
                },
                coords={
                    "time": ("time", group["time"].values),
                    "lat": ("time", group["lat"].values),
                    "lon": ("time", group["lon"].values),
                },
                attrs={
                    "max_sustained_wind_unit": "kn",
                    "central_pressure_unit": "mb",
                    "name": track_name,
                    "sid": track_name,
                    "orig_event_flag": True,
                    "data_provider": "STORM",
                    "id_no": idx[0] * 1000 + idx[1],
                    "category": group["category"].max(),
                },
            )
        )
    return TCTracks(data)
