from pathlib import Path

import xarray as xr

from rich import print

import warnings
from dask.array import PerformanceWarning
warnings.simplefilter("ignore", FutureWarning)
warnings.filterwarnings(
    "ignore",
    category=PerformanceWarning,
)


import earthml
from earthml import (
    Settings,
    get_experiment_configs,
)
from earthml.metrics import calculate_save_and_subset_climatologies


def main() -> None:
    # Get configs

    experiments_root = Path("/work/cmcc/jd19424/ML/MLBC/experiments/weather_atmo")

    variables = [
        "t2m",
        "mslp",
        "d2m",
        "u10",
        "v10",
        "tcc",
        # "sst",
        # "tprate",
    ]
    regions = ["ConUS",] # World, ConUS, Europe, Pacific or None to accept all # TOD rerun europe and pacific

    settings = get_experiment_configs(experiments_root, variables, regions)

    # Selection settings

    # ConUS
    # lat_range = (50, 25)
    # lon_range = (-130, -60)
    # Europe
    # lat_range = (80, 30)
    # lon_range = (-30, 60)
    # Pacific
    # lat_range = (20, -20)
    # lon_range = (-195, -135)
    # World or whole region
    lat_range = None
    lon_range = None

    leadtime_units = "hours"
    clim_period = "dayofyear_hour" # "dayofyear", "day", "month", "year", "day_hour", "dayofyear_hour", "month_hour"
    clim_rolling_window = 31

    time_range = None
    # time_range = ("2018-01-01", "2022-12-31")

    interpolate = False
    build_analysis = True
    force_clim_recalc = True

    print(f"Found {len(settings)} matching experiment(s).")

    for i, s in enumerate(settings, start=1):
        clim_time_range = (s.train_start, s.train_end)
        lat_lon = list(s.region.values()) if s.region is not None else [None, None]
        valid_lat_range = lat_lon[0] if lat_range is None else lat_range
        valid_lon_range = lat_lon[1] if lon_range is None else lon_range

        print(
            f"[{i}/{len(settings)}] Generating {clim_period} climatology for "
            f"{s.var_an}/{s.var_fc} in {s.region_name} "
            f"(lon={valid_lon_range}, lat={valid_lat_range})\n"
            f"Period: {clim_time_range[0]} - {clim_time_range[1]} (rolling={clim_rolling_window})\n"
            f"Experiment: {s.output_name}"
        )

        fc_clim, an_clim, mlfc_clim = calculate_save_and_subset_climatologies(
            s,
            leadtime_units=leadtime_units,
            force=force_clim_recalc,
            clim_period=clim_period,
            rolling_window=clim_rolling_window,
            rolling_center=True,
            rolling_min_periods=1,
            lat_range=valid_lat_range,
            lon_range=valid_lon_range,
            time_range=clim_time_range,
            time_start=None,
            interpolate=interpolate,
            engine="zarr",
            build_analysis=build_analysis,
            coord_rename_fc=None,
            coord_rename_an=None,
        )
        mlfc_clim = mlfc_clim.assign_coords(leadtime=s.leadtimes) if mlfc_clim is not None else None

        print(fc_clim)



if __name__ == "__main__":
    main()
