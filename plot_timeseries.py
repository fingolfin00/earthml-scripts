from pathlib import Path
from typing import Literal

import warnings
from dask.array import PerformanceWarning

import numpy as np
import xarray as xr
from matplotlib.path import Path as MplPath

from earthml import (
    LeadtimeUnit,
    ClimPeriod,
    get_experiment_configs,
    get_and_subset_datasets,
    aggregate_leadtime_ds,
)
from earthml.metrics import (
    LeadtimeAgg,
    stack_hour_clim,
    groupby_period,
    calculate_save_and_subset_climatologies,
)
from earthml.plots import (
    safe_label,
    lead_label,
    plot_field_timeseries,
    VARIABLE_NAMES,
    VARIABLE_UNITS,
)
from locations import CITY_LOCATIONS, SEA_LOCATIONS

warnings.simplefilter("ignore", FutureWarning)
warnings.filterwarnings("ignore", category=PerformanceWarning)


# ==========================================================
# Paths
# ==========================================================

# Seasonal
EXP_NAME = "experiments"
# EXP_NAME = "experiment_new"

# Weather
# EXP_NAME = "weather_atmo"
# EXP_NAME = "weather_atmo_encoding"

EXPERIMENTS_ROOT = Path(
    f"/Users/jacopodallaglio/ML/training/seasonal/{EXP_NAME}"
    # f"/work/cmcc/jd19424/ML/MLBC/experiments/{EXP_NAME}"
)
OUTPUT_ROOT = Path(
    f"/Users/jacopodallaglio/ML/training/seasonal/{EXP_NAME}"
    # f"/work/cmcc/jd19424/ML/MLBC/plots/{EXP_NAME}"
)

# ==========================================================
# Plot generation / representation
# ==========================================================

CATEGORY: Literal["raw", "error", "absolute_error", "anomaly", "anomaly_error"] = "anomaly"
PLOT_MLFC = True
INCLUDE_CLIM_FC = True
REGENERATE_PLOTS = True

# ==========================================================
# Ensemble representation
# ==========================================================

PLOT_ENS_MEAN = True
PLOT_SINGLE_MEMBERS = True
OFFSET_PLOTS = False
SERIES_OFFSETS = {
    "Forecast": 0.0,
    "Clim-corrected forecast": 2.0,
    "Corrected forecast": 4.0,
    "Analysis": 6.0,
}

# ==========================================================
# Plot settings
# ==========================================================

PLOT_TITLE = True
PLOT_LABELS = True
TITLE_SIZE = None
LABEL_SIZE = None
TICK_SIZE = None
DPI = 300

# ==========================================================
# Rolling mean
# ==========================================================

# Number of samples, not days.
ROLLING_MEAN_WINDOW = None
# ROLLING_MEAN_WINDOW = 30
ROLLING_MEAN_CENTER = True
ROLLING_MEAN_MIN_PERIODS = 1

# ==========================================================
# Data processing
# ==========================================================

INTERPOLATE = True  # seasonal
# INTERPOLATE = False  # weather
BUILD_ANALYSIS = True
RECALCULATE_CLIMATOLOGY = False

# ==========================================================
# Climatology
# ==========================================================

CLIM_PERIOD: ClimPeriod = ClimPeriod.MONTH
CLIM_ROLLING_WINDOW = None
# CLIM_PERIOD: ClimPeriod = ClimPeriod.DAYOFYEAR_HOUR
# CLIM_ROLLING_WINDOW = 31

# ==========================================================
# Time selection
# ==========================================================

TIME_RANGE: tuple[str, str] | Literal["train", "val", "test", "train_val", "full"] = "test"
# TIME_RANGE = ("2025-01-01", "2025-10-31")

CLIM_TIME_RANGE: tuple[str, str] | Literal["train", "val", "test", "train_val", "full"] = "train"
INFERENCE_PERIOD: tuple[str, str] | None = None

# ==========================================================
# Lead-time aggregation and selection
# ==========================================================

LEADTIME_UNITS = LeadtimeUnit.MONTHS  # seasonal
# LEADTIME_UNITS = LeadtimeUnit.HOURS  # weather
LEADTIME_AGG_MODE: LeadtimeAgg = "single"
LEADTIMES = None  # all lead times
# LEADTIMES = [24, 48, 72]

# ==========================================================
# Experiment selection
# ==========================================================

VARIABLES = [
    # Atmosphere
    # "mslp",
    "t2m",
    # "d2m",
    # "u10",
    # "v10",
    # "sst",
    # "tprate",
    # "tcc",
    # Ocean
    # "mlotst",
    # "ssh",
    # "sss",
    # "t20d",
]

REGIONS = [
    # "ConUS",
    # "Europe",
    # "Pacific",
    "World",
    # None,
]

FILTERS = dict(
    var_fc=VARIABLES,
    region_name=REGIONS,
    net_name="SmaAt_UNet",
    # net_name="ConvNeXtTransformerUNet",
    # target_mode="anomaly_residual",
    # seasonal_encoding=True,
    # ensemble_encoding=True,
    # input_realization_avg=True,
    # extra_suffix_folder="random_split",
)

# ==========================================================
# Spatial selection
# ==========================================================

LAT_RANGE = None
LON_RANGE = None
# LAT_RANGE = (50, 25)
# LON_RANGE = (-130, -60)

# LOCATIONS = {
#     name: loc
#     for name, loc in (SEA_LOCATIONS | CITY_LOCATIONS).items()
#     if name in (
#         "newyork", "rome", "lagos", "gulfstream", "mediterranean", "nino34"
#     )
# }
# LOCATIONS = SEA_LOCATIONS
# LOCATIONS = CITY_LOCATIONS
LOCATIONS = {"all": None}


def resolve_time_range(s, selection):
    if isinstance(selection, tuple):
        return selection
    ranges = {
        "train": (s.train_start, s.train_end),
        "val": (s.val_start, s.val_end),
        "test": (s.test_start, s.test_end),
        "train_val": (s.train_start, s.val_end),
        "full": (s.train_start, s.test_end),
    }
    return ranges[selection]


def main() -> None:
    settings = get_experiment_configs(EXPERIMENTS_ROOT, **FILTERS)
    print(f"Found {len(settings)} matching experiment(s).")

    n = 0

    for s in settings:

        # ======================================================
        # Time range
        # ======================================================

        valid_time_range = resolve_time_range(s, TIME_RANGE)
        clim_time_range = resolve_time_range(s, CLIM_TIME_RANGE)

        mlfc_path = None
        if INFERENCE_PERIOD is not None:
            if TIME_RANGE == "full":
                valid_time_range = INFERENCE_PERIOD
            inference_start, inference_end = INFERENCE_PERIOD
            mlfc_path = (
                s.exp_dir
                / "inference"
                / f"{inference_start}_{inference_end}"
                / "test_corrected.zarr"
            )

        # ======================================================
        # Spatial range
        # ======================================================

        lat_lon = (
            list(s.region.values())
            if s.region is not None
            else [None, None]
        )

        valid_lat_range = (
            lat_lon[0]
            if LAT_RANGE is None
            else LAT_RANGE
        )

        valid_lon_range = (
            lat_lon[1]
            if LON_RANGE is None
            else LON_RANGE
        )

        # ======================================================
        # Lead-time dimension
        # ======================================================

        leadtime_agg_coord = (
            "leadtime"
            if LEADTIME_AGG_MODE == "single"
            else "leadtime_seasonal"
        )

        print(
            f"Generate {LEADTIME_AGG_MODE} {CATEGORY} timeseries "
            f"for {(s.var_an, s.var_fc)} in {s.region_name} "
            f"(lon={valid_lon_range}, lat={valid_lat_range})"
        )

        # ======================================================
        # Data
        # ======================================================

        fc, an, mlfc = get_and_subset_datasets(
            s,
            leadtime_units=LEADTIME_UNITS,
            lat_range=valid_lat_range,
            lon_range=valid_lon_range,
            time_range=valid_time_range,
            interpolate=INTERPOLATE,
            mlfc_path=mlfc_path,
        )

        if mlfc is not None:
            mlfc = mlfc.assign_coords(
                leadtime=s.leadtimes
            )

        if not PLOT_MLFC:
            mlfc = None

        # ======================================================
        # Climatologies
        # ======================================================

        fc_clim, an_clim, mlfc_clim = (
            calculate_save_and_subset_climatologies(
                s,
                leadtime_units=LEADTIME_UNITS,
                force=RECALCULATE_CLIMATOLOGY,
                clim_period=CLIM_PERIOD,
                rolling_window=CLIM_ROLLING_WINDOW,
                rolling_center=True,
                rolling_min_periods=1,
                lat_range=valid_lat_range,
                lon_range=valid_lon_range,
                time_range=clim_time_range,
                time_start=None,
                interpolate=INTERPOLATE,
                engine="zarr",
                build_analysis=BUILD_ANALYSIS,
                coord_rename_fc=None,
                coord_rename_an=None,
            )
        )

        if mlfc_clim is not None:
            mlfc_clim = mlfc_clim.assign_coords(
                leadtime=s.leadtimes
            )

        if not PLOT_MLFC:
            mlfc_clim = None

        # Restrict all datasets to the requested lead times.
        if LEADTIMES is not None:
            missing = set(LEADTIMES) - set(s.leadtimes)
            if missing:
                raise ValueError(
                    f"Unavailable leadtimes {sorted(missing)}; "
                    f"available: {list(s.leadtimes)}"
                )
            fc = fc.sel({fc.earthml.guessed_dims.leadtime: LEADTIMES})
            an = an.sel({an.earthml.guessed_dims.leadtime: LEADTIMES})
            fc_clim = fc_clim.sel({fc_clim.earthml.guessed_dims.leadtime: LEADTIMES})
            an_clim = an_clim.sel({an_clim.earthml.guessed_dims.leadtime: LEADTIMES})
            if mlfc is not None:
                mlfc = mlfc.sel({mlfc.earthml.guessed_dims.leadtime: LEADTIMES})
            if mlfc_clim is not None:
                mlfc_clim = mlfc_clim.sel({mlfc_clim.earthml.guessed_dims.leadtime: LEADTIMES})

        # ======================================================
        # Lead-time aggregation
        # ======================================================

        if (
            LEADTIME_AGG_MODE != "single"
            and s.seasonal_leadtime_windows is not None
        ):
            an = aggregate_leadtime_ds(
                ds=an,
                windows=s.seasonal_leadtime_windows,
                leadtime_dim=an.earthml.guessed_dims.leadtime,
                leadtime_agg_coord=leadtime_agg_coord,
            )

            fc = aggregate_leadtime_ds(
                ds=fc,
                windows=s.seasonal_leadtime_windows,
                leadtime_dim=fc.earthml.guessed_dims.leadtime,
                leadtime_agg_coord=leadtime_agg_coord,
            )

            if mlfc is not None:
                mlfc = aggregate_leadtime_ds(
                    ds=mlfc,
                    windows=s.seasonal_leadtime_windows,
                    leadtime_dim=mlfc.earthml.guessed_dims.leadtime,
                    leadtime_agg_coord=leadtime_agg_coord,
                )

            if an_clim is not None:
                an_clim = aggregate_leadtime_ds(
                    ds=an_clim,
                    windows=s.seasonal_leadtime_windows,
                    leadtime_dim=an_clim.earthml.guessed_dims.leadtime,
                    leadtime_agg_coord=leadtime_agg_coord,
                )

            if fc_clim is not None:
                fc_clim = aggregate_leadtime_ds(
                    ds=fc_clim,
                    windows=s.seasonal_leadtime_windows,
                    leadtime_dim=fc_clim.earthml.guessed_dims.leadtime,
                    leadtime_agg_coord=leadtime_agg_coord,
                )

            if mlfc_clim is not None:
                mlfc_clim = aggregate_leadtime_ds(
                    ds=mlfc_clim,
                    windows=s.seasonal_leadtime_windows,
                    leadtime_dim=mlfc_clim.earthml.guessed_dims.leadtime,
                    leadtime_agg_coord=leadtime_agg_coord,
                )

        # ======================================================
        # Unit conversion
        # ======================================================

        fc[s.var_fc] = _convert_kelvin_to_celsius(
            fc[s.var_fc],
            var=s.var_fc,
        )

        an[s.var_an] = _convert_kelvin_to_celsius(
            an[s.var_an],
            var=s.var_an,
        )

        if mlfc is not None:
            mlfc[s.var_fc] = _convert_kelvin_to_celsius(
                mlfc[s.var_fc],
                var=s.var_fc,
            )

        fc_clim[s.var_fc] = _convert_kelvin_to_celsius(
            fc_clim[s.var_fc],
            var=s.var_fc,
        )

        an_clim[s.var_an] = _convert_kelvin_to_celsius(
            an_clim[s.var_an],
            var=s.var_an,
        )

        if mlfc_clim is not None:
            mlfc_clim[s.var_fc] = _convert_kelvin_to_celsius(
                mlfc_clim[s.var_fc],
                var=s.var_fc,
            )

        # ======================================================
        # Dimensions
        # ======================================================

        time_dim = fc.earthml.guessed_dims.time
        lat_dim = fc.earthml.guessed_dims.latitude
        lon_dim = fc.earthml.guessed_dims.longitude
        realization_dim = fc.earthml.guessed_dims.realization

        # ======================================================
        # Fields and climatologies
        # ======================================================

        fc_da = fc[s.var_fc]
        an_da = an[s.var_an]

        fc_clim_da = fc_clim[s.var_fc]
        an_clim_da = an_clim[s.var_an]

        mlfc_da = (
            mlfc[s.var_fc]
            if mlfc is not None
            else None
        )

        mlfc_clim_da = (
            mlfc_clim[s.var_fc]
            if mlfc_clim is not None
            else None
        )

        fc_da, fc_clim_da = xr.unify_chunks(
            fc_da,
            fc_clim_da,
        )

        an_da, an_clim_da = xr.unify_chunks(
            an_da,
            an_clim_da,
        )

        if (
            mlfc_da is not None
            and mlfc_clim_da is not None
        ):
            mlfc_da, mlfc_clim_da = xr.unify_chunks(
                mlfc_da,
                mlfc_clim_da,
            )

        fc_clim_da = stack_hour_clim(
            fc_clim_da,
            CLIM_PERIOD,
        )

        an_clim_da = stack_hour_clim(
            an_clim_da,
            CLIM_PERIOD,
        )

        mlfc_clim_da = (
            stack_hour_clim(
                mlfc_clim_da,
                CLIM_PERIOD,
            )
            if mlfc_clim_da is not None
            else None
        )

        # ======================================================
        # Anomalies
        # ======================================================

        fc_anom_da = (
            groupby_period(
                fc_da,
                time_dim,
                CLIM_PERIOD,
            )
            - fc_clim_da
        )

        an_anom_da = (
            groupby_period(
                an_da,
                time_dim,
                CLIM_PERIOD,
            )
            - an_clim_da
        )

        mlfc_anom_da = (
            groupby_period(
                mlfc_da,
                time_dim,
                CLIM_PERIOD,
            )
            - mlfc_clim_da
            if (
                mlfc_da is not None
                and mlfc_clim_da is not None
            )
            else None
        )

        # ======================================================
        # Climatology-corrected forecast
        # ======================================================

        # Analysis climatology expanded onto the full time axis.
        an_clim_for_time_da = (
            an_da - an_anom_da
        )

        # Classical grid-point climatological bias correction:
        #
        #     fc_corrected = fc - fc_clim + an_clim
        #
        fc_clim_corrected_da = (
            fc_anom_da
            + an_clim_for_time_da
        )

        # ======================================================
        # Timeseries category
        # ======================================================

        if CATEGORY == "anomaly":
            fc_ts_da = fc_anom_da
            fc_ts_clim_corrected_da = fc_anom_da
            an_ts_da = an_anom_da
            mlfc_ts_da = mlfc_anom_da
            category_title = " anomaly "

        elif CATEGORY == "anomaly_error":
            fc_ts_da = (
                fc_anom_da - an_anom_da
            )

            # A climatology-only correction changes the mean
            # climatological component, not the forecast anomaly.
            fc_ts_clim_corrected_da = (
                fc_anom_da - an_anom_da
            )

            mlfc_ts_da = (
                mlfc_anom_da - an_anom_da
                if mlfc_anom_da is not None
                else None
            )

            an_ts_da = None
            category_title = " anomaly error "

        elif CATEGORY == "error":
            fc_ts_da = (
                fc_da - an_da
            )

            fc_ts_clim_corrected_da = (
                fc_clim_corrected_da - an_da
            )

            mlfc_ts_da = (
                mlfc_da - an_da
                if mlfc_da is not None
                else None
            )

            an_ts_da = None
            category_title = " error "

        elif CATEGORY == "absolute_error":
            fc_ts_da = abs(
                fc_da - an_da
            )

            fc_ts_clim_corrected_da = abs(
                fc_clim_corrected_da - an_da
            )

            mlfc_ts_da = (
                abs(mlfc_da - an_da)
                if mlfc_da is not None
                else None
            )

            an_ts_da = None
            category_title = " absolute error "

        else:
            fc_ts_da = fc_da
            fc_ts_clim_corrected_da = (
                fc_clim_corrected_da
            )
            an_ts_da = an_da
            mlfc_ts_da = mlfc_da
            category_title = ""

        for location, location_config in LOCATIONS.items():

            fc_ts_loc_da = subset_timeseries_region(
                fc_ts_da,
                location_config,
            )

            fc_ts_clim_corrected_loc_da = subset_timeseries_region(
                fc_ts_clim_corrected_da,
                location_config,
            )

            an_ts_loc_da = subset_timeseries_region(
                an_ts_da,
                location_config,
            )

            mlfc_ts_loc_da = subset_timeseries_region(
                mlfc_ts_da,
                location_config,
            )

            # ======================================================
            # Lead-time plots
            # ======================================================

            for lt in fc[leadtime_agg_coord].values:

                # --------------------------------------------------
                # Forecast
                # --------------------------------------------------

                fc_ts_lead_da = fc_ts_loc_da.sel(
                    {leadtime_agg_coord: lt}
                )

                fc_ts_lead_da = apply_rolling_mean(
                    fc_ts_lead_da,
                    time_dim=time_dim,
                    window=ROLLING_MEAN_WINDOW,
                    center=ROLLING_MEAN_CENTER,
                    min_periods=ROLLING_MEAN_MIN_PERIODS,
                )

                if (
                    realization_dim is not None
                    and realization_dim in fc_ts_lead_da.dims
                ):
                    fc_ts_lead_da_ens_mean = (
                        fc_ts_lead_da.mean(
                            realization_dim,
                            skipna=True,
                        )
                        if PLOT_ENS_MEAN
                        else None
                    )

                else:
                    fc_ts_lead_da_ens_mean = (
                        fc_ts_lead_da
                    )

                # --------------------------------------------------
                # Analysis
                # --------------------------------------------------

                an_ts_lead_da = (
                    an_ts_loc_da.sel(
                        {leadtime_agg_coord: lt}
                    )
                    if an_ts_loc_da is not None
                    else None
                )

                an_ts_lead_da = apply_rolling_mean(
                    an_ts_lead_da,
                    time_dim=time_dim,
                    window=ROLLING_MEAN_WINDOW,
                    center=ROLLING_MEAN_CENTER,
                    min_periods=ROLLING_MEAN_MIN_PERIODS,
                )

                # --------------------------------------------------
                # ML-corrected forecast
                # --------------------------------------------------

                mlfc_ts_lead_da = (
                    mlfc_ts_loc_da.sel(
                        {leadtime_agg_coord: lt}
                    )
                    if mlfc_ts_loc_da is not None
                    else None
                )

                mlfc_ts_lead_da = apply_rolling_mean(
                    mlfc_ts_lead_da,
                    time_dim=time_dim,
                    window=ROLLING_MEAN_WINDOW,
                    center=ROLLING_MEAN_CENTER,
                    min_periods=ROLLING_MEAN_MIN_PERIODS,
                )

                if mlfc_ts_lead_da is not None:
                    if (
                        realization_dim is not None
                        and realization_dim
                        in mlfc_ts_lead_da.dims
                    ):
                        mlfc_ts_lead_da_ens_mean = (
                            mlfc_ts_lead_da.mean(
                                realization_dim,
                                skipna=True,
                            )
                            if PLOT_ENS_MEAN
                            else None
                        )

                    else:
                        mlfc_ts_lead_da_ens_mean = (
                            mlfc_ts_lead_da
                        )

                else:
                    mlfc_ts_lead_da_ens_mean = None

                # --------------------------------------------------
                # Climatology-corrected forecast
                # --------------------------------------------------

                fc_ts_lead_clim_corrected_da = (
                    fc_ts_clim_corrected_loc_da.sel(
                        {leadtime_agg_coord: lt}
                    )
                )

                fc_ts_lead_clim_corrected_da = (
                    apply_rolling_mean(
                        fc_ts_lead_clim_corrected_da,
                        time_dim=time_dim,
                        window=ROLLING_MEAN_WINDOW,
                        center=ROLLING_MEAN_CENTER,
                        min_periods=ROLLING_MEAN_MIN_PERIODS,
                    )
                )

                if (
                    realization_dim is not None
                    and realization_dim
                    in fc_ts_lead_clim_corrected_da.dims
                ):
                    fc_ts_lead_clim_corrected_da_ens_mean = (
                        fc_ts_lead_clim_corrected_da.mean(
                            realization_dim,
                            skipna=True,
                        )
                        if PLOT_ENS_MEAN
                        else None
                    )

                else:
                    fc_ts_lead_clim_corrected_da_ens_mean = (
                        fc_ts_lead_clim_corrected_da
                    )

                # ==================================================
                # Series to plot
                # ==================================================

                series = {
                    "Forecast": fc_ts_lead_da_ens_mean,
                }

                member_series = {
                    "Forecast": fc_ts_lead_da,
                }

                if mlfc_ts_lead_da is not None:
                    series[
                        "Corrected forecast"
                    ] = mlfc_ts_lead_da_ens_mean

                    member_series[
                        "Corrected forecast"
                    ] = mlfc_ts_lead_da

                if an_ts_lead_da is not None:
                    series["Analysis"] = (
                        an_ts_lead_da
                    )

                if (
                    INCLUDE_CLIM_FC
                    and CATEGORY in (
                        "raw",
                        "error",
                        "absolute_error",
                    )
                ):
                    series[
                        "Clim-corrected forecast"
                    ] = (
                        fc_ts_lead_clim_corrected_da_ens_mean
                    )

                    member_series[
                        "Clim-corrected forecast"
                    ] = (
                        fc_ts_lead_clim_corrected_da
                    )

                # ==================================================
                # Output
                # ==================================================

                label = safe_label(
                    lead_label(
                        fc_ts_lead_da,
                        lt,
                        leadtime_agg_coord,
                    )
                )

                rolling_label = (
                    f" roll {ROLLING_MEAN_WINDOW} "
                    if ROLLING_MEAN_WINDOW is not None
                    else ""
                )

                rolling_filename = (
                    f"_roll{ROLLING_MEAN_WINDOW}"
                    if ROLLING_MEAN_WINDOW is not None
                    else ""
                )

                out_file = (
                    OUTPUT_ROOT / s.output_name / "plots"
                    / "timeseries"
                    / "fields"
                    / CATEGORY
                    / (
                        f"time_{safe_label(valid_time_range)}"
                        f"_loc_{location}"
                    )
                    / LEADTIME_AGG_MODE
                    / (
                        f"{s.var_fc}_lead_{label}_"
                        f"{CATEGORY}{rolling_filename}_"
                        f"timeseries.png"
                    )
                )

                if (
                    out_file.exists()
                    and not REGENERATE_PLOTS
                ):
                    continue

                title = (
                    f"{VARIABLE_NAMES.get(s.var_fc, s.var_fc.upper())}"
                    f"{category_title}"
                    f"{rolling_label}"
                    f"· lead={label}"
                    if PLOT_TITLE
                    else ""
                )

                plot_field_timeseries(
                    series=series,
                    member_series=member_series,
                    var=s.var_fc,
                    title=title,
                    out_file=out_file,
                    time_dim=time_dim,
                    spatial_dims=(
                        lat_dim,
                        lon_dim,
                    ),
                    realization_dim=realization_dim,
                    train_end=s.train_end if valid_time_range[0] <= s.train_end <= valid_time_range[1] else None,
                    val_end=s.val_end if valid_time_range[0] <= s.val_end <= valid_time_range[1] else None,
                    plot_single_members=(
                        PLOT_SINGLE_MEMBERS
                    ),
                    member_linestyle="-",
                    series_linestyle=(
                        "--"
                        if PLOT_ENS_MEAN
                        else "-"
                    ),
                    plot_title=PLOT_TITLE,
                    plot_labels=PLOT_LABELS,
                    title_size=TITLE_SIZE,
                    label_size=LABEL_SIZE,
                    tick_size=TICK_SIZE,
                    dpi=DPI,
                    series_offsets=(
                        SERIES_OFFSETS
                        if OFFSET_PLOTS
                        else None
                    ),
                )

                n += 1

    print(f"Done. Saved {n} plots.")


def apply_rolling_mean(
    da: xr.DataArray | None,
    *,
    time_dim: str,
    window: int | None,
    center: bool,
    min_periods: int,
) -> xr.DataArray | None:
    if da is None or window is None:
        return da

    return da.rolling(
        {time_dim: window},
        center=center,
        min_periods=min_periods,
    ).mean(skipna=True)


def subset_timeseries_region(
    da: xr.DataArray | None,
    region: dict | None,
) -> xr.DataArray | None:
    if da is None or region is None:
        return da

    lat_dim = da.earthml.guessed_dims.latitude
    lon_dim = da.earthml.guessed_dims.longitude

    # Normalize longitudes to [-180, 180].
    if float(da[lon_dim].max()) > 180:
        lon = ((da[lon_dim] + 180) % 360) - 180
        da = da.assign_coords({lon_dim: lon}).sortby(lon_dim)

    region_type = region.get("type", "rectangle")

    # ==========================================================
    # Rectangle
    # ==========================================================

    if region_type == "rectangle":
        lat0, lat1 = region["lat_range"]
        lon0, lon1 = region["lon_range"]

        # Latitude ordering.
        lat_ascending = (
            float(da[lat_dim].values[0])
            < float(da[lat_dim].values[-1])
        )

        if lat_ascending:
            lat0, lat1 = sorted((lat0, lat1))
        else:
            lat0, lat1 = sorted(
                (lat0, lat1),
                reverse=True,
            )

        da = da.sel({
            lat_dim: slice(lat0, lat1),
        })

        # Longitude bounds -> [-180, 180].
        lon0 = ((lon0 + 180) % 360) - 180
        lon1 = ((lon1 + 180) % 360) - 180

        if lon0 <= lon1:
            da = da.sel({
                lon_dim: slice(lon0, lon1),
            })

        else:
            # Dateline-crossing rectangle.
            da = da.where(
                (da[lon_dim] >= lon0)
                | (da[lon_dim] <= lon1),
                drop=True,
            )

        return da

    # ==========================================================
    # Polygon
    # ==========================================================

    if region_type == "polygon":
        coordinates = np.asarray(
            region["coordinates"],
            dtype=float,
        )

        # Normalize polygon longitudes too.
        coordinates[:, 0] = (
            (coordinates[:, 0] + 180) % 360
        ) - 180

        lon2d, lat2d = np.meshgrid(
            da[lon_dim].values,
            da[lat_dim].values,
        )

        points = np.column_stack([
            lon2d.ravel(),
            lat2d.ravel(),
        ])

        polygon = MplPath(coordinates)

        mask = polygon.contains_points(
            points
        ).reshape(lat2d.shape)

        mask_da = xr.DataArray(
            mask,
            coords={
                lat_dim: da[lat_dim],
                lon_dim: da[lon_dim],
            },
            dims=(lat_dim, lon_dim),
        )

        return da.where(mask_da)

    raise ValueError(
        f"Unsupported region type {region_type!r}. "
        "Choose 'rectangle' or 'polygon'."
    )


def _convert_kelvin_to_celsius(
    da: xr.DataArray,
    *,
    var: str,
) -> xr.DataArray:
    unit = da.attrs.get("units") or VARIABLE_UNITS.get(var, "")

    if unit not in {"K", "Kelvin", "kelvin"}:
        return da

    da = da - 273.15
    da.attrs["units"] = "°C"

    return da


if __name__ == "__main__":
    main()
