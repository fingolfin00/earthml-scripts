from typing import Literal
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from matplotlib.path import Path as MplPath

import warnings
from dask.array import PerformanceWarning
warnings.simplefilter("ignore", FutureWarning)
warnings.filterwarnings(
    "ignore",
    category=PerformanceWarning,
)

import earthml
from earthml import (
    LeadtimeUnit,
    ClimPeriod,
    get_experiment_configs,
    get_and_subset_datasets,
)
from earthml.metrics import (
    LeadtimeAgg,
    is_deterministic,
    is_probabilistic,
    get_metrics,
    calculate_save_and_subset_climatologies,
    stack_hour_clim,
    groupby_period,
    build_metric_improvements,
)
from earthml.plots import (
    safe_label,
    lead_label,
    plot_timeseries,
)
from locations import CITY_LOCATIONS, SEA_LOCATIONS


# ==========================================================
# Paths
# ==========================================================

EXP_NAME = "weather_atmo"
# EXP_NAME = "weather_atmo_encoding_short"
# EXP_NAME = "weather_atmo_encoding"
# EXP_NAME = "weather_atmo_seed"
# EXP_NAME = "weather_atmo_normalization"
# EXP_NAME = "weather_atmo_variables"
# EXP_NAME = "weather_atmo_normalization_ablation"
# EXP_NAME = "weather_atmo_sample_independent"
# EXP_NAME = "weather_atmo_one_year_test"
# EXP_NAME = "weather_atmo_ablation_fixed_val"
# EXP_NAME = "weather_atmo_short_zero_vs_replicate_padding"

EXPERIMENTS_ROOT = Path(
    "/Users/jacopodallaglio/ML/training/seasonal/experiments"
    # f"/work/cmcc/jd19424/ML/MLBC/experiments/{EXP_NAME}"
)

OUTPUT_ROOT = Path(
    "/Users/jacopodallaglio/ML/training/seasonal/plots"
    # f"/work/cmcc/jd19424/ML/MLBC/plots/{EXP_NAME}"
)

OROGRAPHY_PATH = Path(
    "/Users/jacopodallaglio/ML/training/seasonal/data/input/era5_orography.zarr"
    # "/work/cmcc/jd19424/ML/MLBC/data/orography/era5_orography.zarr"
)

# ==========================================================
# Plot generation
# ==========================================================

PLOT_ABSOLUTE_TIMESERIES = True
PLOT_IMPROVEMENT_TIMESERIES = True

# ==========================================================
# Global flag
# ==========================================================

INTERPOLATE = True # seasonal
# INTERPOLATE = False # weather

BUILD_ANALYSIS = True

RECALCULATE_CLIMATOLOGY = False
REGENERATE_PLOTS = True

# ==========================================================
# Plot settings
# ==========================================================

PLOT_TITLE = True
PLOT_LABELS = True
PLOT_LEGEND = True

TIMESERIES_ZERO_LINE = True
TIMESERIES_MEAN_LINE = True

TITLE_SIZE = None
LABEL_SIZE = None
TICK_SIZE = None
DPI = 300

# ==========================================================
# Absolute timeseries configuration
# ==========================================================

PLOT_MODELS = (
    "fc",
    "mlfc",
)

METRIC_LIMITS = None

# ==========================================================
# Improvement timeseries configuration
# ==========================================================

METRIC_IMPROVEMENT_LIMITS = {
    "Δ": (-0.5, 1.2),
    # "Δ": None,
    "%": None,
    "normalized": None,
}

MODEL_COMPARISONS = (
    ("fc", "mlfc"),
)
IMPROVEMENT_UNITS = ("%", "Δ")
TIMESERIES_IMPROVED_PERCENTAGE = True

# ==========================================================
# Rolling mean
# ==========================================================

# Number of metric samples, not number of days.
ROLLING_MEAN_WINDOW = None
# rolling_mean_window = 30
ROLLING_MEAN_CENTER = True
ROLLING_MEAN_MIN_PERIODS = 1

# ==========================================================
# Timeseries extrema
# ==========================================================

EXTREMA_PERIODS: tuple[
    Literal["train", "val", "test", "full"],
    ...
] = (
    "test",
)

EXTREMA_N_MIN = 20
EXTREMA_N_MAX = 20

EXTREMA_ANNOTATE = False
EXTREMA_PRINT = True

# ==========================================================
# Timeseries annotations
# ==========================================================

ANNOTATE_TIMES: tuple[str, ...] = (
    # "2025-08-17 12:00", # W4 mslp 72h
    # "2025-08-18 12:00", # B1 mslp 72h
)

# ==========================================================
# Climatology
# ==========================================================

CLIM_PERIOD: ClimPeriod = ClimPeriod.MONTH
CLIM_ROLLING_WINDOW = None

PERIOD_REFERENCE: Literal["init", "valid"] = "init"

# ==========================================================
# Time selection
# ==========================================================

TIME_RANGE: tuple[str, str] | Literal[
    "train", "val", "test", "train_val", "full"
] = (
    "test"
    # ("2025-02-01", "2025-02-28")
)

CLIM_TIME_RANGE: tuple[str, str] | Literal[
    "train", "val", "test", "train_val", "full"
] = (
    "train" # seasonal
    # "train_val" # weather
)

INFERENCE_PERIOD: tuple[str, str] | None = None

PERIODS_REQUESTED = ["all"]

# ==========================================================
# Lead-time aggregation and selection
# ==========================================================

LEADTIME_UNITS = LeadtimeUnit.MONTHS
# LEADTIME_UNITS = LeadtimeUnit.HOURS

LEADTIME_AGG_MODE: LeadtimeAgg = "single"

LEADTIMES = None # use all experiment leadtimes
# LEADTIMES = [72]
# LEADTIMES = [24, 48, 72]

# ==========================================================
# Metrics
# ==========================================================

METRICS = [
    # ------------------------------------------------------
    # Deterministic absolute fields
    # ------------------------------------------------------
    "bias",
    "rmse",
    "scc",
    # "mae",
    # "mse",
    # "nrmse",
    # "r2",

    # ------------------------------------------------------
    # Gradients
    # ------------------------------------------------------
    # "fc_grad_mag",
    # "an_grad_mag",
    # "grad_rmse",

    # ------------------------------------------------------
    # Deterministic anomaly fields
    # ------------------------------------------------------
    # "bias_anom",
    # "rmse_anom",
    # "scc_anom",
    # "mae_anom",
    # "mse_anom",
    # "nrmse_anom",
    # "r2_anom",

    # ------------------------------------------------------
    # Anomaly gradients
    # ------------------------------------------------------
    # "fc_anom_grad_mag",
    # "an_anom_grad_mag",
    # "grad_rmse_anom",

    # ------------------------------------------------------
    # Skill vs climatology
    # ------------------------------------------------------
    # "mse_skill_clim",
    # "rmse_skill_clim",
    # "mae_skill_clim",

    # "mse_anom_skill_clim",
    # "rmse_anom_skill_clim",
    # "mae_anom_skill_clim",

    # ------------------------------------------------------
    # Ensemble / probabilistic
    # ------------------------------------------------------
    # "ens_member_rmse",
    # "mean_member_rmse",
    # "spread",
    # "spread_skill_ratio",
    # "crps",

    # ------------------------------------------------------
    # Brier terciles
    # ------------------------------------------------------
    # "brier_lower",
    # "brier_middle",
    # "brier_upper",

    # ------------------------------------------------------
    # Ensemble / probabilistic anomalies
    # ------------------------------------------------------
    # "ens_member_rmse_anom",
    # "mean_member_rmse_anom",
    # "spread_anom",
    # "spread_anom_skill_ratio",
    # "crps_anom",

    # ------------------------------------------------------
    # Anomaly Brier terciles
    # ------------------------------------------------------
    # "brier_anom_lower",
    # "brier_anom_middle",
    # "brier_anom_upper",

    # ------------------------------------------------------
    # Ensemble skill vs climatology
    # ------------------------------------------------------
    # "ens_member_mse_skill_clim",
    # "mean_member_mse_skill_clim",
    # "ens_member_mse_anom_skill_clim",
    # "mean_member_mse_anom_skill_clim",
]

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
    "ConUS",
    # "Europe",
    # "Pacific",
    # "World",
    # None,
]

FILTERS = dict(
    var_fc=VARIABLES,
    region_name=REGIONS,
    # net_name="SmaAt_UNet",
    # target_mode="residual",
    loss_name="GeoMaskedMSELoss",
    # train_start="2019-10-14",
    # train_subsamples=1000,
    # train_subsamples=None,
    # normalization_input_mean="sample",
    # normalization_input_std="sample",
    # normalization_target_mean="sample",
    # normalization_target_std="sample",
    # sample_target_mean="input",
    # sample_target_std="input",
    # extra_suffix_folder="",
)

# ==========================================================
# Spatial subset
# ==========================================================

# LOCATIONS = SEA_LOCATIONS
# LOCATIONS = CITY_LOCATIONS
LOCATIONS = {"all": None}

# LOCATIONS = {
#     "Central ConUS": {
#         "type": "rectangle",
#         "lat_range": (30, 45),
#         "lon_range": (-108, -90),
#     },
# }


def main() -> None:
    if not (PLOT_ABSOLUTE_TIMESERIES or PLOT_IMPROVEMENT_TIMESERIES):
        raise ValueError("No timeseries plot types enabled")

    metrics_to_compute = list(METRICS)

    # ==========================================================
    # Experiment selection
    # ==========================================================

    settings = get_experiment_configs(
        EXPERIMENTS_ROOT,
        **FILTERS,
    )

    # Skip unfinished experiments
    split_paths = {
        "train": "output_mlfc_train",
        "val": "output_mlfc_val",
        "test": "output_mlfc_test",
    }

    available_settings = []

    for s in settings:
        if not getattr(s, split_paths[TIME_RANGE]).exists():
            print(
                f"Skipping unfinished experiment: {s.output_name} "
                f"(missing MLFC: {', '.join(TIME_RANGE)})"
            )
            continue

        available_settings.append(s)

    settings = available_settings

    if not settings:
        print("No completed experiments available")
        return

    print(f"Found {len(settings)} matching experiment(s)")

    # ==========================================================
    # Evaluation timestamps for each experiment
    # ==========================================================

    # Temporal encoding can shorten MLFC outputs
    if TIME_RANGE in ("train", "val", "test"):
        output_splits = (TIME_RANGE,)
    else:
        # For composite or explicit date ranges, gather all available
        # corrected-output splits, then filter by each experiment's range
        output_splits = ("train", "val", "test")

    available_settings = []
    evaluation_times_by_experiment = {}

    for s in settings:
        if TIME_RANGE == "train":
            requested_range = (s.train_start, s.train_end)
        elif TIME_RANGE == "val":
            requested_range = (s.val_start, s.val_end)
        elif TIME_RANGE == "test":
            requested_range = (s.test_start, s.test_end)
        elif TIME_RANGE == "train_val":
            requested_range = (s.train_start, s.val_end)
        elif TIME_RANGE == "full":
            requested_range = (s.train_start, s.test_end)
        elif isinstance(TIME_RANGE, tuple):
            requested_range = TIME_RANGE
        else:
            raise ValueError(f"Invalid TIME_RANGE={TIME_RANGE!r}")

        if INFERENCE_PERIOD is not None:
            start, end = INFERENCE_PERIOD

            paths = [
                s.exp_dir / "inference" / f"{start}_{end}" / "test_corrected.zarr"
            ]

            # Same requested-window behavior as the data-loading branch
            requested_start, requested_end = map(pd.Timestamp, requested_range)
            inference_start, inference_end = map(pd.Timestamp, INFERENCE_PERIOD)

            if not (inference_start <= requested_start and requested_end <= inference_end):
                requested_range = INFERENCE_PERIOD
        else:
            paths = [getattr(s, f"output_mlfc_{split}") for split in output_splits]

        missing_paths = [path for path in paths if not path.exists()]
        if missing_paths:
            print(
                f"Skipping incomplete experiment {s.exp_dir.name}: "
                f"missing {', '.join(path.name for path in missing_paths)}"
            )
            continue

        available_times = None
        for path in paths:
            with xr.open_zarr(path, consolidated=False) as ds:
                split_times = pd.DatetimeIndex(ds["time"].values)

            if not split_times.is_unique:
                raise ValueError(f"Duplicate timestamps in {path}")

            available_times = (
                split_times
                if available_times is None
                else available_times.union(split_times)
            )

        requested_start = pd.Timestamp(requested_range[0])
        requested_end = pd.Timestamp(requested_range[1])

        if isinstance(requested_range[1], str) and len(requested_range[1]) == 10:
            requested_end += pd.Timedelta(days=1)
            in_range = (
                (available_times >= requested_start)
                & (available_times < requested_end)
            )
        else:
            in_range = (
                (available_times >= requested_start)
                & (available_times <= requested_end)
            )

        available_times = available_times[in_range]

        if len(available_times) == 0:
            print(f"Skipping experiment {s.exp_dir.name}: no corrected timestamps in requested range")
            continue

        evaluation_times_by_experiment[s.exp_dir.name] = available_times.sort_values()
        available_settings.append(s)

    settings = available_settings
    if not settings:
        raise ValueError("No experiments with corrected-forecast timestamps in the requested range")

    print(f"Processing {len(settings)} experiment(s) with per-experiment timestamps")

    n = 0

    for s in settings:
        # Time range
        if TIME_RANGE == "train":
            valid_time_range = (s.train_start, s.train_end)
        elif TIME_RANGE == "val":
            valid_time_range = (s.val_start, s.val_end)
        elif TIME_RANGE == "test":
            valid_time_range = (s.test_start, s.test_end)
        elif TIME_RANGE == "train_val":
            valid_time_range = (s.train_start, s.val_end)
        elif TIME_RANGE == "full":
            valid_time_range = (s.train_start, s.test_end)
        elif isinstance(TIME_RANGE, tuple):
            valid_time_range = TIME_RANGE
        else:
            raise ValueError(f"Invalid time_range={TIME_RANGE}")

        mlfc_path = None

        if INFERENCE_PERIOD is not None:
            valid_start, valid_end = map(
                pd.Timestamp,
                valid_time_range,
            )
            inference_start, inference_end = map(
                pd.Timestamp,
                INFERENCE_PERIOD,
            )

            if (
                inference_start <= valid_start
                and valid_end <= inference_end
            ):
                valid_time_range = valid_time_range
            else:
                valid_time_range = INFERENCE_PERIOD

            inference_start_str, inference_end_str = INFERENCE_PERIOD

            mlfc_path = (
                s.exp_dir
                / "inference"
                / f"{inference_start_str}_{inference_end_str}"
                / "test_corrected.zarr"
            )

        # Climatology time range
        if CLIM_TIME_RANGE == "train":
            valid_clim_time_range = (s.train_start, s.train_end)
        elif CLIM_TIME_RANGE == "val":
            valid_clim_time_range = (s.val_start, s.val_end)
        elif CLIM_TIME_RANGE == "test":
            valid_clim_time_range = (s.test_start, s.test_end)
        elif CLIM_TIME_RANGE == "train_val":
            valid_clim_time_range = (s.train_start, s.val_end)
        elif CLIM_TIME_RANGE == "full":
            valid_clim_time_range = (s.train_start, s.test_end)
        elif isinstance(CLIM_TIME_RANGE, tuple):
            valid_clim_time_range = CLIM_TIME_RANGE
        else:
            raise ValueError(f"Invalid clim_time_range={CLIM_TIME_RANGE}")

        # Whole available region spatial coordinates
        whole_region_lat_lon = (
            list(s.region.values())
            if s.region is not None
            else [None, None]
        )

        valid_lat_range = whole_region_lat_lon[0]
        valid_lon_range = whole_region_lat_lon[1]

        leadtime_agg_coord = (
            "leadtime"
            if LEADTIME_AGG_MODE == "single"
            else "leadtime_seasonal"
        )

        period_dim = f"{PERIOD_REFERENCE}_{CLIM_PERIOD.value}"
        period_label = period_dim.replace("_", " ").capitalize()

        # Select and validate leadtimes
        selected_leadtimes = (
            s.leadtimes
            if LEADTIMES is None
            else LEADTIMES
        )

        missing_leadtimes = (
            set(selected_leadtimes)
            - set(s.leadtimes)
        )

        if missing_leadtimes:
            raise ValueError(
                f"Requested leadtimes {sorted(missing_leadtimes)} "
                f"are not available for experiment. "
                f"Available leadtimes: {list(s.leadtimes)}"
            )

        print(
            f"Generate {LEADTIME_AGG_MODE} timeseries "
            f"grouped by {period_dim} for "
            f"{s.var_an, s.var_fc} in {s.region_name} "
            f"(lon={valid_lon_range}, lat={valid_lat_range})"
            f", leadtimes={selected_leadtimes}"
            f", time={valid_time_range}"
            f", clim={valid_clim_time_range}"
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

        # Align FC, analysis, and MLFC within this experiment only.
        # Separate experiment plots retain their own maximum sample coverage.
        evaluation_times = evaluation_times_by_experiment[s.exp_dir.name]
        print(f"Evaluation timestamps for {s.exp_dir.name}: {len(evaluation_times)}")
        for name, ds in (("fc", fc), ("an", an), ("mlfc", mlfc)):
            if ds is None:
                raise ValueError(f"Missing {name} dataset for {s.exp_dir.name}")
            missing_times = evaluation_times.difference(ds.indexes["time"])
            if len(missing_times):
                raise ValueError(
                    f"{name} is missing {len(missing_times)} evaluation "
                    f"timestamps in {s.exp_dir.name}; first: {missing_times[0]}"
                )

        fc = fc.sel(time=evaluation_times)
        an = an.sel(time=evaluation_times)
        mlfc = mlfc.sel(time=evaluation_times)

        mlfc = mlfc.assign_coords(
            leadtime=s.leadtimes
        )

        (
            fc_clim,
            an_clim,
            mlfc_clim,
        ) = calculate_save_and_subset_climatologies(
            s,
            leadtime_units=LEADTIME_UNITS,
            force=RECALCULATE_CLIMATOLOGY,
            clim_period=CLIM_PERIOD,
            rolling_window=CLIM_ROLLING_WINDOW,
            rolling_center=True,
            rolling_min_periods=1,
            lat_range=valid_lat_range,
            lon_range=valid_lon_range,
            time_range=valid_clim_time_range,
            time_start=None,
            interpolate=INTERPOLATE,
            engine="zarr",
            build_analysis=BUILD_ANALYSIS,
            coord_rename_fc=None,
            coord_rename_an=None,
        )

        if mlfc_clim is not None:
            mlfc_clim = mlfc_clim.assign_coords(
                leadtime=s.leadtimes
            )

        leadtime_dim = fc.earthml.guessed_dims.leadtime

        fc = fc.sel(
            {leadtime_dim: selected_leadtimes}
        )
        an = an.sel(
            {leadtime_dim: selected_leadtimes}
        )

        if mlfc is not None:
            mlfc = mlfc.sel(
                {leadtime_dim: selected_leadtimes}
            )

        fc_clim = fc_clim.sel(
            {leadtime_dim: selected_leadtimes}
        )
        an_clim = an_clim.sel(
            {leadtime_dim: selected_leadtimes}
        )

        if mlfc_clim is not None:
            mlfc_clim = mlfc_clim.sel(
                {leadtime_dim: selected_leadtimes}
            )

        # ======================================================
        # Climatological forecast
        # ======================================================

        fc_clim_da = stack_hour_clim(
            fc_clim[s.var_fc],
            CLIM_PERIOD,
        )
        an_clim_da = stack_hour_clim(
            an_clim[s.var_an],
            CLIM_PERIOD,
        )

        fc_anom_da = (
            groupby_period(
                fc[s.var_fc],
                an.earthml.guessed_dims.time,
                CLIM_PERIOD,
            )
            - fc_clim_da
        )

        clim_fc = (
            groupby_period(
                fc_anom_da,
                an.earthml.guessed_dims.time,
                CLIM_PERIOD,
            )
            + an_clim_da
        ).to_dataset(name=s.var_fc)

        realization_dim = fc.earthml.guessed_dims.realization

        an_clim_for_fc = an_clim

        if (
            realization_dim is not None
            and realization_dim in fc.dims
            and realization_dim not in an_clim_for_fc.dims
        ):
            an_clim_for_fc = (
                an_clim_for_fc.expand_dims(
                    {
                        realization_dim:
                        fc[realization_dim]
                    }
                )
            )

        model_datasets = {
            "fc": fc,
            "clim-fc": clim_fc,
            "mlfc": mlfc,
        }

        model_climatologies = {
            "fc": fc_clim,
            "clim-fc": an_clim_for_fc,
            "mlfc": mlfc_clim,
        }

        # ======================================================
        # Calculate + plot timeseries by spatial subregion
        # ======================================================

        for location, location_config in LOCATIONS.items():
            print(f"Spatial subregion: {location}")
            an_loc = subset_timeseries_region(
                an,
                location_config,
            )
            an_clim_loc = subset_timeseries_region(
                an_clim,
                location_config,
            )

            metric_ts_by_model: dict[str, xr.Dataset] = {}
            metric_models = tuple(dict.fromkeys(
                [
                    *(PLOT_MODELS if PLOT_ABSOLUTE_TIMESERIES else ()),
                    *(
                        model
                        for comparison in (MODEL_COMPARISONS if PLOT_IMPROVEMENT_TIMESERIES else ())
                        for model in comparison
                    ),
                ]
            ))

            for model in metric_models:
                ds = subset_timeseries_region(
                    model_datasets[model],
                    location_config,
                )

                ds_clim = subset_timeseries_region(
                    model_climatologies[model],
                    location_config,
                )

                if ds is None or ds_clim is None:
                    continue

                deterministic_metrics = [
                    m
                    for m in metrics_to_compute
                    if is_deterministic(m)
                ]

                probabilistic_metrics = [
                    m
                    for m in metrics_to_compute
                    if is_probabilistic(m)
                ]

                metric_ts_det = xr.Dataset()
                metric_ts_prob = xr.Dataset()
                if deterministic_metrics:
                    print(
                        f"Get {model} deterministic metric timeseries "
                        f"for {location}"
                    )

                    metric_ts_det = get_metrics(
                        an=an_loc,
                        fc=ds,
                        var=s.var_fc,
                        metric_kind="timeseries",
                        leadtime_agg=LEADTIME_AGG_MODE,
                        realization_agg=True,
                        an_clim=an_clim_loc,
                        fc_clim=ds_clim,
                        orography_path=OROGRAPHY_PATH,
                        metrics=deterministic_metrics,
                        leadtime_windows=s.seasonal_leadtime_windows,
                        leadtime_agg_coord=leadtime_agg_coord,
                        clim_period=CLIM_PERIOD,
                        period_reference=PERIOD_REFERENCE,
                        period_dim=period_dim,
                        periods_requested=PERIODS_REQUESTED,
                        leadtime_unit=LEADTIME_UNITS,
                        align=False,
                        fair_correction=False,
                    )

                if probabilistic_metrics:
                    print(
                        f"Get {model} probabilistic metric timeseries "
                        f"for {location}"
                    )

                    metric_ts_prob = get_metrics(
                        an=an_loc,
                        fc=ds,
                        var=s.var_fc,
                        metric_kind="timeseries",
                        leadtime_agg=LEADTIME_AGG_MODE,
                        realization_agg=False,
                        an_clim=an_clim_loc,
                        fc_clim=ds_clim,
                        orography_path=OROGRAPHY_PATH,
                        metrics=probabilistic_metrics,
                        leadtime_windows=s.seasonal_leadtime_windows,
                        leadtime_agg_coord=leadtime_agg_coord,
                        clim_period=CLIM_PERIOD,
                        period_reference=PERIOD_REFERENCE,
                        period_dim=period_dim,
                        periods_requested=PERIODS_REQUESTED,
                        leadtime_unit=LEADTIME_UNITS,
                        align=False,
                        fair_correction=False,
                    )

                metric_ts_by_model[model] = xr.merge(
                    [metric_ts_det, metric_ts_prob]
                )

            available_models = [
                model
                for model in PLOT_MODELS
                if model in metric_ts_by_model
            ]

            available_metrics = [
                m
                for m in METRICS
                if all(
                    m in metric_ts_by_model[model]
                    for model in available_models
                )
            ]

            # Construct common labels
            rolling_label = (
                f"\nroll {ROLLING_MEAN_WINDOW}"
                if ROLLING_MEAN_WINDOW is not None
                else ""
            )
            rolling_filename = (
                f"_roll{ROLLING_MEAN_WINDOW}"
                if ROLLING_MEAN_WINDOW is not None
                else ""
            )

            location_label = f" · {location}" if location_config is not None else ""

            title_suffix = rolling_label + location_label

            if PLOT_ABSOLUTE_TIMESERIES:
                print(
                    f"Plotting metrics {available_metrics} "
                    f"for models {available_models} in {location}"
                )

            common_plot_kwargs = dict(
                var=s.var_fc,
                time_range=valid_time_range,
                leadtime_dim=leadtime_agg_coord,
                train_start=s.train_start,
                train_end=s.train_end,
                val_end=s.val_end,
                test_end=s.test_end,
                extrema_periods=EXTREMA_PERIODS,
                extrema_n_min=EXTREMA_N_MIN,
                extrema_n_max=EXTREMA_N_MAX,
                extrema_annotate=EXTREMA_ANNOTATE,
                extrema_print=EXTREMA_PRINT,
                annotate_times=ANNOTATE_TIMES,
                timeseries_mean_line=TIMESERIES_MEAN_LINE,
                plot_title=PLOT_TITLE,
                plot_labels=PLOT_LABELS,
                plot_legend=PLOT_LEGEND,
                x_label=period_label,
                title_suffix=title_suffix,
                title_size=TITLE_SIZE,
                label_size=LABEL_SIZE,
                tick_size=TICK_SIZE,
                dpi=DPI,
            )

            # ==================================================
            # Plot FC and MLFC
            # ==================================================

            if PLOT_ABSOLUTE_TIMESERIES:
                for m in available_metrics:
                    das = [
                        apply_rolling_mean(
                            metric_ts_by_model[model][m],
                            window=ROLLING_MEAN_WINDOW,
                            center=ROLLING_MEAN_CENTER,
                            min_periods=ROLLING_MEAN_MIN_PERIODS,
                        )
                        for model in available_models
                    ]

                    for lead_value in das[0][leadtime_agg_coord].values:
                        label = safe_label(
                            lead_label(
                                das[0],
                                lead_value,
                                leadtime_agg_coord,
                            )
                        )

                        common_path = (
                            Path("timeseries")
                            / "absolute"
                            / safe_label(period_dim)
                            / safe_label("all")
                            / (
                                f"time_{safe_label(valid_time_range)}"
                                f"_loc_{safe_label(location)}"
                            )
                            / m
                            / LEADTIME_AGG_MODE
                        )

                        filename = (
                            f"{s.var_fc}_{m}_"
                            f"{'-'.join(available_models)}"
                            f"_lead_{label}{rolling_filename}.png"
                        )

                        out_file = OUTPUT_ROOT / s.output_name / common_path / filename
                        if out_file.exists() and not REGENERATE_PLOTS:
                            continue

                        print(f"Saving timeseries {out_file}")

                        plot_timeseries(
                            das,
                            metric=m,
                            models=available_models,
                            lead_value=lead_value,
                            out_file=out_file,
                            ylim=METRIC_LIMITS,
                            timeseries_zero_line=TIMESERIES_ZERO_LINE and "bias" in m,
                            **common_plot_kwargs,
                        )

                        n += 1

            # ==================================================
            # Plot metric improvements
            # ==================================================

            if PLOT_IMPROVEMENT_TIMESERIES:
                for baseline_model, target_model in MODEL_COMPARISONS:
                    if baseline_model not in metric_ts_by_model:
                        continue
                    if target_model not in metric_ts_by_model:
                        continue

                    baseline_ds = metric_ts_by_model[baseline_model]
                    target_ds = metric_ts_by_model[target_model]

                    comparison_metrics = [
                        m
                        for m in METRICS
                        if m in baseline_ds and m in target_ds
                    ]

                    for m in comparison_metrics:
                        improvements = build_metric_improvements(
                            baseline_ds,
                            target_ds,
                            metric=m,
                            baseline_model=baseline_model,
                            target_model=target_model,
                            improvement_units=IMPROVEMENT_UNITS,
                        )

                        for comparison_model, comparison_da in improvements.items():
                            improvement_unit = get_improvement_unit(
                                comparison_model
                            )
                            comparison_da = apply_rolling_mean(
                                comparison_da,
                                window=ROLLING_MEAN_WINDOW,
                                center=ROLLING_MEAN_CENTER,
                                min_periods=ROLLING_MEAN_MIN_PERIODS,
                            )

                            for lead_value in comparison_da[leadtime_agg_coord].values:
                                label = safe_label(
                                    lead_label(
                                        comparison_da,
                                        lead_value,
                                        leadtime_agg_coord,
                                    )
                                )

                                common_path = (
                                    Path("timeseries")
                                    / "comparisons"
                                    / safe_label(comparison_model)
                                    / safe_label(period_dim)
                                    / safe_label("all")
                                    / (
                                        f"time_{safe_label(valid_time_range)}"
                                        f"_loc_{safe_label(location)}"
                                    )
                                    / m
                                    / LEADTIME_AGG_MODE
                                )

                                filename = (
                                    f"{s.var_fc}_{m}_"
                                    f"{safe_label(comparison_model)}_"
                                    f"lead_{label}{rolling_filename}.png"
                                )

                                out_file = OUTPUT_ROOT / s.output_name / common_path / filename

                                if out_file.exists() and not REGENERATE_PLOTS:
                                    continue

                                print(
                                    f"Saving improvement timeseries {out_file}"
                                )

                                plot_timeseries(
                                    [comparison_da],
                                    metric=m,
                                    models=[comparison_model],
                                    lead_value=lead_value,
                                    out_file=out_file,
                                    ylim=METRIC_IMPROVEMENT_LIMITS[improvement_unit],





                                    timeseries_zero_line=TIMESERIES_ZERO_LINE,

                                    timeseries_improved_percentage=TIMESERIES_IMPROVED_PERCENTAGE,
                                    **common_plot_kwargs,

                                    improvement_unit=improvement_unit,
                                )

                                n += 1

    print(f"Done. Saved {n} plots.")


def get_improvement_unit(
    comparison_model: str,
) -> Literal["%", "Δ", "normalized"]:
    if comparison_model.endswith("_percentage"):
        return "%"
    if comparison_model.endswith("_difference"):
        return "Δ"
    if comparison_model.endswith("_normalized"):
        return "normalized"
    raise ValueError(
        f"Cannot determine improvement unit from {comparison_model!r}."
    )

def apply_rolling_mean(
    da: xr.DataArray | None,
    *,
    window: int | None,
    center: bool,
    min_periods: int,
) -> xr.DataArray | None:
    if da is None or window is None:
        return da

    time_dim = da.earthml.guessed_dims.time
    if time_dim is None:
        raise ValueError("Could not determine time dimension for rolling mean.")

    return da.rolling(
        {time_dim: window},
        center=center,
        min_periods=min_periods,
    ).mean(skipna=True)


def subset_timeseries_region(
    obj: xr.DataArray | xr.Dataset | None,
    region: dict | None,
) -> xr.DataArray | xr.Dataset | None:
    if obj is None or region is None:
        return obj

    lat_dim = obj.earthml.guessed_dims.latitude
    lon_dim = obj.earthml.guessed_dims.longitude
    if lat_dim is None or lon_dim is None:
        raise ValueError("Could not determine latitude/longitude dimensions.")

    # Normalize longitudes to [-180, 180].
    if float(obj[lon_dim].max()) > 180:
        lon = ((obj[lon_dim] + 180) % 360) - 180
        obj = obj.assign_coords({lon_dim: lon}).sortby(lon_dim)

    region_type = region.get("type", "rectangle")
    if region_type == "rectangle":
        lat0, lat1 = region["lat_range"]
        lon0, lon1 = region["lon_range"]
        lat_ascending = (
            float(obj[lat_dim].values[0])
            < float(obj[lat_dim].values[-1])
        )
        if lat_ascending:
            lat0, lat1 = sorted((lat0, lat1))
        else:
            lat0, lat1 = sorted((lat0, lat1), reverse=True)

        obj = obj.sel({lat_dim: slice(lat0, lat1)})

        lon0 = ((lon0 + 180) % 360) - 180
        lon1 = ((lon1 + 180) % 360) - 180
        if lon0 <= lon1:
            obj = obj.sel({lon_dim: slice(lon0, lon1)})
        else:
            obj = obj.where(
                (obj[lon_dim] >= lon0) | (obj[lon_dim] <= lon1),
                drop=True,
            )

        return obj

    if region_type == "polygon":
        coordinates = np.asarray(region["coordinates"], dtype=float).copy()
        coordinates[:, 0] = ((coordinates[:, 0] + 180) % 360) - 180

        lon2d, lat2d = np.meshgrid(
            obj[lon_dim].values,
            obj[lat_dim].values,
        )

        points = np.column_stack([lon2d.ravel(), lat2d.ravel()])

        polygon = MplPath(coordinates)

        mask = polygon.contains_points(points).reshape(lat2d.shape)
        mask_da = xr.DataArray(
            mask,
            coords={
                lat_dim: obj[lat_dim],
                lon_dim: obj[lon_dim],
            },
            dims=(lat_dim, lon_dim),
        )

        return obj.where(mask_da)

    raise ValueError(
        f"Unsupported region type {region_type!r}. "
        "Choose 'rectangle' or 'polygon'."
    )


if __name__ == "__main__":
    main()
