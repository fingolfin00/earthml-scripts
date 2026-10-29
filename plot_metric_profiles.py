from typing import Literal
from collections import defaultdict
from collections.abc import Sequence

from pathlib import Path

import numpy as np
import xarray as xr

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

import warnings
from dask.array import PerformanceWarning

warnings.simplefilter("ignore", FutureWarning)
warnings.filterwarnings(
    "ignore",
    category=PerformanceWarning,
)

from dask.diagnostics.progress import ProgressBar

from earthml import (
    LeadtimeUnit,
    ClimPeriod,
    get_experiment_configs,
    get_and_subset_datasets,
)
from earthml.metrics import (
    LeadtimeAgg,
    MetricAgg,
    is_deterministic,
    is_probabilistic,
    get_scalar_metrics,
    ImprovementAgg,
    get_aggregated_metric_improvements,
    calculate_save_and_subset_climatologies,
    stack_hour_clim,
    groupby_period,
    get_required_improvement_metrics,
)
from earthml.plots import (
    safe_label,
    plot_profile,
)

# ==========================================================
# Paths
# ==========================================================

EXP_NAME = "weather_atmo"
# EXP_NAME = "weather_atmo_encoding"
# EXP_NAME = "weather_atmo_seed"
# EXP_NAME = "weather_atmo_normalization"
# EXP_NAME = "weather_atmo_variables"
# EXP_NAME = "weather_atmo_ablation_fixed_val"
# EXP_NAME = "weather_atmo_short_zero_vs_replicate_padding"

EXPERIMENTS_ROOT = Path(
    "/Users/jacopodallaglio/ML/training/seasonal/experiments"
    # f"/work/cmcc/jd19424/ML/MLBC/experiments/{EXP_NAME}"
)

COMMON_PLOT_DIR = Path(
    "/Users/jacopodallaglio/ML/training/seasonal/plots"
    # f"/work/cmcc/jd19424/ML/MLBC/plots/{EXP_NAME}/common"
)

# ==========================================================
# Model settings
# ==========================================================

PLOT_MODELS = (
    "fc",
    "mlfc",
    "clim-fc",
)

MODEL_COMPARISONS = (
    ("fc", "mlfc"),
    # ("clim-fc", "mlfc"),
    # ("fc", "clim-fc"),
)

# ==========================================================
# Global flags
# ==========================================================

INTERPOLATE = True # seasonal
# INTERPOLATE = False # weather

BUILD_ANALYSIS = True

RECALCULATE_CLIMATOLOGY = False
REGENERATE_PLOTS = True
# Compute metric datasets once before repeated plotting
MATERIALIZE_ONCE = False

# Plot individual ensemble members in addition to the aggregate
PLOT_MEMBERS = False

# ==========================================================
# Plot generation
# ==========================================================

# Individual experiments
PLOT_INDIVIDUAL_LEADTIME_PROFILES = False
PLOT_INDIVIDUAL_CLIMATOLOGICAL_PROFILES = False

# Combined variables
PLOT_COMBINED_VARIABLE_LEADTIME_PROFILES = False
PLOT_COMBINED_VARIABLE_CLIMATOLOGICAL_PROFILES = False

# Combined experiments
PLOT_COMBINED_EXPERIMENT_LEADTIME_PROFILES = False
PLOT_COMBINED_EXPERIMENT_CLIMATOLOGICAL_PROFILES = False
PLOT_COMBINED_EXPERIMENT_AXIS_PROFILES = True

# ==========================================================
# Time selection
# ==========================================================

SPLIT_RANGE: list[Literal["train", "val", "test"]] = [
    # "train",
    # "val",
    "test",
]

TIME_RANGE = None
# TIME_RANGE = ("2018-01-01", "2022-12-31")

INFERENCE_PERIOD = None
# INFERENCE_PERIOD = ("2025-01-01", "2025-10-31")

# None -> automatically select periods based on plot requests
# Otherwise, explicitly select the requested periods
PERIODS_REQUESTED = None
# PERIODS_REQUESTED = ["01", "12"]

# ==========================================================
# Climatology
# ==========================================================

CLIM_TIME_RANGE: tuple[str, str] | Literal[
    "train", "val", "test", "train_val", "full"
] = (
    "train" # seasonal
    # "train_val" # weather
)

CLIM_PERIOD: ClimPeriod = ClimPeriod.MONTH
CLIM_ROLLING_WINDOW = None

# Period grouping reference:
#   "init"  -> group metrics by forecast initialization time
#   "valid" -> group metrics by forecast valid time (init + lead time)
PERIOD_REFERENCE = "init"


# CLIM_PERIOD: ClimPeriod = ClimPeriod.DAYOFYEAR_HOUR
# CLIM_ROLLING_WINDOW = 31

# CLIM_PERIOD: ClimPeriod = ClimPeriod.DAYOFYEAR
# CLIM_ROLLING_WINDOW = None

# ==========================================================
# Metric aggregation
# ==========================================================

METRIC_AGG_MODE: MetricAgg = "spatial_avg"
# "timeseries_avg"
# "spatial_avg"
# "global"

IMPROVEMENT_AGG_MODE: ImprovementAgg = "before"
# "after": aggregate model metrics, then calculate improvement
# "before": calculate local improvement, then aggregate

LEADTIME_AGG_MODE: LeadtimeAgg = "single"
# "single"
# "aggregated" 
# "seasonal_window"

LEADTIME_UNITS = LeadtimeUnit.MONTHS # seasonal
# LEADTIME_UNITS = LeadtimeUnit.HOURS # weather

LEADTIME_PROFILE_LABEL = f"Lead time [{LEADTIME_UNITS.value}]"

# ==========================================================
# Metrics
# ==========================================================

METRICS = [
    # Deterministic metrics - absolute fields
    # "bias",
    # "mae",
    # "mse",
    "rmse",
    # "nrmse",
    # "corr",
    # "scc",
    # "r2",
    # "fc_std",
    # "an_std",
    # "std_ratio",

    # Gradient metrics
    # "fc_grad_mag",
    # "an_grad_mag",
    # "grad_rmse",

    # MSE decomposition / calibration diagnostics
    # "mse_bias_component",
    # "mse_std_component",
    # "mse_corr_component",
    # "crmse",
    # "regression_slope",

    # Deterministic metrics - anomaly fields
    # "bias_anom",
    # "mae_anom",
    # "mse_anom",
    # "rmse_anom",
    # "nrmse_anom",
    # "acc",
    # "r2_anom",
    # "fc_anom_std",
    # "an_anom_std",
    # "std_ratio_anom",

    # Gradient metrics
    # "fc_anom_grad_mag",
    # "an_anom_grad_mag",
    # "grad_rmse_anom",

    # Anomaly MSE decomposition / calibration diagnostics
    # "mse_bias_component_anom",
    # "mse_std_component_anom",
    # "mse_corr_component_anom",
    # "crmse_anom",
    # "regression_slope_anom",

    # Skill scores vs climatology
    # "mse_skill_clim",
    # "mae_anom_skill_clim",
    # "mse_anom_skill_clim",
    # "rmse_anom_skill_clim",
    # "ens_member_mse_anom_skill_clim",
    # "mean_member_mse_anom_skill_clim",

    # Ensemble / probabilistic metrics - absolute fields
    # "ens_member_rmse",
    # "mean_member_rmse",
    # "spread",
    # "spread_skill_ratio",
    # "crps",
    # "rank_histogram",

    # Ensemble / probabilistic metrics - anomaly fields
    # "ens_member_rmse_anom",
    # "mean_member_rmse_anom",
    # "spread_anom",
    # "spread_anom_skill_ratio",
    # "crps_anom",
    # "rank_histogram_anom",

    # ROC AUC - anomaly terciles
    # "roc_anom_lower",
    # "roc_anom_middle",
    # "roc_anom_upper",
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
    # "ConUS",
    # "Europe",
    # "Pacific",
    "World",
    # None,
]

FILTERS = dict(
    var_fc=VARIABLES,
    region_name=REGIONS,

    # net_name="ConvNeXtTransformerUNet",
    # net_name="SmaAt_UNet",

    # target_mode="analysis",

    # seasonal_encoding=True,
    # ensemble_encoding=True,
    # input_realization_avg=True,
    # channel_representation="variable",

    # normalization_input_mean="sample",
    # normalization_input_std="sample",
    # normalization_target_mean="sample",
    # normalization_target_std="sample",
    # sample_target_mean="input",
    # sample_target_std="input",

    loss_name="GeoMaskedMSELoss",
    # loss_name="GeoMaskedMSEMultiScaleLoss",
    # loss_name="SpatialDegradationMSELoss",

    # train_start="2019-10-14",

    # separate_training_by_init_period=None,
    # separate_training_by_init_period=ClimPeriod.MONTH,

    # extra_suffix_folder="264samples_consecutive",
    # extra_suffix_folder="NOAA_copy",
    # extra_suffix_folder="",
)

# ==========================================================
# Spatial subset
# ==========================================================

# ConUS
# LAT_RANGE = (50, 25)
# LON_RANGE = (-130, -60)

# Europe
# LAT_RANGE = (80, 30)
# LON_RANGE = (-30, 60)

# Pacific
# LAT_RANGE = (20, -20)
# LON_RANGE = (-195, -135)

# Whole configured region
LAT_RANGE = None
LON_RANGE = None

# ==========================================================
# Common profile styling
# ==========================================================

PLOT_TITLE = True
PLOT_LEGEND = True

FIGSIZE = (16, 12)
TITLE_SIZE = None
LABEL_SIZE = 20
TICK_SIZE = 15
DPI = 300

# title_strftime = "%Y"     # seasonal
TITLE_STRFTIME = "%m.%Y"    # weather

MODEL_LABELS = {
    "fc": "FC",
    "mlfc": "MLFC",
    "clim-fc": "Clim-corrected FC",
}

COMPARISON_LABELS_MAP = {
    ("fc", "mlfc", "percentage"): "MLFC vs FC [%]",
    ("fc", "mlfc", "difference"): "MLFC vs FC [Δ]",
    ("fc", "mlfc", "normalized"): "MLFC vs FC [norm.]",
}

# Custom profile bands (lower and upper offsets)
PROFILE_BAND_BOUNDS = {
    # "rmse": (-0.1, 0.1),
    # "mae": (-0.1, 0.1),
    # "bias": (-0.2, 0.2),
}

PROFILE_BAND_ALPHA = 0.18

# ==========================================================
# Fixed y-axis limits
# ==========================================================

# Missing metrics, or explicit None values, use automatic limits
PROFILE_YLIMS = {
    "absolute": {
        # Weather: day-of-year climatology
        # "bias": (-1, 1),
        # "rmse": (0, 2),
        # "corr": (0.5, 1.0),

        # Weather: monthly climatology
        # "bias": (-0.5, 0.5),
        # "rmse": (0.8, 1.8),   # only FC
        # "rmse": (0.5, 1.8),

        # "corr": (0.8, 1.0),   # only FC
        # "corr": (0.5, 1.0),
    },

    "percentage": {
        # "rmse": (-10, 10),
    },

    "difference": {
        # "bias": (0, 0.2), # temporal avg
    #     "bias": (-0.1, 0.1), # spatial avg
    #     "rmse": (0, 0.2),
    #     "mae": (0, 0.2),
    #     "scc": (0.0015, 0.0035),
    #     "corr": (0.0005, 0.0035),
    },

    "normalized": {
        # "rmse": (-1, 1),
    },
}

# ==========================================================
# Climatological profiles
# ==========================================================
# climatological period in x axis, e.g. month, curves correspond to selected leadtimes

# If True, put all selected lead times in the same figure
# If False, generate one figure per lead time
COMBINE_CLIMATOLOGICAL_PROFILE_LEADTIMES = True

# How lead times are distinguished when combined:
#   "shades" -> shades of each model color
#   "colors" -> one distinct color per lead time
CLIMATOLOGICAL_LEADTIME_COLORS = "shades"

# None -> use every available lead time
# WANTED_CLIMATOLOGICAL_PROFILE_LEADTIMES = [12, 24, 36, 48, 60, 72]
# WANTED_CLIMATOLOGICAL_PROFILE_LEADTIMES = [72]
WANTED_CLIMATOLOGICAL_PROFILE_LEADTIMES = None

# ==========================================================
# Combined-variable settings
# ==========================================================

# Plot one lead-time profile containing all selected variables

# Raw sources
# COMBINED_VARIABLE_SOURCE = "fc"
# COMBINED_VARIABLE_SOURCE = "mlfc"
# COMBINED_VARIABLE_SOURCE = "clim-fc"

# MLFC improvement over FC
COMBINED_VARIABLE_SOURCE = "mlfc-fc-difference"
# COMBINED_VARIABLE_SOURCE = "mlfc-fc-percentage"
# COMBINED_VARIABLE_SOURCE = "mlfc-fc-normalized"

COMBINED_VARIABLE_IS_IMPROVEMENT = COMBINED_VARIABLE_SOURCE.startswith("mlfc-fc-")

COMBINED_VARIABLE_PLOT_FOLDER = "variable_comparison"
COMBINED_VARIABLE_PLOT_LEGEND = False

# Combined-variable climatological profiles:
#   x = climatological period
#   color = variable
#   shade = lead time
#
# If True, put all selected lead times in one figure
# If False, generate one figure per lead time
COMBINE_COMBINED_VARIABLE_CLIMATOLOGICAL_LEADTIMES = False

# None -> use every available lead time
COMBINED_VARIABLE_CLIMATOLOGICAL_LEADTIMES = [72,]
# COMBINED_VARIABLE_CLIMATOLOGICAL_LEADTIMES = None

# When lead times are combined, use shades of each variable's base color
COMBINED_VARIABLE_CLIMATOLOGICAL_LEADTIME_COLORS = "shades"

VARIABLE_LABELS = {
    "mslp": "MSLP",
    "t2m": "T2M",
    "d2m": "D2M",
    "u10": "U10",
    "v10": "V10",
    "tcc": "TCC",

    "sst": "SST",
    "ssh": "SSH",
    "tprate": "TP rate",
}

VARIABLE_COLORS = {
    "mslp": "tab:blue",
    "t2m": "tab:red",
    "d2m": "tab:orange",
    "u10": "tab:green",
    "v10": "tab:purple",
    "tcc": "tab:brown",

    "sst": "tab:pink",
    "ssh": "tab:cyan",
    "tprate": "tab:olive",
}

# ==========================================================
# Combined-experiment profiles
# ==========================================================

COMBINED_PLOT_FOLDER = "profile_comparison"
COMPARISON_NAME = "encoding"

COMPARISON_LABELS = None
COMPARISON_COLORS = None

COMPARISON_LABEL_FIELDS = [
    "target_mode",
    "net_name",
    "climatology_encoding",
    "climatology_error_encoding",
    "temporal_evolution_encoding",
    "seasonal_encoding",
    "spatial_encoding",
    "orography_encoding",
    "normalization_input_mean",
    "normalization_input_std",
    "normalization_target_mean",
    "normalization_target_std",
]

# COMPARISON_NAME = "ablation_fixed_val_samples"

# COMPARISON_LABELS = [
#     "1000",
#     "1200",
#     "1250",
#     "1300",
#     "1500",
#     "2000",
#     "264",
#     "500",
#     "2474",  # base experiment
# ]

# COMPARISON_NAME = "normalization_modes"

# COMPARISON_LABELS = [
#     r"Channel $\mu$ only",
#     r"Channel $\mu/\sigma$",
#     r"Channel $\mu$, sample $\sigma$",
#     r"Sample $\mu$ only",
#     r"Sample $\mu$, channel $\sigma$",
#     r"Sample $\mu/\sigma$ (input $\rightarrow$ target)",
#     r"Sample $\mu/\sigma$ (target $\rightarrow$ target)",
# ]

# COMPARISON_NAME = "channel_mean_std_variability"

# COMPARISON_LABELS = [
#     r"Channel $\mu/\sigma$ 42 seed (1)",
#     r"Channel $\mu/\sigma$ 42 seed (2)",
#     r"Channel $\mu/\sigma$ 42 seed (3)",
#     r"Channel $\mu/\sigma$ 20 seed",
#     r"Channel $\mu/\sigma$ 40 seed",
#     r"Channel $\mu/\sigma$ 41 seed",
#     r"Channel $\mu/\sigma$ 43 seed",
#     r"Channel $\mu/\sigma$ 44 seed",
# ]

# COMPARISON_NAME = "encoding"

# COMPARISON_LABELS = [
#     r"Sample $\mu/\sigma$ all encoding ConvNeXt",
#     r"Sample $\mu/\sigma$ clim and clim err encoding",
#     r"Sample $\mu/\sigma$ clim encoding",
#     # r"Sample $\mu/\sigma$ clim err encoding",
#     r"Sample $\mu/\sigma$ no encoding ConvNeXt",
#     r"Sample $\mu/\sigma$ no encoding",
#     # r"Sample $\mu/\sigma$ oro encoding",
#     r"Sample $\mu/\sigma$ seasonal encoding",
#     r"Sample $\mu/\sigma$ spatial encoding",
#     # r"Sample $\mu/\sigma$ oro, seasonal, spatial, temp -1,+1 encoding",
#     # r"Sample $\mu/\sigma$ spatial, temp -1,+1 encoding",
#     # r"Channel $\mu/\sigma$ temp -4,-3,-2,-1 encoding",
#     r"Sample $\mu/\sigma$ temp -4,-3,-2,-1 encoding",
#     r"Sample $\mu/\sigma$ oro, seasonal, spatial, temp -4,-3,-2,-1 encoding",
#     r"Residual sample $\mu/\sigma$, channel target all encoding",
#     r"Residual sample $\mu/\sigma$, channel target no encoding",
#     r"Residual sample $\mu/\sigma$, channel target no encoding ConvNeXt",
# ]

# COMPARISON_COLORS = {
#     "fc": "tab:blue",
#     "clim-fc": "tab:orange",
#     COMPARISON_LABELS[0]: "darkviolet",
#     COMPARISON_LABELS[1]: "forestgreen",
#     COMPARISON_LABELS[2]: "deeppink",
#     COMPARISON_LABELS[3]: "firebrick",
#     COMPARISON_LABELS[4]: "dimgray",
#     COMPARISON_LABELS[5]: "teal",
#     COMPARISON_LABELS[6]: "darkgoldenrod",
#     COMPARISON_LABELS[7]: "navy",
#     COMPARISON_LABELS[8]: "orchid",
#     COMPARISON_LABELS[9]: "lime",
#     COMPARISON_LABELS[10]: "red",
#     COMPARISON_LABELS[11]: "blue",
# }

# ==========================================================
# Combined-experiment climatological profiles
# ==========================================================

# x = climatological period, e.g. month
# color = experiment
# shade/style = lead time when stacked
#
# Applies to both absolute MLFC metrics and MLFC-vs-FC improvements
# None -> use every available lead time
# COMBINED_EXPERIMENT_CLIMATOLOGICAL_LEADTIMES = [72]
COMBINED_EXPERIMENT_CLIMATOLOGICAL_LEADTIMES = None

# True -> stack all selected lead times in the same climatology figure
# False -> generate one climatology figure per lead time
COMBINE_COMBINED_EXPERIMENT_CLIMATOLOGICAL_LEADTIMES = False

# How lead times are distinguished in a stacked climatology figure:
#   "shades" -> keep one base color per experiment and vary shade by lead time
#   "colors" -> use distinct colors for lead times
COMBINED_EXPERIMENT_CLIMATOLOGICAL_LEADTIME_COLORS = "shades"
COMBINED_EXPERIMENT_CLIMATOLOGICAL_LEADTIME_SHADE_RANGE = (0.2, 1.0)

# ==========================================================
# Experiment-axis profiles
# ==========================================================

# x = experiment label
# y = metric at one selected lead time and climatological period
# This is especially useful for ablation studies such as sample size

# None -> use every available lead time
# combined_experiment_axis_leadtimes = None
COMBINED_EXPERIMENT_AXIS_LEADTIMES = [72]

# Combine all selected lead times in one experiment-axis figure
COMBINE_COMBINED_EXPERIMENT_AXIS_LEADTIMES = False

# When combined: "shades" or "colors" for lead-time curves
COMBINED_EXPERIMENT_AXIS_LEADTIME_COLORS = "colors"
COMBINED_EXPERIMENT_AXIS_LEADTIME_SHADE_RANGE = (0.2, 1.0)

# Optional horizontal reference lines in absolute experiment-axis profiles
# combined_experiment_axis_reference_lines = ("fc", "clim-fc")
COMBINED_EXPERIMENT_AXIS_REFERENCE_LINES = () # disable

# Sort labels numerically when they start with a number, e.g.
# "264 samples", "500 samples", ..., "2474 samples"
COMBINED_EXPERIMENT_AXIS_SORT_NUMERIC = False

# Connect points after sorting. The experiment-specific marker colors
# still come from comparison_colors.
COMBINED_EXPERIMENT_AXIS_CONNECT = True

# None -> use comparison_name with underscores replaced by spaces
# COMBINED_EXPERIMENT_AXIS_LABEL = "Training samples"
COMBINED_EXPERIMENT_AXIS_LABEL = None
COMBINED_EXPERIMENT_AXIS_TICK_ROTATION = 45


def main() -> None:
    period_profile_label = get_period_profile_label(
        PERIOD_REFERENCE,
        CLIM_PERIOD,
    )

    # Shared plotting defaults
    plot_kwargs = dict(
        plot_title=PLOT_TITLE,
        label_size=LABEL_SIZE,
        tick_size=TICK_SIZE,
        dpi=DPI,
        figsize=FIGSIZE,
    )

    # Common arguments for leadtime and climatological curves
    profile_kwargs = dict(
        **plot_kwargs,
        realization_dim="realization",
        spread="std",
        title_strftime=TITLE_STRFTIME,
        title_size=TITLE_SIZE,
        profile_band_bounds=PROFILE_BAND_BOUNDS,
        profile_band_alpha=PROFILE_BAND_ALPHA,
    )

    # Combined-variable and combined-experiment curves always display markers.
    combined_profile_kwargs = dict(
        **profile_kwargs,
        profile_markers=True,
    )

    # ==========================================================
    # Experiment selection
    # ==========================================================

    settings = get_experiment_configs(
        EXPERIMENTS_ROOT,
        **FILTERS,
    )

    print(f"Found {len(settings)} matching experiment(s)")

    # ==========================================================
    # Generate combined profile labels, colors and linestyles
    # ==========================================================

    plot_combined_experiments = (
        PLOT_COMBINED_EXPERIMENT_LEADTIME_PROFILES
        or PLOT_COMBINED_EXPERIMENT_CLIMATOLOGICAL_PROFILES
        or PLOT_COMBINED_EXPERIMENT_AXIS_PROFILES
    )

    if plot_combined_experiments:
        if COMPARISON_LABELS is None:
            fields = [
                name
                for name in COMPARISON_LABEL_FIELDS
                if len({str(getattr(s, name)) for s in settings}) > 1
            ]

            comparison_labels = [
                ", ".join(
                    f"{name.removesuffix('_encoding').replace('_', ' ')}={getattr(s, name)}"
                    for name in fields
                ) or "Default"
                for s in settings
            ]
        else:
            comparison_labels = COMPARISON_LABELS

        if len(settings) != len(comparison_labels):
            raise ValueError("Number of experiments and labels differ")

        if len(set(comparison_labels)) != len(comparison_labels):
            raise ValueError("Comparison labels are not unique")

        comparison_labels_by_experiment = {
            s.exp_dir.name: label
            for s, label in zip(settings, comparison_labels, strict=True)
        }

        colors = plt.get_cmap("tab20").resampled(len(comparison_labels))

        comparison_colors = (
            COMPARISON_COLORS
            if COMPARISON_COLORS is not None
            else {
                **{"fc": "tab:blue", "clim-fc": "tab:orange"},
                **{
                    label: colors(i)
                    for i, label in enumerate(comparison_labels)
                },
            }
        )

        comparison_linestyles = {
            model: "--" if model in {"fc", "clim-fc"} else "-"
            for model in comparison_colors
        }

    # ==========================================================
    # Skip experiments without completed forecast outputs
    # ==========================================================

    # Skip unfinished experiments
    split_paths = {
        "train": "output_mlfc_train",
        "val": "output_mlfc_val",
        "test": "output_mlfc_test",
    }

    available_settings = []

    for s in settings:
        missing = [
            split
            for split in SPLIT_RANGE
            if not getattr(s, split_paths[split]).exists()
        ]

        if missing:
            print(
                f"Skipping unfinished experiment: {s.output_name} "
                f"(missing MLFC: {', '.join(missing)})"
            )
            continue

        available_settings.append(s)

    settings = available_settings

    if not settings:
        print("No completed experiments available")
        return

    print(f"Found {len(settings)} matching experiment(s)")

    # ==========================================================
    # Common evaluation timestamps across experiments
    # ==========================================================

    if (
        not SPLIT_RANGE
        or len(set(SPLIT_RANGE)) != len(SPLIT_RANGE)
        or not set(SPLIT_RANGE) <= {"train", "val", "test"}
    ):
        raise ValueError(f"Invalid split_range={SPLIT_RANGE}")

    split_paths = {
        "train": "output_mlfc_train",
        "val": "output_mlfc_val",
        "test": "output_mlfc_test",
    }

    common_times = None

    for s in settings:
        if INFERENCE_PERIOD is None:
            available_times = None

            for split in SPLIT_RANGE:
                path = getattr(s, split_paths[split])

                with xr.open_zarr(path, consolidated=False) as ds:
                    split_times = ds.indexes["time"]

                available_times = (
                    split_times
                    if available_times is None
                    else available_times.union(split_times)
                )

        else:
            inference_start, inference_end = INFERENCE_PERIOD

            path = (
                s.exp_dir
                / "inference"
                / f"{inference_start}_{inference_end}"
                / "test_corrected.zarr"
            )

            with xr.open_zarr(path, consolidated=False) as ds:
                available_times = ds.indexes["time"]

        if TIME_RANGE is not None:
            available_times = available_times[
                (available_times >= TIME_RANGE[0])
                & (available_times <= TIME_RANGE[1])
            ]

        common_times = (
            available_times
            if common_times is None
            else common_times.intersection(available_times)
        )

    if common_times is None or len(common_times) == 0:
        raise ValueError(
            "No common evaluation timestamps across experiments."
        )

    common_times = common_times.sort_values()

    print(
        f"Common evaluation period: "
        f"{common_times[0]} to {common_times[-1]} "
        f"({len(common_times)} samples)"
    )

    # Add auxiliary metrics required for improvement representations
    # e.g. standard deviation for normalized improvements
    metrics_to_compute = get_required_improvement_metrics(
        list(METRICS),
    )

    # ==========================================================
    # Requested periods
    # ==========================================================

    if PERIODS_REQUESTED is None:

        need_individual_periods = any((
            PLOT_INDIVIDUAL_CLIMATOLOGICAL_PROFILES,
            PLOT_COMBINED_VARIABLE_CLIMATOLOGICAL_PROFILES,
            PLOT_COMBINED_EXPERIMENT_CLIMATOLOGICAL_PROFILES,
        ))

        periods_requested = ["all"]

        if need_individual_periods:
            if CLIM_PERIOD == ClimPeriod.MONTH:
                periods_requested += [
                    f"{month:02d}" for month in range(1, 13)
                ]
            else:
                raise ValueError(f"Automatic period selection not implemented for {CLIM_PERIOD}")

    else:
        periods_requested = list(PERIODS_REQUESTED)

    # ==========================================================
    # Combined-experiment storage
    # ==========================================================

    # Grouping dicts contain everything that must be common for curves to share one lead-time axis

    # Groups across experiments
    combined_groups = defaultdict(
        lambda: {
            "fc": None,
            "fc_fields": None,
            "fc_members": None,
            "clim-fc": None,
            "clim-fc_fields": None,
            "clim-fc_members": None,
            "mlfc": [],
            "mlfc_fields": [],
            "mlfc_members": [],
            "spatial_reference": None,
            "labels": [],
            "settings": [],
            "valid_time_range": None,
            "valid_lat_range": None,
            "valid_lon_range": None,
            "period_dim": None,
            "leadtime_agg_coord": None,
        }
    )

    # Groups across variables
    combined_variable_groups = defaultdict(
        lambda: {
            "metrics": {},
            "valid_time_range": None,
            "valid_lat_range": None,
            "valid_lon_range": None,
            "period_dim": None,
            "leadtime_agg_coord": None,
            "variant": "absolute",
            "improvement_unit": None,
        }
    )

    n = 0

    # ==========================================================
    # Experiments
    # ==========================================================

    for s in settings:

        # ------------------------------------------------------
        # Time range
        # ------------------------------------------------------

        valid_time_range = (
            common_times[0].strftime("%Y-%m-%d"),
            common_times[-1].strftime("%Y-%m-%d"),
        )

        if INFERENCE_PERIOD is None:
            mlfc_path = None
        else:
            inference_start, inference_end = INFERENCE_PERIOD

            mlfc_path = (
                s.exp_dir
                / "inference"
                / f"{inference_start}_{inference_end}"
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
            raise ValueError(f"Invalid CLIM_TIME_RANGE={CLIM_TIME_RANGE!r}")

        # ------------------------------------------------------
        # Spatial range
        # ------------------------------------------------------

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

        # ------------------------------------------------------
        # Dimension names
        # ------------------------------------------------------

        leadtime_agg_coord = (
            "leadtime"
            if LEADTIME_AGG_MODE == "single"
            else "leadtime_seasonal"
        )

        period_dim = f"{PERIOD_REFERENCE}_{CLIM_PERIOD.value}"

        print(
            f"Generate {LEADTIME_AGG_MODE} profiles "
            f"for {(s.var_an, s.var_fc)} in {s.region_name} "
            f"(lon={valid_lon_range}, lat={valid_lat_range})"
        )

        # ------------------------------------------------------
        # Data
        # ------------------------------------------------------

        fc, an, mlfc = get_and_subset_datasets(
            s,
            leadtime_units=LEADTIME_UNITS,
            lat_range=valid_lat_range,
            lon_range=valid_lon_range,
            time_range=valid_time_range,
            interpolate=INTERPOLATE,
            mlfc_path=mlfc_path,
        )

        fc = fc.sel(time=common_times)
        an = an.sel(time=common_times)
        mlfc = mlfc.sel(time=common_times)

        mlfc = mlfc.assign_coords(
            leadtime=s.leadtimes
        )
   
        # ------------------------------------------------------
        # Climatologies
        # ------------------------------------------------------

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
                time_range=valid_clim_time_range,
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

        # ------------------------------------------------------
        # Leadtime subset
        # ------------------------------------------------------

        leadtime_dim = fc.earthml.guessed_dims.leadtime

        fc = fc.sel(
            {leadtime_dim: s.leadtimes}
        )

        an = an.sel(
            {leadtime_dim: s.leadtimes}
        )

        fc_clim = fc_clim.sel(
            {leadtime_dim: s.leadtimes}
        )

        an_clim = an_clim.sel(
            {leadtime_dim: s.leadtimes}
        )

        # ------------------------------------------------------
        # Climatology-corrected forecast
        # ------------------------------------------------------

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
                fc.earthml.guessed_dims.time,
                CLIM_PERIOD,
            )
            - fc_clim_da
        )

        clim_fc = (
            groupby_period(
                fc_anom_da,
                fc.earthml.guessed_dims.time,
                CLIM_PERIOD,
            )
            + an_clim_da
        ).to_dataset(
            name=s.var_fc
        )


        # ------------------------------------------------------
        # Climatology realization handling
        # ------------------------------------------------------

        an_clim_for_fc = an_clim

        realization_dim = (
            fc.earthml.guessed_dims.realization
        )

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

        # ------------------------------------------------------
        # Available models
        # ------------------------------------------------------

        datasets = {
            "fc": (
                fc,
                fc_clim,
            ),
            "mlfc": (
                mlfc,
                mlfc_clim,
            ),
        }

        plot_combined_variables = (
            PLOT_COMBINED_VARIABLE_LEADTIME_PROFILES
            or PLOT_COMBINED_VARIABLE_CLIMATOLOGICAL_PROFILES
        )

        need_clim_fc = (
            "clim-fc" in PLOT_MODELS
            or any(
                "clim-fc" in comparison
                for comparison in MODEL_COMPARISONS
            )
            or (
                plot_combined_variables
                and COMBINED_VARIABLE_SOURCE == "clim-fc"
            )
        )

        if need_clim_fc:
            datasets["clim-fc"] = (
                clim_fc,
                an_clim_for_fc,
            )

        # ------------------------------------------------------
        # Metrics to compute
        # ------------------------------------------------------

        deterministic_metrics = [
            metric
            for metric in metrics_to_compute
            if is_deterministic(metric)
        ]

        probabilistic_metrics = [
            metric
            for metric in metrics_to_compute
            if is_probabilistic(metric)
        ]


        metrics_by_model: dict[
            str,
            xr.Dataset,
        ] = {}

        metric_fields_by_model: dict[
            str,
            xr.Dataset,
        ] = {}

        members_by_model: dict[
            str,
            xr.Dataset,
        ] = {}

        # ------------------------------------------------------
        # Compute metrics
        # ------------------------------------------------------

        for model, (
            ds,
            ds_clim,
        ) in datasets.items():

            if ds is None or ds_clim is None:
                continue

            print(
                f"Get {model} profile metrics"
            )

            metric_kwargs = dict(
                an=an,
                fc=ds,
                var=s.var_fc,
                leadtime_windows=s.seasonal_leadtime_windows,
                an_clim=an_clim,
                fc_clim=ds_clim,
                metric_agg_mode=METRIC_AGG_MODE,
                leadtime_agg_mode=LEADTIME_AGG_MODE,
                leadtime_agg_coord=leadtime_agg_coord,
                clim_period=CLIM_PERIOD,
                period_reference=PERIOD_REFERENCE,
                period_dim=period_dim,
                periods_requested=periods_requested,
                leadtime_unit=LEADTIME_UNITS,
            )

            metric_parts = []
            metric_field_parts = []

            # --------------------------------------------------
            # Deterministic / ensemble-mean metrics
            # --------------------------------------------------

            if deterministic_metrics:
                profile_part, field_part = get_scalar_metrics(
                    metrics=deterministic_metrics,
                    realization_agg=True,
                    **metric_kwargs,
                )

                metric_parts.append(profile_part)
                metric_field_parts.append(field_part)

            # --------------------------------------------------
            # Probabilistic metrics
            # --------------------------------------------------

            if probabilistic_metrics:
                profile_part, field_part = get_scalar_metrics(
                    metrics=probabilistic_metrics,
                    realization_agg=False,
                    **metric_kwargs,
                )

                metric_parts.append(profile_part)
                metric_field_parts.append(field_part)

            if metric_parts:
                metrics_by_model[model] = xr.merge(
                    metric_parts
                )

                metric_fields_by_model[model] = xr.merge(
                    metric_field_parts
                )

            # --------------------------------------------------
            # Individual-member deterministic metrics
            # --------------------------------------------------

            if (
                PLOT_MEMBERS
                and deterministic_metrics
            ):
                member_profiles, _ = get_scalar_metrics(
                    metrics=deterministic_metrics,
                    realization_agg=False,
                    **metric_kwargs,
                )

                members_by_model[model] = member_profiles

        if not metrics_by_model:
            print(
                f"No metrics available for "
                f"{s.output_name}"
            )
            continue

        # ------------------------------------------------------
        # Optional materialization
        # ------------------------------------------------------

        if MATERIALIZE_ONCE:
            with ProgressBar():

                metrics_by_model = {
                    model: ds.compute()
                    for model, ds
                    in metrics_by_model.items()
                }

                metric_fields_by_model = {
                    model: ds.compute()
                    for model, ds
                    in metric_fields_by_model.items()
                }

                members_by_model = {
                    model: ds.compute()
                    for model, ds
                    in members_by_model.items()
                }

        comparison_source_by_model = metric_fields_by_model

        # ------------------------------------------------------
        # Available periods
        # ------------------------------------------------------

        reference_ds = next(
            iter(metrics_by_model.values())
        )

        available_periods = [
            str(value)
            for value in reference_ds[
                period_dim
            ].values
            if str(value) in periods_requested
        ]

        leadtime_profile_periods = (
            available_periods
            if PLOT_INDIVIDUAL_LEADTIME_PROFILES
            else []
        )

        climatological_profile_leadtimes = (
            get_climatological_profile_leadtimes(
                reference_ds,
                leadtime_dim=leadtime_agg_coord,
                wanted_leadtimes=(
                    WANTED_CLIMATOLOGICAL_PROFILE_LEADTIMES
                ),
            )
            if PLOT_INDIVIDUAL_CLIMATOLOGICAL_PROFILES
            else []
        )

        # ------------------------------------------------------
        # Individual experiment plots
        # ------------------------------------------------------

        if (
            PLOT_INDIVIDUAL_LEADTIME_PROFILES
            or PLOT_INDIVIDUAL_CLIMATOLOGICAL_PROFILES
        ):

            # ==================================================
            # Absolute profiles
            # ==================================================

            available_models = tuple(
                model
                for model in PLOT_MODELS
                if model in metrics_by_model
            )
            available_metrics = [
                metric
                for metric in METRICS
                if (
                    metric != "rank_histogram"
                    and all(
                        metric
                        in metrics_by_model[
                            model
                        ]
                        for model
                        in available_models
                    )
                )
            ]

            print(
                f"Plotting absolute metric profiles "
                f"{available_metrics} "
                f"for periods {available_periods} "
                f"for exp {s.output_name}"
            )

            for metric in available_metrics:

                das = [
                    metrics_by_model[
                        model
                    ][metric]
                    for model
                    in available_models
                ]

                das_member = []

                for model in available_models:
                    member_ds = (
                        members_by_model.get(
                            model
                        )
                    )

                    if (
                        member_ds is not None
                        and metric in member_ds
                    ):
                        das_member.append(
                            member_ds[metric]
                        )

                    else:
                        das_member.append(None)


                for period_value in leadtime_profile_periods:

                    common_path = (
                        Path("profiles")
                        / "absolute"
                        / safe_label(period_dim)
                        / safe_label(
                            period_value
                        )
                        / (
                            f"time_"
                            f"{safe_label(valid_time_range)}"
                            f"_lat_"
                            f"{safe_label(valid_lat_range)}"
                            f"_lon_"
                            f"{safe_label(valid_lon_range)}"
                        )
                        / metric
                        / METRIC_AGG_MODE
                        / IMPROVEMENT_AGG_MODE
                    )

                    filename = (
                        f"{s.var_fc}_"
                        f"{metric}_"
                        f"{LEADTIME_AGG_MODE}lt.png"
                    )

                    out_file = (
                        s.plot_dir
                        / common_path
                        / filename
                    )

                    if (
                        out_file.exists()
                        and not REGENERATE_PLOTS
                    ):
                        continue

                    print(f"Saving profile {out_file}")

                    plot_profile(
                        das=das,
                        var=s.var_fc,
                        metric=metric,
                        select_value=period_value,
                        models=available_models,
                        labels=tuple(
                            MODEL_LABELS.get(model, model)
                            for model in available_models
                        ),
                        out_file=out_file,
                        time_range=valid_time_range,
                        das_member=das_member,
                        profile_dim=leadtime_agg_coord,
                        profile_label=LEADTIME_PROFILE_LABEL,
                        select_dim=period_dim,
                        select_unit=CLIM_PERIOD.value,
                        plot_single_members=PLOT_MEMBERS,
                        plot_legend=PLOT_LEGEND,
                        ylim=get_profile_ylim(
                            PROFILE_YLIMS,
                            metric,
                            variant="absolute",
                        ),
                        **profile_kwargs,
                    )

                    n += 1

                # ----------------------------------------------
                # Climatological-period profile
                # ----------------------------------------------

                if PLOT_INDIVIDUAL_CLIMATOLOGICAL_PROFILES:
                    climatological_das = [
                        prepare_climatological_profile_da(
                            da,
                            period_dim=period_dim,
                            clim_period=CLIM_PERIOD,
                        )
                        for da in das
                    ]

                    climatological_das_member = [
                        prepare_climatological_profile_da(
                            da,
                            period_dim=period_dim,
                            clim_period=CLIM_PERIOD,
                        )
                        for da in das_member
                    ]

                    leadtime_groups = (
                        [climatological_profile_leadtimes]
                        if COMBINE_CLIMATOLOGICAL_PROFILE_LEADTIMES
                        else [
                            [lead_value]
                            for lead_value in climatological_profile_leadtimes
                        ]
                    )

                    for lead_values in leadtime_groups:

                        combined = len(lead_values) > 1

                        leadtime_label = (
                            "all_leadtimes"
                            if combined
                            else f"leadtime_{safe_label(lead_values[0])}"
                        )

                        common_path = (
                            Path("profiles")
                            / "climatology"
                            / "absolute"
                            / leadtime_label
                            / (
                                f"time_"
                                f"{safe_label(valid_time_range)}"
                                f"_lat_"
                                f"{safe_label(valid_lat_range)}"
                                f"_lon_"
                                f"{safe_label(valid_lon_range)}"
                            )
                            / metric
                            / METRIC_AGG_MODE
                            / IMPROVEMENT_AGG_MODE
                        )

                        filename = (
                            f"{s.var_fc}_"
                            f"{metric}_"
                            f"{safe_label(period_dim)}_"
                            f"{leadtime_label}.png"
                        )

                        out_file = (
                            s.plot_dir
                            / common_path
                            / filename
                        )

                        if (
                            out_file.exists()
                            and not REGENERATE_PLOTS
                        ):
                            continue

                        print(f"Saving climatological profile {out_file}")

                        plot_das = climatological_das
                        plot_models_current = available_models
                        plot_labels_current = tuple(
                            MODEL_LABELS.get(model, model)
                            for model in available_models
                        )
                        plot_das_member = climatological_das_member
                        plot_colors = None
                        plot_linestyles = None
                        plot_select_dim = leadtime_agg_coord
                        plot_select_value = lead_values[0]
                        plot_select_unit = LEADTIME_UNITS.value

                        if combined:
                            (
                                plot_das,
                                plot_models_current,
                                plot_labels_current,
                                plot_das_member,
                                plot_colors,
                                plot_linestyles,
                            ) = expand_climatological_leadtimes(
                                climatological_das,
                                available_models,
                                lead_values,
                                leadtime_dim=leadtime_agg_coord,
                                leadtime_unit=LEADTIME_UNITS.value,
                                color_mode=CLIMATOLOGICAL_LEADTIME_COLORS,
                                labels=plot_labels_current,
                                das_member=climatological_das_member,
                            )

                            plot_select_dim = None
                            plot_select_value = "all"
                            plot_select_unit = None

                        plot_profile(
                            das=plot_das,
                            var=s.var_fc,
                            metric=metric,
                            models=plot_models_current,
                            labels=plot_labels_current,
                            out_file=out_file,
                            time_range=valid_time_range,
                            das_member=plot_das_member,
                            profile_dim=period_dim,
                            profile_label=period_profile_label,
                            select_dim=plot_select_dim,
                            select_value=plot_select_value,
                            select_unit=plot_select_unit,
                            plot_single_members=PLOT_MEMBERS,
                            plot_legend=PLOT_LEGEND,
                            model_colors=plot_colors,
                            model_linestyles=plot_linestyles,
                            ylim=get_profile_ylim(
                                PROFILE_YLIMS,
                                metric,
                                variant="absolute",
                            ),
                            **profile_kwargs,
                        )

                        n += 1

            # ==================================================
            # Model-comparison profiles
            # ==================================================

            for (
                baseline_model,
                target_model,
            ) in MODEL_COMPARISONS:

                if (
                    baseline_model
                    not in metrics_by_model
                ):
                    continue

                if (
                    target_model
                    not in metrics_by_model
                ):
                    continue

                baseline_ds = comparison_source_by_model[baseline_model]
                target_ds = comparison_source_by_model[target_model]

                comparison_metrics = [
                    metric
                    for metric in METRICS
                    if (
                        metric != "rank_histogram"
                        and metric in baseline_ds
                        and metric in target_ds
                    )
                ]

                for metric in comparison_metrics:

                    improvements = get_aggregated_metric_improvements(
                        baseline_ds,
                        target_ds,
                        metric=metric,
                        baseline_model=baseline_model,
                        target_model=target_model,
                        an=an,
                        metric_agg_mode=METRIC_AGG_MODE,
                        improvement_agg_mode=IMPROVEMENT_AGG_MODE,
                    )

                    for comparison_model, comparison_da in improvements.items():

                        (
                            variant,
                            improvement_unit,
                            comparison_label,
                        ) = get_comparison_plot_meta(
                            baseline_model,
                            target_model,
                            comparison_model,
                            COMPARISON_LABELS_MAP,
                        )

                        for period_value in leadtime_profile_periods:

                            common_path = (
                                Path("profiles")
                                / "comparisons"
                                / safe_label(
                                    comparison_model
                                )
                                / safe_label(period_dim)
                                / safe_label(
                                    period_value
                                )
                                / (
                                    f"time_"
                                    f"{safe_label(valid_time_range)}"
                                    f"_lat_"
                                    f"{safe_label(valid_lat_range)}"
                                    f"_lon_"
                                    f"{safe_label(valid_lon_range)}"
                                )
                                / metric
                                / METRIC_AGG_MODE
                                / IMPROVEMENT_AGG_MODE
                            )

                            filename = (
                                f"{s.var_fc}_"
                                f"{metric}_"
                                f"{safe_label(comparison_model)}_"
                                f"{LEADTIME_AGG_MODE}lt.png"
                            )

                            out_file = (
                                s.plot_dir
                                / common_path
                                / filename
                            )

                            if (
                                out_file.exists()
                                and not REGENERATE_PLOTS
                            ):
                                continue

                            print(f"Saving comparison profile {out_file}")

                            plot_profile(
                                das=[
                                    comparison_da
                                ],
                                var=s.var_fc,
                                metric=metric,
                                models=(
                                    comparison_model,
                                ),
                                labels=(comparison_label,),
                                out_file=out_file,
                                time_range=valid_time_range,
                                das_member=[None],
                                profile_dim=leadtime_agg_coord,
                                profile_label=LEADTIME_PROFILE_LABEL,
                                select_value=period_value,
                                select_dim=period_dim,
                                select_unit=CLIM_PERIOD.value,
                                plot_single_members=False,
                                plot_legend=PLOT_LEGEND,
                                improvement_unit=improvement_unit,
                                ylim=get_profile_ylim(
                                    PROFILE_YLIMS,
                                    metric,
                                    variant=variant,
                                ),
                                **profile_kwargs,
                            )

                            n += 1

                        # --------------------------------------
                        # Climatological-period profile
                        # --------------------------------------

                        if PLOT_INDIVIDUAL_CLIMATOLOGICAL_PROFILES:
                            climatological_comparison_da = (
                                prepare_climatological_profile_da(
                                    comparison_da,
                                    period_dim=period_dim,
                                    clim_period=CLIM_PERIOD,
                                )
                            )

                            leadtime_groups = (
                                [climatological_profile_leadtimes]
                                if COMBINE_CLIMATOLOGICAL_PROFILE_LEADTIMES
                                else [
                                    [lead_value]
                                    for lead_value in climatological_profile_leadtimes
                                ]
                            )

                            for lead_values in leadtime_groups:

                                combined = len(lead_values) > 1

                                leadtime_label = (
                                    "all_leadtimes"
                                    if combined
                                    else f"leadtime_{safe_label(lead_values[0])}"
                                )

                                common_path = (
                                    Path("profiles")
                                    / "climatology"
                                    / "comparisons"
                                    / safe_label(
                                        comparison_model
                                    )
                                    / leadtime_label
                                    / (
                                        f"time_"
                                        f"{safe_label(valid_time_range)}"
                                        f"_lat_"
                                        f"{safe_label(valid_lat_range)}"
                                        f"_lon_"
                                        f"{safe_label(valid_lon_range)}"
                                    )
                                    / metric
                                    / METRIC_AGG_MODE
                                    / IMPROVEMENT_AGG_MODE
                                )

                                filename = (
                                    f"{s.var_fc}_"
                                    f"{metric}_"
                                    f"{safe_label(comparison_model)}_"
                                    f"{safe_label(period_dim)}_"
                                    f"{leadtime_label}.png"
                                )

                                out_file = (
                                    s.plot_dir
                                    / common_path
                                    / filename
                                )

                                if (
                                    out_file.exists()
                                    and not REGENERATE_PLOTS
                                ):
                                    continue

                                print(f"Saving climatological comparison profile {out_file}")

                                plot_das = [
                                    climatological_comparison_da
                                ]
                                plot_models_current = (
                                    comparison_model,
                                )
                                plot_labels_current = (comparison_label,)
                                plot_colors = None
                                plot_linestyles = None
                                plot_select_dim = leadtime_agg_coord
                                plot_select_value = lead_values[0]
                                plot_select_unit = LEADTIME_UNITS.value

                                if combined:
                                    (
                                        plot_das,
                                        plot_models_current,
                                        plot_labels_current,
                                        _,
                                        plot_colors,
                                        plot_linestyles,
                                    ) = expand_climatological_leadtimes(
                                        plot_das,
                                        plot_models_current,
                                        lead_values,
                                        leadtime_dim=leadtime_agg_coord,
                                        leadtime_unit=LEADTIME_UNITS.value,
                                        color_mode=CLIMATOLOGICAL_LEADTIME_COLORS,
                                        labels=plot_labels_current,
                                    )

                                    plot_select_dim = None
                                    plot_select_value = "all"
                                    plot_select_unit = None


                                plot_profile(
                                    das=plot_das,
                                    var=s.var_fc,
                                    metric=metric,
                                    models=plot_models_current,
                                    labels=plot_labels_current,
                                    out_file=out_file,
                                    time_range=valid_time_range,
                                    das_member=[None] * len(plot_das),
                                    profile_dim=period_dim,
                                    profile_label=period_profile_label,
                                    select_dim=plot_select_dim,
                                    select_value=plot_select_value,
                                    select_unit=plot_select_unit,
                                    plot_single_members=False,
                                    plot_legend=PLOT_LEGEND,
                                    improvement_unit=improvement_unit,
                                    model_colors=plot_colors,
                                    model_linestyles=plot_linestyles,
                                    ylim=get_profile_ylim(
                                        PROFILE_YLIMS,
                                        metric,
                                        variant=variant,
                                    ),
                                    **profile_kwargs,
                                )

                                n += 1

        # ======================================================
        # Accumulate combined variables
        # ======================================================

        if plot_combined_variables:

            combined_variable_metrics = None
            combined_variable_variant = "absolute"
            combined_variable_improvement_unit = None

            if COMBINED_VARIABLE_SOURCE in metrics_by_model:
                combined_variable_metrics = metrics_by_model[
                    COMBINED_VARIABLE_SOURCE
                ]

            elif COMBINED_VARIABLE_IS_IMPROVEMENT:
                requested_variant = COMBINED_VARIABLE_SOURCE.removeprefix(
                    "mlfc-fc-"
                )

                if requested_variant not in IMPROVEMENT_UNITS:
                    raise ValueError(
                        "Unsupported combined-variable improvement "
                        f"variant {requested_variant!r}."
                    )

                if {"fc", "mlfc"} <= comparison_source_by_model.keys():
                    combined_variable_parts = {}

                    for metric in METRICS:
                        if (
                            metric == "rank_histogram"
                            or metric not in comparison_source_by_model["fc"]
                            or metric not in comparison_source_by_model["mlfc"]
                        ):
                            continue

                        improvements = get_aggregated_metric_improvements(
                            comparison_source_by_model["fc"],
                            comparison_source_by_model["mlfc"],
                            metric=metric,
                            baseline_model="fc",
                            target_model="mlfc",
                            an=an,
                            metric_agg_mode=METRIC_AGG_MODE,
                            improvement_agg_mode=IMPROVEMENT_AGG_MODE,
                        )

                        matching = [
                            da
                            for name, da in improvements.items()
                            if get_improvement_variant(name) == requested_variant
                        ]

                        if matching:
                            combined_variable_parts[metric] = (
                                matching[0]
                            )

                    if combined_variable_parts:
                        combined_variable_metrics = xr.Dataset(
                            combined_variable_parts
                        )
                        combined_variable_variant = requested_variant
                        combined_variable_improvement_unit = (
                            IMPROVEMENT_UNITS[requested_variant]
                        )

            else:
                raise ValueError(
                    "Unsupported combined_variable_source="
                    f"{COMBINED_VARIABLE_SOURCE!r}. Expected a raw model "
                    '("fc", "mlfc", "clim-fc") or one of '
                    '"mlfc-fc-difference", "mlfc-fc-percentage", '
                    '"mlfc-fc-normalized".'
                )

            if combined_variable_metrics is not None:
                combined_variable_key = (
                    s.region_name,
                    safe_label(valid_time_range),
                    safe_label(valid_lat_range),
                    safe_label(valid_lon_range),
                    tuple(s.leadtimes),
                    LEADTIME_AGG_MODE,
                    METRIC_AGG_MODE,
                    str(CLIM_PERIOD),
                    PERIOD_REFERENCE,
                    combined_variable_variant,
                    combined_variable_improvement_unit,
                )

                variable_group = combined_variable_groups[
                    combined_variable_key
                ]

                variable_group["valid_time_range"] = valid_time_range
                variable_group["valid_lat_range"] = valid_lat_range
                variable_group["valid_lon_range"] = valid_lon_range
                variable_group["period_dim"] = period_dim
                variable_group["leadtime_agg_coord"] = leadtime_agg_coord
                variable_group["variant"] = combined_variable_variant
                variable_group["improvement_unit"] = (
                    combined_variable_improvement_unit
                )

                if s.var_fc in variable_group["metrics"]:
                    raise ValueError(
                        "Combined-variable plotting found more than one "
                        f"matching experiment for variable {s.var_fc!r} "
                        "within the same plotting group. Narrow the "
                        "experiment selection so there is one experiment "
                        "per variable."
                    )

                variable_group["metrics"][s.var_fc] = (
                    combined_variable_metrics
                )

        # ======================================================
        # Accumulate combined experiments
        # ======================================================

        if (
            plot_combined_experiments
            and "mlfc" in metrics_by_model
        ):
            comparison_key = (
                s.var_fc,
                s.region_name,
                safe_label(
                    valid_time_range
                ),
                safe_label(
                    valid_lat_range
                ),
                safe_label(
                    valid_lon_range
                ),
                tuple(s.leadtimes),
                LEADTIME_AGG_MODE,
                METRIC_AGG_MODE,
                str(CLIM_PERIOD),
                PERIOD_REFERENCE,
            )

            group = combined_groups[
                comparison_key
            ]

            group[
                "valid_time_range"
            ] = valid_time_range

            group[
                "valid_lat_range"
            ] = valid_lat_range

            group[
                "valid_lon_range"
            ] = valid_lon_range

            group[
                "period_dim"
            ] = period_dim

            group[
                "leadtime_agg_coord"
            ] = leadtime_agg_coord

            if group["spatial_reference"] is None:
                group["spatial_reference"] = an

            # FC is common to experiments in the same group.
            if (
                group["fc"] is None
                and "fc" in metrics_by_model
            ):
                group["fc"] = (
                    metrics_by_model["fc"]
                )

                group["fc_fields"] = (
                    metric_fields_by_model["fc"]
                )

                group["fc_members"] = (
                    members_by_model.get(
                        "fc"
                    )
                )

            if (
                need_clim_fc
                and group["clim-fc"] is None
                and "clim-fc"
                in metrics_by_model
            ):
                group["clim-fc"] = (
                    metrics_by_model[
                        "clim-fc"
                    ]
                )

                group["clim-fc_fields"] = (
                    metric_fields_by_model[
                        "clim-fc"
                    ]
                )

                group[
                    "clim-fc_members"
                ] = members_by_model.get(
                    "clim-fc"
                )

            group["mlfc"].append(
                metrics_by_model[
                    "mlfc"
                ]
            )

            group["mlfc_fields"].append(
                metric_fields_by_model[
                    "mlfc"
                ]
            )

            group[
                "mlfc_members"
            ].append(
                members_by_model.get(
                    "mlfc"
                )
            )

            group["labels"].append(
                s.output_name
            )

            group["settings"].append(
                s
            )

    # ==========================================================
    # Combined-variable profile plots
    # ==========================================================

    if plot_combined_variables:

        for variable_group in combined_variable_groups.values():

            metrics_by_variable = variable_group["metrics"]

            if not metrics_by_variable:
                continue

            variables_current = tuple(
                var
                for var in VARIABLES
                if var in metrics_by_variable
            )

            if not variables_current:
                continue

            valid_time_range = variable_group[
                "valid_time_range"
            ]
            valid_lat_range = variable_group[
                "valid_lat_range"
            ]
            valid_lon_range = variable_group[
                "valid_lon_range"
            ]
            period_dim = variable_group["period_dim"]
            leadtime_agg_coord = variable_group[
                "leadtime_agg_coord"
            ]
            combined_variable_variant = variable_group["variant"]
            combined_variable_improvement_unit = variable_group[
                "improvement_unit"
            ]

            reference_ds = metrics_by_variable[
                variables_current[0]
            ]

            available_periods = [
                str(value)
                for value in reference_ds[period_dim].values
                if str(value) in periods_requested
            ]

            available_metrics = [
                metric
                for metric in METRICS
                if (
                    metric != "rank_histogram"
                    and all(
                        metric in metrics_by_variable[var]
                        for var in variables_current
                    )
                )
            ]

            for metric in available_metrics:

                das = [
                    metrics_by_variable[var][metric]
                    for var in variables_current
                ]

                labels = tuple(
                    VARIABLE_LABELS.get(var, var)
                    for var in variables_current
                )

                colors = {
                    var: VARIABLE_COLORS[var]
                    for var in variables_current
                    if var in VARIABLE_COLORS
                }

                # ----------------------------------------------
                # Lead-time profile
                # ----------------------------------------------

                if PLOT_COMBINED_VARIABLE_LEADTIME_PROFILES:

                    for period_value in available_periods:

                        common_path = (
                            Path("profiles")
                            / COMBINED_VARIABLE_PLOT_FOLDER
                            / safe_label(COMBINED_VARIABLE_SOURCE)
                            / safe_label(period_dim)
                            / safe_label(period_value)
                            / (
                                f"time_"
                                f"{safe_label(valid_time_range)}"
                                f"_lat_"
                                f"{safe_label(valid_lat_range)}"
                                f"_lon_"
                                f"{safe_label(valid_lon_range)}"
                            )
                            / metric
                            / METRIC_AGG_MODE
                            / IMPROVEMENT_AGG_MODE
                        )

                        filename = (
                            f"all_variables_"
                            f"{COMBINED_VARIABLE_SOURCE}_"
                            f"{metric}_"
                            f"{LEADTIME_AGG_MODE}lt.png"
                        )

                        out_file = (
                            COMMON_PLOT_DIR
                            / common_path
                            / filename
                        )

                        if (
                            out_file.exists()
                            and not REGENERATE_PLOTS
                        ):
                            continue

                        print(f"Saving combined-variable lead-time profile {out_file}")

                        plot_profile(
                            das=das,
                            var=variables_current[0],
                            metric=metric,
                            models=variables_current,
                            labels=labels,
                            out_file=out_file,
                            time_range=valid_time_range,
                            das_member=[None] * len(das),
                            profile_dim=leadtime_agg_coord,
                            profile_label=LEADTIME_PROFILE_LABEL,
                            select_value=period_value,
                            select_dim=period_dim,
                            select_unit=CLIM_PERIOD.value,
                            plot_single_members=False,
                            plot_legend=COMBINED_VARIABLE_PLOT_LEGEND,
                            improvement_unit=combined_variable_improvement_unit,
                            model_colors=colors,
                            ylim=get_profile_ylim(
                                PROFILE_YLIMS,
                                metric,
                                variant=combined_variable_variant,
                            ),
                            **combined_profile_kwargs,
                        )

                        n += 1

                # ----------------------------------------------
                # Climatological-period profile
                # ----------------------------------------------

                if PLOT_COMBINED_VARIABLE_CLIMATOLOGICAL_PROFILES:

                    climatological_das = [
                        prepare_climatological_profile_da(
                            da,
                            period_dim=period_dim,
                            clim_period=CLIM_PERIOD,
                        )
                        for da in das
                    ]

                    climatological_leadtimes = (
                        get_climatological_profile_leadtimes(
                            reference_ds,
                            leadtime_dim=leadtime_agg_coord,
                            wanted_leadtimes=(
                                COMBINED_VARIABLE_CLIMATOLOGICAL_LEADTIMES
                            ),
                        )
                    )

                    leadtime_groups = (
                        [climatological_leadtimes]
                        if COMBINE_COMBINED_VARIABLE_CLIMATOLOGICAL_LEADTIMES
                        else [
                            [lead_value]
                            for lead_value in climatological_leadtimes
                        ]
                    )

                    for lead_values in leadtime_groups:

                        if not lead_values:
                            continue

                        combined = len(lead_values) > 1

                        leadtime_label = (
                            "all_leadtimes"
                            if combined
                            else f"leadtime_{safe_label(lead_values[0])}"
                        )

                        plot_das = climatological_das
                        plot_models_current = variables_current
                        plot_labels_current = labels
                        plot_colors = colors
                        plot_linestyles = None
                        plot_select_dim = leadtime_agg_coord
                        plot_select_value = lead_values[0]
                        plot_select_unit = LEADTIME_UNITS.value

                        if combined:
                            (
                                plot_das,
                                plot_models_current,
                                plot_labels_current,
                                _,
                                plot_colors,
                                plot_linestyles,
                            ) = expand_climatological_leadtimes(
                                climatological_das,
                                variables_current,
                                lead_values,
                                leadtime_dim=leadtime_agg_coord,
                                leadtime_unit=LEADTIME_UNITS.value,
                                color_mode=COMBINED_VARIABLE_CLIMATOLOGICAL_LEADTIME_COLORS,
                                labels=labels,
                                model_colors=colors,
                            )

                            plot_select_dim = None
                            plot_select_value = "all"
                            plot_select_unit = None

                        common_path = (
                            Path("profiles")
                            / COMBINED_VARIABLE_PLOT_FOLDER
                            / safe_label(COMBINED_VARIABLE_SOURCE)
                            / "climatology"
                            / leadtime_label
                            / (
                                f"time_"
                                f"{safe_label(valid_time_range)}"
                                f"_lat_"
                                f"{safe_label(valid_lat_range)}"
                                f"_lon_"
                                f"{safe_label(valid_lon_range)}"
                            )
                            / metric
                            / METRIC_AGG_MODE
                            / IMPROVEMENT_AGG_MODE
                        )

                        filename = (
                            f"all_variables_"
                            f"{COMBINED_VARIABLE_SOURCE}_"
                            f"{metric}_"
                            f"{safe_label(period_dim)}_"
                            f"{leadtime_label}.png"
                        )

                        out_file = (
                            COMMON_PLOT_DIR
                            / common_path
                            / filename
                        )

                        if (
                            out_file.exists()
                            and not REGENERATE_PLOTS
                        ):
                            continue

                        print(f"Saving combined-variable climatological profile {out_file}")

                        plot_profile(
                            das=plot_das,
                            var=variables_current[0],
                            metric=metric,
                            models=plot_models_current,
                            labels=plot_labels_current,
                            out_file=out_file,
                            time_range=valid_time_range,
                            das_member=[None] * len(plot_das),
                            profile_dim=period_dim,
                            profile_label=period_profile_label,
                            select_dim=plot_select_dim,
                            select_value=plot_select_value,
                            select_unit=plot_select_unit,
                            plot_single_members=False,
                            plot_legend=COMBINED_VARIABLE_PLOT_LEGEND,
                            improvement_unit=combined_variable_improvement_unit,
                            model_colors=plot_colors,
                            model_linestyles=plot_linestyles,
                            ylim=get_profile_ylim(
                                PROFILE_YLIMS,
                                metric,
                                variant=combined_variable_variant,
                            ),
                            **combined_profile_kwargs,
                        )

                        n += 1

    # ==========================================================
    # Combined experiment profile plots
    # ==========================================================

    if plot_combined_experiments:

        for (
            comparison_key,
            group,
        ) in combined_groups.items():

            if not group["mlfc"]:
                continue

            common_s = group["settings"][0]

            group_comparison_labels = [
                comparison_labels_by_experiment[s.exp_dir.name]
                for s in group["settings"]
            ]

            print("\nCombined experiment mapping:")

            for label, s, ds in zip(
                group_comparison_labels,
                group["settings"],
                group["mlfc"],
                strict=True,
            ):
                print(
                    f"{label:>15} <- "
                    f"{s.exp_dir.name}"
                )

            valid_time_range = (
                group[
                    "valid_time_range"
                ]
            )

            valid_lat_range = (
                group[
                    "valid_lat_range"
                ]
            )

            valid_lon_range = (
                group[
                    "valid_lon_range"
                ]
            )

            period_dim = (
                group["period_dim"]
            )

            leadtime_agg_coord = (
                group[
                    "leadtime_agg_coord"
                ]
            )

            # ==================================================
            # Build combined model dictionary
            # ==================================================

            das_by_model: dict[
                str,
                xr.Dataset,
            ] = {}

            fields_by_model: dict[
                str,
                xr.Dataset,
            ] = {}

            members_by_combined_model: dict[
                str,
                xr.Dataset,
            ] = {}


            if group["fc"] is not None:
                das_by_model["fc"] = (
                    group["fc"]
                )
                fields_by_model["fc"] = (
                    group["fc_fields"]
                )

                if (
                    group["fc_members"]
                    is not None
                ):
                    members_by_combined_model[
                        "fc"
                    ] = group[
                        "fc_members"
                    ]

            if (
                need_clim_fc
                and group["clim-fc"]
                is not None
            ):
                das_by_model[
                    "clim-fc"
                ] = group[
                    "clim-fc"
                ]
                fields_by_model[
                    "clim-fc"
                ] = group[
                    "clim-fc_fields"
                ]

                if (
                    group[
                        "clim-fc_members"
                    ]
                    is not None
                ):
                    members_by_combined_model[
                        "clim-fc"
                    ] = group[
                        "clim-fc_members"
                    ]

            if (
                len(group_comparison_labels)
                != len(group["mlfc"])
            ):
                raise ValueError(
                    "group_comparison_labels must contain "
                    "one label for each combined "
                    "experiment. "
                    f"Got {len(group_comparison_labels)} "
                    f"labels for "
                    f"{len(group['mlfc'])} "
                    "experiments."
                )

            for (
                experiment_label,
                mlfc_metrics,
                mlfc_fields,
                mlfc_members,
            ) in zip(
                group_comparison_labels,
                group["mlfc"],
                group["mlfc_fields"],
                group["mlfc_members"],
                strict=True,
            ):
                das_by_model[
                    experiment_label
                ] = mlfc_metrics
                fields_by_model[
                    experiment_label
                ] = mlfc_fields

                if mlfc_members is not None:
                    members_by_combined_model[
                        experiment_label
                    ] = mlfc_members

            comparison_source_by_model = fields_by_model

            # ==================================================
            # Available periods
            # ==================================================

            reference_ds = next(
                iter(das_by_model.values())
            )

            available_periods = [
                str(value)
                for value
                in reference_ds[
                    period_dim
                ].values
                if str(value)
                in periods_requested
            ]

            leadtime_profile_periods = (
                available_periods
                if PLOT_COMBINED_EXPERIMENT_LEADTIME_PROFILES
                else []
            )

            experiment_axis_leadtimes = (
                get_climatological_profile_leadtimes(
                    reference_ds,
                    leadtime_dim=leadtime_agg_coord,
                    wanted_leadtimes=(
                        COMBINED_EXPERIMENT_AXIS_LEADTIMES
                    ),
                )
                if PLOT_COMBINED_EXPERIMENT_AXIS_PROFILES
                else []
            )

            combined_experiment_climatological_profile_leadtimes = (
                get_climatological_profile_leadtimes(
                    reference_ds,
                    leadtime_dim=leadtime_agg_coord,
                    wanted_leadtimes=(
                        COMBINED_EXPERIMENT_CLIMATOLOGICAL_LEADTIMES
                    ),
                )
                if PLOT_COMBINED_EXPERIMENT_CLIMATOLOGICAL_PROFILES
                else []
            )

            # ==================================================
            # Absolute combined profiles
            # ==================================================

            available_models = tuple(
                model
                for model in das_by_model
                if (
                    model in PLOT_MODELS
                    or (
                        "mlfc" in PLOT_MODELS
                        and model in group_comparison_labels
                    )
                )
            )

            available_metrics = [
                metric
                for metric in METRICS
                if (
                    metric != "rank_histogram"
                    and all(
                        metric
                        in das_by_model[
                            model
                        ]
                        for model
                        in available_models
                    )
                )
            ]

            print(
                "Plotting combined experiment "
                f"profiles {available_metrics} "
                f"for periods {available_periods}: "
                f"{group['labels']}"
            )

            for metric in available_metrics:

                das = [
                    das_by_model[
                        model
                    ][metric]
                    for model
                    in available_models
                ]

                das_member = []

                for model in available_models:

                    member_ds = (
                        members_by_combined_model.get(
                            model
                        )
                    )

                    if (
                        member_ds is not None
                        and metric in member_ds
                    ):
                        das_member.append(
                            member_ds[metric]
                        )

                    else:
                        das_member.append(None)

                # ----------------------------------------------
                # Experiment-axis profile
                # ----------------------------------------------
                if PLOT_COMBINED_EXPERIMENT_AXIS_PROFILES:
                    leadtime_groups = (
                        [experiment_axis_leadtimes]
                        if COMBINE_COMBINED_EXPERIMENT_AXIS_LEADTIMES
                        else [[lead] for lead in experiment_axis_leadtimes]
                    )

                    for period_value in available_periods:
                        for lead_values in leadtime_groups:
                            if not lead_values:
                                continue
                            combined = len(lead_values) > 1
                            leadtime_label = (
                                "all_leadtimes" if combined
                                else f"leadtime_{safe_label(lead_values[0])}"
                            )

                            experiment_entries = [
                                (model, das_by_model[model][metric])
                                for model in group_comparison_labels
                                if model in das_by_model
                            ]
                            plot_das, plot_models, plot_labels, plot_colors = (
                                build_experiment_axis_leadtime_curves(
                                    experiment_entries,
                                    lead_values=lead_values,
                                    period_dim=period_dim,
                                    period_value=period_value,
                                    leadtime_dim=leadtime_agg_coord,
                                    leadtime_unit=LEADTIME_UNITS.value,
                                    sort_numeric=COMBINED_EXPERIMENT_AXIS_SORT_NUMERIC,
                                    color_mode=COMBINED_EXPERIMENT_AXIS_LEADTIME_COLORS,
                                    shade_range=COMBINED_EXPERIMENT_AXIS_LEADTIME_SHADE_RANGE,
                                )
                            )

                            # Add constant-valued baseline curves over the
                            # experiment axis, separately for each lead time.
                            plot_das = list(plot_das)
                            plot_models = list(plot_models)
                            plot_labels = list(plot_labels)
                            plot_colors = dict(plot_colors)
                            plot_linestyles = {model: "-" for model in plot_models}

                            for reference_model in COMBINED_EXPERIMENT_AXIS_REFERENCE_LINES:
                                if reference_model not in ("fc", "clim-fc"):
                                    raise ValueError(
                                        f"Unsupported experiment-axis reference: {reference_model!r}"
                                    )
                                reference_ds = das_by_model.get(reference_model)
                                if reference_ds is None or metric not in reference_ds:
                                    continue

                                for index, lead_value in enumerate(lead_values):
                                    reference_value = (
                                        reference_ds[metric]
                                        .sel({period_dim: period_value, leadtime_agg_coord: lead_value})
                                        .squeeze(drop=True)
                                    )
                                    if reference_value.size != 1:
                                        raise ValueError(
                                            f"Expected scalar {reference_model} reference at "
                                            f"lead time {lead_value}, got {reference_value.shape}"
                                        )
                                    reference_value = float(reference_value.compute().item())
                                    reference_key = f"{reference_model}_leadtime_{index}"
                                    plot_das.append(
                                        xr.full_like(plot_das[index], reference_value)
                                    )
                                    plot_models.append(reference_key)
                                    plot_labels.append(
                                        f"{MODEL_LABELS.get(reference_model, reference_model)} "
                                        f"· {plot_labels[index]}"
                                    )
                                    plot_colors[reference_key] = plot_colors[f"leadtime_{index}"]
                                    plot_linestyles[reference_key] = (
                                        "--" if reference_model == "fc" else ":"
                                    )

                            has_references = len(plot_models) > len(lead_values)

                            common_path = (
                                Path("profiles")
                                / COMBINED_PLOT_FOLDER
                                / COMPARISON_NAME
                                / "experiment_axis"
                                / "absolute"
                                / safe_label(period_dim)
                                / safe_label(period_value)
                                / leadtime_label
                                / (
                                    f"time_{safe_label(valid_time_range)}"
                                    f"_lat_{safe_label(valid_lat_range)}"
                                    f"_lon_{safe_label(valid_lon_range)}"
                                )
                                / metric
                                / METRIC_AGG_MODE
                                / IMPROVEMENT_AGG_MODE
                            )
                            filename = (
                                f"{common_s.var_fc}_{metric}_experiment_axis_"
                                f"{leadtime_label}.png"
                            )
                            out_file = COMMON_PLOT_DIR / common_path / filename
                            if out_file.exists() and not REGENERATE_PLOTS:
                                continue
                            print(f"Saving combined experiment-axis profile {out_file}")

                            plot_profile(
                                das=plot_das if combined or has_references else plot_das[0],
                                var=common_s.var_fc,
                                metric=metric,
                                models=plot_models if combined or has_references else "mlfc",
                                labels=plot_labels if combined or has_references else "MLFC",
                                out_file=out_file,
                                time_range=valid_time_range,
                                profile_dim="experiment",
                                profile_label=(
                                    COMBINED_EXPERIMENT_AXIS_LABEL
                                    if COMBINED_EXPERIMENT_AXIS_LABEL is not None
                                    else COMPARISON_NAME.replace("_", " ")
                                ),
                                select_dim=None,
                                select_value="all",
                                profile_markers=(
                                    {
                                        model: not model.startswith(
                                            ("fc_leadtime_", "clim-fc_leadtime_")
                                        )
                                        for model in plot_models
                                    }
                                    if combined or has_references else True
                                ),
                                profile_point_colors=None if combined else comparison_colors,
                                model_colors=plot_colors if combined or has_references else None,
                                model_linestyles=plot_linestyles if has_references else None,
                                profile_connect=COMBINED_EXPERIMENT_AXIS_CONNECT,
                                profile_tick_rotation=COMBINED_EXPERIMENT_AXIS_TICK_ROTATION,
                                plot_legend=combined or has_references,
                                ylim=get_profile_ylim(
                                    PROFILE_YLIMS, metric, variant="absolute",
                                ),
                                **plot_kwargs,
                            )
                            n += 1

                for period_value in leadtime_profile_periods:

                    common_path = (
                        Path("profiles")
                        / COMBINED_PLOT_FOLDER
                        / COMPARISON_NAME
                        / "absolute"
                        / safe_label(period_dim)
                        / safe_label(
                            period_value
                        )
                        / (
                            f"time_"
                            f"{safe_label(valid_time_range)}"
                            f"_lat_"
                            f"{safe_label(valid_lat_range)}"
                            f"_lon_"
                            f"{safe_label(valid_lon_range)}"
                        )
                        / metric
                        / METRIC_AGG_MODE
                        / IMPROVEMENT_AGG_MODE
                    )

                    filename = (
                        f"{common_s.var_fc}_"
                        f"{metric}_"
                        f"{LEADTIME_AGG_MODE}lt.png"
                    )

                    out_file = (
                        COMMON_PLOT_DIR
                        / common_path
                        / filename
                    )

                    if (
                        out_file.exists()
                        and not REGENERATE_PLOTS
                    ):
                        continue

                    print(f"Saving combined profile {out_file}")

                    plot_profile(
                        das=das,
                        var=common_s.var_fc,
                        metric=metric,
                        models=available_models,
                        labels=tuple(
                            MODEL_LABELS.get(model, model)
                            for model in available_models
                        ),
                        out_file=out_file,
                        time_range=valid_time_range,
                        das_member=das_member,
                        profile_dim=leadtime_agg_coord,
                        profile_label=LEADTIME_PROFILE_LABEL,
                        select_value=period_value,
                        select_dim=period_dim,
                        select_unit=CLIM_PERIOD.value,
                        plot_single_members=PLOT_MEMBERS,
                        plot_legend=PLOT_LEGEND,
                        model_colors=comparison_colors,
                        model_linestyles=comparison_linestyles,
                        ylim=get_profile_ylim(
                            PROFILE_YLIMS,
                            metric,
                            variant="absolute",
                        ),
                        **combined_profile_kwargs,
                    )

                    n += 1

                # ----------------------------------------------
                # Climatological-period profile
                # ----------------------------------------------

                if PLOT_COMBINED_EXPERIMENT_CLIMATOLOGICAL_PROFILES:
                    climatological_das = [
                        prepare_climatological_profile_da(
                            da,
                            period_dim=period_dim,
                            clim_period=CLIM_PERIOD,
                        )
                        for da in das
                    ]

                    climatological_das_member = [
                        prepare_climatological_profile_da(
                            da,
                            period_dim=period_dim,
                            clim_period=CLIM_PERIOD,
                        )
                        for da in das_member
                    ]

                    leadtime_groups = (
                        [combined_experiment_climatological_profile_leadtimes]
                        if COMBINE_COMBINED_EXPERIMENT_CLIMATOLOGICAL_LEADTIMES
                        else [
                            [lead_value]
                            for lead_value in combined_experiment_climatological_profile_leadtimes
                        ]
                    )

                    for lead_values in leadtime_groups:

                        combined = len(lead_values) > 1

                        leadtime_label = (
                            "all_leadtimes"
                            if combined
                            else f"leadtime_{safe_label(lead_values[0])}"
                        )

                        common_path = (
                            Path("profiles")
                            / COMBINED_PLOT_FOLDER
                            / COMPARISON_NAME
                            / "climatology"
                            / "absolute"
                            / leadtime_label
                            / (
                                f"time_"
                                f"{safe_label(valid_time_range)}"
                                f"_lat_"
                                f"{safe_label(valid_lat_range)}"
                                f"_lon_"
                                f"{safe_label(valid_lon_range)}"
                            )
                            / metric
                            / METRIC_AGG_MODE
                            / IMPROVEMENT_AGG_MODE
                        )

                        filename = (
                            f"{common_s.var_fc}_"
                            f"{metric}_"
                            f"{safe_label(period_dim)}_"
                            f"{leadtime_label}.png"
                        )

                        out_file = (
                            COMMON_PLOT_DIR
                            / common_path
                            / filename
                        )

                        if (
                            out_file.exists()
                            and not REGENERATE_PLOTS
                        ):
                            continue

                        print(f"Saving combined climatological profile {out_file}")

                        plot_das = climatological_das
                        plot_models_current = available_models
                        plot_labels_current = tuple(
                            MODEL_LABELS.get(model, model)
                            for model in available_models
                        )
                        plot_das_member = climatological_das_member
                        plot_colors = comparison_colors
                        plot_linestyles = comparison_linestyles
                        plot_select_dim = leadtime_agg_coord
                        plot_select_value = lead_values[0]
                        plot_select_unit = LEADTIME_UNITS.value

                        if combined:
                            (
                                plot_das,
                                plot_models_current,
                                plot_labels_current,
                                plot_das_member,
                                plot_colors,
                                plot_linestyles,
                            ) = expand_climatological_leadtimes(
                                climatological_das,
                                available_models,
                                lead_values,
                                leadtime_dim=leadtime_agg_coord,
                                leadtime_unit=LEADTIME_UNITS.value,
                                color_mode=COMBINED_EXPERIMENT_CLIMATOLOGICAL_LEADTIME_COLORS,
                                shade_range=COMBINED_EXPERIMENT_CLIMATOLOGICAL_LEADTIME_SHADE_RANGE,
                                labels=plot_labels_current,
                                das_member=climatological_das_member,
                                model_colors=comparison_colors,
                                model_linestyles=comparison_linestyles,
                            )

                            plot_select_dim = None
                            plot_select_value = "all"
                            plot_select_unit = None

                        plot_profile(
                            das=plot_das,
                            var=common_s.var_fc,
                            metric=metric,
                            models=plot_models_current,
                            labels=plot_labels_current,
                            out_file=out_file,
                            time_range=valid_time_range,
                            das_member=plot_das_member,
                            profile_dim=period_dim,
                            profile_label=period_profile_label,
                            select_dim=plot_select_dim,
                            select_value=plot_select_value,
                            select_unit=plot_select_unit,
                            plot_single_members=PLOT_MEMBERS,
                            plot_legend=PLOT_LEGEND,
                            model_colors=plot_colors,
                            model_linestyles=plot_linestyles,
                            ylim=get_profile_ylim(
                                PROFILE_YLIMS,
                                metric,
                                variant="absolute",
                            ),
                            **combined_profile_kwargs,
                        )

                        n += 1

            # ==================================================
            # Combined comparison profiles
            #
            # "mlfc" in model_comparisons represents each
            # selected experiment independently.
            # ==================================================

            for (
                baseline_model,
                target_model,
            ) in MODEL_COMPARISONS:

                # ----------------------------------------------
                # Multiple ML experiments against common
                # baseline.
                # ----------------------------------------------

                if target_model == "mlfc":

                    if (
                        baseline_model
                        not in das_by_model
                    ):
                        continue

                    baseline_ds = (
                        comparison_source_by_model[
                            baseline_model
                        ]
                    )

                    for metric in METRICS:

                        if (
                            metric
                            == "rank_histogram"
                        ):
                            continue

                        if metric not in baseline_ds:
                            continue

                        comparison_by_variant: dict[
                            str,
                            list[
                                tuple[
                                    str,
                                    xr.DataArray,
                                ]
                            ],
                        ] = {}


                        for (
                            experiment_label
                        ) in group_comparison_labels:

                            if (
                                experiment_label
                                not in das_by_model
                            ):
                                continue

                            target_ds = (
                                comparison_source_by_model[
                                    experiment_label
                                ]
                            )

                            if (
                                metric
                                not in target_ds
                            ):
                                continue

                            improvements = (
                                get_aggregated_metric_improvements(
                                    baseline_ds,
                                    target_ds,
                                    metric=metric,
                                    baseline_model=(
                                        baseline_model
                                    ),
                                    target_model=(
                                        experiment_label
                                    ),
                                    an=group["spatial_reference"],
                                    metric_agg_mode=METRIC_AGG_MODE,
                                    improvement_agg_mode=IMPROVEMENT_AGG_MODE,
                                )
                            )

                            for (
                                generated_name,
                                comparison_da,
                            ) in improvements.items():

                                variant = (
                                    get_improvement_variant(
                                        generated_name
                                    )
                                )

                                comparison_by_variant.setdefault(
                                    variant,
                                    [],
                                ).append(
                                    (
                                        experiment_label,
                                        comparison_da,
                                    )
                                )

                        for (
                            variant,
                            entries,
                        ) in (
                            comparison_by_variant.items()
                        ):

                            if not entries:
                                continue

                            comparison_models = tuple(
                                label
                                for label, _
                                in entries
                            )

                            comparison_das = [
                                da
                                for _, da
                                in entries
                            ]

                            comparison_variant = (
                                f"{target_model}_vs_"
                                f"{baseline_model}_"
                                f"{variant}"
                            )

                            improvement_unit = {
                                "percentage": "%",
                                "difference": "Δ",
                                "normalized": "normalized",
                            }[variant]

                            # ----------------------------------
                            # Experiment-axis comparison profile
                            # ----------------------------------

                            if PLOT_COMBINED_EXPERIMENT_AXIS_PROFILES:
                                leadtime_groups = (
                                    [experiment_axis_leadtimes]
                                    if COMBINE_COMBINED_EXPERIMENT_AXIS_LEADTIMES
                                    else [[lead] for lead in experiment_axis_leadtimes]
                                )
                                for period_value in available_periods:
                                    for lead_values in leadtime_groups:
                                        if not lead_values:
                                            continue
                                        combined = len(lead_values) > 1
                                        leadtime_label = (
                                            "all_leadtimes" if combined
                                            else f"leadtime_{safe_label(lead_values[0])}"
                                        )
                                        plot_das, plot_models, plot_labels, plot_colors = (
                                            build_experiment_axis_leadtime_curves(
                                                entries,
                                                lead_values=lead_values,
                                                period_dim=period_dim,
                                                period_value=period_value,
                                                leadtime_dim=leadtime_agg_coord,
                                                leadtime_unit=LEADTIME_UNITS.value,
                                                sort_numeric=COMBINED_EXPERIMENT_AXIS_SORT_NUMERIC,
                                                color_mode=COMBINED_EXPERIMENT_AXIS_LEADTIME_COLORS,
                                            )
                                        )

                                        common_path = (
                                            Path("profiles")
                                            / COMBINED_PLOT_FOLDER
                                            / COMPARISON_NAME
                                            / "experiment_axis"
                                            / "comparisons"
                                            / safe_label(comparison_variant)
                                            / safe_label(period_dim)
                                            / safe_label(period_value)
                                            / leadtime_label
                                            / (
                                                f"time_{safe_label(valid_time_range)}"
                                                f"_lat_{safe_label(valid_lat_range)}"
                                                f"_lon_{safe_label(valid_lon_range)}"
                                            )
                                            / metric
                                            / METRIC_AGG_MODE
                                            / IMPROVEMENT_AGG_MODE
                                        )

                                        filename = (
                                            f"{common_s.var_fc}_{metric}_"
                                            f"{safe_label(comparison_variant)}_"
                                            f"experiment_axis_{leadtime_label}.png"
                                        )

                                        out_file = COMMON_PLOT_DIR / common_path / filename
                                        if out_file.exists() and not REGENERATE_PLOTS:
                                            continue

                                        print(f"Saving combined comparison experiment-axis profile {out_file}")

                                        plot_profile(
                                            das=plot_das if combined else plot_das[0],
                                            var=common_s.var_fc,
                                            metric=metric,
                                            models=plot_models if combined else "mlfc",
                                            labels=plot_labels if combined else "MLFC",
                                            out_file=out_file,
                                            time_range=valid_time_range,
                                            profile_dim="experiment",
                                            profile_label=(
                                                COMBINED_EXPERIMENT_AXIS_LABEL
                                                if COMBINED_EXPERIMENT_AXIS_LABEL is not None
                                                else COMPARISON_NAME.replace("_", " ")
                                            ),
                                            select_dim=None,
                                            select_value="all",
                                            profile_markers=True,
                                            profile_point_colors=None if combined else comparison_colors,
                                            model_colors=plot_colors if combined else None,
                                            profile_connect=COMBINED_EXPERIMENT_AXIS_CONNECT,
                                            profile_tick_rotation=COMBINED_EXPERIMENT_AXIS_TICK_ROTATION,
                                            profile_zero_line=False,
                                            improvement_unit=improvement_unit,
                                            plot_legend=combined,
                                            ylim=get_profile_ylim(
                                                PROFILE_YLIMS, metric, variant=variant,
                                            ),
                                            **plot_kwargs,
                                        )
                                        n += 1

                            for (
                                period_value
                            ) in leadtime_profile_periods:

                                common_path = (
                                    Path("profiles")
                                    / COMBINED_PLOT_FOLDER
                                    / COMPARISON_NAME
                                    / "comparisons"
                                    / safe_label(
                                        comparison_variant
                                    )
                                    / safe_label(period_dim)
                                    / safe_label(
                                        period_value
                                    )
                                    / (
                                        f"time_"
                                        f"{safe_label(valid_time_range)}"
                                        f"_lat_"
                                        f"{safe_label(valid_lat_range)}"
                                        f"_lon_"
                                        f"{safe_label(valid_lon_range)}"
                                    )
                                    / metric
                                    / METRIC_AGG_MODE
                                    / IMPROVEMENT_AGG_MODE
                                )

                                filename = (
                                    f"{common_s.var_fc}_"
                                    f"{metric}_"
                                    f"{safe_label(comparison_variant)}_"
                                    f"{LEADTIME_AGG_MODE}lt.png"
                                )

                                out_file = (
                                    COMMON_PLOT_DIR
                                    / common_path
                                    / filename
                                )

                                if (
                                    out_file.exists()
                                    and not REGENERATE_PLOTS
                                ):
                                    continue

                                print(f"Saving combined comparison profile {out_file}")

                                plot_profile(
                                    das=comparison_das,
                                    var=common_s.var_fc,
                                    metric=metric,
                                    models=comparison_models,
                                    labels=comparison_models,
                                    out_file=out_file,
                                    time_range=valid_time_range,
                                    das_member=[
                                        None
                                        for _
                                        in comparison_das
                                    ],
                                    profile_dim=leadtime_agg_coord,
                                    profile_label=LEADTIME_PROFILE_LABEL,
                                    select_value=period_value,
                                    select_dim=period_dim,
                                    select_unit=CLIM_PERIOD.value,
                                    plot_single_members=False,
                                    plot_legend=PLOT_LEGEND,
                                    improvement_unit=improvement_unit,
                                    model_colors=comparison_colors,
                                    model_linestyles=comparison_linestyles,
                                    ylim=get_profile_ylim(
                                        PROFILE_YLIMS,
                                        metric,
                                        variant=variant,
                                    ),
                                    **combined_profile_kwargs,
                                )

                                n += 1

                            # ----------------------------------
                            # Climatological-period profiles
                            # ----------------------------------

                            if PLOT_COMBINED_EXPERIMENT_CLIMATOLOGICAL_PROFILES:
                                climatological_comparison_das = [
                                    prepare_climatological_profile_da(
                                        da,
                                        period_dim=period_dim,
                                        clim_period=CLIM_PERIOD,
                                    )
                                    for da in comparison_das
                                ]

                                leadtime_groups = (
                                    [combined_experiment_climatological_profile_leadtimes]
                                    if COMBINE_COMBINED_EXPERIMENT_CLIMATOLOGICAL_LEADTIMES
                                    else [
                                        [lead_value]
                                        for lead_value in combined_experiment_climatological_profile_leadtimes
                                    ]
                                )

                                for lead_values in leadtime_groups:

                                    combined = len(lead_values) > 1

                                    leadtime_label = (
                                        "all_leadtimes"
                                        if combined
                                        else f"leadtime_{safe_label(lead_values[0])}"
                                    )

                                    common_path = (
                                        Path("profiles")
                                        / COMBINED_PLOT_FOLDER
                                        / COMPARISON_NAME
                                        / "climatology"
                                        / "comparisons"
                                        / safe_label(
                                            comparison_variant
                                        )
                                        / leadtime_label
                                        / (
                                            f"time_"
                                            f"{safe_label(valid_time_range)}"
                                            f"_lat_"
                                            f"{safe_label(valid_lat_range)}"
                                            f"_lon_"
                                            f"{safe_label(valid_lon_range)}"
                                        )
                                        / metric
                                        / METRIC_AGG_MODE
                                        / IMPROVEMENT_AGG_MODE
                                    )

                                    filename = (
                                        f"{common_s.var_fc}_"
                                        f"{metric}_"
                                        f"{safe_label(comparison_variant)}_"
                                        f"{safe_label(period_dim)}_"
                                        f"{leadtime_label}.png"
                                    )

                                    out_file = (
                                        COMMON_PLOT_DIR
                                        / common_path
                                        / filename
                                    )

                                    if (
                                        out_file.exists()
                                        and not REGENERATE_PLOTS
                                    ):
                                        continue

                                    print(f"Saving combined climatological comparison profile {out_file}")

                                    plot_das = climatological_comparison_das
                                    plot_models_current = comparison_models
                                    plot_labels_current = comparison_models
                                    plot_colors = comparison_colors
                                    plot_linestyles = comparison_linestyles
                                    plot_select_dim = leadtime_agg_coord
                                    plot_select_value = lead_values[0]
                                    plot_select_unit = LEADTIME_UNITS.value

                                    if combined:
                                        (
                                            plot_das,
                                            plot_models_current,
                                            plot_labels_current,
                                            _,
                                            plot_colors,
                                            plot_linestyles,
                                        ) = expand_climatological_leadtimes(
                                            climatological_comparison_das,
                                            comparison_models,
                                            lead_values,
                                            leadtime_dim=leadtime_agg_coord,
                                            leadtime_unit=LEADTIME_UNITS.value,
                                            color_mode=(
                                                COMBINED_EXPERIMENT_CLIMATOLOGICAL_LEADTIME_COLORS
                                            ),
                                            labels=plot_labels_current,
                                            model_colors=comparison_colors,
                                            model_linestyles=comparison_linestyles,
                                        )

                                        plot_select_dim = None
                                        plot_select_value = "all"
                                        plot_select_unit = None

                                    plot_profile(
                                        das=plot_das,
                                        var=common_s.var_fc,
                                        metric=metric,
                                        models=plot_models_current,
                                        labels=plot_labels_current,
                                        out_file=out_file,
                                        time_range=valid_time_range,
                                        das_member=[None] * len(plot_das),
                                        profile_dim=period_dim,
                                        profile_label=period_profile_label,
                                        select_dim=plot_select_dim,
                                        select_value=plot_select_value,
                                        select_unit=plot_select_unit,
                                        plot_single_members=False,
                                        plot_legend=PLOT_LEGEND,
                                        improvement_unit=improvement_unit,
                                        model_colors=plot_colors,
                                        model_linestyles=plot_linestyles,
                                        ylim=get_profile_ylim(
                                            PROFILE_YLIMS,
                                            metric,
                                            variant=variant,
                                        ),
                                        **combined_profile_kwargs,
                                    )

                                    n += 1

                # ----------------------------------------------
                # Comparison between common non-ML models,
                # e.g. FC -> clim-FC.
                # ----------------------------------------------

                else:

                    if (
                        baseline_model
                        not in das_by_model
                    ):
                        continue

                    if (
                        target_model
                        not in das_by_model
                    ):
                        continue

                    baseline_ds = (
                        comparison_source_by_model[
                            baseline_model
                        ]
                    )

                    target_ds = (
                        comparison_source_by_model[
                            target_model
                        ]
                    )

                    comparison_metrics = [
                        metric
                        for metric in METRICS
                        if (
                            metric
                            != "rank_histogram"
                            and metric
                            in baseline_ds
                            and metric
                            in target_ds
                        )
                    ]

                    for metric in comparison_metrics:

                        improvements = (
                            get_aggregated_metric_improvements(
                                baseline_ds,
                                target_ds,
                                metric=metric,
                                baseline_model=baseline_model,
                                target_model=target_model,
                                an=group["spatial_reference"],
                                metric_agg_mode=METRIC_AGG_MODE,
                                improvement_agg_mode=IMPROVEMENT_AGG_MODE,
                            )
                        )

                        for (
                            comparison_model,
                            comparison_da,
                        ) in improvements.items():

                            (
                                variant,
                                improvement_unit,
                                comparison_label,
                            ) = get_comparison_plot_meta(
                                baseline_model,
                                target_model,
                                comparison_model,
                                COMPARISON_LABELS_MAP,
                            )

                            for (
                                period_value
                            ) in leadtime_profile_periods:

                                common_path = (
                                    Path("profiles")
                                    / COMBINED_PLOT_FOLDER
                                    / COMPARISON_NAME
                                    / "comparisons"
                                    / safe_label(
                                        comparison_model
                                    )
                                    / safe_label(period_dim)
                                    / safe_label(
                                        period_value
                                    )
                                    / (
                                        f"time_"
                                        f"{safe_label(valid_time_range)}"
                                        f"_lat_"
                                        f"{safe_label(valid_lat_range)}"
                                        f"_lon_"
                                        f"{safe_label(valid_lon_range)}"
                                    )
                                    / metric
                                    / METRIC_AGG_MODE
                                    / IMPROVEMENT_AGG_MODE
                                )

                                filename = (
                                    f"{common_s.var_fc}_"
                                    f"{metric}_"
                                    f"{safe_label(comparison_model)}_"
                                    f"{LEADTIME_AGG_MODE}lt.png"
                                )

                                out_file = (
                                    COMMON_PLOT_DIR
                                    / common_path
                                    / filename
                                )

                                if (
                                    out_file.exists()
                                    and not REGENERATE_PLOTS
                                ):
                                    continue

                                print(f"Saving combined comparison profile {out_file}")

                                plot_profile(
                                    das=[
                                        comparison_da
                                    ],
                                    var=common_s.var_fc,
                                    metric=metric,
                                    models=(
                                        comparison_model,
                                    ),
                                    labels=(comparison_label,),
                                    out_file=out_file,
                                    time_range=valid_time_range,
                                    das_member=[None],
                                    profile_dim=leadtime_agg_coord,
                                    profile_label=LEADTIME_PROFILE_LABEL,
                                    select_value=period_value,
                                    select_dim=period_dim,
                                    select_unit=CLIM_PERIOD.value,
                                    plot_single_members=False,
                                    plot_legend=PLOT_LEGEND,
                                    improvement_unit=improvement_unit,
                                    ylim=get_profile_ylim(
                                        PROFILE_YLIMS,
                                        metric,
                                        variant=variant,
                                    ),
                                    **combined_profile_kwargs,
                                )

                                n += 1

                            # ----------------------------------
                            # Climatological-period profile
                            # ----------------------------------

                            if PLOT_COMBINED_EXPERIMENT_CLIMATOLOGICAL_PROFILES:
                                climatological_comparison_da = (
                                    prepare_climatological_profile_da(
                                        comparison_da,
                                        period_dim=period_dim,
                                        clim_period=CLIM_PERIOD,
                                    )
                                )

                                leadtime_groups = (
                                    [combined_experiment_climatological_profile_leadtimes]
                                    if COMBINE_COMBINED_EXPERIMENT_CLIMATOLOGICAL_LEADTIMES
                                    else [
                                        [lead_value]
                                        for lead_value in combined_experiment_climatological_profile_leadtimes
                                    ]
                                )

                                for lead_values in leadtime_groups:

                                    combined = len(lead_values) > 1

                                    leadtime_label = (
                                        "all_leadtimes"
                                        if combined
                                        else f"leadtime_{safe_label(lead_values[0])}"
                                    )

                                    common_path = (
                                        Path("profiles")
                                        / COMBINED_PLOT_FOLDER
                                        / COMPARISON_NAME
                                        / "climatology"
                                        / "comparisons"
                                        / safe_label(
                                            comparison_model
                                        )
                                        / leadtime_label
                                        / (
                                            f"time_"
                                            f"{safe_label(valid_time_range)}"
                                            f"_lat_"
                                            f"{safe_label(valid_lat_range)}"
                                            f"_lon_"
                                            f"{safe_label(valid_lon_range)}"
                                        )
                                        / metric
                                        / METRIC_AGG_MODE
                                        / IMPROVEMENT_AGG_MODE
                                    )

                                    filename = (
                                        f"{common_s.var_fc}_"
                                        f"{metric}_"
                                        f"{safe_label(comparison_model)}_"
                                        f"{safe_label(period_dim)}_"
                                        f"{leadtime_label}.png"
                                    )

                                    out_file = (
                                        COMMON_PLOT_DIR
                                        / common_path
                                        / filename
                                    )

                                    if (
                                        out_file.exists()
                                        and not REGENERATE_PLOTS
                                    ):
                                        continue

                                    print(f"Saving combined climatological comparison profile {out_file}")

                                    plot_das = [
                                        climatological_comparison_da
                                    ]
                                    plot_models_current = (
                                        comparison_model,
                                    )
                                    plot_labels_current = (comparison_label,)
                                    plot_colors = None
                                    plot_select_dim = leadtime_agg_coord
                                    plot_select_value = lead_values[0]
                                    plot_select_unit = LEADTIME_UNITS.value

                                    if combined:
                                        (
                                            plot_das,
                                            plot_models_current,
                                            plot_labels_current,
                                            _,
                                            plot_colors,
                                            _,
                                        ) = expand_climatological_leadtimes(
                                            plot_das,
                                            plot_models_current,
                                            lead_values,
                                            leadtime_dim=leadtime_agg_coord,
                                            leadtime_unit=LEADTIME_UNITS.value,
                                            labels=plot_labels_current,
                                        )

                                        plot_select_dim = None
                                        plot_select_value = "all"
                                        plot_select_unit = None

                                    plot_profile(
                                        das=plot_das,
                                        var=common_s.var_fc,
                                        metric=metric,
                                        models=plot_models_current,
                                        labels=plot_labels_current,
                                        out_file=out_file,
                                        time_range=valid_time_range,
                                        das_member=[None] * len(plot_das),
                                        profile_dim=period_dim,
                                        profile_label=period_profile_label,
                                        select_dim=plot_select_dim,
                                        select_value=plot_select_value,
                                        select_unit=plot_select_unit,
                                        plot_single_members=False,
                                        plot_legend=PLOT_LEGEND,
                                        improvement_unit=improvement_unit,
                                        model_colors=plot_colors,
                                        ylim=get_profile_ylim(
                                            PROFILE_YLIMS,
                                            metric,
                                            variant=variant,
                                        ),
                                        **combined_profile_kwargs,
                                    )

                                    n += 1

    print(
        f"Done. Saved {n} plots."
    )


def get_period_profile_label(
    period_reference: str,
    clim_period: ClimPeriod,
) -> str:
    """Return a human-readable x-axis label for climatological profiles."""
    reference_label = {
        "init": "Initialization",
        "valid": "Valid",
    }.get(period_reference)

    if reference_label is None:
        raise ValueError(
            f"Unsupported period_reference={period_reference!r}. "
            "Choose 'init' or 'valid'."
        )

    period_value = clim_period.value
    period_label = {
        "month": "month",
        "day": "day of month",
        "dayofyear": "day of year",
        "year": "year",
        "month_hour": "month/hour",
        "day_hour": "day/hour",
        "dayofyear_hour": "day of year/hour",
    }.get(
        period_value,
        period_value.replace("_", " "),
    )

    return f"{reference_label} {period_label}"



def get_improvement_variant(
    comparison_name: str,
) -> str:
    """
    Extract the improvement representation from a model-comparison
    name generated by get_aggregated_metric_improvements().
    """
    if comparison_name.endswith("_percentage"):
        return "percentage"

    if comparison_name.endswith("_difference"):
        return "difference"

    if comparison_name.endswith("_normalized"):
        return "normalized"

    raise ValueError(
        f"Cannot determine improvement representation "
        f"from comparison name {comparison_name!r}."
    )


IMPROVEMENT_UNITS = {
    "percentage": "%",
    "difference": "Δ",
    "normalized": "normalized",
}


def get_comparison_plot_meta(
    baseline_model: str,
    target_model: str,
    comparison_model: str,
    comparison_labels_map: dict,
) -> tuple[str, str, str]:
    """Return comparison variant, plotting unit, and display label."""
    variant = get_improvement_variant(comparison_model)
    improvement_unit = IMPROVEMENT_UNITS[variant]
    comparison_label = comparison_labels_map.get(
        (baseline_model, target_model, variant),
        comparison_model,
    )
    return variant, improvement_unit, comparison_label


def get_profile_ylim(
    profile_ylims: dict,
    metric: str,
    *,
    variant: str = "absolute",
) -> tuple[float, float] | None:
    """Return optional y-axis limits for a metric-profile variant."""
    return profile_ylims.get(variant, {}).get(metric)


def prepare_climatological_profile_da(
    da: xr.DataArray | None,
    *,
    period_dim: str,
    clim_period: ClimPeriod,
) -> xr.DataArray | None:
    """Prepare a metric DataArray for a climatological-period profile.

    The aggregate ``"all"`` period is removed. Month and day-of-year
    coordinates are converted to integers so ``plot_profile`` can format the
    climatological x axis correctly.
    """
    if da is None:
        return None

    if period_dim not in da.dims:
        raise ValueError(
            f"Climatological period dimension {period_dim!r} not found "
            f"in {da.dims}."
        )

    period_values = [
        value
        for value in da[period_dim].values
        if str(value) != "all"
    ]

    da = da.sel({period_dim: period_values})

    if clim_period in {
        ClimPeriod.MONTH,
        ClimPeriod.DAYOFYEAR,
    }:
        period_values = np.asarray(
            [int(value) for value in da[period_dim].values],
            dtype=int,
        )
        da = da.assign_coords({period_dim: period_values})

    return da


def get_climatological_profile_leadtimes(
    reference: xr.DataArray | xr.Dataset,
    *,
    leadtime_dim: str,
    wanted_leadtimes: list[object] | tuple[object, ...] | None,
) -> list[object]:
    """Return lead times to use as fixed values in climatological profiles."""
    available = list(reference[leadtime_dim].values)

    if wanted_leadtimes is None:
        return available

    wanted = {str(value) for value in wanted_leadtimes}

    return [
        value
        for value in available
        if str(value) in wanted
    ]

# ==========================================================
# Color and shading
# ==========================================================

ColorMode = Literal["shades", "colors"]
Color = str | tuple[float, float, float]

def get_leadtime_colors(
    n_leadtimes: int,
    *,
    color_mode: ColorMode = "shades",
    base_color: str = "tab:blue",
    shade_range: tuple[float, float] = (0.4, 1.0),
) -> list[Color]:
    """Return one color per lead time."""
    if color_mode not in {"shades", "colors"}:
        raise ValueError(f"Unsupported color_mode={color_mode!r}")

    if not (0.0 <= shade_range[0] <= shade_range[1] <= 1.0):
        raise ValueError(f"Invalid shade_range={shade_range!r}")

    default_colors = matplotlib.rcParams[
        "axes.prop_cycle"
    ].by_key()["color"]

    if color_mode == "colors":
        return [
            default_colors[i % len(default_colors)]
            for i in range(n_leadtimes)
        ]

    base_rgb = np.asarray(to_rgb(base_color))
    fractions = np.linspace(*shade_range, n_leadtimes)

    return [
        tuple(
            float(value)
            for value in 1.0 - (1.0 - base_rgb) * fraction
        )
        for fraction in fractions
    ]


def expand_climatological_leadtimes(
    das: Sequence[xr.DataArray],
    models: Sequence[str],
    lead_values: Sequence[int | float],
    *,
    leadtime_dim: str,
    leadtime_unit: str | None,
    color_mode: ColorMode = "shades",
    das_member: Sequence[xr.DataArray | None] | None = None,
    labels: Sequence[str] | None = None,
    model_colors: dict[str, str] | None = None,
    model_linestyles: dict[str, str] | None = None,
    shade_range: tuple[float, float] = (0.4, 1.0),
) -> tuple[
    list[xr.DataArray],
    tuple[str, ...],
    tuple[str, ...],
    list[xr.DataArray | None],
    dict[str, Color],
    dict[str, str],
]:
    """Expand model x lead time into explicitly styled profile curves."""
    models = list(models)
    labels = models if labels is None else list(labels)

    if len(labels) != len(models):
        raise ValueError(
            "Select the same number of labels and models. "
            f"Got {len(labels)} labels and {len(models)} models."
        )

    das_member = (
        [None] * len(das)
        if das_member is None
        else list(das_member)
    )

    if len(das) != len(models) or len(das_member) != len(models):
        raise ValueError(
            "das, models, and das_member must have the same length."
        )

    default_colors = matplotlib.rcParams[
        "axes.prop_cycle"
    ].by_key()["color"]

    unit = leadtime_unit[:1].upper() if leadtime_unit else ""

    expanded_das: list[xr.DataArray] = []
    expanded_models: list[str] = []
    expanded_labels: list[str] = []
    expanded_das_member: list[xr.DataArray | None] = []
    expanded_colors: dict[str, Color] = {}
    expanded_linestyles: dict[str, str] = {}

    for model_index, (
        model,
        display_model,
        da,
        da_member,
    ) in enumerate(
        zip(models, labels, das, das_member, strict=True)
    ):
        base_color = (
            model_colors[model]
            if model_colors is not None and model in model_colors
            else default_colors[model_index % len(default_colors)]
        )

        leadtime_colors = get_leadtime_colors(
            len(lead_values),
            color_mode=color_mode,
            base_color=base_color,
            shade_range=shade_range,
        )

        linestyle = (
            model_linestyles.get(model, "-")
            if model_linestyles is not None
            else "-"
        )

        for lead_index, lead_value in enumerate(lead_values):
            label = f"{model} · {lead_value}{unit}"
            display_label = f"{display_model} · {lead_value}{unit}"

            expanded_das.append(
                da.sel({leadtime_dim: lead_value})
            )

            expanded_models.append(label)
            expanded_labels.append(display_label)

            expanded_das_member.append(
                None
                if da_member is None
                else da_member.sel({leadtime_dim: lead_value})
            )

            expanded_colors[label] = leadtime_colors[lead_index]
            expanded_linestyles[label] = linestyle

    return (
        expanded_das,
        tuple(expanded_models),
        tuple(expanded_labels),
        expanded_das_member,
        expanded_colors,
        expanded_linestyles,
    )


def build_experiment_axis_leadtime_curves(
    entries: Sequence[tuple[str, xr.DataArray]],
    *,
    lead_values: Sequence[int | float],
    period_dim: str,
    period_value: str,
    leadtime_dim: str,
    leadtime_unit: str,
    sort_numeric: bool,
    color_mode: ColorMode = "shades",
    shade_range: tuple[float, float] = (0.4, 1.0),
) -> tuple[
    list[xr.DataArray],
    tuple[str, ...],
    tuple[str, ...],
    dict[str, Color],
]:
    """Build one experiment-axis curve per lead time, with consistent styling."""
    default_colors = matplotlib.rcParams[
        "axes.prop_cycle"
    ].by_key()["color"]

    leadtime_colors = get_leadtime_colors(
        len(lead_values),
        color_mode=color_mode,
        base_color=default_colors[0],
        shade_range=shade_range,
    )

    unit = leadtime_unit[:1].upper() if leadtime_unit else ""

    das: list[xr.DataArray] = []
    models: list[str] = []
    labels: list[str] = []
    colors: dict[str, Color] = {}

    for index, lead in enumerate(lead_values):
        model = f"leadtime_{index}"
        label = f"{lead}{unit}"

        das.append(
            build_experiment_axis_da(
                entries,
                period_dim=period_dim,
                period_value=period_value,
                leadtime_dim=leadtime_dim,
                leadtime_value=lead,
                sort_numeric=sort_numeric,
            )
        )

        models.append(model)
        labels.append(label)
        colors[model] = leadtime_colors[index]

    return das, tuple(models), tuple(labels), colors


def build_experiment_axis_da(
    entries: list[tuple[str, xr.DataArray]],
    *,
    period_dim: str,
    period_value,
    leadtime_dim: str,
    leadtime_value,
    sort_numeric: bool = True,
) -> xr.DataArray:

    selected = []

    for label, da in entries:
        current = (
            da.sel({
                period_dim: period_value,
                leadtime_dim: leadtime_value,
            })
            .squeeze(drop=True)
        )

        if current.size != 1:
            raise ValueError(
                f"Expected one scalar for experiment {label!r}, "
                f"got dims={current.dims}, shape={current.shape}."
            )

        selected.append((label, current))

    if sort_numeric:
        selected.sort(
            key=lambda item: float(
                item[0].strip().split(maxsplit=1)[0]
            )
        )

    return xr.concat(
        [
            da.expand_dims(experiment=[label])
            for label, da in selected
        ],
        dim="experiment",
    )


if __name__ == "__main__":
    main()
