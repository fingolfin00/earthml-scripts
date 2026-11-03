from typing import Literal
from pathlib import Path

import xarray as xr

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
    MetricKind,
    is_deterministic,
    is_probabilistic,
    get_metrics,
    calculate_save_and_subset_climatologies,
    stack_hour_clim,
    groupby_period,
    build_metric_improvements,
    get_required_improvement_metrics,
    get_metric_improvement_significance,
)
from earthml.plots import (
    safe_label,
    lead_label,
    PlotMode,
    plot_map,
)

from settings_plot_seasonal import VARIABLE_PLOT_CONFIG, IMPROVEMENT_PLOT_CONFIG
# from settings_plot_weather_atmo import VARIABLE_PLOT_CONFIG, IMPROVEMENT_PLOT_CONFIG

from locations import CITY_LOCATIONS, SEA_LOCATIONS


# ==========================================================
# Paths
# ==========================================================

# Seasonal
EXP_NAME = "experiments"
# EXP_NAME = "experiment_new"

# Weather
# EXP_NAME = "weather_atmo"
# EXP_NAME = "weather_atmo_encoding"
# EXP_NAME = "weather_atmo_normalization"
# EXP_NAME = "weather_atmo_ablation_fixed_val"

EXPERIMENTS_ROOT = Path(
    f"/Users/jacopodallaglio/ML/training/seasonal/{EXP_NAME}"
    # f"/work/cmcc/jd19424/ML/MLBC/experiments/{EXP_NAME}"
)

OUTPUT_ROOT = Path(
    f"/Users/jacopodallaglio/ML/training/seasonal/{EXP_NAME}"
    # f"/work/cmcc/jd19424/ML/MLBC/plots/{EXP_NAME}"
)

# Shared FC and clim-FC maps
COMMON_PLOT_DIR = OUTPUT_ROOT / "common"

OROGRAPHY_PATH = Path(
    "/Users/jacopodallaglio/ML/training/seasonal/data/input/era5_orography.zarr"
    # "/work/cmcc/jd19424/ML/MLBC/data/orography/era5_orography.zarr"
)

# ==========================================================
# Plot settings
# ==========================================================

METRIC_KIND: MetricKind = "maps"

PLOT_TYPE: Literal["pcolormesh", "contourf"] = "contourf"

PLOT_TITLE = True
PLOT_LABELS = True

TITLE_SIZE = None
LABEL_SIZE = None
TICK_SIZE = None
DPI = 300

TITLE_STRFTIME = "%Y" # seasonal
# title_strftime = "%m.%Y" # weather

PLOT_MODELS = (
    "fc",
    # "clim-fc",
    "mlfc",
)

REGENERATE_PLOTS = (
    # "fc",
    # "clim-fc",
    # "mlfc",
    # "improvement",
)

# ==========================================================
# Model comparisons
# ==========================================================

MODEL_COMPARISONS = (
    ("fc", "mlfc"),
    # ("mlfc", "clim-fc"),
    # ("fc", "clim-fc"),
)

# ==========================================================
# Statistical significance
# ==========================================================

PLOT_SIGNIFICANCE = False

SIGNIFICANCE_N_BOOTSTRAP = 200
SIGNIFICANCE_BLOCK_SIZE = 1
SIGNIFICANCE_CONFIDENCE_LEVEL = 0.95
SIGNIFICANCE_SEED = 42

SIGNIFICANCE_STRIDE = 3
SIGNIFICANCE_SIZE = 2.0
SIGNIFICANCE_ALPHA = 0.6

# ==========================================================
# Data processing
# ==========================================================

INTERPOLATE = True # seasonal
# interpolate = False # weather
BUILD_ANALYSIS = True

RECALCULATE_CLIMATOLOGY = False

# ==========================================================
# Climatology
# ==========================================================

CLIM_PERIOD: ClimPeriod = ClimPeriod.MONTH
CLIM_ROLLING_WINDOW = None

# clim_period: ClimPeriod = ClimPeriod.DAYOFYEAR_HOUR
# clim_rolling_window = 31

# Period grouping reference:
#   "init"  -> group by forecast initialization time
#   "valid" -> group by forecast valid time (init + lead time)
PERIOD_REFERENCE: Literal["init", "valid"] = "init"

# ==========================================================
# Regional boxes
# ==========================================================

LOCATIONS = SEA_LOCATIONS | CITY_LOCATIONS

PLOT_LOCATIONS = [
    # Cities
    # "newyork",
    # "miami",
    # "chicago",
    # "denver",
    # "losangeles",
    # "seattle",
    # "houston",

    # Ocean
    # "gulfstream",
    # "kuroshio",
    # "north_atlantic",
    # "equatorial_pacific",
    "nino34",
]

# None / [] -> no boxes
HIGHLIGHTED_REGIONS = [
    {
        **LOCATIONS[name],
        "name": name.replace("_", " ").title(),
        "edgecolor": "black",
        "facecolor": "none",
        "linewidth": 3,
        "linestyle": "-",
    }
    for name in PLOT_LOCATIONS
]

# ==========================================================
# Time selection
# ==========================================================

TIME_RANGE: tuple[str, str] | Literal["train", "val", "test", "train_val", "full"] = "test"
# TIME_RANGE = ("2018-01-01", "2022-12-31")

CLIM_TIME_RANGE: tuple[str, str] | Literal["train", "val", "test", "train_val", "full"] = "train"
# CLIM_TIME_RANGE = "train_val"

INFERENCE_PERIOD = None
# INFERENCE_PERIOD = ("2025-01-01", "2025-10-10")
# INFERENCE_PERIOD = ("2025-01-01", "2025-05-12")

PERIODS_REQUESTED = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "all",
]

# ==========================================================
# Leadtime
# ==========================================================

LEADTIMES = None  # All experiment lead times
# LEADTIMES = [24, 48, 72]

LEADTIME_UNITS = LeadtimeUnit.MONTHS # seasonal
# LEADTIME_UNITS = LeadtimeUnit.HOURS # weather

LEADTIME_AGG_MODE: LeadtimeAgg = "aggregated"
# "single" for weather
# "aggregated". for seasonal
# "seasonal_window"

HOVMOLLER_TIME_AGG: ClimPeriod | None = ClimPeriod.MONTH

# ==========================================================
# Metrics
# ==========================================================

METRICS = [
    # Orography
    # "orography",
    # "orography_grad_mag",

    # Deterministic metrics - absolute fields
    "bias",
    # "mae",
    # "mse",
    # "rmse",
    # "nrmse",
    # "r2",
    # "corr",
    # "kendall_tau",
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
    "bias_anom",
    "mae_anom",
    # "mse_anom",
    "rmse_anom",
    # "nrmse_anom",
    # "r2_anom",
    "acc",
    # "kendall_tau_anom",
    # "fc_anom_std",
    # "an_anom_std",
    "std_ratio_anom",

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

    # Event-based probabilistic metrics - absolute terciles
    # "brier_lower",
    # "brier_middle",
    # "brier_upper",
    # "roc_lower",
    # "roc_middle",
    # "roc_upper",

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

    # Ensemble / probabilistic metrics - anomaly fields
    # "ens_member_rmse_anom",
    # "mean_member_rmse_anom",
    # "spread_anom",
    "spread_anom_skill_ratio",
    "crps_anom",

    # Event-based probabilistic metrics - anomaly terciles
    # "brier_anom_lower",
    # "brier_anom_middle",
    # "brier_anom_upper",
    # "roc_anom_lower",
    # "roc_anom_middle",
    # "roc_anom_upper",
]

# ==========================================================
# Variables and regions
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

# ==========================================================
# Spatial subset
# ==========================================================

# ConUS
# lat_range = (50, 25)
# lon_range = (-130, -60)

# Europe
# lat_range = (80, 30)
# lon_range = (-30, 60)

# Pacific
# lat_range = (20, -20)
# lon_range = (-195, -135)

# Whole configured region
LAT_RANGE = None
LON_RANGE = None


# ==========================================================
# Experiment selection
# ==========================================================

FILTERS = dict(
    var_fc=VARIABLES,
    region_name=REGIONS,
    # net_name="ConvNeXtTransformerUNet",
    net_name="SmaAt_UNet",
    # target_mode="analysis",
    # input_realization_avg=False,
    loss_name="GeoMaskedMSELoss",
    # train_start="2019-10-14",
    # train_subsamples=1000,
    # seasonal_encoding=True,
    # ensemble_encoding=True,
    # separate_training_by_init_period=ClimPeriod.MONTH,
    # extra_suffix_folder="",
)

def resolve_time_range(s, value: tuple[str, str] | str) -> tuple[str, str]:
    if isinstance(value, tuple):
        return value
    ranges = {
        "train": (s.train_start, s.train_end),
        "val": (s.val_start, s.val_end),
        "test": (s.test_start, s.test_end),
        "train_val": (s.train_start, s.val_end),
        "full": (s.train_start, s.test_end),
    }
    if value not in ranges:
        raise ValueError(f"Invalid time range: {value!r}")
    return ranges[value]


def main() -> None:
    settings = get_experiment_configs(EXPERIMENTS_ROOT, **FILTERS)

    print(f"Found {len(settings)} matching experiment(s).")

    # Add std to compute normalized diff metrics
    metrics_to_compute = get_required_improvement_metrics(list(METRICS))

    n = 0
    for s in settings:
        valid_time_range = resolve_time_range(s, TIME_RANGE)
        clim_time_range = resolve_time_range(s, CLIM_TIME_RANGE)
        mlfc_path = None
        if INFERENCE_PERIOD is not None:
            inference_start, inference_end = INFERENCE_PERIOD
            mlfc_path = (
                s.exp_dir / "inference"
                / f"{inference_start}_{inference_end}"
                / "test_corrected.zarr"
            )
            if TIME_RANGE == "test":
                valid_time_range = INFERENCE_PERIOD

        lat_lon = list(s.region.values()) if s.region is not None else [None, None]
        valid_lat_range = lat_lon[0] if LAT_RANGE is None else LAT_RANGE
        valid_lon_range = lat_lon[1] if LON_RANGE is None else LON_RANGE

        leadtime_agg_coord = "leadtime" if LEADTIME_AGG_MODE=="single" else "leadtime_seasonal"
        period_dim = f"{PERIOD_REFERENCE}_{CLIM_PERIOD.value}"

        if PERIOD_REFERENCE == "valid" and LEADTIME_AGG_MODE == "aggregated":
            raise ValueError(
                "period_reference='valid' is not supported with "
                "leadtime_agg_mode='aggregated' because an aggregated "
                "forecast has no unique valid time. Use 'single' or "
                "'seasonal_window'."
            )

        if PERIOD_REFERENCE == "valid" and PLOT_SIGNIFICANCE:
            raise ValueError(
                "Valid-time grouping is not yet supported by "
                "get_metric_improvement_significance(). Disable "
                "plot_significance or extend the significance helper with "
                "period_reference and leadtime_unit support first."
            )

        print(
            f"Generate {LEADTIME_AGG_MODE} metric maps "
            f"grouped by {period_dim} for {s.var_an, s.var_fc} "
            f"in {s.region_name} (lon={valid_lon_range}, lat={valid_lat_range})"
        )

        fc, an, mlfc = get_and_subset_datasets(
            s,
            leadtime_units=LEADTIME_UNITS,
            lat_range=valid_lat_range,
            lon_range=valid_lon_range,
            time_range=valid_time_range,
            interpolate=INTERPOLATE,
            mlfc_path=mlfc_path,
        )
        mlfc = mlfc.assign_coords(leadtime=s.leadtimes) if mlfc is not None else None

        fc_clim, an_clim, mlfc_clim = calculate_save_and_subset_climatologies(
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
        mlfc_clim = mlfc_clim.assign_coords(leadtime=s.leadtimes) if mlfc_clim is not None else None

        leadtime_dim = fc.earthml.guessed_dims.leadtime
        selected_leadtimes = s.leadtimes if LEADTIMES is None else LEADTIMES
        missing_leadtimes = set(selected_leadtimes) - set(s.leadtimes)
        if missing_leadtimes:
            raise ValueError(
                f"Requested leadtimes {sorted(missing_leadtimes)} are unavailable; "
                f"available: {list(s.leadtimes)}"
            )
        fc = fc.sel({leadtime_dim: selected_leadtimes})
        an = an.sel({leadtime_dim: selected_leadtimes})
        fc_clim = fc_clim.sel({leadtime_dim: selected_leadtimes})
        an_clim = an_clim.sel({leadtime_dim: selected_leadtimes})
        if mlfc is not None:
            mlfc = mlfc.sel({leadtime_dim: selected_leadtimes})
        if mlfc_clim is not None:
            mlfc_clim = mlfc_clim.sel({leadtime_dim: selected_leadtimes})

        # fc corrected with analysis clim
        fc_clim_da = stack_hour_clim(fc_clim[s.var_fc], CLIM_PERIOD)
        an_clim_da = stack_hour_clim(an_clim[s.var_an], CLIM_PERIOD)

        fc_anom_da = groupby_period(fc[s.var_fc], an.earthml.guessed_dims.time, CLIM_PERIOD) - fc_clim_da
        clim_fc = (groupby_period(fc_anom_da, an.earthml.guessed_dims.time, CLIM_PERIOD) + an_clim_da).to_dataset(name=s.var_fc)

        realization_dim = fc.earthml.guessed_dims.realization
        an_clim_for_fc = an_clim
        if (
            realization_dim is not None
            and realization_dim in fc.dims
            and realization_dim not in an_clim_for_fc.dims
        ):
            an_clim_for_fc = an_clim_for_fc.expand_dims(
                {
                    realization_dim: fc[realization_dim]
                }
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

        model_plot_folders = {
            "fc": COMMON_PLOT_DIR,
            "clim-fc": COMMON_PLOT_DIR,
            "mlfc": OUTPUT_ROOT / s.output_name / "plots",
        }

        metric_maps_by_model: dict[str, xr.Dataset] = {}
        significance_by_comparison: dict[tuple[str, str, str], xr.Dataset] = {}

        required_models = tuple(dict.fromkeys([
            *PLOT_MODELS,
            *(m for pair in MODEL_COMPARISONS for m in pair),
        ]))
        for model in required_models:
            ds = model_datasets[model]
            ds_clim = model_climatologies[model]

            if ds is None or ds_clim is None:
                continue

            deterministic_metrics = [
                m for m in metrics_to_compute
                if is_deterministic(m)
            ]

            probabilistic_metrics = [
                m for m in metrics_to_compute
                if is_probabilistic(m)
            ]

            metric_maps_det = xr.Dataset()
            metric_maps_prob = xr.Dataset()
            if len(deterministic_metrics) != 0:
                print(f"Get {model} deterministic metric maps")
                metric_maps_det = get_metrics(
                    an=an,
                    fc=ds,
                    var=s.var_fc,
                    metric_kind=METRIC_KIND,
                    leadtime_agg=LEADTIME_AGG_MODE, # "single", "aggregated", "seasonal_window"
                    realization_agg=True,
                    an_clim=an_clim,
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
            if len(probabilistic_metrics) != 0:
                print(f"Get {model} probabilistic metric maps")
                metric_maps_prob = get_metrics(
                    an=an,
                    fc=ds,
                    var=s.var_fc,
                    metric_kind=METRIC_KIND,
                    leadtime_agg=LEADTIME_AGG_MODE, # "single", "aggregated", "seasonal_window"
                    realization_agg=False,
                    an_clim=an_clim,
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

            metric_maps = xr.merge([metric_maps_det, metric_maps_prob])

            if METRIC_KIND in {"time_lon", "time_lat"}:
                time_dim = metric_maps.earthml.guessed_dims.time

                if time_dim is None or time_dim not in metric_maps.dims:
                    raise ValueError(
                        f"Cannot calculate monthly Hovmöller: "
                        f"no time dimension found in {metric_maps.dims}"
                    )

                # TODO support _hour groups
                metric_maps = metric_maps.groupby(f"{time_dim}.{HOVMOLLER_TIME_AGG}").mean(time_dim, skipna=True)

            # =====================================================
            # Statistical significance of model comparisons
            # =====================================================

            metric_maps_by_model[model] = metric_maps

            if (
                PLOT_SIGNIFICANCE
                and METRIC_KIND == "maps"
            ):
                for baseline_model, target_model in MODEL_COMPARISONS:

                    if target_model != model:
                        continue

                    if baseline_model not in metric_maps_by_model:
                        continue

                    baseline_ds = model_datasets[baseline_model]
                    target_ds = model_datasets[target_model]

                    baseline_clim_ds = model_climatologies[baseline_model]
                    target_clim_ds = model_climatologies[target_model]

                    if (
                        baseline_ds is None
                        or target_ds is None
                        or baseline_clim_ds is None
                        or target_clim_ds is None
                    ):
                        continue

                    significance_metrics = [
                        m
                        for m in METRICS
                        if (
                            m != "rank_histogram"
                            and m in metric_maps_by_model[baseline_model]
                            and m in metric_maps_by_model[target_model]
                        )
                    ]

                    for m in significance_metrics:
                        print(
                            f"Get significance for {m} "
                            f"({baseline_model} -> {target_model})"
                        )

                        significance_by_comparison[
                            baseline_model,
                            target_model,
                            m,
                        ] = get_metric_improvement_significance(
                            an=an,
                            fc=baseline_ds,
                            mlfc=target_ds,
                            var_fc=s.var_fc,
                            var_an=s.var_an,
                            metric=m,
                            metric_kind="maps",
                            leadtime_agg=LEADTIME_AGG_MODE,
                            realization_agg=is_deterministic(m),
                            fc_clim=baseline_clim_ds,
                            mlfc_clim=target_clim_ds,
                            an_clim=an_clim,
                            leadtime_windows=s.seasonal_leadtime_windows,
                            leadtime_agg_coord=leadtime_agg_coord,
                            clim_period=CLIM_PERIOD,
                            period_dim=period_dim,
                            periods_requested=PERIODS_REQUESTED,
                            n_bootstrap=SIGNIFICANCE_N_BOOTSTRAP,
                            block_size=SIGNIFICANCE_BLOCK_SIZE,
                            confidence_level=SIGNIFICANCE_CONFIDENCE_LEVEL,
                            seed=SIGNIFICANCE_SEED,
                            align=False,
                            fair_correction=False,
                        )

            available_metrics = [
                str(x) for x in metric_maps.data_vars
                if str(x) in METRICS and str(x) != "rank_histogram"
            ]

            available_periods = [
                str(x) for x in metric_maps[period_dim].values
                if str(x) in PERIODS_REQUESTED
            ]

            print(f"Plotting {model} metrics {available_metrics} for periods {available_periods} for exp {s.output_name}")

            for m in available_metrics:
                # ----------------------------------------------------------
                # DataArrays to plot
                #
                # value:
                #   (
                #       dataarray,
                #       baseline_model,
                #       target_model,
                #   )
                #
                # baseline/target are None for ordinary metric maps.
                # ----------------------------------------------------------

                dataarrays_to_plot: dict[
                    str,
                    tuple[xr.DataArray, str | None, str | None],
                ] = (
                    {model: (metric_maps[m], None, None)}
                    if model in PLOT_MODELS else {}
                )

                # ==========================================================
                # Build requested model comparisons
                # ==========================================================

                for baseline_model, target_model in MODEL_COMPARISONS:

                    # Comparison is generated when its target model
                    # is the model currently being processed.
                    if target_model != model:
                        continue

                    if baseline_model not in metric_maps_by_model:
                        continue

                    if m not in metric_maps_by_model[baseline_model]:
                        continue

                    improvements = build_metric_improvements(
                        metric_maps_by_model[baseline_model],
                        metric_maps_by_model[target_model],
                        metric=m,
                        baseline_model=baseline_model,
                        target_model=target_model,
                    )

                    for plot_model, dataarray in improvements.items():
                        dataarrays_to_plot[plot_model] = (
                            dataarray,
                            baseline_model,
                            target_model,
                        )

                # ==========================================================
                # Plot
                # ==========================================================

                for (
                    plot_model,
                    (
                        dataarray,
                        comparison_baseline,
                        comparison_target,
                    ),
                ) in dataarrays_to_plot.items():

                    is_improvement = (
                        comparison_baseline is not None
                        and comparison_target is not None
                    )

                    for period_value in available_periods:
                        for lead_value in dataarray[leadtime_agg_coord].values:
                            label = safe_label(
                                lead_label(
                                    dataarray,
                                    lead_value,
                                    leadtime_agg_coord,
                                )
                            )

                            common_path = (
                                Path(METRIC_KIND)
                                / "metrics"
                                / safe_label(period_dim)
                                / safe_label(period_value)
                                / (
                                    f"time_{safe_label(valid_time_range)}"
                                    f"_lat_{safe_label(LAT_RANGE)}"
                                    f"_lon_{safe_label(LON_RANGE)}"
                                )
                                / m
                                / LEADTIME_AGG_MODE
                            )

                            filename = (
                                f"{s.var_fc}_{m}_{plot_model}"
                                f"_lead_{label}.png"
                            )

                            out_file = (
                                model_plot_folders[model]
                                / common_path
                                / filename
                            )

                            link = (
                                s.plot_dir
                                / common_path
                                / filename
                            )

                            # ==================================================
                            # Significance for this exact comparison
                            # ==================================================

                            significance = None

                            if (
                                PLOT_SIGNIFICANCE
                                and is_improvement
                            ):
                                significance_key = (
                                    comparison_baseline,
                                    comparison_target,
                                    m,
                                )

                                if (
                                    significance_key
                                    in significance_by_comparison
                                ):
                                    significance = (
                                        significance_by_comparison[
                                            significance_key
                                        ]["significant"]
                                    )

                            force_regenerate = (
                                "improvement" in REGENERATE_PLOTS
                                if is_improvement
                                else model in REGENERATE_PLOTS
                            )

                            # ==================================================
                            # Normal plot
                            # ==================================================

                            if (
                                not out_file.exists()
                                or force_regenerate
                            ):
                                print(f"Saving map {out_file}")

                                plot_map(
                                    dataarray,
                                    var=s.var_fc,
                                    metric=m,
                                    model=plot_model,
                                    start_period=period_value,
                                    lead_value=lead_value,
                                    out_file=out_file,
                                    time_range=valid_time_range,
                                    leadtime_dim=leadtime_agg_coord,
                                    leadtime_units=LEADTIME_UNITS,
                                    period_dim=period_dim,
                                    clim_period=(
                                        None
                                        if METRIC_KIND == "map"
                                        else HOVMOLLER_TIME_AGG
                                    ),
                                    var_plot_config=VARIABLE_PLOT_CONFIG,
                                    impro_plot_config=IMPROVEMENT_PLOT_CONFIG,
                                    plot_kind=METRIC_KIND,
                                    plot_type=PLOT_TYPE,
                                    plot_title=PLOT_TITLE,
                                    plot_labels=PLOT_LABELS,
                                    title_size=TITLE_SIZE,
                                    label_size=LABEL_SIZE,
                                    tick_size=TICK_SIZE,
                                    dpi=DPI,
                                    title_strftime=TITLE_STRFTIME,
                                    significance=None,
                                    regions=HIGHLIGHTED_REGIONS if METRIC_KIND == "maps" else None,
                                )

                                n += 1

                            # ==================================================
                            # Additional plot with significance stippling
                            # ==================================================

                            if significance is not None:
                                significance_out_file = (
                                    out_file.parent
                                    / (
                                        f"{out_file.stem}"
                                        f"_significance"
                                        f"{out_file.suffix}"
                                    )
                                )

                                if (
                                    not significance_out_file.exists()
                                    or force_regenerate
                                ):
                                    print(
                                        "Saving significance map "
                                        f"{significance_out_file}"
                                    )

                                    plot_map(
                                        dataarray,
                                        var=s.var_fc,
                                        metric=m,
                                        model=plot_model,
                                        start_period=period_value,
                                        lead_value=lead_value,
                                        out_file=significance_out_file,
                                        time_range=valid_time_range,
                                        leadtime_dim=leadtime_agg_coord,
                                        leadtime_units=LEADTIME_UNITS,
                                        period_dim=period_dim,
                                        clim_period=(
                                            None
                                            if METRIC_KIND == "map"
                                            else HOVMOLLER_TIME_AGG
                                        ),
                                        var_plot_config=VARIABLE_PLOT_CONFIG,
                                        impro_plot_config=IMPROVEMENT_PLOT_CONFIG,
                                        plot_kind=METRIC_KIND,
                                        plot_type=PLOT_TYPE,
                                        plot_title=PLOT_TITLE,
                                        plot_labels=PLOT_LABELS,
                                        title_size=TITLE_SIZE,
                                        label_size=LABEL_SIZE,
                                        tick_size=TICK_SIZE,
                                        dpi=DPI,
                                        title_strftime=TITLE_STRFTIME,
                                        significance=significance,
                                        significance_stride=(
                                            SIGNIFICANCE_STRIDE
                                        ),
                                        significance_size=(
                                            SIGNIFICANCE_SIZE
                                        ),
                                        significance_alpha=(
                                            SIGNIFICANCE_ALPHA
                                        ),
                                        regions=HIGHLIGHTED_REGIONS if METRIC_KIND == "maps" else None,
                                    )

                                    n += 1

                            if (
                                model in ("fc", "clim-fc")
                                and not link.exists()
                            ):
                                link.parent.mkdir(
                                    parents=True,
                                    exist_ok=True,
                                )
                                link.symlink_to(
                                    out_file.resolve()
                                )

    print(f"Done. Saved {n} plots.")


if __name__ == "__main__":
    main()
