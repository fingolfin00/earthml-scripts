from pathlib import Path
from collections import defaultdict

import numpy as np
import xarray as xr

import matplotlib
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
    get_metrics,
    calculate_save_and_subset_climatologies,
    stack_hour_clim,
    groupby_period,
    build_metric_improvements,
    get_required_improvement_metrics,
)
from earthml.plots import (
    safe_label,
    PlotMode,
    plot_profile,
)


def main() -> None:
    # ==========================================================
    # Paths
    # ==========================================================

    experiments_root = Path(
        "/Users/jacopodallaglio/ML/training/seasonal/experiments"
        # "/work/cmcc/jd19424/ML/MLBC/experiments/weather_atmo"
    )

    common_plot_dir = Path(
        "/Users/jacopodallaglio/ML/training/seasonal/plots"
        # "/work/cmcc/jd19424/ML/MLBC/plots/weather_atmo/common"
    )

    # ==========================================================
    # Plot settings
    # ==========================================================

    plot_mode: PlotMode = "profiles"
    regenerate_plots = True

    # ==========================================================
    # Common profile styling
    # ==========================================================

    plot_title = False
    plot_legend = False

    title_size = None
    label_size = 20
    tick_size = 15
    dpi = 300

    title_strftime = "%Y"     # seasonal
    # title_strftime = "%m.%Y"    # weather

    model_labels = {
        "fc": "FC",
        "mlfc": "MLFC",
        "clim-fc": "Clim-FC",
    }

    comparison_labels_map = {
        ("fc", "mlfc", "percentage"): "MLFC vs FC [%]",
        ("fc", "mlfc", "difference"): "MLFC vs FC [Δ]",
        ("fc", "mlfc", "normalized"): "MLFC vs FC [norm.]",
    }

    # ==========================================================
    # Profile y-axis limits
    # ==========================================================
    #
    # Missing metrics, or explicit None values, use automatic limits.

    profile_ylims = {
        "absolute": {
            # Weather: day-of-year climatology
            # "bias": (-1, 1),
            # "rmse": (0, 2),
            # "corr": (0.5, 1.0),

            # Weather: monthly climatology
            "bias": (-0.5, 0.5),
            "rmse": (0.8, 1.8), # only fc
            # "rmse": (0.5, 1.8),
            "corr": (0.8, 1.0), # only fc
            # "corr": (0.5, 1.0),
        },

        "percentage": {
            # "rmse": (-50, 50),
        },

        "difference": {
            # Weather: only FC
            "rmse": (0, 0.4),
        },

        "normalized": {
            # "rmse": (-1, 1),
        },
    }

    # ==========================================================
    # Individual experiment profiles
    # ==========================================================

    plot_individual_experiments = False

    # Plot individual ensemble members in addition to the aggregate.
    plot_members = False

    # ==========================================================
    # Lead-time profiles
    # ==========================================================
    #
    # x = lead time
    # one curve for each selected model
    # one figure for each selected climatological period.

    plot_leadtime_profiles = False

    # ==========================================================
    # Climatological profiles
    # ==========================================================
    #
    # x = climatological period, e.g. month
    # curves correspond to selected lead times.

    plot_climatological_profiles = False

    # If True, put all selected lead times in the same figure.
    # If False, generate one figure per lead time.
    combine_climatological_profile_leadtimes = True

    # How lead times are distinguished when combined:
    #   "shades" -> shades of each model color
    #   "colors" -> one distinct color per lead time
    climatological_leadtime_colors = "colors"

    # None -> use every available lead time.
    # wanted_climatological_profile_leadtimes = [24, 48, 72]
    # wanted_climatological_profile_leadtimes = [72]
    wanted_climatological_profile_leadtimes = None

    # ==========================================================
    # Combined-variable settings
    # ==========================================================

    # Plot one lead-time profile containing all selected variables.
    # Raw sources: "fc", "mlfc", "clim-fc".
    # MLFC-vs-FC improvements:
    #   "mlfc-fc-difference"
    #   "mlfc-fc-percentage"
    #   "mlfc-fc-normalized"
    plot_combined_variables = True

    # combined_variable_source = "fc"
    # combined_variable_source = "mlfc"
    # combined_variable_source = "clim-fc"

    # MLFC improvement over FC
    # combined_variable_source = "mlfc-fc-difference"
    combined_variable_source = "mlfc-fc-percentage"
    # combined_variable_source = "mlfc-fc-normalized"

    combined_variable_is_improvement = (
        combined_variable_source.startswith("mlfc-fc-")
    )

    combined_variable_plot_folder = "variable_comparison"
    combined_variable_plot_legend = True

    variable_labels = {
        "mslp": "MSLP",
        "t2m": "T2M",
        "d2m": "D2M",
        "u10": "U10",
        "v10": "V10",
        "tcc": "TCC",
    }

    variable_colors = {
        "mslp": "tab:blue",
        "t2m": "tab:red",
        "d2m": "tab:orange",
        "u10": "tab:green",
        "v10": "tab:purple",
        "tcc": "tab:brown",
    }

    # ==========================================================
    # Combined-experiment profiles
    # ==========================================================
    #
    # Compare several ML experiments for the same variable.

    plot_combined_experiments = False

    combined_plot_folder = "profile_comparison"

    comparison_name = "smaatunet-convnexttransformer"

    comparison_labels = [
        "ConvNeXt reanalysis",
        "ConvNeXt reanalysis ens mean",
        "SmaAt-UNet anomaly residual ens mean",
        "SmaAt-UNet anomaly residual",
    ]

    comparison_colors = {
        "fc": "tab:blue",
        "clim-fc": "tab:orange",

        "ConvNeXt reanalysis": "green",
        "ConvNeXt reanalysis ens mean": "green",

        "SmaAt-UNet anomaly residual ens mean": "red",
        "SmaAt-UNet anomaly residual": "red",
    }

    comparison_linestyles = {
        "fc": "-",
        "clim-fc": "-",

        "ConvNeXt reanalysis": "-",
        "ConvNeXt reanalysis ens mean": "--",

        "SmaAt-UNet anomaly residual ens mean": "--",
        "SmaAt-UNet anomaly residual": "-",
    }

    # ==========================================================
    # Model settings
    # ==========================================================

    plot_models = (
        "fc",
        # "mlfc",
        # "clim-fc",
    )

    model_comparisons = (
        # ("fc", "mlfc"),
        # ("clim-fc", "mlfc"),
        # ("fc", "clim-fc"),
    )

    need_clim_fc = (
        "clim-fc" in plot_models
        or any(
            "clim-fc" in comparison
            for comparison in model_comparisons
        )
        or (
            plot_combined_variables
            and combined_variable_source == "clim-fc"
        )
    )

    # ==========================================================
    # Data processing
    # ==========================================================

    interpolate = True # seasonal
    # interpolate = False # weather
    build_analysis = True

    recalculate_climatology = False

    # Compute metric datasets once before repeated plotting.
    materialize_once = False

    # ==========================================================
    # Climatology
    # ==========================================================

    clim_period: ClimPeriod = ClimPeriod.MONTH
    clim_rolling_window = None

    # Period grouping reference:
    #   "init"  -> group metrics by forecast initialization time
    #   "valid" -> group metrics by forecast valid time (init + lead time)
    period_reference = "valid"

    period_profile_label = get_period_profile_label(
        period_reference,
        clim_period,
    )

    # clim_period: ClimPeriod = ClimPeriod.DAYOFYEAR_HOUR
    # clim_rolling_window = 31

    # clim_period: ClimPeriod = ClimPeriod.DAYOFYEAR
    # clim_rolling_window = None

    # ==========================================================
    # Time selection
    # ==========================================================

    time_range = None
    # time_range = ("2018-01-01", "2022-12-31")

    inference_period = None
    # inference_period = ("2025-01-01", "2025-10-31")

    periods_requested = [
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
    # Metric aggregation
    # ==========================================================

    metric_agg_mode: MetricAgg = "spatial_avg"
    # "spatial_avg"
    # "global"

    leadtime_agg_mode: LeadtimeAgg = "single"
    # "single" for weather
    # "aggregated" for seasonal
    # "seasonal_window"

    leadtime_units = LeadtimeUnit.MONTHS # seasonal
    # leadtime_units = LeadtimeUnit.HOURS # weather

    leadtime_profile_label = f"Lead time [{leadtime_units.value}]"

    # ==========================================================
    # Metrics
    # ==========================================================

    metrics = [
        # ======================================================
        # Deterministic metrics - absolute fields
        # ======================================================
        "bias",
        # "mae",
        # "mse",
        "rmse",
        # "nrmse",
        # "corr",
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

        # ======================================================
        # Deterministic metrics - anomaly fields
        # ======================================================
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

        # ======================================================
        # Skill scores vs climatology
        # ======================================================
        # "mse_skill_clim",
        # "mae_anom_skill_clim",
        # "mse_anom_skill_clim",
        # "rmse_anom_skill_clim",
        # "ens_member_mse_anom_skill_clim",
        # "mean_member_mse_anom_skill_clim",

        # ======================================================
        # Ensemble / probabilistic metrics - absolute fields
        # ======================================================
        # "ens_member_rmse",
        # "mean_member_rmse",
        # "spread",
        # "spread_skill_ratio",
        # "crps",
        # "rank_histogram",

        # ======================================================
        # Ensemble / probabilistic metrics - anomaly fields
        # ======================================================
        # "ens_member_rmse_anom",
        # "mean_member_rmse_anom",
        # "spread_anom",
        # "spread_anom_skill_ratio",
        # "crps_anom",
        # "rank_histogram_anom",

        # ======================================================
        # ROC AUC - anomaly terciles
        # ======================================================
        # "roc_anom_lower",
        # "roc_anom_middle",
        # "roc_anom_upper",
    ]

    # ==========================================================
    # Variables and regions
    # ==========================================================

    variables = [
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

    regions = [
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
    lat_range = None
    lon_range = None

    # ==========================================================
    # Experiment selection
    # ==========================================================

    settings = get_experiment_configs(
        experiments_root,
        var_fc=variables,
        region_name=regions,

        # net_name="ConvNeXtTransformerUNet",
        net_name="SmaAt_UNet",

        target_mode="analysis",

        # seasonal_encoding=True,
        # ensemble_encoding=True,
        # input_realization_avg=True,
        # channel_representation="variable",

        loss_name="GeoMaskedMSELoss",
        # loss_name="GeoMaskedMSEMultiScaleLoss",
        # loss_name="SpatialDegradationMSELoss",

        # separate_training_by_init_period=None,
        # separate_training_by_init_period=ClimPeriod.MONTH,
        # pretrain_norm="full",

        # extra_suffix_folder="264samples_consecutive",
        # extra_suffix_folder="NOAA_copy",
        extra_suffix_folder="",
    )

    print(f"Found {len(settings)} matching experiment(s).")

    # Add auxiliary metrics required for improvement
    # representations, e.g. standard deviation for normalized
    # improvements.
    metrics_to_compute = get_required_improvement_metrics(
        list(metrics)
    )

    # ==========================================================
    # Combined-experiment storage
    # ==========================================================

    combined_groups = defaultdict(
        lambda: {
            "fc": None,
            "fc_members": None,
            "clim-fc": None,
            "clim-fc_members": None,
            "mlfc": [],
            "mlfc_members": [],
            "labels": [],
            "settings": [],
            "valid_time_range": None,
            "valid_lat_range": None,
            "valid_lon_range": None,
            "period_dim": None,
            "leadtime_agg_coord": None,
        }
    )

    # Same idea as combined_groups, but grouped across variables rather than
    # experiments. The grouping key contains everything that must be common
    # for curves to share one lead-time axis.
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

        # ======================================================
        # Time range
        # ======================================================

        if inference_period is None:
            valid_time_range = (
                (s.test_start, s.test_end)
                # (s.train_start, s.train_end)
                # (s.train_start, s.test_end)
                if time_range is None
                else time_range
            )

            mlfc_path = None

        else:
            valid_time_range = (
                inference_period
                if time_range is None
                else time_range
            )

            inference_start, inference_end = inference_period

            mlfc_path = (
                s.exp_dir
                / "inference"
                / f"{inference_start}_{inference_end}"
                / "test_corrected.zarr"
            )

        # clim_time_range = (s.train_start, s.val_end) # weather (very few years)
        clim_time_range = (s.train_start, s.train_end) # seasonal
        # clim_time_range = (s.train_start, s.test_end) # seasonal

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
            if lat_range is None
            else lat_range
        )

        valid_lon_range = (
            lat_lon[1]
            if lon_range is None
            else lon_range
        )

        # ======================================================
        # Dimension names
        # ======================================================

        leadtime_agg_coord = (
            "leadtime"
            if leadtime_agg_mode == "single"
            else "leadtime_seasonal"
        )

        period_dim = f"{period_reference}_{clim_period.value}"

        print(
            f"Generate {leadtime_agg_mode} {plot_mode} "
            f"for {(s.var_an, s.var_fc)} in {s.region_name} "
            f"(lon={valid_lon_range}, lat={valid_lat_range})"
        )

        # ======================================================
        # Data
        # ======================================================

        fc, an, mlfc = get_and_subset_datasets(
            s,
            leadtime_units=leadtime_units,
            lat_range=valid_lat_range,
            lon_range=valid_lon_range,
            time_range=valid_time_range,
            interpolate=interpolate,
            mlfc_path=mlfc_path,
        )

        if mlfc is not None:
            mlfc = mlfc.assign_coords(
                leadtime=s.leadtimes
            )

        # ======================================================
        # Climatologies
        # ======================================================

        fc_clim, an_clim, mlfc_clim = (
            calculate_save_and_subset_climatologies(
                s,
                leadtime_units=leadtime_units,
                force=recalculate_climatology,
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
        )

        if mlfc_clim is not None:
            mlfc_clim = mlfc_clim.assign_coords(
                leadtime=s.leadtimes
            )

        # ======================================================
        # Lead-time subset
        # ======================================================

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

        # ======================================================
        # Climatology-corrected forecast
        # ======================================================

        fc_clim_da = stack_hour_clim(
            fc_clim[s.var_fc],
            clim_period,
        )

        an_clim_da = stack_hour_clim(
            an_clim[s.var_an],
            clim_period,
        )

        fc_anom_da = (
            groupby_period(
                fc[s.var_fc],
                fc.earthml.guessed_dims.time,
                clim_period,
            )
            - fc_clim_da
        )

        clim_fc = (
            groupby_period(
                fc_anom_da,
                fc.earthml.guessed_dims.time,
                clim_period,
            )
            + an_clim_da
        ).to_dataset(
            name=s.var_fc
        )


        # ======================================================
        # Climatology realization handling
        # ======================================================

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

        # ======================================================
        # Available models
        # ======================================================

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

        if need_clim_fc:
            datasets["clim-fc"] = (
                clim_fc,
                an_clim_for_fc,
            )


        if plot_mode not in {
            "profiles",
            "all",
        }:
            continue

        # ======================================================
        # Metrics to compute
        # ======================================================

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

        members_by_model: dict[
            str,
            xr.Dataset,
        ] = {}

        # ======================================================
        # Compute metrics
        # ======================================================

        for model, (
            ds,
            ds_clim,
        ) in datasets.items():

            if ds is None or ds_clim is None:
                continue

            print(
                f"Get {model} profile metrics"
            )

            metric_parts = []

            # --------------------------------------------------
            # Deterministic / ensemble-mean metrics
            # --------------------------------------------------

            if deterministic_metrics:
                metric_parts.append(
                    get_profile_metrics(
                        s=s,
                        an=an,
                        fc=ds,
                        an_clim=an_clim,
                        fc_clim=ds_clim,
                        metrics=deterministic_metrics,
                        realization_agg=True,
                        metric_agg_mode=metric_agg_mode,
                        leadtime_agg_mode=leadtime_agg_mode,
                        leadtime_agg_coord=leadtime_agg_coord,
                        clim_period=clim_period,
                        period_reference=period_reference,
                        period_dim=period_dim,
                        periods_requested=periods_requested,
                        leadtime_unit=leadtime_units,
                    )
                )

            # --------------------------------------------------
            # Probabilistic metrics
            # --------------------------------------------------

            if probabilistic_metrics:
                metric_parts.append(
                    get_profile_metrics(
                        s=s,
                        an=an,
                        fc=ds,
                        an_clim=an_clim,
                        fc_clim=ds_clim,
                        metrics=probabilistic_metrics,
                        realization_agg=False,
                        metric_agg_mode=metric_agg_mode,
                        leadtime_agg_mode=leadtime_agg_mode,
                        leadtime_agg_coord=leadtime_agg_coord,
                        clim_period=clim_period,
                        period_reference=period_reference,
                        period_dim=period_dim,
                        periods_requested=periods_requested,
                        leadtime_unit=leadtime_units,
                    )
                )

            if metric_parts:
                metrics_by_model[model] = xr.merge(
                    metric_parts
                )

            # --------------------------------------------------
            # Individual-member deterministic metrics
            # --------------------------------------------------

            if (
                plot_members
                and deterministic_metrics
            ):
                members_by_model[model] = (
                    get_profile_metrics(
                        s=s,
                        an=an,
                        fc=ds,
                        an_clim=an_clim,
                        fc_clim=ds_clim,
                        metrics=deterministic_metrics,
                        realization_agg=False,
                        metric_agg_mode=metric_agg_mode,
                        leadtime_agg_mode=leadtime_agg_mode,
                        leadtime_agg_coord=leadtime_agg_coord,
                        clim_period=clim_period,
                        period_reference=period_reference,
                        period_dim=period_dim,
                        periods_requested=periods_requested,
                        leadtime_unit=leadtime_units,
                    )
                )

        if not metrics_by_model:
            print(
                f"No metrics available for "
                f"{s.output_name}"
            )
            continue

        # ======================================================
        # Optional materialization
        # ======================================================

        if materialize_once:
            with ProgressBar():

                metrics_by_model = {
                    model: ds.compute()
                    for model, ds
                    in metrics_by_model.items()
                }

                members_by_model = {
                    model: ds.compute()
                    for model, ds
                    in members_by_model.items()
                }

        # ======================================================
        # Available periods
        # ======================================================

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
            if plot_leadtime_profiles
            else []
        )

        climatological_profile_leadtimes = (
            get_climatological_profile_leadtimes(
                reference_ds,
                leadtime_dim=leadtime_agg_coord,
                wanted_leadtimes=(
                    wanted_climatological_profile_leadtimes
                ),
            )
            if plot_climatological_profiles
            else []
        )

        # ======================================================
        # Individual experiment plots
        # ======================================================

        if plot_individual_experiments:

            # ==================================================
            # Absolute profiles
            # ==================================================

            available_models = tuple(
                model
                for model in plot_models
                if model in metrics_by_model
            )
            available_metrics = [
                metric
                for metric in metrics
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
                        / metric_agg_mode
                    )

                    filename = (
                        f"{s.var_fc}_"
                        f"{metric}_"
                        f"{leadtime_agg_mode}lt.png"
                    )

                    out_file = (
                        s.plot_dir
                        / common_path
                        / filename
                    )

                    if (
                        out_file.exists()
                        and not regenerate_plots
                    ):
                        continue

                    print(
                        f"Saving profile "
                        f"{out_file}"
                    )

                    plot_profile(
                        das=das,
                        var=s.var_fc,
                        metric=metric,
                        select_value=period_value,
                        models=available_models,
                        labels=tuple(
                            model_labels.get(model, model)
                            for model in available_models
                        ),
                        out_file=out_file,
                        time_range=valid_time_range,
                        das_member=das_member,
                        profile_dim=leadtime_agg_coord,
                        profile_label=leadtime_profile_label,
                        select_dim=period_dim,
                        select_unit=clim_period.value,
                        realization_dim="realization",
                        spread="std",
                        plot_single_members=plot_members,
                        plot_title=plot_title,
                        title_strftime=title_strftime,
                        plot_legend=plot_legend,
                        title_size=title_size,
                        label_size=label_size,
                        tick_size=tick_size,
                        dpi=dpi,
                        ylim=get_profile_ylim(
                            profile_ylims,
                            metric,
                            variant="absolute",
                        ),
                    )

                    n += 1

                # ----------------------------------------------
                # Climatological-period profile
                # ----------------------------------------------

                if plot_climatological_profiles:
                    climatological_das = [
                        prepare_climatological_profile_da(
                            da,
                            period_dim=period_dim,
                            clim_period=clim_period,
                        )
                        for da in das
                    ]

                    climatological_das_member = [
                        prepare_climatological_profile_da(
                            da,
                            period_dim=period_dim,
                            clim_period=clim_period,
                        )
                        for da in das_member
                    ]

                    leadtime_groups = (
                        [climatological_profile_leadtimes]
                        if combine_climatological_profile_leadtimes
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
                            / metric_agg_mode
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
                            and not regenerate_plots
                        ):
                            continue

                        print(
                            "Saving climatological profile "
                            f"{out_file}"
                        )

                        plot_das = climatological_das
                        plot_models_current = available_models
                        plot_labels_current = tuple(
                            model_labels.get(model, model)
                            for model in available_models
                        )
                        plot_das_member = climatological_das_member
                        plot_colors = None
                        plot_linestyles = None
                        plot_select_dim = leadtime_agg_coord
                        plot_select_value = lead_values[0]
                        plot_select_unit = leadtime_units.value

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
                                leadtime_unit=leadtime_units.value,
                                color_mode=climatological_leadtime_colors,
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
                            realization_dim="realization",
                            spread="std",
                            plot_single_members=plot_members,
                            plot_title=plot_title,
                            title_strftime=title_strftime,
                            plot_legend=plot_legend,
                            title_size=title_size,
                            label_size=label_size,
                            tick_size=tick_size,
                            dpi=dpi,
                            model_colors=plot_colors,
                            model_linestyles=plot_linestyles,
                            ylim=get_profile_ylim(
                                profile_ylims,
                                metric,
                                variant="absolute",
                            ),
                        )

                        n += 1

            # ==================================================
            # Model-comparison profiles
            # ==================================================

            for (
                baseline_model,
                target_model,
            ) in model_comparisons:

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

                baseline_ds = (
                    metrics_by_model[
                        baseline_model
                    ]
                )

                target_ds = (
                    metrics_by_model[
                        target_model
                    ]
                )

                comparison_metrics = [
                    metric
                    for metric in metrics
                    if (
                        metric != "rank_histogram"
                        and metric in baseline_ds
                        and metric in target_ds
                    )
                ]

                for metric in comparison_metrics:

                    improvements = (
                        build_metric_improvements(
                            baseline_ds,
                            target_ds,
                            metric=metric,
                            baseline_model=baseline_model,
                            target_model=target_model,
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
                            comparison_labels_map,
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
                                / metric_agg_mode
                            )

                            filename = (
                                f"{s.var_fc}_"
                                f"{metric}_"
                                f"{safe_label(comparison_model)}_"
                                f"{leadtime_agg_mode}lt.png"
                            )

                            out_file = (
                                s.plot_dir
                                / common_path
                                / filename
                            )

                            if (
                                out_file.exists()
                                and not regenerate_plots
                            ):
                                continue

                            print(
                                "Saving comparison "
                                f"profile {out_file}"
                            )

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
                                profile_label=leadtime_profile_label,
                                select_value=period_value,
                                select_dim=period_dim,
                                select_unit=clim_period.value,
                                realization_dim="realization",
                                spread="std",
                                plot_single_members=False,
                                plot_title=plot_title,
                                title_strftime=title_strftime,
                                plot_legend=plot_legend,
                                title_size=title_size,
                                label_size=label_size,
                                tick_size=tick_size,
                                dpi=dpi,
                                improvement_unit=improvement_unit,
                                ylim=get_profile_ylim(
                                    profile_ylims,
                                    metric,
                                    variant=variant,
                                ),
                            )

                            n += 1

                        # --------------------------------------
                        # Climatological-period profile
                        # --------------------------------------

                        if plot_climatological_profiles:
                            climatological_comparison_da = (
                                prepare_climatological_profile_da(
                                    comparison_da,
                                    period_dim=period_dim,
                                    clim_period=clim_period,
                                )
                            )

                            leadtime_groups = (
                                [climatological_profile_leadtimes]
                                if combine_climatological_profile_leadtimes
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
                                    / metric_agg_mode
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
                                    and not regenerate_plots
                                ):
                                    continue

                                print(
                                    "Saving climatological "
                                    "comparison profile "
                                    f"{out_file}"
                                )

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
                                plot_select_unit = leadtime_units.value

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
                                        leadtime_unit=leadtime_units.value,
                                        color_mode=climatological_leadtime_colors,
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
                                    realization_dim="realization",
                                    spread="std",
                                    plot_single_members=False,
                                    plot_title=plot_title,
                                    title_strftime=title_strftime,
                                    plot_legend=plot_legend,
                                    title_size=title_size,
                                    label_size=label_size,
                                    tick_size=tick_size,
                                    dpi=dpi,
                                    improvement_unit=improvement_unit,
                                    model_colors=plot_colors,
                                    model_linestyles=plot_linestyles,
                                    ylim=get_profile_ylim(
                                        profile_ylims,
                                        metric,
                                        variant=variant,
                                    ),
                                )

                                n += 1

        # ======================================================
        # Accumulate combined variables
        # ======================================================

        if plot_combined_variables:

            combined_variable_metrics = None
            combined_variable_variant = "absolute"
            combined_variable_improvement_unit = None

            if combined_variable_source in metrics_by_model:
                combined_variable_metrics = metrics_by_model[
                    combined_variable_source
                ]

            elif combined_variable_is_improvement:
                requested_variant = combined_variable_source.removeprefix(
                    "mlfc-fc-"
                )

                if requested_variant not in IMPROVEMENT_UNITS:
                    raise ValueError(
                        "Unsupported combined-variable improvement "
                        f"variant {requested_variant!r}."
                    )

                if {"fc", "mlfc"} <= metrics_by_model.keys():
                    combined_variable_parts = {}

                    for metric in metrics:
                        if (
                            metric == "rank_histogram"
                            or metric not in metrics_by_model["fc"]
                            or metric not in metrics_by_model["mlfc"]
                        ):
                            continue

                        improvements = build_metric_improvements(
                            metrics_by_model["fc"],
                            metrics_by_model["mlfc"],
                            metric=metric,
                            baseline_model="fc",
                            target_model="mlfc",
                        )

                        matching = [
                            da
                            for name, da in improvements.items()
                            if get_improvement_variant(name) == requested_variant
                        ]

                        if matching:
                            combined_variable_parts[metric] = matching[0]

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
                    f"{combined_variable_source!r}. Expected a raw model "
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
                    leadtime_agg_mode,
                    metric_agg_mode,
                    str(clim_period),
                    period_reference,
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
                leadtime_agg_mode,
                metric_agg_mode,
                str(clim_period),
                period_reference,
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


            # FC is common to experiments in the same group.
            if (
                group["fc"] is None
                and "fc" in metrics_by_model
            ):
                group["fc"] = (
                    metrics_by_model["fc"]
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
    # Combined-variable lead-time profile plots
    # ==========================================================

    if plot_combined_variables:

        for variable_group in combined_variable_groups.values():

            metrics_by_variable = variable_group["metrics"]

            if not metrics_by_variable:
                continue

            variables_current = tuple(
                var
                for var in variables
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
                for metric in metrics
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
                    variable_labels.get(var, var)
                    for var in variables_current
                )

                colors = {
                    var: variable_colors[var]
                    for var in variables_current
                    if var in variable_colors
                }

                for period_value in available_periods:

                    common_path = (
                        Path("profiles")
                        / combined_variable_plot_folder
                        / safe_label(combined_variable_source)
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
                        / metric_agg_mode
                    )

                    filename = (
                        f"all_variables_"
                        f"{combined_variable_source}_"
                        f"{metric}_"
                        f"{leadtime_agg_mode}lt.png"
                    )

                    out_file = (
                        common_plot_dir
                        / common_path
                        / filename
                    )

                    if (
                        out_file.exists()
                        and not regenerate_plots
                    ):
                        continue

                    print(
                        "Saving combined-variable profile "
                        f"{out_file}"
                    )

                    plot_profile(
                        das=das,
                        # plot_profile still expects a variable name for
                        # metric/unit formatting. The plotted curves are
                        # identified by models/labels below.
                        var=variables_current[0],
                        metric=metric,
                        models=variables_current,
                        labels=labels,
                        out_file=out_file,
                        time_range=valid_time_range,
                        das_member=[None] * len(das),
                        profile_dim=leadtime_agg_coord,
                        profile_label=leadtime_profile_label,
                        select_value=period_value,
                        select_dim=period_dim,
                        select_unit=clim_period.value,
                        realization_dim="realization",
                        spread="std",
                        plot_single_members=False,
                        plot_title=plot_title,
                        title_strftime=title_strftime,
                        plot_legend=combined_variable_plot_legend,
                        title_size=title_size,
                        label_size=label_size,
                        tick_size=tick_size,
                        dpi=dpi,
                        improvement_unit=(
                            combined_variable_improvement_unit
                        ),
                        model_colors=colors,
                        ylim=get_profile_ylim(
                            profile_ylims,
                            metric,
                            variant=combined_variable_variant,
                        ),
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

            common_s = (
                group["settings"][0]
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

            members_by_combined_model: dict[
                str,
                xr.Dataset,
            ] = {}


            if group["fc"] is not None:
                das_by_model["fc"] = (
                    group["fc"]
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
                len(comparison_labels)
                != len(group["mlfc"])
            ):
                raise ValueError(
                    "comparison_labels must contain "
                    "one label for each combined "
                    "experiment. "
                    f"Got {len(comparison_labels)} "
                    f"labels for "
                    f"{len(group['mlfc'])} "
                    "experiments."
                )

            for (
                experiment_label,
                mlfc_metrics,
                mlfc_members,
            ) in zip(
                comparison_labels,
                group["mlfc"],
                group["mlfc_members"],
                strict=True,
            ):
                das_by_model[
                    experiment_label
                ] = mlfc_metrics

                if mlfc_members is not None:
                    members_by_combined_model[
                        experiment_label
                    ] = mlfc_members

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
                if plot_leadtime_profiles
                else []
            )

            climatological_profile_leadtimes = (
                get_climatological_profile_leadtimes(
                    reference_ds,
                    leadtime_dim=leadtime_agg_coord,
                    wanted_leadtimes=wanted_climatological_profile_leadtimes,
                )
                if plot_climatological_profiles
                else []
            )

            # ==================================================
            # Absolute combined profiles
            # ==================================================

            available_models = tuple(
                model
                for model in das_by_model
                if (
                    model in plot_models
                    or (
                        "mlfc" in plot_models
                        and model in comparison_labels
                    )
                )
            )

            available_metrics = [
                metric
                for metric in metrics
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

                for period_value in leadtime_profile_periods:

                    common_path = (
                        Path("profiles")
                        / combined_plot_folder
                        / comparison_name
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
                        / metric_agg_mode
                    )

                    filename = (
                        f"{common_s.var_fc}_"
                        f"{metric}_"
                        f"{leadtime_agg_mode}lt.png"
                    )

                    out_file = (
                        common_plot_dir
                        / common_path
                        / filename
                    )

                    if (
                        out_file.exists()
                        and not regenerate_plots
                    ):
                        continue

                    print(
                        "Saving combined profile "
                        f"{out_file}"
                    )

                    plot_profile(
                        das=das,
                        var=common_s.var_fc,
                        metric=metric,
                        models=available_models,
                        labels=tuple(
                            model_labels.get(model, model)
                            for model in available_models
                        ),
                        out_file=out_file,
                        time_range=valid_time_range,
                        das_member=das_member,
                        profile_dim=leadtime_agg_coord,
                        profile_label=leadtime_profile_label,
                        select_value=period_value,
                        select_dim=period_dim,
                        select_unit=clim_period.value,
                        realization_dim="realization",
                        spread="std",
                        plot_single_members=plot_members,
                        plot_title=plot_title,
                        title_strftime=title_strftime,
                        plot_legend=plot_legend,
                        title_size=title_size,
                        label_size=label_size,
                        tick_size=tick_size,
                        dpi=dpi,
                        model_colors=comparison_colors,
                        model_linestyles=comparison_linestyles,
                        ylim=get_profile_ylim(
                            profile_ylims,
                            metric,
                            variant="absolute",
                        ),
                    )

                    n += 1

                # ----------------------------------------------
                # Climatological-period profile
                # ----------------------------------------------

                if plot_climatological_profiles:
                    climatological_das = [
                        prepare_climatological_profile_da(
                            da,
                            period_dim=period_dim,
                            clim_period=clim_period,
                        )
                        for da in das
                    ]

                    climatological_das_member = [
                        prepare_climatological_profile_da(
                            da,
                            period_dim=period_dim,
                            clim_period=clim_period,
                        )
                        for da in das_member
                    ]

                    leadtime_groups = (
                        [climatological_profile_leadtimes]
                        if combine_climatological_profile_leadtimes
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
                            / combined_plot_folder
                            / comparison_name
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
                            / metric_agg_mode
                        )

                        filename = (
                            f"{common_s.var_fc}_"
                            f"{metric}_"
                            f"{safe_label(period_dim)}_"
                            f"{leadtime_label}.png"
                        )

                        out_file = (
                            common_plot_dir
                            / common_path
                            / filename
                        )

                        if (
                            out_file.exists()
                            and not regenerate_plots
                        ):
                            continue

                        print(
                            "Saving combined climatological "
                            f"profile {out_file}"
                        )

                        plot_das = climatological_das
                        plot_models_current = available_models
                        plot_labels_current = tuple(
                            model_labels.get(model, model)
                            for model in available_models
                        )
                        plot_das_member = climatological_das_member
                        plot_colors = comparison_colors
                        plot_linestyles = comparison_linestyles
                        plot_select_dim = leadtime_agg_coord
                        plot_select_value = lead_values[0]
                        plot_select_unit = leadtime_units.value

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
                                leadtime_unit=leadtime_units.value,
                                color_mode=climatological_leadtime_colors,
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
                            realization_dim="realization",
                            spread="std",
                            plot_single_members=plot_members,
                            plot_title=plot_title,
                            title_strftime=title_strftime,
                            plot_legend=plot_legend,
                            title_size=title_size,
                            label_size=label_size,
                            tick_size=tick_size,
                            dpi=dpi,
                            model_colors=plot_colors,
                            model_linestyles=plot_linestyles,
                            ylim=get_profile_ylim(
                                profile_ylims,
                                metric,
                                variant="absolute",
                            ),
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
            ) in model_comparisons:

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
                        das_by_model[
                            baseline_model
                        ]
                    )

                    for metric in metrics:

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
                        ) in comparison_labels:

                            if (
                                experiment_label
                                not in das_by_model
                            ):
                                continue

                            target_ds = (
                                das_by_model[
                                    experiment_label
                                ]
                            )

                            if (
                                metric
                                not in target_ds
                            ):
                                continue

                            improvements = (
                                build_metric_improvements(
                                    baseline_ds,
                                    target_ds,
                                    metric=metric,
                                    baseline_model=(
                                        baseline_model
                                    ),
                                    target_model=(
                                        experiment_label
                                    ),
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

                            for (
                                period_value
                            ) in leadtime_profile_periods:

                                common_path = (
                                    Path("profiles")
                                    / combined_plot_folder
                                    / comparison_name
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
                                    / metric_agg_mode
                                )

                                filename = (
                                    f"{common_s.var_fc}_"
                                    f"{metric}_"
                                    f"{safe_label(comparison_variant)}_"
                                    f"{leadtime_agg_mode}lt.png"
                                )

                                out_file = (
                                    common_plot_dir
                                    / common_path
                                    / filename
                                )

                                if (
                                    out_file.exists()
                                    and not regenerate_plots
                                ):
                                    continue

                                print(
                                    "Saving combined "
                                    "comparison profile "
                                    f"{out_file}"
                                )

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
                                    profile_label=leadtime_profile_label,
                                    select_value=period_value,
                                    select_dim=period_dim,
                                    select_unit=clim_period.value,
                                    realization_dim="realization",
                                    spread="std",
                                    plot_single_members=False,
                                    plot_title=plot_title,
                                    title_strftime=title_strftime,
                                    plot_legend=plot_legend,
                                    title_size=title_size,
                                    label_size=label_size,
                                    tick_size=tick_size,
                                    dpi=dpi,
                                    improvement_unit=improvement_unit,
                                    model_colors=comparison_colors,
                                    model_linestyles=comparison_linestyles,
                                    ylim=get_profile_ylim(
                                        profile_ylims,
                                        metric,
                                        variant=variant,
                                    ),
                                )

                                n += 1

                            # ----------------------------------
                            # Climatological-period profiles
                            # ----------------------------------

                            if plot_climatological_profiles:
                                climatological_comparison_das = [
                                    prepare_climatological_profile_da(
                                        da,
                                        period_dim=period_dim,
                                        clim_period=clim_period,
                                    )
                                    for da in comparison_das
                                ]

                                leadtime_groups = (
                                    [climatological_profile_leadtimes]
                                    if combine_climatological_profile_leadtimes
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
                                        / combined_plot_folder
                                        / comparison_name
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
                                        / metric_agg_mode
                                    )

                                    filename = (
                                        f"{common_s.var_fc}_"
                                        f"{metric}_"
                                        f"{safe_label(comparison_variant)}_"
                                        f"{safe_label(period_dim)}_"
                                        f"{leadtime_label}.png"
                                    )

                                    out_file = (
                                        common_plot_dir
                                        / common_path
                                        / filename
                                    )

                                    if (
                                        out_file.exists()
                                        and not regenerate_plots
                                    ):
                                        continue

                                    print(
                                        "Saving combined climatological "
                                        "comparison profile "
                                        f"{out_file}"
                                    )

                                    plot_das = climatological_comparison_das
                                    plot_models_current = comparison_models
                                    plot_labels_current = comparison_models
                                    plot_colors = comparison_colors
                                    plot_linestyles = comparison_linestyles
                                    plot_select_dim = leadtime_agg_coord
                                    plot_select_value = lead_values[0]
                                    plot_select_unit = leadtime_units.value

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
                                            leadtime_unit=leadtime_units.value,
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
                                        realization_dim="realization",
                                        spread="std",
                                        plot_single_members=False,
                                        plot_title=plot_title,
                                        title_strftime=title_strftime,
                                        plot_legend=plot_legend,
                                        title_size=title_size,
                                        label_size=label_size,
                                        tick_size=tick_size,
                                        dpi=dpi,
                                        improvement_unit=improvement_unit,
                                        model_colors=plot_colors,
                                        model_linestyles=plot_linestyles,
                                        ylim=get_profile_ylim(
                                            profile_ylims,
                                            metric,
                                            variant=variant,
                                        ),
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
                        das_by_model[
                            baseline_model
                        ]
                    )

                    target_ds = (
                        das_by_model[
                            target_model
                        ]
                    )

                    comparison_metrics = [
                        metric
                        for metric in metrics
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
                            build_metric_improvements(
                                baseline_ds,
                                target_ds,
                                metric=metric,
                                baseline_model=baseline_model,
                                target_model=target_model,
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
                                comparison_labels_map,
                            )

                            for (
                                period_value
                            ) in leadtime_profile_periods:

                                common_path = (
                                    Path("profiles")
                                    / combined_plot_folder
                                    / comparison_name
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
                                    / metric_agg_mode
                                )

                                filename = (
                                    f"{common_s.var_fc}_"
                                    f"{metric}_"
                                    f"{safe_label(comparison_model)}_"
                                    f"{leadtime_agg_mode}lt.png"
                                )

                                out_file = (
                                    common_plot_dir
                                    / common_path
                                    / filename
                                )

                                if (
                                    out_file.exists()
                                    and not regenerate_plots
                                ):
                                    continue

                                print(
                                    "Saving combined "
                                    "comparison profile "
                                    f"{out_file}"
                                )

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
                                    profile_label=leadtime_profile_label,
                                    select_value=period_value,
                                    select_dim=period_dim,
                                    select_unit=clim_period.value,
                                    realization_dim="realization",
                                    spread="std",
                                    plot_single_members=False,
                                    plot_title=plot_title,
                                    title_strftime=title_strftime,
                                    plot_legend=plot_legend,
                                    title_size=title_size,
                                    label_size=label_size,
                                    tick_size=tick_size,
                                    dpi=dpi,
                                    improvement_unit=improvement_unit,
                                    ylim=get_profile_ylim(
                                        profile_ylims,
                                        metric,
                                        variant=variant,
                                    ),
                                )

                                n += 1

                            # ----------------------------------
                            # Climatological-period profile
                            # ----------------------------------

                            if plot_climatological_profiles:
                                climatological_comparison_da = (
                                    prepare_climatological_profile_da(
                                        comparison_da,
                                        period_dim=period_dim,
                                        clim_period=clim_period,
                                    )
                                )

                                leadtime_groups = (
                                    [climatological_profile_leadtimes]
                                    if combine_climatological_profile_leadtimes
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
                                        / combined_plot_folder
                                        / comparison_name
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
                                        / metric_agg_mode
                                    )

                                    filename = (
                                        f"{common_s.var_fc}_"
                                        f"{metric}_"
                                        f"{safe_label(comparison_model)}_"
                                        f"{safe_label(period_dim)}_"
                                        f"{leadtime_label}.png"
                                    )

                                    out_file = (
                                        common_plot_dir
                                        / common_path
                                        / filename
                                    )

                                    if (
                                        out_file.exists()
                                        and not regenerate_plots
                                    ):
                                        continue

                                    print(
                                        "Saving combined climatological "
                                        "comparison profile "
                                        f"{out_file}"
                                    )

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
                                    plot_select_unit = leadtime_units.value

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
                                            leadtime_unit=leadtime_units.value,
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
                                        realization_dim="realization",
                                        spread="std",
                                        plot_single_members=False,
                                        plot_title=plot_title,
                                        title_strftime=title_strftime,
                                        plot_legend=plot_legend,
                                        title_size=title_size,
                                        label_size=label_size,
                                        tick_size=tick_size,
                                        dpi=dpi,
                                        improvement_unit=improvement_unit,
                                        model_colors=plot_colors,
                                        ylim=get_profile_ylim(
                                            profile_ylims,
                                            metric,
                                            variant=variant,
                                        ),
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


def get_profile_metrics(
    *,
    s,
    an,
    fc,
    an_clim,
    fc_clim,
    metrics,
    realization_agg,
    metric_agg_mode,
    leadtime_agg_mode,
    leadtime_agg_coord,
    clim_period,
    period_reference,
    period_dim,
    periods_requested,
    leadtime_unit,
):
    metric_kind = (
        "scalar"
        if metric_agg_mode == "global"
        else "maps"
    )

    ds = get_metrics(
        an=an,
        fc=fc,
        var=s.var_fc,
        metric_kind=metric_kind,
        leadtime_agg=leadtime_agg_mode,
        realization_agg=realization_agg,
        an_clim=an_clim,
        fc_clim=fc_clim,
        metrics=metrics,
        leadtime_windows=s.seasonal_leadtime_windows,
        leadtime_agg_coord=leadtime_agg_coord,
        clim_period=clim_period,
        period_reference=period_reference,
        period_dim=period_dim,
        periods_requested=periods_requested,
        leadtime_unit=leadtime_unit,
    )

    if metric_agg_mode == "global":
        return ds

    if metric_agg_mode == "spatial_avg":
        lat_dim = fc.earthml.guessed_dims.latitude
        lon_dim = fc.earthml.guessed_dims.longitude

        weights = np.cos(np.deg2rad(fc[lat_dim]))

        return ds.weighted(weights).mean(
            dim=(lat_dim, lon_dim)
        )

    raise ValueError(
        f"Unsupported metric_agg_mode={metric_agg_mode!r}"
    )


def get_improvement_variant(
    comparison_name: str,
) -> str:
    """
    Extract the improvement representation from a model-comparison
    name generated by build_metric_improvements().
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


def expand_climatological_leadtimes(
    das,
    models,
    lead_values,
    *,
    leadtime_dim: str,
    leadtime_unit: str | None,
    color_mode: str = "shades",
    das_member=None,
    labels=None,
    model_colors: dict[str, object] | None = None,
    model_linestyles: dict[str, str] | None = None,
):
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

    if color_mode not in {"shades", "colors"}:
        raise ValueError(
            f"Unsupported color_mode={color_mode!r}. "
            "Choose 'shades' or 'colors'."
        )

    default_colors = matplotlib.rcParams[
        "axes.prop_cycle"
    ].by_key()["color"]

    shade_fractions = np.linspace(
        0.4,
        1.0,
        len(lead_values),
    )

    unit = (
        leadtime_unit[:1].upper()
        if leadtime_unit
        else ""
    )

    expanded_das = []
    expanded_models = []
    expanded_labels = []
    expanded_das_member = []
    expanded_colors = {}
    expanded_linestyles = {}

    for model_index, (
        model,
        display_model,
        da,
        da_member,
    ) in enumerate(
        zip(
            models,
            labels,
            das,
            das_member,
            strict=True,
        )
    ):
        base_color = (
            model_colors[model]
            if model_colors is not None
            and model in model_colors
            else default_colors[
                model_index % len(default_colors)
            ]
        )

        base_rgb = np.asarray(
            to_rgb(base_color)
        )

        linestyle = (
            model_linestyles.get(model, "-")
            if model_linestyles is not None
            else "-"
        )

        for lead_index, lead_value in enumerate(
            lead_values
        ):
            if color_mode == "shades":
                fraction = shade_fractions[
                    lead_index
                ]

                color = tuple(
                    1.0
                    - (1.0 - base_rgb)
                    * fraction
                )

            else:
                color = default_colors[
                    lead_index % len(default_colors)
                ]

            label = (
                f"{model} · {lead_value}{unit}"
            )

            display_label = (
                f"{display_model} · "
                f"{lead_value}{unit}"
            )

            expanded_das.append(
                da.sel(
                    {
                        leadtime_dim:
                            lead_value
                    }
                )
            )

            expanded_models.append(
                label
            )

            expanded_labels.append(
                display_label
            )

            expanded_das_member.append(
                None
                if da_member is None
                else da_member.sel(
                    {
                        leadtime_dim:
                            lead_value
                    }
                )
            )

            expanded_colors[
                label
            ] = color

            expanded_linestyles[
                label
            ] = linestyle

    return (
        expanded_das,
        tuple(expanded_models),
        tuple(expanded_labels),
        expanded_das_member,
        expanded_colors,
        expanded_linestyles,
    )


if __name__ == "__main__":
    main()
