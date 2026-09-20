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

    orography_path = Path(
        "/Users/jacopodallaglio/ML/training/seasonal/data/input/era5_orography.zarr"
        # "/work/cmcc/jd19424/ML/MLBC/data/orography/era5_orography.zarr"
    )

    # ==========================================================
    # Plot settings
    # ==========================================================

    plot_mode: PlotMode = "maps"
    metric_kind: MetricKind = "maps"

    plot_type: Literal["pcolormesh", "contourf"] = "contourf"

    plot_title = True
    plot_labels = True

    title_size = None
    label_size = None
    tick_size = None
    dpi = 300

    title_strftime = "%Y" # seasonal
    # title_strftime = "%m.%Y" # weather

    plot_models = (
        "fc",
        # "clim-fc",
        # "mlfc",
    )

    regenerate_plots = (
        # "fc",
        # "clim-fc",
        # "mlfc",
        # "improvement",
    )

    # ==========================================================
    # Model comparisons
    # ==========================================================

    model_comparisons = (
        ("fc", "mlfc"),
        # ("clim-fc", "mlfc"),
        # ("fc", "clim-fc"),
    )

    # ==========================================================
    # Statistical significance
    # ==========================================================

    plot_significance = False

    significance_n_bootstrap = 200
    significance_block_size = 1
    significance_confidence_level = 0.95
    significance_seed = 42

    significance_stride = 3
    significance_size = 2.0
    significance_alpha = 0.6

    # ==========================================================
    # Data processing
    # ==========================================================

    interpolate = True # seasonal
    # interpolate = False # weather
    build_analysis = True

    recalculate_climatology = False

    # ==========================================================
    # Climatology
    # ==========================================================

    clim_period: ClimPeriod = ClimPeriod.MONTH
    clim_rolling_window = None

    # clim_period: ClimPeriod = ClimPeriod.DAYOFYEAR_HOUR
    # clim_rolling_window = 31

    # Period grouping reference:
    #   "init"  -> group by forecast initialization time
    #   "valid" -> group by forecast valid time (init + lead time)
    period_reference: Literal["init", "valid"] = "init"

    # ==========================================================
    # Time selection
    # ==========================================================

    time_range = None
    # time_range = ("2018-01-01", "2022-12-31")

    inference_period = None
    # inference_period = ("2025-01-01", "2025-10-10")
    # inference_period = ("2025-01-01", "2025-05-12")

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
    # Lead-time aggregation
    # ==========================================================

    leadtime_units = LeadtimeUnit.MONTHS # seasonal
    # leadtime_units = LeadtimeUnit.HOURS # weather

    leadtime_agg_mode: LeadtimeAgg = "aggregated"
    # "single" for weather
    # "aggregated". for seasonal
    # "seasonal_window"

    hovmoller_time_agg: ClimPeriod | None = ClimPeriod.MONTH

    # ==========================================================
    # Metrics
    # ==========================================================

    metrics = [
        # ======================================================
        # Orography
        # ======================================================
        # "orography",
        # "orography_grad_mag",

        # ======================================================
        # Deterministic metrics - absolute fields
        # ======================================================
        # "bias",
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

        # test_end="2025-10-01",

        # target_mode="anomaly",

        # seasonal_encoding=True,
        # ensemble_encoding=True,

        # channel_representation="variable",

        # loss_name="VarNormMaskMSELoss",
        # loss_name="GeoMaskedMSEMultiScaleLoss",
        # loss_name="SpatialDegradationMSELoss",

        # separate_training_by_init_period=None,
        # separate_training_by_init_period=ClimPeriod.MONTH,

        # extra_suffix_folder="264samples_randomsamples",
        # extra_suffix_folder="264samples_consecutive",
        # extra_suffix_folder="NOAA_copy",
        # extra_suffix_folder="",
    )

    print(f"Found {len(settings)} matching experiment(s).")

    # Add std to compute normalized diff metrics
    metrics_to_compute = get_required_improvement_metrics(list(metrics))

    n = 0
    for s in settings:
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
        clim_time_range = (s.train_start, s.train_end)
        # clim_time_range = (s.train_start, s.val_end)
        # clim_time_range = (s.train_start, s.test_end)
        # clim_time_range = ("2019-10-14", "2024-12-31")

        lat_lon = list(s.region.values()) if s.region is not None else [None, None]
        valid_lat_range = lat_lon[0] if lat_range is None else lat_range
        valid_lon_range = lat_lon[1] if lon_range is None else lon_range

        leadtime_agg_coord = "leadtime" if leadtime_agg_mode=="single" else "leadtime_seasonal"
        period_dim = f"{period_reference}_{clim_period.value}"

        if period_reference == "valid" and leadtime_agg_mode == "aggregated":
            raise ValueError(
                "period_reference='valid' is not supported with "
                "leadtime_agg_mode='aggregated' because an aggregated "
                "forecast has no unique valid time. Use 'single' or "
                "'seasonal_window'."
            )

        if period_reference == "valid" and plot_significance:
            raise ValueError(
                "Valid-time grouping is not yet supported by "
                "get_metric_improvement_significance(). Disable "
                "plot_significance or extend the significance helper with "
                "period_reference and leadtime_unit support first."
            )

        print(
            f"Generate {leadtime_agg_mode} {plot_mode} "
            f"grouped by {period_dim} for {s.var_an, s.var_fc} "
            f"in {s.region_name} (lon={valid_lon_range}, lat={valid_lat_range})"
        )

        fc, an, mlfc = get_and_subset_datasets(
            s,
            leadtime_units=leadtime_units,
            lat_range=valid_lat_range,
            lon_range=valid_lon_range,
            time_range=valid_time_range,
            interpolate=interpolate,
            mlfc_path=mlfc_path,
        )
        mlfc = mlfc.assign_coords(leadtime=s.leadtimes) if mlfc is not None else None

        fc_clim, an_clim, mlfc_clim = calculate_save_and_subset_climatologies(
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
        mlfc_clim = mlfc_clim.assign_coords(leadtime=s.leadtimes) if mlfc_clim is not None else None

        leadtime_dim = fc.earthml.guessed_dims.leadtime
        fc = fc.sel({leadtime_dim: s.leadtimes})
        an = an.sel({leadtime_dim: s.leadtimes})
        fc_clim = fc_clim.sel({leadtime_dim: s.leadtimes})
        an_clim = an_clim.sel({leadtime_dim: s.leadtimes})

        # fc corrected with analysis clim
        fc_clim_da = stack_hour_clim(fc_clim[s.var_fc], clim_period)
        an_clim_da = stack_hour_clim(an_clim[s.var_an], clim_period)

        fc_anom_da = groupby_period(fc[s.var_fc], fc.earthml.guessed_dims.time, clim_period) - fc_clim_da
        clim_fc = (groupby_period(fc_anom_da, fc.earthml.guessed_dims.time, clim_period) + an_clim_da).to_dataset(name=s.var_fc)

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
            "fc": common_plot_dir,
            "clim-fc": common_plot_dir,
            "mlfc": s.plot_dir,
        }

        metric_maps_by_model: dict[str, xr.Dataset] = {}
        significance_by_comparison: dict[tuple[str, str, str], xr.Dataset] = {}

        for model in plot_models:
            ds = model_datasets[model]
            ds_clim = model_climatologies[model]

            if ds is None or ds_clim is None:
                continue

            if plot_mode in {"maps", "all"}:
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
                        metric_kind=metric_kind,
                        leadtime_agg=leadtime_agg_mode, # "single", "aggregated", "seasonal_window"
                        realization_agg=True,
                        an_clim=an_clim,
                        fc_clim=ds_clim,
                        orography_path=orography_path,
                        metrics=deterministic_metrics,
                        leadtime_windows=s.seasonal_leadtime_windows,
                        leadtime_agg_coord=leadtime_agg_coord,
                        clim_period=clim_period,
                        period_reference=period_reference,
                        period_dim=period_dim,
                        periods_requested=periods_requested,
                        leadtime_unit=leadtime_units,
                        align=False,
                        fair_correction=False,
                    )
                if len(probabilistic_metrics) != 0:
                    print(f"Get {model} probabilistic metric maps")
                    metric_maps_prob = get_metrics(
                        an=an,
                        fc=ds,
                        var=s.var_fc,
                        metric_kind=metric_kind,
                        leadtime_agg=leadtime_agg_mode, # "single", "aggregated", "seasonal_window"
                        realization_agg=False,
                        an_clim=an_clim,
                        fc_clim=ds_clim,
                        orography_path=orography_path,
                        metrics=probabilistic_metrics,
                        leadtime_windows=s.seasonal_leadtime_windows,
                        leadtime_agg_coord=leadtime_agg_coord,
                        clim_period=clim_period,
                        period_reference=period_reference,
                        period_dim=period_dim,
                        periods_requested=periods_requested,
                        leadtime_unit=leadtime_units,
                        align=False,
                        fair_correction=False,
                    )

                metric_maps = xr.merge([metric_maps_det, metric_maps_prob])

                if metric_kind in {"time_lon", "time_lat"}:
                    time_dim = metric_maps.earthml.guessed_dims.time

                    if time_dim is None or time_dim not in metric_maps.dims:
                        raise ValueError(
                            f"Cannot calculate monthly Hovmöller: "
                            f"no time dimension found in {metric_maps.dims}"
                        )

                    # TODO support _hour groups
                    metric_maps = metric_maps.groupby(f"{time_dim}.{hovmoller_time_agg}").mean(time_dim, skipna=True)

                # =====================================================
                # Statistical significance of model comparisons
                # =====================================================

                metric_maps_by_model[model] = metric_maps

                if (
                    plot_significance
                    and metric_kind == "maps"
                ):
                    for baseline_model, target_model in model_comparisons:

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
                            for m in metrics
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
                                leadtime_agg=leadtime_agg_mode,
                                realization_agg=is_deterministic(m),
                                fc_clim=baseline_clim_ds,
                                mlfc_clim=target_clim_ds,
                                an_clim=an_clim,
                                leadtime_windows=s.seasonal_leadtime_windows,
                                leadtime_agg_coord=leadtime_agg_coord,
                                clim_period=clim_period,
                                period_dim=period_dim,
                                periods_requested=periods_requested,
                                n_bootstrap=significance_n_bootstrap,
                                block_size=significance_block_size,
                                confidence_level=significance_confidence_level,
                                seed=significance_seed,
                                align=False,
                                fair_correction=False,
                            )

                available_metrics = [
                    str(x) for x in metric_maps.data_vars
                    if str(x) in metrics and str(x) != "rank_histogram"
                ]

                available_periods = [
                    str(x) for x in metric_maps[period_dim].values
                    if str(x) in periods_requested
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
                    ] = {
                        model: (
                            metric_maps[m],
                            None,
                            None,
                        ),
                    }

                    # ==========================================================
                    # Build requested model comparisons
                    # ==========================================================

                    for baseline_model, target_model in model_comparisons:

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
                                    Path(metric_kind)
                                    / safe_label(period_dim)
                                    / safe_label(period_value)
                                    / (
                                        f"time_{safe_label(valid_time_range)}"
                                        f"_lat_{safe_label(lat_range)}"
                                        f"_lon_{safe_label(lon_range)}"
                                    )
                                    / m
                                    / leadtime_agg_mode
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
                                    plot_significance
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
                                    "improvement" in regenerate_plots
                                    if is_improvement
                                    else model in regenerate_plots
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
                                        leadtime_units=leadtime_units,
                                        period_dim=period_dim,
                                        clim_period=(
                                            None
                                            if metric_kind == "map"
                                            else hovmoller_time_agg
                                        ),
                                        var_plot_config=VARIABLE_PLOT_CONFIG,
                                        impro_plot_config=IMPROVEMENT_PLOT_CONFIG,
                                        plot_kind=metric_kind,
                                        plot_type=plot_type,
                                        plot_title=plot_title,
                                        plot_labels=plot_labels,
                                        title_size=title_size,
                                        label_size=label_size,
                                        tick_size=tick_size,
                                        dpi=dpi,
                                        title_strftime=title_strftime,
                                        significance=None,
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
                                            leadtime_units=leadtime_units,
                                            period_dim=period_dim,
                                            clim_period=(
                                                None
                                                if metric_kind == "map"
                                                else hovmoller_time_agg
                                            ),
                                            var_plot_config=VARIABLE_PLOT_CONFIG,
                                            impro_plot_config=IMPROVEMENT_PLOT_CONFIG,
                                            plot_kind=metric_kind,
                                            plot_type=plot_type,
                                            plot_title=plot_title,
                                            plot_labels=plot_labels,
                                            title_size=title_size,
                                            label_size=label_size,
                                            tick_size=tick_size,
                                            dpi=dpi,
                                            title_strftime=title_strftime,
                                            significance=significance,
                                            significance_stride=(
                                                significance_stride
                                            ),
                                            significance_size=(
                                                significance_size
                                            ),
                                            significance_alpha=(
                                                significance_alpha
                                            ),
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



    # if plot_mode in {"histograms", "all"} and "rank_histogram" in metrics:
    #     for s in settings:
    #         time_range = (s.train_start, s.test_end)
    #         metric_scalar_members = calculate_metric_kind(
    #             s,
    #             metric_kind="scalar",
    #             leadtime_agg="leadtime_month",
    #             realization_agg="member",
    #             lat_range=lat_range,
    #             lon_range=lon_range,
    #             time_range=time_range,
    #         )["leadtime_month"]["scalar"]

    #         da_members_fc = metric_scalar_members["fc"]["rank_histogram"]
    #         da_members_mlfc = metric_scalar_members["mlfc"]["rank_histogram"]

    #         out_file = (
    #             s.plot_dir / "histograms" / s.var_fc / "leadtime_month"
    #             / "all_months"
    #             / f"time_{safe_label(time_range)}_lat_{safe_label(lat_range)}_lon_{safe_label(lon_range)}"
    #             / f"{s.var_fc}_rank_histogram_fc-mlfc_startmonth_all.png"
    #         )

    #         print(f"Saving rank histogram {out_file}")

    #         plot_rank_histogram(
    #             [da_members_fc, da_members_mlfc],
    #             var=s.var_fc,
    #             metric="rank_histogram",
    #             period_value="all",
    #             models=["fc", "mlfc"],
    #             out_file=out_file,
    #             time_range=time_range,
    #         )
    #         n += 1

    print(f"Done. Saved {n} plots.")


if __name__ == "__main__":
    main()
