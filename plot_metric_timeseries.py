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
)
from earthml.plots import (
    safe_label,
    lead_label,
    plot_timeseries,
)


def main() -> None:
    # ==========================================================
    # Paths
    # ==========================================================

    # exp_name = "weather_atmo"
    exp_name = "weather_atmo_ablation_fixed_val"
    # exp_name = "weather_atmo_short_zero_vs_replicate_padding"

    experiments_root = Path(f"/work/cmcc/jd19424/ML/MLBC/experiments/{exp_name}")

    orography_path = Path("/work/cmcc/jd19424/ML/MLBC/data/orography/era5_orography.zarr")

    # ==========================================================
    # Plot settings
    # ==========================================================

    metric_kind: MetricKind = "timeseries"

    regenerate_plots = True

    plot_models = (
        "fc",
        "mlfc",
    )

    # ==========================================================
    # Model comparisons
    # ==========================================================

    model_comparisons = (
        ("fc", "mlfc"),
    )

    # ==========================================================
    # Data processing
    # ==========================================================

    interpolate = False
    build_analysis = True

    recalculate_climatology = False

    # ==========================================================
    # Climatology
    # ==========================================================

    clim_period: ClimPeriod = ClimPeriod.MONTH
    clim_rolling_window = None

    period_reference: Literal["init", "valid"] = "init"

    # ==========================================================
    # Time selection
    # ==========================================================

    time_range = None
    inference_period = None

    periods_requested = [
        "all",
    ]

    # ==========================================================
    # Lead-time aggregation
    # ==========================================================

    leadtime_units = LeadtimeUnit.HOURS

    leadtime_agg_mode: LeadtimeAgg = "single"

    # ==========================================================
    # Metrics
    # ==========================================================

    metrics = [
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

    # metrics_to_compute = get_required_improvement_metrics(
    #     list(metrics)
    # )
    metrics_to_compute = list(metrics)

    # ==========================================================
    # Variables and regions
    # ==========================================================

    variables = [
        "t2m",
    ]

    regions = [
        "ConUS",
    ]

    # ==========================================================
    # Spatial subset
    # ==========================================================

    lat_range = (50, 25)
    lon_range = (-130, -60)

    # ==========================================================
    # Experiment selection
    # ==========================================================

    settings = get_experiment_configs(
        experiments_root,
        var_fc=variables,
        region_name=regions,
        net_name="SmaAt_UNet",
        target_mode="analysis",
        loss_name="GeoMaskedMSELoss",
        train_start="2019-10-14",
    )

    print(f"Found {len(settings)} matching experiment(s).")

    n = 0

    for s in settings:
        if inference_period is None:
            valid_time_range = (
                # (s.test_start, s.test_end)
                (s.train_start, s.test_end)
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

        clim_time_range = (
            s.train_start,
            s.train_end,
            # s.val_end,
        )

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

        leadtime_agg_coord = (
            "leadtime"
            if leadtime_agg_mode == "single"
            else "leadtime_seasonal"
        )

        period_dim = (
            f"{period_reference}_{clim_period.value}"
        )

        print(
            f"Generate {leadtime_agg_mode} timeseries "
            f"grouped by {period_dim} for "
            f"{s.var_an, s.var_fc} in {s.region_name} "
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

        (
            fc_clim,
            an_clim,
            mlfc_clim,
        ) = calculate_save_and_subset_climatologies(
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

        if mlfc_clim is not None:
            mlfc_clim = mlfc_clim.assign_coords(
                leadtime=s.leadtimes
            )

        leadtime_dim = fc.earthml.guessed_dims.leadtime

        fc = fc.sel({leadtime_dim: s.leadtimes})
        an = an.sel({leadtime_dim: s.leadtimes})

        fc_clim = fc_clim.sel(
            {leadtime_dim: s.leadtimes}
        )
        an_clim = an_clim.sel(
            {leadtime_dim: s.leadtimes}
        )

        # ======================================================
        # Climatological forecast
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
                an.earthml.guessed_dims.time,
                clim_period,
            )
            - fc_clim_da
        )

        clim_fc = (
            groupby_period(
                fc_anom_da,
                an.earthml.guessed_dims.time,
                clim_period,
            )
            + an_clim_da
        ).to_dataset(name=s.var_fc)

        realization_dim = (
            fc.earthml.guessed_dims.realization
        )

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
        # Calculate timeseries
        # ======================================================

        metric_ts_by_model: dict[
            str,
            xr.Dataset,
        ] = {}

        for model in plot_models:
            ds = model_datasets[model]
            ds_clim = model_climatologies[model]

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
                    f"Get {model} deterministic "
                    f"metric timeseries"
                )

                metric_ts_det = get_metrics(
                    an=an,
                    fc=ds,
                    var=s.var_fc,
                    metric_kind=metric_kind,
                    leadtime_agg=leadtime_agg_mode,
                    realization_agg=True,
                    an_clim=an_clim,
                    fc_clim=ds_clim,
                    orography_path=orography_path,
                    metrics=deterministic_metrics,
                    leadtime_windows=(
                        s.seasonal_leadtime_windows
                    ),
                    leadtime_agg_coord=(
                        leadtime_agg_coord
                    ),
                    clim_period=clim_period,
                    period_reference=period_reference,
                    period_dim=period_dim,
                    periods_requested=periods_requested,
                    leadtime_unit=leadtime_units,
                    align=False,
                    fair_correction=False,
                )

            if probabilistic_metrics:
                print(
                    f"Get {model} probabilistic "
                    f"metric timeseries"
                )

                metric_ts_prob = get_metrics(
                    an=an,
                    fc=ds,
                    var=s.var_fc,
                    metric_kind=metric_kind,
                    leadtime_agg=leadtime_agg_mode,
                    realization_agg=False,
                    an_clim=an_clim,
                    fc_clim=ds_clim,
                    orography_path=orography_path,
                    metrics=probabilistic_metrics,
                    leadtime_windows=(
                        s.seasonal_leadtime_windows
                    ),
                    leadtime_agg_coord=(
                        leadtime_agg_coord
                    ),
                    clim_period=clim_period,
                    period_reference=period_reference,
                    period_dim=period_dim,
                    periods_requested=periods_requested,
                    leadtime_unit=leadtime_units,
                    align=False,
                    fair_correction=False,
                )

            metric_ts_by_model[model] = xr.merge(
                [
                    metric_ts_det,
                    metric_ts_prob,
                ]
            )

        # ======================================================
        # Plot FC and MLFC together
        # ======================================================

        available_models = [
            model
            for model in plot_models
            if model in metric_ts_by_model
        ]

        if not available_models:
            continue

        available_metrics = [
            m
            for m in metrics
            if all(
                m in metric_ts_by_model[model]
                for model in available_models
            )
        ]

        print(
            f"Plotting metrics {available_metrics} "
            f"for models {available_models}"
        )

        for m in available_metrics:
            das = [
                metric_ts_by_model[model][m]
                for model in available_models
            ]

            for lead_value in das[0][
                leadtime_agg_coord
            ].values:
                label = safe_label(
                    lead_label(
                        das[0],
                        lead_value,
                        leadtime_agg_coord,
                    )
                )

                common_path = (
                    Path("timeseries")
                    / safe_label(period_dim)
                    / safe_label("all")
                    / (
                        f"time_{safe_label(valid_time_range)}"
                        f"_lat_{safe_label(valid_lat_range)}"
                        f"_lon_{safe_label(valid_lon_range)}"
                    )
                    / m
                    / leadtime_agg_mode
                )

                filename = (
                    f"{s.var_fc}_{m}_"
                    f"{'-'.join(available_models)}"
                    f"_lead_{label}.png"
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
                    f"Saving timeseries {out_file}"
                )

                plot_timeseries(
                    das,
                    var=s.var_fc,
                    metric=m,
                    models=available_models,
                    lead_value=lead_value,
                    out_file=out_file,
                    time_range=valid_time_range,
                    leadtime_dim=leadtime_agg_coord,
                )

                n += 1

    print(f"Done. Saved {n} plots.")


if __name__ == "__main__":
    main()
