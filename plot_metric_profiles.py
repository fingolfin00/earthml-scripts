from pathlib import Path

import numpy as np
import xarray as xr

import warnings
from dask.array import PerformanceWarning
warnings.simplefilter("ignore", FutureWarning)
warnings.filterwarnings(
    "ignore",
    category=PerformanceWarning,
)

from dask.diagnostics.progress import ProgressBar

import earthml
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
)
from earthml.plots import (
    safe_label,
    lead_label,
    PlotMode,
    plot_profile,
)


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
    period_dim,
    wanted_start_periods,
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
        period_dim=period_dim,
        periods_requested=wanted_start_periods,
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
        f"Unsupported metric_agg_mode={metric_agg_mode}"
    )


def main() -> None:
    experiments_root = Path("/Users/jacopodallaglio/ML/training/seasonal/experiments")

    plot_mode: PlotMode = "profiles"
    regenerate_plots = False

    force_clim_recalc = False
    interpolate = True
    build_analysis = True
    materialize_once = False

    metrics = [
        # ==========================================================
        # Deterministic Metrics (Absolute Fields)
        # ==========================================================
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

        # ==========================================================
        # Deterministic Metrics (Anomaly Fields)
        # ==========================================================
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

        # ==========================================================
        # Skill Scores vs Climatology
        # ==========================================================
        # "mse_skill_clim",
        # "mae_anom_skill_clim",
        # "mse_anom_skill_clim",
        # "rmse_anom_skill_clim",
        # "ens_member_mse_anom_skill_clim",
        # "mean_member_mse_anom_skill_clim",

        # ==========================================================
        # Ensemble / Probabilistic Metrics (Absolute Fields)
        # ==========================================================
        # "ens_member_rmse",
        # "mean_member_rmse",
        # "spread",
        # "spread_skill_ratio",
        # "crps",
        # "rank_histogram",

        # ==========================================================
        # Ensemble / Probabilistic Metrics (Anomaly Fields)
        # ==========================================================
        # "ens_member_rmse_anom",
        # "mean_member_rmse_anom",
        # "spread_anom",
        # "spread_anom_skill_ratio",
        # "crps_anom",
        # "rank_histogram_anom",

        # ==========================================================
        # ROC AUC (Anomaly Terciles)
        # ==========================================================
        # "roc_anom_lower",
        # "roc_anom_middle",
        # "roc_anom_upper",
    ]

    variables = [
        # Atmo
        "mslp",
        # "t2m",
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
        # None, # accept all
    ]

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

    wanted_start_periods = [
        "01",
        "05",
        "08",
        "10",
        "all",
    ]

    leadtime_units = LeadtimeUnit.MONTHS
    clim_period: ClimPeriod = ClimPeriod.MONTH # "dayofyear", "day", "month", "year", "day_hour", "dayofyear_hour", "month_hour"
    clim_rolling_window = None

    time_range = None
    # time_range = ("2018-01-01", "2022-12-31")

    metric_agg_mode: MetricAgg = "spatial_avg" # "spatial_avg", "global", "spatial_rmse"
    leadtime_agg_mode: LeadtimeAgg = "single" # "single", "aggregated", "seasonal_window"
    plot_members = True

    settings = get_experiment_configs(
        experiments_root,
        var_fc=variables,
        region_name=regions,
        # net_name="SmaAt_UNet",
        net_name="ConvNeXtTransformerUNet",
        # target_mode="anomaly_residual",
        # extra_suffix_folder="random_split",
    )

    print(f"Found {len(settings)} matching experiment(s).")

    n = 0
    for s in settings:
        valid_time_range = (s.train_start, s.train_end) if time_range is None else time_range
        # valid_time_range = (s.test_start, s.test_end) if time_range is None else time_range
        clim_time_range = (s.train_start, s.train_end)

        lat_lon = list(s.region.values()) if s.region is not None else [None, None]
        valid_lat_range = lat_lon[0] if lat_range is None else lat_range
        valid_lon_range = lat_lon[1] if lon_range is None else lon_range

        leadtime_agg_coord = "leadtime" if leadtime_agg_mode=="single" else "leadtime_seasonal"

        print(f"Generate {leadtime_agg_mode} {plot_mode} for {s.var_an, s.var_fc} in {s.region_name} (lon={valid_lon_range}, lat={valid_lat_range})")

        fc, an, mlfc = get_and_subset_datasets(
            s,
            leadtime_units=leadtime_units,
            lat_range=valid_lat_range,
            lon_range=valid_lon_range,
            time_range=valid_time_range,
            interpolate=interpolate,
        )

        if mlfc is not None:
            mlfc = mlfc.assign_coords(leadtime=s.leadtimes)

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

        if mlfc_clim is not None:
            mlfc_clim = mlfc_clim.assign_coords(
                leadtime=s.leadtimes
            )

        leadtime_dim = fc.earthml.guessed_dims.leadtime

        fc = fc.sel({leadtime_dim: s.leadtimes})
        an = an.sel({leadtime_dim: s.leadtimes})
        fc_clim = fc_clim.sel({leadtime_dim: s.leadtimes})
        an_clim = an_clim.sel({leadtime_dim: s.leadtimes})

        # ----------------------------------------------------------
        # Forecast corrected only by replacing forecast climatology
        # with analysis climatology.
        # ----------------------------------------------------------

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
        ).to_dataset(name=s.var_fc)

        # Analysis climatology is the appropriate forecast
        # climatology for clim-fc.
        an_clim_for_fc = an_clim

        realization_dim = fc.earthml.guessed_dims.realization

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

        datasets = {
            "fc": (fc, fc_clim),
            "clim-fc": (clim_fc, an_clim_for_fc),
            "mlfc": (mlfc, mlfc_clim),
        }

        if plot_mode in {"profiles", "all"}:
            deterministic_metrics = [
                m for m in metrics
                if is_deterministic(m)
            ]

            probabilistic_metrics = [
                m for m in metrics
                if is_probabilistic(m)
            ]

            metrics_by_model: dict[str, xr.Dataset] = {}
            members_by_model: dict[str, xr.Dataset] = {}

            for model, (ds, ds_clim) in datasets.items():
                if ds is None or ds_clim is None:
                    continue

                print(f"Get {model} profile metrics")

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
                            period_dim=f"start_{leadtime_units}",
                            wanted_start_periods=wanted_start_periods,
                        )
                    )

                # --------------------------------------------------
                # Probabilistic metrics require realizations
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
                            period_dim=f"start_{leadtime_units}",
                            wanted_start_periods=wanted_start_periods,
                        )
                    )

                if metric_parts:
                    metrics_by_model[model] = xr.merge(metric_parts)

                # --------------------------------------------------
                # Individual-member deterministic metrics
                # --------------------------------------------------

                if plot_members and deterministic_metrics:
                    members_by_model[model] = get_profile_metrics(
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
                        period_dim=f"start_{leadtime_units}",
                        wanted_start_periods=wanted_start_periods,
                    )

            if not metrics_by_model:
                print(f"No metrics available for {s.output_name}")
                continue

            if materialize_once:
                with ProgressBar():
                    metrics_by_model = {
                        model: ds.compute()
                        for model, ds in metrics_by_model.items()
                    }

                    members_by_model = {
                        model: ds.compute()
                        for model, ds in members_by_model.items()
                    }

            # Keep the requested model order.
            available_models = tuple(
                model
                for model in datasets.keys()
                if model in metrics_by_model
            )

            # Metric must be available for every plotted model.
            available_metrics = [
                metric
                for metric in metrics
                if metric != "rank_histogram"
                and all(
                    metric in metrics_by_model[model].data_vars
                    for model in available_models
                )
            ]

            reference_ds = metrics_by_model[available_models[0]]

            start_periods = [
                str(x)
                for x in reference_ds[
                    f"start_{leadtime_units}"
                ].values
                if str(x) in wanted_start_periods
            ]

            print(
                f"Plotting metric profiles {available_metrics} "
                f"for periods {start_periods} "
                f"for exp {s.output_name}"
            )

            for metric in available_metrics:
                das = [
                    metrics_by_model[model][metric]
                    for model in available_models
                ]

                das_member = []

                for model in available_models:
                    member_ds = members_by_model.get(model)

                    if (
                        member_ds is not None
                        and metric in member_ds.data_vars
                    ):
                        das_member.append(member_ds[metric])
                    else:
                        das_member.append(None)

                for start_period in start_periods:
                    out_file = (
                        s.plot_dir
                        / "profiles"
                        / safe_label(start_period)
                        / (
                            f"time_{safe_label(valid_time_range)}"
                            f"_lat_{safe_label(valid_lat_range)}"
                            f"_lon_{safe_label(valid_lon_range)}"
                        )
                        / metric
                        / metric_agg_mode
                        / (
                            f"{s.var_fc}_{metric}_"
                            f"{leadtime_agg_mode}lt.png"
                        )
                    )

                    if out_file.exists() and not regenerate_plots:
                        continue

                    print(f"Saving profile {out_file}")

                    plot_profile(
                        das=das,
                        var=s.var_fc,
                        metric=metric,
                        start_period=start_period,
                        models=available_models,
                        out_file=out_file,
                        time_range=valid_time_range,
                        das_member=das_member,
                        leadtime_dim=leadtime_agg_coord,
                        leadtime_unit=leadtime_units,
                        period_dim=f"start_{leadtime_units}",
                        realization_dim="realization",
                        spread="std",
                        plot_single_members=plot_members,
                    )

                    n += 1

    print(f"Done. Saved {n} plots.")


if __name__ == "__main__":
    main()
