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
    Settings,
    get_experiment_configs,
)
from earthml.metrics import (
    is_deterministic,
    is_probabilistic,
    get_metrics,
    LeadtimeAgg,
)
from earthml.plots import (
    safe_label,
    lead_label,
    PlotMode,
    plot_map,
    plot_profile,
    plot_rank_histogram,
    plot_timeseries
)

from plot_scatter import get_scalar_metrics

from settings_plot_seasonal import VARIABLE_PLOT_CONFIG, IMPROVEMENT_PLOT_CONFIG


def main() -> None:
    experiments_root = Path("/Users/jacopodallaglio/ML/training/seasonal/experiments")

    plot_mode: PlotMode = "profiles"

    plot_mlfc = True

    force_clim_recalc = False
    interpolate = True
    build_analysis = True

    metrics = [
        # ==========================================================
        # Deterministic Metrics (Absolute Fields)
        # ==========================================================
        # "bias",
        # "mae",
        # "mse",
        # "rmse",
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
        "rmse_anom",
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
        "t2m",
        "d2m",
        "u10",
        # "v10",
        # "sst",
        # "tprate",
        # Ocean
        # "mlotst",
        # "ssh",
        # "sss",
        # "t20d",
    ]
    regions = ["World"] # World, ConUS or None to accept all

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
        # "01",
        "05",
        # "08",
        # "10",
        "all",
    ]

    leadtime_units = "months"
    clim_period = "month" # "dayofyear", "day", "month", "year", "day_hour", "dayofyear_hour", "month_hour"
    clim_rolling_window = None

    time_range = None
    # time_range = ("2018-01-01", "2022-12-31")

    metric_agg_mode = "spatial_avg" # "spatial_avg", "global", "spatial_rmse"
    leadtime_agg_mode = "single" # "single", "aggregated", "seasonal_window"
    plot_members = True

    settings = get_experiment_configs(experiments_root, variables, regions)

    print(f"Found {len(settings)} matching experiment(s).")

    n = 0
    for s in settings:
        valid_time_range = (s.train_start, s.test_end) if time_range is None else time_range
        clim_time_range = (s.train_start, s.test_end)

        lat_lon = list(s.region.values()) if s.region is not None else [None, None]
        valid_lat_range = lat_lon[0] if lat_range is None else lat_range
        valid_lon_range = lat_lon[1] if lon_range is None else lon_range

        leadtime_agg_coord = "leadtime" if leadtime_agg_mode=="single" else "leadtime_seasonal"

        print(f"Generate {leadtime_agg_mode} {plot_mode} for {s.var_an, s.var_fc} in {s.region_name} (lon={valid_lon_range}, lat={valid_lat_range})")


        if plot_mode in {"profiles", "all"}:
            deterministic_metrics = [m for m in metrics if is_deterministic(m)]
            probabilistic_metrics = [m for m in metrics if is_probabilistic(m)]

            metrics_fc_det, metrics_mlfc_det = xr.Dataset(), xr.Dataset()
            metrics_fc_ds_members, metrics_mlfc_ds_members = xr.Dataset(), xr.Dataset()
            metrics_fc_prob, metrics_mlfc_prob = xr.Dataset(), xr.Dataset()
            if len(deterministic_metrics) != 0:
                metrics_fc_det, metrics_mlfc_det = get_scalar_metrics(
                    s=s,
                    fc_metrics=deterministic_metrics,
                    mlfc_metrics=deterministic_metrics,
                    metric_agg_mode=metric_agg_mode,
                    leadtime_agg=leadtime_agg_mode,
                    realization_agg=True,
                    lat_range=lat_range,
                    lon_range=lon_range,
                    time_range=time_range,
                    clim_period=clim_period,
                    clim_rolling_window=clim_rolling_window,
                    clim_time_range=clim_time_range,
                    leadtime_units=leadtime_units,
                    force_clim_recalc=force_clim_recalc,
                    period_dim=f"start_{leadtime_units}",
                    wanted_start_periods=wanted_start_periods,
                    interpolate=interpolate,
                    build_analysis=build_analysis,
                )

                if plot_members:
                    metrics_fc_ds_members, metrics_mlfc_ds_members = get_scalar_metrics(
                        s=s,
                        fc_metrics=deterministic_metrics,
                        mlfc_metrics=deterministic_metrics,
                        metric_agg_mode=metric_agg_mode,
                        leadtime_agg=leadtime_agg_mode,
                        realization_agg=False,
                        lat_range=lat_range,
                        lon_range=lon_range,
                        time_range=time_range,
                        clim_period=clim_period,
                        clim_rolling_window=clim_rolling_window,
                        clim_time_range=clim_time_range,
                        leadtime_units=leadtime_units,
                        force_clim_recalc=force_clim_recalc,
                        period_dim=f"start_{leadtime_units}",
                        wanted_start_periods=wanted_start_periods,
                        interpolate=interpolate,
                        build_analysis=build_analysis,
                    )
            if len(probabilistic_metrics) != 0:
                metrics_fc_prob, metrics_mlfc_prob = get_scalar_metrics(
                    s=s,
                    fc_metrics=probabilistic_metrics,
                    mlfc_metrics=probabilistic_metrics,
                    metric_agg_mode=metric_agg_mode,
                    leadtime_agg=leadtime_agg_mode,
                    realization_agg=False,
                    lat_range=lat_range,
                    lon_range=lon_range,
                    time_range=time_range,
                    clim_period=clim_period,
                    clim_rolling_window=clim_rolling_window,
                    clim_time_range=clim_time_range,
                    leadtime_units=leadtime_units,
                    force_clim_recalc=force_clim_recalc,
                    period_dim=f"start_{leadtime_units}",
                    wanted_start_periods=wanted_start_periods,
                    interpolate=interpolate,
                    build_analysis=build_analysis,
                )

            metrics_fc_ds = xr.merge([metrics_fc_det, metrics_fc_prob])
            metrics_mlfc_ds = xr.merge([metrics_mlfc_det, metrics_mlfc_prob])

            available_metrics = [
                str(x) for x in metrics_fc_ds.data_vars
                if str(x) in metrics and str(x) != "rank_histogram"
            ]

            start_periods = [
                str(x) for x in metrics_fc_ds[f"start_{leadtime_units}"].values
                if str(x) in wanted_start_periods
            ]

            print(f"Plotting metric profiles {available_metrics} for periods {start_periods} for exp {s.output_name}")

            for m in available_metrics:
                for start_period in start_periods:
                    out_file = (
                        s.plot_dir / "profiles"
                        / safe_label(start_period)
                        / f"time_{safe_label(valid_time_range)}_lat_{safe_label(lat_range)}_lon_{safe_label(lon_range)}"
                        / m
                        / metric_agg_mode
                        / f"{s.var_fc}_{m}_{leadtime_agg_mode}lt.png"
                    )

                    print(f"Saving profile {out_file}")

                    plot_profile(
                        das=[metrics_fc_ds[m], metrics_mlfc_ds[m]],
                        var=s.var_fc,
                        metric=m,
                        start_period=start_period,
                        models=("fc", "mlfc"),
                        out_file=out_file,
                        time_range=valid_time_range,
                        das_member=[metrics_fc_ds_members[m], metrics_mlfc_ds_members[m]],
                        leadtime_dim=leadtime_agg_coord,
                        leadtime_unit=leadtime_units,
                        period_dim=f"start_{leadtime_units}",
                        realization_dim="realization",
                        spread="std",
                    )

                    n += 1

    print(f"Done. Saved {n} plots.")


if __name__ == "__main__":
    main()
