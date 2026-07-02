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
    get_and_subset_datasets,
)
from earthml.metrics import (
    is_deterministic,
    is_probabilistic,
    get_metrics,
    calculate_save_and_subset_climatologies,
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


# =============================================================================
# Colormaps and plotting configuration
# =============================================================================

VARIABLE_PLOT_CONFIG = {
    "mslp": {
        "bias": {
            "vmin": -12,
            "vmax": 12,
            "ticks": [-12, -10, -8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8, 10, 12],
        },
        "mae": {
            "vmin": 0,
            "vmax": 10,
            "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10],
        },
        "rmse": {
            "vmin": 0,
            "vmax": 10,
            "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10],
        },
        "rmse_anom": {
            "vmin": 0,
            "vmax": 9,
            "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        },
        "ens_member_rmse": {
            "vmin": 0,
            "vmax": 9,
            "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        },
        "ens_member_rmse_anom": {
            "vmin": 0,
            "vmax": 9,
            "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        },
        "mean_member_rmse_anom": {
            "vmin": 0,
            "vmax": 9,
            "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        },
    },

    "t2m": {
        "bias": {
            "vmin": -6,
            "vmax": 6,
            "ticks": [-6, -4, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 4, 6],
        },
        "mae": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 1.5, 2, 4, 6],
        },
        "rmse": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6],
        },
        "rmse_anom": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6],
        },
        "crps": {
            "vmin": 0,
            "vmax": 8,
            "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8],
        },
    },

    "u10": {
        "bias": {
            "vmin": -3,
            "vmax": 3,
            "ticks": [-3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3],
        },
        "mae": {
            "vmin": 0,
            "vmax": 5,
            "ticks": [0, 0.5, 1, 2, 3, 4, 5],
        },
        "rmse": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6],
        },
        "rmse_anom": {
            "vmin": 0,
            "vmax": 3,
            "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3],
        },
    },

    "v10": {
        "bias": {
            "vmin": -3,
            "vmax": 3,
            "ticks": [-4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3],
        },
        "mae": {
            "vmin": 0,
            "vmax": 5,
            "ticks": [0, 0.5, 1, 2, 3, 4, 5],
        },
        "rmse": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6],
        },
        "rmse_anom": {
            "vmin": 0,
            "vmax": 3,
            "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3],
        },
    },

    "sst": {
        "bias": {
            "vmin": -6,
            "vmax": 6,
            "ticks": [-6, -4, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 4, 6],
        },
        "mae": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 1.5, 2, 4, 6],
        },
        "rmse": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6],
        },
        "rmse_anom": {
            "vmin": 0,
            "vmax": 6,
            "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6],
        },
    },
}

IMPROVEMENT_PLOT_CONFIG = {
    "mslp": {
        "bias": {
            "vmin": -400,
            "vmax": 100,
            "ticks": [-400, -300, -200, -100, 0, 25, 50, 75, 100],
        },
        "rmse": {
            "vmin": -200,
            "vmax": 100,
            "ticks": [-200, -150, -100, -50, 0, 25, 50, 75, 100],
        },
        "rmse_anom": {
            "vmin": -100,
            "vmax": 100,
            "ticks": [-100, -75, -50, -25, 0, 25, 50, 75, 100],
        },
    },
    "t2m": {
        "bias": {
            "vmin": -400,
            "vmax": 100,
            "ticks": [-400, -300, -200, -100, 0, 25, 50, 75, 100],
        },
        "rmse": {
            "vmin": -100,
            "vmax": 100,
            "ticks": [-100, -75, -50, -25, 0, 25, 50, 75, 100],
        },
        "rmse_anom": {
            "vmin": -100,
            "vmax": 100,
            "ticks": [-100, -75, -50, -25, 0, 25, 50, 75, 100],
        },
    },
}


def main() -> None:
    experiments_root = Path("/Users/jacopodallaglio/ML/training/seasonal/experiments")

    plot_mode: PlotMode = "maps"

    plot_mlfc = False

    force_clim_recalc = True
    interpolate = True

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
        # "t2m",
        "mslp",
        # "d2m",
        # "u10",
        # "v10",
        # "sst",
        # "tprate",
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

    wanted_start_periods = ["05"]

    leadtime_units = "months"
    clim_period = "month" # "dayofyear", "day", "month", "year", "day_hour", "dayofyear_hour", "month_hour"
    clim_rolling_window = None

    time_range = None
    # time_range = ("2018-01-01", "2022-12-31")

    leadtime_agg_coord = "leadtime_seasonal"

    settings = get_experiment_configs(experiments_root, variables, regions)

    print(f"Found {len(settings)} matching experiment(s).")

    n = 0
    for s in settings:
        valid_time_range = (s.test_start, s.test_end) if time_range is None else time_range
        clim_time_range = (s.train_start, s.train_end)
        lat_lon = list(s.region.values()) if s.region is not None else [None, None]
        valid_lat_range = lat_lon[0] if lat_range is None else lat_range
        valid_lon_range = lat_lon[1] if lon_range is None else lon_range

        print(f"Generate {plot_mode} for {s.var_an, s.var_fc} in {s.region_name} (lon={valid_lon_range}, lat={valid_lat_range})")

        fc, an, mlfc = get_and_subset_datasets(
            s,
            leadtime_units=leadtime_units,
            lat_range=valid_lat_range,
            lon_range=valid_lon_range,
            time_range=valid_time_range,
            interpolate=interpolate,
        )
        mlfc = mlfc.assign_coords(leadtime=s.leadtimes) if mlfc is not None else None

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

        models = ("fc", "mlfc")
        ds_plot = (fc, mlfc) if plot_mlfc else (fc,)
        ds_clim_plot = (fc_clim, mlfc_clim) if plot_mlfc else (fc_clim,)

        for ds, ds_clim, model in zip(ds_plot, ds_clim_plot, models):
            if plot_mode in {"maps", "all"}:
                deterministic_metrics = [m for m in metrics if is_deterministic(m)]
                probabilistic_metrics = [m for m in metrics if is_probabilistic(m)]

                metric_maps_det = xr.Dataset()
                metric_maps_prob = xr.Dataset()
                if len(deterministic_metrics) != 0:
                    print(f"Get {model} deterministic metric maps")
                    metric_maps_det = get_metrics(
                        an=an,
                        fc=ds,
                        var=s.var_fc,
                        metric_kind="maps",
                        leadtime_agg=True,
                        realization_agg=True,
                        an_clim=an_clim,
                        fc_clim=ds_clim,
                        metrics=deterministic_metrics,
                        leadtime_windows=s.seasonal_leadtime_windows,
                        leadtime_agg_coord=leadtime_agg_coord,
                        clim_period=clim_period,
                        period_dim=f"start_{leadtime_units}",
                        periods_requested=wanted_start_periods,
                        align=False,
                    )
                if len(probabilistic_metrics) != 0:
                    print(f"Get {model} probabilistic metric maps")
                    metric_maps_prob = get_metrics(
                        an=an,
                        fc=ds,
                        var=s.var_fc,
                        metric_kind="maps",
                        leadtime_agg=True,
                        realization_agg=False,
                        an_clim=an_clim,
                        fc_clim=ds_clim,
                        metrics=probabilistic_metrics,
                        leadtime_windows=s.seasonal_leadtime_windows,
                        leadtime_agg_coord=leadtime_agg_coord,
                        clim_period=clim_period,
                        period_dim=f"start_{leadtime_units}",
                        periods_requested=wanted_start_periods,
                        align=False,
                    )

                metric_maps = xr.merge([metric_maps_det, metric_maps_prob])

                available_metrics = [
                    str(x) for x in metric_maps.data_vars
                    if str(x) in metrics and str(x) != "rank_histogram"
                ]

                start_periods = [
                    str(x) for x in metric_maps[f"start_{leadtime_units}"].values
                    if str(x) in wanted_start_periods
                ]

                print(f"Plotting {model} metrics {available_metrics} for periods {start_periods} for exp {s.output_name}")

                for m in available_metrics:

                    dataarrays_to_plot = {model: metric_maps[m]}

                    # if m in METRIC_IMPROVEMENT:
                    #     datasets_to_plot["mlfc_vs_fc"] = xr.Dataset(
                    #         {m: METRIC_IMPROVEMENT[m](fc_ds[m], mlfc_ds[m])}
                    #     )

                    for start_period in start_periods:
                        for lead_value in metric_maps[m][leadtime_agg_coord].values:
                            label = safe_label(lead_label(metric_maps[m], lead_value, leadtime_agg_coord))

                            out_file = (
                                s.plot_dir / "maps"
                                / safe_label(start_period)
                                / f"time_{safe_label(valid_time_range)}_lat_{safe_label(lat_range)}_lon_{safe_label(lon_range)}"
                                / m
                                / f"{s.var_fc}_{m}_{model}_lead_{label}.png"
                            )

                            print(f"Saving map {out_file}")

                            plot_map(
                                metric_maps[m],
                                var=s.var_fc,
                                metric=m,
                                model=model,
                                start_period=start_period,
                                lead_value=lead_value,
                                out_file=out_file,
                                time_range=valid_time_range,
                                leadtime_dim=leadtime_agg_coord,
                                period_dim=f"start_{leadtime_units}",
                                var_plot_config=VARIABLE_PLOT_CONFIG,
                                impro_plot_config=IMPROVEMENT_PLOT_CONFIG,
                                plot_type="contourf",
                                title_strftime="%Y",
                            )
                            n += 1

    # if plot_mode in {"profiles", "all"}:
    #     probabilistic_metrics = ["crps", "spread", "spread_skill_ratio", "ens_member_rmse", "ens_member_rmse_anom"]
    #     groups = defaultdict(list)
    #     for s in settings:
    #         groups[
    #             s.comparison_key(
    #                 # ignore={"root_dir", "region_name", "region"}
    #                 ignore={"root_dir", "region_name", "region", "trainer_precision"}
    #             )
    #         ].append(s)
        
    #     for group in groups.values():
    #         metric_scalar_ens_mean_fc_by_region: dict[str, xr.Dataset] = {}
    #         metric_scalar_ens_mean_mlfc_by_region: dict[str, xr.Dataset] = {}
    #         metric_scalar_members_fc_by_region: dict[str, xr.Dataset] = {}
    #         metric_scalar_members_mlfc_by_region: dict[str, xr.Dataset] = {}
    #         # metric_maps_ens_mean_fc_by_region: dict[str, xr.Dataset] = {}
    #         # metric_maps_ens_mean_mlfc_by_region: dict[str, xr.Dataset] = {}
    #         # metric_maps_members_fc_by_region: dict[str, xr.Dataset] = {}
    #         # metric_maps_members_mlfc_by_region: dict[str, xr.Dataset] = {}

    #         common_s: Settings = next(iter(group))
    #         time_range = (common_s.train_start, common_s.test_end)
    #         for s in group:
    #             print("Get metrics for:", s.var_fc, s.region_name)
    #             metric_scalar_ens_mean = calculate_metric_kind(
    #                 s,
    #                 metric_kind="scalar",
    #                 leadtime_agg="leadtime_month",
    #                 realization_agg="ensemble_mean",
    #                 lat_range=lat_range,
    #                 lon_range=lon_range,
    #                 time_range=time_range,
    #             )["leadtime_month"]["scalar"]

    #             metric_scalar_members = calculate_metric_kind(
    #                 s,
    #                 metric_kind="scalar",
    #                 leadtime_agg="leadtime_month",
    #                 realization_agg="member",
    #                 lat_range=lat_range,
    #                 lon_range=lon_range,
    #                 time_range=time_range,
    #             )["leadtime_month"]["scalar"]

    #             # metric_maps_ens_mean = calculate_metric_kind(
    #             #     s,
    #             #     metric_kind="maps",
    #             #     leadtime_agg="leadtime_month",
    #             #     realization_agg="ensemble_mean",
    #             #     lat_range=lat_range,
    #             #     lon_range=lon_range,
    #             #     time_range=time_range,
    #             # )["leadtime_month"]["maps"]

    #             # metric_maps_members = calculate_metric_kind(
    #             #     s,
    #             #     metric_kind="maps",
    #             #     leadtime_agg="leadtime_month",
    #             #     realization_agg="member",
    #             #     lat_range=lat_range,
    #             #     lon_range=lon_range,
    #             #     time_range=time_range,
    #             # )["leadtime_month"]["maps"]

    #             metric_scalar_ens_mean_fc_by_region[s.region_name] = metric_scalar_ens_mean["fc_ens_mean"]
    #             metric_scalar_ens_mean_mlfc_by_region[s.region_name] = metric_scalar_ens_mean["mlfc_ens_mean"]
    #             metric_scalar_members_fc_by_region[s.region_name] = metric_scalar_members["fc"]
    #             metric_scalar_members_mlfc_by_region[s.region_name] = metric_scalar_members["mlfc"]
    #             # metric_maps_ens_mean_fc_by_region[s.region_name] = metric_maps_ens_mean["fc_ens_mean"]
    #             # metric_maps_ens_mean_mlfc_by_region[s.region_name] = metric_maps_ens_mean["mlfc_ens_mean"]
    #             # metric_maps_members_fc_by_region[s.region_name] = metric_maps_members["fc"]
    #             # metric_maps_members_mlfc_by_region[s.region_name] = metric_maps_members["mlfc"]

    #         available_metrics = [str(x) for x in next(iter(metric_scalar_members_fc_by_region.values())).data_vars if str(x) in metrics and str(x) != "rank_histogram"]
    #         print("Available metrics for profiles", available_metrics)
    #         models = (
    #             ["fc"]
    #             # + ["global_fc", "time_avg_fc"]
    #             + [str(region_name) + "_mlfc" for region_name in regions]
    #             # + [str(region_name) + "_global_mlfc" for region_name in regions]
    #             # + [str(region_name) + "_time_avg_mlfc" for region_name in regions]
    #         )
    #         print("Models compared in profiles", models)
    #         # weights = np.cos(np.deg2rad(next(iter(metric_maps_members_fc_by_region.values()))[spatial_dims[0]]))

    #         section = "all_dims"
    #         kind = "profiles"
    #         start_period = "all"
    #         for m in available_metrics:
    #             if m in probabilistic_metrics:
    #                 das_ens_mean = (
    #                     [metric_scalar_members_fc_by_region[regions[0]][m]]
    #                     # + [metric_maps_members_fc_by_region[regions[0]][m].weighted(weights).mean(spatial_dims)]
    #                     + [ds[m] for ds in metric_scalar_members_mlfc_by_region.values()]
    #                     # + [ds[m].weighted(weights).mean(spatial_dims) for ds in metric_maps_members_mlfc_by_region.values()]
    #                 )
    #                 das_member = None
    #             else:
    #                 das_ens_mean = (
    #                     [metric_scalar_ens_mean_fc_by_region[regions[0]][m]]
    #                     # + [metric_maps_ens_mean_fc_by_region[regions[0]][m].weighted(weights).mean(spatial_dims)] 
    #                     + [ds[m] for ds in metric_scalar_ens_mean_mlfc_by_region.values()]
    #                     # + [ds[m].weighted(weights).mean(spatial_dims) for ds in metric_maps_ens_mean_mlfc_by_region.values()]
    #                 )
    #                 das_member = (
    #                     [metric_scalar_members_fc_by_region[regions[0]][m]]
    #                     # + [metric_maps_members_fc_by_region[regions[0]][m].weighted(weights).mean(spatial_dims)]
    #                     + [ds[m] for ds in metric_scalar_members_mlfc_by_region.values()]
    #                     # + [ds[m].weighted(weights).mean(spatial_dims) for ds in metric_maps_members_mlfc_by_region.values()]
    #                 )

    #             print(das_ens_mean)
    #             out_file = (
    #                 common_s.plot_dir / "profiles" / common_s.var_fc / "leadtime_month"
    #                 / safe_label(start_period)
    #                 / f"time_{safe_label(time_range)}_lat_{safe_label(lat_range)}_lon_{safe_label(lon_range)}"
    #                 / f"{common_s.var_fc}_{m}_{section}_{kind}_fc-mlfc_startmonth_{safe_label(start_period)}.png"
    #             )

    #             print(f"Saving profile {out_file}")

    #             plot_profile(
    #                 das=das_ens_mean,
    #                 var=common_s.var_fc,
    #                 metric=m,
    #                 start_period=start_period,
    #                 models=models,
    #                 out_file=out_file,
    #                 time_range=time_range,
    #                 das_member=das_member,
    #                 leadtime_dim=leadtime_dim,
    #                 realization_dim=realization_dim,
    #                 spread="std",
    #             )
    #             n += 1

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
    #             start_period="all",
    #             models=["fc", "mlfc"],
    #             out_file=out_file,
    #             time_range=time_range,
    #         )
    #         n += 1
        
    # if plot_mode in {"timeseries", "all"}:
    #     groups = defaultdict(list)
    #     for s in settings:
    #         groups[
    #             s.comparison_key(
    #                 ignore={"root_dir", "region_name", "region"}
    #             )
    #         ].append(s)
        
    #     for group in groups.values():
    #         metric_ts_ens_mean_fc_by_region: dict[str, xr.Dataset] = {}
    #         metric_ts_ens_mean_mlfc_by_region: dict[str, xr.Dataset] = {}
    #         metric_ts_members_fc_by_region: dict[str, xr.Dataset] = {}
    #         metric_ts_members_mlfc_by_region: dict[str, xr.Dataset] = {}

    #         common_s: Settings = next(iter(group))
    #         time_range = (common_s.train_start, common_s.test_end)
    #         for s in group:
    #             print("Get metrics for:", s.var_fc, s.region_name)
    #             metric_ts_ens_mean = calculate_metric_kind(
    #                 s,
    #                 metric_kind="timeseries",
    #                 leadtime_agg="leadtime_month",
    #                 realization_agg="ensemble_mean",
    #                 lat_range=lat_range,
    #                 lon_range=lon_range,
    #                 time_range=time_range,
    #             )["leadtime_month"]["timeseries"]

    #             metric_ts_members = calculate_metric_kind(
    #                 s,
    #                 metric_kind="timeseries",
    #                 leadtime_agg="leadtime_month",
    #                 realization_agg="member",
    #                 lat_range=lat_range,
    #                 lon_range=lon_range,
    #                 time_range=time_range,
    #             )["leadtime_month"]["timeseries"]

    #             metric_ts_ens_mean_fc_by_region[s.region_name] = metric_ts_ens_mean["fc_ens_mean"]
    #             metric_ts_ens_mean_mlfc_by_region[s.region_name] = metric_ts_ens_mean["mlfc_ens_mean"]
    #             metric_ts_members_fc_by_region[s.region_name] = metric_ts_members["fc"]
    #             metric_ts_members_mlfc_by_region[s.region_name] = metric_ts_members["mlfc"]

    #         available_metrics = [str(x) for x in next(iter(metric_ts_ens_mean_fc_by_region.values())).data_vars if str(x) in metrics]
    #         print("Available metrics for timeseries", available_metrics)
    #         models = ["fc"] + [str(region_name) + "_mlfc" for region_name in regions]
    #         print("Models compared in timeseries", models)

    #         section = "all_dims"
    #         kind = "timeseries"
    #         start_period = "all"
    #         for m in available_metrics:
    #             das_ens_mean = [metric_ts_ens_mean_fc_by_region[regions[0]][m]] + [ds[m] for ds in metric_ts_ens_mean_mlfc_by_region.values()]
    #             das_member = [metric_ts_members_fc_by_region[regions[0]][m]] + [ds[m] for ds in metric_ts_members_mlfc_by_region.values()]

    #             for lead_value in das_ens_mean[0][leadtime_dim].values:
    #                 label = safe_label(lead_label(das_ens_mean[0], lead_value, leadtime_dim))
    #                 out_file = (
    #                     common_s.plot_dir / "timeseries"/ common_s.var_fc / "leadtime_month"
    #                     / "all_months"
    #                     / f"time_{safe_label(time_range)}_lat_{safe_label(lat_range)}_lon_{safe_label(lon_range)}"
    #                     / f"{common_s.var_fc}_{m}_{section}_{kind}_lead_{label}.png"
    #                 )

    #                 print(f"Saving timeseries {out_file}")

    #                 plot_timeseries(
    #                     das=das_ens_mean,
    #                     var=common_s.var_fc,
    #                     metric=m,
    #                     start_period="all",
    #                     models=models,
    #                     out_file=out_file,
    #                     lead_value=lead_value,
    #                     time_range=time_range,
    #                     das_member=das_member,
    #                     leadtime_dim=leadtime_dim,
    #                     realization_dim=realization_dim,
    #                     spread="std",
    #                 )
    #                 n += 1

    print(f"Done. Saved {n} plots.")


if __name__ == "__main__":
    main()
