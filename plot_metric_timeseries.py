from typing import Literal
from pathlib import Path

import numpy as np
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
    MetricKind,
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

    regenerate_plots = False

    plot_title = True
    plot_labels = True
    plot_legend = True

    title_size = None
    label_size = None
    tick_size = None
    dpi = 300

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
    # Rolling mean
    # ==========================================================

    # Number of metric samples, not number of days.
    rolling_mean_window = None
    # rolling_mean_window = 30
    rolling_mean_center = True
    rolling_mean_min_periods = 1

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

    metrics_to_compute = list(metrics)

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
        "ConUS",
        # "Europe",
        # "Pacific",
        # "World",
        # None,
    ]

    # ==========================================================
    # Spatial subset
    # ==========================================================

    # ConUS
    lat_range = (50, 25)
    lon_range = (-130, -60)

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
    # Timeseries spatial subregions
    # ==========================================================

    # locations = SEA_LOCATIONS
    # locations = CITY_LOCATIONS
    locations = {"all": None}

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
        # train_subsamples=1000,
        train_subsamples=None,
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

        for location, location_config in locations.items():
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
                    *plot_models,
                    *(
                        model
                        for comparison in model_comparisons
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
                        metric_kind=metric_kind,
                        leadtime_agg=leadtime_agg_mode,
                        realization_agg=True,
                        an_clim=an_clim_loc,
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

                if probabilistic_metrics:
                    print(
                        f"Get {model} probabilistic metric timeseries "
                        f"for {location}"
                    )

                    metric_ts_prob = get_metrics(
                        an=an_loc,
                        fc=ds,
                        var=s.var_fc,
                        metric_kind=metric_kind,
                        leadtime_agg=leadtime_agg_mode,
                        realization_agg=False,
                        an_clim=an_clim_loc,
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

                metric_ts_by_model[model] = xr.merge(
                    [metric_ts_det, metric_ts_prob]
                )

            # ==================================================
            # Plot FC and MLFC together
            # ==================================================

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
                f"for models {available_models} in {location}"
            )

            for m in available_metrics:
                das = [
                    apply_rolling_mean(
                        metric_ts_by_model[model][m],
                        window=rolling_mean_window,
                        center=rolling_mean_center,
                        min_periods=rolling_mean_min_periods,
                    ).compute()
                    for model in available_models
                ]

                for lead_value in das[0][leadtime_agg_coord].values:
                    for model, da in zip(available_models, das):
                        print_timeseries_extrema(
                            da,
                            model=model,
                            metric=m,
                            lead_value=lead_value,
                            leadtime_dim=leadtime_agg_coord,
                        )

                    label = safe_label(
                        lead_label(
                            das[0],
                            lead_value,
                            leadtime_agg_coord,
                        )
                    )

                    rolling_label = (
                        f" roll {rolling_mean_window}"
                        if rolling_mean_window is not None
                        else ""
                    )

                    rolling_filename = (
                        f"_roll{rolling_mean_window}"
                        if rolling_mean_window is not None
                        else ""
                    )

                    common_path = (
                        Path("timeseries")
                        / safe_label(period_dim)
                        / safe_label("all")
                        / (
                            f"time_{safe_label(valid_time_range)}"
                            f"_loc_{safe_label(location)}"
                        )
                        / m
                        / leadtime_agg_mode
                    )

                    filename = (
                        f"{s.var_fc}_{m}_"
                        f"{'-'.join(available_models)}"
                        f"_lead_{label}{rolling_filename}.png"
                    )

                    out_file = s.plot_dir / common_path / filename
                    if out_file.exists() and not regenerate_plots:
                        continue

                    print(f"Saving timeseries {out_file}")

                    plot_timeseries(
                        das,
                        var=s.var_fc,
                        metric=m,
                        models=available_models,
                        lead_value=lead_value,
                        out_file=out_file,
                        time_range=valid_time_range,
                        leadtime_dim=leadtime_agg_coord,
                        train_end=(
                            s.train_end
                            if valid_time_range[0] <= s.train_end <= valid_time_range[1]
                            else None
                        ),
                        val_end=(
                            s.val_end
                            if valid_time_range[0] <= s.val_end <= valid_time_range[1]
                            else None
                        ),
                        plot_title=plot_title,
                        plot_labels=plot_labels,
                        title_suffix=rolling_label,
                        plot_legend=plot_legend,
                        title_size=title_size,
                        label_size=label_size,
                        tick_size=tick_size,
                        dpi=dpi,
                    )
                    n += 1

            # ==================================================
            # Plot metric improvements
            # ==================================================

            for baseline_model, target_model in model_comparisons:
                if baseline_model not in metric_ts_by_model:
                    continue
                if target_model not in metric_ts_by_model:
                    continue

                baseline_ds = metric_ts_by_model[baseline_model]
                target_ds = metric_ts_by_model[target_model]

                comparison_metrics = [
                    m
                    for m in metrics
                    if m in baseline_ds and m in target_ds
                ]

                for m in comparison_metrics:
                    improvements = build_metric_improvements(
                        baseline_ds,
                        target_ds,
                        metric=m,
                        baseline_model=baseline_model,
                        target_model=target_model,
                        improvement_units=("%", "Δ"),
                    )

                    for comparison_model, comparison_da in improvements.items():
                        improvement_unit = get_improvement_unit(
                            comparison_model
                        )
                        comparison_da = apply_rolling_mean(
                            comparison_da,
                            window=rolling_mean_window,
                            center=rolling_mean_center,
                            min_periods=rolling_mean_min_periods,
                        ).compute()

                        for lead_value in comparison_da[leadtime_agg_coord].values:
                            print_timeseries_extrema(
                                comparison_da,
                                model=comparison_model,
                                metric=m,
                                lead_value=lead_value,
                                leadtime_dim=leadtime_agg_coord,
                            )

                            label = safe_label(
                                lead_label(
                                    comparison_da,
                                    lead_value,
                                    leadtime_agg_coord,
                                )
                            )

                            rolling_label = (
                                f" roll {rolling_mean_window}"
                                if rolling_mean_window is not None
                                else ""
                            )

                            rolling_filename = (
                                f"_roll{rolling_mean_window}"
                                if rolling_mean_window is not None
                                else ""
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
                                / leadtime_agg_mode
                            )

                            filename = (
                                f"{s.var_fc}_{m}_"
                                f"{safe_label(comparison_model)}_"
                                f"lead_{label}{rolling_filename}.png"
                            )

                            out_file = s.plot_dir / common_path / filename

                            if out_file.exists() and not regenerate_plots:
                                continue

                            print(
                                f"Saving improvement timeseries {out_file}"
                            )

                            plot_timeseries(
                                [comparison_da],
                                var=s.var_fc,
                                metric=m,
                                models=[comparison_model],
                                lead_value=lead_value,
                                out_file=out_file,
                                time_range=valid_time_range,
                                leadtime_dim=leadtime_agg_coord,
                                train_end=(
                                    s.train_end
                                    if valid_time_range[0] <= s.train_end <= valid_time_range[1]
                                    else None
                                ),
                                val_end=(
                                    s.val_end
                                    if valid_time_range[0] <= s.val_end <= valid_time_range[1]
                                    else None
                                ),
                                plot_title=plot_title,
                                plot_labels=plot_labels,
                                title_suffix=rolling_label,
                                plot_legend=plot_legend,
                                improvement_unit=improvement_unit,
                                timeseries_zero_line=True,
                                title_size=title_size,
                                label_size=label_size,
                                tick_size=tick_size,
                                dpi=dpi,
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


def print_timeseries_extrema(
    da: xr.DataArray,
    *,
    model: str,
    metric: str,
    lead_value,
    leadtime_dim: str,
) -> None:
    """Print min/max values and their time locations."""
    time_dim = da.earthml.guessed_dims.time

    if time_dim is None:
        raise ValueError("Could not determine time dimension.")

    ts = da.sel({leadtime_dim: lead_value}).squeeze(drop=True)

    valid = ts.where(np.isfinite(ts), drop=True)

    if valid.size == 0:
        print(
            f"  {model:>16s} | {metric} | lead={lead_value}: "
            "no finite values"
        )
        return

    min_idx = int(valid.argmin(dim=time_dim))
    max_idx = int(valid.argmax(dim=time_dim))

    min_value = float(valid.isel({time_dim: min_idx}))
    max_value = float(valid.isel({time_dim: max_idx}))

    min_time = valid[time_dim].isel({time_dim: min_idx}).values
    max_time = valid[time_dim].isel({time_dim: max_idx}).values

    print(
        f"  {model:>16s} | {metric} | lead={lead_value}\n"
        f"    min = {min_value:.6g} @ "
        f"{np.datetime_as_string(min_time, unit='h')}\n"
        f"    max = {max_value:.6g} @ "
        f"{np.datetime_as_string(max_time, unit='h')}"
    )


if __name__ == "__main__":
    main()
