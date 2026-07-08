from pathlib import Path
from collections import defaultdict
from typing import Literal, Sequence

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")

from earthml import (
    LeadtimeUnit,
    Settings,
    get_experiment_configs,
)
from earthml.metrics import (
    get_scalar_metrics,
    LeadtimeAgg,
    MetricAgg,
    ClimPeriod,
)
from earthml.plots import (
    plot_metric_diff_scatter,
    ScatterPoint,
    metric_improvement,
    get_total_months,
    safe_label,
    METRIC_NAMES,
    VARIABLE_NAMES,
)


def iter_scalar_points(
    settings: Sequence[Settings],
    *,
    forecast_metric: str,
    diff_metric: str,
    metric_agg_mode: MetricAgg,
    leadtime_agg: LeadtimeAgg,
    realization_agg: bool,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
    time_range: tuple[str, str] | None = None,
    clim_period: ClimPeriod = "month",
    clim_rolling_window: int | None = None,
    clim_time_range: tuple[str, str] | None = None,
    leadtime_units: LeadtimeUnit = LeadtimeUnit.MONTHS,
    leadtime_agg_coord: str = "leadtime",
    force_clim_recalc: bool = False,
    period_dim: str = "start_months",
    wanted_start_periods: Sequence[str] | None = None,
    interpolate: bool = False,
    build_analysis: bool = True,
) -> list[ScatterPoint]:
    points: list[ScatterPoint] = []

    for s in settings:
        metrics_fc_da, metrics_mlfc_da = get_scalar_metrics(
            s=s,
            fc_metrics=[forecast_metric, diff_metric],
            mlfc_metrics=diff_metric,
            metric_agg_mode=metric_agg_mode,
            leadtime_agg=leadtime_agg,
            realization_agg=realization_agg,
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=time_range,
            clim_period=clim_period,
            clim_rolling_window=clim_rolling_window,
            clim_time_range=clim_time_range,
            leadtime_units=leadtime_units,
            force_clim_recalc=force_clim_recalc,
            period_dim=period_dim,
            wanted_start_periods=wanted_start_periods,
            interpolate=interpolate,
            build_analysis=build_analysis,
        )

        if metric_agg_mode == "global":
            y_da = metrics_fc_da[forecast_metric]
            x_da = metric_improvement(metrics_fc_da[diff_metric], metrics_mlfc_da[diff_metric], diff_metric)

        elif metric_agg_mode in ("spatial_avg", "spatial_rmse"):
            lat_dim = metrics_fc_da.earthml.guessed_dims.latitude
            lon_dim = metrics_fc_da.earthml.guessed_dims.longitude
            weights = np.cos(np.deg2rad(metrics_fc_da[lat_dim]))
            if metric_agg_mode == "spatial_rmse":
                y_da = np.sqrt(metrics_fc_da[forecast_metric])
                x_da = metric_improvement(
                    np.sqrt(metrics_fc_da[diff_metric].weighted(weights).mean(dim=(lat_dim, lon_dim))),
                    np.sqrt(metrics_mlfc_da[diff_metric].weighted(weights).mean(dim=(lat_dim, lon_dim))),
                    diff_metric,
                )
            else:
                y_da = metrics_fc_da[forecast_metric]
                x_da = metric_improvement(
                    metrics_fc_da[diff_metric].weighted(weights).mean(dim=(lat_dim, lon_dim)),
                    metrics_mlfc_da[diff_metric].weighted(weights).mean(dim=(lat_dim, lon_dim)),
                    diff_metric,
                )

        else:
            raise ValueError(f"metric_agg_mode={metric_agg_mode} not suppoerted.")

        common_dims = tuple(dim for dim in y_da.dims if dim in x_da.dims)

        stacked = xr.Dataset({"x": x_da, "y": y_da}).stack(point=common_dims)

        for i in range(stacked.sizes["point"]):
            x = float(stacked["x"].isel(point=i).values)
            y = float(stacked["y"].isel(point=i).values)

            if not np.isfinite(x) or not np.isfinite(y):
                continue

            coords = {
                dim: stacked.indexes["point"][i][j]
                for j, dim in enumerate(common_dims)
            }

            points.append(
                ScatterPoint(
                    x=x,
                    y=y,
                    variable=s.var_fc,
                    region=s.region_name,
                    leadtime=coords.get(leadtime_agg_coord, "all"),
                    start_month=coords.get(period_dim, "all"),
                    total_months=get_total_months(s),
                    experiment=s.output_name,
                )
            )

    return points


def main() -> None:
    experiments_root = Path("/Users/jacopodallaglio/ML/training/seasonal/experiments")

    force_clim_recalc = False
    interpolate = True
    build_analysis = True

    plot_dir = Path("/Users/jacopodallaglio/ML/training/seasonal/plots/scatter")

    variables = [
        "mslp",
        "t2m",
        "d2m",
        "u10",
        "v10",
        "sst",
    ]

    regions = ["World"]

    settings = get_experiment_configs(experiments_root, variables, regions)

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

    time_range = None
    # time_range = ("2018-01-01", "2022-12-31")

    wanted_start_months = ["all"]

    leadtime_agg_mode = "single" # "single", "aggregated", "seasonal_window"

    leadtime_units = LeadtimeUnit.MONTHS
    clim_period = "month" # "dayofyear", "day", "month", "year", "day_hour", "dayofyear_hour", "month_hour"
    clim_rolling_window = None

    metric_agg_mode = "global" # "spatial_avg", "global", "spatial_rmse"

    forecast_metric = "r2_anom"
    diff_metric = "nrmse_anom"

    groups = defaultdict(list)
    for s in settings:
        groups[
            s.comparison_key(
                ignore={
                    "root_dir",
                    "region_name",
                    "region",
                    "trainer_precision",
                    "var_fc",
                    "var_an",
                }
            )
        ].append(s)

    print(f"Found {len(groups)} groups")
    n = 0

    leadtime_agg_coord = "leadtime" if leadtime_agg_mode=="single" else "leadtime_seasonal"

    all_points: list[ScatterPoint] = []

    for group in groups.values():
        common_s: Settings = next(iter(group))
        valid_time_range = (common_s.train_start, common_s.test_end) if time_range is None else time_range
        clim_time_range = (common_s.train_start, common_s.test_end)

        points = iter_scalar_points(
            group,
            forecast_metric=forecast_metric,
            diff_metric=diff_metric,
            metric_agg_mode=metric_agg_mode,
            leadtime_agg=leadtime_agg_mode,
            realization_agg=True,
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=valid_time_range,
            clim_time_range=clim_time_range,
            leadtime_units=leadtime_units,
            leadtime_agg_coord=leadtime_agg_coord,
            force_clim_recalc=force_clim_recalc,
            clim_period=clim_period,
            clim_rolling_window=clim_rolling_window,
            period_dim=f"start_{leadtime_units}",
            wanted_start_periods=wanted_start_months,
            interpolate=interpolate,
            build_analysis=build_analysis,
        )

        all_points.extend(points)

    out_file = (
        plot_dir
        / f"time_{safe_label(time_range)}_lat_{safe_label(lat_range)}_lon_{safe_label(lon_range)}"
        / f"{forecast_metric}_vs_{diff_metric}_improvement_all_variables_{metric_agg_mode}.png"
    )

    print(f"Saving combined diff scatter {out_file}")

    plot_metric_diff_scatter(
        all_points,
        forecast_metric=forecast_metric,
        diff_metric=diff_metric,
        out_file=out_file,
        color_by="variable",
        marker_by="leadtime",
        shade_by="total_months",
        title=(
            f"All variables {metric_agg_mode} metrics · "
            f"{METRIC_NAMES.get(forecast_metric, forecast_metric)} vs "
            f"{METRIC_NAMES.get(diff_metric, diff_metric)} improvement"
        ),
        xlabel=f"{METRIC_NAMES.get(diff_metric, diff_metric)} improvement of MLFC vs FC",
        ylabel=f"FC {METRIC_NAMES.get(forecast_metric, forecast_metric)}",
        fit_lines=False,
    )

    print(f"Done. Saved 1 combined diff scatter plot with {len(all_points)} points.")


if __name__ == "__main__":
    main()
