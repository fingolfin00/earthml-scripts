from pathlib import Path
from collections import defaultdict
from typing import Literal, Sequence

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")

from dask.diagnostics.progress import ProgressBar

from earthml import (
    Settings,
    get_experiment_configs,
    get_and_subset_datasets,
    calculate_save_and_subset_climatologies,
)
from earthml.metrics import get_metrics, LeadtimeAgg
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
    metric_agg_mode: Literal["spatial_avg", "global"],
    leadtime_agg: LeadtimeAgg,
    realization_agg: bool,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
    time_range: tuple[str, str] | None = None,
    clim_period: str = "month",
    clim_rolling_window: int | None = None,
    clim_time_range: tuple[str, str] | None = None,
    leadtime_units: str = "months",
    leadtime_agg_coord: str = "leadtime",
    force_clim_recalc: bool = False,
    period_dim: str = "start_months",
    wanted_start_periods: Sequence[str] | None = None,
    interpolate: bool = False,
    build_analysis: bool = True,
) -> list[ScatterPoint]:
    points: list[ScatterPoint] = []

    for s in settings:
        valid_time_range = (s.train_start, s.test_end) if time_range is None else time_range
        fc, an, mlfc = get_and_subset_datasets(
            s,
            leadtime_units=leadtime_units,
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=time_range,
            interpolate=interpolate,
        )

        if mlfc is None:
            raise ValueError("ML-corrected forecast must be present to produce scatter plot.")

        fc_clim, an_clim, mlfc_clim = calculate_save_and_subset_climatologies(
            s,
            leadtime_units=leadtime_units,
            force=force_clim_recalc,
            clim_period=clim_period,
            rolling_window=clim_rolling_window,
            rolling_center=True,
            rolling_min_periods=1,
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=clim_time_range,
            time_start=None,
            interpolate=interpolate,
            engine="zarr",
            build_analysis=build_analysis,
            coord_rename_fc=None,
            coord_rename_an=None,
        )

        print(f"Calculating {metric_agg_mode} scalar metrics [fc={forecast_metric}, diff={diff_metric}] for {s.output_name}")

        if metric_agg_mode == "global":
            metric_kind = "scalar"
        elif metric_agg_mode == "spatial_avg":
            metric_kind = "maps"
        else:
            raise ValueError(f"metric_agg_mode={metric_agg_mode} not available. Choose between: 'spatial_avg', 'global'")

        metric_scalar_fc = get_metrics(
            an=an,
            fc=fc,
            var=s.var_fc,
            metric_kind=metric_kind,
            leadtime_agg=leadtime_agg,
            realization_agg=realization_agg,
            an_clim=an_clim,
            fc_clim=fc_clim,
            metrics=[forecast_metric, diff_metric],
            leadtime_windows=s.seasonal_leadtime_windows,
            leadtime_agg_coord=leadtime_agg_coord,
            clim_period=clim_period,
            period_dim=period_dim,
            periods_requested=wanted_start_periods,
        )

        metric_scalar_mlfc = get_metrics(
            an=an,
            fc=mlfc,
            var=s.var_fc,
            metric_kind=metric_kind,
            leadtime_agg=leadtime_agg,
            realization_agg=realization_agg,
            an_clim=an_clim,
            fc_clim=mlfc_clim,
            metrics=diff_metric,
            leadtime_windows=s.seasonal_leadtime_windows,
            leadtime_agg_coord=leadtime_agg_coord,
            clim_period=clim_period,
            period_dim=period_dim,
            periods_requested=wanted_start_periods,
        )

        if forecast_metric not in metric_scalar_fc.data_vars:
            print(f"Skipping {s.output_name}: missing forecast metric {forecast_metric!r}")
            continue

        if diff_metric not in metric_scalar_fc.data_vars or diff_metric not in metric_scalar_mlfc.data_vars:
            print(f"Skipping {s.output_name}: missing diff metric {diff_metric!r}")
            continue

        if metric_agg_mode == "global":
            y_da = metric_scalar_fc[forecast_metric]
            x_da = metric_improvement(metric_scalar_fc[diff_metric], metric_scalar_mlfc[diff_metric], diff_metric)
        else:
            lat_dim = fc.earthml.guessed_dims.latitude
            lon_dim = fc.earthml.guessed_dims.longitude
            weights = np.cos(np.deg2rad(fc[lat_dim]))

            y_da = metric_scalar_fc[forecast_metric].weighted(weights).mean(dim=(lat_dim, lon_dim))

            x_da = metric_improvement(
                metric_scalar_fc[diff_metric].weighted(weights).mean(dim=(lat_dim, lon_dim)),
                metric_scalar_mlfc[diff_metric].weighted(weights).mean(dim=(lat_dim, lon_dim)),
                diff_metric,
            )

        common_dims = tuple(dim for dim in y_da.dims if dim in x_da.dims)
        y_da, x_da = xr.align(y_da, x_da, join="inner")

        if period_dim in common_dims and wanted_start_periods is not None:
            periods = [m for m in y_da[period_dim].values if str(m) in wanted_start_periods]
            if not periods:
                continue
            y_da = y_da.sel({period_dim: periods})
            x_da = x_da.sel({period_dim: periods})

        with ProgressBar():
            y_da = y_da.compute()
            x_da = x_da.compute()

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

    leadtime_units = "months"
    clim_period = "month" # "dayofyear", "day", "month", "year", "day_hour", "dayofyear_hour", "month_hour"
    clim_rolling_window = None

    metric_agg_mode = "spatial_avg" # "spatial_avg", "global"

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
