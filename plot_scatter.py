from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import Literal, Sequence

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

from dask.diagnostics.progress import ProgressBar

from earthml import (
    Settings,
    get_experiment_configs,
    get_and_subset_datasets,
    calculate_save_and_subset_climatologies,
)
from earthml.metrics import get_metrics
from earthml.plots import (
    plot_metric_diff_scatter,
    ScatterPoint,
    metric_improvement,
    get_total_months,
    safe_label,
    METRIC_NAMES,
    VARIABLE_NAMES,
    plot_metric_diff_scatter,
)


def iter_scalar_points(
    settings: Sequence[Settings],
    *,
    forecast_metric: str,
    diff_metric: str,
    leadtime_agg: bool,
    realization_agg: bool,
    lat_range,
    lon_range,
    time_range,
    leadtime_agg_coord: str = "leadtime",
    force_clim_recalc: bool = False,
    clim_period: str = "month",
    period_dim: str = "start_month",
    wanted_start_periods: Sequence[str] | None = None,
) -> list[ScatterPoint]:
    points: list[ScatterPoint] = []

    for s in settings:
        valid_time_range = (s.train_start, s.test_end) if time_range is None else time_range
        fc, an, mlfc = get_and_subset_datasets(
            s,
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=valid_time_range,
        )

        if mlfc is None:
            raise ValueError("ML-corrected forecast must be present to produce scatter plot.")

        fc_clim, an_clim, mlfc_clim = calculate_save_and_subset_climatologies(
            s,
            force=force_clim_recalc,
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=(s.train_start, s.test_end),
        )

        print(f"Calculating scalar metrics for {s.output_name}")

        metric_scalar_fc = get_metrics(
            an=an,
            fc=fc,
            var=s.var_fc,
            metric_kind="scalar",
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
            metric_kind="scalar",
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

        y_da = metric_scalar_fc[forecast_metric]
        x_da = metric_improvement(metric_scalar_fc[diff_metric], metric_scalar_mlfc[diff_metric], diff_metric)

        common_dims = tuple(dim for dim in y_da.dims if dim in x_da.dims)
        y_da, x_da = xr.align(y_da, x_da, join="inner")

        if period_dim in common_dims and wanted_start_periods is not None:
            months = [m for m in y_da[period_dim].values if str(m) in wanted_start_periods]
            if not months:
                continue
            y_da = y_da.sel(init_month=months)
            x_da = x_da.sel(init_month=months)

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
    experiments_root = Path("./experiments")
    plot_dir = Path("./plots")

    variables = [
        "mslp",
        "t2m",
        "d2m",
        "u10",
        "v10",
        "sst",
    ]

    regions = ["ConUS"]

    lat_range = None
    lon_range = None

    wanted_start_months = ["all"]

    forecast_metric = "acc"
    diff_metric = "nrmse_anom"

    settings = get_experiment_configs(experiments_root, variables, regions)

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

    for group in groups.values():
        common_s: Settings = next(iter(group))
        time_range = (common_s.train_start, common_s.test_end)

        points = iter_scalar_points(
            group,
            forecast_metric=forecast_metric,
            diff_metric=diff_metric,
            leadtime_agg=False,
            realization_agg=True,
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=(common_s.train_start, common_s.test_end),
            wanted_start_periods=wanted_start_months,
        )

        var_label = "multi_variable"

        out_file = (
            plot_dir
            / "diff_scatter"
            / var_label
            / f"time_{safe_label(time_range)}_lat_{safe_label(lat_range)}_lon_{safe_label(lon_range)}"
            / f"{forecast_metric}_vs_{diff_metric}_improvement_multi_variable.png"
        )

        print(f"Saving diff scatter {out_file}")

        plot_metric_diff_scatter(
            points,
            forecast_metric=forecast_metric,
            diff_metric=diff_metric,
            out_file=out_file,
            color_by="variable",
            marker_by="leadtime",
            shade_by="total_months",
            title=(
                f"{VARIABLE_NAMES.get(common_s.var_fc, common_s.var_fc)} · "
                f"{METRIC_NAMES.get(forecast_metric, forecast_metric)} vs "
                f"{METRIC_NAMES.get(diff_metric, diff_metric)} improvement"
            ),
            xlabel=f"{METRIC_NAMES.get(diff_metric, diff_metric)} improvement of MLFC vs FC",
            ylabel=f"FC {METRIC_NAMES.get(forecast_metric, forecast_metric)}",
            fit_lines=False,
        )

        n += 1

    print(f"Done. Saved {n} diff scatter plot(s).")


if __name__ == "__main__":
    main()
