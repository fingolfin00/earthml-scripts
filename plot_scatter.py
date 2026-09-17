from pathlib import Path
from collections import defaultdict
from typing import Literal, Sequence

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")

from earthml import (
    LeadtimeUnit,
    ClimPeriod,
    Settings,
    get_experiment_configs,
)
from earthml.metrics import (
    get_scalar_metrics,
    build_metric_improvement,
    LeadtimeAgg,
    MetricAgg,
    ImprovementUnit,
)
from earthml.plots import (
    plot_metric_diff_scatter,
    ScatterPoint,
    get_total_months,
    safe_label,
    METRIC_NAMES,
    VARIABLE_NAMES,
)


def spatial_mean(
    da: xr.DataArray,
    *,
    lat_dim: str,
    lon_dim: str,
) -> xr.DataArray:
    """
    Area-weighted spatial mean.
    """
    weights = np.cos(
        np.deg2rad(da[lat_dim])
    )

    return da.weighted(weights).mean(
        dim=(lat_dim, lon_dim),
        skipna=True,
    )


def spatial_rms(
    da: xr.DataArray,
    *,
    lat_dim: str,
    lon_dim: str,
) -> xr.DataArray:
    """
    Area-weighted spatial root-mean-square.
    """
    weights = np.cos(
        np.deg2rad(da[lat_dim])
    )

    return np.sqrt(
        (da**2)
        .weighted(weights)
        .mean(
            dim=(lat_dim, lon_dim),
            skipna=True,
        )
    )


def aggregate_metric(
    da: xr.DataArray,
    *,
    metric_agg_mode: MetricAgg,
    lat_dim: str | None,
    lon_dim: str | None,
) -> xr.DataArray:
    """
    Apply the requested spatial aggregation to a metric field.
    """
    if metric_agg_mode == "global":
        return da

    if lat_dim is None or lon_dim is None:
        raise ValueError(
            f"Spatial aggregation {metric_agg_mode!r} "
            "requires latitude and longitude dimensions."
        )

    if metric_agg_mode == "spatial_avg":
        return spatial_mean(
            da,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
        )

    if metric_agg_mode == "spatial_rmse":
        return spatial_rms(
            da,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
        )

    raise ValueError(
        f"Unsupported metric_agg_mode="
        f"{metric_agg_mode!r}."
    )


def iter_scalar_points(
    settings: Sequence[Settings],
    *,
    forecast_metric: str,
    diff_metric: str,
    improvement_unit: ImprovementUnit,
    metric_agg_mode: MetricAgg,
    leadtime_agg: LeadtimeAgg,
    realization_agg: bool,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
    time_range: tuple[str, str] | None = None,
    clim_period: ClimPeriod = ClimPeriod.MONTH,
    clim_rolling_window: int | None = None,
    clim_time_range: tuple[str, str] | None = None,
    leadtime_units: LeadtimeUnit = LeadtimeUnit.MONTHS,
    leadtime_agg_coord: str = "leadtime",
    recalculate_climatology: bool = False,
    period_dim: str = "start_month",
    wanted_start_periods: Sequence[str] | None = None,
    interpolate: bool = False,
    build_analysis: bool = True,
) -> list[ScatterPoint]:

    points: list[ScatterPoint] = []

    for s in settings:

        # ======================================================
        # Metrics
        # ======================================================

        metrics_fc_ds, metrics_mlfc_ds = (
            get_scalar_metrics(
                s=s,
                fc_metrics=[
                    forecast_metric,
                    diff_metric,
                ],
                mlfc_metrics=[
                    diff_metric,
                ],
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
                force_clim_recalc=(
                    recalculate_climatology
                ),
                period_dim=period_dim,
                wanted_start_periods=(
                    wanted_start_periods
                ),
                interpolate=interpolate,
                build_analysis=build_analysis,
            )
        )


        # ======================================================
        # Spatial dimensions
        # ======================================================

        lat_dim = (
            metrics_fc_ds
            .earthml
            .guessed_dims
            .latitude
        )

        lon_dim = (
            metrics_fc_ds
            .earthml
            .guessed_dims
            .longitude
        )


        # ======================================================
        # Y axis:
        # forecast metric
        # ======================================================

        y_da = aggregate_metric(
            metrics_fc_ds[forecast_metric],
            metric_agg_mode=metric_agg_mode,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
        )


        # ======================================================
        # X axis:
        # target vs baseline improvement
        # ======================================================

        baseline_da = aggregate_metric(
            metrics_fc_ds[diff_metric],
            metric_agg_mode=metric_agg_mode,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
        )

        target_da = aggregate_metric(
            metrics_mlfc_ds[diff_metric],
            metric_agg_mode=metric_agg_mode,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
        )

        baseline_da, target_da = xr.align(
            baseline_da,
            target_da,
            join="exact",
        )

        x_da = build_metric_improvement(
            baseline_da,
            target_da,
            metric=diff_metric,
            improvement_unit=improvement_unit,
        )


        # ======================================================
        # Align X and Y
        # ======================================================

        x_da, y_da = xr.align(
            x_da,
            y_da,
            join="inner",
        )

        common_dims = tuple(
            dim
            for dim in y_da.dims
            if dim in x_da.dims
        )


        # ======================================================
        # Convert to scatter points
        # ======================================================

        point_ds = xr.Dataset(
            {
                "x": x_da,
                "y": y_da,
            }
        )

        if common_dims:
            stacked = point_ds.stack(
                point=common_dims
            )

        else:
            stacked = point_ds.expand_dims(
                point=[0]
            )


        for i in range(
            stacked.sizes["point"]
        ):
            x = float(
                stacked["x"]
                .isel(point=i)
                .values
            )

            y = float(
                stacked["y"]
                .isel(point=i)
                .values
            )

            if (
                not np.isfinite(x)
                or not np.isfinite(y)
            ):
                continue


            coords = {}

            for dim in common_dims:
                if dim in stacked.coords:
                    value = (
                        stacked[dim]
                        .isel(point=i)
                        .values
                    )

                    if np.ndim(value) == 0:
                        value = value.item()

                    coords[dim] = value


            points.append(
                ScatterPoint(
                    x=x,
                    y=y,
                    variable=s.var_fc,
                    region=s.region_name,
                    leadtime=coords.get(
                        leadtime_agg_coord,
                        "all",
                    ),
                    start_month=coords.get(
                        period_dim,
                        "all",
                    ),
                    total_months=(
                        get_total_months(s)
                    ),
                    experiment=s.output_name,
                )
            )

    return points


def main() -> None:

    # ==========================================================
    # Paths
    # ==========================================================

    experiments_root = Path(
        "/Users/jacopodallaglio/ML/"
        "training/seasonal/experiments"
    )

    plot_dir = Path(
        "/Users/jacopodallaglio/ML/"
        "training/seasonal/plots/scatter"
    )


    # ==========================================================
    # Plot settings
    # ==========================================================

    plot_title = True

    regenerate_plots = True

    fit_lines = False


    # ==========================================================
    # Scatter configuration
    # ==========================================================

    # Y axis:
    # quality of the original forecast.
    forecast_metric = "r2_anom"

    # X axis:
    # improvement of MLFC relative to FC.
    diff_metric = "nrmse_anom"

    improvement_unit: ImprovementUnit = "%"
    # "%"
    # "Δ"

    color_by = "variable"
    marker_by = "leadtime"
    shade_by = "total_months"


    # ==========================================================
    # Data processing
    # ==========================================================

    interpolate = True
    build_analysis = True

    recalculate_climatology = False

    realization_agg = True


    # ==========================================================
    # Climatology
    # ==========================================================

    clim_period: ClimPeriod = ClimPeriod.MONTH
    clim_rolling_window = None

    # Alternative:
    #
    # clim_period = ClimPeriod.DAYOFYEAR_HOUR
    # clim_rolling_window = 31


    # ==========================================================
    # Time selection
    # ==========================================================

    time_range = None
    # time_range = (
    #     "2018-01-01",
    #     "2022-12-31",
    # )

    wanted_start_periods = [
        "all",
    ]


    # ==========================================================
    # Lead-time aggregation
    # ==========================================================

    leadtime_units = LeadtimeUnit.MONTHS

    leadtime_agg_mode: LeadtimeAgg = "single"
    # "single"
    # "aggregated"
    # "seasonal_window"


    # ==========================================================
    # Metric aggregation
    # ==========================================================

    metric_agg_mode: MetricAgg = "global"
    # "global"
    # "spatial_avg"
    # "spatial_rmse"


    # ==========================================================
    # Variables and regions
    # ==========================================================

    variables = [
        # Atmosphere
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

        net_name="SmaAt_UNet",

        target_mode="anomaly_residual",

        extra_suffix_folder="random_split",

        # seasonal_encoding=True,
        # ensemble_encoding=True,
        # input_realization_avg=True,
        # channel_representation="variable",
    )

    print(
        f"Found {len(settings)} "
        f"matching experiment(s)."
    )


    # ==========================================================
    # Group comparable experiments
    # ==========================================================

    groups = defaultdict(list)

    for s in settings:
        key = s.comparison_key(
            ignore={
                "root_dir",
                "region_name",
                "region",
                "trainer_precision",
                "var_fc",
                "var_an",
            }
        )

        groups[key].append(s)

    print(
        f"Found {len(groups)} groups"
    )


    # ==========================================================
    # Dimension names
    # ==========================================================

    leadtime_agg_coord = (
        "leadtime"
        if leadtime_agg_mode == "single"
        else "leadtime_seasonal"
    )

    period_dim = f"start_{clim_period}"


    # ==========================================================
    # Collect scatter points
    # ==========================================================

    all_points: list[ScatterPoint] = []

    used_time_ranges: set[
        tuple[str, str]
    ] = set()


    for group in groups.values():

        common_s: Settings = next(
            iter(group)
        )


        # ------------------------------------------------------
        # Evaluation range
        # ------------------------------------------------------

        valid_time_range = (
            (
                common_s.train_start,
                common_s.test_end,
            )
            if time_range is None
            else time_range
        )

        used_time_ranges.add(
            tuple(valid_time_range)
        )


        # ------------------------------------------------------
        # Climatology range
        #
        # Deliberately stop at train_end to avoid using
        # validation/test information in the climatology.
        # ------------------------------------------------------

        clim_time_range = (
            common_s.train_start,
            common_s.train_end,
        )


        points = iter_scalar_points(
            group,
            forecast_metric=forecast_metric,
            diff_metric=diff_metric,
            improvement_unit=improvement_unit,
            metric_agg_mode=metric_agg_mode,
            leadtime_agg=leadtime_agg_mode,
            realization_agg=realization_agg,
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=valid_time_range,
            clim_time_range=clim_time_range,
            leadtime_units=leadtime_units,
            leadtime_agg_coord=leadtime_agg_coord,
            recalculate_climatology=(
                recalculate_climatology
            ),
            clim_period=clim_period,
            clim_rolling_window=(
                clim_rolling_window
            ),
            period_dim=period_dim,
            wanted_start_periods=(
                wanted_start_periods
            ),
            interpolate=interpolate,
            build_analysis=build_analysis,
        )

        all_points.extend(points)


    # ==========================================================
    # Output naming
    # ==========================================================

    improvement_suffix = {
        "%": "percentage",
        "Δ": "difference",
    }[improvement_unit]


    if len(used_time_ranges) == 1:
        output_time_range: object = next(
            iter(used_time_ranges)
        )
    else:
        output_time_range = "mixed"


    common_path = (
        Path(
            f"time_"
            f"{safe_label(output_time_range)}"
            f"_lat_{safe_label(lat_range)}"
            f"_lon_{safe_label(lon_range)}"
        )
        / metric_agg_mode
    )


    filename = (
        f"{forecast_metric}_vs_"
        f"{diff_metric}_"
        f"{improvement_suffix}_"
        f"all_variables.png"
    )


    out_file = (
        plot_dir
        / common_path
        / filename
    )


    if (
        out_file.exists()
        and not regenerate_plots
    ):
        print(
            f"Scatter already exists: "
            f"{out_file}"
        )
        return


    # ==========================================================
    # Labels
    # ==========================================================

    forecast_metric_name = (
        METRIC_NAMES.get(
            forecast_metric,
            forecast_metric,
        )
    )

    diff_metric_name = (
        METRIC_NAMES.get(
            diff_metric,
            diff_metric,
        )
    )


    if improvement_unit == "%":
        improvement_label = (
            f"{diff_metric_name} improvement "
            f"of MLFC vs FC (%)"
        )

    else:
        improvement_label = (
            f"{diff_metric_name} improvement "
            f"difference of MLFC vs FC"
        )


    title = (
        (
            f"All variables · "
            f"{metric_agg_mode} metrics · "
            f"{forecast_metric_name} vs "
            f"{diff_metric_name} improvement"
        )
        if plot_title
        else None
    )


    # ==========================================================
    # Plot
    # ==========================================================

    print(
        f"Saving combined scatter "
        f"{out_file}"
    )

    plot_metric_diff_scatter(
        all_points,
        forecast_metric=forecast_metric,
        diff_metric=diff_metric,
        out_file=out_file,

        color_by=color_by,
        marker_by=marker_by,
        shade_by=shade_by,

        title=title,
        xlabel=improvement_label,
        ylabel=f"FC {forecast_metric_name}",

        figsize=(12, 12),
        cmap_name="tab10",
        markers=("o", "s", "^", "D", "v", "P", "X"),

        point_size=22,
        edgecolor="black",
        linewidth=0.3,

        shade_strength=0.65,

        fit_lines=fit_lines,
        fit_min_points=3,

        shade_improvement_region=True,
    )


    print(
        f"Done. Saved combined scatter "
        f"with {len(all_points)} points."
    )


if __name__ == "__main__":
    main()
