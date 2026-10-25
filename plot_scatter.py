from pathlib import Path
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")

from earthml import (
    LeadtimeUnit,
    ClimPeriod,
    Settings,
    get_experiment_configs,
    get_and_subset_datasets,
)
from earthml.metrics import (
    get_scalar_metrics,
    ImprovementAgg,
    get_aggregated_metric_improvements,
    get_required_improvement_metrics,
    calculate_save_and_subset_climatologies,
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


def main() -> None:

    # ==========================================================
    # Paths
    # ==========================================================

    exp_name = "weather_atmo"
    # exp_name = "weather_atmo_ablation_fixed_val"
    # exp_name = "weather_atmo_short_zero_vs_replicate_padding"

    experiments_root = Path(
        "/Users/jacopodallaglio/ML/training/seasonal/experiments"
        # f"/work/cmcc/jd19424/ML/MLBC/experiments/{exp_name}"
    )
    common_plot_dir = Path(
        "/Users/jacopodallaglio/ML/training/seasonal/plots/scatter"
        # f"/work/cmcc/jd19424/ML/MLBC/plots/{exp_name}/scatter"
    )


    # ==========================================================
    # Plot settings
    # ==========================================================

    plot_title = True
    plot_labels = True

    title_size = None
    label_size = None
    tick_size = None
    dpi = 300

    regenerate_plots = True

    fit_lines = False

    # ==========================================================
    # Scatter configuration
    # ==========================================================

    # Y axis:
    # quality of the original forecast.
    forecast_metric = "r2"

    # X axis:
    # improvement of MLFC relative to FC.
    diff_metric = "rmse"

    improvement_unit: ImprovementUnit = "Δ"
    # "%"
    # "Δ"
    # "normalized"

    color_by = "experiment"
    # color_by = "variable"
    marker_by = "leadtime"
    shade_by = "total_months"

    # ==========================================================
    # Category custom labels
    # ==========================================================

    def experiment_label(s: Settings) -> str:
        return (
            f"IM={s.normalization_input_mean}, "
            f"IS={s.normalization_input_std}, "
            f"TM={s.normalization_target_mean}, "
            f"TS={s.normalization_target_std}"
            + (
                f", SM={s.sample_target_mean}, SS={s.sample_target_std}"
                if s.normalization_input_mean == "sample"
                or s.normalization_input_std == "sample"
                or s.normalization_target_mean == "sample"
                or s.normalization_target_std == "sample"
                else ""
            )
        )

    category_labels = {
        "leadtime": {
            12: "12h",
            24: "24h",
            36: "36h",
            48: "48h",
            60: "60h",
            72: "72h",
        },
    }

    # ==========================================================
    # Data processing
    # ==========================================================

    interpolate = True # seasonal
    # interpolate = False # weather
    build_analysis = True

    recalculate_climatology = False

    realization_agg = True # seasonal
    # realization_agg = False # weather

    # ==========================================================
    # Climatology
    # ==========================================================

    clim_period: ClimPeriod = ClimPeriod.MONTH
    clim_rolling_window = None

    # clim_period = ClimPeriod.DAYOFYEAR_HOUR
    # clim_rolling_window = 31

    # ==========================================================
    # Time selection
    # ==========================================================

    time_range = None
    # time_range = ("2018-01-01", "2022-12-31")

    wanted_start_periods = [
        # "01",
        # "02",
        # "03",
        # "04",
        # "05",
        # "06",
        # "07",
        # "08",
        # "09",
        # "10",
        # "11",
        # "12",
        "all",
    ]

    # ==========================================================
    # Lead-time aggregation
    # ==========================================================

    leadtime_units = LeadtimeUnit.MONTHS # seasonal
    # leadtime_units = LeadtimeUnit.HOURS # weather

    leadtime_agg_mode: LeadtimeAgg = "aggregated"
    # leadtime_agg_mode: LeadtimeAgg = "single"
    # "single" for weather
    # "aggregated" for seasonal
    # "seasonal_window"

    # ==========================================================
    # Metric aggregation
    # ==========================================================

    metric_agg_mode: MetricAgg = "timeseries_avg"
    # "global"
    # "spatial_avg"
    # "timeseries_avg"

    improvement_agg_mode: ImprovementAgg = "before"
    # "after": aggregate intermediate metric fields first, then calculate improvements
    # "before": calculate improvements from intermediate metric fields, then aggregate each improvement

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

        net_name="SmaAt_UNet",

        # target_mode="analysis",

        # extra_suffix_folder="random_split",
        # extra_suffix_folder="",

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
    # Experiment labels
    # ==========================================================

    experiment_labels = {
        s.output_name: experiment_label(s)
        for s in settings
    }

    category_labels["experiment"] = experiment_labels

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

    period_reference = "init"
    period_dim = f"{period_reference}_{clim_period.value}"

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
                # common_s.train_start,
                common_s.test_start,
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
        # ------------------------------------------------------

        clim_time_range = (
            common_s.train_start,
            common_s.train_end,
            # common_s.val_end,
        )

        points = iter_scalar_points(
            group,
            forecast_metric=forecast_metric,
            diff_metric=diff_metric,
            improvement_unit=improvement_unit,
            improvement_agg_mode=improvement_agg_mode,
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
            period_reference=period_reference,
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
        "normalized": "normalized",
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
        / improvement_agg_mode
    )

    filename = (
        f"{forecast_metric}_vs_"
        f"{diff_metric}_"
        f"{improvement_suffix}_"
        f"all_variables.png"
    )

    out_file = (
        common_plot_dir
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

        category_labels=category_labels,

        title=title,
        xlabel=improvement_label,
        ylabel=f"FC {forecast_metric_name}",
        plot_title=plot_title,
        plot_labels=plot_labels,
        title_size=title_size,
        label_size=label_size,
        tick_size=tick_size,
        dpi=dpi,

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



def iter_scalar_points(
    settings: Sequence[Settings],
    *,
    forecast_metric: str,
    diff_metric: str,
    improvement_unit: ImprovementUnit,
    improvement_agg_mode: ImprovementAgg,
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
    period_dim: str = "init_month",
    period_reference: str = "init",
    wanted_start_periods: Sequence[str] | None = None,
    interpolate: bool = False,
    build_analysis: bool = True,
) -> list[ScatterPoint]:

    points: list[ScatterPoint] = []

    for s in settings:

        # ======================================================
        # Data and climatologies (loaded once per experiment)
        # ======================================================

        fc, an, mlfc = get_and_subset_datasets(
            s,
            leadtime_units=leadtime_units,
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=time_range,
            interpolate=interpolate,
        )

        if mlfc is None:
            raise ValueError(
                f"ML-corrected forecast is not available for {s.output_name}."
            )

        fc_clim, an_clim, mlfc_clim = (
            calculate_save_and_subset_climatologies(
                s,
                leadtime_units=leadtime_units,
                force=recalculate_climatology,
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
        )

        # ======================================================
        # Metric profiles and intermediate fields
        # ======================================================

        common_kwargs = dict(
            an=an,
            var=s.var_fc,
            realization_agg=realization_agg,
            metric_agg_mode=metric_agg_mode,
            leadtime_agg_mode=leadtime_agg,
            leadtime_agg_coord=leadtime_agg_coord,
            clim_period=clim_period,
            period_reference=period_reference,
            period_dim=period_dim,
            periods_requested=wanted_start_periods,
            leadtime_unit=leadtime_units,
            leadtime_windows=s.seasonal_leadtime_windows,
        )

        metrics_fc_ds, metrics_fc_fields = get_scalar_metrics(
            fc=fc,
            fc_clim=fc_clim,
            an_clim=an_clim,
            metrics=get_required_improvement_metrics(
                list(dict.fromkeys((forecast_metric, diff_metric))),
                improvement_units=(improvement_unit,),
            ),
            **common_kwargs,
        )

        metrics_mlfc_ds, metrics_mlfc_fields = get_scalar_metrics(
            fc=mlfc,
            fc_clim=mlfc_clim,
            an_clim=an_clim,
            metrics=[diff_metric],
            **common_kwargs,
        )

        # ======================================================
        # Y axis: aggregated FC metric
        # ======================================================

        y_da = metrics_fc_ds[forecast_metric]

        # ======================================================
        # X axis: selected order of improvement and aggregation
        # ======================================================

        improvements = get_aggregated_metric_improvements(
            baseline_fields=metrics_fc_fields,
            target_fields=metrics_mlfc_fields,
            an=an,
            metric=diff_metric,
            baseline_model="fc",
            target_model="mlfc",
            metric_agg_mode=metric_agg_mode,
            improvement_agg_mode=improvement_agg_mode,
            improvement_units=(improvement_unit,),
        )

        improvement_suffix = {
            "%": "percentage",
            "Δ": "difference",
            "normalized": "normalized",
        }[improvement_unit]
        x_da = improvements[f"mlfc_vs_fc_{improvement_suffix}"]

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


if __name__ == "__main__":
    main()
