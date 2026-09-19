from pathlib import Path
from typing import Literal

import warnings

import xarray as xr
from dask.array import PerformanceWarning

from earthml import (
    LeadtimeUnit,
    ClimPeriod,
    get_experiment_configs,
    get_and_subset_datasets,
    aggregate_leadtime_ds,
)
from earthml.metrics import (
    LeadtimeAgg,
    stack_hour_clim,
    groupby_period,
    calculate_save_and_subset_climatologies,
)
from earthml.plots import (
    safe_label,
    lead_label,
    PlotMode,
    plot_field_timeseries,
    VARIABLE_NAMES,
    VARIABLE_UNITS,
)

from cities import CITY_LOCATIONS

warnings.simplefilter("ignore", FutureWarning)
warnings.filterwarnings(
    "ignore",
    category=PerformanceWarning,
)


def main() -> None:

    # ==========================================================
    # Paths
    # ==========================================================

    experiments_root = Path(
        "/Users/jacopodallaglio/ML/training/seasonal/experiments"
        # "/work/cmcc/jd19424/ML/MLBC/experiments/weather_atmo"
    )


    # ==========================================================
    # Plot settings
    # ==========================================================

    plot_mode: PlotMode = "timeseries"

    plot_title = True
    plot_labels = True

    title_size = None
    label_size = None
    tick_size = None
    dpi = 300

    plot_mlfc = True
    include_clim_fc = True

    plot_ens_mean = False
    plot_single_members = False

    offset_plots = False

    regenerate_plots = True

    # ==========================================================
    # Timeseries representation
    # ==========================================================

    category: Literal[
        "raw",
        "error",
        "absolute_error",
        "anomaly",
        "anomaly_error",
    ] = "absolute_error"

    # ==========================================================
    # Rolling mean
    # ==========================================================

    rolling_mean_window = None
    # Number of samples, not number of days.
    # Example for 12-hour data:
    #     30 samples = 15 days
    #     60 samples = 30 days
    # rolling_mean_window = 30

    rolling_mean_center = True
    rolling_mean_min_periods = 1


    # ==========================================================
    # Offset plots
    # ==========================================================

    series_offsets = {
        "Forecast": 0.0,
        "Clim-corrected forecast": 2.0,
        "Corrected forecast": 4.0,
        "Analysis": 6.0,
    }

    # ==========================================================
    # Data processing
    # ==========================================================

    interpolate = True # seasonal
    # interpolate = False # weather
    build_analysis = True

    recalculate_climatology = False

    # ==========================================================
    # Climatology
    # ==========================================================

    clim_period: ClimPeriod = ClimPeriod.MONTH
    clim_rolling_window = None

    # clim_period: ClimPeriod = ClimPeriod.DAYOFYEAR_HOUR
    # clim_rolling_window = 31

    # ==========================================================
    # Time selection
    # ==========================================================

    time_range = None
    # time_range = ("2018-01-01", "2022-12-31")

    inference_period = None
    # inference_period = ("2025-01-01", "2025-10-31")

    # ==========================================================
    # Lead-time aggregation
    # ==========================================================

    leadtime_units = LeadtimeUnit.MONTHS # seasonal
    # leadtime_units = LeadtimeUnit.HOURS # weather

    leadtime_agg_mode: LeadtimeAgg = "aggregated"
    # "single" for weather
    # "aggregated" for seasonal
    # "seasonal_window"

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
    # Spatial subset (from training experiment)
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
    # lat_range = None
    # lon_range = None

    # ==========================================================
    # Timeseries spatial subregions
    # ==========================================================

    locations = CITY_LOCATIONS

    # ==========================================================
    # Experiment selection
    # ==========================================================

    settings = get_experiment_configs(
        experiments_root,
        var_fc=variables,
        region_name=regions,

        net_name="SmaAt_UNet",
        # net_name="ConvNeXtTransformerUNet",

        # target_mode="anomaly_residual",
        # seasonal_encoding=True,
        # ensemble_encoding=True,
        # input_realization_avg=True,
        # extra_suffix_folder="random_split",
    )

    print(f"Found {len(settings)} matching experiment(s).")

    n = 0

    for s in settings:

        # ======================================================
        # Time range
        # ======================================================

        if inference_period is None:
            valid_time_range = (
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
        )

        # ======================================================
        # Spatial range
        # ======================================================

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

        # ======================================================
        # Lead-time dimension
        # ======================================================

        leadtime_agg_coord = (
            "leadtime"
            if leadtime_agg_mode == "single"
            else "leadtime_seasonal"
        )

        print(
            f"Generate {leadtime_agg_mode} {category} {plot_mode} "
            f"for {(s.var_an, s.var_fc)} in {s.region_name} "
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

        if not plot_mlfc:
            mlfc = None

        # ======================================================
        # Climatologies
        # ======================================================

        fc_clim, an_clim, mlfc_clim = (
            calculate_save_and_subset_climatologies(
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
        )

        if mlfc_clim is not None:
            mlfc_clim = mlfc_clim.assign_coords(
                leadtime=s.leadtimes
            )

        if not plot_mlfc:
            mlfc_clim = None

        # ======================================================
        # Lead-time aggregation
        # ======================================================

        if (
            leadtime_agg_mode != "single"
            and s.seasonal_leadtime_windows is not None
        ):
            an = aggregate_leadtime_ds(
                ds=an,
                windows=s.seasonal_leadtime_windows,
                leadtime_dim=an.earthml.guessed_dims.leadtime,
                leadtime_agg_coord=leadtime_agg_coord,
            )

            fc = aggregate_leadtime_ds(
                ds=fc,
                windows=s.seasonal_leadtime_windows,
                leadtime_dim=fc.earthml.guessed_dims.leadtime,
                leadtime_agg_coord=leadtime_agg_coord,
            )

            if mlfc is not None:
                mlfc = aggregate_leadtime_ds(
                    ds=mlfc,
                    windows=s.seasonal_leadtime_windows,
                    leadtime_dim=mlfc.earthml.guessed_dims.leadtime,
                    leadtime_agg_coord=leadtime_agg_coord,
                )

            if an_clim is not None:
                an_clim = aggregate_leadtime_ds(
                    ds=an_clim,
                    windows=s.seasonal_leadtime_windows,
                    leadtime_dim=an_clim.earthml.guessed_dims.leadtime,
                    leadtime_agg_coord=leadtime_agg_coord,
                )

            if fc_clim is not None:
                fc_clim = aggregate_leadtime_ds(
                    ds=fc_clim,
                    windows=s.seasonal_leadtime_windows,
                    leadtime_dim=fc_clim.earthml.guessed_dims.leadtime,
                    leadtime_agg_coord=leadtime_agg_coord,
                )

            if mlfc_clim is not None:
                mlfc_clim = aggregate_leadtime_ds(
                    ds=mlfc_clim,
                    windows=s.seasonal_leadtime_windows,
                    leadtime_dim=mlfc_clim.earthml.guessed_dims.leadtime,
                    leadtime_agg_coord=leadtime_agg_coord,
                )

        # ======================================================
        # Unit conversion
        # ======================================================

        fc[s.var_fc] = _convert_kelvin_to_celsius(
            fc[s.var_fc],
            var=s.var_fc,
        )

        an[s.var_an] = _convert_kelvin_to_celsius(
            an[s.var_an],
            var=s.var_an,
        )

        if mlfc is not None:
            mlfc[s.var_fc] = _convert_kelvin_to_celsius(
                mlfc[s.var_fc],
                var=s.var_fc,
            )

        fc_clim[s.var_fc] = _convert_kelvin_to_celsius(
            fc_clim[s.var_fc],
            var=s.var_fc,
        )

        an_clim[s.var_an] = _convert_kelvin_to_celsius(
            an_clim[s.var_an],
            var=s.var_an,
        )

        if mlfc_clim is not None:
            mlfc_clim[s.var_fc] = _convert_kelvin_to_celsius(
                mlfc_clim[s.var_fc],
                var=s.var_fc,
            )

        # ======================================================
        # Dimensions
        # ======================================================

        time_dim = fc.earthml.guessed_dims.time
        lat_dim = fc.earthml.guessed_dims.latitude
        lon_dim = fc.earthml.guessed_dims.longitude
        realization_dim = fc.earthml.guessed_dims.realization

        # ======================================================
        # Fields and climatologies
        # ======================================================

        fc_da = fc[s.var_fc]
        an_da = an[s.var_an]

        fc_clim_da = fc_clim[s.var_fc]
        an_clim_da = an_clim[s.var_an]

        mlfc_da = (
            mlfc[s.var_fc]
            if mlfc is not None
            else None
        )

        mlfc_clim_da = (
            mlfc_clim[s.var_fc]
            if mlfc_clim is not None
            else None
        )

        fc_da, fc_clim_da = xr.unify_chunks(
            fc_da,
            fc_clim_da,
        )

        an_da, an_clim_da = xr.unify_chunks(
            an_da,
            an_clim_da,
        )

        if (
            mlfc_da is not None
            and mlfc_clim_da is not None
        ):
            mlfc_da, mlfc_clim_da = xr.unify_chunks(
                mlfc_da,
                mlfc_clim_da,
            )

        fc_clim_da = stack_hour_clim(
            fc_clim_da,
            clim_period,
        )

        an_clim_da = stack_hour_clim(
            an_clim_da,
            clim_period,
        )

        mlfc_clim_da = (
            stack_hour_clim(
                mlfc_clim_da,
                clim_period,
            )
            if mlfc_clim_da is not None
            else None
        )

        # ======================================================
        # Anomalies
        # ======================================================

        fc_anom_da = (
            groupby_period(
                fc_da,
                time_dim,
                clim_period,
            )
            - fc_clim_da
        )

        an_anom_da = (
            groupby_period(
                an_da,
                time_dim,
                clim_period,
            )
            - an_clim_da
        )

        mlfc_anom_da = (
            groupby_period(
                mlfc_da,
                time_dim,
                clim_period,
            )
            - mlfc_clim_da
            if (
                mlfc_da is not None
                and mlfc_clim_da is not None
            )
            else None
        )

        # ======================================================
        # Climatology-corrected forecast
        # ======================================================

        # Analysis climatology expanded onto the full time axis.
        an_clim_for_time_da = (
            an_da - an_anom_da
        )

        # Classical grid-point climatological bias correction:
        #
        #     fc_corrected = fc - fc_clim + an_clim
        #
        fc_clim_corrected_da = (
            fc_anom_da
            + an_clim_for_time_da
        )

        # ======================================================
        # Timeseries category
        # ======================================================

        if category == "anomaly":
            fc_ts_da = fc_anom_da
            fc_ts_clim_corrected_da = fc_anom_da
            an_ts_da = an_anom_da
            mlfc_ts_da = mlfc_anom_da
            category_title = " anomaly "

        elif category == "anomaly_error":
            fc_ts_da = (
                fc_anom_da - an_anom_da
            )

            # A climatology-only correction changes the mean
            # climatological component, not the forecast anomaly.
            fc_ts_clim_corrected_da = (
                fc_anom_da - an_anom_da
            )

            mlfc_ts_da = (
                mlfc_anom_da - an_anom_da
                if mlfc_anom_da is not None
                else None
            )

            an_ts_da = None
            category_title = " anomaly error "

        elif category == "error":
            fc_ts_da = (
                fc_da - an_da
            )

            fc_ts_clim_corrected_da = (
                fc_clim_corrected_da - an_da
            )

            mlfc_ts_da = (
                mlfc_da - an_da
                if mlfc_da is not None
                else None
            )

            an_ts_da = None
            category_title = " error "

        elif category == "absolute_error":
            fc_ts_da = abs(
                fc_da - an_da
            )

            fc_ts_clim_corrected_da = abs(
                fc_clim_corrected_da - an_da
            )

            mlfc_ts_da = (
                abs(mlfc_da - an_da)
                if mlfc_da is not None
                else None
            )

            an_ts_da = None
            category_title = " absolute error "

        else:
            fc_ts_da = fc_da
            fc_ts_clim_corrected_da = (
                fc_clim_corrected_da
            )
            an_ts_da = an_da
            mlfc_ts_da = mlfc_da
            category_title = ""

        for location, location_config in locations.items():
            timeseries_lat_range = location_config["lat_range"]
            timeseries_lon_range = location_config["lon_range"]

            # Restrict timeseries to selected rectangular region
            fc_ts_loc_da = subset_timeseries_region(
                fc_ts_da,
                timeseries_lat_range,
                timeseries_lon_range,
            )

            fc_ts_clim_corrected_loc_da = subset_timeseries_region(
                fc_ts_clim_corrected_da,
                timeseries_lat_range,
                timeseries_lon_range,
            )

            an_ts_loc_da = subset_timeseries_region(
                an_ts_da,
                timeseries_lat_range,
                timeseries_lon_range,
            )

            mlfc_ts_loc_da = subset_timeseries_region(
                mlfc_ts_da,
                timeseries_lat_range,
                timeseries_lon_range,
            )

            # ======================================================
            # Lead-time plots
            # ======================================================

            for lt in fc[leadtime_agg_coord].values:

                # --------------------------------------------------
                # Forecast
                # --------------------------------------------------

                fc_ts_lead_da = fc_ts_loc_da.sel(
                    {leadtime_agg_coord: lt}
                )

                fc_ts_lead_da = apply_rolling_mean(
                    fc_ts_lead_da,
                    time_dim=time_dim,
                    window=rolling_mean_window,
                    center=rolling_mean_center,
                    min_periods=rolling_mean_min_periods,
                )

                if (
                    realization_dim is not None
                    and realization_dim in fc_ts_lead_da.dims
                ):
                    fc_ts_lead_da_ens_mean = (
                        fc_ts_lead_da.mean(
                            realization_dim
                        )
                        if plot_ens_mean
                        else None
                    )

                else:
                    fc_ts_lead_da_ens_mean = (
                        fc_ts_lead_da
                    )

                # --------------------------------------------------
                # Analysis
                # --------------------------------------------------

                an_ts_lead_da = (
                    an_ts_loc_da.sel(
                        {leadtime_agg_coord: lt}
                    )
                    if an_ts_loc_da is not None
                    else None
                )

                an_ts_lead_da = apply_rolling_mean(
                    an_ts_lead_da,
                    time_dim=time_dim,
                    window=rolling_mean_window,
                    center=rolling_mean_center,
                    min_periods=rolling_mean_min_periods,
                )

                # --------------------------------------------------
                # ML-corrected forecast
                # --------------------------------------------------

                mlfc_ts_lead_da = (
                    mlfc_ts_loc_da.sel(
                        {leadtime_agg_coord: lt}
                    )
                    if mlfc_ts_loc_da is not None
                    else None
                )

                mlfc_ts_lead_da = apply_rolling_mean(
                    mlfc_ts_lead_da,
                    time_dim=time_dim,
                    window=rolling_mean_window,
                    center=rolling_mean_center,
                    min_periods=rolling_mean_min_periods,
                )

                if mlfc_ts_lead_da is not None:
                    if (
                        realization_dim is not None
                        and realization_dim
                        in mlfc_ts_lead_da.dims
                    ):
                        mlfc_ts_lead_da_ens_mean = (
                            mlfc_ts_lead_da.mean(
                                realization_dim
                            )
                            if plot_ens_mean
                            else None
                        )

                    else:
                        mlfc_ts_lead_da_ens_mean = (
                            mlfc_ts_lead_da
                        )

                else:
                    mlfc_ts_lead_da_ens_mean = None

                # --------------------------------------------------
                # Climatology-corrected forecast
                # --------------------------------------------------

                fc_ts_lead_clim_corrected_da = (
                    fc_ts_clim_corrected_loc_da.sel(
                        {leadtime_agg_coord: lt}
                    )
                )

                fc_ts_lead_clim_corrected_da = (
                    apply_rolling_mean(
                        fc_ts_lead_clim_corrected_da,
                        time_dim=time_dim,
                        window=rolling_mean_window,
                        center=rolling_mean_center,
                        min_periods=rolling_mean_min_periods,
                    )
                )

                if (
                    realization_dim is not None
                    and realization_dim
                    in fc_ts_lead_clim_corrected_da.dims
                ):
                    fc_ts_lead_clim_corrected_da_ens_mean = (
                        fc_ts_lead_clim_corrected_da.mean(
                            realization_dim
                        )
                        if plot_ens_mean
                        else None
                    )

                else:
                    fc_ts_lead_clim_corrected_da_ens_mean = (
                        fc_ts_lead_clim_corrected_da
                    )

                # ==================================================
                # Series to plot
                # ==================================================

                series = {
                    "Forecast": fc_ts_lead_da_ens_mean,
                }

                member_series = {
                    "Forecast": fc_ts_lead_da,
                }

                if mlfc_ts_lead_da is not None:
                    series[
                        "Corrected forecast"
                    ] = mlfc_ts_lead_da_ens_mean

                    member_series[
                        "Corrected forecast"
                    ] = mlfc_ts_lead_da

                if an_ts_lead_da is not None:
                    series["Analysis"] = (
                        an_ts_lead_da
                    )

                if (
                    include_clim_fc
                    and category in (
                        "raw",
                        "error",
                        "absolute_error",
                    )
                ):
                    series[
                        "Clim-corrected forecast"
                    ] = (
                        fc_ts_lead_clim_corrected_da_ens_mean
                    )

                    member_series[
                        "Clim-corrected forecast"
                    ] = (
                        fc_ts_lead_clim_corrected_da
                    )

                # ==================================================
                # Output
                # ==================================================

                label = safe_label(
                    lead_label(
                        fc_ts_lead_da,
                        lt,
                        leadtime_agg_coord,
                    )
                )

                rolling_label = (
                    f" roll {rolling_mean_window} "
                    if rolling_mean_window is not None
                    else ""
                )

                rolling_filename = (
                    f"_roll{rolling_mean_window}"
                    if rolling_mean_window is not None
                    else ""
                )

                out_file = (
                    s.plot_dir
                    / "timeseries"
                    / category
                    / (
                        f"time_{safe_label(valid_time_range)}"
                        f"_loc_{location}"
                    )
                    / leadtime_agg_mode
                    / (
                        f"{s.var_fc}_lead_{label}_"
                        f"{category}{rolling_filename}_"
                        f"timeseries.png"
                    )
                )

                if (
                    out_file.exists()
                    and not regenerate_plots
                ):
                    continue

                title = (
                    f"{VARIABLE_NAMES.get(s.var_fc, s.var_fc.upper())}"
                    f"{category_title}"
                    f"{rolling_label}"
                    f"· lead={label}"
                    if plot_title
                    else ""
                )

                plot_field_timeseries(
                    series=series,
                    member_series=member_series,
                    var=s.var_fc,
                    title=title,
                    out_file=out_file,
                    time_dim=time_dim,
                    spatial_dims=(
                        lat_dim,
                        lon_dim,
                    ),
                    realization_dim=realization_dim,
                    train_end=s.train_end if valid_time_range[0] <= s.train_end <= valid_time_range[1] else None,
                    val_end=s.val_end if valid_time_range[0] <= s.val_end <= valid_time_range[1] else None,
                    plot_single_members=(
                        plot_single_members
                    ),
                    member_linestyle="-",
                    series_linestyle=(
                        "--"
                        if plot_ens_mean
                        else "-"
                    ),
                    plot_title=plot_title,
                    plot_labels=plot_labels,
                    title_size=title_size,
                    label_size=label_size,
                    tick_size=tick_size,
                    dpi=dpi,
                    series_offsets=(
                        series_offsets
                        if offset_plots
                        else None
                    ),
                )

                n += 1

    print(f"Done. Saved {n} plots.")


def apply_rolling_mean(
    da: xr.DataArray | None,
    *,
    time_dim: str,
    window: int | None,
    center: bool,
    min_periods: int,
) -> xr.DataArray | None:
    if da is None or window is None:
        return da

    return da.rolling(
        {time_dim: window},
        center=center,
        min_periods=min_periods,
    ).mean()


def subset_timeseries_region(
    da: xr.DataArray | None,
    lat_range: tuple | None,
    lon_range: tuple | None,
) -> xr.DataArray | None:
    if da is None:
        return None

    lat_dim = da.earthml.guessed_dims.latitude
    lon_dim = da.earthml.guessed_dims.longitude

    if lat_range is not None:
        da = da.sel({
            lat_dim: slice(*lat_range)
        })

    if lon_range is not None:
        da = da.sel({
            lon_dim: slice(*lon_range)
        })

    return da


def _convert_kelvin_to_celsius(
    da: xr.DataArray,
    *,
    var: str,
) -> xr.DataArray:
    unit = da.attrs.get("units") or VARIABLE_UNITS.get(var, "")

    if unit not in {"K", "Kelvin", "kelvin"}:
        return da

    da = da - 273.15
    da.attrs["units"] = "°C"

    return da


if __name__ == "__main__":
    main()
