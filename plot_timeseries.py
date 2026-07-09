from typing import Literal
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
    LeadtimeUnit,
    ClimPeriod,
    get_experiment_configs,
    get_and_subset_datasets,
    aggregate_leadtime_ds,
)
from earthml.metrics import (
    LeadtimeAgg,
    MetricAgg,
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
)


def main() -> None:
    experiments_root = Path("/Users/jacopodallaglio/ML/training/seasonal/experiments")

    plot_mode: PlotMode = "timeseries"
    separate_plots = False 
    regenerate_plots = False

    force_clim_recalc = False
    interpolate = True
    build_analysis = True

    variables = [
        # Atmo
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

    leadtime_units = LeadtimeUnit.MONTHS
    clim_period: ClimPeriod = ClimPeriod.MONTH # "dayofyear", "day", "month", "year", "day_hour", "dayofyear_hour", "month_hour"
    clim_rolling_window = None

    time_range = None
    # time_range = ("2018-01-01", "2022-12-31")

    leadtime_agg_mode: LeadtimeAgg = "aggregated" # "single", "aggregated", "seasonal_window"
    plot_ens_mean = False
    category: Literal[
        "raw",
        "residual",
        "anomaly",
        "anomaly_residual",
    ] = "anomaly"

    rolling_mean_window = None # e.g. 3, 5, 12, None disables
    rolling_mean_center = True
    rolling_mean_min_periods = 1

    series_offsets = {
        "Forecast": 0.0,
        "Corrected forecast": 2.0,
        "Analysis": 4.0,
    }

    settings = get_experiment_configs(
        experiments_root,
        var_fc=variables,
        region_name=regions,
        net_name="SmaAt_UNet",
        target_mode="anomaly_residual",
        extra_suffix_folder="random_split",
    )

    print(f"Found {len(settings)} matching experiment(s).")

    n = 0
    for s in settings:
        valid_time_range = (s.train_start, s.test_end) if time_range is None else time_range
        clim_time_range = (s.train_start, s.train_end)

        lat_lon = list(s.region.values()) if s.region is not None else [None, None]
        valid_lat_range = lat_lon[0] if lat_range is None else lat_range
        valid_lon_range = lat_lon[1] if lon_range is None else lon_range

        leadtime_agg_coord = "leadtime" if leadtime_agg_mode=="single" else "leadtime_seasonal"

        print(f"Generate {leadtime_agg_mode} {category} {plot_mode} for {s.var_an, s.var_fc} in {s.region_name} (lon={valid_lon_range}, lat={valid_lat_range})")

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

        # Aggregate if requested
        if leadtime_agg_mode != "single" and s.seasonal_leadtime_windows is not None:
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
            mlfc = aggregate_leadtime_ds(
                ds=mlfc,
                windows=s.seasonal_leadtime_windows,
                leadtime_dim=mlfc.earthml.guessed_dims.leadtime,
                leadtime_agg_coord=leadtime_agg_coord,
            ) if mlfc is not None else None
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

        time_dim = fc.earthml.guessed_dims.time
        lat_dim = fc.earthml.guessed_dims.latitude
        lon_dim = fc.earthml.guessed_dims.longitude
        realization_dim = fc.earthml.guessed_dims.realization

        fc_da, an_da = fc[s.var_fc], an[s.var_an]
        fc_clim_da, an_clim_da = fc_clim[s.var_fc], an_clim[s.var_an]
        mlfc_da = mlfc[s.var_fc] if mlfc is not None else None
        mlfc_clim_da = mlfc_clim[s.var_fc] if mlfc_clim is not None else None

        fc_da, fc_clim_da = xr.unify_chunks(fc_da, fc_clim_da)
        an_da, an_clim_da = xr.unify_chunks(an_da, an_clim_da)
        if mlfc_da is not None and mlfc_clim_da is not None:
            mlfc_da, mlfc_clim_da = xr.unify_chunks(mlfc_da, mlfc_clim_da)

        fc_clim_da = stack_hour_clim(fc_clim_da, clim_period)
        an_clim_da = stack_hour_clim(an_clim_da, clim_period)
        mlfc_clim_da = stack_hour_clim(mlfc_clim_da, clim_period) if mlfc_clim_da is not None else None

        fc_anom_da = groupby_period(fc_da, time_dim, clim_period) - fc_clim_da
        an_anom_da = groupby_period(an_da, time_dim, clim_period) - an_clim_da
        mlfc_anom_da = (
            groupby_period(mlfc_da, time_dim, clim_period) - mlfc_clim_da
            if mlfc_da is not None and mlfc_clim_da is not None
            else None
        )

        if category == "anomaly":
            fc_ts_da, an_ts_da = fc_anom_da, an_anom_da
            mlfc_ts_da = mlfc_anom_da if mlfc_anom_da is not None else None
            category_title = " anomaly "
        elif category == "anomaly_residual":
            fc_ts_da = fc_anom_da - an_anom_da
            mlfc_ts_da = mlfc_anom_da - an_anom_da if mlfc_anom_da is not None else None
            an_ts_da = None
            category_title = " anomaly residual "
        elif category == "residual":
            fc_ts_da = fc_da - an_da
            mlfc_ts_da = mlfc_da - an_da if mlfc_da is not None else None
            an_ts_da = None
            category_title = " residual "
        else:
            fc_ts_da, an_ts_da = fc_da, an_da
            mlfc_ts_da = mlfc_da if mlfc_da is not None else None
            category_title = ""

        for lt in fc[leadtime_agg_coord].values:
            fc_ts_lead_da = fc_ts_da.sel({leadtime_agg_coord: lt})

            fc_ts_lead_da = fc_ts_lead_da.chunk({time_dim: rolling_mean_window}).rolling(
                {time_dim: rolling_mean_window},
                center=rolling_mean_center,
                min_periods=rolling_mean_min_periods,
            ).mean() if rolling_mean_window is not None else fc_ts_lead_da

            if realization_dim is not None:
                fc_ts_lead_da_ens_mean = fc_ts_lead_da.mean(realization_dim) if plot_ens_mean else None
            else:
                fc_ts_lead_da_ens_mean = fc_ts_lead_da

            an_ts_lead_da = an_ts_da.sel({leadtime_agg_coord: lt}) if an_ts_da is not None else None

            an_ts_lead_da = an_ts_lead_da.chunk({time_dim: rolling_mean_window}).rolling(
                {time_dim: rolling_mean_window},
                center=rolling_mean_center,
                min_periods=rolling_mean_min_periods,
            ).mean() if rolling_mean_window is not None and an_ts_lead_da is not None else an_ts_lead_da

            mlfc_ts_lead_da = mlfc_ts_da.sel({leadtime_agg_coord: lt}) if mlfc_ts_da is not None else None

            if mlfc_ts_lead_da is not None:
                mlfc_ts_lead_da = mlfc_ts_lead_da.chunk({time_dim: rolling_mean_window}).rolling(
                    {time_dim: rolling_mean_window},
                    center=rolling_mean_center,
                    min_periods=rolling_mean_min_periods,
                ).mean() if rolling_mean_window is not None else mlfc_ts_lead_da

                if realization_dim is not None:
                    mlfc_ts_lead_da_ens_mean = mlfc_ts_lead_da.mean(realization_dim) if plot_ens_mean else None
                else:
                    mlfc_ts_lead_da_ens_mean = mlfc_ts_lead_da
            else:
                mlfc_ts_lead_da_ens_mean = None

            if mlfc_ts_lead_da is None:
                series = {
                    "Forecast": fc_ts_lead_da_ens_mean,
                    "Analysis": an_ts_lead_da if an_ts_lead_da is not None else None,
                }
                member_series = {
                    "Forecast": fc_ts_lead_da,
                }
            else:
                series = {
                    "Forecast": fc_ts_lead_da_ens_mean,
                    "Corrected forecast": mlfc_ts_lead_da_ens_mean,
                    "Analysis": an_ts_lead_da if an_ts_lead_da is not None else None,
                }
                member_series = {
                    "Forecast": fc_ts_lead_da,
                    "Corrected forecast": mlfc_ts_lead_da,
                }

            label = safe_label(lead_label(fc_ts_lead_da, lt, leadtime_agg_coord))
            rolling_label = f" roll {rolling_mean_window} " if rolling_mean_window is not None else ""
            rolling_filename = f"_roll{rolling_mean_window}" if rolling_mean_window is not None else ""

            out_file = (
                s.plot_dir / "timeseries" / category
                / f"time_{safe_label(valid_time_range)}_lat_{safe_label(lat_range)}_lon_{safe_label(lon_range)}"
                / leadtime_agg_mode
                / f"{s.var_fc}_lead_{label}_{category}{rolling_filename}_timeseries.png"
            )

            if out_file.exists() and regenerate_plots == False:
                continue

            plot_field_timeseries(
                series=series,
                member_series=member_series,
                var=s.var_fc,
                title=f"{VARIABLE_NAMES.get(s.var_fc, s.var_fc.upper())}{category_title}{rolling_label}· lead={label}",
                out_file=out_file,
                time_dim=time_dim,
                spatial_dims=(lat_dim, lon_dim),
                realization_dim=realization_dim,
                train_end=s.train_end,
                plot_single_members=False,
                member_linestyle="-",
                series_linestyle="--" if plot_ens_mean else "-",
                series_offsets=series_offsets if separate_plots else None
            )
            n += 1

    print(f"Done. Saved {n} plots.")


if __name__ == "__main__":
    main()
