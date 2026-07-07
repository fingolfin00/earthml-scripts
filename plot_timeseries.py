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
    aggregate_leadtime_ds,
)
from earthml.metrics import (
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

from settings_plot_seasonal import VARIABLE_PLOT_CONFIG, IMPROVEMENT_PLOT_CONFIG


def main() -> None:
    experiments_root = Path("/Users/jacopodallaglio/ML/training/seasonal/experiments")

    plot_mode: PlotMode = "timeseries"

    force_clim_recalc = False
    interpolate = True
    build_analysis = True

    variables = [
        # Atmo
        "mslp",
        "t2m",
        "d2m",
        "u10",
        "v10",
        "sst",
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

    leadtime_units = "months"
    clim_period = "month" # "dayofyear", "day", "month", "year", "day_hour", "dayofyear_hour", "month_hour"
    clim_rolling_window = None

    time_range = None
    # time_range = ("2018-01-01", "2022-12-31")

    leadtime_agg_mode = "aggregated" # "single", "aggregated", "seasonal_window"
    ens_mean = False
    category = "anomaly_residual" # "raw", "anomaly", "anomaly_residual"

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

        fc_anom = groupby_period(fc[s.var_fc], time_dim, clim_period) - fc_clim
        an_anom = groupby_period(an[s.var_an], time_dim, clim_period) - an_clim
        mlfc_anom = groupby_period(mlfc[s.var_fc], time_dim, clim_period) - mlfc_clim if mlfc is not None and mlfc_clim is not None else None

        if category == "anomaly":
            fc, an = fc_anom[s.var_fc], an_anom[s.var_an]
            mlfc = mlfc_anom[s.var_fc] if mlfc_anom is not None else None
            category_title = " anomaly "
        elif category == "anomaly_residual":
            fc = fc_anom[s.var_fc] - an_anom[s.var_an]
            mlfc = mlfc_anom[s.var_fc] - an_anom[s.var_an] if mlfc_anom is not None else None
            an = None
            category_title = " anomaly residual "
        else:
            fc, an = fc[s.var_fc], an[s.var_an]
            mlfc = mlfc[s.var_fc] if mlfc is not None else None
            category_title = ""

        for lt in fc[leadtime_agg_coord].values:
            fc_lead = fc.sel({leadtime_agg_coord: lt})
            fc_lead_ens_mean = fc_lead.mean(realization_dim)
            an_lead = an.sel({leadtime_agg_coord: lt}) if an is not None else None
            mlfc_lead = mlfc.sel({leadtime_agg_coord: lt}) if mlfc is not None else None
            mlfc_lead_ens_mean = mlfc_lead.mean(realization_dim) if mlfc_lead is not None else None

            if mlfc_lead is None:
                series = {
                    "Forecast": fc_lead_ens_mean if ens_mean else None,
                    "Analysis": an_lead if an is not None else None,
                }
                member_series = {
                    "Forecast": fc_lead,
                }
            else:
                series = {
                    "Forecast": fc_lead_ens_mean if ens_mean else None,
                    "Corrected forecast": mlfc_lead_ens_mean if ens_mean else None,
                    "Analysis": an_lead if an is not None else None,
                }
                member_series = {
                    "Forecast": fc_lead,
                    "Corrected forecast": mlfc_lead,
                }

            label = safe_label(lead_label(fc, lt, leadtime_agg_coord))

            out_file = (
                s.plot_dir / "timeseries" / category
                / f"time_{safe_label(valid_time_range)}_lat_{safe_label(lat_range)}_lon_{safe_label(lon_range)}"
                / leadtime_agg_mode
                / f"{s.var_fc}_lead_{label}_timeseries.png"
            )

            plot_field_timeseries(
                series=series,
                member_series=member_series,
                var=s.var_fc,
                title=f"{VARIABLE_NAMES.get(s.var_fc, s.var_fc.upper())}{category_title}· lead={label}",
                out_file=out_file,
                time_dim=time_dim,
                spatial_dims=(lat_dim, lon_dim),
                train_end=s.train_end,
                plot_single_members=False,
                member_linestyle="-",
                series_linestyle="--" if ens_mean else "-",
            )
            n += 1

    print(f"Done. Saved {n} plots.")


if __name__ == "__main__":
    main()
