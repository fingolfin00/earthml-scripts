from typing import Literal

from pathlib import Path

from rich import print

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import earthml
from earthml import (
    Settings,
    get_experiment_configs,
    get_and_subset_datasets,
    calculate_save_and_subset_climatologies,
)
from earthml.plots import (
    plot_field_timeseries,
    plot_field_map,
    safe_label,
    VARIABLE_NAMES,
)


def main() -> None:
    experiments_root = Path("./experiments")
    force_clim_recalc = False

    variables = ["mslp"]
    regions = ["World"] # or None to accept all

    # ConUS
    lat_range = (50, 25)
    lon_range = (-130, -60)
    # World
    # lat_range = None
    # lon_range = None
    time_range = None
    # time_range = ("2018-01-01", "2022-12-31")

    settings = get_experiment_configs(experiments_root, variables, regions)

    print(f"Found {len(settings)} matching experiment(s).")

    n = 0
    for s in settings:
        print(f"Plotting fields for {s.output_name}")

        out_dir = s.plot_dir / "fields"

        valid_time_range = (s.train_start, s.test_end) if time_range is None else time_range
        fc, an, mlfc = get_and_subset_datasets(
            s,
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=valid_time_range,
        )

        fc_clim, an_clim, mlfc_clim = calculate_save_and_subset_climatologies(
            s,
            force=force_clim_recalc,
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=(s.train_start, s.test_end),
        )

        var = s.var_fc
        time_dim = fc.earthml.guessed_dims.time
        lat_dim = fc.earthml.guessed_dims.latitude
        lon_dim = fc.earthml.guessed_dims.longitude
        leadtime_dim = fc.earthml.guessed_dims.leadtime
        realization_dim = fc.earthml.guessed_dims.realization
        for lt in fc[leadtime_dim].values:
            fc_lead = fc[var].sel({leadtime_dim: lt})
            fc_lead_ens_mean = fc_lead.mean(realization_dim)
            an_lead = an[var].sel({leadtime_dim: lt})
            mlfc_lead = mlfc[var].sel({leadtime_dim: lt}) if mlfc is not None else None
            mlfc_lead_ens_mean = mlfc_lead.mean(realization_dim) if mlfc_lead is not None else None

            if mlfc_lead is None:
                series = {
                    "Forecast": fc_lead_ens_mean,
                    "Analysis": an_lead,
                }
                member_series = {
                    "Forecast": fc_lead,
                }
            else:
                series = {
                    "Forecast": fc_lead_ens_mean,
                    "Corrected forecast": mlfc_lead_ens_mean,
                    "Analysis": an_lead,
                }
                member_series = {
                    "Forecast": fc_lead,
                    "Corrected forecast": mlfc_lead,
                }
            plot_field_timeseries(
                series=series,
                member_series=member_series,
                var=var,
                title=f"{VARIABLE_NAMES.get(var, var.upper())} · lead {lt}",
                out_file=out_dir / "timeseries" / "raw" / f"{var}_lead_{safe_label(lt)}_timeseries.png",
                time_dim=time_dim,
                spatial_dims=(lat_dim, lon_dim),
                train_end=s.train_end,
            )
            n += 1

            # Ensemble mean maps over time
            map_items = {
                "forecast": fc_lead_ens_mean.mean(time_dim, skipna=True),
                "analysis": an_lead.mean(time_dim, skipna=True),
            }

            if mlfc_lead_ens_mean is not None:
                map_items["corrected_forecast"] = mlfc_lead_ens_mean.mean(time_dim, skipna=True)
                map_items["forecast_minus_analysis"] = (fc_lead_ens_mean - an_lead).mean(time_dim, skipna=True)
                map_items["corrected_minus_analysis"] = (mlfc_lead_ens_mean - an_lead).mean(time_dim, skipna=True)
                map_items["forecast_minus_corrected"] = (fc_lead_ens_mean - mlfc_lead_ens_mean).mean(time_dim, skipna=True)

            for field_name, da_map in map_items.items():
                is_difference = "minus" in field_name
                plot_field_map(
                    da_map,
                    var=var,
                    title=f"{VARIABLE_NAMES.get(var, var.upper())} · {field_name.replace('_', ' ')} · lead {lt}",
                    out_file=out_dir / "maps" / field_name / f"{var}_{field_name}_lead_{safe_label(lt)}.png",
                    cmap="RdBu_r" if is_difference else "jet",
                    centered=is_difference,
                )
                n += 1

            # Anomaly timeseries
            fc_clim_lead = fc_clim[var].sel({leadtime_dim: lt})
            an_clim_lead = an_clim[var].sel({leadtime_dim: lt})
            mlfc_clim_lead = mlfc_clim[var].sel({leadtime_dim: lt}) if mlfc_clim is not None else None

            fc_anom = fc_lead.groupby(f"{time_dim}.month") - fc_clim_lead
            an_anom = an_lead.groupby(f"{time_dim}.month") - an_clim_lead

            fc_member_anom = fc_lead.groupby(f"{time_dim}.month") - fc_clim_lead

            mlfc_anom = None
            mlfc_member_anom = None
            if mlfc_lead is not None and mlfc_clim_lead is not None:
                mlfc_anom = mlfc_lead.groupby(f"{time_dim}.month") - mlfc_clim_lead.mean(realization_dim)
                mlfc_member_anom = mlfc_lead.groupby(f"{time_dim}.month") - mlfc_clim_lead
                series = {
                    "Forecast": fc_anom.mean(realization_dim),
                    "Corrected forecast": mlfc_anom.mean(realization_dim),
                    "Analysis": an_anom,
                }
                member_series = {
                    "Forecast": fc_member_anom,
                    "Corrected forecast": mlfc_member_anom,
                }
            else:
                series = {
                    "Forecast": fc_anom.mean(realization_dim),
                    "Analysis": an_anom,
                }
                member_series = {
                    "Forecast": fc_anom,
                }

            plot_field_timeseries(
                series=series,
                member_series=member_series,
                var=var,
                title=f"{VARIABLE_NAMES.get(var, var.upper())} anomaly · lead {safe_label(lt)}",
                out_file=out_dir / "timeseries" / "raw" / f"{var}_lead_{safe_label(lt)}_anomaly_timeseries.png",
                time_dim=time_dim,
                spatial_dims=(lat_dim, lon_dim),
                train_end=s.train_end,
            )
            n += 1

    print(f"Done. Saved {n} field plots.")


if __name__ == "__main__":
    main()
