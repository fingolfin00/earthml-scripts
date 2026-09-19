from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

import earthml
from earthml import (
    LeadtimeUnit,
    ClimPeriod,
    get_experiment_configs,
    get_and_subset_datasets,
)

from earthml.metrics import (
    calculate_save_and_subset_climatologies,
    stack_hour_clim,
    groupby_period,
)

from earthml.plots import (
    safe_label,
    plot_field_map,
    FieldModel,
    VARIABLE_UNITS,
)

from cities import CITY_LOCATIONS


FieldDifference = Literal[
    "fc-an",
    "clim-fc-an",
    "mlfc-an",
    "clim-fc-fc",
    "mlfc-fc",
    "mlfc-clim-fc",
    "mlfc-fc-abs-error",
    "mlfc-fc-abs-anomaly-error",
]


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
    plot_type = "pcolormesh"
    plot_figsize = (12, 8)

    # cmap = "cmocean:thermal"
    cmap = "cmasher:pride"
    # cmap = "cmasher:torch"
    # anomaly_cmap = "cmasher:fusion"
    anomaly_cmap = "cmocean:balance"
    # difference_cmap = "cmasher:fusion"
    difference_cmap = "cmocean:balance"

    plot_title = True
    plot_labels = True

    title_size = None
    label_size = None
    tick_size = None
    dpi = 300

    plot_title_strftime = "%m.%Y" # seasonal
    # plot_title_strftime = "%d.%m.%Y %H:%M" # weather

    regenerate_plots = False

    # ==========================================================
    # Spatial subregion
    # ==========================================================

    locations = CITY_LOCATIONS

    plot_locations = [
        # "newyork",
        # "miami",
        # "chicago",
        # "denver",
        # "losangeles",
        # "seattle",
        # "houston",
    ]

    rectangles = [
        {
            "lon_range": locations[location]["lon_range"],
            "lat_range": locations[location]["lat_range"],
            "edgecolor": "black",
            "facecolor": "none",
            "linewidth": 3,
            "linestyle": "-",
        }
        for location in plot_locations
    ]

    # ==========================================================
    # Plot limits
    # ==========================================================

    # Raw fields
    common_scale = True
    # robust_quantiles = (0.01, 0.99)  # None -> true min/max
    robust_quantiles = None  # None -> true min/max
    vmin, vmax = -50, 50 # t2m seasonal
    # vmin, vmax = -30, 30 # t2m weather
    # vmin, vmax = None, None
    raw_levels = 21
    raw_centered = True

    # Anomalies
    common_anomaly_scale = True
    # anomaly_quantile = 0.99 # None -> true max(abs(x))
    anomaly_quantile = None
    anomaly_vmax = 6 # t2m
    # anomaly_vmax = None
    anomaly_levels = 13

    # Raw differences
    common_difference_scale = True
    # difference_quantile = 0.99
    difference_quantile = None
    difference_vmax = 7 # t2m
    # difference_vmax = None
    difference_levels = 15

    # Anomaly differences
    common_anomaly_difference_scale = True
    # anomaly_difference_quantile = 0.99
    anomaly_difference_quantile = None
    anomaly_difference_vmax = 6 # t2m
    # anomaly_difference_vmax = None
    anomaly_difference_levels = 13

    # ==========================================================
    # Fields to plot
    # ==========================================================

    plot_models: tuple[FieldModel, ...] = (
        "an",
        "fc",
        # "clim-fc",
        "mlfc",
    )

    plot_anomaly_models: tuple[FieldModel, ...] = (
        "an",
        "fc",
        # "clim-fc",
        "mlfc",
    )

    common_anomaly_scale = True

    # Forecasts with realizations:
    #
    # "mean"   -> ensemble mean
    # "member" -> selected member
    realization_mode: Literal[
        "mean",
        "member",
    ] = "mean"

    realization = 0

    # ==========================================================
    # Differences
    # ==========================================================

    plot_differences: tuple[FieldDifference, ...] = (
        "fc-an",
        # "clim-fc-an",
        "mlfc-an",
        # "clim-fc-fc",
        "mlfc-fc",
        # "mlfc-clim-fc",
        "mlfc-fc-abs-error",
    )

    # Unique anomaly differences. Since clim-fc anomaly == fc anomaly,
    # the other combinations would be exact duplicates or identically zero.
    plot_anomaly_differences: tuple[FieldDifference, ...] = (
        "fc-an",
        "mlfc-an",
        "mlfc-fc",
        "mlfc-fc-abs-anomaly-error",
    )

    # ==========================================================
    # Climatology
    # ==========================================================

    clim_period: ClimPeriod = ClimPeriod.MONTH
    clim_rolling_window = None

    # Weather alternative:
    #
    # clim_period = ClimPeriod.DAYOFYEAR_HOUR
    # clim_rolling_window = 31

    recalculate_climatology = False

    build_analysis = True

    # ==========================================================
    # Time selection
    # ==========================================================

    # Dataset loading range.
    time_range = None

    # time_range = (
    #     "2025-01-01",
    #     "2025-05-12",
    # )

    inference_period = None

    # Individual initialization times.
    #
    # None -> all available times.

    # wanted_times = None

    wanted_times = [
        # seasonal
        "1993-01-01", # first train
        "1994-01-01",
        "2000-01-01",
        "2012-01-01",
        "2014-12-01",
        "2024-01-01",
        "1993-05-01",
        "1994-05-01",
        "1994-05-01",
        "2000-05-01",
        "2012-05-01",
        "2024-05-01",
        # weather
        # "2019-10-14", # first train
        # "2020-01-01",
        # "2020-03-01",
        # "2020-05-01",
        # "2020-08-01",
        # "2020-10-01",
        # "2020-12-01",
        # "2021-01-01",
        # "2021-03-01",
        # "2021-05-01",
        # "2021-08-01",
        # "2021-10-01",
        # "2021-12-01",
        # "2022-01-01",
        # "2022-03-01",
        # "2022-05-01",
        # "2022-08-01",
        # "2022-10-01",
        # "2022-12-01",
        # "2023-01-01",
        # "2023-03-01",
        # "2023-05-01",
        # "2023-08-01",
        # "2023-10-01",
        # "2023-12-01",
        # "2023-12-31", # last train
        # "2024-01-01", # first val
        # "2024-03-01",
        # "2024-05-01",
        # "2024-08-01",
        # "2024-12-31", # last val
        # "2025-01-01", # first test
        # "2025-03-01",
        # "2025-05-01",
        # "2025-08-01",
        # "2025-09-30" # last test
    ]

    # Individual lead times.
    #
    # None -> all experiment lead times.

    wanted_leadtimes = None

    # Seasonal:
    # wanted_leadtimes = [1, 2, 3, 4, 5, 6]

    # Weather:
    # wanted_leadtimes = [24, 72]

    # ==========================================================
    # Data processing
    # ==========================================================

    interpolate = True # seasonal
    # interpolate = False # weather

    leadtime_units = LeadtimeUnit.MONTHS # seasonal
    # leadtime_units = LeadtimeUnit.HOURS # weather

    # ==========================================================
    # Variables and regions
    # ==========================================================

    variables = [
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

    # Whole configured region.
    lat_range = None
    lon_range = None

    # ==========================================================
    # Experiment selection
    # ==========================================================

    settings = get_experiment_configs(
        experiments_root,
        var_fc=variables,
        region_name=regions,

        # net_name="ConvNeXtTransformerUNet",
        net_name="SmaAt_UNet",

        # target_mode="analysis",

        # input_realization_avg=False,

        # loss_name="GeoMaskedMSELoss",

        # seasonal_encoding=True,
        # ensemble_encoding=True,

        # separate_training_by_init_period=None,
        # separate_training_by_init_period=ClimPeriod.MONTH,

        # extra_suffix_folder="",
        # extra_suffix_folder="NOAA_copy",
    )

    print(
        f"Found {len(settings)} matching experiment(s)."
    )

    n = 0

    for s in settings:

        # ======================================================
        # Time range / inference path
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

            inference_start, inference_end = (
                inference_period
            )

            mlfc_path = (
                s.exp_dir
                / "inference"
                / f"{inference_start}_{inference_end}"
                / "test_corrected.zarr"
            )

        clim_time_range = (
            s.train_start,
            s.test_end,
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

        print(
            f"Generate field maps for "
            f"{(s.var_an, s.var_fc)} "
            f"in {s.region_name} "
            f"(lon={valid_lon_range}, "
            f"lat={valid_lat_range})"
        )

        # ======================================================
        # Load datasets
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

        # ======================================================
        # Lead-time selection
        # ======================================================

        leadtime_dim = (
            fc.earthml.guessed_dims.leadtime
        )

        time_dim = (
            fc.earthml.guessed_dims.time
        )

        if leadtime_dim is None:
            raise ValueError(
                "Could not determine forecast "
                "lead-time dimension."
            )

        if time_dim is None:
            raise ValueError(
                "Could not determine forecast time dimension."
            )

        fc = fc.sel(
            {leadtime_dim: s.leadtimes}
        )

        an = an.sel(
            {leadtime_dim: s.leadtimes}
        )

        fc_clim = fc_clim.sel(
            {leadtime_dim: s.leadtimes}
        )

        an_clim = an_clim.sel(
            {leadtime_dim: s.leadtimes}
        )

        if mlfc is not None:
            mlfc = mlfc.sel(
                {leadtime_dim: s.leadtimes}
            )

        if mlfc_clim is not None:
            mlfc_clim[s.var_fc] = _convert_kelvin_to_celsius(
                mlfc_clim[s.var_fc],
                var=s.var_fc,
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

        fc_clim[s.var_fc] = _convert_kelvin_to_celsius(
            fc_clim[s.var_fc],
            var=s.var_fc,
        )

        an_clim[s.var_an] = _convert_kelvin_to_celsius(
            an_clim[s.var_an],
            var=s.var_an,
        )

        if mlfc is not None:
            mlfc[s.var_fc] = _convert_kelvin_to_celsius(
                mlfc[s.var_fc],
                var=s.var_fc,
            )

        # ======================================================
        # Build clim-fc
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
                time_dim,
                clim_period,
            )
            - fc_clim_da
        )

        an_anom_da = (
            groupby_period(
                an[s.var_an],
                time_dim,
                clim_period,
            )
            - an_clim_da
        )

        clim_fc_da = (
            groupby_period(
                fc_anom_da,
                time_dim,
                clim_period,
            )
            + an_clim_da
        )

        if mlfc is not None and mlfc_clim is not None:
            mlfc_clim_da = stack_hour_clim(
                mlfc_clim[s.var_fc],
                clim_period,
            )

            mlfc_anom_da = (
                groupby_period(
                    mlfc[s.var_fc],
                    time_dim,
                    clim_period,
                )
                - mlfc_clim_da
            )
        else:
            mlfc_anom_da = None

        # ======================================================
        # Dataset collections
        # ======================================================

        model_fields: dict[
            FieldModel,
            xr.DataArray | None,
        ] = {
            "an": an[s.var_an],
            "fc": fc[s.var_fc],
            "clim-fc": clim_fc_da,
            "mlfc": (
                mlfc[s.var_fc]
                if mlfc is not None
                else None
            ),
        }

        anomaly_fields: dict[
            FieldModel,
            xr.DataArray | None,
        ] = {
            "an": an_anom_da,
            "fc": fc_anom_da,
            # By construction, clim-fc = fc anomaly + analysis climatology.
            "clim-fc": fc_anom_da,
            "mlfc": mlfc_anom_da,
        }

        # ======================================================
        # Select times
        # ======================================================

        available_times = (
            fc[time_dim].values
        )

        if wanted_times is None:
            selected_times = available_times

        else:
            requested = pd.to_datetime(
                wanted_times
            ).to_numpy()

            selected_times = [
                t
                for t in available_times
                if np.any(requested == t)
            ]

            missing_times = [
                t
                for t in requested
                if not np.any(
                    available_times == t
                )
            ]

            if missing_times:
                print(
                    "WARNING: requested times "
                    "not available: "
                    f"{missing_times}"
                )

        # ======================================================
        # Select leads
        # ======================================================

        available_leads = (
            fc[leadtime_dim].values
        )

        if wanted_leadtimes is None:
            selected_leads = (
                available_leads
            )

        else:
            selected_leads = [
                lead
                for lead in wanted_leadtimes
                if lead in available_leads
            ]

        # ======================================================
        # Plot
        # ======================================================

        model_names = {
            "an": "Analysis",
            "fc": "Forecast",
            "clim-fc": "Climatology-corrected forecast",
            "mlfc": "ML-corrected forecast",
        }

        difference_names = {
            "fc-an": "Forecast - Analysis",
            "clim-fc-an": "Climatology-corrected forecast - Analysis",
            "mlfc-an": "ML-corrected forecast - Analysis",
            "clim-fc-fc": "Climatology correction",
            "mlfc-fc": "ML correction",
            "mlfc-clim-fc": "ML-corrected - Climatology-corrected forecast",
            "mlfc-fc-abs-error": "ML-corrected - Forecast absolute error",
        }

        anomaly_difference_names = {
            "fc-an": "Forecast anomaly - Analysis anomaly",
            "mlfc-an": "ML-corrected anomaly - Analysis anomaly",
            "mlfc-fc": "ML-corrected anomaly - Forecast anomaly",
            "mlfc-fc-abs-anomaly-error": "ML-corrected - Forecast absolute anomaly error",
        }

        if leadtime_units == LeadtimeUnit.HOURS:
            lead_unit_label = "h"
        elif leadtime_units == LeadtimeUnit.MONTHS:
            lead_unit_label = "M"
        else:
            lead_unit_label = ""

        for time_value in selected_times:

            for lead_value in selected_leads:

                # Labels
                init_time = pd.Timestamp(time_value)

                if leadtime_units == LeadtimeUnit.HOURS:
                    valid_time = init_time + pd.to_timedelta(lead_value, unit="h")

                elif leadtime_units == LeadtimeUnit.MONTHS:
                    valid_time = init_time + pd.DateOffset(months=int(lead_value))

                else:
                    raise ValueError(f"Unsupported leadtime unit: {leadtime_units}")

                init_title = init_time.strftime(plot_title_strftime)
                valid_title = valid_time.strftime(plot_title_strftime)

                time_label = pd.Timestamp(time_value).strftime(
                    "%Y%m%dT%H%M"
                )
                time_title = pd.Timestamp(time_value).strftime(
                    plot_title_strftime
                )
                lead_label = safe_label(lead_value)

                # Select the raw fields and anomaly fields once.
                selected: dict[str, dict[FieldModel, xr.DataArray]] = {
                    "fields": {},
                    "anomalies": {},
                }

                for kind, source in (
                    ("fields", model_fields),
                    ("anomalies", anomaly_fields),
                ):
                    for model, da in source.items():
                        if da is None:
                            continue

                        selected[kind][model] = _select_field(
                            da,
                            time_value=time_value,
                            lead_value=lead_value,
                            realization_mode=(
                                "mean"
                                if model == "an"
                                else realization_mode
                            ),
                            realization=realization,
                        )

                # --------------------------------------------------
                # Direct field/anomaly maps
                # --------------------------------------------------

                raw_fields = {
                    model: selected["fields"][model]
                    for model in plot_models
                    if model in selected["fields"]
                }

                plot_vmin = vmin
                plot_vmax = vmax

                if raw_fields and common_scale:
                    auto_vmin, auto_vmax = _get_common_limits(
                        raw_fields,
                        quantiles=robust_quantiles,
                    )

                    if plot_vmin is None:
                        plot_vmin = auto_vmin
                    if plot_vmax is None:
                        plot_vmax = auto_vmax

                selected_anomalies = {
                    model: selected["anomalies"][model]
                    for model in plot_anomaly_models
                    if model in selected["anomalies"]
                }

                if anomaly_vmax is not None:
                    anom_vmax = float(anomaly_vmax)
                elif selected_anomalies and common_anomaly_scale:
                    anom_vmax = _get_symmetric_limit(
                        selected_anomalies,
                        quantile=anomaly_quantile,
                    )
                else:
                    anom_vmax = None

                collections = (
                    (
                        "absolute",
                        raw_fields,
                        cmap,
                        raw_centered, # centered
                        plot_vmin,
                        plot_vmax,
                        raw_levels,
                        model_names,
                    ),
                    (
                        "anomalies",
                        selected_anomalies,
                        anomaly_cmap,
                        True, # centered
                        -anom_vmax if anom_vmax is not None else None,
                        anom_vmax,
                        anomaly_levels,
                        {
                            model: f"{name} anomaly"
                            for model, name in model_names.items()
                        },
                    ),
                )

                for (
                    kind,
                    fields,
                    map_cmap,
                    centered,
                    map_vmin,
                    map_vmax,
                    map_levels,
                    names,
                ) in collections:
                    for model, field in fields.items():
                        common_path = (
                            Path("fields")
                            / kind
                            / model
                            / s.var_fc
                            / (
                                f"time_{safe_label(valid_time_range)}"
                                f"_lat_{safe_label(valid_lat_range)}"
                                f"_lon_{safe_label(valid_lon_range)}"
                            )
                        )

                        suffix = "_anom" if kind == "anomalies" else ""
                        filename = (
                            f"{s.var_fc}_{model}{suffix}"
                            f"_init_{time_label}"
                            f"_lead_{lead_label}.png"
                        )
                        out_file = s.plot_dir / common_path / filename

                        if out_file.exists() and not regenerate_plots:
                            continue

                        title = (
                            f"{names[model]} · start {init_title} · "
                            f"valid {valid_title} · lead {lead_value}{lead_unit_label}"
                            if plot_title
                            else None
                        )
                        print(f"Saving {kind} map {out_file}")

                        plot_field_map(
                            field,
                            var=s.var_fc,
                            title=title,
                            out_file=out_file,
                            cmap=map_cmap,
                            centered=centered,
                            vmin=map_vmin,
                            vmax=map_vmax,
                            levels=map_levels,
                            plot_type=plot_type,
                            figsize=plot_figsize,
                            rectangles=rectangles,
                            plot_title=plot_title,
                            plot_labels=plot_labels,
                            title_size=title_size,
                            label_size=label_size,
                            tick_size=tick_size,
                            dpi=dpi,
                        )
                        n += 1

                # --------------------------------------------------
                # Raw and anomaly differences
                # --------------------------------------------------

                difference_sets = (
                    (
                        "differences",
                        selected["fields"],
                        plot_differences,
                        difference_names,
                    ),
                    (
                        "anomaly_differences",
                        selected["anomalies"],
                        plot_anomaly_differences,
                        anomaly_difference_names,
                    ),
                )

                for kind, fields, requested, names in difference_sets:
                    differences = _build_field_differences(
                        fields,
                        requested,
                    )

                    if not differences:
                        continue

                    if kind == "anomaly_differences":
                        manual_vmax = anomaly_difference_vmax
                        diff_levels = anomaly_difference_levels
                        use_common_scale = common_anomaly_difference_scale
                        quantile = anomaly_difference_quantile
                    else:
                        manual_vmax = difference_vmax
                        diff_levels = difference_levels
                        use_common_scale = common_difference_scale
                        quantile = difference_quantile

                    if manual_vmax is not None:
                        diff_vmax = float(manual_vmax)
                    elif use_common_scale:
                        diff_vmax = _get_symmetric_limit(
                            differences,
                            quantile=quantile,
                        )
                    else:
                        diff_vmax = None

                    for diff_name, field in differences.items():
                        common_path = (
                            Path("fields")
                            / kind
                            / diff_name
                            / s.var_fc
                            / (
                                f"time_{safe_label(valid_time_range)}"
                                f"_lat_{safe_label(valid_lat_range)}"
                                f"_lon_{safe_label(valid_lon_range)}"
                            )
                        )

                        suffix = (
                            "_anom_diff"
                            if kind == "anomaly_differences"
                            else ""
                        )

                        filename = (
                            f"{s.var_fc}_{diff_name}{suffix}"
                            f"_init_{time_label}"
                            f"_lead_{lead_label}.png"
                        )

                        out_file = s.plot_dir / common_path / filename

                        if out_file.exists() and not regenerate_plots:
                            continue

                        title = (
                            f"{names[diff_name]} · start {init_title} · "
                            f"valid {valid_title} · lead {lead_value}{lead_unit_label}"
                            if plot_title
                            else None
                        )

                        print(f"Saving {kind} map {out_file}")

                        plot_field_map(
                            field,
                            var=s.var_fc,
                            title=title,
                            out_file=out_file,
                            cmap=difference_cmap,
                            centered=True,
                            vmin=-diff_vmax if diff_vmax is not None else None,
                            vmax=diff_vmax,
                            levels=diff_levels,
                            plot_type=plot_type,
                            figsize=plot_figsize,
                            rectangles=rectangles,
                            plot_title=plot_title,
                            plot_labels=plot_labels,
                            title_size=title_size,
                            label_size=label_size,
                            tick_size=tick_size,
                            dpi=dpi,
                        )

                        n += 1

    print(
        f"Done. Saved {n} plots."
    )


def _select_field(
    da: xr.DataArray,
    *,
    time_value,
    lead_value,
    realization_mode: Literal["mean", "member"] = "mean",
    realization: int | str | None = None,
) -> xr.DataArray:
    """
    Select one 2-D spatial field.

    Expected dimensions before selection are approximately:

        time x leadtime x [realization] x latitude x longitude

    Analysis usually has no realization dimension.
    """

    time_dim = da.earthml.guessed_dims.time
    leadtime_dim = da.earthml.guessed_dims.leadtime
    realization_dim = da.earthml.guessed_dims.realization

    if time_dim is None:
        raise ValueError(
            f"Could not determine time dimension for {da.name!r}. "
            f"Available dimensions: {da.dims}"
        )

    if leadtime_dim is None:
        raise ValueError(
            f"Could not determine lead-time dimension for {da.name!r}. "
            f"Available dimensions: {da.dims}"
        )

    da = da.sel(
        {
            time_dim: time_value,
            leadtime_dim: lead_value,
        }
    )

    if (
        realization_dim is not None
        and realization_dim in da.dims
    ):
        if realization_mode == "mean":
            da = da.mean(
                realization_dim,
                skipna=True,
            )

        elif realization_mode == "member":
            if realization is None:
                raise ValueError(
                    "realization must be specified when "
                    "realization_mode='member'."
                )

            try:
                da = da.sel(
                    {realization_dim: realization}
                )

            except (KeyError, ValueError):
                if not isinstance(realization, int):
                    raise

                da = da.isel(
                    {realization_dim: realization}
                )

        else:
            raise ValueError(
                f"Unsupported "
                f"realization_mode={realization_mode!r}."
            )

    return da.squeeze(drop=True)


def _get_common_limits(
    fields: dict[str, xr.DataArray],
    *,
    quantiles: tuple[float, float] | None = (0.01, 0.99),
) -> tuple[float, float]:
    """Return common non-symmetric limits across fields."""
    values = [
        finite
        for da in fields.values()
        if (finite := np.asarray(da.values)[np.isfinite(da.values)]).size
    ]

    if not values:
        raise ValueError("No finite values available to determine plot limits.")

    values = np.concatenate(values)

    if quantiles is None:
        return float(values.min()), float(values.max())

    return (
        float(np.quantile(values, quantiles[0])),
        float(np.quantile(values, quantiles[1])),
    )


def _get_symmetric_limit(
    fields: dict[str, xr.DataArray],
    *,
    quantile: float | None = 0.99,
) -> float:
    """Return a positive symmetric limit around zero."""
    values = [
        finite
        for da in fields.values()
        if (finite := np.abs(
            np.asarray(da.values)[np.isfinite(da.values)]
        )).size
    ]

    if not values:
        raise ValueError("No finite values available to determine plot limits.")

    values = np.concatenate(values)

    if quantile is None:
        vmax = float(values.max())
    else:
        vmax = float(np.quantile(values, quantile))

    # Avoid invalid vmin == vmax == 0.
    return vmax if vmax > 0 else 1.0


def _build_field_differences(
    fields: dict[FieldModel, xr.DataArray],
    requested: tuple[FieldDifference, ...],
) -> dict[FieldDifference, xr.DataArray]:

    pairs: dict[
        FieldDifference,
        tuple[FieldModel, FieldModel],
    ] = {
        "fc-an": ("fc", "an"),
        "clim-fc-an": ("clim-fc", "an"),
        "mlfc-an": ("mlfc", "an"),
        "clim-fc-fc": ("clim-fc", "fc"),
        "mlfc-fc": ("mlfc", "fc"),
        "mlfc-clim-fc": ("mlfc", "clim-fc"),
    }

    differences: dict[
        FieldDifference,
        xr.DataArray,
    ] = {}

    for name in requested:
        if name in {
            "mlfc-fc-abs-error",
            "mlfc-fc-abs-anomaly-error",
        }:
            required = ("mlfc", "fc", "an")

            if any(field not in fields for field in required):
                continue

            mlfc, fc, an = xr.align(
                fields["mlfc"],
                fields["fc"],
                fields["an"],
                join="exact",
            )

            differences[name] = (
                abs(mlfc - an)
                - abs(fc - an)
            )

            continue

        lhs_name, rhs_name = pairs[name]

        if (
            lhs_name not in fields
            or rhs_name not in fields
        ):
            continue

        lhs, rhs = xr.align(
            fields[lhs_name],
            fields[rhs_name],
            join="exact",
        )

        differences[name] = lhs - rhs

    return differences


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
