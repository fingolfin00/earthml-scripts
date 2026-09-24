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
)
from earthml.metrics import (
    LeadtimeAgg,
    MetricKind,
    get_metrics,
    calculate_save_and_subset_climatologies,
)
from earthml.plots import (
    safe_label,
    lead_label,
    plot_rank_histogram,
)


def main() -> None:
    # ==========================================================
    # Paths
    # ==========================================================

    # exp_name = "weather_atmo"
    exp_name = "weather_atmo_ablation_fixed_val"
    # exp_name = "weather_atmo_short_zero_vs_replicate_padding"

    experiments_root = Path(
        # "/Users/jacopodallaglio/ML/training/seasonal/experiments"
        f"/work/cmcc/jd19424/ML/MLBC/experiments/{exp_name}"
    )

    # ==========================================================
    # Plot settings
    # ==========================================================

    metric_kind: MetricKind = "scalar"

    plot_title = True
    plot_labels = True

    title_size = None
    label_size = 14
    tick_size = 12
    dpi = 200

    plot_models = (
        "fc",
        "mlfc",
    )

    regenerate_plots = True

    # ==========================================================
    # Data processing
    # ==========================================================

    # interpolate = True  # seasonal
    interpolate = False  # weather

    build_analysis = True
    recalculate_climatology = False

    # ==========================================================
    # Climatology
    # ==========================================================

    clim_period: ClimPeriod = ClimPeriod.MONTH
    clim_rolling_window = None

    # clim_period: ClimPeriod = ClimPeriod.DAYOFYEAR_HOUR
    # clim_rolling_window = 31

    # Period grouping reference:
    #   "init"  -> group by forecast initialization time
    #   "valid" -> group by forecast valid time
    period_reference: Literal["init", "valid"] = "init"

    # ==========================================================
    # Time selection
    # ==========================================================

    time_range = None
    # time_range = ("2018-01-01", "2022-12-31")

    inference_period = None
    # inference_period = ("2025-01-01", "2025-10-10")

    periods_requested = [
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

    # leadtime_units = LeadtimeUnit.MONTHS  # seasonal
    leadtime_units = LeadtimeUnit.HOURS  # weather

    leadtime_agg_mode: LeadtimeAgg = "single"
    # "single"
    # "aggregated"
    # "seasonal_window"

    # ==========================================================
    # Metrics
    # ==========================================================

    metrics = [
        # ======================================================
        # Ensemble rank histograms - absolute fields
        # ======================================================
        "rank_histogram",

        # ======================================================
        # Ensemble rank histograms - anomaly fields
        # ======================================================
        # "rank_histogram_anom",
    ]

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
        "ConUS",
        # "Europe",
        # "Pacific",
        # "World",
        # None,
    ]

    # ==========================================================
    # Spatial subset
    # ==========================================================

    # ConUS
    lat_range = (50, 25)
    lon_range = (-130, -60)

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
    # Experiment selection
    # ==========================================================

    settings = get_experiment_configs(
        experiments_root,
        var_fc=variables,
        region_name=regions,

        # net_name="ConvNeXtTransformerUNet",
        net_name="SmaAt_UNet",

        target_mode="analysis",

        # input_realization_avg=False,

        loss_name="GeoMaskedMSELoss",

        train_start="2019-10-14",
        train_subsamples=1000,

        # seasonal_encoding=True,
        # ensemble_encoding=True,

        # separate_training_by_init_period=None,
        # separate_training_by_init_period=ClimPeriod.MONTH,
    )

    print(f"Found {len(settings)} matching experiment(s).")

    n = 0

    for s in settings:
        # ======================================================
        # Time range
        # ======================================================

        if inference_period is None:
            valid_time_range = (
                (s.test_start, s.test_end)
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
            s.val_end,
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
        # Dimensions / grouping
        # ======================================================

        leadtime_agg_coord = (
            "leadtime"
            if leadtime_agg_mode == "single"
            else "leadtime_seasonal"
        )

        period_dim = (
            f"{period_reference}_{clim_period.value}"
        )

        if (
            period_reference == "valid"
            and leadtime_agg_mode == "aggregated"
        ):
            raise ValueError(
                "period_reference='valid' is not supported with "
                "leadtime_agg_mode='aggregated' because an aggregated "
                "forecast has no unique valid time. Use 'single' or "
                "'seasonal_window'."
            )

        print(
            f"Generate {leadtime_agg_mode} histograms "
            f"grouped by {period_dim} for "
            f"{s.var_an, s.var_fc} in {s.region_name} "
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

        # ======================================================
        # Climatologies
        # ======================================================

        (
            fc_clim,
            an_clim,
            mlfc_clim,
        ) = calculate_save_and_subset_climatologies(
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

        if mlfc_clim is not None:
            mlfc_clim = mlfc_clim.assign_coords(
                leadtime=s.leadtimes
            )

        # ======================================================
        # Lead-time selection
        # ======================================================

        leadtime_dim = fc.earthml.guessed_dims.leadtime

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

        # ======================================================
        # Models
        # ======================================================

        model_datasets = {
            "fc": fc,
            "mlfc": mlfc,
        }

        model_climatologies = {
            "fc": fc_clim,
            "mlfc": mlfc_clim,
        }

        # ======================================================
        # Calculate histograms
        # ======================================================

        metric_histograms_by_model: dict[
            str,
            xr.Dataset,
        ] = {}

        for model in plot_models:
            ds = model_datasets[model]
            ds_clim = model_climatologies[model]

            if ds is None or ds_clim is None:
                continue

            print(
                f"Get {model} rank histograms"
            )

            metric_histograms_by_model[model] = get_metrics(
                an=an,
                fc=ds,
                var=s.var_fc,
                metric_kind=metric_kind,
                leadtime_agg=leadtime_agg_mode,

                # Rank histograms require ensemble members.
                realization_agg=False,

                an_clim=an_clim,
                fc_clim=ds_clim,

                metrics=metrics,

                leadtime_windows=(
                    s.seasonal_leadtime_windows
                ),

                leadtime_agg_coord=(
                    leadtime_agg_coord
                ),

                clim_period=clim_period,

                period_reference=period_reference,
                period_dim=period_dim,
                periods_requested=periods_requested,

                leadtime_unit=leadtime_units,

                align=False,
                fair_correction=False,
            )

        # ======================================================
        # Available models
        # ======================================================

        available_models = [
            model
            for model in plot_models
            if model in metric_histograms_by_model
        ]

        if not available_models:
            continue

        # ======================================================
        # Available metrics
        # ======================================================

        available_metrics = [
            metric
            for metric in metrics
            if all(
                metric in metric_histograms_by_model[model]
                for model in available_models
            )
        ]

        if not available_metrics:
            continue

        print(
            f"Plotting histogram metrics "
            f"{available_metrics} for models "
            f"{available_models}"
        )

        # ======================================================
        # Periods
        # ======================================================

        reference_ds = (
            metric_histograms_by_model[
                available_models[0]
            ]
        )

        available_periods = [
            str(value)
            for value in reference_ds[period_dim].values
            if str(value) in periods_requested
        ]

        # ======================================================
        # Plot
        # ======================================================

        for metric in available_metrics:
            for period_value in available_periods:

                reference_da = (
                    metric_histograms_by_model[
                        available_models[0]
                    ][metric]
                )

                for lead_value in reference_da[
                    leadtime_agg_coord
                ].values:

                    label = safe_label(
                        lead_label(
                            reference_da,
                            lead_value,
                            leadtime_agg_coord,
                        )
                    )

                    # --------------------------------------------------
                    # Select the exact lead time and period BEFORE
                    # plot_rank_histogram().
                    #
                    # plot_rank_histogram reduces every remaining
                    # dimension except "rank".
                    # --------------------------------------------------

                    histogram_das = []

                    for model in available_models:
                        da = (
                            metric_histograms_by_model[
                                model
                            ][metric]
                        )

                        selectors = {}

                        if period_dim in da.dims:
                            selectors[
                                period_dim
                            ] = period_value

                        if leadtime_agg_coord in da.dims:
                            selectors[
                                leadtime_agg_coord
                            ] = lead_value

                        if selectors:
                            da = da.sel(
                                selectors
                            )

                        da = da.squeeze(
                            drop=True
                        )

                        histogram_das.append(
                            da
                        )

                    # --------------------------------------------------
                    # Output
                    # --------------------------------------------------

                    common_path = (
                        Path("histograms")
                        / safe_label(period_dim)
                        / safe_label(period_value)
                        / (
                            f"time_"
                            f"{safe_label(valid_time_range)}"
                            f"_lat_"
                            f"{safe_label(valid_lat_range)}"
                            f"_lon_"
                            f"{safe_label(valid_lon_range)}"
                        )
                        / metric
                        / leadtime_agg_mode
                    )

                    filename = (
                        f"{s.var_fc}_{metric}_"
                        f"{'-'.join(available_models)}"
                        f"_lead_{label}.png"
                    )

                    out_file = (
                        s.plot_dir
                        / common_path
                        / filename
                    )

                    if (
                        out_file.exists()
                        and not regenerate_plots
                    ):
                        continue

                    print(
                        f"Saving histogram "
                        f"{out_file}"
                    )

                    plot_rank_histogram(
                        histogram_das,
                        var=s.var_fc,
                        metric=metric,
                        models=available_models,
                        start_period=period_value,
                        out_file=out_file,
                        time_range=valid_time_range,
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


if __name__ == "__main__":
    main()
