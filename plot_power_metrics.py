from typing import Literal
from pathlib import Path

import numpy as np
import xarray as xr

import warnings
from dask.array import PerformanceWarning
from dask.diagnostics.progress import ProgressBar

warnings.simplefilter("ignore", FutureWarning)
warnings.filterwarnings(
    "ignore",
    category=PerformanceWarning,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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
    is_power,
    get_metrics,
    calculate_save_and_subset_climatologies,
)
from earthml.plots import (
    safe_label,
    lead_label,
)


SpectrumType = Literal[
    "spatial",
    "temporal",
    "zonal",
    "meridional",
]


def main() -> None:
    # ==========================================================
    # Paths
    # ==========================================================

    # exp_name = "weather_atmo"
    exp_name = "weather_atmo_ablation_fixed_val"
    # exp_name = "weather_atmo_short_zero_vs_replicate_padding"

    experiments_root = Path(
        "/Users/jacopodallaglio/ML/training/seasonal/experiments"
        # f"/work/cmcc/jd19424/ML/MLBC/experiments/{exp_name}"
    )

    # ==========================================================
    # Spectrum settings
    # ==========================================================

    spectrum_type: SpectrumType = "spatial"

    metric_kind = get_power_metric_kind(
        spectrum_type
    )

    # Ensemble mean before calculating spectra.
    realization_agg = True

    dpi = 200
    regenerate_plots = True

    plot_models = (
        "fc",
        "mlfc",
    )

    # ==========================================================
    # Data processing
    # ==========================================================

    interpolate = False
    build_analysis = True
    recalculate_climatology = False

    # ==========================================================
    # Climatology
    # ==========================================================

    clim_period: ClimPeriod = ClimPeriod.MONTH
    clim_rolling_window = None

    period_reference: Literal[
        "init",
        "valid",
    ] = "init"

    # ==========================================================
    # Time selection
    # ==========================================================

    time_range = None
    inference_period = None

    periods_requested = [
        "all",
    ]

    # ==========================================================
    # Lead-time aggregation
    # ==========================================================

    leadtime_units = LeadtimeUnit.HOURS

    leadtime_agg_mode: LeadtimeAgg = "single"

    # ==========================================================
    # Power metrics
    # ==========================================================

    metrics = [
        # Raw field spectra
        "fc_power_spectrum",
        "an_power_spectrum",

        # Spatial isotropic spectra.
        # Only generated when spectrum_type == "spatial".
        "fc_isotropic_power_spectrum",
        "an_isotropic_power_spectrum",

        # FC / analysis power ratio
        "power_spectrum_ratio",

        # Anomaly spectra
        # "fc_anom_power_spectrum",
        # "an_anom_power_spectrum",
        # "fc_anom_isotropic_power_spectrum",
        # "an_anom_isotropic_power_spectrum",
        # "power_spectrum_ratio_anom",
    ]

    power_metrics = [
        metric
        for metric in metrics
        if is_power(metric)
    ]

    if not power_metrics:
        raise ValueError(
            "No power-spectrum metrics selected."
        )

    # Isotropic spectra exist only for spatial transforms.
    if spectrum_type != "spatial":
        power_metrics = [
            metric
            for metric in power_metrics
            if "isotropic" not in metric
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
        target_mode="analysis",
        loss_name="GeoMaskedMSELoss",
        train_start="2019-10-14",
        # train_subsamples=1000,
    )

    print(
        f"Found {len(settings)} matching experiment(s)."
    )

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
            s.train_end,
            # s.val_end,
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
        # Coordinates
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
                "leadtime_agg_mode='aggregated'."
            )

        print(
            f"Generate {spectrum_type} spectra for "
            f"{s.var_an, s.var_fc} in {s.region_name} "
            f"(lon={valid_lon_range}, "
            f"lat={valid_lat_range})"
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
            mlfc_clim = (
                mlfc_clim.assign_coords(
                    leadtime=s.leadtimes
                )
            )

        leadtime_dim = (
            fc.earthml.guessed_dims.leadtime
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
        # Calculate power metrics
        # ======================================================

        spectra_by_model: dict[
            str,
            xr.Dataset,
        ] = {}

        for model in plot_models:
            ds = model_datasets[model]
            ds_clim = model_climatologies[model]

            if ds is None or ds_clim is None:
                continue

            print(
                f"Get {model} {spectrum_type} spectra"
            )

            spectra = get_metrics(
                an=an,
                fc=ds,
                var=s.var_fc,
                metric_kind=metric_kind,
                leadtime_agg=leadtime_agg_mode,
                realization_agg=realization_agg,
                an_clim=an_clim,
                fc_clim=ds_clim,
                metrics=power_metrics,
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

            # --------------------------------------------------
            # FFT does not necessarily consume every dimension.
            #
            # Example spatial spectrum:
            #
            #     input:
            #       time, lat, lon
            #
            #     FFT over lat/lon:
            #       time, freq_lat, freq_lon
            #
            # Average the remaining samples so the plotting
            # representation is purely spectral.
            # --------------------------------------------------

            reduced_vars = {}

            for metric, da in spectra.data_vars.items():
                reduced_vars[metric] = (
                    reduce_non_spectral_dims(
                        da,
                        leadtime_dim=leadtime_agg_coord,
                        period_dim=period_dim,
                    )
                )

            spectra_by_model[model] = xr.Dataset(
                reduced_vars
            )

        # ======================================================
        # Available models / metrics
        # ======================================================

        available_models = [
            model
            for model in plot_models
            if model in spectra_by_model
        ]

        if not available_models:
            continue

        available_metrics = [
            metric
            for metric in power_metrics
            if all(
                metric in spectra_by_model[model]
                for model in available_models
            )
        ]

        if not available_metrics:
            continue

        reference_ds = spectra_by_model[
            available_models[0]
        ]

        available_periods = [
            str(value)
            for value in reference_ds[
                period_dim
            ].values
            if str(value) in periods_requested
        ]

        # ======================================================
        # Plot
        # ======================================================

        for metric in available_metrics:
            reference_da = (
                reference_ds[metric]
            )

            for period_value in available_periods:
                for lead_value in reference_da[
                    leadtime_agg_coord
                ].values:

                    label_lt = safe_label(
                        lead_label(
                            reference_da,
                            lead_value,
                            leadtime_agg_coord,
                        )
                    )

                    spectra_to_plot = []

                    for model in available_models:
                        da = spectra_by_model[
                            model
                        ][metric]

                        da = da.sel(
                            {
                                period_dim:
                                    period_value,
                                leadtime_agg_coord:
                                    lead_value,
                            }
                        ).squeeze(
                            drop=True
                        )

                        spectra_to_plot.append(
                            da
                        )

                    common_path = (
                        Path("spectra")
                        / spectrum_type
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

                    # ------------------------------------------
                    # 1-D spectrum
                    # ------------------------------------------

                    n_freq = len(
                        frequency_dims(
                            spectra_to_plot[0]
                        )
                    )

                    if n_freq == 1:
                        filename = (
                            f"{s.var_fc}_{metric}_"
                            f"{'-'.join(available_models)}"
                            f"_lead_{label_lt}.png"
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
                            f"Saving 1-D spectrum "
                            f"{out_file}"
                        )

                        ratio = (
                            metric
                            in {
                                "power_spectrum_ratio",
                                "power_spectrum_ratio_anom",
                            }
                        )

                        plot_spectrum_1d(
                            spectra_to_plot,
                            labels=list(
                                available_models
                            ),
                            metric=metric,
                            out_file=out_file,
                            title=(
                                f"{s.var_fc} · "
                                f"{metric} · "
                                f"{spectrum_type} · "
                                f"lead={label_lt}"
                            ),
                            ratio=ratio,
                            dpi=dpi,
                        )

                        n += 1

                    # ------------------------------------------
                    # Raw 2-D spatial PSD
                    # ------------------------------------------

                    elif n_freq == 2:
                        for model, da in zip(
                            available_models,
                            spectra_to_plot,
                            strict=True,
                        ):
                            filename = (
                                f"{s.var_fc}_{metric}_"
                                f"{model}"
                                f"_lead_{label_lt}.png"
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
                                f"Saving 2-D spectrum "
                                f"{out_file}"
                            )

                            plot_spectrum_2d(
                                da,
                                out_file=out_file,
                                title=(
                                    f"{s.var_fc} · "
                                    f"{metric} · "
                                    f"{model} · "
                                    f"lead={label_lt}"
                                ),
                                dpi=dpi,
                            )

                            n += 1

                    else:
                        raise ValueError(
                            f"Unsupported spectral output for "
                            f"{metric!r}: "
                            f"{spectra_to_plot[0].dims}. "
                            f"Frequency dimensions: "
                            f"{frequency_dims(spectra_to_plot[0])}"
                        )

    print(
        f"Done. Saved {n} plots."
    )


def get_power_metric_kind(
    spectrum_type: SpectrumType,
) -> MetricKind:
    """
    Select reduction dimensions through the existing MetricKind API.

    spatial:
        dims = (latitude, longitude)

    temporal:
        dims = (time,)

    zonal:
        dims = (longitude,)

    meridional:
        dims = (latitude,)
    """
    return {
        "spatial": "timeseries",
        "temporal": "maps",
        "zonal": "time_lat",
        "meridional": "time_lon",
    }[spectrum_type]


def frequency_dims(
    da: xr.DataArray,
) -> list[str]:
    return [
        dim
        for dim in da.dims
        if dim.startswith("freq_")
    ]


def reduce_non_spectral_dims(
    da: xr.DataArray,
    *,
    leadtime_dim: str,
    period_dim: str,
) -> xr.DataArray:
    """
    Average dimensions that remain after the FFT.

    Keep only:
        frequency dimension(s)
        leadtime
        requested climatological period
    """
    freq_dims = set(frequency_dims(da))

    keep_dims = freq_dims | {
        leadtime_dim,
        period_dim,
    }

    reduce_dims = [
        dim
        for dim in da.dims
        if dim not in keep_dims
    ]

    if reduce_dims:
        da = da.mean(
            reduce_dims,
            skipna=True,
        )

    return da


def plot_spectrum_1d(
    das: list[xr.DataArray],
    *,
    labels: list[str],
    metric: str,
    out_file: Path,
    title: str,
    ratio: bool = False,
    dpi: int = 200,
) -> None:
    if len(das) != len(labels):
        raise ValueError(
            "Select same number of spectra and labels."
        )

    fig, ax = plt.subplots(
        figsize=(9, 6),
    )

    for da, label in zip(
        das,
        labels,
        strict=True,
    ):
        freq_dims = frequency_dims(da)

        if len(freq_dims) != 1:
            raise ValueError(
                f"Expected one frequency dimension for 1-D spectrum, "
                f"got {freq_dims} from {da.dims}."
            )

        freq_dim = freq_dims[0]

        with ProgressBar():
            da = da.compute()

        x = np.asarray(
            da[freq_dim].values,
            dtype=float,
        )

        y = np.asarray(
            da.values,
            dtype=float,
        )

        # xrft spectra may contain zero frequency.
        valid = (
            np.isfinite(x)
            & np.isfinite(y)
            & (x > 0)
        )

        if not ratio:
            valid &= y > 0

        x = x[valid]
        y = y[valid]

        ax.plot(
            x,
            y,
            linewidth=1.5,
            label=label,
        )

    ax.set_xscale("log")

    if ratio:
        ax.axhline(
            1.0,
            linewidth=1.0,
            linestyle="--",
            alpha=0.7,
        )
        ax.set_ylabel("Power ratio")
    else:
        ax.set_yscale("log")
        ax.set_ylabel("Power spectral density")

    ax.set_xlabel("Frequency")
    ax.set_title(title)

    ax.grid(
        True,
        alpha=0.3,
    )
    ax.legend()

    out_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()

    fig.savefig(
        out_file,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_spectrum_2d(
    da: xr.DataArray,
    *,
    out_file: Path,
    title: str,
    dpi: int = 200,
) -> None:
    freq_dims = frequency_dims(da)

    if len(freq_dims) != 2:
        raise ValueError(
            f"Expected two frequency dimensions for 2-D spectrum, "
            f"got {freq_dims} from {da.dims}."
        )

    y_dim, x_dim = freq_dims

    with ProgressBar():
        da = da.compute()

    da = da.where(
        np.isfinite(da)
        & (da > 0)
    )

    positive = da.where(
        da > 0,
        drop=True,
    )

    if positive.size == 0:
        raise ValueError(
            "2-D power spectrum contains no positive finite values."
        )

    vmin = float(
        positive.min()
    )
    vmax = float(
        positive.max()
    )

    fig, ax = plt.subplots(
        figsize=(8, 7),
    )

    im = ax.pcolormesh(
        da[x_dim],
        da[y_dim],
        da,
        shading="auto",
        norm=LogNorm(
            vmin=vmin,
            vmax=vmax,
        ),
    )

    ax.set_xlabel(x_dim)
    ax.set_ylabel(y_dim)
    ax.set_title(title)

    cb = fig.colorbar(
        im,
        ax=ax,
    )

    cb.set_label(
        "Power spectral density"
    )

    out_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()

    fig.savefig(
        out_file,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)


if __name__ == "__main__":
    main()
