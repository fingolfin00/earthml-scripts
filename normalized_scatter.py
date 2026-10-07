from pathlib import Path
from typing import Literal, Sequence

from itertools import combinations

import numpy as np
import pandas as pd
import xarray as xr

from scipy.stats import spearmanr
import torch

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tqdm.auto import tqdm

from earthml import (
    Settings,
    ClimPeriod,
    LeadtimeUnit,
    MonthlyNormalize,
    Normalize,
    XarrayDataset,
    get_experiment_configs,
    get_and_subset_datasets,
)

from earthml.metrics import (
    ImprovementUnit,
    PeriodReference,
    get_metrics,
    build_metric_improvements,
)

from train import (
    convert_to_xarray,
    make_train_test_datasets_for_leadtime,
)

# ============================================================================
# Configuration
# ============================================================================

# EXP_NAME = "weather_atmo"
EXP_NAME = "weather_atmo_one_year_test"

EXPERIMENTS_ROOT = Path(
    f"/work/cmcc/jd19424/ML/MLBC/experiments/{EXP_NAME}"
)

OUTPUT_ROOT = Path(
    f"/work/cmcc/jd19424/ML/MLBC/plots/{EXP_NAME}"
)

INTERPOLATE_ANALYSIS = False # weather

PLOT_SPECTRAL_WAVENUMBER_CORRELATION = True

# ----------------------------------------------------------------------------
# Experiment selection
# ----------------------------------------------------------------------------

FILTERS = dict(
    var_fc="t2m",
    var_an="t2m",
    region_name="ConUS",
)

# None -> all experiment leadtimes
LEADTIMES: list[int] | None = [72]

# ----------------------------------------------------------------------------
# Diagnostic spatial region
# ----------------------------------------------------------------------------

# Optional box used consistently for normalized-input statistics, spectral
# diagnostics, FC/AN/MLFC diagnostic fields, and performance (e.g. RMSE
# improvement). None keeps the corresponding full experiment dimension
# Best/worst rankings therefore refer to performance inside this box
DIAGNOSTIC_LAT_RANGE: tuple[float, float] | None = None
DIAGNOSTIC_LON_RANGE: tuple[float, float] | None = None

# DIAGNOSTIC_LAT_RANGE = (30, 45)
# DIAGNOSTIC_LON_RANGE = (-108, -90)

# ----------------------------------------------------------------------------
# Time range selection
# ----------------------------------------------------------------------------

SPLIT_RANGE: Literal["train", "val", "test"] = "test"

# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

LEGEND_LOCATION = "upper right"

# Number of best/worst performance timesteps to annotate.
# Performance is currently available for the test split only.
# ANNOTATE_BEST_TIMESTEPS = 30
ANNOTATE_BEST_TIMESTEPS = 10
ANNOTATE_WORST_TIMESTEPS = 10

SELECTED_SAMPLES_TO_ANNOTATE: set[str] | None = {
    "B5",
    "W4",
}

ANNOTATION_FONTSIZE = 5
ANNOTATION_DATE_FORMAT = "%Y-%m-%d:%H:%M"

# ----------------------------------------------------------------------------
# Performance and coloring
# ----------------------------------------------------------------------------

PERFORMANCE_METRICS_NORMALIZED = False

PERFORMANCE_METRIC = "rmse"

PERFORMANCE_IMPROVEMENT_UNIT: ImprovementUnit = "Δ"
PERFORMANCE_REFERENCE_TIME_PERIOD: PeriodReference = "init"

# Positive improvement = corrected forecast is better.
#
# RdBu:
#   negative -> red
#   zero     -> white
#   positive -> blue
PERFORMANCE_CMAP = "RdBu"

# Robust symmetric color range:
# e.g. 0.98 ignores the outermost 2% when setting vmax.
# Values outside the range are clipped to the endpoint colors.
PERFORMANCE_COLOR_QUANTILE = 0.90

# Non-performance-colored scatter points use the same neutral appearance
# regardless of whether they belong to train, validation, or test.
SCATTER_COLOR = "0.45" # grey
SCATTER_SIZE = 18
SCATTER_ALPHA = 0.35
SCATTER_MARKER_SIZE = 5.0

# ----------------------------------------------------------------------------
# Spectral settings
# ----------------------------------------------------------------------------

SPECTRAL_BATCH_SIZE = 32
HIGH_FREQ_THRESHOLD = 0.5
# Number of annuli used for the wavenumber-resolved spectral diagnostic
SPECTRAL_N_BINS = 30

# ----------------------------------------------------------------------------
# Performance diagnostics
# ----------------------------------------------------------------------------

# Scalar per-timestep diagnostics to compare against performance.
# Single-field metrics are evaluated independently for every
# DIAGNOSTIC_FIELDS entry; pairwise metrics are evaluated for
# generated field combinations.
DIAGNOSTIC_METRICS: Sequence[Literal[
    # Single field
    "mean",
    "std",
    "rms_laplacian",
    "mean_abs_laplacian",
    "low_frequency_power_fraction",
    "high_frequency_power_fraction",
    "high_low_power_ratio",
    "spectral_centroid",

    # Pairwise
    "correlation",
    "cosine",
    "rms_ratio",
    "projection",
    "orthogonal_rms",
    "mean_product",
]] = (
    # "mean",
    # "std",
    # "spectral_centroid",

    "correlation",
    "cosine",
    "rms_ratio",
    "orthogonal_rms",
    "mean_product",
)

# Order is important for PAIRWISE_METRICS
DIAGNOSTIC_FIELDS: Sequence[Literal[
    "input",
    "target",
    "corrected",
    "fc_error",
    "mlfc_error",
    "ideal_correction",
    "ml_correction",
]] = (
    # "input",

    "ml_correction",
    "ideal_correction",
)

# ----------------------------------------------------------------------------
# Metric classes
# ----------------------------------------------------------------------------

SINGLE_FIELD_METRICS = {
    "mean",
    "std",
    "rms_laplacian",
    "mean_abs_laplacian",
    "low_frequency_power_fraction",
    "high_frequency_power_fraction",
    "high_low_power_ratio",
    "spectral_centroid",
}

PAIRWISE_METRICS = {
    "correlation",
    "cosine",
    "rms_ratio",
    "projection",
    "orthogonal_rms",
    "mean_product",
}

SPECTRAL_METRICS = {
    "low_frequency_power_fraction",
    "high_frequency_power_fraction",
    "high_low_power_ratio",
    "spectral_centroid",
}

# ----------------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------------

FIELD_LABELS = {
    "input": "FC",
    "target": "analysis",
    "corrected": "MLFC",
    "fc_error": "FC error",
    "mlfc_error": "MLFC error",
    "ideal_correction": "ideal correction",
    "ml_correction": "ML correction",
}

METRIC_LABELS = {
    "mean": "Spatial mean",
    "std": "Spatial standard deviation",
    "rms_laplacian": "RMS Laplacian",
    "mean_abs_laplacian": "Mean |Laplacian|",
    "correlation": "Spatial correlation",
    "cosine": "Cosine similarity",
    "rms_ratio": "RMS ratio",
    "projection": "Projection coefficient",
    "orthogonal_rms": "Orthogonal RMS",
    "mean_product": "Mean product",
    "low_frequency_power_fraction": "Low-frequency power fraction",
    "high_frequency_power_fraction": "High-frequency power fraction",
    "high_low_power_ratio": "High / low-frequency power ratio",
    "spectral_centroid": "Normalized spectral centroid",
}

# ============================================================================
# Build metrics for one leadtime
# ============================================================================

def get_leadtime_metrics(
    s: Settings,
    leadtime: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(
        "=" * 80
    )

    print(
        f"{s.output_name}: "
        f"leadtime={leadtime} "
        f"{s.leadtime_unit}"
    )
    print(
        "Diagnostic spatial region: "
        f"lat={DIAGNOSTIC_LAT_RANGE}, lon={DIAGNOSTIC_LON_RANGE}"
    )

    # ------------------------------------------------------------
    # Generate datasets
    # ------------------------------------------------------------

    datasets = make_train_test_datasets_for_leadtime(
        forecast_ds_path=(
            s.input_dir
            / f"{s.model_fc}_{s.var_fc}.zarr"
        ),
        analysis_ds_path=(
            s.input_dir
            / f"{s.model_an}_{s.var_an}.zarr"
        ),
        leadtime=leadtime,
        leadtime_unit=LeadtimeUnit(s.leadtime_unit),
        train_start=s.train_start,
        train_end=s.train_end,
        val_start=s.val_start,
        val_end=s.val_end,
        test_start=s.test_start,
        test_end=s.test_end,
        target_mode=s.target_mode,
        clim_period=s.clim_period,
        forecast_vars=[s.var_fc],
        analysis_vars=[s.var_an],
        region=s.region,
        dataset_kwargs=dataset_kwargs_from_settings(s),
        seasonal_encoding=(
            s.seasonal_encoding
            and s.channel_representation != "init_period"
        ),
        ensemble_encoding=s.ensemble_encoding,
        spatial_encoding=s.spatial_encoding,
        interpolate_analysis=INTERPOLATE_ANALYSIS,
        materialize=False,
    )

    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    test_dataset = datasets["test"]

    if not isinstance(train_dataset, XarrayDataset):
        raise NotImplementedError(
            "Train dataset must be an XarrayDataset"
        )

    if (
        val_dataset is not None
        and not isinstance(val_dataset, XarrayDataset)
    ):
        raise NotImplementedError(
            "Validation dataset must be an XarrayDataset or None"
        )

    if not isinstance(test_dataset, XarrayDataset):
        raise NotImplementedError(
            "Test dataset must be an XarrayDataset"
        )

    normalize_input = make_normalizer(
        s,
        train_dataset,
        dim="x",
    )

    normalize_target = make_normalizer(
        s,
        train_dataset,
        dim="y",
    )

    fc, an, mlfc = get_leadtime_xarray_datasets(
        s=s,
        time_range=SPLIT_RANGE,
        leadtime=leadtime,
    )

    if mlfc is None:
        raise ValueError("MLFC dataset is required to calculate metrics improvement")

    norm_fc, norm_an, norm_mlfc = get_normalized_datasets(
        s=s,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        normalize_input=normalize_input,
        normalize_target=normalize_target,
        fc=fc,
        mlfc=mlfc,
        time_range=SPLIT_RANGE,
    )

    # Subset
    fc = subset_diagnostic_dataset_region(fc, DIAGNOSTIC_LAT_RANGE, DIAGNOSTIC_LON_RANGE)
    an = subset_diagnostic_dataset_region(an, DIAGNOSTIC_LAT_RANGE, DIAGNOSTIC_LON_RANGE)
    mlfc = subset_diagnostic_dataset_region(mlfc, DIAGNOSTIC_LAT_RANGE, DIAGNOSTIC_LON_RANGE)

    norm_fc = subset_diagnostic_dataset_region(norm_fc, DIAGNOSTIC_LAT_RANGE, DIAGNOSTIC_LON_RANGE)
    norm_an = subset_diagnostic_dataset_region(norm_an, DIAGNOSTIC_LAT_RANGE, DIAGNOSTIC_LON_RANGE)
    norm_mlfc = subset_diagnostic_dataset_region(norm_mlfc, DIAGNOSTIC_LAT_RANGE, DIAGNOSTIC_LON_RANGE)

    # ------------------------------------------------------------
    # Metric performance
    # ------------------------------------------------------------

    print(f"Calculating {"normalized " if PERFORMANCE_METRICS_NORMALIZED else ""}{PERFORMANCE_METRIC} improvement...")

    # Select performance metric datasets
    if PERFORMANCE_METRICS_NORMALIZED:
        perf_metric_baseline = norm_fc
        perf_metric_target = norm_an
        perf_metric_corrected = norm_mlfc
    else:
        perf_metric_baseline = fc
        perf_metric_target = an
        perf_metric_corrected = mlfc

    metric_df = get_metric_improvement_timeseries(
        s=s,
        baseline=perf_metric_baseline,
        target=perf_metric_target,
        corrected=perf_metric_corrected,
        leadtime=leadtime,
        metric=PERFORMANCE_METRIC,
        improvement_unit=PERFORMANCE_IMPROVEMENT_UNIT,
        period_reference=PERFORMANCE_REFERENCE_TIME_PERIOD,
    )

    # ------------------------------------------------------------
    # Normalized spectral metrics
    # ------------------------------------------------------------

    spectral_scalar_df, spectral_df = get_spectral_metrics(
        s=s,
        baseline=norm_fc,
        target=norm_an,
        corrected=norm_mlfc,
        leadtime=leadtime,
    )

    if not spectral_scalar_df.empty:
        metric_df = metric_df.merge(
            spectral_scalar_df,
            on="time",
            how="left",
            validate="one_to_one",
        )

    if not spectral_df.empty:
        spectral_df = spectral_df.merge(
            metric_df[
                [
                    "time",
                    "metric_improvement",
                ]
            ],
            on="time",
            how="left",
            validate="many_to_one",
        )

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------

    print(f"Calculating normalized diagnostics...")

    forecast_diagnostics = tuple(
        metric
        for metric in DIAGNOSTIC_METRICS
        if metric not in SPECTRAL_METRICS
    )

    if forecast_diagnostics:
        print(
            "Calculating forecast/analysis diagnostics: "
            + ", ".join(forecast_diagnostics)
        )

        diagnostics_df = get_diagnostic_timeseries(
            fc=norm_fc,
            an=norm_an,
            mlfc=norm_mlfc,
            s=s,
        )

        metric_df = metric_df.merge(
            diagnostics_df,
            on="time",
            how="left",
            validate="one_to_one",
        )

    return metric_df, spectral_df

# ============================================================================
# Main
# ============================================================================

def main() -> None:
    settings = (
        get_experiment_configs(
            EXPERIMENTS_ROOT,
            **FILTERS,
        )
    )

    if not settings:
        raise RuntimeError(
            "No matching experiments found."
        )

    print(f"Found {len(settings)} matching experiment(s)")

    for s in settings:
        experiment_output_path = (
            OUTPUT_ROOT
            / s.output_name
            / "perf_vs_diag_scatter"
            / diagnostic_region_label()
        )

        leadtimes = (
            [
                int(x)
                for x in s.leadtimes
            ]
            if LEADTIMES is None
            else LEADTIMES
        )

        for leadtime in leadtimes:
            metrics_df, spectral_df = get_leadtime_metrics(
                s,
                leadtime,
            )

            leadtime_output_path = (
                experiment_output_path
                / f"leadtime_{leadtime}"
            )

            leadtime_output_path.mkdir(
                parents=True,
                exist_ok=True,
            )

            metrics_df.to_csv(
                leadtime_output_path
                / "metrics.csv",
                index=False,
            )

            if "metric_improvement" in metrics_df.columns:
                for diagnostic in diagnostic_names():
                    if diagnostic not in metrics_df.columns:
                        continue

                    plot_diagnostic_vs_performance(
                        metrics_df,
                        diagnostic=diagnostic,
                        output_path=(
                            leadtime_output_path
                            / f"{"normalized_" if PERFORMANCE_METRICS_NORMALIZED else ""}{PERFORMANCE_METRIC}_improvement_vs_{diagnostic}.png"
                        ),
                    )

            if (
                PLOT_SPECTRAL_WAVENUMBER_CORRELATION
                and not spectral_df.empty
            ):
                spectral_df.to_csv(
                    leadtime_output_path
                    / "radial_power_spectra.csv",
                    index=False,
                )

                for field_name in DIAGNOSTIC_FIELDS:
                    field_spectral_df = spectral_df.loc[
                        spectral_df["field"] == field_name
                    ].copy()

                    if field_spectral_df.empty:
                        continue

                    correlation_df = (
                        plot_spearman_vs_spatial_wavenumber(
                            spectral_df=field_spectral_df,
                            output_path=(
                                leadtime_output_path
                                / (
                                    f"{field_name}_"
                                    "spearman_vs_spatial_wavenumber.png"
                                )
                            ),
                            title_suffix=f" ({FIELD_LABELS[field_name]})",
                        )
                    )

                    if not correlation_df.empty:
                        correlation_df.to_csv(
                            leadtime_output_path
                            / (
                                f"{field_name}_"
                                "spearman_vs_spatial_wavenumber.csv"
                            ),
                            index=False,
                        )

            print_summary(metrics_df)

# ============================================================================
# Normalization
# ============================================================================

def make_normalizer(
    s,
    dataset: XarrayDataset,
    dim: Literal["x", "y"],
):
    """
    Reproduce input normalization used during training
    """
    if s.normalization == "monthly":
        norm_class = MonthlyNormalize
    elif s.normalization == "full":
        norm_class = Normalize
    else:
        raise ValueError(
            f"Unsupported normalization={s.normalization!r}"
        )

    exclude_channels = None

    if dim == "x":
        n_excluded_channels = 0

        if (
            s.seasonal_encoding
            and s.channel_representation != "init_period"
        ):
            n_excluded_channels += 4

        if s.spatial_encoding:
            n_excluded_channels += 4

        exclude_channels = (
            tuple(range(-n_excluded_channels, 0))
            if n_excluded_channels > 0
            else None
        )

    return norm_class(
        mode=s.normalization_mode,
        exclude_channels=exclude_channels,
    ).fit(
        dataset,
        dim=dim,
    )

# ============================================================================
# Helpers
# ============================================================================

def dataset_kwargs_from_settings(s) -> dict:
    return {
        "target_realization_avg": s.target_realization_avg,
        "channel_representation": s.channel_representation,
        "init_period_dim": s.init_period_dim,
        "output_realizations": s.output_realizations,
        "torch_mask": s.torch_mask,
        "fill_nan_value": s.fill_nan_value,
    }


def _longitude_bounds_for_coord(
    coord: np.ndarray,
    bounds: tuple[float, float],
) -> tuple[float, float]:
    """Convert longitude bounds to the coordinate convention when needed."""
    values = np.asarray(coord, dtype=float)
    start, end = map(float, bounds)

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return start, end

    # Dataset uses 0..360 while requested bounds use -180..180.
    if finite.min() >= 0.0 and (start < 0.0 or end < 0.0):
        start %= 360.0
        end %= 360.0

    return start, end


def _coordinate_indices(
    coord: np.ndarray,
    bounds: tuple[float, float] | None,
    *,
    longitude: bool = False,
) -> np.ndarray:
    """Return coordinate indices inside inclusive bounds."""
    values = np.asarray(coord, dtype=float)

    if bounds is None:
        return np.arange(values.size, dtype=int)

    start, end = map(float, bounds)
    if longitude:
        start, end = _longitude_bounds_for_coord(values, (start, end))

    if longitude and start > end:
        # Bounds cross the longitude seam in 0..360 coordinates.
        selected = (values >= start) | (values <= end)
    else:
        lower = min(start, end)
        upper = max(start, end)
        selected = (values >= lower) & (values <= upper)

    indices = np.flatnonzero(selected)
    if indices.size == 0:
        kind = "longitude" if longitude else "latitude"
        raise ValueError(
            f"Diagnostic {kind} range {bounds} selects no grid points. "
            f"Available coordinate range is "
            f"({np.nanmin(values):g}, {np.nanmax(values):g})."
        )

    return indices


def subset_diagnostic_dataset_region(
    ds: xr.Dataset,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
) -> xr.Dataset:
    if lat_range is None and lon_range is None:
        return ds

    lat_dim = ds.earthml.guessed_dims.latitude
    lon_dim = ds.earthml.guessed_dims.longitude

    if lat_dim is None or lon_dim is None:
        raise ValueError(
            "Could not determine latitude/longitude dimensions."
        )

    lat_indices = _coordinate_indices(
        np.asarray(ds[lat_dim].values),
        lat_range,
    )
    lon_indices = _coordinate_indices(
        np.asarray(ds[lon_dim].values),
        lon_range,
        longitude=True,
    )

    return ds.isel({
        lat_dim: lat_indices,
        lon_dim: lon_indices,
    })


def diagnostic_spatial_dims(
    da: xr.DataArray,
    template_ds: xr.Dataset,
) -> tuple[list[str], str]:
    """Return spatial dimensions and the time dimension for diagnostics."""
    latitude_dim = template_ds.earthml.guessed_dims.latitude
    longitude_dim = template_ds.earthml.guessed_dims.longitude
    time_dim = template_ds.earthml.guessed_dims.time

    if latitude_dim is None or latitude_dim not in da.dims:
        raise ValueError("Could not determine latitude dimension.")
    if longitude_dim is None or longitude_dim not in da.dims:
        raise ValueError("Could not determine longitude dimension.")
    if time_dim is None or time_dim not in da.dims:
        raise ValueError("Could not determine time dimension.")

    return [latitude_dim, longitude_dim], time_dim


def diagnostic_region_label() -> str:
    """Filesystem-safe label for the configured diagnostic spatial box."""
    if DIAGNOSTIC_LAT_RANGE is None and DIAGNOSTIC_LON_RANGE is None:
        return "full_domain"

    parts = []
    if DIAGNOSTIC_LAT_RANGE is not None:
        lo, hi = sorted(map(float, DIAGNOSTIC_LAT_RANGE))
        parts.append(f"lat_{lo:g}_{hi:g}")
    if DIAGNOSTIC_LON_RANGE is not None:
        lo, hi = sorted(map(float, DIAGNOSTIC_LON_RANGE))
        parts.append(f"lon_{lo:g}_{hi:g}")

    return "_".join(parts).replace("-", "m").replace(".", "p")


def diagnostic_names() -> list[str]:
    single_cases, pairwise_cases = diagnostic_cases()

    names = [
        f"{field}_{metric}"
        for field, metric in single_cases
    ]

    names.extend(
        f"{field_a}_vs_{field_b}_{metric}"
        for field_a, field_b, metric in pairwise_cases
    )

    return names


def get_leadtime_xarray_datasets(
    s: Settings,
    time_range: Literal["train", "val", "test"] | tuple[str, str],
    leadtime: int,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset | None]:
    """
    Get input, target and prediction datasets for requested time_range
    """
    if time_range == "train":
        time_range = (s.train_start, s.train_end)
    elif time_range == "val":
        time_range = (s.val_start, s.val_end)
    elif time_range == "test":
        time_range = (s.test_start, s.test_end)
    elif isinstance(time_range, tuple):
        time_range = time_range
    else:
        raise ValueError(f"Time range {time_range} not supported")

    leadtime_units = (
        s.leadtime_unit
        if isinstance(s.leadtime_unit, LeadtimeUnit)
        else LeadtimeUnit(s.leadtime_unit)
    )

    lat_lon = (
        list(s.region.values())
        if s.region is not None
        else [None, None]
    )

    # Take full domain, subset occurs later
    fc, an, mlfc = get_and_subset_datasets(
        s,
        leadtime_units=leadtime_units,
        lat_range=lat_lon[0],
        lon_range=lat_lon[1],
        time_range=time_range,
        interpolate=INTERPOLATE_ANALYSIS,
        mlfc_path=None,
    )


    leadtime_dim = fc.earthml.guessed_dims.leadtime
    if leadtime_dim is None:
        raise ValueError("Could not determine leadtime dimension.")

    # Ensure all three datasets use only the requested lead
    fc = fc.sel({
        leadtime_dim: [leadtime]
    })
    an = an.sel({
        leadtime_dim: [leadtime]
    })
    mlfc = mlfc.sel({
        leadtime_dim: [leadtime]
    }) if mlfc is not None else None

    return fc, an, mlfc


def get_transformed_xy(
    dataset: XarrayDataset,
) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []

    for i in range(len(dataset)):
        x, y, _, _ = dataset[i]

        xs.append(x)
        ys.append(y)

    return (
        torch.stack(xs, dim=0),
        torch.stack(ys, dim=0),
    )


def _select_split_dataset(
    *,
    train_dataset: XarrayDataset,
    val_dataset: XarrayDataset | None,
    test_dataset: XarrayDataset,
    time_range: Literal["train", "val", "test"],
) -> XarrayDataset:
    if time_range == "train":
        return train_dataset

    if time_range == "val":
        if val_dataset is None:
            raise ValueError(
                "Validation dataset is not available."
            )
        return val_dataset

    if time_range == "test":
        return test_dataset

    raise ValueError(
        f"Unsupported time_range={time_range!r}"
    )


def _squeeze_single_leadtime(
    ds: xr.Dataset,
) -> xr.Dataset:
    leadtime_dim = ds.earthml.guessed_dims.leadtime

    if leadtime_dim is None:
        return ds

    if ds.sizes[leadtime_dim] != 1:
        raise ValueError(
            "Expected exactly one selected leadtime, "
            f"got {ds.sizes[leadtime_dim]}"
        )

    return ds.squeeze(
        leadtime_dim,
        drop=True,
    )


def _normalized_target_tensor(
    *,
    reference_dataset: XarrayDataset,
    target_ds: xr.Dataset,
    normalize_target: Normalize | MonthlyNormalize,
    s: Settings,
) -> torch.Tensor:
    target_dataset = XarrayDataset(
        input_ds=reference_dataset.input_ds,
        target_ds=target_ds,
        **dataset_kwargs_from_settings(s),
    )

    target_dataset.transform_y = normalize_target

    _, target_norm = get_transformed_xy(
        target_dataset
    )

    return target_norm


def get_normalized_datasets(
    s: Settings,
    *,
    train_dataset: XarrayDataset,
    val_dataset: XarrayDataset | None,
    test_dataset: XarrayDataset,
    normalize_input: Normalize | MonthlyNormalize,
    normalize_target: Normalize | MonthlyNormalize,
    fc: xr.Dataset,
    mlfc: xr.Dataset,
    time_range: Literal["train", "val", "test"],
) -> tuple[
    xr.Dataset,
    xr.Dataset,
    xr.Dataset,
]:
    # ------------------------------------------------------------
    # Select split and reproduce training normalization
    # ------------------------------------------------------------

    dataset = _select_split_dataset(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        time_range=time_range,
    )

    dataset.transform_x = normalize_input
    dataset.transform_y = normalize_target

    input_norm, target_norm = get_transformed_xy(
        dataset
    )

    # ------------------------------------------------------------
    # Construct model prediction in network target space
    # ------------------------------------------------------------

    if s.target_mode == "analysis":
        prediction_target_ds = (
            mlfc[[s.var_fc]]
            .rename({
                s.var_fc: s.var_an,
            })
        )

    elif s.target_mode == "residual":
        fc_base = fc[s.var_fc]

        realization_dim = (
            fc_base.earthml.guessed_dims.realization
        )

        if (
            realization_dim is not None
            and realization_dim in fc_base.dims
        ):
            fc_base = fc_base.mean(
                realization_dim,
                skipna=True,
            )

        prediction_target_ds = (
            mlfc[s.var_fc] - fc_base
        ).to_dataset(
            name=s.var_an
        )

    elif s.target_mode == "residual_realization":
        prediction_target_ds = (
            mlfc[s.var_fc] - fc[s.var_fc]
        ).to_dataset(
            name=s.var_an
        )

    else:
        raise NotImplementedError(
            "Normalized datasets are not implemented for "
            f"target_mode={s.target_mode!r}"
        )

    prediction_target_ds = _squeeze_single_leadtime(
        prediction_target_ds
    )

    prediction_norm = _normalized_target_tensor(
        reference_dataset=dataset,
        target_ds=prediction_target_ds,
        normalize_target=normalize_target,
        s=s,
    )

    # ------------------------------------------------------------
    # Construct baseline in normalized space
    # ------------------------------------------------------------

    if s.target_mode == "analysis":
        # Exact normalized network input
        baseline_norm = input_norm

    elif s.target_mode in {
        "residual",
        "residual_realization",
    }:
        # "No correction" in residual target space
        baseline_target_ds = xr.zeros_like(
            prediction_target_ds
        )

        baseline_norm = _normalized_target_tensor(
            reference_dataset=dataset,
            target_ds=baseline_target_ds,
            normalize_target=normalize_target,
            s=s,
        )

    else:
        raise NotImplementedError(
            "Normalized baseline is not implemented for "
            f"target_mode={s.target_mode!r}"
        )

    # ------------------------------------------------------------
    # Validate tensor shapes
    # ------------------------------------------------------------

    if prediction_norm.shape != target_norm.shape:
        raise ValueError(
            "Normalized prediction/target shape mismatch: "
            f"{prediction_norm.shape} != {target_norm.shape}"
        )

    if baseline_norm.shape != target_norm.shape:
        raise ValueError(
            "Normalized baseline/target shape mismatch: "
            f"{baseline_norm.shape} != {target_norm.shape}"
        )

    # ------------------------------------------------------------
    # Convert tensors back to labelled xarray datasets
    # ------------------------------------------------------------

    baseline_norm_ds = convert_to_xarray(
        baseline_norm,
        dataset,
        [s.var_fc],
    )

    target_norm_ds = convert_to_xarray(
        target_norm,
        dataset,
        [s.var_an],
    )

    prediction_norm_ds = convert_to_xarray(
        prediction_norm,
        dataset,
        [s.var_fc],
    )

    # ------------------------------------------------------------
    # Restore selected leadtime dimension
    # ------------------------------------------------------------

    leadtime_dim = fc.earthml.guessed_dims.leadtime

    if leadtime_dim is None:
        raise ValueError(
            "Could not determine leadtime dimension "
            "from physical forecast."
        )

    if fc.sizes[leadtime_dim] != 1:
        raise ValueError(
            "Expected exactly one selected leadtime, "
            f"got {fc.sizes[leadtime_dim]}"
        )

    leadtime_coord = fc[leadtime_dim].values

    def restore_leadtime(
        ds: xr.Dataset,
    ) -> xr.Dataset:
        return ds.expand_dims({
            leadtime_dim: leadtime_coord,
        })

    return (
        restore_leadtime(baseline_norm_ds),
        restore_leadtime(target_norm_ds),
        restore_leadtime(prediction_norm_ds),
    )

# ============================================================================
# Spectral diagnostics
# ============================================================================

def make_spectral_geometry(
    height: int,
    width: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Precompute quantities that depend only on grid geometry.

    Returns
    -------
    window
        2-D Hann window, shape (H, W).

    radial_k
        Normalized radial wavenumber for the full FFT grid,
        shape (H, W), ranging from 0 to 1.

    nonzero
        Boolean mask excluding the zero-wavenumber mode.
    """
    window_y = torch.hann_window(
        height,
        periodic=False,
        dtype=dtype,
        device=device,
    )

    window_x = torch.hann_window(
        width,
        periodic=False,
        dtype=dtype,
        device=device,
    )

    window = (
        window_y[:, None]
        * window_x[None, :]
    )

    ky = torch.fft.fftfreq(
        height,
        dtype=dtype,
        device=device,
    )

    kx = torch.fft.fftfreq(
        width,
        dtype=dtype,
        device=device,
    )

    radial_k = torch.sqrt(
        ky[:, None] ** 2
        + kx[None, :] ** 2
    )

    k_max = radial_k.max()

    if (
        not torch.isfinite(k_max)
        or k_max <= 0
    ):
        raise ValueError(
            "Could not construct normalized radial "
            "wavenumber grid."
        )

    radial_k = radial_k / k_max

    nonzero = radial_k > 0

    return (
        window,
        radial_k,
        nonzero,
    )


def collect_spectral_diagnostics(
    da: xr.DataArray,
    field_name: str,
    *,
    template_ds: xr.Dataset,
    leadtime: int,
    n_bins: int,
    high_frequency_threshold: float,
    collect_radial_spectrum: bool,
    batch_size: int = SPECTRAL_BATCH_SIZE,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Calculate scalar spectral diagnostics and radial spectra
    using batched I/O and batched FFTs.

    One FFT per timestep, but timesteps are processed together
    to avoid repeated small xarray/Zarr reads.
    """
    if not 0.0 < high_frequency_threshold < 1.0:
        raise ValueError(
            "high_frequency_threshold must be "
            "strictly between 0 and 1."
        )

    spatial_dims, time_dim = diagnostic_spatial_dims(
        da,
        template_ds,
    )

    extra_dims = [
        dim
        for dim in da.dims
        if dim not in {
            time_dim,
            *spatial_dims,
        }
    ]

    for dim in extra_dims:
        if da.sizes[dim] != 1:
            raise ValueError(
                f"Unexpected non-singleton dimension {dim!r} "
                f"with size {da.sizes[dim]} in spectral diagnostic."
            )

        da = da.squeeze(
            dim,
            drop=True,
        )

    da = da.transpose(
        time_dim,
        *spatial_dims,
    )

    n_times = da.sizes[time_dim]
    height = da.sizes[spatial_dims[0]]
    width = da.sizes[spatial_dims[1]]

    times = pd.to_datetime(
        da[time_dim].values
    )

    # ------------------------------------------------------------
    # Geometry: calculated once
    # ------------------------------------------------------------

    window, radial_k_grid, nonzero = make_spectral_geometry(
        height,
        width,
        dtype=torch.float32,
        device="cpu",
    )

    radial_k = radial_k_grid[nonzero]

    # Spectral masks calculated once
    high_mask = (
        radial_k
        >= high_frequency_threshold
    )
    low_mask = ~high_mask

    # ------------------------------------------------------------
    # Radial-bin mapping: calculated once
    # ------------------------------------------------------------

    if collect_radial_spectrum:
        bin_index = torch.floor(
            radial_k * n_bins
        ).long()

        bin_index = torch.clamp(
            bin_index,
            min=0,
            max=n_bins - 1,
        )

        wavenumber_centers = (
            (
                torch.arange(
                    n_bins,
                    dtype=torch.float32,
                )
                + 0.5
            )
            / n_bins
        ).numpy()

    scalar_frames = []
    spectral_frames = []

    # ------------------------------------------------------------
    # Process multiple timesteps at once
    # ------------------------------------------------------------

    batch_starts = range(
        0,
        n_times,
        batch_size,
    )

    for start in tqdm(
        batch_starts,
        total=(
            n_times + batch_size - 1
        ) // batch_size,
        desc=f"{field_name}: spectra",
        unit="batch",
        dynamic_ncols=True,
    ):
        stop = min(
            start + batch_size,
            n_times,
        )

        # one xarray/Zarr read for the entire batch
        block_values = np.asarray(
            da.isel(
                {
                    time_dim: slice(
                        start,
                        stop,
                    )
                }
            ).values,
            dtype=np.float32,
        )

        fields = torch.from_numpy(
            block_values
        )

        valid = torch.isfinite(
            fields
        )

        batch_count = (
            stop - start
        )

        # --------------------------------------------------------
        # Spatial mean independently for every timestep
        # --------------------------------------------------------

        valid_count = valid.sum(
            dim=(-2, -1)
        )

        field_sum = torch.where(
            valid,
            fields,
            torch.zeros_like(fields),
        ).sum(
            dim=(-2, -1)
        )

        means = (
            field_sum
            / valid_count.clamp_min(1)
        )

        fields = torch.where(
            valid,
            fields
            - means[:, None, None],
            torch.zeros_like(fields),
        )

        # --------------------------------------------------------
        # Window
        # --------------------------------------------------------

        fields = (
            fields
            * window[None, :, :]
        )

        # --------------------------------------------------------
        # BATCHED FFT
        #
        # Input:
        #   (B, H, W)
        #
        # fft2 operates independently over H,W for all B timesteps.
        # --------------------------------------------------------

        fft = torch.fft.fft2(
            fields,
            dim=(-2, -1),
        )

        power_full = (
            fft.real.square()
            + fft.imag.square()
        )

        # Flatten H,W and retain only non-zero modes.
        power = power_full.reshape(
            batch_count,
            -1,
        )[
            :,
            nonzero.reshape(-1),
        ]

        total_power = power.sum(
            dim=1
        )

        good = (
            torch.isfinite(total_power)
            & (total_power > 0)
            & (valid_count > 0)
        )

        # --------------------------------------------------------
        # Scalar spectral quantities
        # --------------------------------------------------------

        spectral_centroid = torch.full(
            (batch_count,),
            torch.nan,
            dtype=torch.float32,
        )

        low_fraction = torch.full_like(
            spectral_centroid,
            torch.nan,
        )

        high_fraction = torch.full_like(
            spectral_centroid,
            torch.nan,
        )

        high_low_ratio = torch.full_like(
            spectral_centroid,
            torch.nan,
        )

        if good.any():
            centroid_numerator = (
                power
                * radial_k[None, :]
            ).sum(
                dim=1
            )

            spectral_centroid[good] = (
                centroid_numerator[good]
                / total_power[good]
            )

            p_low = power[
                :,
                low_mask,
            ].sum(
                dim=1
            )

            p_high = power[
                :,
                high_mask,
            ].sum(
                dim=1
            )

            low_fraction[good] = (
                p_low[good]
                / total_power[good]
            )

            high_fraction[good] = (
                p_high[good]
                / total_power[good]
            )

            ratio_good = (
                good
                & (p_low > 0)
            )

            high_low_ratio[
                ratio_good
            ] = (
                p_high[ratio_good]
                / p_low[ratio_good]
            )

        batch_times = times[
            start:stop
        ]

        scalar_frames.append(
            pd.DataFrame({
                "time": batch_times,
                "low_frequency_power_fraction":
                    low_fraction.numpy(),
                "high_frequency_power_fraction":
                    high_fraction.numpy(),
                "high_low_power_ratio":
                    high_low_ratio.numpy(),
                "spectral_centroid":
                    spectral_centroid.numpy(),
            })
        )

        # --------------------------------------------------------
        # Radially binned power
        # --------------------------------------------------------

        if collect_radial_spectrum:
            binned_power = torch.zeros(
                (
                    batch_count,
                    n_bins,
                ),
                dtype=torch.float32,
            )

            expanded_bin_index = (
                bin_index[None, :]
                .expand(
                    batch_count,
                    -1,
                )
            )

            binned_power.scatter_add_(
                1,
                expanded_bin_index,
                power,
            )

            power_fraction = torch.full(
                (
                    batch_count,
                    n_bins,
                ),
                torch.nan,
                dtype=torch.float32,
            )

            power_fraction[good] = (
                binned_power[good]
                / total_power[
                    good,
                    None,
                ]
            )

            power_fraction_np = (
                power_fraction.numpy()
            )

            spectral_frames.append(
                pd.DataFrame({
                    "leadtime": leadtime,
                    "time": np.repeat(
                        batch_times,
                        n_bins,
                    ),
                    "wavenumber": np.tile(
                        wavenumber_centers,
                        batch_count,
                    ),
                    "power_fraction":
                        power_fraction_np.reshape(-1),
                })
            )

    scalar_df = (
        pd.concat(
            scalar_frames,
            ignore_index=True,
        )
        if scalar_frames
        else pd.DataFrame()
    )

    spectral_df = (
        pd.concat(
            spectral_frames,
            ignore_index=True,
        )
        if spectral_frames
        else pd.DataFrame()
    )

    return (
        scalar_df,
        spectral_df,
    )

# ============================================================================
# Diagnostic core functions
# ============================================================================

def rms_over_dims(
    da: xr.DataArray,
    dims: list[str],
) -> xr.DataArray:
    return np.sqrt(
        (da**2).mean(
            dim=dims,
            skipna=True,
        )
    )


def laplacian_metric(
    da: xr.DataArray,
    *,
    spatial_dims: list[str],
    metric: Literal[
        "rms_laplacian",
        "mean_abs_laplacian",
    ],
) -> xr.DataArray:
    """Unscaled five-point grid-cell Laplacian aggregated spatially."""
    latitude_dim, longitude_dim = spatial_dims

    north = da.shift({latitude_dim: -1})
    south = da.shift({latitude_dim: 1})
    east = da.shift({longitude_dim: -1})
    west = da.shift({longitude_dim: 1})

    valid = (
        np.isfinite(da)
        & np.isfinite(north)
        & np.isfinite(south)
        & np.isfinite(east)
        & np.isfinite(west)
    )

    laplacian = (
        north
        + south
        + east
        + west
        - 4.0 * da
    ).where(valid)

    if metric == "rms_laplacian":
        return rms_over_dims(
            laplacian,
            spatial_dims,
        )

    if metric == "mean_abs_laplacian":
        return np.abs(laplacian).mean(
            dim=spatial_dims,
            skipna=True,
        )

    raise ValueError(f"Unsupported Laplacian metric: {metric!r}")


def pairwise_spatial_correlation(
    a: xr.DataArray,
    b: xr.DataArray,
    *,
    spatial_dims: list[str],
) -> xr.DataArray:
    """Pearson correlation across space for each timestep."""
    valid = np.isfinite(a) & np.isfinite(b)
    a = a.where(valid)
    b = b.where(valid)

    a_anom = a - a.mean(
        dim=spatial_dims,
        skipna=True,
    )
    b_anom = b - b.mean(
        dim=spatial_dims,
        skipna=True,
    )

    covariance = (a_anom * b_anom).mean(
        dim=spatial_dims,
        skipna=True,
    )
    denominator = np.sqrt(
        (a_anom**2).mean(
            dim=spatial_dims,
            skipna=True,
        )
        * (b_anom**2).mean(
            dim=spatial_dims,
            skipna=True,
        )
    )

    return covariance / denominator


def pairwise_cosine_similarity(
    a: xr.DataArray,
    b: xr.DataArray,
    *,
    spatial_dims: list[str],
) -> xr.DataArray:
    """Cosine similarity across space for each timestep."""
    valid = np.isfinite(a) & np.isfinite(b)
    a = a.where(valid)
    b = b.where(valid)

    numerator = (a * b).mean(
        dim=spatial_dims,
        skipna=True,
    )
    denominator = np.sqrt(
        (a**2).mean(
            dim=spatial_dims,
            skipna=True,
        )
        * (b**2).mean(
            dim=spatial_dims,
            skipna=True,
        )
    )

    return numerator / denominator

# ============================================================================
# Diagnostics collectors
# ============================================================================

def build_fields(
    baseline: xr.DataArray,
    target: xr.DataArray,
    corrected: xr.DataArray,
) -> dict[str, xr.DataArray]:
    return {
        "input": baseline,
        "target": target,
        "corrected": corrected,
        "fc_error": baseline - target,
        "mlfc_error": corrected - target,
        "ideal_correction": target - baseline,
        "ml_correction": corrected - baseline,
    }


def diagnostic_cases() -> tuple[
    list[tuple[str, str]],
    list[tuple[str, str, str]],
]:
    single_cases = []
    pairwise_cases = []

    for metric in DIAGNOSTIC_METRICS:
        if metric in SINGLE_FIELD_METRICS:
            for field in DIAGNOSTIC_FIELDS:
                single_cases.append(
                    (field, metric)
                )

        elif metric in PAIRWISE_METRICS:
            for field_a, field_b in combinations(
                DIAGNOSTIC_FIELDS,
                2,
            ):
                pairwise_cases.append(
                    (
                        field_a,
                        field_b,
                        metric,
                    )
                )

        else:
            raise ValueError(
                f"Unsupported diagnostic metric: {metric!r}"
            )

    return single_cases, pairwise_cases


def calculate_single_field_diagnostic(
    field: xr.DataArray,
    metric: str,
    *,
    spatial_dims: list[str],
) -> xr.DataArray:

    if metric == "mean":
        return field.mean(
            dim=spatial_dims,
            skipna=True,
        )

    if metric == "std":
        return field.std(
            dim=spatial_dims,
            skipna=True,
        )

    if metric in {
        "rms_laplacian",
        "mean_abs_laplacian",
    }:
        return laplacian_metric(
            field,
            spatial_dims=spatial_dims,
            metric=metric,
        )

    raise ValueError(
        f"Unsupported single-field metric: {metric!r}"
    )


def calculate_pairwise_diagnostic(
    field_a: xr.DataArray,
    field_b: xr.DataArray,
    metric: str,
    *,
    spatial_dims: list[str],
) -> xr.DataArray:
    if metric == "correlation":
        return pairwise_spatial_correlation(
            field_a,
            field_b,
            spatial_dims=spatial_dims,
        )

    if metric == "cosine":
        return pairwise_cosine_similarity(
            field_a,
            field_b,
            spatial_dims=spatial_dims,
        )

    if metric == "rms_ratio":
        return (
            rms_over_dims(field_a, spatial_dims)
            / rms_over_dims(field_b, spatial_dims)
        )

    if metric == "projection":
        valid = (
            np.isfinite(field_a)
            & np.isfinite(field_b)
        )

        a = field_a.where(valid)
        b = field_b.where(valid)

        return (
            (a * b).mean(
                dim=spatial_dims,
                skipna=True,
            )
            / (b**2).mean(
                dim=spatial_dims,
                skipna=True,
            )
        )

    if metric == "orthogonal_rms":
        valid = (
            np.isfinite(field_a)
            & np.isfinite(field_b)
        )

        a = field_a.where(valid)
        b = field_b.where(valid)

        alpha = (
            (a * b).mean(
                dim=spatial_dims,
                skipna=True,
            )
            / (b**2).mean(
                dim=spatial_dims,
                skipna=True,
            )
        )

        orthogonal = a - alpha * b

        return rms_over_dims(
            orthogonal,
            spatial_dims,
        )

    if metric == "mean_product":
        valid = (
            np.isfinite(field_a)
            & np.isfinite(field_b)
        )

        return (
            field_a.where(valid)
            * field_b.where(valid)
        ).mean(
            dim=spatial_dims,
            skipna=True,
        )

    raise ValueError(
        f"Unsupported pairwise metric: {metric!r}"
    )


def get_diagnostic_timeseries(
    *,
    fc: xr.Dataset,
    an: xr.Dataset,
    mlfc: xr.Dataset,
    s: Settings,
) -> pd.DataFrame:
    fc_da = fc[s.var_fc]
    an_da = an[s.var_an]
    mlfc_da = mlfc[s.var_fc]

    spatial_dims, time_dim = diagnostic_spatial_dims(
        da=fc_da,
        template_ds=fc,
    )

    fields = build_fields(
        fc_da,
        an_da,
        mlfc_da,
    )

    result = pd.DataFrame({
        "time": pd.to_datetime(
            fc_da[time_dim].values
        ),
    })

    def add_result(
        name: str,
        values: xr.DataArray,
    ) -> None:
        values = values.squeeze(drop=True)

        extra_dims = [
            dim
            for dim in values.dims
            if dim != time_dim
        ]

        if extra_dims:
            raise ValueError(
                f"{name!r} still has dimensions "
                f"{values.dims}"
            )

        result[name] = pd.to_numeric(
            values.values,
            errors="coerce",
        )

    single_cases, pairwise_cases = (
        diagnostic_cases()
    )

    for field_name, metric in single_cases:
        if metric in SPECTRAL_METRICS:
            continue

        values = calculate_single_field_diagnostic(
            fields[field_name],
            metric,
            spatial_dims=spatial_dims,
        )

        add_result(
            f"{field_name}_{metric}",
            values,
        )

    for (
        field_a,
        field_b,
        metric,
    ) in pairwise_cases:
        values = calculate_pairwise_diagnostic(
            fields[field_a],
            fields[field_b],
            metric,
            spatial_dims=spatial_dims,
        )

        add_result(
            f"{field_a}_vs_{field_b}_{metric}",
            values,
        )

    return (
        result
        .drop_duplicates(subset="time")
        .sort_values("time")
        .reset_index(drop=True)
    )

# ============================================================================
# Metric performance
# ============================================================================

def get_metric_improvement_timeseries(
    s: Settings,
    baseline: xr.Dataset,
    target: xr.Dataset,
    corrected: xr.Dataset,
    leadtime: int,
    metric: str,
    improvement_unit: ImprovementUnit = PERFORMANCE_IMPROVEMENT_UNIT,
    period_reference: PeriodReference = PERFORMANCE_REFERENCE_TIME_PERIOD,

) -> pd.DataFrame:
    """
    Calculate baseline -> corrected metric improvement for every initialization.
    Positive improvement always means MLFC is better.
    """
    clim_period = (
        s.clim_period
        if isinstance(s.clim_period, ClimPeriod)
        else ClimPeriod(s.clim_period)
    )

    # ------------------------------------------------------------
    # Baseline forecast metric
    # ------------------------------------------------------------

    print("Calculating FC performance metric...", flush=True)

    fc_metrics = get_metrics(
        an=target,
        fc=baseline,
        var=s.var_fc,
        metric_kind="timeseries",
        leadtime_agg="single",
        realization_agg=True,
        an_clim=None,
        fc_clim=None,
        orography_path=None,
        metrics=[metric],
        leadtime_windows=s.seasonal_leadtime_windows,
        leadtime_agg_coord="leadtime",
        clim_period=clim_period,
        period_reference=period_reference,
        period_dim=f"init_{clim_period.value}",
        periods_requested=["all"],
        leadtime_unit=s.leadtime_unit,
        align=False,
        fair_correction=False,
    )

    # ------------------------------------------------------------
    # Corrected forecast metric
    # ------------------------------------------------------------

    print("Calculating MLFC performance metric...", flush=True)

    mlfc_metrics = get_metrics(
        an=target,
        fc=corrected,
        var=s.var_fc,
        metric_kind="timeseries",
        leadtime_agg="single",
        realization_agg=True,
        an_clim=None,
        fc_clim=None,
        orography_path=None,
        metrics=[metric],
        leadtime_windows=s.seasonal_leadtime_windows,
        leadtime_agg_coord="leadtime",
        clim_period=clim_period,
        period_reference=period_reference,
        period_dim=f"init_{clim_period.value}",
        periods_requested=["all"],
        leadtime_unit=s.leadtime_unit,
        align=False,
        fair_correction=False,
    )

    # ------------------------------------------------------------
    # Improvement
    # ------------------------------------------------------------

    print("Building metric improvement...", flush=True)

    improvements = build_metric_improvements(
        fc_metrics,
        mlfc_metrics,
        metric=metric,
        baseline_model="fc",
        target_model="mlfc",
        improvement_units=(
            improvement_unit,
        ),
    )

    if len(improvements) != 1:
        raise RuntimeError(
            "Expected exactly one improvement DataArray, "
            f"got {list(improvements)}"
        )

    comparison_model, improvement_da = next(
        iter(improvements.items())
    )

    print(f"Using performance metric: {metric} for {comparison_model}")

    # Select requested lead
    if (
        "leadtime" in improvement_da.dims
        or "leadtime" in improvement_da.coords
    ):
        improvement_da = improvement_da.sel(
            leadtime=leadtime
        )

    # Remove singleton dimensions such as period="all"
    improvement_da = improvement_da.squeeze(
        drop=True
    )

    time_dim = improvement_da.earthml.guessed_dims.time
    if time_dim is None:
        raise ValueError("Could not determine time dimension of metric improvement")

    # After selecting the lead and period, this should be a one-dimensional timeseries
    extra_dims = [
        dim
        for dim in improvement_da.dims
        if dim != time_dim
    ]

    if extra_dims:
        raise ValueError(f"Metric improvement still contains unexpected dimensions: {improvement_da.dims}")

    metric_df = pd.DataFrame({
        "time": pd.to_datetime(
            improvement_da[
                time_dim
            ].values
        ),
        "metric_improvement": (
            improvement_da.values
        ),
    })

    metric_df["metric_improvement"] = (
        pd.to_numeric(
            metric_df["metric_improvement"],
            errors="coerce",
        )
    )

    metric_df = (
        metric_df
        .drop_duplicates(
            subset="time"
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    return metric_df

# ============================================================================
# Spectral metrics
# ============================================================================

def get_spectral_metrics(
    s: Settings,
    baseline: xr.Dataset,
    target: xr.Dataset,
    corrected: xr.Dataset,
    leadtime: int,
):
    diagnostic_fields = build_fields(
        baseline[s.var_fc],
        target[s.var_an],
        corrected[s.var_fc],
    )

    need_spectral_diagnostics = any(
        metric in SPECTRAL_METRICS
        for metric in DIAGNOSTIC_METRICS
    )

    need_spectral_work = (
        need_spectral_diagnostics
        or PLOT_SPECTRAL_WAVENUMBER_CORRELATION
    )

    spectral_scalar_frames = []
    spectral_frames = []

    if need_spectral_work:
        requested_spectral_metrics = [
            metric
            for metric in DIAGNOSTIC_METRICS
            if metric in SPECTRAL_METRICS
        ]

        for field_name in DIAGNOSTIC_FIELDS:
            field = diagnostic_fields[field_name]

            print(
                f"Calculating {field_name} spectral diagnostics "
                f"for {field.sizes}...",
                flush=True,
            )

            scalar_df, radial_df = collect_spectral_diagnostics(
                da=field,
                field_name=field_name,
                template_ds=baseline,
                leadtime=leadtime,
                n_bins=SPECTRAL_N_BINS,
                high_frequency_threshold=HIGH_FREQ_THRESHOLD,
                collect_radial_spectrum=(
                    PLOT_SPECTRAL_WAVENUMBER_CORRELATION
                ),
            )

            if (
                requested_spectral_metrics
                and not scalar_df.empty
            ):
                scalar_df = scalar_df[
                    [
                        "time",
                        *requested_spectral_metrics,
                    ]
                ].rename(
                    columns={
                        metric: f"{field_name}_{metric}"
                        for metric in requested_spectral_metrics
                    }
                )

                spectral_scalar_frames.append(
                    scalar_df
                )

            if not radial_df.empty:
                radial_df = radial_df.copy()
                radial_df["field"] = field_name

                spectral_frames.append(
                    radial_df
                )

    spectral_scalar_df = (
        spectral_scalar_frames[0]
        if spectral_scalar_frames
        else pd.DataFrame()
    )

    for frame in spectral_scalar_frames[1:]:
        spectral_scalar_df = spectral_scalar_df.merge(
            frame,
            on="time",
            how="outer",
            validate="one_to_one",
        )

    spectral_df = (
        pd.concat(
            spectral_frames,
            ignore_index=True,
        )
        if spectral_frames
        else pd.DataFrame()
    )

    return spectral_scalar_df, spectral_df

# ============================================================================
# Performance coloring
# ============================================================================

def make_performance_norm(
    metrics_df: pd.DataFrame,
) -> mcolors.TwoSlopeNorm | None:
    """
    Build a robust symmetric normalization centered at zero.
    """
    if (
        "metric_improvement"
        not in metrics_df.columns
    ):
        return None

    values = (
        pd.to_numeric(
            metrics_df["metric_improvement"],
            errors="coerce",
        )
        .dropna()
        .to_numpy(dtype=float)
    )

    values = values[np.isfinite(values)]
    if values.size == 0:
        return None

    abs_values = np.abs(values)

    vmax = float(
        np.quantile(
            abs_values,
            PERFORMANCE_COLOR_QUANTILE,
        )
    )

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(
            np.nanmax(abs_values)
        )
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    return mcolors.TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0.0,
        vmax=vmax,
    )


def performance_color(
    value,
    *,
    cmap,
    norm,
):
    normalized = np.clip(
        norm(value),
        0.0,
        1.0,
    )

    return cmap(normalized)


# ============================================================================
# Performance extrema
# ============================================================================

def get_performance_extrema(
    stats: pd.DataFrame,
    *,
    n_best: int,
    n_worst: int,
) -> pd.DataFrame:
    """
    Return unique best/worst timesteps according to metric improvement.

    Positive improvement means the corrected forecast is better, so the
    largest values are the best timesteps and the smallest values are the
    worst timesteps.

    Performance is currently attached only to test rows. If one initialization
    corresponds to multiple samples, it still appears only once here.
    """
    columns = [
        "time",
        "metric_improvement",
        "extreme",
        "rank",
    ]

    if "metric_improvement" not in stats.columns:
        return pd.DataFrame(columns=columns)

    perf = (
        stats.loc[
            stats["metric_improvement"].notna(),
            [
                "time",
                "metric_improvement",
            ],
        ]
        .drop_duplicates(subset="time")
        .sort_values("metric_improvement")
        .reset_index(drop=True)
    )

    if perf.empty:
        return pd.DataFrame(columns=columns)

    n_best = max(int(n_best), 0)
    n_worst = max(int(n_worst), 0)

    frames = []
    if n_worst > 0:
        worst = perf.head(n_worst).copy()

        worst["extreme"] = "worst"
        worst["rank"] = np.arange(1, len(worst) + 1)

        frames.append(worst)

    if n_best > 0:
        best = (
            perf.tail(n_best)
            .sort_values(
                "metric_improvement",
                ascending=False,
            )
            .copy()
        )

        best["extreme"] = "best"
        best["rank"] = np.arange(1, len(best) + 1)

        frames.append(best)

    if not frames:
        return pd.DataFrame(columns=columns)
    return (
        pd.concat(
            frames,
            ignore_index=True,
        )
        .drop_duplicates(subset="time")
        .reset_index(drop=True)
    )

def format_extrema_label(row: pd.Series) -> str:
    prefix = (
        "B"
        if row["extreme"] == "best"
        else "W"
    )

    return f"{prefix}{int(row['rank'])}"

def format_extrema_annotation(row: pd.Series) -> str:
    time = pd.Timestamp(row["time"])

    return (
        f"{format_extrema_label(row)} "
        f"{time.strftime(ANNOTATION_DATE_FORMAT)}"
    )

# ============================================================================
# Plotting helpers
# ============================================================================

def annotate_nonoverlapping(
    ax,
    text: str,
    *,
    xy: tuple[float, float],
    occupied: list,
    rotation: float = 0.0,
) -> None:
    """
    Annotate a point using a leader line while attempting to avoid
    overlap with previously placed annotations.
    """
    fig = ax.figure

    # Need a renderer for label bounding-box calculations.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    x, _ = xy
    x_center = np.mean(ax.get_xlim())

    # Prefer putting labels outward from the plot center.
    direction = -1 if x < x_center else 1

    candidate_offsets = [
        (direction * 8, 8),
        (-direction * 8, 8),
        (direction * 12, 16),
        (-direction * 12, 16),
        (direction * 18, 24),
        (-direction * 18, 24),
        (direction * 24, 32),
        (-direction * 24, 32),
        (direction * 32, 40),
        (-direction * 32, 40),
        (direction * 40, 50),
        (-direction * 40, 50),
    ]

    axes_bbox = ax.get_window_extent(renderer)

    annotation = None

    for x_offset, y_offset in candidate_offsets:
        candidate = ax.annotate(
            text,
            xy=xy,
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha="left" if x_offset > 0 else "right",
            va="bottom",
            fontsize=ANNOTATION_FONTSIZE,
            rotation=rotation,
            color="black",
            bbox={
                "boxstyle": "round,pad=0.10",
                "facecolor": "white",
                "edgecolor": "0.5",
                "linewidth": 0.3,
                "alpha": 0.75,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": "0.25",
                "linewidth": 0.6,
                "shrinkA": 1,
                "shrinkB": 2,
            },
            annotation_clip=True,
            zorder=8,
        )

        bbox = candidate.get_window_extent(
            renderer
        ).expanded(
            1.05,
            1.10,
        )

        inside_axes = (
            bbox.x0 >= axes_bbox.x0
            and bbox.x1 <= axes_bbox.x1
            and bbox.y0 >= axes_bbox.y0
            and bbox.y1 <= axes_bbox.y1
        )

        overlaps = any(
            bbox.overlaps(existing)
            for existing in occupied
        )

        if inside_axes and not overlaps:
            annotation = candidate
            occupied.append(bbox)
            break

        candidate.remove()

    # If every candidate overlaps, keep the last-resort position.
    if annotation is None:
        x_offset, y_offset = candidate_offsets[-1]

        annotation = ax.annotate(
            text,
            xy=xy,
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha="left" if x_offset > 0 else "right",
            va="bottom",
            fontsize=ANNOTATION_FONTSIZE,
            rotation=rotation,
            color="black",
            bbox={
                "boxstyle": "round,pad=0.10",
                "facecolor": "white",
                "edgecolor": "0.5",
                "linewidth": 0.3,
                "alpha": 0.75,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": "0.25",
                "linewidth": 0.6,
                "shrinkA": 1,
                "shrinkB": 2,
            },
            annotation_clip=True,
            zorder=8,
        )

        occupied.append(
            annotation.get_window_extent(renderer).expanded(
                1.05,
                1.10,
            )
        )

# ============================================================================
# Diagnostic vs performance
# ============================================================================

def diagnostic_label(name: str) -> str:
    single_cases, pairwise_cases = diagnostic_cases()

    for field, metric in single_cases:
        if name == f"{field}_{metric}":
            return (
                f"{METRIC_LABELS[metric]} "
                f"({FIELD_LABELS[field]})"
            )

    for field_a, field_b, metric in pairwise_cases:
        if name == f"{field_a}_vs_{field_b}_{metric}":
            return (
                f"{METRIC_LABELS[metric]} "
                f"({FIELD_LABELS[field_a]} vs "
                f"{FIELD_LABELS[field_b]})"
            )

    return name


def plot_diagnostic_vs_performance(
    metrics_df: pd.DataFrame,
    *,
    diagnostic: str,
    output_path: Path,
    title_suffix: str = "",
) -> None:
    """Scatter one scalar timestep diagnostic against metric improvement."""
    required_columns = {
        "time",
        "metric_improvement",
        diagnostic,
    }

    if not required_columns.issubset(metrics_df.columns):
        return

    df = (
        metrics_df[
            [
                "time",
                "metric_improvement",
                diagnostic,
            ]
        ]
        .drop_duplicates(subset="time")
        .copy()
    )

    df["metric_improvement"] = pd.to_numeric(
        df["metric_improvement"],
        errors="coerce",
    )
    df[diagnostic] = pd.to_numeric(
        df[diagnostic],
        errors="coerce",
    )

    valid = (
        np.isfinite(df["metric_improvement"])
        & np.isfinite(df[diagnostic])
    )
    df = df.loc[valid].copy()

    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        df[diagnostic],
        df["metric_improvement"],
        s=SCATTER_SIZE,
        color=SCATTER_COLOR,
        alpha=SCATTER_ALPHA,
        edgecolors="none",
        zorder=3,
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        color="black",
        alpha=0.7,
        zorder=2,
    )

    rho = np.nan
    p_value = np.nan
    if len(df) >= 2:
        correlation = spearmanr(
            df[diagnostic].to_numpy(dtype=float),
            df["metric_improvement"].to_numpy(dtype=float),
            nan_policy="omit",
        )
        rho = float(correlation.statistic)
        p_value = float(correlation.pvalue)

    correlation_lines = [f"n = {len(df)}"]
    if np.isfinite(rho):
        correlation_lines.append(rf"Spearman $\rho$ = {rho:.3f}")
    if np.isfinite(p_value):
        correlation_lines.append(f"p = {p_value:.3g}")

    # Calculate and filter extrema
    extrema = get_performance_extrema(
        df,
        n_best=ANNOTATE_BEST_TIMESTEPS,
        n_worst=ANNOTATE_WORST_TIMESTEPS,
    )

    if SELECTED_SAMPLES_TO_ANNOTATE is not None:
        extrema = extrema[
            extrema.apply(
                format_extrema_label,
                axis=1,
            ).isin(SELECTED_SAMPLES_TO_ANNOTATE)
        ]

    occupied_annotations = []
    performance_norm = make_performance_norm(metrics_df)
    performance_cmap = plt.get_cmap(PERFORMANCE_CMAP)

    for _, row in extrema.iterrows():
        time = pd.Timestamp(row["time"])
        point = df[df["time"] == time]
        if point.empty:
            continue

        x = float(point[diagnostic].iloc[0])
        y = float(point["metric_improvement"].iloc[0])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue

        marker_color = (
            performance_color(
                y,
                cmap=performance_cmap,
                norm=performance_norm,
            )
            if performance_norm is not None
            else SCATTER_COLOR
        )

        ax.plot(
            x,
            y,
            marker="o",
            linestyle="none",
            markersize=SCATTER_MARKER_SIZE,
            markerfacecolor=marker_color,
            markeredgecolor="black",
            markeredgewidth=0.8,
            zorder=7,
        )

        annotate_nonoverlapping(
            ax,
            format_extrema_annotation(row),
            xy=(x, y),
            occupied=occupied_annotations,
        )

    ax.set_xlabel(diagnostic_label(diagnostic))
    ax.set_ylabel(
        f"{"Normalized " if PERFORMANCE_METRICS_NORMALIZED else ""}"
        f"{PERFORMANCE_METRIC.upper()} "
        f"improvement ({PERFORMANCE_IMPROVEMENT_UNIT})"
    )
    ax.set_title(
        f"Performance vs {diagnostic_label(diagnostic)}"
        f"{title_suffix}\n"
        f"{" · ".join(correlation_lines)}"
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_spearman_vs_spatial_wavenumber(
    spectral_df: pd.DataFrame,
    *,
    output_path: Path,
    title_suffix: str = "",
):
    """
    Plot Spearman correlation between normalized radial-bin power fraction
    and performance as a function of normalized spatial wavenumber.
    """
    required = {
        "time",
        "wavenumber",
        "power_fraction",
        "metric_improvement",
    }

    if spectral_df.empty or not required.issubset(spectral_df.columns):
        return pd.DataFrame()

    df = spectral_df[list(required)].copy()

    if df.empty:
        return pd.DataFrame()

    # If one initialization corresponds to multiple dataset samples, average
    # their spectral power before correlating so each timestep counts once
    df = (
        df.groupby(
            ["time", "wavenumber"],
            as_index=False,
        )
        .agg(
            power_fraction=("power_fraction", "mean"),
            metric_improvement=("metric_improvement", "first"),
        )
)

    rows = []
    for wavenumber, group in df.groupby("wavenumber", sort=True):
        valid = (
            group[["power_fraction", "metric_improvement"]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        rho = np.nan
        p_value = np.nan

        if (
            len(valid) >= 3
            and valid["power_fraction"].nunique() > 1
            and valid["metric_improvement"].nunique() > 1
        ):
            correlation = spearmanr(
                valid["power_fraction"].to_numpy(dtype=float),
                valid["metric_improvement"].to_numpy(dtype=float),
                nan_policy="omit",
            )
            rho = float(correlation.statistic)
            p_value = float(correlation.pvalue)

        rows.append({
            "wavenumber": float(wavenumber),
            "rho": rho,
            "p_value": p_value,
            "n": len(valid),
        })

    correlation_df = pd.DataFrame(rows).sort_values("wavenumber")
    valid_rho = correlation_df[np.isfinite(correlation_df["rho"])]

    if valid_rho.empty:
        return correlation_df

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        valid_rho["wavenumber"],
        valid_rho["rho"],
        s=26,
        color=SCATTER_COLOR,
        alpha=0.85,
        zorder=3,
    )

    # A light connecting line makes persistent scale ranges easier to see
    # while the markers remain the actual wavenumber-bin correlations
    ax.plot(
        valid_rho["wavenumber"],
        valid_rho["rho"],
        linewidth=0.8,
        color=SCATTER_COLOR,
        alpha=0.6,
        zorder=2,
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        color="black",
        alpha=0.7,
        zorder=1,
    )

    ax.axvline(
        HIGH_FREQ_THRESHOLD,
        linestyle=":",
        linewidth=1.0,
        color="black",
        alpha=0.7,
        label=f"current high-frequency threshold = {HIGH_FREQ_THRESHOLD:g}",
        zorder=1,
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-1.0, 1.0)

    ax.set_xlabel("Normalized radial spatial wavenumber")
    ax.set_ylabel(
        rf"Spearman $\rho$(power fraction, "
        f"{PERFORMANCE_METRIC.upper()} improvement)"
    )

    ax.set_title(
        f"{PERFORMANCE_METRIC.upper()} improvement vs spatial wavenumber"
        f"{title_suffix}"
    )

    ax.legend(
        loc=LEGEND_LOCATION,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return correlation_df

# ============================================================================
# Summary
# ============================================================================

def print_summary(
    metrics_df: pd.DataFrame,
) -> None:
    print()

    if "metric_improvement" in metrics_df.columns:
        performance = (
            pd.to_numeric(
                metrics_df["metric_improvement"],
                errors="coerce",
            )
            .dropna()
        )

        if not performance.empty:
            print(
                f"{PERFORMANCE_METRIC.upper()} "
                f"improvement "
                f"({PERFORMANCE_IMPROVEMENT_UNIT}):"
            )
            print(
                performance.describe().to_string()
            )
            print()

    print("Diagnostics vs performance:")

    for diagnostic in diagnostic_names():
        if diagnostic not in metrics_df.columns:
            continue

        diagnostic_perf = (
            metrics_df[
                [
                    "time",
                    "metric_improvement",
                    diagnostic,
                ]
            ]
            .drop_duplicates(subset="time")
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        if diagnostic_perf.empty:
            continue

        line = f"  {diagnostic}: "

        if (
            len(diagnostic_perf) >= 2
            and diagnostic_perf[diagnostic].nunique() > 1
            and diagnostic_perf["metric_improvement"].nunique() > 1
        ):
            correlation = spearmanr(
                diagnostic_perf[diagnostic],
                diagnostic_perf["metric_improvement"],
                nan_policy="omit",
            )

            line += (
                f"rho={float(correlation.statistic):.4f}, "
                f"p={float(correlation.pvalue):.4g}"
            )
        else:
            line += "insufficient variation"

        print(line)

    print()


if __name__ == "__main__":
    main()
