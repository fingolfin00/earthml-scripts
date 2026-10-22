from pathlib import Path
from contextlib import contextmanager
from time import perf_counter
from itertools import combinations
from typing import Sequence
import torch
from typing import Literal, TypedDict

import numpy as np
import pandas as pd
import xarray as xr

from scipy import stats as scipy_stats
from scipy.special import logsumexp
from scipy.stats import gaussian_kde, spearmanr

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from tqdm.auto import tqdm

from earthml import (
    Settings,
    ClimPeriod,
    MonthlyNormalize,
    Normalize,
    get_and_subset_datasets,
    LeadtimeUnit,
    XarrayDataset,
    get_experiment_configs,
)

from earthml.metrics import (
    ImprovementUnit,
    PeriodReference,
    get_metrics,
    build_metric_improvements,
)

from train import convert_to_xarray, make_train_test_datasets_for_leadtime

# ============================================================================
# Configuration
# ============================================================================

# EXP_NAME = "weather_atmo"
# EXP_NAME = "weather_atmo_variables"
EXP_NAME = "weather_atmo_normalization_ablation"
# EXP_NAME = "weather_atmo_normalization_ablation_72h"
# EXP_NAME = "weather_atmo_one_year_test"

EXPERIMENTS_ROOT = Path(
    f"/work/cmcc/jd19424/ML/MLBC/experiments/{EXP_NAME}"
)

OUTPUT_ROOT = Path(
    f"/work/cmcc/jd19424/ML/MLBC/plots/{EXP_NAME}"
)

INTERPOLATE_ANALYSIS = False # weather

PLOT_DISTRIBUTIONS = True
PLOT_DIAGNOSTIC_SCATTERS = True
PLOT_SPECTRAL_WAVENUMBER_CORRELATION = False

DEBUG_TIMING = False

# ----------------------------------------------------------------------------
# Experiment selection
# ----------------------------------------------------------------------------

FILTERS = dict(
    region_name="ConUS",
    # normalization_mode="channel",
    # normalization_mode="sample",
)

VARIABLES: list[str] | None = [
    # "mslp",
    "t2m",
    # "d2m",
    # "u10",
    # "v10",
    # "tcc",
]

# None -> all experiment leadtimes
LEADTIMES: list[int] | None = None
# LEADTIMES: list[int] | None = [72]

# Optional box used consistently for normalized-input statistics, spectral
# diagnostics, FC/AN/MLFC diagnostic fields, and performance (e.g. RMSE
# improvement). None keeps the corresponding full experiment dimension.
# Best/worst rankings therefore refer to performance inside this box.
LAT_RANGE: tuple[float, float] | None = None
LON_RANGE: tuple[float, float] | None = None

# LAT_RANGE = (30, 45)
# LON_RANGE = (-108, -90)

# ----------------------------------------------------------------------------
# Distribution data selection
# ----------------------------------------------------------------------------

NORMALIZED_DISTRIBUTION = True

DISTRIBUTION_SPLIT_RANGE: list[Literal["train", "val", "test"]] = [
    "train",
    "val",
    "test",
]

DISTRIBUTION_FIELD: Literal[
    "input",
    "target",
    "corrected",
    "fc_error",
    "mlfc_error",
    "ideal_correction",
    "ml_correction",
] = "target"

# ----------------------------------------------------------------------------
# Distribution plotting
# ----------------------------------------------------------------------------

DISTRIBUTION_QUANTITIES = (
    "mean",
    "std",
    "skewness",
    "excess_kurtosis",
    # "mean_zscore",
    "mean_abs_zscore",
    # "std_zscore",
    "std_abs_zscore",
    # "skewness_zscore",
    "skewness_abs_zscore",
    # "excess_kurtosis_zscore",
    "excess_kurtosis_abs_zscore",
)

PLOT_MONTHLY = False

DISTRIBUTION_PLOT: Literal[
    "hist",
    "kde",
    "both",
] = "kde"

BINS = 50
KDE_POINTS = 1024

# None -> automatic scaling
DISTRIBUTION_XLIM_MEAN: tuple[float, float] | None = None
DISTRIBUTION_XLIM_STD: tuple[float, float] | None = None
# DISTRIBUTION_XLIM_MEAN = (-2.0, 2.0)
# DISTRIBUTION_XLIM_STD = (0.0, 2.0)

SPLIT_COLORS = {
    "train": "darkgreen",
    "val":   "darkorchid",
    "test":  "royalblue",
}

# Distribution line transparency
DISTRIBUTION_LINE_ALPHA = 0.9

REFERENCE_LABEL_MODE: Literal["axis", "legend"] = "legend"
LEGEND_LOCATION = "best"
# LEGEND_LOCATION = "upper right"

ANNOTATE_BEST_TIMESTEPS = 0
ANNOTATE_WORST_TIMESTEPS = 0

SELECTED_DATES_TO_ANNOTATE: set[str] | None = None
# SELECTED_DATES_TO_ANNOTATE: set[str] | None = {
#     "2025-08-18 12:00", # B1 mslp 72h
#     "2025-08-17 12:00", # W4 mslp 72h
# }

ANNOTATION_FONTSIZE = 7
ANNOTATION_DATE_FORMAT = "%Y-%m-%d:%H:%M"

DISTRIBUTION_FIGSIZE = (10, 7)
MEAN_STD_FIGSIZE = (10, 8)

DISTRIBUTION_LINEWIDTH = 2

# ----------------------------------------------------------------------------
# Distribution fit
# ----------------------------------------------------------------------------

SHOW_DISTRIBUTION_FIT = True
DISTRIBUTION_FIT_SPLIT: Literal["train", "val", "test"] = "test"

FIT_SELECTION_CRITERION: Literal["aic", "bic"] = "bic"
DISTRIBUTION_FIT: Literal[
    "auto",
    "normal",
    "student_t",
    "lognormal",
    "gamma",
    "gaussian_mixture_2",
] = "normal"

FIT_DISTRIBUTION_SUMMARY = False
PRINT_DISTRIBUTION_SUMMARY = False

DISTRIBUTION_FIT_ALPHA = 0.4
DISTRIBUTION_FIT_LINESTYLE = "--"

# ----------------------------------------------------------------------------
# Reference drawing
# ----------------------------------------------------------------------------

ReferenceMode = Literal[
    "global",
    "median",
    "sample_mean",
    "sample_std",
    "sample_min",
    "sample_max",
    "q05",
    "q25",
    "q75",
    "q95",
    "mean_minus_std",
    "mean_plus_std",
]

ReferenceSpec = tuple[str, ReferenceMode]

ReferenceBandSpec = tuple[
    str,
    ReferenceMode,
    ReferenceMode,
]

DISTRIBUTION_REFERENCES_MEAN: list[ReferenceSpec] = [
    # ("input", "median"),
    # ("input", "sample_mean"),
    # ("input", "global"),
    # ("target", "median"),
    # ("target", "sample_mean"),
    # ("target", "global"),
]

DISTRIBUTION_REFERENCES_STD: list[ReferenceSpec] = [
    # ("input", "median"),
    # ("input", "sample_mean"),
    # ("input", "global"),
    # ("target", "median"),
    # ("target", "sample_mean"),
    # ("target", "global"),
]

DISTRIBUTION_REFERENCE_BANDS_MEAN: list[ReferenceBandSpec] = [
    # ("input", "q25", "q75"),
    # ("target", "q25", "q75"),
]

DISTRIBUTION_REFERENCE_BANDS_STD: list[ReferenceBandSpec] = [
    # ("input", "q25", "q75"),
    # ("target", "q25", "q75"),
]

DISTRIBUTION_REFERENCE_BANDS: list[ReferenceBandSpec] = [
    # ("input", "q25", "q75"),
    # ("target", "q25", "q75"),
]

MEAN_STD_REFERENCES: list[ReferenceSpec] = [
    # ("input", "global"),
    ("target", "global"),
    # ("corrected", "global"),
    # ("input", "sample_mean"),
    # ("target", "sample_mean"),
]

MEAN_STD_REFERENCE_BANDS: list[ReferenceBandSpec] = [
    # ("input", "q25", "q75"),
    # ("target", "q25", "q75"),
]

REFERENCE_BAND_ALPHA = 0.12

REFERENCE_LINESTYLES = {
    "global": "-",
    "median": ":",
    "sample_mean": "--",
}

REFERENCE_LINEWIDTH = {
    "global": 1,
    "median": 1,
    "sample_mean": 1,
}

REFERENCE_ALPHA = {
    "global": 0.8,
    "median": 0.8,
    "sample_mean": 0.8,
}

REFERENCE_MODE_LABELS = {
    "global": "global avg",
    "median": "median",
    "sample_mean": "avg per-sample",
}

# ----------------------------------------------------------------------------
# Performance and coloring
# ----------------------------------------------------------------------------

COLOR_BY_PERFORMANCE = True
PERFORMANCE_SPLIT: Literal["train", "val", "test"] = "test"

PERFORMANCE_METRIC = "rmse"
PERFORMANCE_METRICS_NORMALIZED = False

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
# Values outside the range are clipped to the endpoint colors
PERFORMANCE_COLOR_QUANTILE = 0.90

# Scatter settings
PERFORMANCE_SCATTER_SIZE = 14
PERFORMANCE_SCATTER_ALPHA = 0.90

# Non-performance-colored scatter points use the same neutral appearance
# regardless of whether they belong to train, validation, or test
# SCATTER_COLOR = "0.45"
SCATTER_COLOR = None
SCATTER_SIZE = 6
SCATTER_ALPHA = 0.1

# ----------------------------------------------------------------------------
# Metric vs performance scatter plots
# ----------------------------------------------------------------------------

SCATTER_SPLITS = ("test",)

# Spectral metric settings
SPECTRAL_BATCH_SIZE = 32
SPECTRAL_N_BINS = 30
HIGH_FREQ_THRESHOLD = 0.5

# Annotation and marker size
SELECTED_SAMPLES_TO_ANNOTATE = None
SCATTER_MARKER_SIZE = 5.0

# Diagnostic quantities computed from the same fields used for distributions
DIAGNOSTIC_METRICS = (
    "mean", "std", "skewness", "excess_kurtosis",
    "mean_abs_zscore", "std_abs_zscore",
    "skewness_abs_zscore", "excess_kurtosis_abs_zscore",
    # "correlation", "cosine", "rms_ratio", "orthogonal_rms",
    "mean_product", "rms_laplacian", "mean_abs_laplacian",
    # "spectral_centroid", "high_frequency_power_fraction",
)

DIAGNOSTIC_FIELDS = ("target",)

SHAPE_METRICS = {"mean", "std", "skewness", "excess_kurtosis"}

ZSCORE_METRICS = {
    f"{name}{suffix}"
    for name in SHAPE_METRICS
    for suffix in ("_zscore", "_abs_zscore")
}

SINGLE_FIELD_METRICS = SHAPE_METRICS | ZSCORE_METRICS | {
    "rms_laplacian", "mean_abs_laplacian",
    "low_frequency_power_fraction", "high_frequency_power_fraction",
    "high_low_power_ratio", "spectral_centroid",
}

PAIRWISE_METRICS = {
    "correlation", "cosine", "rms_ratio", "projection",
    "orthogonal_rms", "mean_product",
}

SPECTRAL_METRICS = {
    "low_frequency_power_fraction", "high_frequency_power_fraction",
    "high_low_power_ratio", "spectral_centroid",
}

# Labels
FIELD_LABELS = {
    "input": "FC", "target": "analysis", "corrected": "MLFC",
    "fc_error": "FC error", "mlfc_error": "MLFC error",
    "ideal_correction": "ideal correction", "ml_correction": "ML correction",
}

METRIC_LABELS = {
    "mean": "Spatial mean", "std": "Spatial standard deviation",
    "skewness": "Spatial skewness", "excess_kurtosis": "Spatial excess kurtosis",
    "rms_laplacian": "RMS Laplacian", "mean_abs_laplacian": "Mean |Laplacian|",
    "correlation": "Spatial correlation", "cosine": "Cosine similarity",
    "rms_ratio": "RMS ratio", "projection": "Projection coefficient",
    "orthogonal_rms": "Orthogonal RMS", "mean_product": "Mean product",
    "low_frequency_power_fraction": "Low-frequency power fraction",
    "high_frequency_power_fraction": "High-frequency power fraction",
    "high_low_power_ratio": "High/low frequency power ratio",
    "spectral_centroid": "Normalized spectral centroid",
}

for _base in SHAPE_METRICS:
    METRIC_LABELS[f"{_base}_zscore"] = f"{_base} training z-score"
    METRIC_LABELS[f"{_base}_abs_zscore"] = f"Absolute {_base} training z-score"

# ============================================================================
# Month names
# ============================================================================

MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

# ============================================================================
# Timing context
# ============================================================================

@contextmanager
def timing(label: str):
    """Wall-clock time including any lazy xarray/Dask computation."""
    if not DEBUG_TIMING:
        yield
        return

    start = perf_counter()
    print(f"[TIME] START {label}", flush=True)

    try:
        yield
    finally:
        print(
            f"[TIME] END   {label}: {perf_counter() - start:.2f}s",
            flush=True,
        )

# ============================================================================
# Shared forecast diagnostic helpers
# ============================================================================

def make_normalizer(
    s: Settings,
    dataset: XarrayDataset,
    dim: Literal["x", "y"],
):
    """
    Reproduce input and target normalization used during training.
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

    kwargs = {
        "mode": s.normalization_mode,
        "exclude_channels": exclude_channels,
    }

    if s.normalization_mode == "fixed":
        if dim == "x":
            kwargs["mean"] = s.fixed_input_mean
            kwargs["std"] = s.fixed_input_std
        else:
            kwargs["mean"] = s.fixed_target_mean
            kwargs["std"] = s.fixed_target_std

    return norm_class(
        **kwargs,
    ).fit(
        dataset,
        dim=dim,
    )


def dataset_kwargs_from_settings(s) -> dict:
    return {
        "target_realization_avg": s.target_realization_avg,
        "channel_representation": s.channel_representation,
        "init_period_dim": s.init_period_dim,
        "output_realizations": s.output_realizations,
        "torch_mask": s.torch_mask,
        "fill_nan_value": s.fill_nan_value,
        "sample_target_stats": s.sample_target_stats,
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


def diagnostic_region_label() -> str:
    if LAT_RANGE is None and LON_RANGE is None:
        return "full_domain"

    parts = []

    if LAT_RANGE is not None:
        lo, hi = sorted(map(float, LAT_RANGE))
        parts.append(f"lat_{lo:g}_{hi:g}")

    if LON_RANGE is not None:
        lo, hi = sorted(map(float, LON_RANGE))
        parts.append(f"lon_{lo:g}_{hi:g}")

    return "_".join(parts).replace("-", "m").replace(".", "p")


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
    batch_size: int = 8,
    num_workers: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
    )

    xs = []
    ys = []

    for x, y, _, _ in tqdm(
        loader,
        desc="Transforming samples",
        unit="batch",
        total=len(loader),
        dynamic_ncols=True,
    ):
        xs.append(x)
        ys.append(y)

    return (
        torch.cat(xs, dim=0),
        torch.cat(ys, dim=0),
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
    normalize_input: Normalize | MonthlyNormalize,
    normalize_target: Normalize | MonthlyNormalize,
    s: Settings,
) -> torch.Tensor:
    target_dataset = XarrayDataset(
        input_ds=reference_dataset.input_ds,
        target_ds=target_ds,
        **dataset_kwargs_from_settings(s),
    )

    target_dataset.transform_x = normalize_input
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
        normalize_input=normalize_input,
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
            normalize_input=normalize_input,
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


def get_required_fields() -> set[str]:
    required = set()

    if PLOT_DISTRIBUTIONS:
        required.add(DISTRIBUTION_FIELD)

        for specs in (
            DISTRIBUTION_REFERENCES_MEAN,
            DISTRIBUTION_REFERENCES_STD,
            MEAN_STD_REFERENCES,
        ):
            required.update(field for field, _ in specs)

        for specs in (
            DISTRIBUTION_REFERENCE_BANDS_MEAN,
            DISTRIBUTION_REFERENCE_BANDS_STD,
            DISTRIBUTION_REFERENCE_BANDS,
            MEAN_STD_REFERENCE_BANDS,
        ):
            required.update(field for field, _, _ in specs)

    if PLOT_DIAGNOSTIC_SCATTERS:
        required.update(DIAGNOSTIC_FIELDS)

    return required


def build_fields(
    baseline: xr.DataArray,
    target: xr.DataArray,
    corrected: xr.DataArray,
) -> dict[str, xr.DataArray]:
    required = get_required_fields()

    fields = {
        "input": lambda: baseline,
        "target": lambda: target,
        "corrected": lambda: corrected,
        "fc_error": lambda: baseline - target,
        "mlfc_error": lambda: corrected - target,
        "ideal_correction": lambda: target - baseline,
        "ml_correction": lambda: corrected - baseline,
    }

    return {
        name: make_field()
        for name, make_field in fields.items()
        if name in required
    }


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
# Build field-distribution statistics for one leadtime
# ============================================================================

def get_leadtime_stats(
    s: Settings,
    leadtime: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(
        "=" * 80
    )

    print(
        f"{s.output_name}\n"
        f"leadtime={leadtime} "
        f"{s.leadtime_unit}, "
        f"splits={DISTRIBUTION_SPLIT_RANGE}"
    )
    print(
        "Selected spatial region: "
        f"lat={LAT_RANGE}, lon={LON_RANGE}"
    )

    # ------------------------------------------------------------
    # Generate datasets
    # ------------------------------------------------------------

    with timing(f"{s.var_fc}/{leadtime} dataset construction"):
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

    normalize_input = None
    normalize_target = None

    space = (
        "normalized"
        if NORMALIZED_DISTRIBUTION
        else "physical"
    )

    if NORMALIZED_DISTRIBUTION:
        with timing(f"{s.var_fc}/{leadtime} fit input normalizer"):
            normalize_input = make_normalizer(
                s,
                train_dataset,
                dim="x",
            )

        with timing(f"{s.var_fc}/{leadtime} fit target normalizer"):
            normalize_target = make_normalizer(
                s,
                train_dataset,
                dim="y",
            )

    frames = []
    diagnostic_frames = []
    spectral_frames = []
    for split in DISTRIBUTION_SPLIT_RANGE:
        print(f"[STAGE] {s.var_fc}/{leadtime}: split={split}", flush=True)
        if split == "val" and val_dataset is None:
            raise ValueError(f"Split {split} not available for split strategy {s.split_strategy}")

        with timing(f"{s.var_fc}/{leadtime}/{split} physical data loading"):
            fc, an, mlfc = get_leadtime_xarray_datasets(
                s=s,
                time_range=split,
                leadtime=leadtime,
            )

        if mlfc is None:
            raise ValueError("MLFC dataset is required to calculate metrics improvement")

        if NORMALIZED_DISTRIBUTION:
            assert normalize_input is not None
            assert normalize_target is not None

            with timing(f"{s.var_fc}/{leadtime}/{split} normalized data"):
                distro_fc, distro_an, distro_mlfc = get_normalized_datasets(
                    s=s,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    test_dataset=test_dataset,
                    normalize_input=normalize_input,
                    normalize_target=normalize_target,
                    fc=fc,
                    mlfc=mlfc,
                    time_range=split,
                )
        else:
            distro_fc = fc
            distro_an = an
            distro_mlfc = mlfc


        # Subset
        distro_fc = subset_diagnostic_dataset_region(distro_fc, LAT_RANGE, LON_RANGE)
        distro_an = subset_diagnostic_dataset_region(distro_an, LAT_RANGE, LON_RANGE)
        distro_mlfc = subset_diagnostic_dataset_region(distro_mlfc, LAT_RANGE, LON_RANGE)

        fields = build_fields(
            baseline=distro_fc[s.var_fc],
            target=distro_an[s.var_an],
            corrected=distro_mlfc[s.var_fc],
        )

        # ------------------------------------------------------------
        # Field and reference (target) statistics
        # ------------------------------------------------------------

        with timing(f"{s.var_fc}/{leadtime}/{split} all field statistics"):
            stats = collect_field_stats(
                fields=fields,
                space=space,
                split=split,
                leadtime=leadtime,
            )

        # ------------------------------------------------------------
        # Metric performance (shared between plot families)
        # ------------------------------------------------------------
        metric_df = None
        if (COLOR_BY_PERFORMANCE or PLOT_DIAGNOSTIC_SCATTERS) and split == PERFORMANCE_SPLIT:
            print(
                "Calculating "
                f"{'normalized ' if PERFORMANCE_METRICS_NORMALIZED else ''}"
                f"{PERFORMANCE_METRIC} improvement in split={split}..."
            )

            with timing(f"{s.var_fc}/{leadtime}/{split} performance metrics"):
                metric_df = get_metric_improvement_timeseries(
                    s=s,
                    baseline=fc,
                    target=an,
                    corrected=mlfc,
                    leadtime=leadtime,
                    metric=PERFORMANCE_METRIC,
                    improvement_unit=PERFORMANCE_IMPROVEMENT_UNIT,
                    period_reference=PERFORMANCE_REFERENCE_TIME_PERIOD,
                )

            stats = add_metric_performance(
                stats,
                metric_df,
            )

        frames.append(stats)

        if PLOT_DIAGNOSTIC_SCATTERS and split in SCATTER_SPLITS:
            # The existing field-statistics frame owns mean/std/skewness/kurtosis.
            # Only compute the supplementary pairwise/laplacian diagnostics here.
            if any(m in PAIRWISE_METRICS or m in {"rms_laplacian", "mean_abs_laplacian"}
                   for m in DIAGNOSTIC_METRICS):
                extra_df = get_diagnostic_timeseries(
                    fc=distro_fc, an=distro_an, mlfc=distro_mlfc, s=s,
                )
            else:
                extra_df = pd.DataFrame({"time": pd.to_datetime(distro_fc[distro_fc.earthml.guessed_dims.time].values)})

            if metric_df is not None:
                extra_df = extra_df.merge(
                    metric_df[["time", "metric_improvement"]],
                    on="time", how="left", validate="one_to_one",
                )
            extra_df["split"] = split

            if any(m in SPECTRAL_METRICS for m in DIAGNOSTIC_METRICS) or PLOT_SPECTRAL_WAVENUMBER_CORRELATION:
                with timing(f"{s.var_fc}/{leadtime}/{split} spectral metrics"):
                    scalar_spectral, radial_spectral = get_spectral_metrics(
                        s=s, baseline=distro_fc, target=distro_an,
                        corrected=distro_mlfc, leadtime=leadtime,
                    )
                if not scalar_spectral.empty:
                    extra_df = extra_df.merge(scalar_spectral, on="time", how="left", validate="one_to_one")
                if not radial_spectral.empty:
                    if metric_df is not None:
                        radial_spectral = radial_spectral.merge(
                            metric_df[["time", "metric_improvement"]],
                            on="time", how="left", validate="many_to_one",
                        )
                    radial_spectral["split"] = split
                    spectral_frames.append(radial_spectral)
            diagnostic_frames.append(extra_df)

    stats = pd.concat(frames, ignore_index=True)

    if "train" in DISTRIBUTION_SPLIT_RANGE and (
        any(q.endswith(("_zscore", "_abs_zscore")) for q in DISTRIBUTION_QUANTITIES)
        or any(m in ZSCORE_METRICS for m in DIAGNOSTIC_METRICS)
    ):
        with timing(f"{s.var_fc}/{leadtime} training z-scores"):
            stats = add_training_zscores(
                stats,
                quantities=(
                    "mean",
                    "std",
                    "skewness",
                    "excess_kurtosis",
                ),
            )

    return (
        stats,
        pd.concat(diagnostic_frames, ignore_index=True) if diagnostic_frames else pd.DataFrame(),
        pd.concat(spectral_frames, ignore_index=True) if spectral_frames else pd.DataFrame(),
    )

# ============================================================================
# Main
# ============================================================================

def merge_scatter_statistics(stats: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Reuse calculated per-map moments and training scores; never recalculate fields."""
    if diagnostics.empty:
        return diagnostics
    requested = [metric for metric in DIAGNOSTIC_METRICS
                 if metric in SHAPE_METRICS or metric in ZSCORE_METRICS]
    if not requested:
        return diagnostics
    rows = stats.loc[
        stats["field"].isin(DIAGNOSTIC_FIELDS)
        & stats["split"].isin(SCATTER_SPLITS)
    ]
    metric_columns = {
        m: (m if m in ZSCORE_METRICS else f"{m}_values")
        for m in requested
    }
    wide = rows[["split", "time", "field", *metric_columns.values()]].copy()
    wide = wide.pivot(index=["split", "time"], columns="field", values=list(metric_columns.values()))
    wide.columns = [f"{field}_{next(k for k,v in metric_columns.items() if v == column)}"
                    for column, field in wide.columns]
    wide = wide.reset_index()
    return diagnostics.merge(wide, on=["split", "time"], how="left", validate="one_to_one")

def main() -> None:
    if not DISTRIBUTION_SPLIT_RANGE or len(set(DISTRIBUTION_SPLIT_RANGE)) != len(DISTRIBUTION_SPLIT_RANGE):
        raise ValueError("DISTRIBUTION_SPLIT_RANGE must contain unique selected splits")

    if COLOR_BY_PERFORMANCE and PERFORMANCE_SPLIT not in DISTRIBUTION_SPLIT_RANGE:
        raise ValueError(
            f"PERFORMANCE_SPLIT={PERFORMANCE_SPLIT!r} "
            "is not in DISTRIBUTION_SPLIT_RANGE"
        )

    if SHOW_DISTRIBUTION_FIT and DISTRIBUTION_FIT_SPLIT not in DISTRIBUTION_SPLIT_RANGE:
        raise ValueError(
            f"DISTRIBUTION_FIT_SPLIT={DISTRIBUTION_FIT_SPLIT!r} "
            "is not in DISTRIBUTION_SPLIT_RANGE"
        )

    if not 0 <= DISTRIBUTION_LINE_ALPHA <= 1:
        raise ValueError("DISTRIBUTION_LINE_ALPHA must be between 0 and 1")

    if (
        any(
            quantity.endswith(("_zscore", "_abs_zscore"))
            for quantity in DISTRIBUTION_QUANTITIES
        )
        and "train" not in DISTRIBUTION_SPLIT_RANGE
    ):
        raise ValueError(
            "Training split is required to compute z-score distributions. "
            "Add 'train' to DISTRIBUTION_SPLIT_RANGE."
        )

    if PLOT_DIAGNOSTIC_SCATTERS and not set(SCATTER_SPLITS).issubset(DISTRIBUTION_SPLIT_RANGE):
        raise ValueError("SCATTER_SPLITS must be a subset of DISTRIBUTION_SPLIT_RANGE")
    if (PLOT_DIAGNOSTIC_SCATTERS and any(m in ZSCORE_METRICS for m in DIAGNOSTIC_METRICS)
            and "train" not in DISTRIBUTION_SPLIT_RANGE):
        raise ValueError("Training split required for training-relative diagnostic z-scores")

    settings = get_experiment_configs(
        EXPERIMENTS_ROOT,
        **FILTERS,
    )

    if VARIABLES is not None:
        settings = [
            s for s in settings
            if s.var_fc in VARIABLES
        ]

    if not settings:
        raise RuntimeError("No matching experiments found")

    print(
        f"Found {len(settings)} "
        "matching experiment(s)."
    )

    distribution_space = (
        "normalized"
        if NORMALIZED_DISTRIBUTION
        else "physical"
    )

    for s in settings:
        experiment_output = (
            OUTPUT_ROOT
            / s.output_name
            / "distribution"
            / distribution_space
            / DISTRIBUTION_FIELD
            / "_".join(DISTRIBUTION_SPLIT_RANGE)
            / diagnostic_region_label()
        )

        experiment_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        leadtimes = (
            [
                int(x)
                for x in s.leadtimes
            ]
            if LEADTIMES is None
            else LEADTIMES
        )

        all_stats = []

        for leadtime in leadtimes:
            with timing(f"{s.var_fc}/{leadtime} total data and statistics"):
                stats, diagnostics, spectral_df = get_leadtime_stats(
                    s,
                    leadtime,
                )

            all_stats.append(
                stats
            )

            leadtime_output = (
                experiment_output
                / f"leadtime_{leadtime}"
            )

            leadtime_output.mkdir(
                parents=True,
                exist_ok=True,
            )

            stats.to_csv(
                leadtime_output
                / "field_stats.csv",
                index=False,
            )

            with timing(f"{s.var_fc}/{leadtime} distribution characterization"):
                distribution_params = collect_distribution_parameters(stats)
            if PRINT_DISTRIBUTION_SUMMARY:
                print_distribution_summary(distribution_params)

            if PLOT_DISTRIBUTIONS:
                with timing(f"{s.var_fc}/{leadtime} distribution figures"):
                    plot_stats(
                        stats=stats,
                        output_dir=leadtime_output,
                        s=s,
                    )

            if PLOT_DIAGNOSTIC_SCATTERS and not diagnostics.empty:
                with timing(f"{s.var_fc}/{leadtime} scatter statistics merge"):
                    diagnostics = merge_scatter_statistics(stats, diagnostics)
                scatter_dir = leadtime_output / "performance_scatter"
                scatter_dir.mkdir(exist_ok=True)
                diagnostics.to_csv(scatter_dir / "metrics.csv", index=False)
                for split in SCATTER_SPLITS:
                    subset_for_summary = diagnostics.loc[diagnostics["split"] == split]
                    if "metric_improvement" in subset_for_summary:
                        print(f"Scatter diagnostics: {split}")
                        print_diagnostic_summary(subset_for_summary)
                for split in SCATTER_SPLITS:
                    subset = diagnostics.loc[diagnostics["split"] == split]
                    for diagnostic in diagnostic_names():
                        if diagnostic not in subset or subset[diagnostic].notna().sum() < 2:
                            continue
                        with timing(f"{s.var_fc}/{leadtime}/{split}/{diagnostic} scatter plot"):
                            plot_diagnostic_vs_performance(
                                subset, diagnostic=diagnostic,
                                output_path=scatter_dir / f"{split}_{PERFORMANCE_METRIC}_vs_{diagnostic}.png",
                            )
                    if not spectral_df.empty:
                        sub_spectrum = spectral_df.loc[spectral_df["split"] == split]
                        if not sub_spectrum.empty:
                            sub_spectrum.to_csv(scatter_dir / f"{split}_radial_power_spectra.csv", index=False)
                            if PLOT_SPECTRAL_WAVENUMBER_CORRELATION:
                                for field_name in DIAGNOSTIC_FIELDS:
                                    part = sub_spectrum.loc[sub_spectrum["field"] == field_name]
                                    if not part.empty:
                                        correlation_df = plot_spearman_vs_spatial_wavenumber(
                                            spectral_df=part,
                                            output_path=scatter_dir / f"{split}_{field_name}_spearman_vs_wavenumber.png",
                                            title_suffix=f" ({FIELD_LABELS[field_name]})",
                                        )
                                        if not correlation_df.empty:
                                            correlation_df.to_csv(scatter_dir / f"{split}_{field_name}_spearman_vs_wavenumber.csv", index=False)

            print_summary(stats)

            distribution_params.to_csv(
                leadtime_output / "distribution_parameters.csv",
                index=False,
            )

        all_stats = pd.concat(
            all_stats,
            ignore_index=True,
        )

        all_stats.to_csv(
            experiment_output
            / "stats_all_leadtimes.csv",
            index=False,
        )

# ============================================================================
# Field statistics
# ============================================================================

class FieldStats(TypedDict):
    min_values: xr.DataArray
    max_values: xr.DataArray
    mean_values: xr.DataArray
    std_values: xr.DataArray
    skewness_values: xr.DataArray
    excess_kurtosis_values: xr.DataArray
    min: float
    max: float
    mean: float
    std: float

def compute_stats(
    da: xr.DataArray,
    spatial_dims: tuple[str, str],
) -> FieldStats:
    mean = da.mean(dim=spatial_dims, skipna=True)
    std = da.std(dim=spatial_dims, skipna=True, ddof=0)

    z = (da - mean) / std.where(std > 1e-12)

    skewness = (z ** 3).mean(
        dim=spatial_dims,
        skipna=True,
    )

    excess_kurtosis = (z ** 4).mean(
        dim=spatial_dims,
        skipna=True,
    ) - 3

    return {
        # Per-sample
        "min_values": da.min(
            dim=spatial_dims,
            skipna=True,
        ),
        "max_values": da.max(
            dim=spatial_dims,
            skipna=True,
        ),
        "mean_values": mean,
        "std_values": std,
        "skewness_values": skewness,
        "excess_kurtosis_values": excess_kurtosis,

        # Scalar
        "min": float(
            da.min(skipna=True).values
        ),
        "max": float(
            da.max(skipna=True).values
        ),
        "mean": float(
            da.mean(skipna=True).values
        ),
        "std": float(
            da.std(
                skipna=True,
                ddof=0,
            ).values
        ),
    }

def collect_field_stats(
    fields: dict[str, xr.DataArray],
    *,
    space: Literal["normalized", "physical"],
    split: str,
    leadtime: int,
) -> pd.DataFrame:
    # Common info
    ref_field = next(iter(fields.values()))

    ref_time_dim = ref_field.earthml.guessed_dims.time
    ref_times = pd.to_datetime(
        ref_field[ref_time_dim].values
    )

    common_frame = pd.DataFrame({
        "space": space,
        "split": split,
        "leadtime": leadtime,
        "sample": np.arange(len(ref_times)),
        "time": ref_times,
        "channel": 0,
        "month": ref_times.month,
    })

    # Per-field stats
    frames = []
    for field_name in fields.keys():
        field = fields[field_name]

        time_dim = field.earthml.guessed_dims.time
        lat_dim = field.earthml.guessed_dims.latitude
        lon_dim = field.earthml.guessed_dims.longitude

        if time_dim is None:
            raise ValueError(
                "Could not determine time dimension of field."
            )

        if lat_dim is None or lon_dim is None:
            raise ValueError(
                "Could not determine spatial dimensions of field."
            )

        times = pd.to_datetime(
            field[time_dim].values
        )

        if not np.array_equal(times, ref_times):
            raise ValueError(
                f"Time coordinates do not match for field {field_name!r}"
            )

        required_dims = {
            time_dim,
            lat_dim,
            lon_dim,
        }

        squeeze_dims = [
            dim
            for dim in field.dims
            if dim not in required_dims
            and field.sizes[dim] == 1
        ]

        field = field.squeeze(
            dim=squeeze_dims,
            drop=True,
        )

        extra_dims = [
            dim
            for dim in field.dims
            if dim not in {
                time_dim,
                lat_dim,
                lon_dim,
            }
        ]

        if extra_dims:
            raise ValueError(
                "Field contains unsupported dimensions after "
                f"alignment: {field.dims}. "
                f"Extra dimensions: {extra_dims}"
            )

        spatial_dims = (
            lat_dim,
            lon_dim,
        )

        with timing(f"{split}/{field_name} compute_stats"):
            field_stats = compute_stats(field, spatial_dims)

        times = pd.to_datetime(
            field[time_dim].values
        )

        frame = common_frame.copy()

        frame["field"] = field_name

        # Per-sample statistics (to_numpy may trigger lazy computation)
        with timing(f"{split}/{field_name} materialize per-sample stats"):
            for statistic in (
                "min",
                "max",
                "mean",
                "std",
                "skewness",
                "excess_kurtosis",
            ):
                frame[f"{statistic}_values"] = (
                    field_stats[f"{statistic}_values"]
                    .transpose(time_dim)
                    .to_numpy()
                )
    
        # Whole-split statistics
        for statistic in ("min", "max", "mean", "std"):
            frame[statistic] = field_stats[statistic]

        frames.append(frame)

    return pd.concat(
        frames,
        axis=0,
        ignore_index=True,
    )

# ============================================================================
# Performance
# ============================================================================

def add_metric_performance(
    stats: pd.DataFrame,
    metric_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach metric improvement to matching timestamps in this split."""
    stats = stats.copy()
    stats["time"] = pd.to_datetime(
        stats["time"]
    )

    metric_df = metric_df.copy()
    metric_df["time"] = pd.to_datetime(
        metric_df["time"]
    )

    result = stats.merge(
        metric_df,
        on="time",
        how="left",
        validate="many_to_one",
    )
    return result

def make_performance_norm(
    stats: pd.DataFrame,
) -> mcolors.TwoSlopeNorm | None:
    """
    Build a robust symmetric normalization centered at zero.
    """
    if (
        "metric_improvement"
        not in stats.columns
    ):
        return None
    values = (
        stats.loc[
            stats["split"].isin(DISTRIBUTION_SPLIT_RANGE),
            "metric_improvement",
        ]
        .dropna()
        .to_numpy(dtype=float)
    )
    values = values[
        np.isfinite(values)
    ]
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
# z-score
# ============================================================================

def add_training_zscores(
    stats: pd.DataFrame,
    quantities: tuple[str, ...],
) -> pd.DataFrame:
    stats = stats.copy()

    for (field, space, channel, leadtime), group in stats.groupby(
        ["field", "space", "channel", "leadtime"]
    ):
        train = group[group["split"] == "train"]

        for quantity in quantities:
            column = f"{quantity}_values"
            train_values = train[column].dropna()

            if train_values.empty:
                continue

            mean = train_values.mean()
            std = train_values.std(ddof=0)

            z_column = f"{quantity}_zscore"
            abs_column = f"{quantity}_abs_zscore"

            if not np.isfinite(std) or std <= 1e-12:
                stats.loc[group.index, [z_column, abs_column]] = np.nan
                continue

            z = (group[column] - mean) / std

            stats.loc[group.index, z_column] = z
            stats.loc[group.index, abs_column] = z.abs()

    return stats

# ============================================================================
# Plotting helpers
# ============================================================================

def distribution_range(
    groups: list[np.ndarray],
) -> tuple[float, float]:
    arrays = [
        values
        for values in groups
        if len(values) > 0
    ]

    if not arrays:
        return 0.0, 1.0

    all_values = np.concatenate(
        arrays
    )

    lo = float(
        np.nanmin(all_values)
    )

    hi = float(
        np.nanmax(all_values)
    )

    if np.isclose(lo, hi):
        padding = max(
            abs(lo) * 0.05,
            1e-6,
        )

    else:
        padding = 0.05 * (
            hi - lo
        )

    return (
        lo - padding,
        hi + padding,
    )


def histogram_density_on_grid(
    values: np.ndarray,
    *,
    edges: np.ndarray,
    x_grid: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the histogram density as a piecewise-constant function
    on x_grid.
    Used as the performance-fill envelope in histogram-only mode.
    """
    density, _ = np.histogram(
        values,
        bins=edges,
        density=True,
    )

    indices = np.searchsorted(
        edges,
        x_grid,
        side="right",
    ) - 1

    result = np.zeros_like(
        x_grid,
        dtype=float,
    )

    valid = (
        (indices >= 0)
        & (indices < len(density))
    )

    result[valid] = density[
        indices[valid]
    ]

    return result


def add_performance_colorbar(
    fig,
    ax,
    *,
    cmap,
    norm,
) -> None:
    sm = plt.cm.ScalarMappable(
        norm=norm,
        cmap=cmap,
    )

    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=ax,
        pad=0.02,
    )

    unit = PERFORMANCE_IMPROVEMENT_UNIT

    cbar.set_label(
        f"{PERFORMANCE_METRIC.upper()} "
        f"improvement ({unit})"
    )

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
# Disitribution characterization
# ============================================================================

def fit_gaussian_mixture_2(
    values: np.ndarray,
    *,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> tuple[tuple[float, float, float, float, float], float] | None:
    """Fit a two-Gaussian mixture with deterministic, multi-start EM.

    Parameters: (weight1, mean1, std1, mean2, std2), with mean1 <= mean2.
    Returns parameters and log likelihood. A mixture can also be unimodal;
    fitting two components does not establish two density peaks.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 10 or np.ptp(x) == 0:
        return None

    overall_std = float(np.std(x))
    min_std = max(overall_std * 0.02, np.finfo(float).tiny)
    best = None
    best_ll = -np.inf

    for q1, q2 in ((0.25, 0.75), (0.1, 0.9), (0.4, 0.6)):
        means = np.quantile(x, [q1, q2]).astype(float)
        stds = np.array([overall_std, overall_std], dtype=float)
        weights = np.array([0.5, 0.5], dtype=float)
        prev_ll = -np.inf

        for _ in range(max_iter):
            log_components = np.stack([
                np.log(weights[j]) + scipy_stats.norm.logpdf(
                    x, loc=means[j], scale=stds[j]
                )
                for j in range(2)
            ], axis=1)
            log_total = logsumexp(log_components, axis=1)
            ll = float(np.sum(log_total))
            if not np.isfinite(ll):
                break
            if abs(ll - prev_ll) <= tol * (1 + abs(ll)):
                break
            prev_ll = ll

            resp = np.exp(log_components - log_total[:, None])
            counts = resp.sum(axis=0)
            if np.any(counts < 0.02 * x.size):
                break
            weights = counts / x.size
            means = (resp * x[:, None]).sum(axis=0) / counts
            variances = (
                resp * (x[:, None] - means) ** 2
            ).sum(axis=0) / counts
            stds = np.sqrt(np.maximum(variances, min_std**2))
        else:
            pass

        # Re-evaluate the converged parameters rather than using stale ll.
        if np.any(weights < 0.02) or np.any(stds <= min_std):
            continue
        log_components = np.stack([
            np.log(weights[j]) + scipy_stats.norm.logpdf(
                x, loc=means[j], scale=stds[j]
            )
            for j in range(2)
        ], axis=1)
        ll = float(np.sum(logsumexp(log_components, axis=1)))
        if np.isfinite(ll) and ll > best_ll:
            order = np.argsort(means)
            weights, means, stds = weights[order], means[order], stds[order]
            best = (
                float(weights[0]), float(means[0]), float(stds[0]),
                float(means[1]), float(stds[1]),
            )
            best_ll = ll

    return (best, best_ll) if best is not None else None


def characterize_distribution(
    values: np.ndarray,
    *,
    fit_distributions: bool = True,
    positive_only: bool = False,
) -> dict[str, float | str]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    n = len(values)

    if n < 3:
        raise ValueError("At least 3 values are required.")

    result = {
        "n": n,
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "median": float(np.median(values)),
        "skewness": float(scipy_stats.skew(values)),
        "excess_kurtosis": float(scipy_stats.kurtosis(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
    }

    result["iqr"] = result["q75"] - result["q25"]
    # Central 90% interval
    result["width_90"] = result["q95"] - result["q05"]

    # Full Width at Half Maximum (FWHM)
    result["fwhm"] = float("nan")

    if np.std(values) > 0:
        kde = gaussian_kde(values)

        bandwidth = np.sqrt(kde.covariance[0, 0])

        x_grid = np.linspace(
            values.min() - 3 * bandwidth,
            values.max() + 3 * bandwidth,
            KDE_POINTS,
        )

        density = kde(x_grid)
        above = density >= density.max() / 2

        if np.any(above):
            result["fwhm"] = float(
                x_grid[above][-1] - x_grid[above][0]
            )

    if not fit_distributions or np.std(values) == 0:
        return result

    candidates = {
        "normal": scipy_stats.norm,
        "student_t": scipy_stats.t,
    }

    if positive_only and np.all(values > 0):
        candidates.update({
            "lognormal": scipy_stats.lognorm,
            "gamma": scipy_stats.gamma,
        })

    best_score = np.inf
    best_name = None

    for name, distribution in candidates.items():
        try:
            if name in {"lognormal", "gamma"}:
                params = distribution.fit(values, floc=0)
                k = len(params) - 1
            else:
                params = distribution.fit(values)
                k = len(params)

            log_likelihood = np.sum(
                distribution.logpdf(values, *params)
            )

            if not np.isfinite(log_likelihood):
                continue

            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood

            result[f"{name}_aic"] = float(aic)
            result[f"{name}_bic"] = float(bic)

            for i, parameter in enumerate(params):
                result[f"{name}_param_{i}"] = float(parameter)

            score = aic if FIT_SELECTION_CRITERION == "aic" else bic
            if score < best_score:
                best_score = score
                best_name = name

        except (ValueError, RuntimeError, FloatingPointError):
            continue

    # Two-component Gaussian mixture: five free parameters.
    mixture = fit_gaussian_mixture_2(values)
    if mixture is not None:
        mixture_params, mixture_ll = mixture
        mixture_aic = 2 * 5 - 2 * mixture_ll
        mixture_bic = 5 * np.log(n) - 2 * mixture_ll
        result["gaussian_mixture_2_aic"] = float(mixture_aic)
        result["gaussian_mixture_2_bic"] = float(mixture_bic)
        for i, parameter in enumerate(mixture_params):
            result[f"gaussian_mixture_2_param_{i}"] = parameter
        score = (mixture_aic if FIT_SELECTION_CRITERION == "aic"
                 else mixture_bic)
        if score < best_score:
            best_score = score
            best_name = "gaussian_mixture_2"

    result["best_fit"] = best_name or "none"
    result["fit_selection_criterion"] = FIT_SELECTION_CRITERION

    return result


def print_distribution_summary(distribution_params: pd.DataFrame) -> None:
    """Summarize candidate fits for each field and per-sample statistic."""
    names = ("normal", "student_t", "lognormal", "gamma", "gaussian_mixture_2")
    print("\n" + "=" * 80)
    print(f"DISTRIBUTION FIT SUMMARY (selected by {FIT_SELECTION_CRITERION.upper()})")
    print("=" * 80)
    for _, row in distribution_params.iterrows():
        header = (f"{row['field']} / {row['quantity']} "
                  f"[{row['space']}, {row['split']}, leadtime={row['leadtime']}, "
                  f"channel={row['channel']}, n={row['n']}]")
        print("\n" + header)
        print(f"  mean={row['mean']:.4g}, std={row['std']:.4g}, "
              f"skew={row['skewness']:.3f}, excess kurtosis={row['excess_kurtosis']:.3f}")
        candidates = []
        for name in names:
            aic, bic = row.get(f"{name}_aic"), row.get(f"{name}_bic")
            if pd.notna(aic) and pd.notna(bic):
                candidates.append((name, float(aic), float(bic)))
        candidates.sort(key=lambda item: item[1 if FIT_SELECTION_CRITERION == "aic" else 2])
        if not candidates:
            print("  No valid parametric fit")
            continue
        for name, aic, bic in candidates:
            marker = "*" if name == row.get("best_fit") else " "
            print(f" {marker} {name:20s} AIC={aic:11.2f} BIC={bic:11.2f}")
        print(f"  * Best by {FIT_SELECTION_CRITERION.upper()}: {row.get('best_fit', 'none')}")
    print()


def collect_distribution_parameters(
    stats: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    group_columns = [
        "space",
        "split",
        "leadtime",
        "field",
        "channel",
    ]

    for keys, group in stats.groupby(group_columns):
        metadata = dict(zip(group_columns, keys))

        for quantity in (
            "min",
            "max",
            "mean",
            "std",
            "skewness",
            "excess_kurtosis",
        ):
            values = group[
                f"{quantity}_values"
            ].to_numpy(dtype=float)

            params = characterize_distribution(
                values,
                fit_distributions=FIT_DISTRIBUTION_SUMMARY,
                positive_only=(quantity == "std"),
            )

            records.append({
                **metadata,
                "quantity": quantity,
                **params,
            })

    return pd.DataFrame(records)

# ============================================================================
# Reference drawing helpers
# ============================================================================

def get_reference_value(
    stats: pd.DataFrame,
    *,
    field: str,
    space: str,
    split: str,
    channel: int,
    quantity: Literal[
        "min",
        "max",
        "mean",
        "std",
        "skewness",
        "excess_kurtosis",
    ],
    mode: ReferenceMode,
) -> float:
    ref_df = stats.loc[
        (stats["field"] == field)
        & (stats["space"] == space)
        & (stats["split"] == split)
        & (stats["channel"] == channel)
    ]

    if ref_df.empty:
        raise ValueError(
            f"No reference statistics for "
            f"field={field!r}, space={space!r}, "
            f"split={split!r}, channel={channel}"
        )

    if mode == "global":
        return float(ref_df[quantity].iloc[0])

    values = (
        ref_df[f"{quantity}_values"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if values.empty:
        return float("nan")

    mean = float(values.mean())
    std = float(values.std(ddof=0))

    if mode == "median":
        return float(values.median())

    if mode == "sample_mean":
        return mean

    if mode == "sample_std":
        return std

    if mode == "sample_min":
        return float(values.min())

    if mode == "sample_max":
        return float(values.max())

    if mode.startswith("q"):
        quantile = int(mode[1:]) / 100
        return float(values.quantile(quantile))

    if mode == "mean_minus_std":
        return mean - std

    if mode == "mean_plus_std":
        return mean + std

    raise ValueError(
        f"Unsupported reference mode={mode!r}"
    )


def get_reference_band(
    stats: pd.DataFrame,
    *,
    field: str,
    space: str,
    split: str,
    channel: int,
    quantity: Literal[
        "min",
        "max",
        "mean",
        "std",
        "skewness",
        "excess_kurtosis",
    ],
    lower_mode: ReferenceMode,
    upper_mode: ReferenceMode,
) -> tuple[float, float]:
    values = [
        get_reference_value(
            stats,
            field=field,
            space=space,
            split=split,
            channel=channel,
            quantity=quantity,
            mode=mode,
        )
        for mode in (lower_mode, upper_mode)
    ]

    return min(values), max(values)


def reference_style(
    split: str,
    mode: ReferenceMode,
) -> dict:
    return {
        "color": SPLIT_COLORS[split],
        "linestyle": REFERENCE_LINESTYLES.get(mode, "-."),
        "linewidth": REFERENCE_LINEWIDTH.get(mode, 1),
        "alpha": REFERENCE_ALPHA.get(mode, 0.8),
    }


def reference_label(
    field: str,
    mode: ReferenceMode,
    value: float | None = None,
) -> str:
    label = (
        f"{field} "
        f"{REFERENCE_MODE_LABELS.get(mode, mode)}"
    )

    if value is not None:
        label += f": {value:.3f}"

    return label


def draw_distribution_reference_bands(
    ax,
    stats: pd.DataFrame,
    *,
    space: str,
    split: str,
    channel: int,
    quantity: Literal["min", "max", "mean", "std"],
    bands: list[ReferenceBandSpec],
) -> list[Patch]:
    handles = []

    for field, lower_mode, upper_mode in bands:
        lower, upper = get_reference_band(
            stats,
            field=field,
            space=space,
            split=split,
            channel=channel,
            quantity=quantity,
            lower_mode=lower_mode,
            upper_mode=upper_mode,
        )

        if not np.isfinite([lower, upper]).all():
            continue

        color = SPLIT_COLORS[split]

        label = (
            f"{split} {field} {lower_mode}–{upper_mode}: "
            f"[{lower:.3f}, {upper:.3f}]"
        )

        ax.axvspan(
            lower,
            upper,
            color=color,
            alpha=REFERENCE_BAND_ALPHA,
            linewidth=0,
            zorder=0,
        )

        handles.append(
            Patch(
                facecolor=color,
                alpha=REFERENCE_BAND_ALPHA,
                label=label,
            )
        )

    return handles


def draw_mean_std_reference_bands(
    ax,
    stats: pd.DataFrame,
    *,
    space: str,
    split: str,
    channel: int,
) -> list[Patch]:
    handles = []

    for field, lower_mode, upper_mode in MEAN_STD_REFERENCE_BANDS:
        x0, x1 = get_reference_band(
            stats,
            field=field,
            space=space,
            split=split,
            channel=channel,
            quantity="mean",
            lower_mode=lower_mode,
            upper_mode=upper_mode,
        )

        y0, y1 = get_reference_band(
            stats,
            field=field,
            space=space,
            split=split,
            channel=channel,
            quantity="std",
            lower_mode=lower_mode,
            upper_mode=upper_mode,
        )

        if not np.isfinite([x0, x1, y0, y1]).all():
            continue

        color = SPLIT_COLORS[split]

        label = f"{split} {field} {lower_mode}–{upper_mode}"

        ax.add_patch(
            Rectangle(
                (x0, y0),
                width=x1 - x0,
                height=y1 - y0,
                facecolor=color,
                edgecolor=color,
                alpha=REFERENCE_BAND_ALPHA,
                linewidth=0.8,
                zorder=1,
            )
        )

        handles.append(
            Patch(
                facecolor=color,
                alpha=REFERENCE_BAND_ALPHA,
                label=label,
            )
        )

    return handles

# ============================================================================
# Distribution plot
# ============================================================================

def plot_distribution(
    stats: pd.DataFrame,
    *,
    field: str,
    space: Literal["normalized", "physical"],
    quantity: Literal["min", "max", "mean", "std"],
    channel: int,
    output_path: Path,
    plot_type: Literal["hist", "kde", "both"] = "both",
    title_suffix: str = "",
    points_to_annotate: pd.DataFrame | None = None,
) -> None:
    if plot_type not in {"hist", "kde", "both"}:
        raise ValueError(
            f"Unsupported plot_type={plot_type!r}"
        )

    # ------------------------------------------------------------
    # Select field, space and channel
    # ------------------------------------------------------------

    channel_df = stats[
        (stats["channel"] == channel)
        & (stats["field"] == field)
        & (stats["space"] == space)
    ]

    # Per-sample spatial statistic
    quantity_column = (
        quantity
        if quantity.endswith(("_zscore", "_abs_zscore"))
        else f"{quantity}_values"
    )

    groups = {
        split: (
            channel_df.loc[channel_df["split"] == split, quantity_column]
            .replace([np.inf, -np.inf], np.nan)
            .dropna().to_numpy(dtype=float)
        )
        for split in DISTRIBUTION_SPLIT_RANGE
    }

    lo, hi = distribution_range(
        list(groups.values())
    )

    edges = np.linspace(
        lo,
        hi,
        BINS + 1,
    )

    n_density_points = KDE_POINTS

    x_grid = np.linspace(
        lo,
        hi,
        n_density_points,
    )

    fig, ax = plt.subplots(
        figsize=DISTRIBUTION_FIGSIZE,
    )

    performance_norm = (
        make_performance_norm(
            channel_df
        )
        if COLOR_BY_PERFORMANCE
        else None
    )

    performance_cmap = plt.get_cmap(
        PERFORMANCE_CMAP
    )

    reference_handles = []
    extra_handles = []
    summary_parts = []
    for split in DISTRIBUTION_SPLIT_RANGE:
        split_density_for_annotations = None

        # ------------------------------------------------------------
        # Plot selected split
        # ------------------------------------------------------------

        values = groups[split]
        n_values = len(values)
        if n_values == 0:
            continue

        color = SPLIT_COLORS[split]

        median = float(
            np.median(values)
        )

        # --------------------------------------------------------
        # Histogram
        # --------------------------------------------------------

        if plot_type in {
            "hist",
            "both",
        }:
            ax.hist(
                values,
                bins=edges,
                histtype="step",
                linewidth=1.5,
                density=True,
                alpha=DISTRIBUTION_LINE_ALPHA,
                color=color,
                label=f"{split} histogram",
                zorder=4,
            )

        # --------------------------------------------------------
        # KDE
        # --------------------------------------------------------

        kde_density = None

        if (
            plot_type in {
                "kde",
                "both",
            }
            and n_values > 1
            and not np.isclose(
                np.std(values),
                0.0,
            )
        ):
            kde = gaussian_kde(values)
            kde_density = kde(x_grid)

            ax.plot(
                x_grid,
                kde_density,
                linewidth=DISTRIBUTION_LINEWIDTH,
                color=color,
                alpha=DISTRIBUTION_LINE_ALPHA,
                label=f"{split} distribution",
                zorder=5,
            )

        if kde_density is not None:
            split_density_for_annotations = kde_density
        elif plot_type == "hist":
            split_density_for_annotations = (
                histogram_density_on_grid(
                    values,
                    edges=edges,
                    x_grid=x_grid,
                )
            )

        # ------------------------------------------------------------
        # Performance-colored density bands + correlation
        # ------------------------------------------------------------

        split_test_perf = pd.DataFrame()

        if (
            COLOR_BY_PERFORMANCE
            and split == PERFORMANCE_SPLIT
            and "metric_improvement" in channel_df.columns
        ):
            split_test_perf = (
                channel_df.loc[
                    channel_df["split"] == split,
                    [quantity_column, "metric_improvement"],
                ]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .copy()
            )

        if (
            not split_test_perf.empty
            and performance_norm is not None
            and split_density_for_annotations is not None
        ):
            sample_values = split_test_perf[
                quantity_column
            ].to_numpy(dtype=float)

            sample_performance = split_test_perf[
                "metric_improvement"
            ].to_numpy(dtype=float)

            # --------------------------------------------------------
            # Sort samples by x coordinate
            # --------------------------------------------------------

            order = np.argsort(sample_values)

            sample_values = sample_values[order]
            sample_performance = sample_performance[order]

            # --------------------------------------------------------
            # Deal with samples having exactly the same x value
            #
            # Multiple samples at the same x coordinate cannot occupy
            # separate horizontal bands. Average their performance.
            # --------------------------------------------------------

            unique_values, inverse = np.unique(
                sample_values,
                return_inverse=True,
            )

            # Average performance for coincident sample values without
            # repeatedly scanning the full sample array.
            counts = np.bincount(inverse)
            unique_performance = (
                np.bincount(inverse, weights=sample_performance) / counts
            )

            # --------------------------------------------------------
            # Construct sample-centered band edges
            # --------------------------------------------------------

            if len(unique_values) == 1:
                # Degenerate case: one unique sample value.
                # Give it a small band around its x position.
                width = (
                    hi - lo
                ) / max(KDE_POINTS, 1)

                sample_edges = np.array([
                    unique_values[0] - 0.5 * width,
                    unique_values[0] + 0.5 * width,
                ])

            else:
                midpoints = (
                    unique_values[:-1]
                    + unique_values[1:]
                ) / 2.0

                sample_edges = np.empty(
                    len(unique_values) + 1,
                    dtype=float,
                )

                sample_edges[1:-1] = midpoints

                # Extend first and last bands symmetrically.
                sample_edges[0] = (
                    unique_values[0]
                    - (
                        midpoints[0]
                        - unique_values[0]
                    )
                )

                sample_edges[-1] = (
                    unique_values[-1]
                    + (
                        unique_values[-1]
                        - midpoints[-1]
                    )
                )

                # Do not color outside the plotted distribution range.
                sample_edges[0] = max(
                    sample_edges[0],
                    lo,
                )
                sample_edges[-1] = min(
                    sample_edges[-1],
                    hi,
                )

            # --------------------------------------------------------
            # Color density under curve using a single QuadMesh.
            # Include both the KDE grid and the exact band boundaries:
            # this preserves the piecewise-linear density outline and
            # the original sample-centered color transitions.
            # --------------------------------------------------------

            mesh_x = np.unique(np.concatenate((
                x_grid,
                np.clip(sample_edges, lo, hi),
            )))
            mesh_y = np.interp(
                mesh_x,
                x_grid,
                split_density_for_annotations,
            )

            # Pick one sample band for each mesh cell midpoint.
            cell_midpoints = 0.5 * (mesh_x[:-1] + mesh_x[1:])
            band_indices = np.searchsorted(
                sample_edges,
                cell_midpoints,
                side="right",
            ) - 1
            band_indices = np.clip(
                band_indices, 0, len(unique_performance) - 1,
            )

            ax.pcolormesh(
                np.vstack((mesh_x, mesh_x)),
                np.vstack((np.zeros_like(mesh_y), mesh_y)),
                unique_performance[band_indices][None, :],
                cmap=performance_cmap,
                norm=performance_norm,
                shading="flat",
                alpha=DISTRIBUTION_LINE_ALPHA,
                edgecolors="none",
                zorder=1,
            )

        # ------------------------------------------------------------
        # Correlation
        # ------------------------------------------------------------

        correlation_text = ""

        if len(split_test_perf) >= 2:
            x = split_test_perf[
                quantity_column
            ].to_numpy(dtype=float)

            y = split_test_perf[
                "metric_improvement"
            ].to_numpy(dtype=float)

            rho, p_value = spearmanr(
                x,
                y,
            )

            if np.isfinite(rho):
                correlation_text = (
                    rf"Spearman $\rho$ "
                    rf"({quantity}-"
                    rf"{PERFORMANCE_IMPROVEMENT_UNIT}"
                    rf"{PERFORMANCE_METRIC.upper()}) "
                    rf"= {rho:.3f}"
                )

                if np.isfinite(p_value):
                    correlation_text += (
                        f" (p={p_value:.3g})"
                    )

        # ------------------------------------------------------------
        # Best/worst timestep annotations
        # ------------------------------------------------------------

        if (
            points_to_annotate is not None
            and split_density_for_annotations is not None
        ):
            occupied_annotations = []

            for _, row in points_to_annotate.iterrows():
                time = pd.Timestamp(row["time"])

                timestep_rows = channel_df[
                    (channel_df["split"] == split)
                    & (channel_df["time"] == time)
                ]

                if timestep_rows.empty:
                    continue

                x = float(
                    timestep_rows[quantity_column].mean()
                )

                y = float(
                    np.interp(
                        x,
                        x_grid,
                        split_density_for_annotations,
                    )
                )

                if not (
                    np.isfinite(x)
                    and np.isfinite(y)
                ):
                    continue

                if (
                    performance_norm is not None
                    and pd.notna(row.get("metric_improvement"))
                ):
                    perf = float(
                        row["metric_improvement"]
                    )
                    marker_color = performance_color(
                        perf,
                        cmap=performance_cmap,
                        norm=performance_norm,
                    )
                else:
                    marker_color = "0.45"

                ax.plot(
                    x,
                    y,
                    marker="o",
                    markersize=4.0,
                    markerfacecolor=marker_color,
                    markeredgecolor="black",
                    markeredgewidth=0.6,
                    zorder=7,
                )

                annotate_nonoverlapping(
                    ax,
                    row["_annotation_label"],
                    xy=(x, y),
                    occupied=occupied_annotations,
                )

        # ------------------------------------------------------------
        # Reference values and lines
        # ------------------------------------------------------------

        if quantity == "mean":
            reference_specs = DISTRIBUTION_REFERENCES_MEAN
            reference_bands = DISTRIBUTION_REFERENCE_BANDS_MEAN
        elif quantity == "std":
            reference_specs = DISTRIBUTION_REFERENCES_STD
            reference_bands = DISTRIBUTION_REFERENCE_BANDS_STD
        else:
            reference_specs = []
            reference_bands = []

        reference_handles.extend(draw_distribution_reference_bands(
            ax,
            stats,
            space=space,
            split=split,
            channel=channel,
            quantity=quantity,
            bands=reference_bands,
        ))


        for ref_field, ref_mode in reference_specs:
            value = get_reference_value(
                stats,
                field=ref_field,
                space=space,
                split=split,
                channel=channel,
                quantity=quantity,
                mode=ref_mode,
            )

            style = reference_style(
                split,
                ref_mode,
            )

            label = f"{split}: " + reference_label(
                ref_field,
                ref_mode,
            )

            ax.axvline(
                value,
                **style,
                zorder=3,
            )


            reference_handles.append(
                Line2D(
                    [0],
                    [0],
                    label=label,
                    **style,
                )
            )

        # ------------------------------------------------------------
        # Theoretical densities
        # ------------------------------------------------------------

        show_fit = (
            SHOW_DISTRIBUTION_FIT
            and split == DISTRIBUTION_FIT_SPLIT
        )

        params = (
            characterize_distribution(
                values,
                fit_distributions=(
                    show_fit and DISTRIBUTION_FIT != "normal"
                ),
                positive_only=(quantity == "std"),
            )
            if n_values >= 3
            else {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "skewness": float("nan"),
                "excess_kurtosis": float("nan"),
            }
        )

        best_fit = (
            params.get("best_fit")
            if DISTRIBUTION_FIT == "auto"
            else DISTRIBUTION_FIT
        )

        if (
            show_fit
            and best_fit not in (None, "none", "normal")
            and f"{best_fit}_aic" not in params
        ):
            raise ValueError(
                f"Distribution {best_fit!r} could not be fitted "
                f"for split={split!r}, quantity={quantity!r}"
            )

        if show_fit and best_fit == "normal":
            params["normal_param_0"] = params["mean"]
            params["normal_param_1"] = params["std"]

        distributions = {
            "normal": scipy_stats.norm,
            "student_t": scipy_stats.t,
            "lognormal": scipy_stats.lognorm,
            "gamma": scipy_stats.gamma,
        }

        distribution_labels = {
            "normal": "Gaussian",
            "student_t": "Student's t",
            "lognormal": "Lognormal",
            "gamma": "Gamma",
            "gaussian_mixture_2": "2-Gaussian mixture",
        }

        if show_fit and best_fit == "gaussian_mixture_2":
            w, mu1, sd1, mu2, sd2 = (
                params[f"gaussian_mixture_2_param_{i}"]
                for i in range(5)
            )
            fitted_density = (
                w * scipy_stats.norm.pdf(x_grid, mu1, sd1)
                + (1 - w) * scipy_stats.norm.pdf(x_grid, mu2, sd2)
            )
        elif show_fit and best_fit in distributions:
            distribution = distributions[best_fit]
            n_params = (
                len(distribution.shapes.split(",")) + 2
                if distribution.shapes else 2
            )
            fitted_params = tuple(
                params[f"{best_fit}_param_{i}"]
                for i in range(n_params)
            )
            fitted_density = distribution.pdf(x_grid, *fitted_params)
        else:
            fitted_density = None

        if fitted_density is not None and np.all(np.isfinite(fitted_density)):
            fit_label = (
                f"{distribution_labels[best_fit]} fit"
                # f" (AIC={params[f'{best_fit}_aic']:.1f}, "
                # f"BIC={params[f'{best_fit}_bic']:.1f})"
            )
            ax.plot(
                x_grid,
                fitted_density,
                linestyle=DISTRIBUTION_FIT_LINESTYLE,
                linewidth=DISTRIBUTION_LINEWIDTH,
                color=color,
                alpha=DISTRIBUTION_FIT_ALPHA,
                label=f"{split}: {fit_label}",
                zorder=6,
            )


        shape_text = (
            rf"$\mu={params['mean']:.3f}$, "
            rf"$\sigma={params['std']:.3f}$"
            "\n"
            rf"IQR $={params.get('iqr', float('nan')):.3f}$, "
            rf"$W_{{90}}={params.get('width_90', float('nan')):.3f}$, "
            rf"FWHM $={params.get('fwhm', float('nan')):.3f}$"
            "\n"
            rf"Skewness $={params['skewness']:.3f}$, "
            rf"Excess kurtosis $={params['excess_kurtosis']:.3f}$"
        )

        extra_handles.append(Line2D([], [], linestyle="none", label=f"{split}: " + shape_text))
        if show_fit and best_fit == "gaussian_mixture_2":
            w, mu1, sd1, mu2, sd2 = (
                params[f"gaussian_mixture_2_param_{i}"] for i in range(5)
            )
            extra_handles.append(Line2D([], [], linestyle="none", label=(
                f"{split}: " + rf"Mixture: $w={w:.2f}$, "
                + rf"$\mu_1={mu1:.2f}$, $\mu_2={mu2:.2f}$"
            )))
        if show_fit and best_fit == "student_t":
            nu = params["student_t_param_0"]
            extra_handles.append(Line2D([], [], linestyle="none", label=(
                f"{split}: " + rf"Student's t $\nu={nu:.2f}$"
            )))

        correlation_text = "" # TODO disabled correlation text for now, we should place it somewhere else
        summary_parts.append(f"{split}: n={n_values}" + (f" ({correlation_text})" if correlation_text else ""))

    # ------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------

    space = (
        "normalized"
        if NORMALIZED_DISTRIBUTION
        else "physical"
    )

    ax.set_xlabel(
        f"Spatial {quantity} of {space} {field}"
    )

    ax.set_ylabel("Probability density")

    ax.set_title(
        f"{space} {field} distribution "
        f"({quantity}) {title_suffix}\n"
        + " · ".join(summary_parts)
    )

    # ------------------------------------------------------------
    # Distribution legend
    # ------------------------------------------------------------

    curve_handles, curve_labels = ax.get_legend_handles_labels()
    legend_handles = list(dict(zip(curve_labels, curve_handles)).values())

    if REFERENCE_LABEL_MODE == "legend":
        legend_handles.extend(reference_handles)
    elif REFERENCE_LABEL_MODE == "axis":
        legend_handles.extend(
            handle for handle in reference_handles
            if isinstance(handle, Patch)
        )
    else:
        raise ValueError(
            f"Unsupported REFERENCE_LABEL_MODE={REFERENCE_LABEL_MODE!r}"
        )

    legend_handles.extend(extra_handles)

    ax.legend(
        handles=legend_handles,
        loc=LEGEND_LOCATION,
        fontsize=7,
        frameon=True,
    )

    # ------------------------------------------------------------
    # Optional fixed X-axis limits
    # ------------------------------------------------------------

    xlim = {
        "mean": DISTRIBUTION_XLIM_MEAN,
        "std": DISTRIBUTION_XLIM_STD,
    }.get(quantity)

    if xlim is not None:
        ax.set_xlim(*xlim)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)

# ============================================================================
# Mean vs std
# ============================================================================

def plot_mean_vs_std(
    stats: pd.DataFrame,
    *,
    field: str,
    space: Literal["normalized", "physical"],
    channel: int,
    output_path: Path,
    title_suffix: str = "",
    points_to_annotate: pd.DataFrame | None = None,
) -> None:
    channel_df = stats[
        (stats["channel"] == channel)
        & (stats["field"] == field)
        & (stats["space"] == space)
    ]

    if channel_df.empty:
        return

    fig, ax = plt.subplots(
        figsize=MEAN_STD_FIGSIZE,
    )

    performance_norm = (
        make_performance_norm(
            channel_df
        )
        if COLOR_BY_PERFORMANCE
        else None
    )

    performance_cmap = plt.get_cmap(
        PERFORMANCE_CMAP
    )

    plotted_performance = False

    # ------------------------------------------------------------
    # Selected splits
    # ------------------------------------------------------------

    occupied_annotations = []
    all_band_handles = []
    reference_handles = []
    for split in DISTRIBUTION_SPLIT_RANGE:
        plot_df = channel_df[channel_df["split"] == split]
        if plot_df.empty:
            continue

        has_performance = (
            COLOR_BY_PERFORMANCE
            and split == PERFORMANCE_SPLIT
            and performance_norm is not None
            and "metric_improvement" in plot_df.columns
        )

        if has_performance:
            valid_perf = (
                plot_df["metric_improvement"]
                .notna()
            )
        else:
            valid_perf = pd.Series(
                False,
                index=plot_df.index,
                dtype=bool,
            )

        # --------------------------------------------------------
        # Points without performance data
        # Same neutral style for train/val/test.
        # --------------------------------------------------------

        ordinary = plot_df[
            ~valid_perf
        ]

        if not ordinary.empty:
            ax.scatter(
                ordinary["mean_values"],
                ordinary["std_values"],
                s=SCATTER_SIZE,
                alpha=SCATTER_ALPHA,
                color=(
                    SCATTER_COLOR
                    if SCATTER_COLOR is not None
                    else SPLIT_COLORS[split]
                ),
                label=split,
                zorder=2,
            )

        # --------------------------------------------------------
        # Points with performance data, color encodes performance
        # --------------------------------------------------------

        performance_df = plot_df[
            valid_perf
        ]

        if not performance_df.empty:
            perf_values = performance_df[
                "metric_improvement"
            ].to_numpy(dtype=float)

            point_colors = performance_color(
                perf_values,
                cmap=performance_cmap,
                norm=performance_norm,
            )

            ax.scatter(
                performance_df["mean_values"],
                performance_df["std_values"],
                c=point_colors,
                s=PERFORMANCE_SCATTER_SIZE,
                alpha=PERFORMANCE_SCATTER_ALPHA,
                label=f"{split} (performance)",
                zorder=4,
            )
            plotted_performance = True

        # ------------------------------------------------------------
        # Best/worst test-timestep annotations
        # ------------------------------------------------------------

        if points_to_annotate is not None:

            for _, row in points_to_annotate.iterrows():
                time = pd.Timestamp(row["time"])

                timestep_rows = channel_df[
                    (channel_df["split"] == split)
                    & (channel_df["time"] == time)
                ]

                if timestep_rows.empty:
                    continue

                x = float(
                    timestep_rows["mean_values"].mean()
                )
                y = float(
                    timestep_rows["std_values"].mean()
                )

                if not (
                    np.isfinite(x)
                    and np.isfinite(y)
                ):
                    continue

                # Thin black ring around annotated points
                ax.scatter(
                    [x],
                    [y],
                    s=PERFORMANCE_SCATTER_SIZE*2,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=7,
                )

                annotate_nonoverlapping(
                    ax,
                    row["_annotation_label"],
                    xy=(x, y),
                    occupied=occupied_annotations,
                )

        # ------------------------------------------------------------
        # Reference bands and points
        # ------------------------------------------------------------

        band_handles = draw_mean_std_reference_bands(
            ax,
            stats,
            space=space,
            split=split,
            channel=channel,
        )

        all_band_handles.extend(band_handles)

        for ref_field, ref_mode in MEAN_STD_REFERENCES:
            mean_value = get_reference_value(
                stats,
                field=ref_field,
                space=space,
                split=split,
                channel=channel,
                quantity="mean",
                mode=ref_mode,
            )

            std_value = get_reference_value(
                stats,
                field=ref_field,
                space=space,
                split=split,
                channel=channel,
                quantity="std",
                mode=ref_mode,
            )

            style = reference_style(
                split,
                ref_mode,
            )

            label = reference_label(
                ref_field,
                ref_mode,
            )

            ax.axvline(
                mean_value,
                **style,
                zorder=2,
            )

            ax.axhline(
                std_value,
                **style,
                zorder=2,
            )

            ax.scatter(
                [mean_value],
                [std_value],
                marker="o",
                s=PERFORMANCE_SCATTER_SIZE * 2,
                edgecolors="black",
                color=style["color"],
                zorder=8,
            )

            reference_handles.append(
                Line2D(
                    [], [],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker="o",
                    markeredgecolor="black",
                    markersize=5,
                    linewidth=style["linewidth"],
                    alpha=style["alpha"],
                    label=(
                        f"{split} {label}: "
                        f"mean={mean_value:.3f}, "
                        f"std={std_value:.3f}"
                    ),
                )
            )

    if plotted_performance:
        add_performance_colorbar(fig, ax, cmap=performance_cmap, norm=performance_norm)

    handles, labels = ax.get_legend_handles_labels()

    handles.extend(all_band_handles)
    handles.extend(reference_handles)

    if handles:
        ax.legend(
            handles=handles,
            loc=LEGEND_LOCATION,
            fontsize=7,
        )

    # ------------------------------------------------------------
    # Labels and title
    # ------------------------------------------------------------

    ax.set_xlabel(f"Spatial mean")
    ax.set_ylabel(f"Spatial standard deviation")

    ax.set_title(
        f"{space} {field} field: "
        "mean vs standard deviation "
        f"{title_suffix}"
    )

    # ------------------------------------------------------------
    # Optional fixed axis limits
    # ------------------------------------------------------------

    if DISTRIBUTION_XLIM_MEAN is not None:
        ax.set_xlim(*DISTRIBUTION_XLIM_MEAN)

    if DISTRIBUTION_XLIM_STD is not None:
        ax.set_ylim(*DISTRIBUTION_XLIM_STD)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)

# ============================================================================
# Monthly plots
# ============================================================================

def plot_monthly_stats(
    stats: pd.DataFrame,
    output_dir: Path,
    points_to_annotate: pd.DataFrame | None = None,
) -> None:
    monthly_output = output_dir / "monthly"

    monthly_output.mkdir(parents=True, exist_ok=True)

    space: Literal["normalized", "physical"] = (
        "normalized" if NORMALIZED_DISTRIBUTION else "physical"
    )

    channels = sorted(stats["channel"].unique())

    months = sorted(stats["month"].dropna().unique())

    for month_value in months:
        month = int(month_value)
        month_stats = stats[
            stats["month"] == month
        ]

        if month_stats.empty:
            continue

        month_name = MONTH_NAMES[month]

        month_dir = (
            monthly_output
            / (
                f"{month:02d}_"
                f"{month_name.lower()}"
            )
        )

        month_dir.mkdir(parents=True, exist_ok=True)

        title_suffix = f" — {month_name}"

        for channel in channels:
            channel_stats = (
                month_stats[
                    month_stats["channel"] == channel
                ]
            )

            if channel_stats.empty:
                continue

            for quantity in DISTRIBUTION_QUANTITIES:
                plot_distribution(
                    month_stats,
                    field=DISTRIBUTION_FIELD,
                    space=space,
                    quantity=quantity,
                    channel=channel,
                    plot_type=DISTRIBUTION_PLOT,
                    title_suffix=title_suffix,
                    output_path=(
                        month_dir
                        / (
                            f"channel_{channel}"
                            f"_{PERFORMANCE_METRIC}_color_encoding"
                            f"_{quantity}_distribution.png"
                        )
                    ),
                    points_to_annotate=points_to_annotate,
                )

            plot_mean_vs_std(
                month_stats,
                field=DISTRIBUTION_FIELD,
                space=space,
                channel=channel,
                title_suffix=title_suffix,
                output_path=(
                    month_dir
                    / (
                        f"channel_{channel}"
                        f"_{PERFORMANCE_METRIC}_color_encoding"
                        "_mean_vs_std.png"
                    )
                ),
                points_to_annotate=points_to_annotate,
            )

# ============================================================================
# All plots
# ============================================================================

def plot_stats(
    stats: pd.DataFrame,
    output_dir: Path,
    s: Settings,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    channels = sorted(
        stats[
            "channel"
        ].unique()
    )

    space: Literal["normalized", "physical"] = (
        "normalized"
        if NORMALIZED_DISTRIBUTION
        else "physical"
    )

    # ------------------------------------------------------------
    # Automatic best/worst timestep annotations
    # ------------------------------------------------------------

    extrema = get_performance_extrema(
        stats,
        n_best=ANNOTATE_BEST_TIMESTEPS,
        n_worst=ANNOTATE_WORST_TIMESTEPS,
    ).copy()

    if extrema.empty:
        extrema["_annotation_label"] = pd.Series(dtype=str)
    else:
        extrema["_annotation_label"] = extrema.apply(
            format_extrema_annotation,
            axis=1,
        )

    # ------------------------------------------------------------
    # Manually selected date annotations
    # ------------------------------------------------------------

    if SELECTED_DATES_TO_ANNOTATE:
        selected_dates = pd.to_datetime(
            list(SELECTED_DATES_TO_ANNOTATE)
        )

        manual = stats.loc[
            (stats["field"] == DISTRIBUTION_FIELD)
            & stats["split"].isin(DISTRIBUTION_SPLIT_RANGE)
            & pd.to_datetime(stats["time"]).isin(selected_dates)
        ].copy()

        manual = manual.drop_duplicates(subset=["split", "time"])

        manual["_annotation_label"] = (
            pd.to_datetime(manual["time"])
            .dt.strftime(ANNOTATION_DATE_FORMAT)
        )

        # Avoid duplicating automatically annotated timesteps.
        manual = manual.loc[
            ~pd.to_datetime(manual["time"]).isin(
                pd.to_datetime(extrema["time"])
            )
        ]

        extrema = pd.concat(
            [extrema, manual],
            ignore_index=True,
        )

    # ------------------------------------------------------------
    # Full-period plots
    # ------------------------------------------------------------

    for channel in channels:
        for quantity in DISTRIBUTION_QUANTITIES:
            plot_distribution(
                stats,
                quantity=quantity,
                field=DISTRIBUTION_FIELD,
                space=space,
                channel=channel,
                title_suffix=s.var_fc,
                plot_type=DISTRIBUTION_PLOT,
                output_path=(
                    output_dir
                    / f"channel_{channel}_{quantity}_distribution.png"
                ),
                points_to_annotate=extrema,
            )

        plot_mean_vs_std(
            stats,
            field=DISTRIBUTION_FIELD,
            space=space,
            channel=channel,
            title_suffix=s.var_fc,
            output_path=(
                output_dir
                / (
                    f"channel_{channel}"
                    f"_{PERFORMANCE_METRIC}_color_encoding"
                    "_mean_vs_std.png"
                )
            ),
            points_to_annotate=extrema,
        )

    # ------------------------------------------------------------
    # Monthly
    # ------------------------------------------------------------

    if PLOT_MONTHLY:
        plot_monthly_stats(
            stats,
            output_dir,
        )

# ============================================================================
# Summary
# ============================================================================


# ============================================================================
# Scalar, pairwise and spectral diagnostics (single shared pipeline)
# ============================================================================

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
        if metric in SPECTRAL_METRICS or metric in SHAPE_METRICS or metric in ZSCORE_METRICS:
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

def diagnostic_names() -> list[str]:
    single_cases, pairwise_cases = diagnostic_cases()
    return (
        [f"{field}_{metric}" for field, metric in single_cases]
        + [f"{a}_vs_{b}_{metric}" for a, b, metric in pairwise_cases]
    )

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
        alpha=0.9,
        # alpha=SCATTER_ALPHA,
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


def print_diagnostic_summary(
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



def print_summary(
    stats: pd.DataFrame,
) -> None:
    summary = (
        stats
        .groupby(
            [
                "leadtime",
                "channel",
                "split",
            ]
        )[
            [
                "mean",
                "std",
            ]
        ]
        .agg(
            [
                "mean",
                "std",
                "median",
                "min",
                "max",
            ]
        )
    )

    print()

    print(
        summary.to_string()
    )

    print()

    if (
        "metric_improvement"
        in stats.columns
    ):
        test_perf = (
            stats.loc[
                stats["split"] == "test",
                [
                    "time",
                    "metric_improvement",
                ],
            ]
            .dropna()
            .drop_duplicates(
                subset="time"
            )
        )

        if not test_perf.empty:
            print(
                f"{PERFORMANCE_METRIC.upper()} "
                f"improvement "
                f"({PERFORMANCE_IMPROVEMENT_UNIT}):"
            )

            print(
                test_perf[
                    "metric_improvement"
                ]
                .describe()
                .to_string()
            )

        print()


if __name__ == "__main__":
    main()
