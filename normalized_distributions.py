from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from scipy.stats import gaussian_kde, spearmanr
import torch

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from earthml import (
    ClimPeriod,
    LeadtimeUnit,
    MonthlyNormalize,
    Normalize,
    XarrayDataset,
    get_experiment_configs,
    get_and_subset_datasets,
)

from earthml.metrics import (
    get_metrics,
    build_metric_improvements,
)

from train import make_train_test_datasets_for_leadtime

# ============================================================================
# Configuration
# ============================================================================

EXP_NAME = "weather_atmo"

EXPERIMENTS_ROOT = Path(
    f"/work/cmcc/jd19424/ML/MLBC/experiments/{EXP_NAME}"
)

OUTPUT_ROOT = Path(
    f"/work/cmcc/jd19424/ML/MLBC/plots/{EXP_NAME}"
)

INTERPOLATE_ANALYSIS = False # weather

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
# improvement). None keeps the corresponding full experiment dimension.
# Best/worst rankings therefore refer to performance inside this box.
DIAGNOSTIC_LAT_RANGE: tuple[float, float] | None = None
DIAGNOSTIC_LON_RANGE: tuple[float, float] | None = None

# DIAGNOSTIC_LAT_RANGE = (30, 45)
# DIAGNOSTIC_LON_RANGE = (-108, -90)

# ----------------------------------------------------------------------------
# Field statistics
# ----------------------------------------------------------------------------

COMPUTE_EXTREMA = True

# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

PLOT_MONTHLY = False

DISTRIBUTION_PLOT: Literal[
    "hist",
    "kde",
    "both",
] = "kde"

BINS = 250

KDE_POINTS = 500

# Splits shown in the distribution and scatter plots.
# Any combination of: "train", "val", "test".
PLOT_SPLITS: tuple[
    Literal["train", "val", "test"],
    ...
] = (
    # "train",
    # "val",
    "test",
)

# Number of best/worst performance timesteps to annotate.
# Performance is currently available for the test split only.
# ANNOTATE_BEST_TIMESTEPS = 30
ANNOTATE_BEST_TIMESTEPS = 4
ANNOTATE_WORST_TIMESTEPS = 4
ANNOTATION_FONTSIZE = 5

ANNOTATION_DATE_FORMAT = "%Y-%m-%d:%H:%M"

SPLIT_COLORS = {
    "train": "C0",
    "val": "C1",
    "test": "C2",
}

COMPUTE_PERFORMANCE = True

# ----------------------------------------------------------------------------
# Performance coloring
# ----------------------------------------------------------------------------

COLOR_BY_PERFORMANCE = True

PERFORMANCE_METRIC = "rmse"

PERFORMANCE_IMPROVEMENT_UNIT: Literal[
    "Δ",
    "%",
] = "Δ"

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

# Scatter settings
PERFORMANCE_SCATTER_SIZE = 14
PERFORMANCE_SCATTER_ALPHA = 0.80

# Non-performance-colored scatter points use the same neutral appearance
# regardless of whether they belong to train, validation, or test.
SCATTER_COLOR = "0.45"
SCATTER_SIZE = 12
SCATTER_ALPHA = 0.35

# ----------------------------------------------------------------------------
# Performance diagnostics
# ----------------------------------------------------------------------------

COMPUTE_DIAGNOSTICS = True

# Scalar per-timestep diagnostics to compare against the performance metric.
# Laplacian diagnostics use DIAGNOSTIC_FIELD; correction/ideal diagnostics
# are defined directly from FC, analysis, and MLFC.
DIAGNOSTIC_METRICS: tuple[Literal[
    "rms_laplacian",
    "mean_abs_laplacian",
    "correction_ideal_correlation",
    "correction_ideal_cosine",
    "correction_rms_ratio",
    "correction_projection",
    "correction_orthogonal_rms",
    "error_correction_mean_product",
    "low_frequency_power_fraction",
    "high_frequency_power_fraction",
    "high_low_power_ratio",
    "spectral_centroid",
], ...] = (
    # "rms_laplacian",
    # "mean_abs_laplacian",
    "correction_ideal_correlation",
    "correction_ideal_cosine",
    # "correction_rms_ratio",
    # "correction_projection",
    # "correction_orthogonal_rms",
    # "error_correction_mean_product",
    # "low_frequency_power_fraction",
    # "high_frequency_power_fraction",
    # "high_low_power_ratio",
    # "spectral_centroid",
)

# Field used only by the Laplacian diagnostics above.
DIAGNOSTIC_FIELD: Literal[
    "fc",
    "an",
    "error", # fc - an
    "correction", # mlfc - fc
    "mlfc",
] = "fc"

DIAGNOSTIC_SCATTER_SIZE = 18
DIAGNOSTIC_SCATTER_ALPHA = 0.75

HIGH_FREQ_THRESHOLD = 0.5

# Number of annuli used for the wavenumber-resolved spectral diagnostic.
SPECTRAL_N_BINS = 30
PLOT_SPECTRAL_WAVENUMBER_CORRELATION = True

# These diagnostics are calculated directly from the normalized input tensor
# in collect_normalized_field_stats(), not from FC/AN/MLFC datasets.
NORMALIZED_INPUT_DIAGNOSTICS = {
    "low_frequency_power_fraction",
    "high_frequency_power_fraction",
    "high_low_power_ratio",
    "spectral_centroid",
}

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
# Normalization
# ============================================================================

def make_input_normalizer(
    s,
    train_dataset: XarrayDataset,
):
    """
    Reproduce input normalization used during training.
    The normalization statistics are fitted only on training data.
    """
    if s.normalization == "monthly":
        norm_class = MonthlyNormalize
    elif s.normalization == "full":
        norm_class = Normalize
    else:
        raise ValueError(
            f"Unsupported normalization={s.normalization!r}"
        )

    # Encoding channels should not be normalized
    n_excluded_channels = 0
    if (
        s.seasonal_encoding
        and s.channel_representation != "init_period"
    ):
        n_excluded_channels += 4
    if s.spatial_encoding:
        n_excluded_channels += 4
    input_excluded_channels = (
        tuple(range(-n_excluded_channels, 0))
        if n_excluded_channels > 0
        else None
    )

    return norm_class(
        mode=s.normalization_mode,
        exclude_channels=input_excluded_channels,
    ).fit(
        train_dataset,
        dim="x",
    )

# ============================================================================
# Dataset helpers
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

def get_sample_times(
    dataset: XarrayDataset,
) -> np.ndarray:
    """
    Return one initialization time for every dataset sample.
    samples_per_init handles cases where realizations are represented
    as independent samples.
    """
    time_dim = dataset.input_ds.earthml.guessed_dims.time
    if time_dim is None:
        raise ValueError(
            "Could not determine input time dimension."
        )
    init_times = dataset.input_ds[time_dim].values
    sample_times = np.repeat(
        init_times,
        dataset.samples_per_init,
    )
    if len(sample_times) != len(dataset):
        raise RuntimeError(
            "Cannot align initialization times with dataset samples: "
            f"{len(sample_times)=}, {len(dataset)=}"
        )
    return sample_times



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


def subset_normalized_tensor_region(
    dataset: XarrayDataset,
    x: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subset normalized input tensors to the configured diagnostic box."""
    if DIAGNOSTIC_LAT_RANGE is None and DIAGNOSTIC_LON_RANGE is None:
        return x, mask

    source = dataset.input_ds
    lat_dim = source.earthml.guessed_dims.latitude
    lon_dim = source.earthml.guessed_dims.longitude

    if lat_dim is None or lon_dim is None:
        raise ValueError(
            "Could not determine latitude/longitude dimensions for "
            "diagnostic-region subsetting."
        )

    latitude = np.asarray(source[lat_dim].values)
    longitude = np.asarray(source[lon_dim].values)

    if x.shape[-2:] != (latitude.size, longitude.size):
        raise ValueError(
            "Normalized input tensor/spatial-coordinate shape mismatch: "
            f"{tuple(x.shape[-2:])} != "
            f"{(latitude.size, longitude.size)}"
        )
    if mask.shape[-2:] != x.shape[-2:]:
        raise ValueError(
            "Mask/input spatial shape mismatch before regional subsetting: "
            f"{tuple(mask.shape[-2:])} != {tuple(x.shape[-2:])}"
        )

    lat_indices = _coordinate_indices(
        latitude,
        DIAGNOSTIC_LAT_RANGE,
    )
    lon_indices = _coordinate_indices(
        longitude,
        DIAGNOSTIC_LON_RANGE,
        longitude=True,
    )

    lat_index = torch.as_tensor(lat_indices, dtype=torch.long, device=x.device)
    lon_index = torch.as_tensor(lon_indices, dtype=torch.long, device=x.device)

    x = x.index_select(-2, lat_index).index_select(-1, lon_index)
    mask = mask.index_select(-2, lat_index).index_select(-1, lon_index)

    return x, mask


def subset_diagnostic_dataarray_region(
    da: xr.DataArray,
) -> xr.DataArray:
    """Subset an aligned FC/AN/MLFC field to the configured diagnostic box."""
    if DIAGNOSTIC_LAT_RANGE is None and DIAGNOSTIC_LON_RANGE is None:
        return da

    lat_dim = da.earthml.guessed_dims.latitude
    lon_dim = da.earthml.guessed_dims.longitude

    if lat_dim is None or lon_dim is None:
        raise ValueError(
            "Could not determine latitude/longitude dimensions for "
            "diagnostic-region subsetting."
        )

    lat_indices = _coordinate_indices(
        np.asarray(da[lat_dim].values),
        DIAGNOSTIC_LAT_RANGE,
    )
    lon_indices = _coordinate_indices(
        np.asarray(da[lon_dim].values),
        DIAGNOSTIC_LON_RANGE,
        longitude=True,
    )

    return da.isel({
        lat_dim: lat_indices,
        lon_dim: lon_indices,
    })


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

# ============================================================================
# Normalized field statistics
# ============================================================================

def masked_field_stats(
    x: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Calculate spatial statistics independently for each physical channel.
    Parameters
    ----------
    x
        Normalized field with shape (C, H, W).
    mask
        Valid-data mask with shape (C, H, W).
    Returns
    -------
    mean, std, minimum, maximum
        Each has shape (C,).
    """
    if x.ndim != 3:
        raise ValueError(
            f"Expected x with shape (C,H,W), got {tuple(x.shape)}"
        )
    if mask.ndim != 3:
        raise ValueError(
            f"Expected mask with shape (C,H,W), got {tuple(mask.shape)}"
        )

    n_channels = mask.shape[0]

    if x.shape[0] < n_channels:
        raise ValueError(
            "x has fewer channels than mask: "
            f"{tuple(x.shape)} vs {tuple(mask.shape)}"
        )
    # Remove any additional encoder channels for these diagnostics.
    x = x[:n_channels]
    if x.shape != mask.shape:
        raise ValueError(
            "x/mask shape mismatch after channel selection: "
            f"{tuple(x.shape)} != {tuple(mask.shape)}"
        )
    means = torch.full(
        (n_channels,),
        torch.nan,
        dtype=torch.float64,
    )
    stds = torch.full_like(
        means,
        torch.nan,
    )
    minima = torch.full_like(
        means,
        torch.nan,
    )
    maxima = torch.full_like(
        means,
        torch.nan,
    )
    for channel in range(n_channels):
        valid = mask[channel].bool()
        values = x[channel][valid].double()
        if values.numel() == 0:
            continue
        means[channel] = values.mean()
        # Population std, consistent with the normalizer.
        stds[channel] = values.std(
            unbiased=False,
        )
        minima[channel] = values.min()
        maxima[channel] = values.max()
    return (
        means,
        stds,
        minima,
        maxima,
    )


def collect_normalized_field_stats(
    dataset: XarrayDataset,
    *,
    split: str,
    leadtime: int,
) -> pd.DataFrame:
    """
    Collect field statistics after input normalization.
    dataset[idx] invokes transform_x, so x here is the actual tensor
    immediately before the NN machinery.
    """
    rows = []
    sample_times = get_sample_times(
        dataset
    )
    for idx in range(len(dataset)):
        x, y, mask, month = dataset[idx]

        x = x.detach().cpu()
        y = y.detach().cpu()

        mask = mask.detach().cpu().bool()

        x, mask = subset_normalized_tensor_region(
            dataset,
            x,
            mask,
        )

        mean, std, minimum, maximum = masked_field_stats(x, mask)

        (
            low_frequency_power_fraction,
            high_frequency_power_fraction,
            high_low_power_ratio,
            spectral_centroid,
        ) = spectral_field_stats(
            x,
            mask,
            high_frequency_threshold=HIGH_FREQ_THRESHOLD,
        )

        time = pd.Timestamp(
            sample_times[idx]
        )
        for channel in range(len(mean)):
            row = {
                "split": split,
                "leadtime": leadtime,
                "sample": idx,
                "time": time,
                "channel": channel,
                "month": int(month),

                "mean": float(mean[channel]),
                "std": float(std[channel]),

                "low_frequency_power_fraction": float(
                    low_frequency_power_fraction[channel]
                ),
                "high_frequency_power_fraction": float(
                    high_frequency_power_fraction[channel]
                ),
                "high_low_power_ratio": float(
                    high_low_power_ratio[channel]
                ),
                "spectral_centroid": float(
                    spectral_centroid[channel]
                ),
            }

            if COMPUTE_EXTREMA:
                row["min"] = float(minimum[channel])
                row["max"] = float(maximum[channel])

            rows.append(row)

    return pd.DataFrame(rows)

# ============================================================================
# Power diagnostics
# ============================================================================

def spectral_power_modes(
    field: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return normalized radial wavenumber and 2-D FFT power for all nonzero modes.

    The field mean is removed over valid points and a 2-D Hann window is
    applied before the FFT to reduce boundary leakage. Invalid grid points are
    filled with zero after mean removal. This is intended for comparisons on
    the same fixed grid/mask across timesteps.
    """
    field = field.double()
    valid = valid.bool()

    if field.ndim != 2 or valid.ndim != 2:
        raise ValueError(
            "Expected field/mask with shape (H,W), got "
            f"{tuple(field.shape)} and {tuple(valid.shape)}"
        )

    if field.shape != valid.shape:
        raise ValueError(
            f"Field/mask shape mismatch: {tuple(field.shape)} != {tuple(valid.shape)}"
        )

    if not valid.any():
        empty = torch.empty(0, dtype=torch.float64, device=field.device)
        return empty, empty

    mean = field[valid].mean()
    field = torch.where(
        valid,
        field - mean,
        torch.zeros_like(field),
    )

    height, width = field.shape

    window_y = torch.hann_window(
        height,
        periodic=False,
        dtype=field.dtype,
        device=field.device,
    )
    window_x = torch.hann_window(
        width,
        periodic=False,
        dtype=field.dtype,
        device=field.device,
    )
    field = field * window_y[:, None] * window_x[None, :]

    # Use the full 2-D FFT so radial annuli contain the complete symmetric
    # spectrum rather than a one-sided rFFT with unequal multiplicities.
    fft = torch.fft.fft2(field)
    power = torch.abs(fft) ** 2

    ky = torch.fft.fftfreq(
        height,
        dtype=field.dtype,
        device=field.device,
    )
    kx = torch.fft.fftfreq(
        width,
        dtype=field.dtype,
        device=field.device,
    )

    radial_k = torch.sqrt(
        ky[:, None] ** 2
        + kx[None, :] ** 2
    )

    k_max = radial_k.max()
    if not torch.isfinite(k_max) or k_max <= 0:
        empty = torch.empty(0, dtype=field.dtype, device=field.device)
        return empty, empty

    radial_k = radial_k / k_max
    nonzero = radial_k > 0

    return radial_k[nonzero], power[nonzero]


def spectral_field_stats(
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    high_frequency_threshold: float = 0.35,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Calculate scalar spectral diagnostics for each physical channel."""
    if x.ndim != 3:
        raise ValueError(
            f"Expected x with shape (C,H,W), got {tuple(x.shape)}"
        )
    if mask.ndim != 3:
        raise ValueError(
            f"Expected mask with shape (C,H,W), got {tuple(mask.shape)}"
        )
    if not 0.0 < high_frequency_threshold < 1.0:
        raise ValueError(
            "high_frequency_threshold must be strictly between 0 and 1."
        )

    n_channels = mask.shape[0]
    x = x[:n_channels]

    low_fraction = torch.full((n_channels,), torch.nan, dtype=torch.float64)
    high_fraction = torch.full_like(low_fraction, torch.nan)
    high_low_ratio = torch.full_like(low_fraction, torch.nan)
    spectral_centroid = torch.full_like(low_fraction, torch.nan)

    for channel in range(n_channels):
        radial_k, power = spectral_power_modes(
            x[channel],
            mask[channel],
        )

        if power.numel() == 0:
            continue

        total_power = power.sum()
        if not torch.isfinite(total_power) or total_power <= 0:
            continue

        high = radial_k >= high_frequency_threshold
        low = radial_k < high_frequency_threshold

        p_high = power[high].sum()
        p_low = power[low].sum()

        low_fraction[channel] = p_low / total_power
        high_fraction[channel] = p_high / total_power

        if p_low > 0:
            high_low_ratio[channel] = p_high / p_low

        spectral_centroid[channel] = (
            radial_k * power
        ).sum() / total_power

    return (
        low_fraction,
        high_fraction,
        high_low_ratio,
        spectral_centroid,
    )


def radial_power_spectrum(
    field: torch.Tensor,
    mask: torch.Tensor,
    *,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return normalized radial-wavenumber bin centers and power fractions.

    Power fractions sum to one across non-empty bins (up to floating-point
    precision) because each annular-bin power is normalized by total nonzero
    wavenumber power.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    radial_k, power = spectral_power_modes(field, mask)

    if power.numel() == 0:
        return (
            np.full(n_bins, np.nan, dtype=float),
            np.full(n_bins, np.nan, dtype=float),
        )

    total_power = power.sum()
    if not torch.isfinite(total_power) or total_power <= 0:
        return (
            np.full(n_bins, np.nan, dtype=float),
            np.full(n_bins, np.nan, dtype=float),
        )

    edges = torch.linspace(
        0.0,
        1.0,
        n_bins + 1,
        dtype=radial_k.dtype,
        device=radial_k.device,
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    spectrum = torch.full_like(centers, torch.nan)

    for index in range(n_bins):
        if index == n_bins - 1:
            in_bin = (
                (radial_k >= edges[index])
                & (radial_k <= edges[index + 1])
            )
        else:
            in_bin = (
                (radial_k >= edges[index])
                & (radial_k < edges[index + 1])
            )

        if in_bin.any():
            spectrum[index] = power[in_bin].sum() / total_power

    return (
        centers.detach().cpu().numpy(),
        spectrum.detach().cpu().numpy(),
    )


def collect_radial_power_spectra(
    dataset: XarrayDataset,
    *,
    split: str,
    leadtime: int,
    n_bins: int,
) -> pd.DataFrame:
    """Collect one normalized radial power spectrum per sample/channel."""
    rows = []
    sample_times = get_sample_times(dataset)

    for idx in range(len(dataset)):
        x, _, mask, _ = dataset[idx]
        x = x.detach().cpu()
        mask = mask.detach().cpu().bool()

        x, mask = subset_normalized_tensor_region(
            dataset,
            x,
            mask,
        )

        n_channels = mask.shape[0]
        x = x[:n_channels]
        time = pd.Timestamp(sample_times[idx])

        for channel in range(n_channels):
            wavenumber, power_fraction = radial_power_spectrum(
                x[channel],
                mask[channel],
                n_bins=n_bins,
            )

            for k, p in zip(wavenumber, power_fraction):
                rows.append({
                    "split": split,
                    "leadtime": leadtime,
                    "sample": idx,
                    "time": time,
                    "channel": channel,
                    "wavenumber": float(k),
                    "power_fraction": float(p),
                })

    return pd.DataFrame(rows)

# ============================================================================
# Performance diagnostics
# ============================================================================

def ensemble_mean_if_present(
    da: xr.DataArray,
    ds: xr.Dataset,
) -> xr.DataArray:
    """Average the realization dimension when present."""
    realization_dim = ds.earthml.guessed_dims.realization

    if (
        realization_dim is not None
        and realization_dim in da.dims
    ):
        da = da.mean(
            dim=realization_dim,
            skipna=True,
        )

    return da


def squeeze_singleton_leadtime(
    da: xr.DataArray,
    ds: xr.Dataset,
) -> xr.DataArray:
    """Remove the selected singleton leadtime dimension if still present."""
    leadtime_dim = ds.earthml.guessed_dims.leadtime

    if (
        leadtime_dim is not None
        and leadtime_dim in da.dims
    ):
        if da.sizes[leadtime_dim] != 1:
            raise ValueError(
                "Expected exactly one leadtime for diagnostic calculation, "
                f"got {da.sizes[leadtime_dim]}."
            )

        da = da.squeeze(
            leadtime_dim,
            drop=True,
        )

    return da


def get_aligned_diagnostic_fields(
    *,
    fc: xr.Dataset,
    an: xr.Dataset,
    mlfc: xr.Dataset,
    var_fc: str,
    var_an: str,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Return ensemble-mean FC, analysis, and MLFC on identical coordinates."""
    if var_fc not in fc:
        raise KeyError(f"{var_fc!r} not found in FC dataset.")

    if var_fc not in mlfc:
        raise KeyError(f"{var_fc!r} not found in MLFC dataset.")

    if var_an in an:
        an_var = var_an
    elif var_fc in an:
        an_var = var_fc
    else:
        raise KeyError(
            f"Neither {var_an!r} nor {var_fc!r} found in analysis dataset."
        )

    fc_da = squeeze_singleton_leadtime(
        ensemble_mean_if_present(fc[var_fc], fc),
        fc,
    )
    an_da = squeeze_singleton_leadtime(
        ensemble_mean_if_present(an[an_var], an),
        an,
    )
    mlfc_da = squeeze_singleton_leadtime(
        ensemble_mean_if_present(mlfc[var_fc], mlfc),
        mlfc,
    )

    return xr.align(
        fc_da,
        an_da,
        mlfc_da,
        join="exact",
    )


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


def get_diagnostic_timeseries(
    *,
    fc: xr.Dataset,
    an: xr.Dataset,
    mlfc: xr.Dataset,
    s,
) -> pd.DataFrame:
    """Calculate all configured scalar diagnostics independently per timestep."""
    fc_da, an_da, mlfc_da = get_aligned_diagnostic_fields(
        fc=fc,
        an=an,
        mlfc=mlfc,
        var_fc=s.var_fc,
        var_an=s.var_an,
    )

    fc_da = subset_diagnostic_dataarray_region(fc_da)
    an_da = subset_diagnostic_dataarray_region(an_da)
    mlfc_da = subset_diagnostic_dataarray_region(mlfc_da)

    spatial_dims, time_dim = diagnostic_spatial_dims(
        fc_da,
        fc,
    )

    error = fc_da - an_da
    ideal_correction = an_da - fc_da
    correction = mlfc_da - fc_da

    field_map = {
        "fc": fc_da,
        "an": an_da,
        "mlfc": mlfc_da,
        "error": error,
        "correction": correction,
    }

    result = pd.DataFrame({
        "time": pd.to_datetime(fc_da[time_dim].values),
    })

    for metric in DIAGNOSTIC_METRICS:
        if metric in NORMALIZED_INPUT_DIAGNOSTICS:
            continue

        if metric in {
            "rms_laplacian",
            "mean_abs_laplacian",
        }:
            values = laplacian_metric(
                field_map[DIAGNOSTIC_FIELD],
                spatial_dims=spatial_dims,
                metric=metric,
            )

        elif metric == "correction_ideal_correlation":
            values = pairwise_spatial_correlation(
                correction,
                ideal_correction,
                spatial_dims=spatial_dims,
            )

        elif metric == "correction_ideal_cosine":
            values = pairwise_cosine_similarity(
                correction,
                ideal_correction,
                spatial_dims=spatial_dims,
            )

        elif metric == "correction_rms_ratio":
            values = (
                rms_over_dims(correction, spatial_dims)
                / rms_over_dims(ideal_correction, spatial_dims)
            )

        elif metric == "correction_projection":
            valid = (
                np.isfinite(correction)
                & np.isfinite(ideal_correction)
            )
            correction_valid = correction.where(valid)
            ideal_valid = ideal_correction.where(valid)
            values = (
                (correction_valid * ideal_valid).mean(
                    dim=spatial_dims,
                    skipna=True,
                )
                / (ideal_valid**2).mean(
                    dim=spatial_dims,
                    skipna=True,
                )
            )

        elif metric == "correction_orthogonal_rms":
            valid = (
                np.isfinite(correction)
                & np.isfinite(ideal_correction)
            )
            correction_valid = correction.where(valid)
            ideal_valid = ideal_correction.where(valid)
            alpha = (
                (correction_valid * ideal_valid).mean(
                    dim=spatial_dims,
                    skipna=True,
                )
                / (ideal_valid**2).mean(
                    dim=spatial_dims,
                    skipna=True,
                )
            )
            orthogonal = correction_valid - alpha * ideal_valid
            values = rms_over_dims(
                orthogonal,
                spatial_dims,
            )

        elif metric == "error_correction_mean_product":
            valid = np.isfinite(error) & np.isfinite(correction)
            values = (
                error.where(valid)
                * correction.where(valid)
            ).mean(
                dim=spatial_dims,
                skipna=True,
            )

        else:
            raise ValueError(
                f"Unsupported diagnostic metric: {metric!r}"
            )

        values = values.squeeze(drop=True)
        extra_dims = [
            dim
            for dim in values.dims
            if dim != time_dim
        ]
        if extra_dims:
            raise ValueError(
                f"Diagnostic {metric!r} still contains unexpected "
                f"dimensions: {values.dims}"
            )

        result[metric] = pd.to_numeric(
            values.values,
            errors="coerce",
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
    s,
    *,
    leadtime: int,
    metric: str,
    improvement_unit: Literal["Δ", "%"],
) -> pd.DataFrame:
    """
    Calculate FC -> MLFC metric improvement for every test initialization.
    The time coordinate is already initialization time, so NO lead-time
    shift is performed.
    Positive improvement always means MLFC is better.
    """
    time_range = (
        s.test_start,
        s.test_end,
    )
    leadtime_units = (
        s.leadtime_unit
        if isinstance(s.leadtime_unit, LeadtimeUnit)
        else LeadtimeUnit(s.leadtime_unit)
    )
    clim_period = (
        s.clim_period
        if isinstance(s.clim_period, ClimPeriod)
        else ClimPeriod(s.clim_period)
    )
    lat_lon = (
        list(s.region.values())
        if s.region is not None
        else [None, None]
    )
    valid_lat_range = (
        DIAGNOSTIC_LAT_RANGE
        if DIAGNOSTIC_LAT_RANGE is not None
        else lat_lon[0]
    )
    valid_lon_range = (
        DIAGNOSTIC_LON_RANGE
        if DIAGNOSTIC_LON_RANGE is not None
        else lat_lon[1]
    )
    fc, an, mlfc = get_and_subset_datasets(
        s,
        leadtime_units=leadtime_units,
        lat_range=valid_lat_range,
        lon_range=valid_lon_range,
        time_range=time_range,
        interpolate=INTERPOLATE_ANALYSIS,
        mlfc_path=None,
    )
    if mlfc is None:
        raise ValueError(
            "MLFC dataset is required to calculate "
            f"{metric!r} improvement."
        )
    leadtime_dim = (
        fc.earthml.guessed_dims.leadtime
    )
    if leadtime_dim is None:
        raise ValueError(
            "Could not determine leadtime dimension."
        )
    # Ensure all three datasets use only the requested lead.
    fc = fc.sel({
        leadtime_dim: [leadtime]
    })
    an = an.sel({
        leadtime_dim: [leadtime]
    })
    mlfc = mlfc.sel({
        leadtime_dim: [leadtime]
    })

    # ------------------------------------------------------------
    # Baseline forecast metric
    # ------------------------------------------------------------

    fc_metrics = get_metrics(
        an=an,
        fc=fc,
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
        period_reference="init",
        period_dim=f"init_{clim_period.value}",
        periods_requested=["all"],
        leadtime_unit=leadtime_units,
        align=False,
        fair_correction=False,
    )

    # ------------------------------------------------------------
    # Corrected forecast metric
    # ------------------------------------------------------------

    mlfc_metrics = get_metrics(
        an=an,
        fc=mlfc,
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
        period_reference="init",
        period_dim=f"init_{clim_period.value}",
        periods_requested=["all"],
        leadtime_unit=leadtime_units,
        align=False,
        fair_correction=False,
    )

    # ------------------------------------------------------------
    # Improvement
    # ------------------------------------------------------------

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

    print(
        f"Using performance metric: "
        f"{comparison_model}"
    )

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

    time_dim = (
        improvement_da
        .earthml
        .guessed_dims
        .time
    )

    if time_dim is None:
        raise ValueError(
            "Could not determine time dimension "
            "of metric improvement."
        )

    # After selecting the lead and period, this should be a
    # one-dimensional timeseries
    extra_dims = [
        dim
        for dim in improvement_da.dims
        if dim != time_dim
    ]

    if extra_dims:
        raise ValueError(
            "Metric improvement still contains unexpected "
            f"dimensions: {improvement_da.dims}"
        )

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

    # ------------------------------------------------------------
    # Per-timestep diagnostics
    # ------------------------------------------------------------

    if COMPUTE_DIAGNOSTICS:
        forecast_diagnostics = tuple(
            metric
            for metric in DIAGNOSTIC_METRICS
            if metric not in NORMALIZED_INPUT_DIAGNOSTICS
        )

        if forecast_diagnostics:
            print(
                "Calculating forecast/analysis diagnostics: "
                + ", ".join(forecast_diagnostics)
            )

            diagnostics_df = get_diagnostic_timeseries(
                fc=fc,
                an=an,
                mlfc=mlfc,
                s=s,
            )

            metric_df = metric_df.merge(
                diagnostics_df,
                on="time",
                how="left",
                validate="one_to_one",
            )

            for diagnostic in forecast_diagnostics:
                n_missing = int(
                    metric_df[diagnostic]
                    .isna()
                    .sum()
                )
                if n_missing:
                    print(
                        f"WARNING: {n_missing}/{len(metric_df)} "
                        f"metric timesteps have no {diagnostic} value."
                    )

    return metric_df


def add_metric_performance(
    stats: pd.DataFrame,
    metric_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add metric improvement to the test rows.
    Train/validation rows remain NaN because performance is calculated
    over the configured test interval.
    """
    stats = stats.copy()
    stats["time"] = pd.to_datetime(
        stats["time"]
    )
    metric_df = metric_df.copy()
    metric_df["time"] = pd.to_datetime(
        metric_df["time"]
    )
    # Only attach performance to test samples.
    test_stats = stats[
        stats["split"] == "test"
    ].merge(
        metric_df,
        on="time",
        how="left",
        validate="many_to_one",
    )
    other_stats = stats[
        stats["split"] != "test"
    ].copy()
    other_stats[
        "metric_improvement"
    ] = np.nan
    result = pd.concat(
        [
            other_stats,
            test_stats,
        ],
        ignore_index=True,
    )
    # Check alignment.
    test_rows = result[
        result["split"] == "test"
    ]
    n_missing = int(
        test_rows[
            "metric_improvement"
        ].isna().sum()
    )
    if n_missing:
        print(
            f"WARNING: {n_missing}/{len(test_rows)} "
            "test rows have no matching metric improvement."
        )
    return result

# ============================================================================
# Performance coloring
# ============================================================================

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
            stats["split"] == "test",
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


def format_extreme_annotation(row: pd.Series) -> str:
    """Compact label used by both distribution and scatter annotations."""
    prefix = (
        "B"
        if row["extreme"] == "best"
        else "W"
    )

    time = pd.Timestamp(row["time"])

    return (
        f"{prefix}{int(row['rank'])} "
        f"{time.strftime(ANNOTATION_DATE_FORMAT)}"
    )

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
# Distribution plot
# ============================================================================

def plot_distribution(
    stats: pd.DataFrame,
    *,
    quantity: str,
    channel: int,
    output_path: Path,
    plot_type: Literal[
        "hist",
        "kde",
        "both",
    ] = "both",
    title_suffix: str = "",
) -> None:
    if plot_type not in {
        "hist",
        "kde",
        "both",
    }:
        raise ValueError(
            f"Unsupported plot_type={plot_type!r}"
        )

    channel_df = stats[
        stats["channel"] == channel
    ]

    available_splits = [
        split
        for split in PLOT_SPLITS
        if split in channel_df["split"].unique()
    ]

    if not available_splits:
        return

    groups = {
        split: (
            channel_df.loc[
                channel_df["split"] == split,
                quantity,
            ]
            .dropna()
            .to_numpy(dtype=float)
        )
        for split in available_splits
    }

    lo, hi = distribution_range(
        list(groups.values())
    )

    edges = np.linspace(
        lo,
        hi,
        BINS + 1,
    )

    x_grid = np.linspace(
        lo,
        hi,
        KDE_POINTS,
    )

    fig, ax = plt.subplots(
        figsize=(8, 5),
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

    # Used only to place best/worst timestep annotations on the
    # test distribution envelope.
    test_density_for_annotations = None

    # ------------------------------------------------------------
    # Plot each selected split
    # ------------------------------------------------------------

    for split in available_splits:
        values = groups[split]

        if len(values) == 0:
            continue

        color = SPLIT_COLORS[split]

        median = float(
            np.median(values)
        )

        label = (
            f"{split} "
            f"(n={len(values)}, "
            f"median={median:.3f})"
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
                alpha=(
                    0.7
                    if plot_type == "both"
                    else 1.0
                ),
                color=color,
                label=(
                    label
                    if plot_type == "hist"
                    else None
                ),
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
            and len(values) > 1
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
                linewidth=2.0,
                color=color,
                label=label,
                zorder=5,
            )

        # Plot split median.
        ax.axvline(
            median,
            color=color,
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
            zorder=3,
        )

        if split == "test":
            if kde_density is not None:
                test_density_for_annotations = kde_density
            elif plot_type == "hist":
                test_density_for_annotations = (
                    histogram_density_on_grid(
                        values,
                        edges=edges,
                        x_grid=x_grid,
                    )
                )

    # ------------------------------------------------------------
    # Performance-colored density bands + correlation
    # ------------------------------------------------------------

    test_perf_df = pd.DataFrame()

    if (
        "test" in available_splits
        and "metric_improvement" in channel_df.columns
    ):
        test_perf_df = (
            channel_df.loc[
                channel_df["split"] == "test",
                [quantity, "metric_improvement"],
            ]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

    if (
        not test_perf_df.empty
        and performance_norm is not None
        and test_density_for_annotations is not None
    ):
        values = test_perf_df[quantity].to_numpy(dtype=float)
        performance = test_perf_df[
            "metric_improvement"
        ].to_numpy(dtype=float)

        # Assign every test sample to one of the same x-bins used by the
        # histogram. Each band gets the plain arithmetic mean performance
        # of the samples in that bin: no spatial/kernel smoothing.
        bin_indices = np.digitize(
            values,
            edges,
            right=False,
        ) - 1

        # Include values exactly equal to the rightmost edge in the last bin.
        bin_indices[values == edges[-1]] = len(edges) - 2

        for bin_index in range(len(edges) - 1):
            in_bin = bin_indices == bin_index
            if not np.any(in_bin):
                continue

            mean_performance = float(
                np.mean(performance[in_bin])
            )

            x0 = edges[bin_index]
            x1 = edges[bin_index + 1]
            band_mask = (x_grid >= x0) & (x_grid <= x1)

            if not np.any(band_mask):
                continue

            ax.fill_between(
                x_grid[band_mask],
                0.0,
                test_density_for_annotations[band_mask],
                color=performance_color(
                    mean_performance,
                    cmap=performance_cmap,
                    norm=performance_norm,
                ),
                alpha=0.95,
                linewidth=0,
                zorder=1,
            )

        add_performance_colorbar(
            fig,
            ax,
            cmap=performance_cmap,
            norm=performance_norm,
        )

    if len(test_perf_df) >= 2:
        rho, p_value = spearmanr(
            test_perf_df[quantity].to_numpy(dtype=float),
            test_perf_df["metric_improvement"].to_numpy(dtype=float),
        )

        if np.isfinite(rho):
            correlation_text = rf"Spearman $\rho$ = {rho:.2f}"
            if np.isfinite(p_value):
                correlation_text += f" (p={p_value:.2g})"

            ax.text(
                0.02,
                0.98,
                correlation_text,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=9,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "0.7",
                    "linewidth": 0.5,
                    "alpha": 0.85,
                },
                zorder=9,
            )

    # ------------------------------------------------------------
    # Best/worst test-timestep annotations
    # ------------------------------------------------------------

    if (
        "test" in available_splits
        and test_density_for_annotations is not None
    ):
        extrema = get_performance_extrema(
            channel_df,
            n_best=ANNOTATE_BEST_TIMESTEPS,
            n_worst=ANNOTATE_WORST_TIMESTEPS,
        )

        occupied_annotations = []

        for _, row in extrema.iterrows():
            time = pd.Timestamp(row["time"])

            timestep_rows = channel_df[
                (channel_df["split"] == "test")
                & (channel_df["time"] == time)
            ]

            if timestep_rows.empty:
                continue

            x = float(
                timestep_rows[quantity].mean()
            )

            y = float(
                np.interp(
                    x,
                    x_grid,
                    test_density_for_annotations,
                )
            )

            if not (
                np.isfinite(x)
                and np.isfinite(y)
            ):
                continue

            if performance_norm is not None:
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
                format_extreme_annotation(row),
                xy=(x, y),
                occupied=occupied_annotations,
            )

    # ------------------------------------------------------------
    # Reference lines
    # ------------------------------------------------------------

    if quantity == "mean":
        ax.axvline(
            0.0,
            linestyle="--",
            linewidth=1.0,
            color="black",
            alpha=0.7,
        )

    elif quantity == "std":
        ax.axvline(
            1.0,
            linestyle="--",
            linewidth=1.0,
            color="black",
            alpha=0.7,
        )

    # ------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------

    ax.set_xlabel(
        f"Spatial {quantity} "
        "of normalized field"
    )
    ax.set_ylabel(
        "Probability density"
    )

    ax.set_title(
        f"Channel {channel}: "
        f"normalized field {quantity}"
        f"{title_suffix}"
    )

    ax.legend()

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
    channel: int,
    output_path: Path,
    title_suffix: str = "",
) -> None:
    channel_df = stats[
        stats["channel"] == channel
    ]

    plot_df = channel_df[
        channel_df["split"].isin(
            PLOT_SPLITS
        )
    ]

    if plot_df.empty:
        return

    fig, ax = plt.subplots(
        figsize=(7, 6),
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

    for split in PLOT_SPLITS:
        df = plot_df[
            plot_df["split"] == split
        ]

        if df.empty:
            continue

        has_performance = (
            COLOR_BY_PERFORMANCE
            and performance_norm is not None
            and "metric_improvement" in df.columns
        )

        if has_performance:
            valid_perf = (
                df["metric_improvement"]
                .notna()
            )
        else:
            valid_perf = pd.Series(
                False,
                index=df.index,
                dtype=bool,
            )

        # --------------------------------------------------------
        # Points without performance data
        # Same neutral style for train/val/test.
        # --------------------------------------------------------

        ordinary = df[
            ~valid_perf
        ]

        if not ordinary.empty:
            ax.scatter(
                ordinary["mean"],
                ordinary["std"],
                s=SCATTER_SIZE,
                alpha=SCATTER_ALPHA,
                color=SCATTER_COLOR,
                label=split,
                zorder=2,
            )

        # --------------------------------------------------------
        # Points with performance data
        # Color encodes performance, not split membership.
        # --------------------------------------------------------

        performance_df = df[
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
                performance_df["mean"],
                performance_df["std"],
                c=point_colors,
                s=PERFORMANCE_SCATTER_SIZE,
                alpha=PERFORMANCE_SCATTER_ALPHA,
                label=split,
                zorder=4,
            )
            plotted_performance = True

    if plotted_performance:
        add_performance_colorbar(
            fig,
            ax,
            cmap=performance_cmap,
            norm=performance_norm,
        )

    # ------------------------------------------------------------
    # Best/worst test-timestep annotations
    # ------------------------------------------------------------

    if "test" in PLOT_SPLITS:
        extrema = get_performance_extrema(
            channel_df,
            n_best=ANNOTATE_BEST_TIMESTEPS,
            n_worst=ANNOTATE_WORST_TIMESTEPS,
        )

        occupied_annotations = []

        for _, row in extrema.iterrows():
            time = pd.Timestamp(row["time"])

            timestep_rows = channel_df[
                (channel_df["split"] == "test")
                & (channel_df["time"] == time)
            ]

            if timestep_rows.empty:
                continue

            x = float(
                timestep_rows["mean"].mean()
            )
            y = float(
                timestep_rows["std"].mean()
            )

            if not (
                np.isfinite(x)
                and np.isfinite(y)
            ):
                continue

            # Thin black ring around annotated points.
            ax.scatter(
                [x],
                [y],
                s=30,
                facecolors="none",
                edgecolors="black",
                linewidths=0.8,
                zorder=7,
            )

            annotate_nonoverlapping(
                ax,
                format_extreme_annotation(row),
                xy=(x, y),
                occupied=occupied_annotations,
            )

    # ------------------------------------------------------------
    # References
    # ------------------------------------------------------------

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        color="black",
        alpha=0.7,
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
        color="black",
        alpha=0.7,
    )

    ax.set_xlabel(
        "Spatial mean"
    )

    ax.set_ylabel(
        "Spatial standard deviation"
    )

    ax.set_title(
        f"Channel {channel}: "
        "normalized input fields"
        f"{title_suffix}"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)

# ============================================================================
# Diagnostic vs performance
# ============================================================================


def diagnostic_label(metric: str) -> str:
    labels = {
        "rms_laplacian": f"RMS Laplacian ({DIAGNOSTIC_FIELD})",
        "mean_abs_laplacian": f"Mean |Laplacian| ({DIAGNOSTIC_FIELD})",
        "correction_ideal_correlation": "Spatial corr(correction, ideal correction)",
        "correction_ideal_cosine": "Cosine(correction, ideal correction)",
        "correction_rms_ratio": "RMS(correction) / RMS(ideal correction)",
        "correction_projection": "Correction projection coefficient α",
        "correction_orthogonal_rms": "RMS orthogonal correction",
        "error_correction_mean_product": "Mean[(FC - analysis) × correction]",
        "low_frequency_power_fraction": "Low-frequency power fraction",
        "high_frequency_power_fraction": "High-frequency power fraction",
        "high_low_power_ratio": "High / low-frequency power ratio",
        "spectral_centroid": "Normalized spectral centroid",
    }
    return labels.get(metric, metric)


def plot_diagnostic_vs_performance(
    stats: pd.DataFrame,
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

    if not required_columns.issubset(stats.columns):
        return

    df = (
        stats.loc[
            stats["split"] == "test",
            [
                "time",
                "metric_improvement",
                diagnostic,
            ],
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
        s=DIAGNOSTIC_SCATTER_SIZE,
        color=SCATTER_COLOR,
        alpha=DIAGNOSTIC_SCATTER_ALPHA,
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

    ax.text(
        0.02,
        0.98,
        "\n".join(correlation_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.6",
            "linewidth": 0.5,
            "alpha": 0.85,
        },
        zorder=6,
    )

    extrema = get_performance_extrema(
        stats,
        n_best=ANNOTATE_BEST_TIMESTEPS,
        n_worst=ANNOTATE_WORST_TIMESTEPS,
    )
    occupied_annotations = []
    performance_norm = make_performance_norm(stats)
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
            markersize=5.0,
            markerfacecolor=marker_color,
            markeredgecolor="black",
            markeredgewidth=0.8,
            zorder=7,
        )

        annotate_nonoverlapping(
            ax,
            format_extreme_annotation(row),
            xy=(x, y),
            occupied=occupied_annotations,
        )

    ax.set_xlabel(diagnostic_label(diagnostic))
    ax.set_ylabel(
        f"{PERFORMANCE_METRIC.upper()} "
        f"improvement ({PERFORMANCE_IMPROVEMENT_UNIT})"
    )
    ax.set_title(
        f"Performance vs {diagnostic_label(diagnostic)}"
        f"{title_suffix}"
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_spearman_vs_spatial_wavenumber(
    spectral_df: pd.DataFrame,
    *,
    channel: int,
    output_path: Path,
    title_suffix: str = "",
) -> pd.DataFrame:
    """
    Plot Spearman correlation between normalized radial-bin power fraction
    and performance as a function of normalized spatial wavenumber.
    """
    required = {
        "time",
        "channel",
        "wavenumber",
        "power_fraction",
        "metric_improvement",
    }
    if spectral_df.empty or not required.issubset(spectral_df.columns):
        return pd.DataFrame()

    df = spectral_df.loc[
        spectral_df["channel"] == channel,
        list(required),
    ].copy()

    if df.empty:
        return pd.DataFrame()

    # If one initialization corresponds to multiple dataset samples, average
    # their spectral power before correlating so each timestep counts once.
    df = (
        df.groupby(
            ["time", "channel", "wavenumber"],
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
    # while the markers remain the actual wavenumber-bin correlations.
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
        f"Channel {channel}: spectral power vs performance"
        f"{title_suffix}"
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return correlation_df

# ============================================================================
# Monthly plots
# ============================================================================

def plot_monthly_stats(
    stats: pd.DataFrame,
    output_dir: Path,
) -> None:
    monthly_output = output_dir / "monthly"

    monthly_output.mkdir(parents=True, exist_ok=True)

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

        if (
            COMPUTE_DIAGNOSTICS
            and "metric_improvement" in month_stats.columns
        ):
            for diagnostic in DIAGNOSTIC_METRICS:
                if diagnostic not in month_stats.columns:
                    continue
                plot_diagnostic_vs_performance(
                    month_stats,
                    diagnostic=diagnostic,
                    title_suffix=title_suffix,
                    output_path=(
                        month_dir
                        / (
                            f"{PERFORMANCE_METRIC}_improvement_vs_"
                            f"{diagnostic}.png"
                        )
                    ),
                )

        for channel in channels:
            channel_stats = (
                month_stats[
                    month_stats[
                        "channel"
                    ] == channel
                ]
            )

            if channel_stats.empty:
                continue

            plot_distribution(
                month_stats,
                quantity="mean",
                channel=channel,
                plot_type=(
                    DISTRIBUTION_PLOT
                ),
                title_suffix=(
                    title_suffix
                ),
                output_path=(
                    month_dir
                    / (
                        f"channel_{channel}"
                        f"_{PERFORMANCE_METRIC}_color_encoding"
                        "_mean_distribution.png"
                    )
                ),
            )

            plot_distribution(
                month_stats,
                quantity="std",
                channel=channel,
                plot_type=(
                    DISTRIBUTION_PLOT
                ),
                title_suffix=(
                    title_suffix
                ),
                output_path=(
                    month_dir
                    / (
                        f"channel_{channel}"
                        f"_{PERFORMANCE_METRIC}_color_encoding"
                        "_std_distribution.png"
                    )
                ),
            )

            plot_mean_vs_std(
                month_stats,
                channel=channel,
                title_suffix=(
                    title_suffix
                ),
                output_path=(
                    month_dir
                    / (
                        f"channel_{channel}"
                        f"_{PERFORMANCE_METRIC}_color_encoding"
                        "_mean_vs_std.png"
                    )
                ),
            )

# ============================================================================
# All plots
# ============================================================================

def plot_stats(
    stats: pd.DataFrame,
    output_dir: Path,
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

    # ------------------------------------------------------------
    # Full-period plots
    # ------------------------------------------------------------

    for channel in channels:
        plot_distribution(
            stats,
            quantity="mean",
            channel=channel,
            plot_type=(
                DISTRIBUTION_PLOT
            ),
            output_path=(
                output_dir
                / (
                    f"channel_{channel}"
                    f"_{PERFORMANCE_METRIC}_color_encoding"
                    "_mean_distribution.png"
                )
            ),
        )

        plot_distribution(
            stats,
            quantity="std",
            channel=channel,
            plot_type=(
                DISTRIBUTION_PLOT
            ),
            output_path=(
                output_dir
                / (
                    f"channel_{channel}"
                    f"_{PERFORMANCE_METRIC}_color_encoding"
                    "_std_distribution.png"
                )
            ),
        )

        plot_mean_vs_std(
            stats,
            channel=channel,
            output_path=(
                output_dir
                / (
                    f"channel_{channel}"
                    f"_{PERFORMANCE_METRIC}_color_encoding"
                    "_mean_vs_std.png"
                )
            ),
        )

    # ------------------------------------------------------------
    # Diagnostics vs performance
    # ------------------------------------------------------------

    if (
        COMPUTE_DIAGNOSTICS
        and "metric_improvement" in stats.columns
    ):
        for diagnostic in DIAGNOSTIC_METRICS:
            if diagnostic not in stats.columns:
                continue
            plot_diagnostic_vs_performance(
                stats,
                diagnostic=diagnostic,
                output_path=(
                    output_dir
                    / (
                        f"{PERFORMANCE_METRIC}_improvement_vs_"
                        f"{DIAGNOSTIC_FIELD}_{diagnostic}.png"
                    )
                ),
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

    if (
        COMPUTE_DIAGNOSTICS
        and "metric_improvement" in stats.columns
    ):
        print("Diagnostics vs performance:")

        for diagnostic in DIAGNOSTIC_METRICS:
            if diagnostic not in stats.columns:
                continue

            diagnostic_perf = (
                stats.loc[
                    stats["split"] == "test",
                    [
                        "time",
                        "metric_improvement",
                        diagnostic,
                    ],
                ]
                .drop_duplicates(subset="time")
                .dropna()
            )

            if diagnostic_perf.empty:
                continue

            line = f"  {diagnostic}: "
            if len(diagnostic_perf) >= 2:
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
                line += "insufficient samples"

            print(line)

        print()

# ============================================================================
# Build normalized diagnostics for one leadtime
# ============================================================================

def process_leadtime(
    s,
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

    datasets = (
        make_train_test_datasets_for_leadtime(
            forecast_ds_path=(
                s.input_dir
                / (
                    f"{s.model_fc}_"
                    f"{s.var_fc}.zarr"
                )
            ),
            analysis_ds_path=(
                s.input_dir
                / (
                    f"{s.model_an}_"
                    f"{s.var_an}.zarr"
                )
            ),
            leadtime=leadtime,
            leadtime_unit=(
                s.leadtime_unit
            ),
            train_start=s.train_start,
            train_end=s.train_end,
            val_start=s.val_start,
            val_end=s.val_end,
            test_start=s.test_start,
            test_end=s.test_end,
            target_mode=s.target_mode,
            clim_period=s.clim_period,
            forecast_vars=[
                s.var_fc
            ],
            analysis_vars=[
                s.var_an
            ],
            region=s.region,
            dataset_kwargs=(
                dataset_kwargs_from_settings(
                    s
                )
            ),
            seasonal_encoding=(
                s.seasonal_encoding
                and (
                    s.channel_representation
                    != "init_period"
                )
            ),
            ensemble_encoding=(
                s.ensemble_encoding
            ),
            input_realization_avg=(
                s.input_realization_avg
            ),
            interpolate_analysis=False,
            materialize=False,
            separate_training_by_init_period=None,
            defer_dataset_creation=False,
        )
    )

    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    test_dataset = datasets["test"]

    if not isinstance(
        train_dataset,
        XarrayDataset,
    ):
        raise TypeError(
            "Expected XarrayDataset "
            "for training data"
        )

    if (
        val_dataset is not None
        and not isinstance(
            val_dataset,
            XarrayDataset,
        )
    ):
        raise TypeError(
            "Expected XarrayDataset "
            "for validation data"
        )

    if not isinstance(
        test_dataset,
        XarrayDataset,
    ):
        raise TypeError(
            "Expected XarrayDataset "
            "for test data"
        )

    # ------------------------------------------------------------
    # Fit on TRAIN only
    # ------------------------------------------------------------

    normalize_input = (
        make_input_normalizer(
            s,
            train_dataset,
        )
    )

    # ------------------------------------------------------------
    # Apply same transform to every split
    # ------------------------------------------------------------

    train_dataset.transform_x = (
        normalize_input
    )

    if val_dataset is not None:
        val_dataset.transform_x = (
            normalize_input
        )

    test_dataset.transform_x = (
        normalize_input
    )

    # Target transformation not needed
    train_dataset.transform_y = None

    if val_dataset is not None:
        val_dataset.transform_y = None

    test_dataset.transform_y = None

    # ------------------------------------------------------------
    # Normalized-field statistics
    # ------------------------------------------------------------

    frames = [
        collect_normalized_field_stats(
            train_dataset,
            split="train",
            leadtime=leadtime,
        )
    ]

    if val_dataset is not None:
        frames.append(
            collect_normalized_field_stats(
                val_dataset,
                split="val",
                leadtime=leadtime,
            )
        )

    frames.append(
        collect_normalized_field_stats(
            test_dataset,
            split="test",
            leadtime=leadtime,
        )
    )

    stats = pd.concat(
        frames,
        ignore_index=True,
    )

    # Wavenumber-resolved spectra are only needed for the test set because
    # performance is currently defined only over the test interval.
    if PLOT_SPECTRAL_WAVENUMBER_CORRELATION:
        spectral_df = collect_radial_power_spectra(
            test_dataset,
            split="test",
            leadtime=leadtime,
            n_bins=SPECTRAL_N_BINS,
        )
    else:
        spectral_df = pd.DataFrame()

    # ------------------------------------------------------------
    # Automatic metric performance
    # ------------------------------------------------------------

    if COMPUTE_PERFORMANCE:
        print(
            f"Calculating "
            f"{PERFORMANCE_METRIC} "
            f"improvement..."
        )

        metric_df = (
            get_metric_improvement_timeseries(
                s,
                leadtime=leadtime,
                metric=PERFORMANCE_METRIC,
                improvement_unit=(
                    PERFORMANCE_IMPROVEMENT_UNIT
                ),
            )
        )

        stats = add_metric_performance(
            stats,
            metric_df,
        )

        if not spectral_df.empty:
            spectral_df = spectral_df.merge(
                metric_df[["time", "metric_improvement"]],
                on="time",
                how="left",
                validate="many_to_one",
            )

    elif not spectral_df.empty:
        spectral_df["metric_improvement"] = np.nan

    return stats, spectral_df

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

    print(
        f"Found {len(settings)} "
        "matching experiment(s)."
    )

    for s in settings:
        experiment_output = (
            OUTPUT_ROOT
            / s.output_name
            / "normalized"
            / "input_diagnostics"
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
            stats, spectral_df = process_leadtime(
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

            plot_stats(
                stats,
                leadtime_output,
            )

            if (
                PLOT_SPECTRAL_WAVENUMBER_CORRELATION
                and not spectral_df.empty
                and "metric_improvement" in spectral_df.columns
            ):
                spectral_df.to_csv(
                    leadtime_output / "radial_power_spectra.csv",
                    index=False,
                )

                for channel in sorted(spectral_df["channel"].unique()):
                    correlation_df = plot_spearman_vs_spatial_wavenumber(
                        spectral_df,
                        channel=int(channel),
                        output_path=(
                            leadtime_output
                            / (
                                f"channel_{int(channel)}_"
                                "spearman_vs_spatial_wavenumber.png"
                            )
                        ),
                    )

                    if not correlation_df.empty:
                        correlation_df.to_csv(
                            leadtime_output
                            / (
                                f"channel_{int(channel)}_"
                                "spearman_vs_spatial_wavenumber.csv"
                            ),
                            index=False,
                        )

            print_summary(stats)

        all_stats = pd.concat(
            all_stats,
            ignore_index=True,
        )

        all_stats.to_csv(
            experiment_output
            / "field_stats_all_leadtimes.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
