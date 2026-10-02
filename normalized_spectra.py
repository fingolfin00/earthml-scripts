from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import xarray as xr

from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

import torch

from earthml import (
    ClimPeriod,
    LeadtimeUnit,
    MonthlyNormalize,
    Normalize,
    XarrayDataset,
    get_and_subset_datasets,
    get_experiment_configs,
)

from earthml.metrics import build_metric_improvements, get_metrics
from earthml.plots import plot_field_map, safe_label

from train import make_train_test_datasets_for_leadtime

import normalized_maps as normalized_maps
from normalized_maps import (
    normalized_sample_fields,
    normalized_fc_in_target_space,
    normalized_corrected_fields,
    subtract_aligned_field_collections,
    load_corrected_dataset,
)

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

INTERPOLATE_ANALYSIS = False  # weather

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

# Optional spatial subset used consistently for spectra, filtered fields,
# and performance/extrema. None -> use the full experiment region
LAT_RANGE: tuple[float, float] | None = None
LON_RANGE: tuple[float, float] | None = None

# Central degradation box
# LAT_RANGE = (30, 45)
# LON_RANGE = (-108, -90)

# ----------------------------------------------------------------------------
# Normalized-input selection
# ----------------------------------------------------------------------------

PLOT_SPLIT: Literal["train", "val", "test"] = "test"

# None -> all physical input channels
PLOT_CHANNELS: list[int] | None = None

# Initialization times to plot. None -> all times in the selected split
WANTED_TIMES: list[str] | None = [
    "2025-03-08T00:00:00", # B1 t2m 72h
    "2025-02-02T00:00:00", # B4 t2m 72h, near W3 in stb-vs-mean scatter
    "2025-02-13T00:00:00", # W1 t2m 72h
    "2025-01-03T00:00:00", # W2 t2m 72h
    "2025-02-06T00:00:00", # W3 t2m 72h
    "2025-02-18T00:00:00", # W4 t2m 72h
]
# WANTED_TIMES: list[str] | None = None

# If multiple network samples correspond to one initialization, the
# performance-facing analyses average their spectra at each wavenumber
AGGREGATE_SAMPLES_PER_INIT = True

# ----------------------------------------------------------------------------
# Field selection
# ----------------------------------------------------------------------------

SPECTRAL_FIELDS: tuple[
    Literal[
        "input",
        "target",
        "corrected",
        "fc_error",
        "mlfc_error",
        "ideal_correction",
        "ml_correction",
    ],
    ...,
] = (
    "input",
    "target",
    "corrected",
    "fc_error",
    "mlfc_error",
    "ideal_correction",
    "ml_correction",
)

SPECTRAL_FIELD_LABELS = {
    "input": "FC input",
    "target": "AN target",
    "corrected": "corrected forecast",
    "fc_error": "FC − AN",
    "mlfc_error": "MLFC − AN",
    "ideal_correction": "Ideal correction (AN − FC)",
    "ml_correction": "ML correction (MLFC − FC)",
}

# ----------------------------------------------------------------------------
# Radial spectrum
# ----------------------------------------------------------------------------

SPECTRAL_N_BINS = 30
# Normalized radial wavenumber. This is shown as a reference line only; the
# full spectrum and wavenumber-resolved correlations do not depend on it
HIGH_FREQ_THRESHOLD = 0.50
# Remove the spatial mean before the FFT so the k=0 component does not dominate
REMOVE_SPATIAL_MEAN = True
# Reduce leakage from the non-periodic CONUS domain edges
APPLY_HANN_WINDOW = True
# Each timestep spectrum is normalized by total nonzero-wavenumber power
NORMALIZE_POWER = True

# ----------------------------------------------------------------------------
# Performance / extrema
# ----------------------------------------------------------------------------

COMPUTE_PERFORMANCE = True

PERFORMANCE_METRIC = "rmse"

PERFORMANCE_IMPROVEMENT_UNIT: Literal["Δ", "%"] = "Δ"

N_BEST_TIMESTEPS = 4
N_WORST_TIMESTEPS = 4

# ----------------------------------------------------------------------------
# Plots / outputs
# ----------------------------------------------------------------------------

PLOT_SELECTED_TIMES = True
PLOT_BEST_WORST_SPECTRA = True
PLOT_SPEARMAN_VS_WAVENUMBER = True
PLOT_ALL_TEST_MEDIAN = True

# Spatial Fourier reconstructions for WANTED_TIMES.
PLOT_SELECTED_FILTERED_FIELDS = True
PLOT_LOWPASS_FIELDS = True
PLOT_HIGHPASS_FIELDS = True
PLOT_HIGHFREQ_RMS_FIELDS = True
PLOT_ORIGINAL_FIELDS = True
HIGHFREQ_RMS_SIGMA = 2.0

# Use the same symmetric color scale for all wanted times within each
# channel/component so B/W maps can be compared directly.
FILTER_FIELD_QUANTILE = 0.99
FILTER_PLOT_TYPE = "pcolormesh"
FILTER_PLOT_CMAP = "RdBu_r"
FILTER_PLOT_LEVELS = 21
FILTER_PLOT_FIGSIZE = (12, 8)

PLOT_FIGSIZE = (8, 5)

PLOT_POWER_LOG_SCALE = True

DPI = 250

REGENERATE_PLOTS = True

# ============================================================================
# WOrkaround
# ============================================================================

normalized_maps.LAT_RANGE = LAT_RANGE
normalized_maps.LON_RANGE = LON_RANGE

# ============================================================================
# Normalization
# ============================================================================

def make_target_normalizer(
    s,
    train_dataset: XarrayDataset,
):
    """Reproduce target normalization used during training."""
    norm_class = get_normalizer_class(s)

    return norm_class(
        mode=s.normalization_mode,
        exclude_channels=None,
    ).fit(
        train_dataset,
        dim="y",
    )


def get_normalizer_class(s):
    if s.normalization == "monthly":
        return MonthlyNormalize

    if s.normalization == "full":
        return Normalize

    raise ValueError(f"Unsupported normalization={s.normalization!r}")


def make_input_normalizer(
    s,
    train_dataset: XarrayDataset,
):
    """Reproduce input normalization used during training."""

    norm_class = get_normalizer_class(s)

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


def get_sample_times(dataset: XarrayDataset) -> np.ndarray:
    """Return one initialization time for every dataset sample."""

    time_dim = dataset.input_ds.earthml.guessed_dims.time
    if time_dim is None:
        raise ValueError("Could not determine input time dimension.")

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


def make_normalized_datasets_for_leadtime(
    s,
    leadtime: int,
) -> tuple[
    dict[str, XarrayDataset | None],
    object,
    object,
]:
    """Build datasets and apply the train-fitted normalization to x only."""

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
        leadtime_unit=s.leadtime_unit,
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
        input_realization_avg=s.input_realization_avg,
        interpolate_analysis=False,
        materialize=False,
        separate_training_by_init_period=None,
        defer_dataset_creation=False,
    )

    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    test_dataset = datasets["test"]

    if not isinstance(train_dataset, XarrayDataset):
        raise TypeError("Expected XarrayDataset for training data.")

    if val_dataset is not None and not isinstance(val_dataset, XarrayDataset):
        raise TypeError("Expected XarrayDataset for validation data.")

    if not isinstance(test_dataset, XarrayDataset):
        raise TypeError("Expected XarrayDataset for test data.")

    normalize_input = make_input_normalizer(
        s,
        train_dataset,
    )

    normalize_target = make_target_normalizer(
        s,
        train_dataset,
    )

    train_dataset.transform_x = normalize_input
    train_dataset.transform_y = normalize_target

    if val_dataset is not None:
        val_dataset.transform_x = normalize_input
        val_dataset.transform_y = normalize_target

    test_dataset.transform_x = normalize_input
    test_dataset.transform_y = normalize_target

    return (
        {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        },
        normalize_input,
        normalize_target,
    )


def _coordinate_indices(
    values: np.ndarray,
    bounds: tuple[float, float] | None,
) -> np.ndarray:
    """Return coordinate indices inside bounds, handling 0..360 longitudes."""

    if bounds is None:
        return np.arange(values.size)

    values = np.asarray(values, dtype=float)

    lo, hi = sorted(float(x) for x in bounds)

    if np.nanmin(values) >= 0.0 and lo < 0.0:
        lo = lo % 360.0
        hi = hi % 360.0
        if lo <= hi:
            selected = (values >= lo) & (values <= hi)
        else:
            selected = (values >= lo) | (values <= hi)
    else:
        selected = (values >= lo) & (values <= hi)

    indices = np.flatnonzero(selected)

    if indices.size == 0:
        raise ValueError(
            f"Requested bounds {bounds} select no coordinates from "
            f"[{np.nanmin(values)}, {np.nanmax(values)}]."
        )

    return indices


def subset_input_tensor_region(
    dataset: XarrayDataset,
    x: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subset normalized input tensor and mask to LAT_RANGE/LON_RANGE."""

    if LAT_RANGE is None and LON_RANGE is None:
        return x, mask

    source = dataset.input_ds

    lat_dim = source.earthml.guessed_dims.latitude
    lon_dim = source.earthml.guessed_dims.longitude
    if lat_dim is None or lon_dim is None:
        raise ValueError("Could not determine input latitude/longitude dimensions.")

    latitude = np.asarray(source[lat_dim].values)
    longitude = np.asarray(source[lon_dim].values)

    lat_indices = _coordinate_indices(latitude, LAT_RANGE)
    lon_indices = _coordinate_indices(longitude, LON_RANGE)

    lat_index = torch.as_tensor(lat_indices, dtype=torch.long, device=x.device)
    lon_index = torch.as_tensor(lon_indices, dtype=torch.long, device=x.device)

    x = x.index_select(-2, lat_index).index_select(-1, lon_index)

    mask = mask.index_select(-2, lat_index).index_select(-1, lon_index)

    return x, mask


def build_spectral_fields(
    s,
    dataset: XarrayDataset,
    corrected_ds: xr.Dataset | None,
    normalize_target,
    *,
    sample_index: int,
    init_time: pd.Timestamp,
    leadtime: int,
) -> dict[str, dict[int, xr.DataArray]]:
    """
    Build normalized fields used by the spectral diagnostics.

    Differences/corrections are calculated in target-normalized space.
    """

    requested = set(SPECTRAL_FIELDS)

    need_input = "input" in requested

    need_target = bool(
        requested
        & {
            "target",
            "fc_error",
            "mlfc_error",
            "ideal_correction",
        }
    )

    need_fc_target = bool(
        requested
        & {
            "fc_error",
            "ideal_correction",
            "ml_correction",
        }
    )

    need_corrected = bool(
        requested
        & {
            "corrected",
            "mlfc_error",
            "ml_correction",
        }
    )

    input_fields = (
        normalized_sample_fields(
            dataset,
            sample_index,
            kind="input",
            channels=PLOT_CHANNELS,
        )
        if need_input
        else {}
    )

    target_fields = (
        normalized_sample_fields(
            dataset,
            sample_index,
            kind="target",
            channels=PLOT_CHANNELS,
        )
        if need_target
        else {}
    )

    fc_target_fields = (
        normalized_fc_in_target_space(
            s,
            dataset,
            normalize_target,
            sample_index=sample_index,
            init_time=init_time,
            leadtime=leadtime,
            channels=PLOT_CHANNELS,
        )
        if need_fc_target
        else {}
    )

    corrected_fields = (
        normalized_corrected_fields(
            s,
            dataset,
            corrected_ds,
            normalize_target,
            sample_index=sample_index,
            init_time=init_time,
            leadtime=leadtime,
            channels=PLOT_CHANNELS,
        )
        if need_corrected and corrected_ds is not None
        else {}
    )

    fc_error = (
        subtract_aligned_field_collections(
            fc_target_fields,
            target_fields,
            name="normalized_fc_minus_target",
            units="normalized difference",
        )
        if "fc_error" in requested
        else {}
    )

    mlfc_error = (
        subtract_aligned_field_collections(
            corrected_fields,
            target_fields,
            name="normalized_corrected_minus_target",
            units="normalized difference",
        )
        if "mlfc_error" in requested
        else {}
    )

    ideal_correction = (
        subtract_aligned_field_collections(
            target_fields,
            fc_target_fields,
            name="normalized_ideal_correction",
            units="normalized correction",
        )
        if "ideal_correction" in requested
        else {}
    )

    ml_correction = (
        subtract_aligned_field_collections(
            corrected_fields,
            fc_target_fields,
            name="normalized_ml_correction",
            units="normalized correction",
        )
        if "ml_correction" in requested
        else {}
    )

    return {
        "input": input_fields,
        "target": target_fields,
        "corrected": corrected_fields,
        "fc_error": fc_error,
        "mlfc_error": mlfc_error,
        "ideal_correction": ideal_correction,
        "ml_correction": ml_correction,
    }


def radial_power_spectrum_dataarray(
    field: xr.DataArray,
    *,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(field.values, dtype=float)

    if values.ndim != 2:
        raise ValueError(
            f"Expected 2-D field, got shape={values.shape}"
        )

    valid = np.isfinite(values)

    tensor = torch.as_tensor(
        np.nan_to_num(values, nan=0.0),
        dtype=torch.float64,
    )
    mask = torch.as_tensor(valid, dtype=torch.bool)

    return radial_power_spectrum(
        tensor,
        mask,
        n_bins=n_bins,
    )

# ============================================================================
# Radial spectra
# ============================================================================

def spectral_power_modes(
    field: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return normalized radial wavenumber and power for every nonzero FFT mode.
    The field is the exact normalized physical input channel presented to the
    network. Invalid pixels are zero-filled after subtracting the valid-domain
    mean. Because the domain/mask is fixed, this is intended as a comparative
    timestep diagnostic rather than a physical isotropic spectrum in km^-1.
    """

    if field.ndim != 2:
        raise ValueError(
            f"Expected field with shape (H,W), got {tuple(field.shape)}"
        )

    if mask.shape != field.shape:
        raise ValueError(
            f"Field/mask mismatch: {tuple(field.shape)} != {tuple(mask.shape)}"
        )

    field = field.double()
    mask = mask.bool()
    valid_values = field[mask]

    if valid_values.numel() == 0:
        empty = torch.empty(0, dtype=field.dtype)
        return empty, empty

    if REMOVE_SPATIAL_MEAN:
        offset = valid_values.mean()
    else:
        offset = torch.zeros((), dtype=field.dtype, device=field.device)

    field = torch.where(
        mask,
        field - offset,
        torch.zeros_like(field),
    )

    height, width = field.shape

    if APPLY_HANN_WINDOW:
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

    # Full FFT keeps complete symmetric annuli, which makes radial-bin power
    # accounting simpler and avoids one-sided mode-count asymmetry
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

    max_k = radial_k.max()

    if not torch.isfinite(max_k) or max_k <= 0:
        empty = torch.empty(0, dtype=field.dtype)
        return empty, empty

    radial_k = radial_k / max_k

    # Remove the DC mode explicitly
    nonzero = radial_k > 0

    return radial_k[nonzero], power[nonzero]


def radial_power_spectrum(
    field: torch.Tensor,
    mask: torch.Tensor,
    *,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized radial-wavenumber bin centers and binned power."""

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
        if not in_bin.any():
            continue
        bin_power = power[in_bin].sum()
        spectrum[index] = (
            bin_power / total_power
            if NORMALIZE_POWER
            else bin_power
        )

    return (
        centers.detach().cpu().numpy(),
        spectrum.detach().cpu().numpy(),
    )


def collect_radial_power_spectra(
    s,
    dataset: XarrayDataset,
    corrected_ds: xr.Dataset | None,
    normalize_target,
    *,
    split: str,
    leadtime: int,
    n_bins: int,
) -> pd.DataFrame:
    """
    Collect radial spectra for WANTED_TIMES only.
    """

    sample_times = pd.to_datetime(
        get_sample_times(dataset)
    )

    # if WANTED_TIMES is None:
    selected_indices = np.arange(len(dataset))
    # else:
    #     wanted = pd.DatetimeIndex(
    #         pd.to_datetime(WANTED_TIMES)
    #     )

    #     available = pd.DatetimeIndex(
    #         sample_times
    #     ).unique()

    #     missing = wanted[~wanted.isin(available)]

    #     if len(missing):
    #         print(
    #             f"WARNING: WANTED_TIMES unavailable in "
    #             f"{split}: {list(missing)}"
    #         )

    #     selected_indices = np.flatnonzero(
    #         pd.DatetimeIndex(sample_times).isin(wanted)
    #     )

    print(
        f"Calculating spectra for "
        f"{len(selected_indices)}/{len(dataset)} samples."
    )

    rows: list[dict] = []
    occurrence_counter: dict[pd.Timestamp, int] = {}

    for position, sample_index in enumerate(
        selected_indices,
        start=1,
    ):
        sample_index = int(sample_index)

        init_time = pd.Timestamp(
            sample_times[sample_index]
        )

        print(
            f"  [{position}/{len(selected_indices)}] "
            f"{init_time}"
        )

        sample_within_init = occurrence_counter.get(
            init_time,
            0,
        )
        occurrence_counter[init_time] = (
            sample_within_init + 1
        )

        fields = build_spectral_fields(
            s,
            dataset,
            corrected_ds,
            normalize_target,
            sample_index=sample_index,
            init_time=init_time,
            leadtime=leadtime,
        )

        for field_name in SPECTRAL_FIELDS:
            channel_fields = fields[field_name]

            for channel, field in channel_fields.items():
                wavenumber, power = (
                    radial_power_spectrum_dataarray(
                        field,
                        n_bins=n_bins,
                    )
                )

                for k, p in zip(
                    wavenumber,
                    power,
                ):
                    rows.append({
                        "split": split,
                        "leadtime": leadtime,
                        "field": field_name,
                        "sample": sample_index,
                        "sample_within_init": (
                            sample_within_init
                        ),
                        "time": init_time,
                        "channel": channel,
                        "wavenumber": float(k),
                        "power": float(p),
                    })

    return pd.DataFrame(rows)


def aggregate_spectra_by_time(
    spectra: pd.DataFrame,
) -> pd.DataFrame:
    """Average multiple network samples belonging to the same initialization."""

    if spectra.empty:
        return spectra.copy()

    if not AGGREGATE_SAMPLES_PER_INIT:
        return spectra.copy()

    return (
        spectra
        .groupby(
            [
                "split",
                "leadtime",
                "field",
                "time",
                "channel",
                "wavenumber",
            ],
            as_index=False,
            sort=True,
        )["power"]
        .mean()
    )

# ============================================================================
# Spatial Fourier-filtered fields
# ============================================================================

def fourier_scale_components(
    field: torch.Tensor,
    mask: torch.Tensor,
    *,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct low- and high-wavenumber components of one input field."""

    if field.ndim != 2:
        raise ValueError(f"Expected field with shape (H,W), got {tuple(field.shape)}")
    if mask.shape != field.shape:
        raise ValueError(f"Field/mask mismatch: {tuple(field.shape)} != {tuple(mask.shape)}")
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must lie strictly between 0 and 1, got {threshold}.")
    field = field.double()
    mask = mask.bool()
    valid_values = field[mask]
    if valid_values.numel() == 0:
        nan_field = torch.full_like(field, torch.nan)
        return nan_field, nan_field.clone()
    offset = valid_values.mean() if REMOVE_SPATIAL_MEAN else torch.zeros((), dtype=field.dtype, device=field.device)
    prepared = torch.where(mask, field - offset, torch.zeros_like(field))
    height, width = prepared.shape
    # Do not apply the Hann window here. It is useful for estimating the
    # spectrum, but would attenuate the reconstructed field near the edges.
    fft = torch.fft.fft2(prepared)
    ky = torch.fft.fftfreq(height, dtype=prepared.dtype, device=prepared.device)
    kx = torch.fft.fftfreq(width, dtype=prepared.dtype, device=prepared.device)
    radial_k = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    max_k = radial_k.max()
    if not torch.isfinite(max_k) or max_k <= 0:
        nan_field = torch.full_like(field, torch.nan)
        return nan_field, nan_field.clone()
    radial_k = radial_k / max_k
    low_filter = radial_k < threshold
    high_filter = radial_k >= threshold
    lowpass = torch.fft.ifft2(fft * low_filter.to(dtype=fft.dtype)).real
    highpass = torch.fft.ifft2(fft * high_filter.to(dtype=fft.dtype)).real
    lowpass = torch.where(mask, lowpass, torch.nan)
    highpass = torch.where(mask, highpass, torch.nan)
    return lowpass, highpass


def high_frequency_rms(
    highpass: torch.Tensor,
    mask: torch.Tensor,
    *,
    sigma: float,
) -> torch.Tensor:
    """Return a smoothed local RMS amplitude of the high-pass component."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    values = highpass.detach().cpu().numpy().astype(float, copy=True)
    valid = mask.detach().cpu().numpy().astype(bool, copy=False)
    squared = np.where(valid, values ** 2, 0.0)
    weights = gaussian_filter(valid.astype(float), sigma=sigma, mode="nearest")
    smoothed_squared = gaussian_filter(squared, sigma=sigma, mode="nearest")
    rms = np.full_like(values, np.nan, dtype=float)
    good = valid & (weights > 0)
    rms[good] = np.sqrt(smoothed_squared[good] / weights[good])
    return torch.from_numpy(rms)


def input_spatial_coordinates(
    dataset: XarrayDataset,
) -> tuple[str, str, np.ndarray, np.ndarray]:
    """Return input latitude/longitude dimension names and coordinates."""

    source = dataset.input_ds

    lat_dim = source.earthml.guessed_dims.latitude
    lon_dim = source.earthml.guessed_dims.longitude
    if lat_dim is None or lon_dim is None:
        raise ValueError(
            "Could not determine input latitude/longitude dimensions."
        )

    return (
        lat_dim,
        lon_dim,
        np.asarray(source[lat_dim].values),
        np.asarray(source[lon_dim].values),
    )


def collect_selected_filtered_fields(
    dataset: XarrayDataset,
) -> list[dict]:
    """Collect original, filtered, and local high-frequency RMS fields for WANTED_TIMES."""

    if WANTED_TIMES is None:
        return []

    wanted = pd.DatetimeIndex(pd.to_datetime(WANTED_TIMES))

    sample_times = pd.to_datetime(get_sample_times(dataset))

    lat_dim, lon_dim, latitude, longitude = input_spatial_coordinates(dataset)

    available = pd.DatetimeIndex(sample_times).unique()
    missing = wanted[~wanted.isin(available)]

    if len(missing):
        print(f"WARNING: WANTED_TIMES unavailable in {PLOT_SPLIT}: {list(missing)}")

    records: list[dict] = []
    occurrence_counter: dict[pd.Timestamp, int] = {}

    for sample_index in range(len(dataset)):
        init_time = pd.Timestamp(sample_times[sample_index])
        if init_time not in wanted:
            continue

        sample_within_init = occurrence_counter.get(init_time, 0)
        occurrence_counter[init_time] = sample_within_init + 1

        x, _, mask, _ = dataset[sample_index]
        x = x.detach().cpu()
        mask = mask.detach().cpu().bool()
        n_channels = mask.shape[0]

        if x.shape[0] < n_channels:
            raise ValueError(f"Input tensor has fewer channels than mask: {tuple(x.shape)} vs {tuple(mask.shape)}")

        x = x[:n_channels]
        x, mask = subset_input_tensor_region(dataset, x, mask)

        lat_indices = _coordinate_indices(latitude, LAT_RANGE)
        lon_indices = _coordinate_indices(longitude, LON_RANGE)
        latitude = latitude[lat_indices]
        longitude = longitude[lon_indices]

        if x.shape[-2:] != (len(latitude), len(longitude)):
            raise ValueError(
                "Input tensor/spatial-coordinate shape mismatch: "
                f"{tuple(x.shape[-2:])} != {(len(latitude), len(longitude))}"
            )

        selected_channels = list(range(n_channels)) if PLOT_CHANNELS is None else PLOT_CHANNELS

        for channel in selected_channels:
            if channel < 0 or channel >= n_channels:
                raise ValueError(f"Requested channel {channel} is unavailable; valid channels are 0..{n_channels - 1}.")

            lowpass, highpass = fourier_scale_components(
                x[channel],
                mask[channel],
                threshold=HIGH_FREQ_THRESHOLD,
            )
            highfreq_rms = high_frequency_rms(
                highpass,
                mask[channel],
                sigma=HIGHFREQ_RMS_SIGMA,
            )

            original = torch.where(mask[channel], x[channel].double(), torch.nan)

            for component, values in (
                ("original", original),
                ("lowpass", lowpass),
                ("highpass", highpass),
                ("highfreq_rms", highfreq_rms),
            ):
                field = xr.DataArray(
                    values.numpy().astype(float, copy=False),
                    dims=(lat_dim, lon_dim),
                    coords={lat_dim: latitude, lon_dim: longitude},
                    name=f"normalized_input_{component}",
                    attrs={
                        "units": "normalized",
                        "long_name": component.replace("_", " "),
                    },
                )

                records.append({
                    "time": init_time,
                    "sample": sample_index,
                    "sample_within_init": sample_within_init,
                    "channel": channel,
                    "component": component,
                    "field": field,
                })
    return records


def shared_component_limit(
    records: list[dict],
    *,
    channel: int,
    component: str,
) -> float:
    """Shared symmetric limit across wanted times for one component."""

    arrays = []
    for record in records:
        if (
            record["channel"] == channel
            and record["component"] == component
        ):
            values = np.asarray(record["field"].values, dtype=float)
            values = np.abs(values[np.isfinite(values)])
            if values.size:
                arrays.append(values)

    if not arrays:
        return 1.0

    values = np.concatenate(arrays)
    vmax = float(np.quantile(values, FILTER_FIELD_QUANTILE))

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.max(values))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    return vmax


def shared_positive_component_limit(
    records: list[dict],
    *,
    channel: int,
    component: str,
) -> float:
    """Shared positive limit across wanted times for one component."""

    arrays = []
    for record in records:
        if record["channel"] == channel and record["component"] == component:
            values = np.asarray(record["field"].values, dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                arrays.append(values)

    if not arrays:
        return 1.0

    values = np.concatenate(arrays)
    vmax = float(np.quantile(values, FILTER_FIELD_QUANTILE))

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.max(values))

    return vmax if np.isfinite(vmax) and vmax > 0 else 1.0


def plot_selected_filtered_fields(
    dataset: XarrayDataset,
    extrema: pd.DataFrame,
    *,
    s,
    leadtime: int,
    output_dir: Path,
) -> None:
    """Plot original, filtered, and local high-frequency RMS fields for WANTED_TIMES."""

    if WANTED_TIMES is None:
        return

    records = collect_selected_filtered_fields(dataset)
    if not records:
        return

    labels: dict[pd.Timestamp, str] = {}
    if not extrema.empty:
        for _, row in extrema.iterrows():
            prefix = "B" if row["extreme"] == "best" else "W"
            labels[pd.Timestamp(row["time"])] = f"{prefix}{int(row['rank'])}"

    channels = sorted({int(record["channel"]) for record in records})
    symmetric_components = ("original", "lowpass", "highpass")

    limits = {
        (channel, component): shared_component_limit(records, channel=channel, component=component)
        for channel in channels
        for component in symmetric_components
    }

    limits.update({
        (channel, "highfreq_rms"): shared_positive_component_limit(
            records,
            channel=channel,
            component="highfreq_rms",
        )
        for channel in channels
    })

    enabled_components = []

    if PLOT_ORIGINAL_FIELDS:
        enabled_components.append("original")

    if PLOT_LOWPASS_FIELDS:
        enabled_components.append("lowpass")

    if PLOT_HIGHPASS_FIELDS:
        enabled_components.append("highpass")

    if PLOT_HIGHFREQ_RMS_FIELDS:
        enabled_components.append("highfreq_rms")

    lead_unit = getattr(s.leadtime_unit, "value", s.leadtime_unit)

    for record in records:
        component = record["component"]
        if component not in enabled_components:
            continue

        channel = int(record["channel"])
        init_time = pd.Timestamp(record["time"])
        sample_within_init = int(record["sample_within_init"])
        field = record["field"]

        label = labels.get(init_time, "")
        label_prefix = f"{label} · " if label else ""

        if component == "original":
            readable_component = "Original normalized input"
            scale_text = ""
        elif component == "lowpass":
            readable_component = "Low-pass normalized input"
            scale_text = f" · k < {HIGH_FREQ_THRESHOLD:g}"
        elif component == "highpass":
            readable_component = "High-pass normalized input"
            scale_text = f" · k >= {HIGH_FREQ_THRESHOLD:g}"
        else:
            readable_component = "High-frequency local RMS"
            scale_text = f" · k >= {HIGH_FREQ_THRESHOLD:g} · sigma={HIGHFREQ_RMS_SIGMA:g}"

        title = (
            f"{label_prefix}{readable_component} · channel {channel}{scale_text} · "
            f"{init_time.strftime('%Y-%m-%d %H:%M')} · lead {leadtime} {lead_unit}"
        )

        component_dir = (
            output_dir
            / "selected_filtered_fields"
            / f"channel_{channel}"
            / component
        )
        component_dir.mkdir(parents=True, exist_ok=True)

        label_token = f"_{label}" if label else ""
        filename = (
            f"{component}{label_token}_init_{init_time.strftime('%Y%m%dT%H%M')}"
            f"_sample_{sample_within_init}.png"
        )

        out_file = component_dir / filename

        if out_file.exists() and not REGENERATE_PLOTS:
            continue

        vmax = limits[(channel, component)]

        positive = component == "highfreq_rms"

        print(f"Saving {component} field {out_file}")

        plot_field_map(
            field,
            var=s.var_fc,
            title=title,
            out_file=out_file,
            cmap=FILTER_PLOT_CMAP,
            centered=not positive,
            vmin=0.0 if positive else -vmax,
            vmax=vmax,
            levels=FILTER_PLOT_LEVELS,
            plot_type=FILTER_PLOT_TYPE,
            figsize=FILTER_PLOT_FIGSIZE,
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
    Positive improvement always means MLFC is better.
    """

    time_range = (s.test_start, s.test_end)

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

    valid_lat_range = LAT_RANGE if LAT_RANGE is not None else lat_lon[0]
    valid_lon_range = LON_RANGE if LON_RANGE is not None else lat_lon[1]

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

    leadtime_dim = fc.earthml.guessed_dims.leadtime
    if leadtime_dim is None:
        raise ValueError("Could not determine leadtime dimension.")

    fc = fc.sel({leadtime_dim: [leadtime]})
    an = an.sel({leadtime_dim: [leadtime]})
    mlfc = mlfc.sel({leadtime_dim: [leadtime]})

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

    improvements = build_metric_improvements(
        fc_metrics,
        mlfc_metrics,
        metric=metric,
        baseline_model="fc",
        target_model="mlfc",
        improvement_units=(improvement_unit,),
    )

    if len(improvements) != 1:
        raise RuntimeError(
            "Expected exactly one improvement DataArray, "
            f"got {list(improvements)}"
        )

    comparison_model, improvement_da = next(iter(improvements.items()))

    print(f"Using performance metric: {comparison_model}")

    if (
        "leadtime" in improvement_da.dims
        or "leadtime" in improvement_da.coords
    ):
        improvement_da = improvement_da.sel(leadtime=leadtime)

    improvement_da = improvement_da.squeeze(drop=True)

    time_dim = improvement_da.earthml.guessed_dims.time
    if time_dim is None:
        raise ValueError(
            "Could not determine time dimension of metric improvement."
        )

    extra_dims = [
        dim
        for dim in improvement_da.dims
        if dim != time_dim
    ]

    if extra_dims:
        raise ValueError(
            "Metric improvement still contains unexpected dimensions: "
            f"{improvement_da.dims}"
        )

    result = pd.DataFrame({
        "time": pd.to_datetime(improvement_da[time_dim].values),
        "metric_improvement": pd.to_numeric(
            improvement_da.values,
            errors="coerce",
        ),
    })

    return (
        result
        .drop_duplicates(subset="time")
        .sort_values("time")
        .reset_index(drop=True)
    )


def get_performance_extrema(
    metric_df: pd.DataFrame,
    *,
    n_best: int,
    n_worst: int,
) -> pd.DataFrame:
    """Return ranked best/worst initialization times by metric improvement."""

    perf = (
        metric_df[["time", "metric_improvement"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .drop_duplicates(subset="time")
        .sort_values("metric_improvement")
        .reset_index(drop=True)
    )

    frames = []

    if n_worst > 0:
        worst = perf.head(n_worst).copy()
        worst["extreme"] = "worst"
        worst["rank"] = np.arange(1, len(worst) + 1)
        frames.append(worst)

    if n_best > 0:
        best = (
            perf.tail(n_best)
            .sort_values("metric_improvement", ascending=False)
            .copy()
        )
        best["extreme"] = "best"
        best["rank"] = np.arange(1, len(best) + 1)
        frames.append(best)

    if not frames:
        return pd.DataFrame(
            columns=["time", "metric_improvement", "extreme", "rank"]
        )

    return pd.concat(frames, ignore_index=True)


def add_extreme_labels(
    df: pd.DataFrame,
    extrema: pd.DataFrame,
) -> pd.DataFrame:
    """Attach B1.. / W1.. labels where available."""

    result = df.copy()
    result["extreme"] = pd.NA
    result["rank"] = pd.NA
    result["extreme_label"] = pd.NA

    if extrema.empty:
        return result

    labels = extrema.copy()
    labels["extreme_label"] = labels.apply(
        lambda row: (
            f"B{int(row['rank'])}"
            if row["extreme"] == "best"
            else f"W{int(row['rank'])}"
        ),
        axis=1,
    )

    result = result.drop(
        columns=["extreme", "rank", "extreme_label"]
    ).merge(
        labels[["time", "extreme", "rank", "extreme_label"]],
        on="time",
        how="left",
        validate="many_to_one",
    )

    return result

# ============================================================================
# Spectral statistics
# ============================================================================

def spearman_vs_wavenumber(
    spectra_with_performance: pd.DataFrame,
    *,
    channel: int,
) -> pd.DataFrame:
    """Spearman(power(k), metric improvement) across initialization times."""

    df = spectra_with_performance[
        spectra_with_performance["channel"] == channel
    ].copy()

    rows = []
    for wavenumber, group in df.groupby("wavenumber", sort=True):
        valid = (
            group[["power", "metric_improvement"]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if len(valid) < 3:
            continue

        result = spearmanr(
            valid["power"].to_numpy(dtype=float),
            valid["metric_improvement"].to_numpy(dtype=float),
        )

        rows.append({
            "wavenumber": float(wavenumber),
            "rho": float(result.statistic),
            "p_value": float(result.pvalue),
            "n": int(len(valid)),
        })

    return pd.DataFrame(rows)


def summarize_spectra(
    spectra: pd.DataFrame,
    *,
    group_columns: list[str],
) -> pd.DataFrame:
    """Median and interquartile spectrum for arbitrary groups."""

    return (
        spectra
        .groupby(
            group_columns + ["channel", "wavenumber"],
            as_index=False,
            sort=True,
        )["power"]
        .agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            mean="mean",
            count="count",
        )
    )

# ============================================================================
# Plotting
# ============================================================================

def _save_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)

    plt.close(fig)


def plot_selected_time_spectra(
    spectra: pd.DataFrame,
    *,
    field_name: str,
    channel: int,
    output_path: Path,
) -> None:
    """Overlay radial spectra for WANTED_TIMES, preserving individual samples."""

    if WANTED_TIMES is None:
        return

    wanted = pd.DatetimeIndex(pd.to_datetime(WANTED_TIMES))

    df = spectra[
        (spectra["channel"] == channel)
        & spectra["time"].isin(wanted)
    ].copy()

    if df.empty:
        return

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

    for (time, sample_within_init), group in df.groupby(
        ["time", "sample_within_init"],
        sort=True,
    ):
        label_prefix = group["extreme_label"].dropna()

        prefix = (
            f"{label_prefix.iloc[0]} "
            if not label_prefix.empty
            else ""
        )
        sample_suffix = (
            f" sample {int(sample_within_init)}"
            if df["sample_within_init"].nunique() > 1
            else ""
        )

        ax.plot(
            group["wavenumber"],
            group["power"],
            linewidth=1.5,
            label=(
                f"{prefix}{pd.Timestamp(time).strftime('%Y-%m-%d %H:%M')}"
                f"{sample_suffix}"
            ),
        )

    if PLOT_POWER_LOG_SCALE:
        ax.set_yscale("log")

    ax.axvline(
        HIGH_FREQ_THRESHOLD,
        linestyle=":",
        linewidth=1.0,
        color="black",
        alpha=0.7,
        label=f"high-k threshold = {HIGH_FREQ_THRESHOLD:g}",
    )

    ax.set_xlabel("Normalized radial spatial wavenumber")
    ax.set_ylabel(
        "Power fraction per radial bin"
        if NORMALIZE_POWER
        else "Radial-bin power"
    )

    label = SPECTRAL_FIELD_LABELS[field_name]
    ax.set_title(f"Normalized {label} radial spectra · channel {channel}")

    ax.legend(fontsize=7)

    _save_figure(fig, output_path)


def plot_best_worst_spectra(
    spectra_by_time: pd.DataFrame,
    extrema: pd.DataFrame,
    *,
    field_name: str,
    channel: int,
    output_path: Path,
) -> pd.DataFrame:
    """Plot median/IQR radial spectra for best and worst timesteps."""

    if extrema.empty:
        return pd.DataFrame()

    # spectra_by_time may already carry B/W annotation columns from
    # add_extreme_labels(). Drop them before merging the canonical extrema
    # table so pandas does not create extreme_x/extreme_y columns
    annotation_columns = [
        column
        for column in (
            "extreme",
            "rank",
            "extreme_label",
            "metric_improvement",
        )
        if column in spectra_by_time.columns
    ]

    tagged = spectra_by_time.drop(
        columns=annotation_columns
    ).merge(
        extrema[["time", "extreme", "rank", "metric_improvement"]],
        on="time",
        how="inner",
        validate="many_to_one",
    )

    tagged = tagged[tagged["channel"] == channel].copy()

    if tagged.empty:
        return pd.DataFrame()

    summary = summarize_spectra(
        tagged,
        group_columns=["extreme"],
    )

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

    for extreme in ("best", "worst"):
        group = summary[summary["extreme"] == extreme]
        if group.empty:
            continue

        x = group["wavenumber"].to_numpy(dtype=float)
        median = group["median"].to_numpy(dtype=float)
        q25 = group["q25"].to_numpy(dtype=float)
        q75 = group["q75"].to_numpy(dtype=float)

        line, = ax.plot(
            x,
            median,
            linewidth=2.0,
            label=f"{extreme} median",
        )

        ax.fill_between(
            x,
            q25,
            q75,
            alpha=0.18,
            color=line.get_color(),
            linewidth=0,
            label=f"{extreme} IQR",
        )

    if PLOT_POWER_LOG_SCALE:
        ax.set_yscale("log")
    ax.axvline(
        HIGH_FREQ_THRESHOLD,
        linestyle=":",
        linewidth=1.0,
        color="black",
        alpha=0.7,
        label=f"high-k threshold = {HIGH_FREQ_THRESHOLD:g}",
    )

    ax.set_xlabel("Normalized radial spatial wavenumber")
    ax.set_ylabel(
        "Power fraction per radial bin"
        if NORMALIZE_POWER
        else "Radial-bin power"
    )

    label = SPECTRAL_FIELD_LABELS[field_name]
    ax.set_title(f"Best vs worst normalized-{label} radial spectra · channel {channel}")

    ax.legend(fontsize=8)

    _save_figure(fig, output_path)

    return summary


def plot_all_test_median_spectrum(
    spectra_by_time: pd.DataFrame,
    *,
    field_name: str,
    channel: int,
    output_path: Path,
) -> pd.DataFrame:
    """Plot the median/IQR spectrum over all selected-split initialization times."""

    df = spectra_by_time[
        spectra_by_time["channel"] == channel
    ].copy()

    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby("wavenumber", as_index=False)["power"]
        .agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            mean="mean",
            count="count",
        )
    )

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    x = summary["wavenumber"].to_numpy(dtype=float)

    line, = ax.plot(
        x,
        summary["median"].to_numpy(dtype=float),
        linewidth=2.0,
        label="median",
    )

    ax.fill_between(
        x,
        summary["q25"].to_numpy(dtype=float),
        summary["q75"].to_numpy(dtype=float),
        alpha=0.18,
        color=line.get_color(),
        linewidth=0,
        label="IQR",
    )

    if PLOT_POWER_LOG_SCALE:
        ax.set_yscale("log")

    ax.axvline(
        HIGH_FREQ_THRESHOLD,
        linestyle=":",
        linewidth=1.0,
        color="black",
        alpha=0.7,
        label=f"high-k threshold = {HIGH_FREQ_THRESHOLD:g}",
    )

    ax.set_xlabel("Normalized radial spatial wavenumber")
    ax.set_ylabel(
        "Power fraction per radial bin"
        if NORMALIZE_POWER
        else "Radial-bin power"
    )

    label = SPECTRAL_FIELD_LABELS[field_name]
    ax.set_title(f"Normalized-{label} radial spectrum · {PLOT_SPLIT} · channel {channel}")

    ax.legend(fontsize=8)

    _save_figure(fig, output_path)

    return summary


def plot_spearman_vs_wavenumber(
    correlation_df: pd.DataFrame,
    *,
    field_name: str,
    channel: int,
    output_path: Path,
) -> None:
    """Plot Spearman(power(k), performance) against normalized wavenumber."""

    if correlation_df.empty:
        return

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

    ax.plot(
        correlation_df["wavenumber"],
        correlation_df["rho"],
        marker="o",
        markersize=4,
        linewidth=1.5,
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        color="black",
        alpha=0.7,
    )

    ax.axvline(
        HIGH_FREQ_THRESHOLD,
        linestyle=":",
        linewidth=1.0,
        color="black",
        alpha=0.7,
        label=f"high-k threshold = {HIGH_FREQ_THRESHOLD:g}",
    )

    ax.set_xlabel("Normalized radial spatial wavenumber")
    ax.set_ylabel(
        f"Spearman $\\rho$ with {PERFORMANCE_METRIC.upper()} improvement"
    )

    label = SPECTRAL_FIELD_LABELS[field_name]
    ax.set_title(
        f"Normalized-{label} spectral power vs ML performance · channel {channel}"
    )

    ax.legend(fontsize=8)

    _save_figure(fig, output_path)

# ============================================================================
# One experiment / leadtime
# ============================================================================

def process_leadtime(
    s,
    leadtime: int,
) -> None:
    print("=" * 80)
    print(
        f"{s.output_name}: leadtime={leadtime} "
        f"{getattr(s.leadtime_unit, 'value', s.leadtime_unit)}"
    )
    print(f"Spatial region: lat={LAT_RANGE}, lon={LON_RANGE}")

    (
        datasets,
        normalize_input,
        normalize_target,
    ) = make_normalized_datasets_for_leadtime(
        s,
        leadtime,
    )

    dataset = datasets[PLOT_SPLIT]

    need_corrected = bool(
        set(SPECTRAL_FIELDS)
        & {
            "corrected",
            "mlfc_error",
            "ml_correction",
        }
    )

    corrected_ds = (
        load_corrected_dataset(s, leadtime)
        if need_corrected
        else None
    )

    if (
        set(SPECTRAL_FIELDS)
        & {
            "target",
            "corrected",
            "fc_error",
            "mlfc_error",
            "ideal_correction",
            "ml_correction",
        }
    ) and s.target_mode != "analysis":
        raise ValueError(
            "These spectral comparisons currently require "
            "target_mode='analysis'."
        )

    print(
        f"Calculating normalized-input radial spectra for {PLOT_SPLIT}..."
    )

    raw_spectra = collect_radial_power_spectra(
        s,
        dataset,
        corrected_ds,
        normalize_target,
        split=PLOT_SPLIT,
        leadtime=leadtime,
        n_bins=SPECTRAL_N_BINS,
    )

    spectra_by_time = aggregate_spectra_by_time(raw_spectra)
    metric_df = pd.DataFrame()
    extrema = pd.DataFrame()

    if COMPUTE_PERFORMANCE:
        if PLOT_SPLIT != "test":
            print(
                "WARNING: performance/extrema plots require PLOT_SPLIT='test'; "
                "skipping them."
            )
        else:
            print(f"Calculating {PERFORMANCE_METRIC} improvement...")
            metric_df = get_metric_improvement_timeseries(
                s,
                leadtime=leadtime,
                metric=PERFORMANCE_METRIC,
                improvement_unit=PERFORMANCE_IMPROVEMENT_UNIT,
            )
            extrema = get_performance_extrema(
                metric_df,
                n_best=N_BEST_TIMESTEPS,
                n_worst=N_WORST_TIMESTEPS,
            )
            raw_spectra = add_extreme_labels(raw_spectra, extrema)
            spectra_by_time = add_extreme_labels(spectra_by_time, extrema)

    output_dir = (
        OUTPUT_ROOT
        / s.output_name
        / "normalized"
        / "radial_spectra"
        / PLOT_SPLIT
        / f"leadtime_{safe_label(leadtime)}"
        / f"lat_{safe_label(LAT_RANGE)}_lon_{safe_label(LON_RANGE)}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    raw_spectra.to_csv(
        output_dir / "radial_power_spectra_samples.csv",
        index=False,
    )

    spectra_by_time.to_csv(
        output_dir / "radial_power_spectra_by_time.csv",
        index=False,
    )

    if not metric_df.empty:
        metric_df.to_csv(
            output_dir / "metric_improvement.csv",
            index=False,
        )
        extrema.to_csv(
            output_dir / "performance_extrema.csv",
            index=False,
        )

    if PLOT_SELECTED_FILTERED_FIELDS and WANTED_TIMES is not None:
        plot_selected_filtered_fields(
            dataset,
            extrema,
            s=s,
            leadtime=leadtime,
            output_dir=output_dir,
        )

    fields = [
        field
        for field in SPECTRAL_FIELDS
        if field in raw_spectra["field"].unique()
    ]

    for field_name in fields:
        field_spectra = raw_spectra[
            raw_spectra["field"] == field_name
        ]

        field_spectra_by_time = spectra_by_time[
            spectra_by_time["field"] == field_name
        ]

        field_dir = output_dir / field_name

        channels = sorted(
            int(channel)
            for channel in field_spectra[
                "channel"
            ].dropna().unique()
        )

        for channel in channels:
            channel_dir = (
                field_dir
                / f"channel_{channel}"
            )
            channel_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            if PLOT_SELECTED_TIMES and WANTED_TIMES is not None:
                selected_file = (
                    channel_dir
                    / "selected_times_radial_spectra.png"
                )

                if REGENERATE_PLOTS or not selected_file.exists():
                    plot_selected_time_spectra(
                        field_spectra,
                        field_name=field_name,
                        channel=channel,
                        output_path=selected_file,
                    )


            if PLOT_ALL_TEST_MEDIAN:
                median_file = (
                    channel_dir
                    / "all_times_median_radial_spectrum.png"
                )

                if REGENERATE_PLOTS or not median_file.exists():
                    all_summary = plot_all_test_median_spectrum(
                        field_spectra_by_time,
                        field_name=field_name,
                        channel=channel,
                        output_path=median_file,
                    )

                    if not all_summary.empty:
                        all_summary.to_csv(
                            channel_dir
                            / "all_times_spectrum_summary.csv",
                            index=False,
                        )


            if PLOT_BEST_WORST_SPECTRA and not extrema.empty:
                bw_file = (
                    channel_dir
                    / "best_vs_worst_radial_spectra.png"
                )

                if REGENERATE_PLOTS or not bw_file.exists():
                    bw_summary = plot_best_worst_spectra(
                            field_spectra_by_time,
                            extrema,
                            field_name=field_name,
                            channel=channel,
                            output_path=bw_file,
                        )

                    if not bw_summary.empty:
                        bw_summary.to_csv(
                            channel_dir
                            / "best_vs_worst_spectrum_summary.csv",
                            index=False,
                        )

            if PLOT_SPEARMAN_VS_WAVENUMBER and not metric_df.empty:
                perf_spectra = field_spectra_by_time.merge(
                    metric_df,
                    on="time",
                    how="left",
                    validate="many_to_one",
                )

                correlation_df = spearman_vs_wavenumber(
                    perf_spectra,
                    channel=channel,
                )

                if not correlation_df.empty:
                    correlation_df.to_csv(
                        channel_dir / "spearman_vs_spatial_wavenumber.csv",
                        index=False,
                    )

                    corr_file = channel_dir / "spearman_vs_spatial_wavenumber.png"

                    if REGENERATE_PLOTS or not corr_file.exists():
                        plot_spearman_vs_wavenumber(
                            correlation_df,
                            field_name=field_name,
                            channel=channel,
                            output_path=corr_file,
                        )

                    high = correlation_df[
                        correlation_df["wavenumber"] >= HIGH_FREQ_THRESHOLD
                    ]
                    low = correlation_df[
                        correlation_df["wavenumber"] < HIGH_FREQ_THRESHOLD
                    ]

                    print(
                        f"Channel {channel}: mean rho below threshold="
                        f"{low['rho'].mean():.4f}, above threshold="
                        f"{high['rho'].mean():.4f}"
                    )

# ============================================================================
# Main
# ============================================================================

def main() -> None:
    settings = get_experiment_configs(
        EXPERIMENTS_ROOT,
        **FILTERS,
    )

    if not settings:
        raise RuntimeError("No matching experiments found.")

    print(f"Found {len(settings)} matching experiment(s).")

    for s in settings:
        leadtimes = (
            [int(x) for x in s.leadtimes]
            if LEADTIMES is None
            else LEADTIMES
        )

        missing_leadtimes = set(leadtimes) - set(int(x) for x in s.leadtimes)
        if missing_leadtimes:
            raise ValueError(
                f"Requested leadtimes {sorted(missing_leadtimes)} are not "
                f"available for {s.output_name}."
            )

        for leadtime in leadtimes:
            process_leadtime(
                s,
                int(leadtime),
            )


if __name__ == "__main__":
    main()
