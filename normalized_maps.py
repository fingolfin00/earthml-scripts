from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
import torch

from earthml import MonthlyNormalize, Normalize, XarrayDataset, get_experiment_configs
from earthml.plots import plot_field_map, safe_label

from train import make_train_test_datasets_for_leadtime

# ============================================================================
# Configuration
# ============================================================================

# EXP_NAME = "weather_atmo"
EXP_NAME = "weather_atmo_orography"

EXPERIMENTS_ROOT = Path(
    f"/work/cmcc/jd19424/ML/MLBC/experiments/{EXP_NAME}"
)

OUTPUT_ROOT = Path(
    f"/work/cmcc/jd19424/ML/MLBC/plots/{EXP_NAME}"
)

FILTERS = dict(
    var_fc="t2m",
    var_an="t2m",
    region_name="ConUS",
)

REGENERATE_PLOTS = True

# None -> all experiment leadtimes
LEADTIMES: list[int] | None = [72]

# Split whose normalized network inputs should be plotted
PLOT_SPLIT: Literal["train", "val", "test"] = "test"

# Initialization times to plot. None -> all times in the selected split
WANTED_TIMES: list[str] | None = [
    "2025-03-08T00:00:00", # B1 t2m 72h
    "2025-02-22T00:00:00", # B2 t2m 72h
    "2025-03-23T00:00:00", # B3 t2m 72h
    "2025-02-02T00:00:00", # B4 t2m 72h, near W3 in stb-vs-mean scatter
    "2025-03-07T00:00:00", # B5 t2m 72h
    "2025-02-21T00:00:00", # B6 t2m 72h
    "2025-04-06T00:00:00", # B7 t2m 72h
    "2025-02-01T00:00:00", # B8 t2m 72h
    "2025-04-07T00:00:00", # B9 t2m 72h
    "2025-03-06T00:00:00", # B10 t2m 72h

    "2025-02-13T00:00:00", # W1 t2m 72h
    "2025-01-03T00:00:00", # W2 t2m 72h
    "2025-02-06T00:00:00", # W3 t2m 72h
    "2025-02-18T00:00:00", # W4 t2m 72h
    "2025-02-16T00:00:00", # W5 t2m 72h
    "2025-01-01T12:00:00", # W6 t2m 72h
    "2025-02-12T12:00:00", # W7 t2m 72h
    "2025-04-24T00:00:00", # W8 t2m 72h
    "2025-03-29T00:00:00", # W9 t2m 72h
    "2025-05-25T12:00:00", # W10 t2m 72h
]
# WANTED_TIMES: list[str] | None = None

# If one initialization corresponds to multiple dataset samples, for example
# when realizations are samples:
#   None -> plot all matching samples
#   int  -> plot only this sample within each initialization (0-based)
SAMPLE_WITHIN_INIT: int | None = None

# What to plot
PLOT_NORMALIZED_INPUT = True
PLOT_NORMALIZED_TARGET = True
PLOT_NORMALIZED_CORRECTED = True
PLOT_INPUT_TARGET_DIFFERENCE = True
PLOT_CORRECTED_TARGET_DIFFERENCE = True

PLOT_FC_SQUARED_ERROR = True
PLOT_MLFC_SQUARED_ERROR = True
PLOT_SQUARED_ERROR_IMPROVEMENT = True

PLOT_ML_CORRECTION = True
PLOT_IDEAL_CORRECTION = True
PLOT_MEAN_ABS_ML_CORRECTION = False

# Channels to plot. None -> all available physical channels
# Extra input encoder channels are intentionally excluded
PLOT_INPUT_CHANNELS: list[int] | None = None
PLOT_TARGET_CHANNELS: list[int] | None = None

# Optional spatial subset applied only for plotting
LAT_RANGE: tuple[float, float] | None = None
LON_RANGE: tuple[float, float] | None = None

# Central degradation box
# LAT_RANGE = (30, 45)
# LON_RANGE = (-108, -90)

# Plot appearance
PLOT_TYPE = "pcolormesh"
PLOT_FIGSIZE = (12, 8)

# PLOT_CMAP = "bids:viridis_r"
# PLOT_CMAP = "cmasher:torch_r"
PLOT_CMAP = "google:turbo"
# PLOT_CMAP = "yorick:ncar_r" # interesting
# PLOT_CMAP = "chrisluts:I_Red"
# PLOT_CMAP = "cmasher:sunburst_r"

PLOT_CMAP_CENTERED = "cmocean:balance"

PLOT_LEVELS = 21
PLOT_TITLE = True
PLOT_LABELS = True

TITLE_SIZE = None
LABEL_SIZE = None
TICK_SIZE = None

DPI = 300

# Normalized fields are naturally centered around zero
# Input and target scales are calculated independently
NORMALIZED_INPUT_VMAX: float | None = None
NORMALIZED_INPUT_QUANTILE = 0.99

NORMALIZED_TARGET_VMAX: float | None = None
NORMALIZED_TARGET_QUANTILE = 0.99

NORMALIZED_CORRECTED_VMAX: float | None = None
NORMALIZED_CORRECTED_QUANTILE = 0.99

NORMALIZED_DIFFERENCE_VMAX: float | None = None
NORMALIZED_DIFFERENCE_QUANTILE = 0.99

NORMALIZED_SQUARED_ERROR_VMAX: float | None = None
NORMALIZED_SQUARED_ERROR_QUANTILE = 0.99

NORMALIZED_SQUARED_ERROR_IMPROVEMENT_VMAX: float | None = None
NORMALIZED_SQUARED_ERROR_IMPROVEMENT_QUANTILE = 0.99

# NORMALIZED_MEAN_ABS_ML_CORRECTION_VMAX: float | None = None
NORMALIZED_MEAN_ABS_ML_CORRECTION_VMAX: float | None = 0.35
NORMALIZED_MEAN_ABS_ML_CORRECTION_QUANTILE = 0.99

PLOT_TITLE_STRFTIME = "%d.%m.%Y %H:%M"

# ============================================================================
# Normalization
# ============================================================================

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

    # Encoding channels should not be normalized.
    n_excluded_channels = 0
    if s.seasonal_encoding and s.channel_representation != "init_period":
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

def normalized_fc_in_target_space(
    s,
    dataset: XarrayDataset,
    normalize_target,
    *,
    sample_index: int,
    init_time: pd.Timestamp,
    leadtime: int,
    channels: list[int] | None,
) -> dict[int, xr.DataArray]:
    if s.target_mode != "analysis":
        raise ValueError(
            "Ideal-correction maps currently require target_mode='analysis'."
        )

    da = dataset.input_ds[s.var_fc]

    time_dim = da.earthml.guessed_dims.time
    lead_dim = da.earthml.guessed_dims.leadtime

    if time_dim is None:
        raise ValueError("Could not determine forecast time dimension.")

    da = da.sel({time_dim: np.datetime64(init_time)})

    if lead_dim is not None and lead_dim in da.dims:
        da = da.sel({lead_dim: leadtime})

    da = da.squeeze(drop=True)

    lat_dim = dataset.target_ds.earthml.guessed_dims.latitude
    lon_dim = dataset.target_ds.earthml.guessed_dims.longitude

    da = da.sel({
        lat_dim: dataset.target_ds[lat_dim],
        lon_dim: dataset.target_ds[lon_dim],
    })

    realization_dim = da.earthml.guessed_dims.realization
    if realization_dim is not None and realization_dim in da.dims:
        if dataset.y.shape[1] == 1:
            da = da.mean(realization_dim, skipna=True)
        else:
            da = da.transpose(realization_dim, lat_dim, lon_dim)

    values = np.asarray(da.values)

    if values.ndim == 2:
        values = values[None, ...]
    elif values.ndim != 3:
        raise ValueError(
            "Expected FC field to resolve to (C,H,W), got "
            f"shape={values.shape}, dims={da.dims}"
        )

    tensor = torch.as_tensor(values, dtype=dataset.y.dtype)
    month = int(dataset.months[sample_index].item())
    tensor = normalize_target(
        tensor,
        months=month,
    ).detach().float().cpu()

    target_mask = dataset.y_mask[sample_index].detach().cpu().bool()

    selected_channels = (
        list(range(tensor.shape[0]))
        if channels is None
        else channels
    )

    latitude = np.asarray(dataset.target_ds[lat_dim].values)
    longitude = np.asarray(dataset.target_ds[lon_dim].values)

    fields = {}

    for channel in selected_channels:
        arr = tensor[channel].numpy().astype(float, copy=True)
        valid = target_mask[channel].numpy()
        arr[~valid] = np.nan

        fields[channel] = subset_spatially(
            xr.DataArray(
                arr,
                dims=(lat_dim, lon_dim),
                coords={
                    lat_dim: latitude,
                    lon_dim: longitude,
                },
                name="normalized_fc_target_space",
                attrs={
                    "units": "normalized",
                    "long_name": (
                        f"FC in target-normalized space "
                        f"channel {channel}"
                    ),
                },
            )
        )

    return fields

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
    sample_times = np.repeat(init_times, dataset.samples_per_init)

    if len(sample_times) != len(dataset):
        raise RuntimeError(
            "Cannot align initialization times with dataset samples: "
            f"{len(sample_times)=}, {len(dataset)=}"
        )

    return sample_times


def make_normalized_datasets_for_leadtime(
    s,
    leadtime: int,
) -> tuple[dict[str, XarrayDataset | None], object, object]:
    """Build datasets and return the train-fitted input/target normalizers."""
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


def get_spatial_coords(
    dataset: XarrayDataset,
    *,
    kind: Literal["input", "target"],
) -> tuple[str, str, np.ndarray, np.ndarray]:
    """Return spatial dimension names and coordinates for input/target tensors."""
    source = dataset.input_ds if kind == "input" else dataset.target_ds
    lat_dim = source.earthml.guessed_dims.latitude
    lon_dim = source.earthml.guessed_dims.longitude

    if lat_dim is None or lon_dim is None:
        raise ValueError(
            f"Could not determine latitude/longitude dimensions from {kind}_ds."
        )

    latitude = np.asarray(source[lat_dim].values)
    longitude = np.asarray(source[lon_dim].values)
    return lat_dim, lon_dim, latitude, longitude


def select_sample_indices(
    dataset: XarrayDataset,
    wanted_times: list[str] | None,
    sample_within_init: int | None,
) -> list[int]:
    """Select dataset sample indices for requested initialization times."""
    sample_times = pd.to_datetime(get_sample_times(dataset))

    if wanted_times is None:
        requested_times = None
    else:
        requested_times = pd.to_datetime(wanted_times)

        available_unique = pd.DatetimeIndex(sample_times).unique()
        missing = requested_times[~requested_times.isin(available_unique)]

        if len(missing) > 0:
            print(
                "WARNING: requested initialization times not available in "
                f"{PLOT_SPLIT}: {list(missing)}"
            )

    indices: list[int] = []

    for time_value in pd.DatetimeIndex(sample_times).unique():
        if requested_times is not None and time_value not in requested_times:
            continue

        matching = np.flatnonzero(sample_times == time_value)

        if sample_within_init is None:
            indices.extend(int(i) for i in matching)
            continue

        if sample_within_init < 0 or sample_within_init >= len(matching):
            print(
                "WARNING: sample_within_init="
                f"{sample_within_init} unavailable for {time_value}; "
                f"this initialization has {len(matching)} sample(s)."
            )
            continue

        indices.append(int(matching[sample_within_init]))

    return indices


def mean_absolute_field_collections(
    collections: list[dict[int, xr.DataArray]],
    *,
    name: str,
    sign: Literal["all", "positive", "negative"] = "all",
) -> dict[int, xr.DataArray]:
    """
    Mean absolute field over samples.

    sign="all":
        mean(|field|)

    sign="positive":
        mean(|field| | field > 0)

    sign="negative":
        mean(|field| | field < 0)
    """
    if not collections:
        return {}

    channels = set(collections[0])

    for fields in collections[1:]:
        if set(fields) != channels:
            raise ValueError(
                f"Channel mismatch while calculating {name}: "
                f"expected {sorted(channels)}, got {sorted(fields)}"
            )

    result: dict[int, xr.DataArray] = {}

    for channel in sorted(channels):
        aligned = xr.align(
            *[fields[channel] for fields in collections],
            join="exact",
        )

        stacked = xr.concat(
            aligned,
            dim="sample",
        )

        if sign == "all":
            values = abs(stacked)

        elif sign == "positive":
            values = abs(stacked).where(stacked > 0)

        elif sign == "negative":
            values = abs(stacked).where(stacked < 0)

        else:
            raise ValueError(f"Unsupported sign={sign!r}")

        mean_abs = values.mean(
            dim="sample",
            skipna=True,
        )

        mean_abs.name = name
        mean_abs.attrs = {
            "units": "normalized correction",
            "long_name": name.replace("_", " "),
        }

        result[channel] = mean_abs

    return result

# ============================================================================
# Tensor -> xarray field
# ============================================================================

def normalized_sample_fields(
    dataset: XarrayDataset,
    sample_index: int,
    *,
    kind: Literal["input", "target"],
    channels: list[int] | None,
) -> dict[int, xr.DataArray]:
    """Return normalized input or target channels for one dataset sample."""
    x, y, _, _ = dataset[sample_index]

    if kind == "input":
        tensor = x.detach().float().cpu()
        mask = dataset.x_mask[sample_index].detach().cpu().bool()
        name = "normalized_input"
    else:
        tensor = y.detach().float().cpu()
        mask = dataset.y_mask[sample_index].detach().cpu().bool()
        name = "normalized_target"

    if tensor.ndim != 3:
        raise ValueError(
            f"Expected normalized {kind} with shape (C,H,W), "
            f"got {tuple(tensor.shape)}"
        )

    if mask.ndim != 3:
        raise ValueError(
            f"Expected {kind} mask with shape (C,H,W), got {tuple(mask.shape)}"
        )

    if kind == "input":
        # Exclude appended encoder/mask channels, matching the diagnostics script.
        n_physical_channels = mask.shape[0]
        if tensor.shape[0] < n_physical_channels:
            raise ValueError(
                "Input tensor has fewer channels than x_mask: "
                f"{tuple(tensor.shape)} vs {tuple(mask.shape)}"
            )
        tensor = tensor[:n_physical_channels]
    elif tensor.shape != mask.shape:
        raise ValueError(
            "Target tensor/y_mask shape mismatch: "
            f"{tuple(tensor.shape)} != {tuple(mask.shape)}"
        )

    if tensor.shape != mask.shape:
        raise ValueError(
            f"{kind} tensor/mask shape mismatch after channel selection: "
            f"{tuple(tensor.shape)} != {tuple(mask.shape)}"
        )

    n_channels = tensor.shape[0]
    selected_channels = (
        list(range(n_channels))
        if channels is None
        else channels
    )

    invalid_channels = [
        channel
        for channel in selected_channels
        if channel < 0 or channel >= n_channels
    ]
    if invalid_channels:
        raise ValueError(
            f"Requested {kind} channels {invalid_channels} are unavailable. "
            f"Valid channels: 0..{n_channels - 1}"
        )

    lat_dim, lon_dim, latitude, longitude = get_spatial_coords(
        dataset,
        kind=kind,
    )

    if tensor.shape[-2:] != (len(latitude), len(longitude)):
        raise ValueError(
            f"{kind} tensor/spatial-coordinate shape mismatch: "
            f"tensor spatial shape={tuple(tensor.shape[-2:])}, "
            f"coordinates={(len(latitude), len(longitude))}"
        )

    fields: dict[int, xr.DataArray] = {}
    for channel in selected_channels:
        values = tensor[channel].numpy().astype(float, copy=True)
        valid = mask[channel].numpy()
        values[~valid] = np.nan

        field = xr.DataArray(
            values,
            dims=(lat_dim, lon_dim),
            coords={
                lat_dim: latitude,
                lon_dim: longitude,
            },
            name=name,
            attrs={
                "units": "normalized",
                "long_name": f"normalized {kind} channel {channel}",
            },
        )
        fields[channel] = subset_spatially(field)

    return fields



def _corrected_store_path(s, leadtime: int) -> Path | None:
    """Return the saved corrected-forecast store for the selected split."""
    if PLOT_SPLIT == "test":
        filename = "test_preds.zarr"
    elif PLOT_SPLIT == "train":
        filename = "train_preds.zarr"
    else:
        return None

    lead_unit = getattr(s.leadtime_unit, "value", s.leadtime_unit)
    candidates = [
        s.exp_dir / f"exp_{leadtime}_{s.leadtime_unit}" / filename,
        s.exp_dir / f"exp_{leadtime}_{lead_unit}" / filename,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def load_corrected_dataset(s, leadtime: int) -> xr.Dataset | None:
    """Load saved corrected forecasts for train/test, if available."""
    store = _corrected_store_path(s, leadtime)
    if store is None:
        print(
            "WARNING: corrected maps are unavailable for validation because "
            "the training workflow does not save val predictions."
        )
        return None

    if not store.exists():
        print(f"WARNING: corrected prediction store not found: {store}")
        return None

    return xr.open_zarr(store, consolidated=False)


def _select_corrected_field(
    corrected_ds: xr.Dataset,
    *,
    var: str,
    init_time: pd.Timestamp,
    leadtime: int,
) -> xr.DataArray:
    """Select one corrected physical field from the saved prediction store."""
    if var not in corrected_ds:
        if len(corrected_ds.data_vars) != 1:
            raise KeyError(
                f"Variable {var!r} not found in corrected dataset. "
                f"Available: {list(corrected_ds.data_vars)}"
            )
        da = next(iter(corrected_ds.data_vars.values()))
    else:
        da = corrected_ds[var]

    time_dim = da.earthml.guessed_dims.time
    lead_dim = da.earthml.guessed_dims.leadtime

    if time_dim is None:
        raise ValueError("Could not determine time dimension of corrected field.")

    da = da.sel({time_dim: np.datetime64(init_time)})

    if lead_dim is not None and lead_dim in da.dims:
        da = da.sel({lead_dim: leadtime})

    return da.squeeze(drop=True)


def normalized_corrected_fields(
    s,
    dataset: XarrayDataset,
    corrected_ds: xr.Dataset,
    normalize_target,
    *,
    sample_index: int,
    init_time: pd.Timestamp,
    leadtime: int,
    channels: list[int] | None,
) -> dict[int, xr.DataArray]:
    """
    Normalize the saved corrected forecast in target-normalized space.

    This comparison is exact for target_mode='analysis'. Other target modes
    represent residual/anomaly quantities, so a reconstructed physical forecast
    cannot be compared directly with normalized y and is rejected explicitly.
    """
    if s.target_mode != "analysis":
        raise ValueError(
            "Normalized corrected/input-target comparisons currently require "
            "target_mode='analysis'. The saved corrected zarr is a reconstructed "
            f"physical forecast, while target_mode={s.target_mode!r} uses a "
            "different target representation."
        )

    da = _select_corrected_field(
        corrected_ds,
        var=s.var_an,
        init_time=init_time,
        leadtime=leadtime,
    )

    lat_dim = dataset.target_ds.earthml.guessed_dims.latitude
    lon_dim = dataset.target_ds.earthml.guessed_dims.longitude
    if lat_dim is None or lon_dim is None:
        raise ValueError("Could not determine target spatial dimensions.")

    # Match target spatial layout exactly.
    da = da.sel({
        lat_dim: dataset.target_ds[lat_dim],
        lon_dim: dataset.target_ds[lon_dim],
    })

    realization_dim = da.earthml.guessed_dims.realization
    if realization_dim is not None and realization_dim in da.dims:
        if dataset.y.shape[1] == 1:
            da = da.mean(realization_dim, skipna=True)
        else:
            da = da.transpose(realization_dim, lat_dim, lon_dim)

    values = np.asarray(da.values)
    if values.ndim == 2:
        values = values[None, ...]
    elif values.ndim != 3:
        raise ValueError(
            "Expected corrected field to resolve to (C,H,W), got "
            f"shape={values.shape}, dims={da.dims}"
        )

    tensor = torch.as_tensor(values, dtype=dataset.y.dtype)
    month = int(dataset.months[sample_index].item())
    tensor = normalize_target(tensor, months=month).detach().float().cpu()

    target_mask = dataset.y_mask[sample_index].detach().cpu().bool()
    if tensor.shape != target_mask.shape:
        raise ValueError(
            "Corrected normalized tensor/target mask shape mismatch: "
            f"{tuple(tensor.shape)} != {tuple(target_mask.shape)}"
        )

    n_channels = tensor.shape[0]
    selected_channels = list(range(n_channels)) if channels is None else channels
    bad = [c for c in selected_channels if c < 0 or c >= n_channels]
    if bad:
        raise ValueError(
            f"Requested corrected channels {bad} are unavailable. "
            f"Valid channels: 0..{n_channels - 1}"
        )

    latitude = np.asarray(dataset.target_ds[lat_dim].values)
    longitude = np.asarray(dataset.target_ds[lon_dim].values)
    fields: dict[int, xr.DataArray] = {}

    for channel in selected_channels:
        arr = tensor[channel].numpy().astype(float, copy=True)
        valid = target_mask[channel].numpy()
        arr[~valid] = np.nan
        fields[channel] = subset_spatially(
            xr.DataArray(
                arr,
                dims=(lat_dim, lon_dim),
                coords={lat_dim: latitude, lon_dim: longitude},
                name="normalized_corrected",
                attrs={
                    "units": "normalized",
                    "long_name": f"normalized corrected channel {channel}",
                },
            )
        )

    return fields

# ============================================================================
# Subtraction helpers
# ============================================================================

def square_field_collection(
    fields: dict[int, xr.DataArray],
    *,
    name: str,
) -> dict[int, xr.DataArray]:
    """Square every field in a channel collection."""
    result: dict[int, xr.DataArray] = {}

    for channel, field in fields.items():
        squared = (field ** 2).copy()
        squared.name = name
        squared.attrs = {
            "units": "normalized squared error",
            "long_name": name.replace("_", " "),
        }
        result[channel] = squared

    return result


def subtract_aligned_field_collections(
    lhs: dict[int, xr.DataArray],
    rhs: dict[int, xr.DataArray],
    *,
    name: str,
    units: str,
) -> dict[int, xr.DataArray]:
    """Subtract matching channel collections."""
    if not lhs or not rhs:
        return {}

    if set(lhs) != set(rhs):
        raise ValueError(
            f"Cannot align channels for {name}: "
            f"lhs={list(lhs)}, rhs={list(rhs)}."
        )

    result: dict[int, xr.DataArray] = {}

    for channel in lhs:
        a, b = xr.align(
            lhs[channel],
            rhs[channel],
            join="exact",
        )

        diff = (a - b).copy()
        diff.name = name
        diff.attrs = {
            "units": units,
            "long_name": name.replace("_", " "),
        }
        result[channel] = diff

    return result


def subtract_field_collections(
    lhs: dict[int, xr.DataArray],
    rhs: dict[int, xr.DataArray],
    *,
    name: str,
) -> dict[int, xr.DataArray]:
    """Subtract channel collections, broadcasting a single channel if needed."""
    if not lhs or not rhs:
        return {}

    pairs: list[tuple[int, int, int]] = []
    lhs_keys = list(lhs)
    rhs_keys = list(rhs)

    if set(lhs_keys) == set(rhs_keys):
        pairs = [(channel, channel, channel) for channel in lhs_keys]
    elif len(rhs_keys) == 1:
        rhs_channel = rhs_keys[0]
        pairs = [(channel, channel, rhs_channel) for channel in lhs_keys]
    elif len(lhs_keys) == 1:
        lhs_channel = lhs_keys[0]
        pairs = [(channel, lhs_channel, channel) for channel in rhs_keys]
    else:
        raise ValueError(
            f"Cannot align channels for {name}: "
            f"lhs={lhs_keys}, rhs={rhs_keys}."
        )

    result: dict[int, xr.DataArray] = {}
    for output_channel, lhs_channel, rhs_channel in pairs:
        a, b = xr.align(lhs[lhs_channel], rhs[rhs_channel], join="exact")
        diff = (a - b).copy()
        diff.name = name
        diff.attrs = {
            "units": "normalized difference",
            "long_name": name.replace("_", " "),
        }
        result[output_channel] = diff

    return result


def subset_spatially(field: xr.DataArray) -> xr.DataArray:
    """Apply optional latitude/longitude plotting subset."""
    lat_dim = field.earthml.guessed_dims.latitude
    lon_dim = field.earthml.guessed_dims.longitude

    if lat_dim is None or lon_dim is None:
        raise ValueError("Could not determine spatial dimensions of field.")

    if LAT_RANGE is not None:
        field = field.sel({lat_dim: _coord_slice(field[lat_dim], LAT_RANGE)})

    if LON_RANGE is not None:
        field = field.sel({lon_dim: _coord_slice(field[lon_dim], LON_RANGE)})

    return field


def _coord_slice(
    coord: xr.DataArray,
    bounds: tuple[float, float],
) -> slice:
    """Build a slice respecting ascending or descending coordinate order."""
    start, end = bounds

    if coord.size < 2:
        return slice(start, end)

    ascending = bool(coord.values[0] <= coord.values[-1])

    if ascending:
        return slice(min(start, end), max(start, end))

    return slice(max(start, end), min(start, end))

# ============================================================================
# Plot scaling
# ============================================================================

def get_symmetric_limit(
    fields: list[xr.DataArray],
    quantile: float | None,
) -> float:
    """Return a common positive symmetric color limit."""
    values = np.concatenate(
        [
            np.asarray(field.values, dtype=float).ravel()
            for field in fields
        ]
    )

    values = np.abs(values[np.isfinite(values)])

    if values.size == 0:
        return 1.0

    if quantile is None:
        vmax = float(np.max(values))
    else:
        vmax = float(np.quantile(values, quantile))

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.max(values))

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    return vmax

# ============================================================================
# Plot one experiment / leadtime
# ============================================================================

def plot_normalized_leadtime(
    s,
    leadtime: int,
) -> int:
    datasets, _, normalize_target = make_normalized_datasets_for_leadtime(
        s,
        leadtime,
    )
    dataset = datasets[PLOT_SPLIT]

    if dataset is None:
        print(
            f"Skipping {s.output_name}, leadtime={leadtime}: "
            f"split {PLOT_SPLIT!r} is unavailable."
        )
        return 0

    need_corrected = (
        PLOT_NORMALIZED_CORRECTED
        or PLOT_CORRECTED_TARGET_DIFFERENCE
        or PLOT_MLFC_SQUARED_ERROR
        or PLOT_SQUARED_ERROR_IMPROVEMENT
        or PLOT_ML_CORRECTION
        or PLOT_MEAN_ABS_ML_CORRECTION
    )

    need_fc_target = (
        PLOT_IDEAL_CORRECTION
        or PLOT_ML_CORRECTION
        or PLOT_MEAN_ABS_ML_CORRECTION
    )

    corrected_ds = (
        load_corrected_dataset(s, leadtime)
        if need_corrected
        else None
    )

    if (
        (PLOT_INPUT_TARGET_DIFFERENCE or need_corrected)
        and s.target_mode != "analysis"
    ):
        raise ValueError(
            "Input/target/corrected comparisons in this script currently "
            "require target_mode='analysis'."
        )

    sample_times = pd.to_datetime(get_sample_times(dataset))
    sample_indices = select_sample_indices(
        dataset,
        WANTED_TIMES,
        SAMPLE_WITHIN_INIT,
    )

    if not sample_indices:
        print(
            f"No matching samples for {s.output_name}, "
            f"leadtime={leadtime}, split={PLOT_SPLIT}."
        )
        return 0

    selected_samples = []
    occurrence_counter: dict[pd.Timestamp, int] = {}

    for sample_index in sample_indices:
        init_time = pd.Timestamp(sample_times[sample_index])
        sample_within_init = occurrence_counter.get(init_time, 0)
        occurrence_counter[init_time] = sample_within_init + 1

        need_input = (
            PLOT_NORMALIZED_INPUT
            or PLOT_INPUT_TARGET_DIFFERENCE
            or PLOT_FC_SQUARED_ERROR
            or PLOT_SQUARED_ERROR_IMPROVEMENT
        )

        need_target = (
            PLOT_NORMALIZED_TARGET
            or PLOT_INPUT_TARGET_DIFFERENCE
            or PLOT_CORRECTED_TARGET_DIFFERENCE
            or PLOT_FC_SQUARED_ERROR
            or PLOT_MLFC_SQUARED_ERROR
            or PLOT_SQUARED_ERROR_IMPROVEMENT
        )

        input_fields = (
            normalized_sample_fields(
                dataset,
                sample_index,
                kind="input",
                channels=PLOT_INPUT_CHANNELS,
            )
            if need_input
            else {}
        )

        target_fields = (
            normalized_sample_fields(
                dataset,
                sample_index,
                kind="target",
                channels=PLOT_TARGET_CHANNELS,
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
                channels=PLOT_TARGET_CHANNELS,
            )
            if need_fc_target
            else {}
        )

        ideal_correction = (
            subtract_aligned_field_collections(
                target_fields,
                fc_target_fields,
                name="normalized_ideal_correction",
                units="normalized correction",
            )
            if PLOT_IDEAL_CORRECTION
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
                channels=PLOT_TARGET_CHANNELS,
            )
            if corrected_ds is not None and need_corrected
            else {}
        )

        ml_correction = (
            subtract_aligned_field_collections(
                corrected_fields,
                fc_target_fields,
                name="normalized_actual_correction",
                units="normalized correction",
            )
            if (
                PLOT_ML_CORRECTION
                or PLOT_MEAN_ABS_ML_CORRECTION
            )
            else {}
        )

        input_target_diff = (
            subtract_field_collections(
                input_fields,
                target_fields,
                name="normalized_input_minus_target",
            )
            if PLOT_INPUT_TARGET_DIFFERENCE
            else {}
        )

        corrected_target_diff = (
            subtract_field_collections(
                corrected_fields,
                target_fields,
                name="normalized_corrected_minus_target",
            )
            if PLOT_CORRECTED_TARGET_DIFFERENCE
            else {}
        )

        fc_squared_error = (
            square_field_collection(
                input_target_diff,
                name="normalized_fc_squared_error",
            )
            if (
                PLOT_FC_SQUARED_ERROR
                or PLOT_SQUARED_ERROR_IMPROVEMENT
            )
            else {}
        )

        mlfc_squared_error = (
            square_field_collection(
                corrected_target_diff,
                name="normalized_mlfc_squared_error",
            )
            if (
                PLOT_MLFC_SQUARED_ERROR
                or PLOT_SQUARED_ERROR_IMPROVEMENT
            )
            else {}
        )

        squared_error_improvement = (
            subtract_aligned_field_collections(
                fc_squared_error,
                mlfc_squared_error,
                name="normalized_squared_error_improvement",
                units="normalized squared error improvement",
            )
            if PLOT_SQUARED_ERROR_IMPROVEMENT
            else {}
        )

        selected_samples.append(
            (
                sample_index,
                init_time,
                sample_within_init,
                input_fields,
                target_fields,
                fc_target_fields,
                ideal_correction,
                corrected_fields,
                ml_correction,
                input_target_diff,
                corrected_target_diff,
                fc_squared_error,
                mlfc_squared_error,
                squared_error_improvement,
            )
        )

    def collect(position: int) -> list[xr.DataArray]:
        return [
            field
            for sample in selected_samples
            for field in sample[position].values()
        ]

    all_input = collect(3)
    all_target = collect(4)
    # all_fc_target = collect(5) # target normalized in forecast space
    all_ideal_correction = collect(6)
    all_corrected = collect(7)
    all_ml_correction = collect(8)
    all_input_diff = collect(9)
    all_corrected_diff = collect(10)
    all_fc_squared_error = collect(11)
    all_mlfc_squared_error = collect(12)
    all_squared_error_improvement = collect(13)

    def resolve_vmax(
        fields: list[xr.DataArray],
        manual: float | None,
        quantile: float | None,
    ) -> float | None:
        if not fields:
            return None
        if manual is not None:
            return float(manual)
        return get_symmetric_limit(fields, quantile)

    input_vmax = resolve_vmax(
        all_input,
        NORMALIZED_INPUT_VMAX,
        NORMALIZED_INPUT_QUANTILE,
    )

    target_vmax = resolve_vmax(
        all_target,
        NORMALIZED_TARGET_VMAX,
        NORMALIZED_TARGET_QUANTILE,
    )

    corrected_vmax = resolve_vmax(
        all_corrected,
        NORMALIZED_CORRECTED_VMAX,
        NORMALIZED_CORRECTED_QUANTILE,
    )

    difference_vmax = resolve_vmax(
        all_input_diff + all_corrected_diff + all_ideal_correction + all_ml_correction,
        NORMALIZED_DIFFERENCE_VMAX,
        NORMALIZED_DIFFERENCE_QUANTILE,
    )

    def get_positive_limit(
        fields: list[xr.DataArray],
        quantile: float | None,
    ) -> float:
        """Return positive upper color limit for nonnegative fields."""
        values = np.concatenate(
            [
                np.asarray(field.values, dtype=float).ravel()
                for field in fields
            ]
        )

        values = values[np.isfinite(values)]

        if values.size == 0:
            return 1.0

        if quantile is None:
            vmax = float(np.max(values))
        else:
            vmax = float(np.quantile(values, quantile))

        if not np.isfinite(vmax) or vmax <= 0:
            vmax = float(np.max(values))

        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0

        return vmax

    squared_error_fields = (
        all_fc_squared_error
        + all_mlfc_squared_error
    )

    if squared_error_fields:
        squared_error_vmax = (
            float(NORMALIZED_SQUARED_ERROR_VMAX)
            if NORMALIZED_SQUARED_ERROR_VMAX is not None
            else get_positive_limit(
                squared_error_fields,
                NORMALIZED_SQUARED_ERROR_QUANTILE,
            )
        )
    else:
        squared_error_vmax = None

    squared_error_improvement_vmax = resolve_vmax(
        all_squared_error_improvement,
        NORMALIZED_SQUARED_ERROR_IMPROVEMENT_VMAX,
        NORMALIZED_SQUARED_ERROR_IMPROVEMENT_QUANTILE,
    )

    lead_unit = getattr(s.leadtime_unit, "value", str(s.leadtime_unit))
    lead_unit_label = {
        "hours": "h",
        "hour": "h",
        "months": "M",
        "month": "M",
    }.get(str(lead_unit).lower(), str(lead_unit))

    saved = 0

    # ----------------------------------------------------------------------------
    # Mean absolute ML corrections
    # ----------------------------------------------------------------------------

    ml_correction_samples = [
        sample[8]
        for sample in selected_samples
        if sample[8]
    ]

    if PLOT_MEAN_ABS_ML_CORRECTION:
        mean_abs_ml_correction = mean_absolute_field_collections(
            ml_correction_samples,
            name="normalized_mean_absolute_ml_correction",
            sign="all",
        )

        mean_abs_positive_ml_correction = mean_absolute_field_collections(
            ml_correction_samples,
            name="normalized_mean_absolute_positive_ml_correction",
            sign="positive",
        )

        mean_abs_negative_ml_correction = mean_absolute_field_collections(
            ml_correction_samples,
            name="normalized_mean_absolute_negative_ml_correction",
            sign="negative",
        )
    else:
        mean_abs_ml_correction = {}
        mean_abs_positive_ml_correction = {}
        mean_abs_negative_ml_correction = {}

        mean_abs_ml_correction_fields = list(
            mean_abs_ml_correction.values()
        )

    mean_abs_ml_correction_fields = (
        list(mean_abs_ml_correction.values())
        + list(mean_abs_positive_ml_correction.values())
        + list(mean_abs_negative_ml_correction.values())
    )

    if mean_abs_ml_correction_fields:
        mean_abs_ml_correction_vmax = (
            float(NORMALIZED_MEAN_ABS_ML_CORRECTION_VMAX)
            if NORMALIZED_MEAN_ABS_ML_CORRECTION_VMAX is not None
            else get_positive_limit(
                mean_abs_ml_correction_fields,
                NORMALIZED_MEAN_ABS_ML_CORRECTION_QUANTILE,
            )
        )
    else:
        mean_abs_ml_correction_vmax = None
        

    aggregate_correction_maps = (
        (
            "mean_absolute_ml_correction",
            mean_abs_ml_correction,
            r"Mean absolute ML correction: $\mathrm{mean}(|MLFC-FC|)$",
        ),
        (
            "mean_absolute_positive_ml_correction",
            mean_abs_positive_ml_correction,
            r"Mean absolute positive ML correction: "
            r"$\mathrm{mean}(|MLFC-FC|\,|\,MLFC-FC>0)$",
        ),
        (
            "mean_absolute_negative_ml_correction",
            mean_abs_negative_ml_correction,
            r"Mean absolute negative ML correction: "
            r"$\mathrm{mean}(|MLFC-FC|\,|\,MLFC-FC<0)$",
        ),
    )

    if (
        PLOT_MEAN_ABS_ML_CORRECTION
        and mean_abs_ml_correction_vmax is not None
    ):
        n_samples = len(selected_samples)

        selected_times = pd.DatetimeIndex(
            sample_times[sample_indices]
        )

        sample_range_label = (
            f"n{len(sample_indices)}"
            f"_{selected_times.min():%Y%m%d}"
            f"-{selected_times.max():%Y%m%d}"
        )

        for output_name, fields, title_prefix in aggregate_correction_maps:
            if not fields:
                continue

            output_dir = (
                OUTPUT_ROOT
                / s.output_name
                / "normalized"
                / "maps"
                / output_name
                / PLOT_SPLIT
                / f"leadtime_{safe_label(leadtime)}"
                / sample_range_label
                / f"lat_{safe_label(LAT_RANGE)}_lon_{safe_label(LON_RANGE)}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            for channel, field in fields.items():
                filename = (
                    f"{s.var_an}_{output_name}"
                    f"_channel_{channel}"
                    f"_lead_{safe_label(leadtime)}.png"
                )
                out_file = output_dir / filename

                if out_file.exists() and not REGENERATE_PLOTS:
                    continue

                valid_values = field.values[np.isfinite(field.values)]

                if valid_values.size:
                    spatial_mean = float(np.mean(valid_values))
                    spatial_std = float(np.std(valid_values, ddof=0))
                else:
                    spatial_mean = np.nan
                    spatial_std = np.nan

                title = (
                    f"{title_prefix}"
                    f" · channel {channel}"
                    f" · {PLOT_SPLIT}"
                    f" · lead {leadtime}{lead_unit_label}"
                    f" · N={n_samples}"
                    f" · mean {spatial_mean:.3f}"
                    f" · std {spatial_std:.3f}"
                    if PLOT_TITLE
                    else None
                )

                lat_dim = field.earthml.guessed_dims.latitude
                lon_dim = field.earthml.guessed_dims.longitude

                if lat_dim is None or lon_dim is None:
                    raise ValueError(
                        "Could not determine latitude/longitude dimensions "
                        f"for {output_name}."
                    )

                print(f"Saving {output_name} map {out_file}")

                plot_field_map(
                    da=field,
                    var=s.var_an,
                    title=title,
                    out_file=out_file,
                    cmap=PLOT_CMAP,
                    centered=False,
                    vmin=0.0,
                    vmax=mean_abs_ml_correction_vmax,
                    levels=PLOT_LEVELS,
                    plot_type=PLOT_TYPE,
                    figsize=PLOT_FIGSIZE,
                    spatial_dims=(lat_dim, lon_dim),
                    regions=None,
                    plot_title=PLOT_TITLE,
                    plot_labels=PLOT_LABELS,
                    title_size=TITLE_SIZE,
                    label_size=LABEL_SIZE,
                    tick_size=TICK_SIZE,
                    dpi=DPI,
                )

                saved += 1

    # ----------------------------------------------------------------------------
    # All other maps
    # ----------------------------------------------------------------------------

    for (
        sample_index,
        init_time,
        sample_within_init,
        input_fields,
        target_fields,
        fc_target_fields,
        ideal_correction,
        corrected_fields,
        ml_correction,
        input_target_diff,
        corrected_target_diff,
        fc_squared_error,
        mlfc_squared_error,
        squared_error_improvement,
    ) in selected_samples:
        time_label = init_time.strftime("%Y%m%dT%H%M")
        time_title = init_time.strftime(PLOT_TITLE_STRFTIME)

        collections = (
            (
                PLOT_NORMALIZED_INPUT,
                "input",
                input_fields,
                input_vmax,
                s.var_fc,
                "Normalized input",
                "input_fields",
                True,
            ),
            (
                PLOT_NORMALIZED_TARGET,
                "target",
                target_fields,
                target_vmax,
                s.var_an,
                f"Normalized target ({s.target_mode})",
                "target_fields",
                True,
            ),
            (
                PLOT_NORMALIZED_CORRECTED,
                "corrected",
                corrected_fields,
                corrected_vmax,
                s.var_an,
                "Normalized corrected input",
                "input_corrected_fields",
                True,
            ),
            (
                PLOT_INPUT_TARGET_DIFFERENCE,
                "input_minus_target",
                input_target_diff,
                difference_vmax,
                s.var_an,
                "Normalized FC - AN",
                "input_minus_target",
                True,
            ),
            (
                PLOT_CORRECTED_TARGET_DIFFERENCE,
                "corrected_minus_target",
                corrected_target_diff,
                difference_vmax,
                s.var_an,
                "Normalized MLFC - AN",
                "input_corrected_minus_target",
                True,
            ),
            (
                PLOT_FC_SQUARED_ERROR,
                "fc_squared_error",
                fc_squared_error,
                squared_error_vmax,
                s.var_an,
                r"Normalized FC squared error: $(FC-AN)^2$",
                "fc_squared_error",
                False,
            ),
            (
                PLOT_MLFC_SQUARED_ERROR,
                "mlfc_squared_error",
                mlfc_squared_error,
                squared_error_vmax,
                s.var_an,
                r"Normalized MLFC squared error: $(MLFC-AN)^2$",
                "mlfc_squared_error",
                False,
            ),
            (
                PLOT_SQUARED_ERROR_IMPROVEMENT,
                "squared_error_improvement",
                squared_error_improvement,
                squared_error_improvement_vmax,
                s.var_an,
                r"Squared-error improvement: $(FC-AN)^2-(MLFC-AN)^2$",
                "squared_error_improvement",
                True,
            ),
            (
                PLOT_IDEAL_CORRECTION,
                "ideal_correction",
                ideal_correction,
                difference_vmax,
                s.var_an,
                r"Ideal correction: $AN-FC$",
                "ideal_correction",
                True,
            ),
            (
                PLOT_ML_CORRECTION,
                "ml_correction",
                ml_correction,
                difference_vmax,
                s.var_an,
                r"ML correction: $MLFC-FC$",
                "ml_correction",
                True,
            ),
        )

        for (
            enabled,
            kind,
            fields,
            vmax,
            var,
            title_prefix,
            output_name,
            centered,
        ) in collections:
            if not enabled or not fields or vmax is None:
                continue

            output_dir = (
                OUTPUT_ROOT
                / s.output_name
                / "normalized"
                / "maps"
                / output_name
                / PLOT_SPLIT
                / f"leadtime_{safe_label(leadtime)}"
                / f"lat_{safe_label(LAT_RANGE)}_lon_{safe_label(LON_RANGE)}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            for channel, field in fields.items():
                filename = (
                    f"{var}_{kind}"
                    f"_channel_{channel}"
                    f"_init_{time_label}"
                    f"_lead_{safe_label(leadtime)}"
                    f"_sample_{sample_within_init}.png"
                )
                out_file = output_dir / filename

                if out_file.exists() and not REGENERATE_PLOTS:
                    continue

                valid_values = field.values[np.isfinite(field.values)]
                if valid_values.size:
                    spatial_mean = float(np.mean(valid_values))
                    spatial_std = float(np.std(valid_values, ddof=0))
                else:
                    spatial_mean = np.nan
                    spatial_std = np.nan

                title = (
                    f"{title_prefix} · channel {channel} · "
                    f"{PLOT_SPLIT} · start {time_title} · "
                    f"lead {leadtime}{lead_unit_label} · "
                    f"mean {spatial_mean:.3f} · std {spatial_std:.3f}"
                    if PLOT_TITLE
                    else None
                )

                print(
                    f"Saving {kind} map {out_file} "
                    f"(dataset sample={sample_index})"
                )

                lat_dim = field.earthml.guessed_dims.latitude
                lon_dim = field.earthml.guessed_dims.longitude
                if lat_dim is None or lon_dim is None:
                    raise ValueError(
                        "Could not determine latitude/longitude dimensions "
                        f"for {kind} field."
                    )

                plot_field_map(
                    field,
                    var=var,
                    title=title,
                    out_file=out_file,
                    cmap=PLOT_CMAP_CENTERED if centered else PLOT_CMAP,
                    centered=centered,
                    vmin=-vmax if centered else 0.0,
                    vmax=vmax,
                    levels=PLOT_LEVELS,
                    plot_type=PLOT_TYPE,
                    figsize=PLOT_FIGSIZE,
                    spatial_dims=(lat_dim, lon_dim),
                    regions=None,
                    plot_title=PLOT_TITLE,
                    plot_labels=PLOT_LABELS,
                    title_size=TITLE_SIZE,
                    label_size=LABEL_SIZE,
                    tick_size=TICK_SIZE,
                    dpi=DPI,
                )

                saved += 1

    if corrected_ds is not None:
        corrected_ds.close()

    return saved


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

    n_saved = 0

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
                f"available for {s.output_name}. "
                f"Available leadtimes: {list(s.leadtimes)}"
            )

        for leadtime in leadtimes:
            print("=" * 80)
            print(
                f"{s.output_name}: normalized input/target/corrected maps, "
                f"split={PLOT_SPLIT}, leadtime={leadtime}"
            )

            n_saved += plot_normalized_leadtime(
                s,
                leadtime,
            )

    print(f"Done. Saved {n_saved} normalized map(s).")


if __name__ == "__main__":
    main()
