from pathlib import Path
from typing import Literal, Sequence, TypedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

import torch

import earthml
from earthml import (
    Settings,
    Leadtime,
    LeadtimeUnit,
    ClimPeriod,
    MonthlyNormalize,
    Normalize,
    XarrayDataset,
    get_experiment_configs,
    get_and_subset_datasets,
    open_zarr,
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

from locations import CITY_LOCATIONS, SEA_LOCATIONS

from train import make_train_test_datasets_for_leadtime

# ============================================================================
# Definitions
# ============================================================================

FieldName = Literal[
    # Original fields
    "input",                        # fc
    "target",                       # an
    "corrected",                    # mlfc
    # Derived fields
    "input_error",                  # fc - an
    "corrected_error",              # mlfc - an
    "ideal_correction",             # an - fc
    "correction",                   # mlfc - fc = (mlfc - an) - (fc - an)
    "input_squared_error",          # (fc - an)^2
    "corrected_squared_error",      # (mlfc - an)^2
    "squared_error_improvement",    # (fc - an)^2 - (mlfc - an)^2 (positive sign mean improvement)
]

AggFieldName = Literal[
    "mean_absolute_correction",
    "mean_absolute_positive_correction",
    "mean_absolute_negative_correction",
]

class ColorScale(TypedDict):
    vmin: float | None
    vmax: float | None
    quantile: float
    centered: bool

TimeRangeSplit = Literal[
    "train",
    "val",
    "test",
    "train_val",
    "full",
]

# ============================================================================
# Configuration
# ============================================================================

# EXP_NAME = "weather_atmo"
EXP_NAME = "weather_atmo_variables"
# EXP_NAME = "weather_atmo_normalization_ablation"
# EXP_NAME = "weather_atmo_sample_independent"
# EXP_NAME = "weather_atmo_orography"
# EXP_NAME = "weather_atmo_one_year_test"

EXPERIMENTS_ROOT = Path(
    f"/work/cmcc/jd19424/ML/MLBC/experiments/{EXP_NAME}"
)

OUTPUT_ROOT = Path(
    f"/work/cmcc/jd19424/ML/MLBC/plots/{EXP_NAME}"
)

REGENERATE_PLOTS = True
BUILD_ANALYSIS = True
# INTERPOLATE = True # seasonal
INTERPOLATE = False # weather
NORMALIZED = True

NORMALIZER: Literal["training", "input", "target"] = "input"

PLOT_FIELDS: tuple[FieldName, ...] = (
    # Errors and corrections
    "input",                        # fc
    "target",                       # an
    "corrected",                    # mlfc
    # Errors and corrections
    "input_error",                  # fc - an
    "corrected_error",              # mlfc - an
    "ideal_correction",             # an - fc
    "correction",                   # mlfc - fc = (mlfc - an) - (fc - an)
    "input_squared_error",          # (fc - an)^2
    "corrected_squared_error",      # (mlfc - an)^2
    "squared_error_improvement",    # (fc - an)^2 - (mlfc - an)^2 (positive sign mean improvement)

    # # Anomalies
    # "input_anomaly",                # fc - fc_clim
    # "target_anomaly",               # an - an_clim
    # "corrected_anomaly",            # mlfc - mlfc_clim
)

AGGREGATED_FIELDS: tuple[AggFieldName, ...] = (
    # "mean_absolute_correction",
    # "mean_absolute_positive_correction",
    # "mean_absolute_negative_correction",
)

PHYSICAL_ANOMALIES = False

CLIM_TIME_RANGE: TimeRangeSplit | tuple[str, str] = "train"
CLIM_PERIOD: ClimPeriod = ClimPeriod.MONTH  # Use your enum's actual member
CLIM_ROLLING_WINDOW: int | None = None

# ----------------------------------------------------------
# Experiment selection
# ----------------------------------------------------------

FILTERS = dict(
    region_name="ConUS",
    # normalization_mode="sample",
    # normalization_mode="none",
    # normalization_mode="channel",
)
VARIABLES: list[str] | None = [
    "mslp",
    "t2m",
    "u10",
    "v10",
    "tcc",
]

LEADTIMES: list[int] | None = [72]
# LEADTIMES: list[int] | None = None # all experiment leadtimes

# ----------------------------------------------------------
# Time selection
# ----------------------------------------------------------

INFERENCE_PERIOD = None
# INFERENCE_PERIOD = ("2025-01-01", "2025-05-12")
# INFERENCE_PERIOD = ("2025-01-01T00:00:00", "2025-01-01T00:00:00")
# INFERENCE_PERIOD = ("2025-05-01T00:00:00", "2025-05-01T00:00:00")
# INFERENCE_PERIOD = ("2025-01-01T00:00:00", "2025-09-30T12:00:00")

# Initialization times to plot. None -> all times in the selected split
WANTED_TIMES: list[str] | None = [
    "2025-08-18T12:00:00", # B1 mslp 72h
    "2025-08-17T12:00:00", # W4 mslp 72h

    # Ten-month test 01-09 2025, channel norm
    # "2025-03-08T00:00:00", # B1 t2m 72h
    # "2025-02-22T00:00:00", # B2 t2m 72h
    # "2025-03-23T00:00:00", # B3 t2m 72h
    # "2025-02-02T00:00:00", # B4 / B5 t2m 72h, near W3 / W4 in std-vs-mean scatter
    # "2025-03-07T00:00:00", # B5 t2m 72h
    # "2025-02-21T00:00:00", # B6 t2m 72h
    # "2025-04-06T00:00:00", # B7 t2m 72h
    # "2025-02-01T00:00:00", # B8 t2m 72h
    # "2025-04-07T00:00:00", # B9 t2m 72h
    # "2025-03-06T00:00:00", # B10 t2m 72h

    # "2025-02-13T00:00:00", # W1 t2m 72h
    # "2025-01-03T00:00:00", # W2 t2m 72h
    # "2025-02-06T00:00:00", # W3 / W4 t2m 72h
    # "2025-02-18T00:00:00", # W4 t2m 72h
    # "2025-02-16T00:00:00", # W5 t2m 72h
    # "2025-01-01T12:00:00", # W6 t2m 72h
    # "2025-02-12T12:00:00", # W7 t2m 72h
    # "2025-04-24T00:00:00", # W8 t2m 72h
    # "2025-03-29T00:00:00", # W9 t2m 72h
    # "2025-05-25T12:00:00", # W10 t2m 72h

    # One year test, channel norm
    # "2025-03-08T00:00:00", # B1 t2m 72h
    # "2025-02-22T00:00:00", # B2 t2m 72h
    # "2025-03-23T00:00:00", # B3 t2m 72h
    # "2025-03-07T00:00:00", # B4 t2m 72h
    # "2025-02-02T00:00:00", # B5 t2m 72h, near W4 in std-vs-mean scatter
    # "2025-02-21T00:00:00", # B6 t2m 72h
    # "2025-04-06T00:00:00", # B7 t2m 72h
    # "2025-04-07T00:00:00", # B8 t2m 72h
    # "2025-02-25T00:00:00", # B9 t2m 72h
    # "2025-02-24T00:00:00", # B10 t2m 72h

    # "2025-02-13T00:00:00", # W1 t2m 72h
    # "2025-02-14T00:00:00", # W2 t2m 72h
    # "2024-11-28T00:00:00", # W3 t2m 72h
    # "2025-02-06T00:00:00", # W4 t2m 72h
    # "2025-03-29T00:00:00", # W5 t2m 72h
    # "2024-11-29T00:00:00", # W6 t2m 72h
    # "2025-01-16T00:00:00", # W7 t2m 72h
    # "2025-04-24T00:00:00", # W8 t2m 72h
    # "2024-10-14T00:00:00", # W9 t2m 72h
    # "2025-05-05T12:00:00", # W10 t2m 72h
]
# WANTED_TIMES: list[str] | None = None

# If one initialization corresponds to multiple dataset samples, for example
# when realizations are samples:
#   None -> plot all matching samples
#   int  -> plot only this sample within each initialization (0-based)
SAMPLE_WITHIN_INIT: int | None = None

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
# PLOT_CMAP = "google:turbo"
PLOT_CMAP = "crameri:batlowW_r"
# PLOT_CMAP = "yorick:ncar_r" # interesting
# PLOT_CMAP = "chrisluts:I_Red"
# PLOT_CMAP = "cmasher:sunburst_r"

# PLOT_CMAP_CENTERED = "cmocean:balance"
PLOT_CMAP_CENTERED = "crameri:vik"

PLOT_LEVELS = 21
PLOT_TITLE = True
PLOT_LABELS = True

TITLE_SIZE = None
LABEL_SIZE = None
TICK_SIZE = None

DPI = 300

PLOT_SCALES: dict[str, dict[FieldName | AggFieldName, ColorScale]] = {
    "normalized": {
        "input": dict(vmin=None, vmax=None, quantile=0.99, centered=True),
        "target": dict(vmin=None, vmax=None, quantile=0.99, centered=True),
        "corrected": dict(vmin=None, vmax=None, quantile=0.99, centered=True),
        "input_error": dict(vmin=None, vmax=None, quantile=0.99, centered=True),
        "corrected_error": dict(vmin=None, vmax=None, quantile=0.99, centered=True),
        "ideal_correction": dict(vmin=None, vmax=None, quantile=0.99, centered=True),
        "correction": dict(vmin=None, vmax=None, quantile=0.99, centered=True),
        "input_squared_error": dict(vmin=0.0, vmax=None, quantile=0.99, centered=False),
        "corrected_squared_error": dict(vmin=0.0, vmax=None, quantile=0.99, centered=False),
        "squared_error_improvement": dict(vmin=None, vmax=None, quantile=0.99, centered=True),
        "mean_absolute_correction": dict(vmin=0.0, vmax=None, quantile=0.99, centered=False),
        "mean_absolute_positive_correction": dict(vmin=0.0, vmax=None, quantile=0.99, centered=False),
        "mean_absolute_negative_correction": dict(vmin=0.0, vmax=None, quantile=0.99, centered=False),
    },
    "physical": {
        "input": dict(vmin=None, vmax=None, quantile=0.99, centered=False),
        "target": dict(vmin=None, vmax=None, quantile=0.99, centered=False),
        "corrected": dict(vmin=None, vmax=None, quantile=0.99, centered=False),
        "input_error": dict(vmin=None, vmax=None, quantile=0.99, centered=True),
        "corrected_error": dict(vmin=None, vmax=None, quantile=0.99, centered=True),
        "ideal_correction": dict(vmin=None, vmax=None, quantile=0.99, centered=True),
        "correction": dict(vmin=None, vmax=None, quantile=0.99, centered=True),
        "input_squared_error": dict(vmin=0.0, vmax=None, quantile=0.99, centered=False),
        "corrected_squared_error": dict(vmin=0.0, vmax=None, quantile=0.99, centered=False),
        "squared_error_improvement": dict(vmin=None, vmax=None, quantile=0.99, centered=True),
        "mean_absolute_correction": dict(vmin=0.0, vmax=None, quantile=0.99, centered=False),
        "mean_absolute_positive_correction": dict(vmin=0.0, vmax=None, quantile=0.99, centered=False),
        "mean_absolute_negative_correction": dict(vmin=0.0, vmax=None, quantile=0.99, centered=False),
    },
}

PLOT_TITLE_STRFTIME = "%d.%m.%Y %H:%M"

# ============================================================================
# Classes
# ============================================================================

@dataclass
class Fields:
    input: xr.DataArray
    target: xr.DataArray
    corrected: xr.DataArray | None = None

    # -------------------------------------------------------------
    # Derived fields
    # -------------------------------------------------------------

    @property
    def input_error(self) -> xr.DataArray:
        return self.input - self.target

    @property
    def corrected_error(self) -> xr.DataArray | None:
        if self.corrected is None:
            return None
        return self.corrected - self.target

    @property
    def ideal_correction(self) -> xr.DataArray:
        return self.target - self.input

    @property
    def correction(self) -> xr.DataArray | None:
        if self.corrected is None:
            return None
        return self.corrected - self.input

    @property
    def input_squared_error(self) -> xr.DataArray:
        return self.input_error ** 2

    @property
    def corrected_squared_error(self) -> xr.DataArray | None:
        error = self.corrected_error
        return None if error is None else error**2

    @property
    def squared_error_improvement(self) -> xr.DataArray | None:
        if self.corrected_squared_error is None:
            return None
        return self.corrected_squared_error - self.input_squared_error

    # -------------------------------------------------------------
    # Aggregation
    # -------------------------------------------------------------

    @staticmethod
    def mean_absolute_correction(
        samples: Sequence["Fields"],
        sign: Literal["all", "positive", "negative"] = "all",
    ) -> xr.DataArray | None:
        corrections = [
            correction
            for fields in samples
            if (correction := fields.correction) is not None
        ]

        if not corrections:
            return None

        stacked = xr.concat(
            corrections,
            dim="sample",
            join="exact",
        )

        if sign == "positive":
            # stacked = stacked.where(stacked > 0) # generated NaN if no positve correction
            stacked = stacked.clip(min=0) # correction of opposite sign is zero
        elif sign == "negative":
            # stacked = stacked.where(stacked < 0)
            stacked = stacked.clip(max=0)
        elif sign != "all":
            raise ValueError(f"Unsupported sign={sign!r}")

        return np.abs(stacked).mean(
            "sample",
            skipna=True,
        )

    @staticmethod
    def mean_absolute_positive_correction(
        samples: Sequence["Fields"],
    ) -> xr.DataArray | None:
        return Fields.mean_absolute_correction(
            samples,
            sign="positive",
        )

    @staticmethod
    def mean_absolute_negative_correction(
        samples: Sequence["Fields"],
    ) -> xr.DataArray | None:
        return Fields.mean_absolute_correction(
            samples,
            sign="negative",
        )

    # -------------------------------------------------------------
    # Methods
    # -------------------------------------------------------------

    def get(self, name: FieldName) -> xr.DataArray | None:
        return getattr(self, name)

    def sel(self, **indexers) -> "Fields":
        return Fields(
            input=self.input.sel(**indexers),
            target=self.target.sel(**indexers),
            corrected=(
                self.corrected.sel(**indexers)
                if self.corrected is not None
                else None
            ),
        )

    def isel(self, **indexers) -> "Fields":
        return Fields(
            input=self.input.isel(indexers),
            target=self.target.isel(indexers),
            corrected=(
                self.corrected.isel(indexers)
                if self.corrected is not None
                else None
            ),
        )

    def squeeze(self) -> "Fields":
        return Fields(
            input=self.input.squeeze(drop=True),
            target=self.target.squeeze(drop=True),
            corrected=(
                self.corrected.squeeze(drop=True)
                if self.corrected is not None
                else None
            ),
        )

# ============================================================================
# Main
# ============================================================================

def main() -> None:
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
                f"{s.output_name}: {"normalized" if NORMALIZED else "physical"} "
                "input/target/corrected maps, "
                f"leadtime={leadtime}"
            )

            selected_samples = get_selected_samples(
                s=s,
                normalized=NORMALIZED,
                leadtime=leadtime,
            )

            if not selected_samples:
                continue

            n_saved += plot_fields(
                s=s,
                normalized=NORMALIZED,
                leadtime=leadtime,
                selected_samples=selected_samples,
            )

            n_saved += plot_aggregated(
                s=s,
                normalized=NORMALIZED,
                leadtime=leadtime,
                selected_samples=selected_samples,
            )

    print(f"Done. Saved {n_saved} map(s).")

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


def get_split_for_time(
    s: Settings,
    time: pd.Timestamp,
) -> Literal["train", "val", "test"]:
    splits = {
        "train": (s.train_start, s.train_end),
        "val": (s.val_start, s.val_end),
        "test": (s.test_start, s.test_end),
    }

    matching = [
        split
        for split, (start, end) in splits.items()
        if pd.Timestamp(start) <= time < pd.Timestamp(end)
    ]

    if len(matching) != 1:
        raise ValueError(
            f"Expected exactly one split for {time}, "
            f"found {matching}"
        )

    return matching[0]

# ======================================================
# Load dataset helpers
# ======================================================

def _select_range(
    s: Settings,
    time_range: tuple[str, str] | TimeRangeSplit,
) -> tuple[str, str]:
    if time_range == "train":
        return (s.train_start, s.train_end)
    elif time_range == "val":
        return (s.val_start, s.val_end)
    elif time_range == "test":
        return (s.test_start, s.test_end)
    elif time_range == "train_val":
        return (s.train_start, s.val_end)
    elif time_range == "full":
        return (s.train_start, s.test_end)
    elif isinstance(time_range, tuple):
        return time_range
    else:
        raise ValueError(f"Invalid time_range={time_range}")


def _get_valid_spatial_range(
    s: Settings,
) -> tuple[str | None, str | None]:
    whole_region_lat_lon = (
        list(s.region.values())
        if s.region is not None
        else [None, None]
    )

    valid_lat_range = whole_region_lat_lon[0]
    valid_lon_range = whole_region_lat_lon[1]

    return valid_lat_range, valid_lon_range


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


def _leadtime_selection(
    s: Settings,
    ds: xr.Dataset,
    wanted_leadtimes: Sequence | None = None,
) -> Sequence[Leadtime]:
    leadtime_dim = ds.earthml.guessed_dims.leadtime
    time_dim = ds.earthml.guessed_dims.time

    if leadtime_dim is None:
        raise ValueError("Could not determine forecast lead-time dimension")

    if time_dim is None:
        raise ValueError("Could not determine forecast time dimension")

    selected_leadtimes = (
        s.leadtimes
        if wanted_leadtimes is None
        else wanted_leadtimes
    )

    missing_leadtimes = (
        set(selected_leadtimes)
        - set(s.leadtimes)
    )

    if missing_leadtimes:
        raise ValueError(
            f"Requested leadtimes {sorted(missing_leadtimes)} "
            f"are not available. "
            f"Available leadtimes: {list(s.leadtimes)}"
        )

    return selected_leadtimes


def _select_plot_channels(
    field: xr.DataArray,
    field_name: FieldName | AggFieldName,
) -> dict[int, xr.DataArray]:
    wanted_channels = (
        PLOT_INPUT_CHANNELS
        if field_name == "input"
        else PLOT_TARGET_CHANNELS
    )

    if "channel" not in field.dims:
        if wanted_channels is not None and 0 not in wanted_channels:
            return {}

        return {0: field}

    available_channels = list(
        range(field.sizes["channel"])
    )

    selected_channels = (
        available_channels
        if wanted_channels is None
        else wanted_channels
    )

    missing = set(selected_channels) - set(available_channels)

    if missing:
        raise ValueError(
            f"Invalid channels {sorted(missing)} for {field_name}. "
            f"Available channels: {available_channels}"
        )

    return {
        channel: field.isel(channel=channel, drop=True)
        for channel in selected_channels
    }

# ======================================================
# Physical fields
# ======================================================

def get_physical_fields(
    s: Settings,
    time_range: tuple[str, str] | TimeRangeSplit,
    wanted_leadtimes: Sequence | None = None,
    wanted_times: Sequence[str] | None = None,
    inference_period: tuple[str, str] | None = None,
    interpolate: bool = True,
) -> Fields:
    valid_time_range = _select_range(s, time_range)

    # Whole available region spatial coordinates
    valid_lat_range, valid_lon_range = _get_valid_spatial_range(s)

    # Select correct MLFC path
    mlfc_path = None

    if inference_period is not None:
        requested_start, requested_end = map(
            pd.Timestamp,
            valid_time_range,
        )

        inference_start, inference_end = map(
            pd.Timestamp,
            inference_period,
        )

        if not (
            inference_start <= requested_start
            and requested_end <= inference_end
        ):
            raise ValueError(
                "Requested time range is not covered by the "
                "inference period:\n"
                f"  Requested: {valid_time_range}\n"
                f"  Inference: {inference_period}"
            )

        inference_start_str, inference_end_str = inference_period

        mlfc_path = (
            s.exp_dir
            / "inference"
            / f"{inference_start_str}_{inference_end_str}"
            / "test_corrected.zarr"
        )

        if not mlfc_path.exists():
            raise FileNotFoundError(
                f"Inference results not found: {mlfc_path}"
            )

    # ------------------------------------------------------
    # Obtain fields
    # ------------------------------------------------------

    fc, an, mlfc = get_and_subset_datasets(
        s,
        leadtime_units=LeadtimeUnit(s.leadtime_unit),
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

    fields = Fields(
        input=fc[s.var_fc],
        target=an[s.var_an],
        corrected=mlfc[s.var_fc] if mlfc is not None else None,
    )

    # ------------------------------------------------------
    # Lead-time selection
    # ------------------------------------------------------

    leadtime_dim = fields.input.earthml.guessed_dims.leadtime

    selected_leadtimes = _leadtime_selection(
        s=s,
        ds=fc,
        wanted_leadtimes=wanted_leadtimes,
    )

    fields = fields.sel(
        **{leadtime_dim: selected_leadtimes}
    )

    # ------------------------------------------------------
    # Initialization-time selection
    # ------------------------------------------------------

    time_dim = fields.input.earthml.guessed_dims.time

    if wanted_times is not None:
        requested_times = pd.to_datetime(wanted_times)

        available_times = pd.DatetimeIndex(
            fields.input[time_dim].values
        )

        missing_times = requested_times[
            ~requested_times.isin(available_times)
        ]

        if len(missing_times) > 0:
            print(
                "WARNING: requested initialization times not available: "
                f"{list(missing_times)}"
            )

        selected_times = requested_times[
            requested_times.isin(available_times)
        ]

        fields = fields.sel(
            **{time_dim: selected_times}
        )

    return fields


def get_physical_anomaly_fields(
    s: Settings,
    fields: Fields,
    clim_time_range: tuple[str, str] | TimeRangeSplit,
    clim_period: ClimPeriod,
    clim_rolling_window: int | None = None,
    interpolate: bool = True,
    recalculate_climatology: bool = True,
    build_analysis: bool = True,
) -> Fields:
    valid_clim_time_range = _select_range(s, clim_time_range)
    valid_lat_range, valid_lon_range = _get_valid_spatial_range(s)

    fc_da, an_da, mlfc_da = fields.input, fields.target, fields.corrected

    # ------------------------------------------------------
    # Climatologies
    # ------------------------------------------------------

    fc_clim, an_clim, mlfc_clim = (
        calculate_save_and_subset_climatologies(
            s,
            leadtime_units=s.leadtime_unit,
            force=recalculate_climatology,
            clim_period=clim_period,
            rolling_window=clim_rolling_window,
            rolling_center=True,
            rolling_min_periods=1,
            lat_range=valid_lat_range,
            lon_range=valid_lon_range,
            time_range=valid_clim_time_range,
            time_start=None,
            interpolate=interpolate,
            engine="zarr",
            build_analysis=build_analysis,
            coord_rename_fc=None,
            coord_rename_an=None,
        )
    )

    clim_fields = Fields(
        input=fc_clim[s.var_fc],
        target=an_clim[s.var_an],
        corrected=(
            mlfc_clim[s.var_fc]
            if mlfc_clim is not None
            else None
        ),
    )

    leadtime_dim = fields.input.earthml.guessed_dims.leadtime
    time_dim = fields.input.earthml.guessed_dims.time

    selected_leadtimes = fields.input[leadtime_dim].values

    clim_fields = clim_fields.sel(
        **{leadtime_dim: selected_leadtimes}
    )

    # ------------------------------------------------------
    # Stack climatologies
    # ------------------------------------------------------

    fc_clim_da = stack_hour_clim(
        clim_fields.input,
        clim_period,
    )

    an_clim_da = stack_hour_clim(
        clim_fields.target,
        clim_period,
    )

    mlfc_clim_da = (
        stack_hour_clim(
            clim_fields.corrected,
            clim_period,
        )
        if clim_fields.corrected is not None
        else None
    )

    # ------------------------------------------------------
    # Calculate anomalies
    # ------------------------------------------------------

    fc_anom_da = (
        groupby_period(
            fc_da,
            time_dim,
            clim_period,
        )
        - fc_clim_da
    )

    an_anom_da = (
        groupby_period(
            an_da,
            time_dim,
            clim_period,
        )
        - an_clim_da
    )

    mlfc_anom_da = (
        groupby_period(
            mlfc_da,
            time_dim,
            clim_period,
        )
        - mlfc_clim_da
        if mlfc_da is not None and mlfc_clim_da is not None
        else None
    )

    return Fields(
        input=fc_anom_da,
        target=an_anom_da,
        corrected=mlfc_anom_da,
    )

# ============================================================================
# Normalized fields
# ============================================================================

def get_normalized_fields(
    s: Settings,
    dataset: XarrayDataset,
    leadtime: int,
    init_time: pd.Timestamp,
    corrected_da: xr.DataArray | None,
    sample_index: int,
    normalize_input,
    normalize_target,
    normalizer: Literal["training", "input", "target"],
) -> Fields:
    """Return populated normalized Fields instance from XarrayDataset."""

    # ----------------------------------------------------------
    # Input and target
    # ----------------------------------------------------------

    x, y, mask, month = dataset[sample_index]

    input_tensor = x.detach().float().cpu()
    target_tensor = y.detach().float().cpu()
    mask_tensor = mask.detach().bool().cpu()

    # ----------------------------------------------------------
    # Select normalization reference
    # ----------------------------------------------------------

    raw_input = dataset.x[sample_index].detach().float().cpu()
    raw_target = dataset.y[sample_index].detach().float().cpu()

    n_channels = raw_target.shape[0]
    raw_fc = raw_input[:n_channels]

    sample_mean = None
    sample_std = None

    if s.normalization_mode == "sample":
        if normalizer in {"training", "input"}:
            reference = raw_input
            reference_normalizer = normalize_input
        else:
            reference = raw_target
            reference_normalizer = normalize_target

        sample_mean, sample_std = (
            reference_normalizer.sample_params(
                reference.unsqueeze(0)
            )
        )

        sample_mean = sample_mean[0]
        sample_std = sample_std[0]


    def _normalize_field(
        tensor: torch.Tensor,
        *,
        field: Literal["input", "target", "corrected"],
    ) -> torch.Tensor:
        if s.normalization_mode == "sample":
            assert sample_mean is not None
            assert sample_std is not None

            return normalize_target(
                tensor,
                mean=sample_mean,
                std=sample_std,
            ).detach().float().cpu()

        selected_normalizer = (
            normalize_input
            if normalizer == "input"
            or (normalizer == "training" and field == "input")
            else normalize_target
        )

        return selected_normalizer(
            tensor,
            months=int(month),
        ).detach().float().cpu()


    if normalizer != "training":
        input_tensor = _normalize_field(
            raw_fc,
            field="input",
        )

        target_tensor = _normalize_field(
            raw_target,
            field="target",
        )

    # Recover spatial coordinates from input dataset
    ref_ds = dataset.input_ds
    lat_dim = ref_ds.earthml.guessed_dims.latitude
    lon_dim = ref_ds.earthml.guessed_dims.longitude

    if lat_dim is None or lon_dim is None:
        raise ValueError(
            f"Could not determine latitude/longitude dimensions"
        )

    latitude = np.asarray(ref_ds[lat_dim].values)
    longitude = np.asarray(ref_ds[lon_dim].values)

    # Validate tensors
    for name, t in (
        ("input", input_tensor),
        ("target", target_tensor),
        ("mask", mask_tensor),
    ):
        if t.ndim != 3:
            raise ValueError(
                f"Expected {name} tensor with shape (C,H,W), got {tuple(t.shape)}"
            )

        if t.shape[-2:] != (len(latitude), len(longitude)):
            raise ValueError(
                f"{name} tensor's spatial-coordinate shape mismatch: "
                f"tensor spatial shape={tuple(t.shape[-2:])}, "
                f"coordinates={(len(latitude), len(longitude))}"
            )

    # ----------------------------------------------------------
    # Corrected forecast
    # ----------------------------------------------------------

    if corrected_da is not None:
        time_dim = corrected_da.earthml.guessed_dims.time
        lead_dim = corrected_da.earthml.guessed_dims.leadtime

        if time_dim is None:
            raise ValueError(
                "Could not determine corrected forecast time dimension."
            )

        corrected_da = corrected_da.sel({
            time_dim: np.datetime64(init_time),
        })

        if lead_dim is not None and lead_dim in corrected_da.dims:
            corrected_da = corrected_da.sel({
                lead_dim: leadtime,
            })

        corrected_da = corrected_da.sel({
            lat_dim: ref_ds[lat_dim],
            lon_dim: ref_ds[lon_dim],
        })

        realization_dim = corrected_da.earthml.guessed_dims.realization

        if realization_dim is not None and realization_dim in corrected_da.dims:
            if target_tensor.shape[0] == 1:
                corrected_da = corrected_da.mean(
                    realization_dim,
                    skipna=True,
                )
            else:
                corrected_da = corrected_da.transpose(
                    realization_dim,
                    lat_dim,
                    lon_dim,
                )

        corrected_da = corrected_da.squeeze(drop=True)

        corrected_values = np.asarray(
            corrected_da.values,
        ).copy()

        if corrected_values.ndim == 2:
            corrected_values = corrected_values[None, ...]

        if corrected_values.ndim != 3:
            raise ValueError(
                f"Unexpected corrected shape={corrected_values.shape}"
            )

        corrected_tensor = torch.as_tensor(
            corrected_values,
            dtype=target_tensor.dtype,
        )

        normalized_corrected_tensor = _normalize_field(
            corrected_tensor,
            field="corrected",
        )
    else:
        normalized_corrected_tensor = None

    # ----------------------------------------------------------
    # Populate Fields
    # ----------------------------------------------------------

    fields: dict[str, xr.DataArray | None] = {}

    for name, tensor in (
        ("input", input_tensor),
        ("target", target_tensor),
        ("corrected", normalized_corrected_tensor),
    ):
        if tensor is None:
            fields[name] = None
            continue

        # Keep only physical channels for the input.
        if name == "input":
            tensor = tensor[:target_tensor.shape[0]]

        if tensor.shape[0] != mask_tensor.shape[0]:
            raise ValueError(
                f"{name} has {tensor.shape[0]} channels, "
                f"but mask has {mask_tensor.shape[0]}"
            )

        values = tensor.numpy().astype(float, copy=True)
        values[~mask_tensor.numpy()] = np.nan

        fields[name] = xr.DataArray(
            values,
            dims=("channel", lat_dim, lon_dim),
            coords={
                "channel": np.arange(tensor.shape[0]),
                lat_dim: latitude,
                lon_dim: longitude,
            },
            name=name,
            attrs={
                "units": "normalized",
                "long_name": f"normalized {name}",
            },
        )

    return Fields(
        input=fields["input"],
        target=fields["target"],
        corrected=fields["corrected"],
    )

# ============================================================================
# Subset helpers
# ============================================================================

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
# Plotting helpers
# ============================================================================

def _get_plot_limits(
    arrays: list[xr.DataArray],
    *,
    scale: ColorScale,
) -> tuple[float, float]:
    vmin = scale["vmin"]
    vmax = scale["vmax"]

    if vmin is not None and vmax is not None:
        if vmin >= vmax:
            raise ValueError(f"Invalid color limits: {vmin=}, {vmax=}")
        return vmin, vmax

    values = np.concatenate([
        np.asarray(a.values, dtype=float).ravel()
        for a in arrays
    ])

    values = values[np.isfinite(values)]

    if values.size == 0:
        return (
            0.0 if vmin is None else vmin,
            1.0 if vmax is None else vmax,
        )

    q = scale["quantile"]

    if not 0.5 < q <= 1.0:
        raise ValueError(f"Invalid quantile: {q}")

    if scale["centered"]:
        auto_vmax = float(np.quantile(np.abs(values), q))

        if vmax is None:
            vmax = auto_vmax

        if vmin is None:
            vmin = -vmax

    else:
        if vmin is None:
            vmin = float(np.quantile(values, 1 - q))

        if vmax is None:
            vmax = float(np.quantile(values, q))

    if vmin == vmax:
        delta = max(abs(vmin) * 0.01, 1e-6)

        if scale["centered"]:
            vmin, vmax = -delta, delta
        else:
            vmax = vmin + delta

    elif vmin > vmax:
        raise ValueError(
            f"Invalid color limits: {vmin=}, {vmax=}"
        )

    return vmin, vmax

# ============================================================================
# Get selected fields
# ============================================================================

def get_selected_samples(
    s: Settings,
    normalized: bool,
    leadtime: int,
) -> list[tuple[int, pd.Timestamp, int, Fields]]:
    if not WANTED_TIMES:
        raise ValueError(
            "WANTED_TIMES must contain at least one initialization time."
        )

    requested_times = pd.DatetimeIndex(
        pd.to_datetime(WANTED_TIMES)
    ).unique().sort_values()

    # ----------------------------------------------------------
    # Determine the split of each requested time
    # ----------------------------------------------------------

    times_by_split: dict[str, list[pd.Timestamp]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for time in requested_times:
        split = get_split_for_time(s, time)
        times_by_split[split].append(time)

    # ----------------------------------------------------------
    # Load physical fields for the requested period
    # ----------------------------------------------------------

    physical_fields = get_physical_fields(
        s=s,
        time_range=(
            str(requested_times.min()),
            str(requested_times.max()),
        ),
        wanted_leadtimes=[leadtime],
        wanted_times=[str(t) for t in requested_times],
        inference_period=INFERENCE_PERIOD,
        interpolate=INTERPOLATE,
    )

    if not normalized and PHYSICAL_ANOMALIES:
        physical_fields = get_physical_anomaly_fields(
            s=s,
            fields=physical_fields,
            clim_time_range=CLIM_TIME_RANGE,
            clim_period=CLIM_PERIOD,
            clim_rolling_window=CLIM_ROLLING_WINDOW,
            interpolate=INTERPOLATE,
            recalculate_climatology=False,
            build_analysis=BUILD_ANALYSIS,
        )

    selected_samples = []

    # ----------------------------------------------------------
    # Normalized samples
    # ----------------------------------------------------------

    if normalized:
        datasets, normalize_input, normalize_target = (
            make_normalized_datasets_for_leadtime(s, leadtime)
        )

        for split, wanted_times in times_by_split.items():
            if not wanted_times:
                continue

            dataset = datasets[split]

            if dataset is None:
                raise ValueError(
                    f"Dataset for split {split!r} is unavailable."
                )

            sample_times = pd.to_datetime(
                get_sample_times(dataset)
            )

            for init_time in wanted_times:
                matching = np.flatnonzero(
                    sample_times == init_time
                )

                if len(matching) == 0:
                    print(
                        f"WARNING: {init_time} not found "
                        f"in {split} dataset."
                    )
                    continue

                indices = (
                    matching
                    if SAMPLE_WITHIN_INIT is None
                    else matching[
                        SAMPLE_WITHIN_INIT:SAMPLE_WITHIN_INIT + 1
                    ]
                )

                for sample_within_init, sample_index in enumerate(indices):
                    fields = get_normalized_fields(
                        s=s,
                        dataset=dataset,
                        leadtime=leadtime,
                        corrected_da=physical_fields.corrected,
                        sample_index=int(sample_index),
                        init_time=init_time,
                        normalize_input=normalize_input,
                        normalize_target=normalize_target,
                        normalizer=NORMALIZER,
                    )

                    selected_samples.append((
                        int(sample_index),
                        init_time,
                        (
                            sample_within_init
                            if SAMPLE_WITHIN_INIT is None
                            else SAMPLE_WITHIN_INIT
                        ),
                        fields,
                    ))

    # ----------------------------------------------------------
    # Physical samples
    # ----------------------------------------------------------

    else:
        time_dim = physical_fields.input.earthml.guessed_dims.time
        leadtime_dim = physical_fields.input.earthml.guessed_dims.leadtime

        if time_dim is None or leadtime_dim is None:
            raise ValueError(
                "Could not determine physical time/leadtime dimensions."
            )

        available_times = pd.DatetimeIndex(
            physical_fields.input[time_dim].values
        )

        for init_time in requested_times:
            matching = np.flatnonzero(
                available_times == init_time
            )

            if len(matching) == 0:
                print(
                    f"WARNING: physical fields unavailable for {init_time}"
                )
                continue

            sample_index = int(matching[0])

            fields = physical_fields.isel(
                **{
                    time_dim: sample_index,
                    leadtime_dim: 0,
                }
            )

            # Convert to Celsius if necessary
            if not PHYSICAL_ANOMALIES:
                fields = Fields(
                    input=_convert_kelvin_to_celsius(
                        fields.input,
                        var=s.var_fc,
                    ),
                    target=_convert_kelvin_to_celsius(
                        fields.target,
                        var=s.var_an,
                    ),
                    corrected=(
                        _convert_kelvin_to_celsius(
                            fields.corrected,
                            var=s.var_fc,
                        )
                        if fields.corrected is not None
                        else None
                    ),
                )

            selected_samples.append((
                sample_index,
                init_time,
                0,
                fields,
            ))

    return selected_samples

# ==========================================================
# Plot selected fields for one leadtime
# ==========================================================

def plot_fields(
    s: Settings,
    normalized: bool,
    leadtime: int,
    selected_samples: list[tuple[int, pd.Timestamp, int, Fields]],
) -> int:
    field_limits: dict[
        tuple[FieldName, int],
        tuple[float, float],
    ] = {}

    space = (
        f"normalized_{NORMALIZER}"
        if normalized
        else "physical_anomaly" if PHYSICAL_ANOMALIES
        else "physical"
    )

    for field_name in PLOT_FIELDS:
        scale = PLOT_SCALES[
            "normalized" if normalized else "physical"
        ][field_name]

        channel_arrays: dict[int, list[xr.DataArray]] = {}

        # Collect fields across all samples, grouped by channel
        for _, _, _, fields in selected_samples:
            field = fields.get(field_name)

            if field is None:
                continue

            for channel, channel_field in _select_plot_channels(
                field,
                field_name,
            ).items():
                channel_arrays.setdefault(channel, []).append(
                    subset_spatially(channel_field)
                )

        # Calculate common limits across samples for each channel
        for channel, arrays in channel_arrays.items():
            field_limits[field_name, channel] = _get_plot_limits(
                arrays,
                scale=scale,
            )

    saved = 0

    output_dir = (
        OUTPUT_ROOT
        / s.output_name
        / "maps"
        / space
        / f"leadtime_{leadtime}"
    )

    for (
        sample_index,
        init_time,
        sample_within_init,
        fields,
    ) in selected_samples:

        time_label = init_time.strftime("%Y%m%dT%H%M")
        time_title = init_time.strftime(PLOT_TITLE_STRFTIME)

        for field_name in PLOT_FIELDS:
            field = fields.get(field_name)

            if field is None:
                continue

            for channel, channel_field in _select_plot_channels(
                field,
                field_name,
            ).items():
                channel_field = subset_spatially(channel_field)

                # --------------------------------------------------
                # Output path
                # --------------------------------------------------

                output_field_dir = (
                    output_dir
                    / field_name
                )

                output_field_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                filename = (
                    f"init_{time_label}"
                    f"_channel_{channel}"
                    f"_sample_{sample_within_init}"
                    f"_leadtime_{leadtime}.png"
                )

                out_file = output_field_dir / filename

                if out_file.exists() and not REGENERATE_PLOTS:
                    continue

                # --------------------------------------------------
                # Plot metadata
                # --------------------------------------------------

                title = (
                    f"{field_name.replace('_', ' ').title()}\n"
                    f"Channel: {channel} · "
                    f"Init: {time_title} · "
                    f"Leadtime: {leadtime} {s.leadtime_unit}"
                )

                lat_dim = channel_field.earthml.guessed_dims.latitude
                lon_dim = channel_field.earthml.guessed_dims.longitude

                if lat_dim is None or lon_dim is None:
                    raise ValueError(
                        f"Could not determine spatial dimensions "
                        f"for {field_name}, channel={channel}"
                    )

                # --------------------------------------------------
                # Plot
                # --------------------------------------------------

                scale = PLOT_SCALES[
                    "normalized" if normalized else "physical"
                ][field_name]

                vmin, vmax = field_limits[field_name, channel]

                plot_field_map(
                    channel_field,
                    var=s.var_an,
                    title=title,
                    out_file=out_file,
                    cmap=(
                        PLOT_CMAP_CENTERED
                        if scale["centered"]
                        else PLOT_CMAP
                    ),
                    centered=scale["centered"],
                    vmin=vmin,
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

    return saved


def plot_aggregated(
    s: Settings,
    normalized: bool,
    leadtime: int,
    selected_samples: list[tuple[int, pd.Timestamp, int, Fields]],
) -> int:
    if not selected_samples:
        return 0

    # ----------------------------------------------------------
    # Aggregate fields
    # ----------------------------------------------------------

    samples = [
        fields
        for _, _, _, fields in selected_samples
    ]

    space = (
        f"normalized_{NORMALIZER}"
        if normalized
        else "physical_anomaly" if PHYSICAL_ANOMALIES
        else "physical"
    )

    init_times = [
        pd.Timestamp(init_time)
        for _, init_time, _, _ in selected_samples
    ]

    first_init = min(init_times).strftime("%Y%m%dT%H%M")
    last_init = max(init_times).strftime("%Y%m%dT%H%M")

    output_dir = (
        OUTPUT_ROOT
        / s.output_name
        / "maps"
        / space
        / f"leadtime_{leadtime}"
        / "aggregated"
        / f"{first_init}_{last_init}"
    )

    saved = 0

    for field_name in AGGREGATED_FIELDS:

        # ------------------------------------------------------
        # Calculate aggregated field
        # ------------------------------------------------------

        field = getattr(Fields, field_name)(samples)

        if field is None:
            continue

        scale = PLOT_SCALES[
            "normalized" if normalized else "physical"
        ][field_name]

        for channel, channel_field in _select_plot_channels(
            field,
            field_name,
        ).items():
            channel_field = subset_spatially(channel_field)

            # --------------------------------------------------
            # Output path
            # --------------------------------------------------
            output_field_dir = (
                output_dir
                / field_name
            )

            output_field_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            out_file = (
                output_field_dir
                / (
                    f"channel_{channel}"
                    f"_leadtime_{leadtime}.png"
                )
            )

            if out_file.exists() and not REGENERATE_PLOTS:
                continue

            # --------------------------------------------------
            # Plot metadata
            # --------------------------------------------------

            title = (
                f"{field_name.replace('_', ' ').title()}\n"
                f"Channel: {channel} · "
                f"Leadtime: {leadtime} {s.leadtime_unit} · "
                f"Samples: {len(samples)}"
            )

            lat_dim = channel_field.earthml.guessed_dims.latitude
            lon_dim = channel_field.earthml.guessed_dims.longitude

            if lat_dim is None or lon_dim is None:
                raise ValueError(
                    f"Could not determine spatial dimensions "
                    f"for {field_name}, channel={channel}"
                )

            # --------------------------------------------------
            # Color scaling
            # --------------------------------------------------

            vmin, vmax = _get_plot_limits(
                [channel_field],
                scale=scale,
            )

            # --------------------------------------------------
            # Plot
            # --------------------------------------------------

            plot_field_map(
                channel_field,
                var=s.var_an,
                title=title,
                out_file=out_file,
                cmap=(
                    PLOT_CMAP_CENTERED
                    if scale["centered"]
                    else PLOT_CMAP
                ),
                centered=scale["centered"],
                vmin=vmin,
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

    return saved


if __name__ == "__main__":
    main()
