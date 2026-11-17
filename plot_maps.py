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
    SplitDataModule,
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
    "input",                        # actual model input (normalized: ALL x channels)
    "baseline",                     # baseline expressed in target space
    "target",                       # actual model target (normalized: y)
    "corrected",                    # model-corrected field expressed in target space
    # Derived fields
    "input_error",                  # fc - an
    "corrected_error",              # mlfc - an
    "ideal_correction",             # an - fc
    "correction",                   # mlfc - fc = (mlfc - an) - (fc - an)
    "input_squared_error",          # (fc - an)^2
    "corrected_squared_error",      # (mlfc - an)^2
    "squared_error_improvement",    # (mlfc - an)^2 - (fc - an)^2  (negative sign mean improvement)
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

# Seasonal
EXP_NAME = "experiments"
# EXP_NAME = "experiment_new"

# Weather
# EXP_NAME = "weather_atmo"
# EXP_NAME = "weather_atmo_architecture"
# EXP_NAME = "weather_atmo_encoding"
# EXP_NAME = "weather_atmo_normalization"
# EXP_NAME = "weather_atmo_variables"
# EXP_NAME = "weather_atmo_normalization_ablation"
# EXP_NAME = "weather_atmo_sample_independent"
# EXP_NAME = "weather_atmo_orography"
# EXP_NAME = "weather_atmo_one_year_test"

EXPERIMENTS_ROOT = Path(
    f"/Users/jacopodallaglio/ML/training/seasonal/{EXP_NAME}"
    # f"/work/cmcc/jd19424/ML/MLBC/experiments/{EXP_NAME}"
)

OUTPUT_ROOT = Path(
    f"/Users/jacopodallaglio/ML/training/seasonal/{EXP_NAME}"
    # f"/work/cmcc/jd19424/ML/MLBC/plots/{EXP_NAME}"
)

REGENERATE_PLOTS = True
BUILD_ANALYSIS = True
# INTERPOLATE = True # seasonal
INTERPOLATE = False # weather
NORMALIZED = True

PLOT_FIELDS: tuple[FieldName, ...] = (
    # Errors and corrections
    "input",                        # network input in normalized mode
    "baseline",                     # baseline in target space
    "target",                       # training target in normalized mode
    "corrected",                    # reconstructed model prediction in target space
    # Errors and corrections
    "input_error",                  # fc - an
    "corrected_error",              # mlfc - an
    "ideal_correction",             # an - fc
    "correction",                   # mlfc - fc = (mlfc - an) - (fc - an)
    "input_squared_error",          # (fc - an)^2
    "corrected_squared_error",      # (mlfc - an)^2
    "squared_error_improvement",    # (mlfc - an)^2 - (fc - an)^2  (negative sign mean improvement)

    # # Anomalies
    # "input_anomaly",                # fc - fc_clim
    # "target_anomaly",               # an - an_clim
    # "corrected_anomaly",            # mlfc - mlfc_clim
)

AGGREGATED_FIELDS: tuple[AggFieldName, ...] = (
    "mean_absolute_correction",
    "mean_absolute_positive_correction",
    "mean_absolute_negative_correction",
)

PHYSICAL_ANOMALIES = False

# CLIM_TIME_RANGE: TimeRangeSplit | tuple[str, str] = "train"
CLIM_TIME_RANGE: TimeRangeSplit | tuple[str, str] = "train_val"
CLIM_PERIOD: ClimPeriod = ClimPeriod.MONTH
# CLIM_PERIOD: ClimPeriod = ClimPeriod.HOUR
CLIM_ROLLING_WINDOW: int | None = None

# ----------------------------------------------------------
# Experiment selection
# ----------------------------------------------------------

VARIABLES = [
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

REGIONS = [
    # "ConUS",
    # "Europe",
    # "Pacific",
    "World",
    # None,
]

FILTERS = dict(
    net_name="SmaAt_UNet",
    target_mode="residual",
    # loss_name="GeoMaskedMSELoss",
    # train_subsamples=264,
    # normalization_input_mean="channel",
    # normalization_input_std="channel",
    # normalization_target_mean=0,
    # normalization_target_std="channel",
    # extra_suffix_folder="depth5_baseC64RR16_KS5_smaaunet",
    # extra_suffix_folder="baseC64RR8_smaaunet",
)

# LEADTIMES: list[int] | None = [72]
LEADTIMES: list[int] | None = None # all experiment leadtimes

# ----------------------------------------------------------
# Time selection
# ----------------------------------------------------------

TIME_SPLIT: TimeRangeSplit = "test"
# "train", "val", "test", "train_val", "full"

INFERENCE_PERIOD: TimeRangeSplit | tuple[str, str] | None = None
# None -> use the experiment default MLFC; "test" -> use inference/<saved test dates>/test_corrected.zarr
# INFERENCE_PERIOD = ("2025-01-01", "2025-05-12")
# INFERENCE_PERIOD = ("2025-01-01T00:00:00", "2025-01-01T00:00:00")

# Initialization times to plot. None -> all times in the selected split
WANTED_TIMES: list[str] | None = [
    # "2025-03-08T00:00:00",
    # "2025-01-17T00:00:00",
    "2025-02-13T00:00:00", # W3
    "2025-02-02T00:00:00", # B6
]
# WANTED_TIMES: list[str] | None = None

# If one initialization corresponds to multiple dataset samples, for example
# when realizations are samples:
#   None -> plot all matching samples
#   int  -> plot only this sample within each initialization (0-based)
SAMPLE_WITHIN_INIT: int | None = None

# Channels to plot. None -> all available channels.
# In normalized mode input includes ALL channels actually passed to the NN,
# including additional forecast variables and encoding channels.
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
        "baseline": dict(vmin=None, vmax=None, quantile=0.99, centered=True),
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
        "baseline": dict(vmin=None, vmax=None, quantile=0.99, centered=False),
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
    baseline: xr.DataArray | None = None  # None means baseline == input (physical fields)

    # -------------------------------------------------------------
    # Derived fields
    # -------------------------------------------------------------

    @property
    def reference_baseline(self) -> xr.DataArray:
        return self.baseline if self.baseline is not None else self.input

    @property
    def input_error(self) -> xr.DataArray:
        return self.reference_baseline - self.target

    @property
    def corrected_error(self) -> xr.DataArray | None:
        if self.corrected is None:
            return None
        return self.corrected - self.target

    @property
    def ideal_correction(self) -> xr.DataArray:
        return self.target - self.reference_baseline

    @property
    def correction(self) -> xr.DataArray | None:
        if self.corrected is None:
            return None
        return self.corrected - self.reference_baseline

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
            baseline=(
                self.baseline.sel(**indexers)
                if self.baseline is not None else None
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
            baseline=(
                self.baseline.isel(indexers)
                if self.baseline is not None else None
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
            baseline=(
                self.baseline.squeeze(drop=True)
                if self.baseline is not None else None
            ),
        )

# ============================================================================
# Main
# ============================================================================

def main() -> None:
    settings = get_experiment_configs(
        EXPERIMENTS_ROOT,
        var_fc=VARIABLES,
        region_name=REGIONS,
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

def make_normalizers(
    s: Settings,
    train_dataset: XarrayDataset,
    val_dataset: XarrayDataset | None,
) -> tuple[Normalize, Normalize]:
    """Fit exactly as inference.py, on the selected training subset."""
    if s.normalization != "full":
        raise NotImplementedError(
            "Monthly normalization is not supported by the current inference.py"
        )

    member_seed = s.seed if isinstance(s.seed, int) else s.seed[0]
    datamodule = SplitDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_fraction=s.train_fraction,
        batch_size=s.batch_size,
        seed=member_seed,
        num_workers=0,
        split_strategy=s.split_strategy,
        shuffle_train=s.shuffle_train_batch,
        pin_memory=None,
        persistent_workers=None,
        drop_last_train=False,
        group_batches_by_month=False,
        train_subsamples=s.train_subsamples,
        val_subsamples=s.val_subsamples,
    )
    datamodule.setup("fit")

    normalize_input = Normalize(
        mean=s.normalization_input_mean,
        std=s.normalization_input_std,
    ).fit(datamodule.train_dataset, dim="x")

    normalize_target = Normalize(
        mean=s.normalization_target_mean,
        std=s.normalization_target_std,
    ).fit(datamodule.train_dataset, dim="y")

    return normalize_input, normalize_target


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
        "sample_target_mean": s.sample_target_mean,
        "sample_target_std": s.sample_target_std,
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
    s: Settings,
    leadtime: int,
) -> tuple[dict[str, XarrayDataset | None], Normalize, Normalize]:
    """Reproduce the input channels, datasets, and training normalizers used by inference.py."""
    if s.regional_training or s.separate_training_by_init_period is not None:
        raise NotImplementedError(
            "These training modes are also unsupported by the current inference.py"
        )

    explicit_split = s.split_strategy == "explicit"
    train_end = s.train_end if explicit_split else s.val_end

    forecast_vars = [s.var_fc, *s.extra_input_variables]
    if len(forecast_vars) != len(set(forecast_vars)):
        raise ValueError("Duplicate forecast input variables")

    orography_ds = (
        xr.open_zarr(s.orography_path)
        if s.orography_encoding and s.orography_path is not None
        else None
    )

    datasets = make_train_test_datasets_for_leadtime(
        forecast_ds_path=[s.input_fc, *s.extra_input_fc],
        analysis_ds_path=s.input_an,
        leadtime=leadtime,
        leadtime_unit=s.leadtime_unit,
        train_start=s.train_start,
        train_end=train_end,
        val_start=s.val_start if explicit_split else None,
        val_end=s.val_end if explicit_split else None,
        test_start=s.test_start,
        test_end=s.test_end,
        target_mode=s.target_mode,
        clim_period=s.clim_period,
        forecast_vars=forecast_vars,
        analysis_vars=[s.var_an],
        region=s.region,
        dataset_kwargs=dataset_kwargs_from_settings(s),
        temporal_evolution_encoding=s.temporal_evolution_encoding,
        climatology_encoding=s.climatology_encoding,
        climatology_error_encoding=s.climatology_error_encoding,
        seasonal_encoding=s.seasonal_encoding and s.channel_representation != "init_period",
        ensemble_encoding=s.ensemble_encoding,
        spatial_encoding=s.spatial_encoding,
        orography_encoding=s.orography_encoding,
        orography_ds=orography_ds,
        input_realization_avg=s.input_realization_avg,
        interpolate_analysis=INTERPOLATE,
        materialize=False,
        separate_training_by_init_period=None,
        defer_dataset_creation=False,
    )

    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    test_dataset = datasets["test"]

    if not isinstance(train_dataset, XarrayDataset):
        raise TypeError("Expected XarrayDataset for training data")
    if val_dataset is not None and not isinstance(val_dataset, XarrayDataset):
        raise TypeError("Expected XarrayDataset for validation data")
    if not isinstance(test_dataset, XarrayDataset):
        raise TypeError("Expected XarrayDataset for test data")

    normalize_input, normalize_target = make_normalizers(
        s, train_dataset, val_dataset,
    )

    # Mirror inference.py's dataset normalization configuration
    train_dataset.sample_target_mean = s.sample_target_mean
    train_dataset.sample_target_std = s.sample_target_std
    test_dataset.sample_target_mean = "input"
    test_dataset.sample_target_std = "input"

    train_dataset.transform_x = normalize_input
    train_dataset.transform_y = normalize_target
    test_dataset.transform_x = normalize_input
    test_dataset.transform_y = normalize_target

    if val_dataset is not None:
        val_dataset.transform_x = normalize_input
        val_dataset.transform_y = normalize_target

    return (
        {"train": train_dataset, "val": val_dataset, "test": test_dataset},
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
    inference_period: TimeRangeSplit | tuple[str, str] | None = None,
    interpolate: bool = True,
) -> Fields:
    valid_time_range = _select_range(s, time_range)

    # Whole available region spatial coordinates
    valid_lat_range, valid_lon_range = _get_valid_spatial_range(s)

    # Select correct MLFC path
    mlfc_path = None

    if inference_period is not None:
        inference_period = _select_range(s, inference_period)
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
    forecast_da: xr.DataArray,
    analysis_da: xr.DataArray,
    corrected_da: xr.DataArray | None,
    sample_index: int,
    normalize_input: Normalize,
    normalize_target: Normalize,
) -> Fields:
    """Plot exactly the input/target tensors seen by the model.

    Error and correction maps use separate *target-space* baseline/ML prediction.
    The physical fields serve only to map the reconstructed ML forecast back
    into training-target space; they are not used as the displayed NN input.
    """
    x, y, _, month = dataset[sample_index]
    input_tensor = x.detach().float().cpu()
    target_tensor = y.detach().float().cpu()
    raw_input = dataset.x[sample_index].detach().float().cpu()
    raw_target = dataset.y[sample_index].detach().float().cpu()
    input_mask = dataset.x_mask[sample_index].detach().bool().cpu()
    target_mask = dataset.y_mask[sample_index].detach().bool().cpu()

    corrected_kwargs: dict[str, torch.Tensor] = {}
    if normalize_target.uses_sample_stats:
        input_mean, input_std = normalize_input.resolve_sample_mean_or_std(
            tensor=raw_input.unsqueeze(0),
            mask=dataset.x_mask[sample_index:sample_index + 1],
        )
        included = normalize_input.included_channels
        if included is None:
            raise ValueError("Input normalizer has no included_channels")
        positions = {channel: pos for pos, channel in enumerate(included)}
        selected = [positions[i] for i in dataset.target_to_input_channel_indices()]
        if normalize_target.mean_spec == "sample":
            corrected_kwargs["mean"] = input_mean[:, selected][0]
        if normalize_target.std_spec == "sample":
            corrected_kwargs["std"] = input_std[:, selected][0]

    ref_ds = dataset.input_ds
    lat_dim = ref_ds.earthml.guessed_dims.latitude
    lon_dim = ref_ds.earthml.guessed_dims.longitude
    if lat_dim is None or lon_dim is None:
        raise ValueError("Could not determine latitude/longitude dimensions")
    latitude = np.asarray(ref_ds[lat_dim].values)
    longitude = np.asarray(ref_ds[lon_dim].values)

    def physical_sample(da: xr.DataArray) -> torch.Tensor:
        time_dim = da.earthml.guessed_dims.time
        lead_dim = da.earthml.guessed_dims.leadtime
        if time_dim is not None and time_dim in da.dims:
            da = da.sel({time_dim: np.datetime64(init_time)})
        if lead_dim is not None and lead_dim in da.dims:
            da = da.sel({lead_dim: leadtime})
        da = da.sel({lat_dim: ref_ds[lat_dim], lon_dim: ref_ds[lon_dim]})
        realization_dim = da.earthml.guessed_dims.realization
        if realization_dim is not None and realization_dim in da.dims:
            n_realizations = da.sizes[realization_dim]
            if dataset.samples_per_init > 1 and n_realizations == dataset.samples_per_init:
                da = da.isel({realization_dim: sample_index % dataset.samples_per_init}, drop=True)
            elif raw_target.shape[0] == n_realizations:
                da = da.transpose(realization_dim, lat_dim, lon_dim)
            else:
                da = da.mean(realization_dim, skipna=True)
        da = da.squeeze(drop=True)
        if da.ndim == 2:
            da = da.transpose(lat_dim, lon_dim).expand_dims(channel=[0])
        elif da.ndim == 3:
            nonspatial = [d for d in da.dims if d not in (lat_dim, lon_dim)]
            if len(nonspatial) != 1:
                raise ValueError(f"Unexpected physical dimensions: {da.dims}")
            da = da.transpose(nonspatial[0], lat_dim, lon_dim)
        else:
            raise ValueError(f"Unexpected physical dimensions: {da.dims}")
        result = torch.as_tensor(np.asarray(da.values).copy(), dtype=raw_target.dtype)
        if result.shape != raw_target.shape:
            raise ValueError(f"Physical shape {result.shape} != target shape {raw_target.shape}")
        return result

    # Target transformations in these modes are additive (analysis, residual,
    # anomaly, anomaly_residual and their realization variants). Consequently
    # adding a physical forecast-minus-analysis difference in unnormalized
    # target space maps the physical forecast into the same training target space.
    supported_modes = {
        "analysis", "residual", "residual_realization", "anomaly",
        "anomaly_residual", "anomaly_residual_realization",
    }
    if s.target_mode not in supported_modes:
        raise ValueError(f"Unsupported target mode for normalized plots: {s.target_mode!r}")

    forecast_phys = physical_sample(forecast_da)
    analysis_phys = physical_sample(analysis_da)
    baseline_raw = raw_target + (forecast_phys - analysis_phys)
    baseline_tensor = normalize_target(
        baseline_raw, months=int(month), **corrected_kwargs,
    ).detach().float().cpu()

    corrected_tensor: torch.Tensor | None = None
    if corrected_da is not None:
        corrected_phys = physical_sample(corrected_da)
        corrected_raw = raw_target + (corrected_phys - analysis_phys)
        corrected_tensor = normalize_target(
            corrected_raw, months=int(month), **corrected_kwargs,
        ).detach().float().cpu()

    def to_dataarray(name: str, tensor: torch.Tensor, mask: torch.Tensor) -> xr.DataArray:
        if tensor.ndim != 3 or tensor.shape != mask.shape:
            raise ValueError(
                f"{name}: expected equal (C,H,W) tensor/mask, "
                f"got {tuple(tensor.shape)} and {tuple(mask.shape)}"
            )
        if tensor.shape[-2:] != (len(latitude), len(longitude)):
            raise ValueError(f"{name}: spatial shape does not match lat/lon coordinates")
        values = tensor.numpy().astype(float, copy=True)
        values[~mask.numpy()] = np.nan
        return xr.DataArray(
            values,
            dims=("channel", lat_dim, lon_dim),
            coords={
                "channel": np.arange(tensor.shape[0]),
                lat_dim: latitude,
                lon_dim: longitude,
            },
            name=name,
            attrs={"units": "normalized", "long_name": f"normalized {name}"},
        )

    return Fields(
        input=to_dataarray("input", input_tensor, input_mask),
        baseline=to_dataarray("baseline", baseline_tensor, target_mask),
        target=to_dataarray("target", target_tensor, target_mask),
        corrected=(
            to_dataarray("corrected", corrected_tensor, target_mask)
            if corrected_tensor is not None else None
        ),
    )

# ============================================================================
# Subset helpers
# ============================================================================

def subset_spatially(field: xr.DataArray) -> xr.DataArray:
    """Apply optional latitude/longitude plotting subset"""
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
    if WANTED_TIMES is None:
        requested_times = None
    else:
        requested_times = pd.DatetimeIndex(
            pd.to_datetime(WANTED_TIMES)
        ).unique().sort_values()

        if requested_times.empty:
            return []

    time_range = TIME_SPLIT

    # ----------------------------------------------------------
    # Load physical fields
    # ----------------------------------------------------------

    physical_fields = get_physical_fields(
        s=s,
        time_range=time_range,
        wanted_leadtimes=[leadtime],
        wanted_times=(
            None
            if requested_times is None
            else [str(t) for t in requested_times]
        ),
        inference_period=INFERENCE_PERIOD,
        interpolate=INTERPOLATE,
    )

    # ----------------------------------------------------------
    # Select available initialization times
    # ----------------------------------------------------------

    time_dim = physical_fields.input.earthml.guessed_dims.time

    if time_dim is None:
        raise ValueError(
            "Could not determine physical initialization-time dimension."
        )

    available_times = pd.DatetimeIndex(
        physical_fields.input[time_dim].values
    )

    if requested_times is None:
        requested_times = available_times.unique().sort_values()

    # ----------------------------------------------------------
    # Determine available samples per split
    # ----------------------------------------------------------

    times_by_split: dict[str, list[pd.Timestamp]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    if normalized:
        datasets, normalize_input, normalize_target = (
            make_normalized_datasets_for_leadtime(s, leadtime)
        )

        selected_splits = {
            "train": ("train",),
            "val": ("val",),
            "test": ("test",),
            "train_val": ("train", "val"),
            "full": ("train", "val", "test"),
        }[TIME_SPLIT]

        for split in selected_splits:
            dataset = datasets[split]
            if dataset is None:
                continue

            sample_times = pd.DatetimeIndex(
                get_sample_times(dataset)
            ).unique().sort_values()

            times_by_split[split] = list(
                sample_times[
                    sample_times.isin(requested_times)
                    & sample_times.isin(available_times)
                ]
            )

    # ----------------------------------------------------------
    # Physical anomalies
    # ----------------------------------------------------------

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
        for split in selected_splits:
            wanted_times = times_by_split[split]
            if not wanted_times:
                continue

            dataset = datasets[split]

            if dataset is None:
                raise ValueError(
                    f"Dataset for split {split!r} is unavailable."
                )

            sample_times = pd.DatetimeIndex(
                get_sample_times(dataset)
            )

            for init_time in wanted_times:
                matching = np.flatnonzero(
                    sample_times == init_time
                )

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
                        forecast_da=physical_fields.input,
                        analysis_da=physical_fields.target,
                        corrected_da=physical_fields.corrected,
                        sample_index=int(sample_index),
                        init_time=init_time,
                        normalize_input=normalize_input,
                        normalize_target=normalize_target,
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
        leadtime_dim = physical_fields.input.earthml.guessed_dims.leadtime

        if leadtime_dim is None:
            raise ValueError(
                "Could not determine physical leadtime dimension."
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
        "normalized"
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
        / "fields"
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
        "normalized"
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
        / "fields"
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
