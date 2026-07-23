from typing import Sequence, Literal, TypedDict

from pathlib import Path

import gc

from rich import print
from rich.console import Console

import numpy as np
import pandas as pd
import xarray as xr

from dask.diagnostics.progress import ProgressBar

import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from earthml import (
    LeadtimeUnit,
    ClimPeriod,
    TargetMode,
    XarrayDataset,
    build_net,
    Normalize,
    MonthlyNormalize,
    SplitDataModule,
    Table,
    Settings,
    calculate_climatology,
    select_clim_for_time,
    open_zarr,
    save_zarr,
    safe_chunk_spec,
)


class LeadtimeDatasets(TypedDict):
    train: XarrayDataset | dict[str, XarrayDataset]
    val: XarrayDataset | dict[str, XarrayDataset] | None
    test: XarrayDataset | dict[str, XarrayDataset]
    x_clim: xr.Dataset | None
    y_clim: xr.Dataset | None

PredictionRecord = tuple[
    int,          # leadtime
    str | None,   # init period
    str,          # regional name
    Path,         # prediction store
]

console = Console()


def compute_leadtimes(
    leadtime_value: int | float,
    leadtime_unit: LeadtimeUnit,
):
    if leadtime_unit in {LeadtimeUnit.MONTHS, LeadtimeUnit.YEARS}:
        if not float(leadtime_value).is_integer():
            raise ValueError(
                f"{leadtime_unit.value} leadtime must be an integer, "
                f"got {leadtime_value}"
            )

    if leadtime_unit == LeadtimeUnit.YEARS:
        return pd.DateOffset(years=int(leadtime_value))
    elif leadtime_unit == LeadtimeUnit.MONTHS:
        return pd.DateOffset(months=int(leadtime_value))
    elif leadtime_unit == LeadtimeUnit.DAYS:
        return pd.Timedelta(days=leadtime_value)
    elif leadtime_unit == LeadtimeUnit.HOURS:
        return pd.Timedelta(hours=leadtime_value)
    raise ValueError(f"Unsupported leadtime_unit={leadtime_unit}")


def add_or_set_leadtime(ds: xr.Dataset, lt: int) -> xr.Dataset:
    # If leadtime already exists as a dimension, select/squeeze it
    if "leadtime" in ds.dims:
        if ds.sizes["leadtime"] == 1:
            ds = ds.isel(leadtime=0, drop=True)
        else:
            raise ValueError(f"Expected one leadtime in saved prediction, got {ds.sizes['leadtime']}")

    # If leadtime exists only as a scalar coordinate or variable, remove it
    if "leadtime" in ds.coords:
        ds = ds.drop_vars("leadtime")
    elif "leadtime" in ds.data_vars:
        ds = ds.drop_vars("leadtime")

    return ds.expand_dims(leadtime=[lt])


def seasonal_cycle_encoder(
    ds: xr.Dataset,
    *,
    leadtime: int | float,
    leadtime_unit: LeadtimeUnit,
    time_dim: str | None = None,
) -> xr.Dataset:
    if time_dim is None:
        time_dim = ds.earthml.guessed_dims.time

    if time_dim is None or time_dim not in ds.dims:
        raise ValueError("Could not determine dataset time dimension.")

    valid_times = compute_valid_times(
        ds[time_dim].values,
        leadtime_value=leadtime,
        leadtime_unit=leadtime_unit,
    )

    valid_times = pd.DatetimeIndex(valid_times)

    init_times = pd.DatetimeIndex(ds[time_dim].values)

    init_phase = (init_times.month.to_numpy() - 1) / 12.0
    init_angle = 2.0 * np.pi * init_phase

    valid_phase = (valid_times.month.to_numpy() - 1) / 12.0
    valid_angle = 2.0 * np.pi * valid_phase

    seasonal = xr.Dataset(
        {
            f"valid_sin": ((time_dim,), np.sin(valid_angle).astype(np.float32)),
            f"valid_cos": ((time_dim,), np.cos(valid_angle).astype(np.float32)),
            "init_sin": ((time_dim,), np.sin(init_angle).astype(np.float32)),
            "init_cos": ((time_dim,), np.cos(init_angle).astype(np.float32)),
        },
        coords={
            time_dim: ds[time_dim],
        },
    )

    return xr.merge(
        [ds, seasonal],
        compat="equals",
        join="exact",
    )


def ensemble_encoder(ds: xr.Dataset) -> xr.Dataset:
    realization_dim = ds.earthml.guessed_dims.realization

    if realization_dim is None:
        raise ValueError(
            "Could not determine the realization dimension."
        )

    encoded: dict[str, xr.DataArray] = {}

    for name, da in ds.data_vars.items():
        if realization_dim not in da.dims:
            continue

        encoded[f"{name}_mean"] = da.mean(
            dim=realization_dim,
            keep_attrs=True,
            skipna=True,
        )

        encoded[f"{name}_spread"] = da.std(
            dim=realization_dim,
            keep_attrs=True,
            skipna=True,
            ddof=0,
        )

    if not encoded:
        raise ValueError(
            f"No variables contain realization dimension "
            f"{realization_dim!r}. Variables: {list(ds.data_vars)}"
        )

    return xr.merge(
        [ds, xr.Dataset(encoded)],
        compat="equals",
        join="exact",
    )


def compute_valid_times(
    init_times,
    leadtime_value: int | float,
    leadtime_unit: LeadtimeUnit,
):
    init_times = pd.DatetimeIndex(init_times)

    leadtime = compute_leadtimes(leadtime_value, leadtime_unit)

    return init_times + leadtime


def extract_period_from_ds(
    fc_ds: xr.Dataset,
    an_ds: xr.Dataset,
    start: str,
    end: str,
    leadtime: int | float,
    leadtime_unit: LeadtimeUnit,
    interpolate_analysis: bool = True,
    materialize: bool = False,
) -> tuple[xr.Dataset, xr.Dataset]:
    time_dim = fc_ds.earthml.guessed_dims.time
    leadtime_dim = fc_ds.earthml.guessed_dims.leadtime

    # Select forecast starts and the requested leadtime
    fc_ds = fc_ds.sel({
        time_dim: slice(start, end),
        leadtime_dim: int(leadtime),
    })

    # Select valid times
    valid_times = compute_valid_times(
        fc_ds[time_dim].values,
        leadtime_value=leadtime,
        leadtime_unit=leadtime_unit,
    )

    valid_times = pd.DatetimeIndex(valid_times)
    an_ds_times = pd.DatetimeIndex(pd.to_datetime(an_ds[time_dim].values))

    ok_times = valid_times.isin(an_ds_times)

    fc_ds = fc_ds.isel({time_dim: ok_times})
    valid_times = valid_times[ok_times]

    an_ds = an_ds.sel({time_dim: valid_times})

    # Make an_ds use forecast start time as its sample axis
    an_ds = an_ds.assign_coords({
        time_dim: fc_ds[time_dim].values
    })

    if interpolate_analysis:
        # Regrid analysis into forecast
        an_ds = an_ds.interp(
            latitude=fc_ds.latitude,
            longitude=fc_ds.longitude,
        )

    fc_ds, an_ds = xr.align(fc_ds, an_ds, join="exact")

    if materialize:
        with ProgressBar():
            fc_ds = fc_ds.load()
            an_ds = an_ds.load()

    return fc_ds, an_ds


def make_leadtime_pair(
    fc_ds: xr.Dataset,
    an_ds: xr.Dataset,
    leadtime: int | float,
    leadtime_unit: LeadtimeUnit,
    start: str,
    end: str,
    fc_clim: xr.Dataset | None,
    an_clim: xr.Dataset | None,
    target_mode: TargetMode = "analysis",
    clim_period: ClimPeriod = ClimPeriod.MONTH,
    seasonal_encoding: bool = False,
    ensemble_encoding: bool = False,
    interpolate_analysis: bool = True,
    materialize: bool = False,
) -> tuple[xr.Dataset, xr.Dataset]:
    def _encode_input(ds: xr.Dataset) -> xr.Dataset:
        if ensemble_encoding:
            ds = ensemble_encoder(ds)

        if seasonal_encoding:
            ds = seasonal_cycle_encoder(
                ds,
                leadtime=leadtime,
                leadtime_unit=leadtime_unit,
            )

        return ds

    fc_period_ds, an_period_ds = extract_period_from_ds(
        fc_ds=fc_ds,
        an_ds=an_ds,
        start=start,
        end=end,
        leadtime=leadtime,
        leadtime_unit=leadtime_unit,
        interpolate_analysis=interpolate_analysis,
        materialize=materialize,
    )

    if fc_period_ds.sizes.get(fc_ds.earthml.guessed_dims.time, 0) == 0:
        raise ValueError(
            f"No forecast samples found for period {start} -> {end}, "
            f"leadtime={leadtime} {leadtime_unit.value}"
        )

    if an_period_ds.sizes.get(an_ds.earthml.guessed_dims.time, 0) == 0:
        raise ValueError(
            f"No analysis samples found for period {start} -> {end}, "
            f"leadtime={leadtime} {leadtime_unit.value}"
        )

    if target_mode == "analysis":
        return _encode_input(fc_period_ds), an_period_ds

    if target_mode == "residual":
        fc_base = (
            fc_period_ds.mean("realization")
            if "realization" in fc_period_ds.dims
            else fc_period_ds
        )
        target_ds = an_period_ds - fc_base
        return _encode_input(fc_period_ds), target_ds

    if target_mode == "residual_realization":
        target_ds = an_period_ds - fc_period_ds
        return _encode_input(fc_period_ds), target_ds

    if fc_clim is None or an_clim is None:
        raise ValueError(
            f"Climatologies are required for target_mode={target_mode!r}"
        )

    fc_clim_for_period = select_clim_for_time(
        fc_clim,
        fc_period_ds[fc_period_ds.earthml.guessed_dims.time].values,
        clim_period,
    )

    an_clim_for_period = select_clim_for_time(
        an_clim,
        an_period_ds[an_period_ds.earthml.guessed_dims.time].values,
        clim_period,
    )

    fc_clim_for_period = fc_clim_for_period.chunk(
        safe_chunk_spec(fc_clim_for_period, fc_clim)
    )
    an_clim_for_period = an_clim_for_period.chunk(
        safe_chunk_spec(an_clim_for_period, an_clim)
    )

    fc_anom = (fc_period_ds - fc_clim_for_period).unify_chunks()

    if fc_anom.sizes.get(fc_anom.earthml.guessed_dims.time, 0) == 0:
        raise ValueError(
            "Climatology alignment removed every time sample. "
            f"Period={start} -> {end}; "
            f"forecast times={fc_period_ds[fc_anom.earthml.guessed_dims.time].values[:3]}; "
            f"climatology dims={fc_clim.dims}"
        )

    an_anom = (an_period_ds - an_clim_for_period).unify_chunks()

    if target_mode == "anomaly":
        return _encode_input(fc_anom), an_anom

    if target_mode == "anomaly_residual":
        fc_anom_base = (
            fc_anom.mean("realization")
            if "realization" in fc_anom.dims
            else fc_anom
        )
        target_anom_ds = an_anom - fc_anom_base
        return _encode_input(fc_anom), target_anom_ds

    if target_mode == "anomaly_residual_realization":
        target_anom_ds = an_anom - fc_anom
        return _encode_input(fc_anom), target_anom_ds

    raise ValueError(f"Unsupported target_mode={target_mode!r}")


def make_train_test_datasets_for_leadtime(
    forecast_ds_path: str | Path,
    analysis_ds_path: str | Path,
    leadtime: int | float,
    leadtime_unit: LeadtimeUnit,
    train_start: str,
    train_end: str,
    val_start: str | None,
    val_end: str | None,
    test_start: str,
    test_end: str,
    target_mode: TargetMode = "analysis",
    clim_period: ClimPeriod = ClimPeriod.MONTH,
    forecast_vars: Sequence[str] | None = None,
    analysis_vars: Sequence[str] | None = None,
    region: dict | None = None,
    dataset_kwargs: dict | None = None,
    seasonal_encoding: bool = False,
    ensemble_encoding: bool = False,
    interpolate_analysis: bool = True,
    materialize: bool = False,
    separate_training_by_init_period: ClimPeriod | None = None,
) -> LeadtimeDatasets:
    dataset_kwargs = dataset_kwargs or {}

    lon = region["lon"] if region is not None else None
    lat = region["lat"] if region is not None else None

    fc_ds = open_zarr(forecast_ds_path)
    an_ds = open_zarr(analysis_ds_path)

    if forecast_vars is not None:
        fc_ds = fc_ds[list(forecast_vars)]

    if analysis_vars is not None:
        an_ds = an_ds[list(analysis_vars)]

    # Select region
    if lat is not None:
        fc_ds = fc_ds.sel(latitude=slice(*lat))
        an_ds = an_ds.sel(latitude=slice(*lat))
    if lon is not None:
        fc_ds = fc_ds.sel(longitude=slice(*lon))
        an_ds = an_ds.sel(longitude=slice(*lon))

    if target_mode in {"anomaly", "anomaly_residual", "anomaly_residual_realization"}:
        fc_clim_ds, an_clim_ds = extract_period_from_ds(
            fc_ds=fc_ds,
            an_ds=an_ds,
            start=train_start,
            end=train_end,
            leadtime=leadtime,
            leadtime_unit=leadtime_unit,
            interpolate_analysis=interpolate_analysis,
        )

        fc_time_dim = fc_ds.earthml.guessed_dims.time
        an_time_dim = an_ds.earthml.guessed_dims.time

        fc_clim = calculate_climatology(fc_clim_ds, fc_time_dim, clim_period)
        an_clim = calculate_climatology(an_clim_ds, an_time_dim, clim_period)

    else:
        fc_clim = None
        an_clim = None

    def _generate_period_datasets(
        x: xr.Dataset,
        y: xr.Dataset,
    ) -> dict[str, XarrayDataset]:
        x_time_dim = x.earthml.guessed_dims.time
        y_time_dim = y.earthml.guessed_dims.time

        if x_time_dim is None or y_time_dim is None:
            raise ValueError("Could not determine input or target time dimension")

        if x.sizes[x_time_dim] != y.sizes[y_time_dim]:
            raise ValueError(
                "Input and target have different sample counts: "
                f"{x.sizes[x_time_dim]} != {y.sizes[y_time_dim]}"
            )

        if separate_training_by_init_period == ClimPeriod.MONTH:
            labels = np.asarray(x[x_time_dim].dt.month.values)
        else:
            raise NotImplementedError(
                f"Currently only separate_training_by_init_period="
                f"{ClimPeriod.MONTH!r} is supported"
            )

        datasets: dict[str, XarrayDataset] = {}

        for label in np.unique(labels):
            print("_generate_period_datasets:", label)
            indices = np.flatnonzero(labels == label)

            datasets[str(label)] = XarrayDataset(
                x.isel({x_time_dim: indices}),
                y.isel({y_time_dim: indices}),
                **dataset_kwargs,
            )

        return datasets

    x_train, y_train = make_leadtime_pair(
        fc_ds=fc_ds,
        an_ds=an_ds,
        leadtime=leadtime,
        leadtime_unit=leadtime_unit,
        start=train_start,
        end=train_end,
        fc_clim=fc_clim,
        an_clim=an_clim,
        target_mode=target_mode,
        clim_period=clim_period,
        seasonal_encoding=seasonal_encoding,
        ensemble_encoding=ensemble_encoding,
        interpolate_analysis=interpolate_analysis,
        materialize=materialize,
    )

    x_train, y_train = drop_zero_valid_target_samples(
        x_train,
        y_train,
        label="train",
    )

    val_ds: dict[str, XarrayDataset] | XarrayDataset | None = None
    if val_start is not None and val_end is not None:
        x_val, y_val = make_leadtime_pair(
            fc_ds=fc_ds,
            an_ds=an_ds,
            leadtime=leadtime,
            leadtime_unit=leadtime_unit,
            start=val_start,
            end=val_end,
            fc_clim=fc_clim,
            an_clim=an_clim,
            target_mode=target_mode,
            clim_period=clim_period,
            seasonal_encoding=seasonal_encoding,
            ensemble_encoding=ensemble_encoding,
            interpolate_analysis=interpolate_analysis,
            materialize=materialize,
        )

        x_val, y_val = drop_zero_valid_target_samples(
            x_val,
            y_val,
            label="validation",
        )

        if separate_training_by_init_period is None:
            val_ds = XarrayDataset(x_val, y_val, **dataset_kwargs)
        else:
            val_ds = _generate_period_datasets(x_val, y_val)

    x_test, y_test = make_leadtime_pair(
        fc_ds=fc_ds,
        an_ds=an_ds,
        leadtime=leadtime,
        leadtime_unit=leadtime_unit,
        start=test_start,
        end=test_end,
        fc_clim=fc_clim,
        an_clim=an_clim,
        target_mode=target_mode,
        clim_period=clim_period,
        seasonal_encoding=seasonal_encoding,
        ensemble_encoding=ensemble_encoding,
        interpolate_analysis=interpolate_analysis,
        materialize=materialize,
    )

    x_test, y_test = drop_zero_valid_target_samples(
        x_test,
        y_test,
        label="test",
    )

    if separate_training_by_init_period is None:
        train_ds: dict[str, XarrayDataset] | XarrayDataset = XarrayDataset(
            x_train,
            y_train,
            **dataset_kwargs,
        )
        test_ds: dict[str, XarrayDataset] | XarrayDataset = XarrayDataset(
            x_test,
            y_test,
            **dataset_kwargs,
        )
    else:
        train_ds = _generate_period_datasets(x_train, y_train)
        test_ds = _generate_period_datasets(x_test, y_test)

    return {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
        "x_clim": fc_clim,
        "y_clim": an_clim,
    }


def drop_zero_valid_target_samples(
    x_ds: xr.Dataset,
    y_ds: xr.Dataset,
    *,
    time_dim: str | None = None,
    label: str = "dataset",
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Remove time samples for which every target value is non-finite.

    A sample is retained when at least one target variable contains at least
    one finite value across all non-time dimensions.
    """
    if time_dim is None:
        time_dim = y_ds.earthml.guessed_dims.time

    if time_dim is None:
        raise ValueError(f"{label}: could not determine target time dimension")

    if time_dim not in y_ds.dims:
        raise ValueError(
            f"{label}: time dimension {time_dim!r} is not present in target"
        )

    valid_by_variable: list[xr.DataArray] = []

    for name, da in y_ds.data_vars.items():
        if time_dim not in da.dims:
            continue

        reduce_dims = tuple(dim for dim in da.dims if dim != time_dim)

        finite = np.isfinite(da)

        if reduce_dims:
            finite = finite.any(dim=reduce_dims)

        valid_by_variable.append(finite)

    if not valid_by_variable:
        raise ValueError(
            f"{label}: no target variables contain time dimension {time_dim!r}"
        )

    valid_sample = valid_by_variable[0]
    for valid in valid_by_variable[1:]:
        valid_sample = valid_sample | valid

    # This is only a one-dimensional boolean vector over time.
    valid_sample = valid_sample.compute()

    keep_indices = np.flatnonzero(valid_sample.values)
    dropped_indices = np.flatnonzero(~valid_sample.values)

    if dropped_indices.size:
        dropped_times = y_ds[time_dim].values[dropped_indices]

        print(
            f"[yellow]{label}: dropping {dropped_indices.size} "
            f"zero-valid target samples out of "
            f"{y_ds.sizes[time_dim]}[/yellow]"
        )

        for index, timestamp in zip(
            dropped_indices,
            dropped_times,
            strict=False,
        ):
            print(
                f"[yellow]  index={index}, time={timestamp}[/yellow]"
            )

    if keep_indices.size == 0:
        raise ValueError(
            f"{label}: every target sample has zero valid elements"
        )

    x_ds = x_ds.isel({time_dim: keep_indices})
    y_ds = y_ds.isel({time_dim: keep_indices})

    x_ds, y_ds = xr.align(
        x_ds,
        y_ds,
        join="exact",
    )

    return x_ds, y_ds


def resolve_accelerator_and_device() -> tuple[str, torch.device]:
    if torch.cuda.is_available():
        return "gpu", torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.device("mps")
    return "cpu", torch.device("cpu")


def init_callbacks(
    ckpt_folder_path: str | Path,
    weights_folder_path: str | Path,
    patience: int,
):
    # Initialize trainer callbacks
    callbacks = []
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        verbose=True,
        mode="min"
    )
    callbacks.append(early_stop_callback)

    # Checkpointing every N epochs
    periodic_checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_folder_path,
        every_n_epochs=1,
        save_last=True,
        save_top_k=0,
        filename="checkpoint",
        enable_version_counter=False,
    )

    callbacks.append(periodic_checkpoint_callback)

    # Model best weights, MUST be the last ModelCheckpoint. Can append from here since ModelCheckpoint will always be the last callbacks
    best_weights_callback = ModelCheckpoint(
        dirpath=weights_folder_path,
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename="weights",
        enable_version_counter=False,
    )
    callbacks.append(best_weights_callback)
    return callbacks


# Save utils
def _infer_RT_from_source(dataset: XarrayDataset) -> tuple[int, int]:
    target_ds = dataset.target_ds

    tdim = "time"
    T_out = int(target_ds.sizes.get(tdim, 1))

    # If target was deterministic/averaged, model output has no ensemble sample axis.
    if getattr(dataset, "realization_as_channel", False):
        if getattr(dataset, "output_realizations", "deterministic") == "ensemble":
            input_ds = dataset.input_ds
            rdim = "realization"
            R_out = int(input_ds.sizes.get(rdim, 1))
        else:
            R_out = 1
    else:
        input_ds = dataset.input_ds
        rdim = "realization"
        R_out = int(input_ds.sizes.get(rdim, 1))

        # If the target is deterministic and repeated across input R, old path needs R_out.
        # Keep this for realization_as_channel=False.
        if getattr(dataset, "target_realization_avg", True):
            R_out = R_out

    return R_out, T_out

def _reconstruct_pred_tensor(
    data: torch.Tensor,     # (N,C,H,W)
    meta_ds: xr.Dataset,    # for slicing/copying coords/attrs
    R_out: int,
    T_out: int,
    realization_as_channel: bool = False,
) -> tuple[torch.Tensor, xr.Dataset]:
    """
    Canonical output ALWAYS (C, R_out, T_out, H, W).
    If output is deterministic, ensures R_out==1 and adds singleton realization dim.
    """
    if data.ndim != 4:
        raise ValueError(f"Expected preds as (N,C,H,W), got {tuple(data.shape)}")

    # (C,N,H,W)
    x = data.permute(1, 0, 2, 3).contiguous()
    C_model, N, H, W = x.shape

    if realization_as_channel:
        # Here N should be time
        if N != T_out:
            raise ValueError(f"realization_as_channel=True expects N==T ({N} != {T_out})")

        if R_out == 1:
            x = x.unsqueeze(1)  # (C,1,T,H,W)
        else:
            # channels contain realizations: C_model == C * R_out
            if C_model % R_out != 0:
                raise ValueError(f"C_model={C_model} not divisible by R_out={R_out}")
            C = C_model // R_out
            # Inverse of dataset packing:
            # (C, T, R, H, W) -> transpose(0, 2, 1, 3, 4) -> (C*R, T, H, W)
            x = x.unflatten(0, (C, R_out)).contiguous()

    else:
        # Here realizations are flattened into N if R_out>1, else N is time
        if R_out == 1:
            if N != T_out:
                raise ValueError(f"deterministic expects N==T ({N} != {T_out})")
            x = x.unsqueeze(1)  # (C,1,T,H,W)
        else:
            if N != T_out * R_out:
                raise ValueError(f"ensemble expects N==T*R ({N} != {T_out*R_out})")
            # time-major flatten (t slow, r fast): (C,N,H,W)->(C,T,R,H,W)->(C,R,T,H,W)
            x = x.unflatten(1, (T_out, R_out)).permute(0, 2, 1, 3, 4).contiguous()

    # meta_ds only sliced to match inferred sizes; never used to infer them
    tdim = "time"
    rdim = "realization"
    ydim = "latitude"
    xdim = "longitude"

    indexers: dict[str, slice] = {}
    if rdim in meta_ds.dims:
        indexers[rdim] = slice(0, R_out)
    if tdim in meta_ds.dims:
        indexers[tdim] = slice(0, T_out)
    if ydim in meta_ds.dims:
        indexers[ydim] = slice(0, H)
    if xdim in meta_ds.dims:
        indexers[xdim] = slice(0, W)
    if indexers:
        meta_ds = meta_ds.isel(indexers)

    return x, meta_ds


def convert_to_xarray(
    data: torch.Tensor,
    dataset: XarrayDataset,
    var_list: Sequence,
):
    # Predictions live on the target grid, so keep target metadata/coords.
    meta_ds = dataset.target_ds

    R_out, T_out = _infer_RT_from_source(dataset)

    if getattr(dataset, "realization_as_channel", False):
        if getattr(dataset, "output_realizations", "deterministic") == "ensemble":
            realization_as_channel = True
        else:
            R_out = 1
            realization_as_channel = False
    else:
        realization_as_channel = False

    input_ds = dataset.input_ds
    target_rdim = meta_ds.earthml.guessed_dims.realization
    input_rdim = input_ds.earthml.guessed_dims.realization
    rdim = target_rdim or input_rdim or "realization"
    tdim = "time"
    ydim = "latitude"
    xdim = "longitude"

    allowed_dims = {tdim, ydim, xdim, rdim, "missed_time"}
    meta_ds = meta_ds.earthml.remove_dims_and_coords(allowed_dims)
    base_order = [rdim, tdim, ydim, xdim]
    order = [d for d in base_order if d in meta_ds.dims]
    order += [d for d in meta_ds.dims if d not in order]
    meta_ds = meta_ds.transpose(*order, missing_dims="ignore")

    input_has_matching_r = (
        input_rdim is not None
        and input_rdim in input_ds.dims
        and int(input_ds.sizes[input_rdim]) == R_out
    )

    # Ensemble predictions should follow the input realization convention
    # when the target metadata is deterministic/singleton.
    if (
        R_out > 1
        and rdim is not None
        and (
            rdim not in meta_ds.dims
            or int(meta_ds.sizes.get(rdim, 1)) != R_out
        )
    ):
        if not input_has_matching_r:
            raise ValueError(
                "Cannot reconstruct ensemble prediction coordinates: "
                f"target realization dim '{rdim}' has size "
                f"{meta_ds.sizes.get(rdim, 'missing')} but expected {R_out}, "
                f"and input realization dim '{input_rdim}' is unavailable."
            )

        if rdim in meta_ds.dims:
            meta_ds = meta_ds.drop_dims(rdim)

        input_rcoord = input_ds.coords.get(input_rdim)
        if input_rcoord is not None:
            if input_rdim != rdim:
                input_rcoord = input_rcoord.rename({input_rdim: rdim})
            meta_ds = meta_ds.assign_coords({rdim: input_rcoord})
        else:
            meta_ds = meta_ds.assign_coords({rdim: np.arange(R_out, dtype=np.int64)})

        base_order = [rdim, tdim, ydim, xdim]
        order = [d for d in base_order if d in meta_ds.dims]
        order += [d for d in meta_ds.dims if d not in order]
        meta_ds = meta_ds.transpose(*order, missing_dims="ignore")

    data_vars, meta_ds = _reconstruct_pred_tensor(
        data,
        meta_ds,
        R_out=R_out,
        T_out=T_out,
        realization_as_channel=realization_as_channel,
    )

    # Use meta dim names if present; otherwise drop them safely.
    dims = []
    if rdim in meta_ds.dims:
        dims.append(rdim)
    dims += [d for d in (tdim, ydim, xdim) if d in meta_ds.dims]

    # If metadata lacks realization dim, write deterministic view
    if rdim not in meta_ds.dims:
        data_vars = data_vars.squeeze(1)  # (C,T,H,W)

    pred_ds_raw = xr.Dataset(
        {
            var: (dims, data_vars[i].cpu().numpy())
            for i, var in enumerate(var_list)
        },
        coords={c: meta_ds.coords[c] for c in meta_ds.coords},
        attrs=meta_ds.attrs,
    )

    return pred_ds_raw


def print_training_recap(
    *,
    settings: Settings,
    exp_ratio: tuple[int, int],
    region_name: str,
    accelerator: str,
    device: torch.device,
    leadtime: int | float | str,
    normalization_name: str,
    n_channels: int,
    n_classes: int,
    longitude_padding: str,
    dry_run: bool,
    force_retrain: bool,
    force_test: bool,
    interpolate_analysis: bool,
    train_input_shape: tuple | None = None,
    train_target_shape: tuple | None = None,
    val_input_shape: tuple | None = None,
    val_target_shape: tuple | None = None,
    test_input_shape: tuple | None = None,
    test_target_shape: tuple | None = None,
    train_idx: list[int] | None = None,
    val_idx: list[int] | None = None,
    exp_name: str | None = None,
    weights_dir: Path | None = None,
    checkpoints_dir: Path | None = None,
) -> None:
    s = settings

    flat_recap = {
        "experiment.number": f"{exp_ratio[0]} / {exp_ratio[1]}",
        "experiment.name": exp_name or s.output_name,
        "experiment.leadtime": f"{leadtime} {s.leadtime_unit.value}",
        "experiment.dry_run": dry_run,
        "experiment.force_retrain": force_retrain,
        "experiment.force_test": force_test,
        "experiment.interpolate_analysis": interpolate_analysis,
        "experiment.torch_workers": s.torch_workers,
        "experiment.separate_training_by_init_period": s.separate_training_by_init_period,
        "experiment.regional_training": f"{s.regional_training}, {region_name} [lat={s.regional_training_lat_size} x lon={s.regional_training_lon_size}]",

        "data.forecast": f"{s.model_fc}/{s.var_fc}",
        "data.analysis": f"{s.model_an}/{s.var_an}",
        "data.full_region": f"{s.region_name} {s.region}",
        "data.train_period": f"{s.train_start} → {s.train_end}",
        "data.val_period": f"{s.val_start} → {s.val_end}",
        "data.test_period": f"{s.test_start} → {s.test_end}",
        "data.train_x": train_input_shape,
        "data.train_y": train_target_shape,
        "data.val_x": val_input_shape,
        "data.val_y": val_target_shape,
        "data.test_x": test_input_shape,
        "data.test_y": test_target_shape,

        "split.strategy": s.split_strategy,
        "split.shuffle_train_batch": s.shuffle_train_batch,
        "split.train_fraction": s.train_fraction,
        "split.train_samples": len(train_idx) if train_idx is not None else None,
        "split.val_samples": len(val_idx) if val_idx is not None else None,
        "split.train_idx": f"{train_idx[0]} → {train_idx[-1]}" if train_idx is not None and len(train_idx) else None,
        "split.val_idx": f"{val_idx[0]} → {val_idx[-1]}" if val_idx is not None and len(val_idx) else None,

        "model.network": s.net_name,
        "model.extra_net_kwargs": s.extra_net_kwargs,
        "model.loss": s.loss_name,
        "model.target_scale_degrees": s.target_scale_degrees if s.loss_name=="GeoMaskedMSELowFreqLoss" else None,
        "model.n_channels": n_channels,
        "model.n_classes": n_classes,
        "model.longitude_padding": longitude_padding,
        "model.channel_representation": s.channel_representation,
        "model.init_period_dim": s.init_period_dim if s.channel_representation=="init_period" else None,
        "model.output_realizations": s.output_realizations,
        "model.norm_layer": s.training_norm,

        "training.normalization": f"{normalization_name}(x), {normalization_name}(y)",
        "training.normalization_mode": s.normalization_mode,
        "training.seasonal_encoding": s.seasonal_encoding and s.channel_representation!="init_period",
        "training.learning_rate": s.init_learning_rate,
        "training.weight_decay": s.weight_decay,
        "training.batch_size": s.batch_size,
        "training.effective_batch_size": s.batch_size * s.accumulate_grad_batches,
        "training.max_epochs": s.max_epochs,
        "training.patience": s.early_stopping_patience,
        "training.precision": s.trainer_precision,
        "training.accelerator": f"{accelerator} ({device})",

        "paths.weights_dir": weights_dir,
        "paths.checkpoints_dir": checkpoints_dir,
    }

    console.print(Table(flat_recap, title="Training recap", twocols=True).table)


def _combine_predictions(
    s: Settings,
    data_type: Literal["train", "val", "test"],
    pred_records: list[
        tuple[int, str | None, str, Path]
    ],
) -> None:
    if not pred_records:
        return

    records_by_leadtime: dict[
        int,
        list[tuple[str | None, str, Path]],
    ] = {}

    for leadtime, init_period, regional_name, path in pred_records:
        records_by_leadtime.setdefault(
            leadtime,
            [],
        ).append(
            (
                init_period,
                regional_name,
                path,
            )
        )

    leadtime_datasets: list[xr.Dataset] = []

    for leadtime in sorted(records_by_leadtime):
        records_for_leadtime = records_by_leadtime[leadtime]

        records_by_init_period: dict[
            str | None,
            list[tuple[str, Path]],
        ] = {}

        for init_period, regional_name, path in records_for_leadtime:
            records_by_init_period.setdefault(
                init_period,
                [],
            ).append(
                (
                    regional_name,
                    path,
                )
            )

        init_period_datasets: list[xr.Dataset] = []

        for init_period in sorted(
            records_by_init_period,
            key=lambda value: (
                int(value) if value is not None else 0
            ),
        ):
            regional_records = records_by_init_period[init_period]

            regional_datasets: list[xr.Dataset] = []

            for regional_name, path in sorted(
                regional_records,
                key=lambda item: item[0],
            ):
                if not path.exists():
                    raise FileNotFoundError(
                        f"Missing {data_type} prediction store for "
                        f"leadtime={leadtime}, "
                        f"init_period={init_period}, "
                        f"region={regional_name}: {path}"
                    )

                regional_datasets.append(
                    open_zarr(path)
                )

            if len(regional_datasets) == 1:
                combined_regions = regional_datasets[0]
            else:
                combined_regions = xr.combine_by_coords(
                    regional_datasets,
                    combine_attrs="override",
                    data_vars="all",
                    coords="minimal",
                    compat="override",
                    join="exact",
                )

                for dim in ("latitude", "longitude"):
                    if dim in combined_regions.coords:
                        _, unique_indices = np.unique(
                            combined_regions[dim].values,
                            return_index=True,
                        )

                        if len(unique_indices) != combined_regions.sizes[dim]:
                            combined_regions = combined_regions.isel(
                                {
                                    dim: np.sort(unique_indices),
                                }
                            )

            init_period_datasets.append(combined_regions)

        if len(init_period_datasets) == 1:
            combined_periods = init_period_datasets[0]
        else:
            combined_periods = xr.concat(
                init_period_datasets,
                dim="time",
                coords="minimal",
                compat="override",
                join="exact",
            )

            combined_periods = combined_periods.sortby("time")

            time_values = combined_periods["time"].values

            if np.unique(time_values).size != time_values.size:
                _, duplicate_counts = np.unique(
                    time_values,
                    return_counts=True,
                )

                raise ValueError(
                    f"Duplicate initialization times while combining "
                    f"{data_type} predictions for "
                    f"leadtime={leadtime}. "
                    f"Maximum duplicate count="
                    f"{duplicate_counts.max()}"
                )

        combined_periods = add_or_set_leadtime(
            combined_periods,
            leadtime,
        )

        leadtime_datasets.append(combined_periods)

    combined_preds = xr.concat(
        leadtime_datasets,
        dim="leadtime",
        coords="minimal",
        compat="override",
        join="outer",
    )

    combined_preds = combined_preds.sortby("leadtime")

    combined_preds = combined_preds.chunk(
        safe_chunk_spec(combined_preds)
    )

    combined_preds["leadtime"].attrs.update(
        {
            "long_name": "forecast lead time",
            "units": s.leadtime_unit.value,
        }
    )

    combined_preds["time"].attrs.update(
        {
            "long_name": "forecast initialization time",
            "standard_name": "forecast_reference_time",
        }
    )

    output_store = (
        s.output_dir / f"{data_type}_corrected.zarr"
    )

    save_zarr(
        combined_preds,
        output_store,
        chunks={"leadtime": 1},
    )

    print(
        f"[green]Final combined {data_type} predictions: "
        f"{dict(combined_preds.sizes)}[/green]"
    )

    for ds in leadtime_datasets:
        try:
            ds.close()
        except Exception:
            pass

    del leadtime_datasets, combined_preds
    gc.collect()


def _test(
    test_trainer: L.Trainer,
    s: Settings,
    model,
    dataset: XarrayDataset,
    normalize_target: Normalize | MonthlyNormalize,
    dataloader: DataLoader,
    preds_store: Path,
    an_clim: xr.Dataset | None = None,
    log_monthly: bool = False,
):
    test_trainer.test(model, dataloaders=dataloader)

    preds_norm = model.test_preds.detach().float().cpu()
    targets_norm = model.test_targets.detach().float().cpu()

    if preds_norm.shape != targets_norm.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: "
            f"{preds_norm.shape} != {targets_norm.shape}"
        )

    error_norm = preds_norm - targets_norm

    model_mse = error_norm.square().mean()
    zero_mse = targets_norm.square().mean()

    masks = model.test_masks

    model_loss = model.compute_loss(
        prediction=preds_norm,
        target=targets_norm,
        mask=masks,
    )

    zero_loss = model.compute_loss(
        prediction=torch.zeros_like(targets_norm),
        target=targets_norm,
        mask=masks,
    )

    print("Normalized tensor diagnostics")
    print("  unweighted raw tensor MSE:", float(model_mse))
    print("  zero-output MSE:", float(zero_mse))
    print("  skill vs zero:", float(1.0 - model_mse / zero_mse))
    print("  model scalar bias:", float(error_norm.mean()))
    print("  target mean:", float(targets_norm.mean()))
    print("  prediction mean:", float(preds_norm.mean()))
    print(
        "prediction std:",
        float(preds_norm.std()),
    )
    print(
        "target std:",
        float(targets_norm.std()),
    )

    print("Loss-consistent normalized diagnostics")
    print("  model loss:", float(model_loss))
    print("  zero-output loss:", float(zero_loss))
    print(
        "  skill vs zero:",
        float(1.0 - model_loss / zero_loss.clamp_min(1e-8)),
    )

    # Per-grid-cell skill against predicting zero residual.
    valid = masks.bool()
    valid_count_map = valid.sum(dim=(0, 1))  # (H, W)

    model_mse_map = (
        error_norm.square()
        .masked_fill(~valid, 0.0)
        .sum(dim=(0, 1))
        / valid_count_map.clamp_min(1)
    )

    zero_mse_map = (
        targets_norm.square()
        .masked_fill(~valid, 0.0)
        .sum(dim=(0, 1))
        / valid_count_map.clamp_min(1)
    )

    valid_grid_cells = valid_count_map > 0
    improved_grid_cells = (
        model_mse_map < zero_mse_map
    ) & valid_grid_cells

    improved_grid_fraction = (
        improved_grid_cells.sum()
        / valid_grid_cells.sum().clamp_min(1)
    )

    print(
        "  improved grid cells:",
        f"{100.0 * float(improved_grid_fraction):.1f}%",
    )

    target_mean_map = targets_norm.mean(dim=0)
    prediction_mean_map = preds_norm.mean(dim=0)
    error_mean_map = error_norm.mean(dim=0)

    print("Target mean map")
    print("  mean abs:", float(target_mean_map.abs().mean()))
    print("  max abs:", float(target_mean_map.abs().max()))

    print("Prediction mean map")
    print("  mean abs:", float(prediction_mean_map.abs().mean()))
    print("  max abs:", float(prediction_mean_map.abs().max()))

    print("Error mean map")
    print("  mean abs:", float(error_mean_map.abs().mean()))
    print("  max abs:", float(error_mean_map.abs().max()))

    if log_monthly:
        preds = model.test_preds.float()
        targets = model.test_targets.float()
        months = model.test_months

        for month in torch.unique(months).sort().values:
            selected = months == month

            pred_month = preds[selected]
            target_month = targets[selected]
            error_month = pred_month - target_month

            model_mse = error_month.square().mean()
            zero_mse = target_month.square().mean()

            target_mean_map = target_month.mean(dim=0)
            pred_mean_map = pred_month.mean(dim=0)
            error_mean_map = error_month.mean(dim=0)

            mask_month = masks[selected]

            model_loss = model.compute_loss(
                prediction=pred_month,
                target=target_month,
                mask=mask_month,
            )

            zero_loss = model.compute_loss(
                prediction=torch.zeros_like(target_month),
                target=target_month,
                mask=mask_month,
            )

            print(f"Month {int(month)}")
            print("  samples:", int(selected.sum()))
            print("  unweighted raw tensor MSE:", float(model_mse))
            print("  weighted loss:", float(model_loss))
            print("  weighted zero loss:", float(zero_loss))
            print(
                "  weighted skill vs zero:",
                float(1.0 - model_loss / zero_loss.clamp_min(1e-8)),
            )
            print("  zero MSE:", float(zero_mse))
            print(
                "  skill vs zero:",
                float(1.0 - model_mse / zero_mse.clamp_min(1e-8)),
            )
            print(
                "  target mean-map abs:",
                float(target_mean_map.abs().mean()),
            )
            print(
                "  prediction mean-map abs:",
                float(pred_mean_map.abs().mean()),
            )
            print(
                "  error mean-map abs:",
                float(error_mean_map.abs().mean()),
            )
            print(
                "  error mean-map max:",
                float(error_mean_map.abs().max()),
            )
            print(
                "prediction std:",
                float(pred_month.std()),
            )
            print(
                "target std:",
                float(target_month.std()),
            )

            valid_month = mask_month.bool()
            valid_count_map_month = valid_month.sum(dim=(0, 1))

            model_mse_map = (
                error_month.square()
                .masked_fill(~valid_month, 0.0)
                .sum(dim=(0, 1))
                / valid_count_map_month.clamp_min(1)
            )

            zero_mse_map = (
                target_month.square()
                .masked_fill(~valid_month, 0.0)
                .sum(dim=(0, 1))
                / valid_count_map_month.clamp_min(1)
            )

            valid_grid_cells_month = valid_count_map_month > 0

            improved_grid_cells = (
                model_mse_map < zero_mse_map
            ) & valid_grid_cells_month

            improved_grid_fraction = (
                improved_grid_cells.sum()
                / valid_grid_cells_month.sum().clamp_min(1)
            )

            print(
                "  improved grid cells:",
                f"{100.0 * float(improved_grid_fraction):.1f}%",
            )

    del model.test_preds
    del model.test_targets
    del model.test_masks
    del model.test_months
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    months = dataset.months[: preds_norm.shape[0]]

    preds = normalize_target.inverse_tensor(preds_norm, months=months)
    preds_ds = convert_to_xarray(preds, dataset, [s.var_an])

    preds_ds = preds_ds.chunk(safe_chunk_spec(preds_ds, dataset.input_ds)).unify_chunks()

    # Reconstruct
    if s.target_mode in ("residual", "residual_realization"):
        fc_base = dataset.input_ds[s.var_fc]
        if s.target_mode == "residual" and "realization" in fc_base.dims:
            fc_base = fc_base.mean("realization")

        fc_base = fc_base.chunk(safe_chunk_spec(fc_base, dataset.input_ds))
        preds_ds = (preds_ds[s.var_an] + fc_base).to_dataset(name=s.var_an)

    elif s.target_mode == "anomaly":
        if an_clim is None:
            raise ValueError("an_clim is required for target_mode='anomaly'.")

        an_clim_for_time = select_clim_for_time(
            an_clim,
            preds_ds.time.values,
            s.clim_period,
        )
        an_clim_for_time = an_clim_for_time.chunk(safe_chunk_spec(an_clim_for_time, dataset.target_ds))

        preds_ds = (preds_ds[s.var_an] + an_clim_for_time[s.var_an]).to_dataset(
            name=s.var_an
        )

    elif s.target_mode in ("anomaly_residual", "anomaly_residual_realization"):
        if an_clim is None:
            raise ValueError("an_clim is required for residual anomaly reconstruction.")


        an_clim_for_time = select_clim_for_time(
            an_clim,
            preds_ds.time.values,
            s.clim_period,
        )

        fc_anom = dataset.input_ds[s.var_fc] # already anomaly

        if s.target_mode == "anomaly_residual" and "realization" in fc_anom.dims:
            fc_anom = fc_anom.mean("realization")

        fc_anom = fc_anom.chunk(safe_chunk_spec(fc_anom, dataset.input_ds))
        an_clim_for_time = an_clim_for_time.chunk(safe_chunk_spec(an_clim_for_time, an_clim))

        corrected = preds_ds[s.var_an] + fc_anom + an_clim_for_time[s.var_an]
        preds_ds = corrected.to_dataset(name=s.var_an)

    save_zarr(
        preds_ds,
        preds_store,
        safe_chunk_spec(preds_ds, dataset.input_ds),
        )

    del preds_norm, preds, preds_ds
    if hasattr(model, "test_preds"):
        del model.test_preds


def _core_train(
    s: Settings,
    exp_name: str,
    region_name: str,
    exp_ratio: tuple[int, int],
    train_dataset: XarrayDataset,
    val_dataset: XarrayDataset | None,
    test_dataset: XarrayDataset,
    x_clim: xr.Dataset | None,
    y_clim: xr.Dataset | None,
    force_retrain: bool,
    force_test: bool,
    dry_run: bool,
    interpolate_analysis: bool,
    device: torch.device,
    accelerator: str,
    leadtime: int | float | str,
    log_monthly: bool = False,

):
    # Create exp dirs
    exp_dir = s.exp_dir / exp_name
    weights_dir = exp_dir / "weights"
    checkpoints_dir = exp_dir / "checkpoints"

    for d in (weights_dir, checkpoints_dir):
        d.mkdir(exist_ok=True, parents=True)

    weights_file = weights_dir / "weights.ckpt"
    last_checkpoint = checkpoints_dir / "last.ckpt"

    train_store, val_store, test_store = experiment_stores(
        s,
        exp_name,
    )

    should_train = force_retrain or not weights_file.exists()

    should_test = (
        force_retrain
        or force_test
        or not train_store.exists()
        or not test_store.exists()
        or (
            val_dataset is not None
            and not val_store.exists()
        )
    )

    if force_retrain:
        print(f"[yellow]Force retrain enabled for leadtime {leadtime}.[/yellow]")

        weights_file.unlink(missing_ok=True)
        last_checkpoint.unlink(missing_ok=True)

    # Normalize
    if s.normalization == "monthly":
        NormClass = MonthlyNormalize
    elif s.normalization == "full":
        NormClass = Normalize
    else:
        raise ValueError(f"normalization={s.normalization} not supported.")

    input_excluded_channels = (
        (-4, -3, -2, -1)
        if s.seasonal_encoding and s.channel_representation!="init_period"
        else None
    )

    normalize_input = NormClass(
        mode=s.normalization_mode,
        exclude_channels=input_excluded_channels,
    ).fit(
        train_dataset,
        dim="x",
    )

    normalize_target = NormClass(
        mode=s.normalization_mode,
        exclude_channels=None,
    ).fit(
        train_dataset,
        dim="y",
    )

    train_dataset.transform_x = normalize_input
    train_dataset.transform_y = normalize_target

    if val_dataset is not None:
        val_dataset.transform_x = normalize_input
        val_dataset.transform_y = normalize_target

    # Loss
    lat_dim = train_dataset.target_ds.earthml.guessed_dims.latitude
    latitudes = torch.as_tensor(
        train_dataset.target_ds[lat_dim].values,
        dtype=torch.float32,
    )

    loss_kwargs = {}

    if s.loss_name == "MaskedMSELoss":
        loss_kwargs = {
            "eps": 1e-8,
        }

    elif s.loss_name in {
        "GeoMSELoss",
        "GeoMaskedMSELoss",
    }:
        loss_kwargs = {
            "latitudes": latitudes,
            "eps": 1e-8,
        }

    elif s.loss_name == "GeoMaskedMSELowFreqLoss":
        target_scale_degrees = s.target_scale_degrees
        grid_spacing = abs(
            float(
                train_dataset.target_ds.latitude.diff("latitude")
                .median()
                .values
            )
        )

        pool_kernel_size = max(
            3,
            round(target_scale_degrees / grid_spacing),
        )

        # Prefer odd windows.
        if pool_kernel_size % 2 == 0:
            pool_kernel_size += 1

        loss_kwargs = {
            "latitudes": latitudes,
            "lambda_low_freq": 0.5,
            "lambda_batch_mean": 2,
            "pool_kernel_size": pool_kernel_size,
            "pool_stride": max(1, pool_kernel_size // 2),
            "eps": 1e-8,
        }

    elif s.loss_name == "VarNormMaskMSELoss":
        loss_kwargs = {
            "variance_type": "spatial", # "channel", "geochannel", "spatial", "temporal", "geotemporal"
            "eps": 1e-6,
            "relative_floor_frac": 1e-3,
            "min_valid_count": 2,
        }

    elif s.loss_name == "HeteroBiasCorrectionLoss":
        loss_kwargs = {
            "lambda_identity": 0.1,
            "bias_scale": 0.5,
            "variance_type": "channel",
            "eps": 1e-6,
        }

    elif s.loss_name == "GaussianNLLFromLogits":
        loss_kwargs = {
            "eps": 1e-6,
        }

    elif s.loss_name == "MSELoss":
        loss_kwargs = {}

    else:
        raise ValueError(f"Unsupported loss configuration: {s.loss_name!r}")

    base_loss_params = {
        "loss": loss_kwargs,
        "net": {},
    }

    # Set input channels
    n_channels = train_dataset.x.shape[1]
    n_classes = train_dataset.y.shape[1] # TODO not sure this works if realization_as_channel is True

    # Initialize model args
    longitude_padding = "circular" if region_name=="World" else "replicate"
    common_net_kwargs = {
        "learning_rate": s.init_learning_rate,
        "weight_decay": s.weight_decay,
        "loss": s.loss_name,
        "loss_params": base_loss_params,
        "norm": s.training_norm,
        "supervised": True,
        "n_channels": n_channels,
        "n_classes": n_classes,
        "longitude_padding": longitude_padding,
        "zero_init_output": True if s.target_mode in (
            "residual",
            "residual_realization",
            "anomaly_residual",
            "anomaly_residual_realization",
        ) else False,
    }

    net_kwargs = {
        **common_net_kwargs,
        **s.extra_net_kwargs,
    }

    # Init model
    model = build_net(
        name=s.net_name,
        **net_kwargs,
    ).to(device)

    # Create train datamodule and split train dataset into train and validation based on self.config.train_percent
    train_datamodule = SplitDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_fraction=s.train_fraction,
        batch_size=s.batch_size,
        seed=s.seed,
        num_workers=s.torch_workers,
        split_strategy=s.split_strategy,
        shuffle_train=s.shuffle_train_batch,
        pin_memory=None, # True if CUDA available
        persistent_workers=None, # True if num_workers > 0
        drop_last_train=False,
    )

    if s.split_strategy == "explicit":
        train_idx, val_idx = None, None
    else:
        train_datamodule.setup("fit")
        train_idx = train_datamodule.train_indices
        val_idx = train_datamodule.val_indices

    # Normalize (use train-fitted normalizers)
    test_dataset.transform_x = normalize_input
    test_dataset.transform_y = normalize_target
    # test_normalize_input  = Normalize().fit(test_dataset, filepath=None, dim='x')
    # test_normalize_target = Normalize().fit(test_dataset, filepath=None, dim='y')
    # test_dataset.transform_x = test_normalize_input
    # test_dataset.transform_y = test_normalize_target

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        pin_memory=(accelerator == "gpu"),
        persistent_workers=False,
    )

    val_test_dataloader = None
    if val_dataset is not None:
        val_test_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            pin_memory=(accelerator == "gpu"),
            persistent_workers=False,
        )

    train_test_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        pin_memory=(accelerator == "gpu"),
        persistent_workers=False,
    )

    print_training_recap(
        settings=s,
        exp_ratio=exp_ratio,
        region_name=region_name,
        accelerator=accelerator,
        device=device,
        leadtime=leadtime,
        normalization_name=type(normalize_input).__name__,
        n_channels=n_channels,
        n_classes=n_classes,
        longitude_padding=longitude_padding,
        dry_run=dry_run,
        force_retrain=force_retrain,
        force_test=force_test,
        interpolate_analysis=interpolate_analysis,
        train_input_shape=tuple(train_dataset.x.shape),
        train_target_shape=tuple(train_dataset.y.shape),
        val_input_shape=tuple(val_dataset.x.shape) if val_dataset is not None else None,
        val_target_shape=tuple(val_dataset.y.shape) if val_dataset is not None else None,
        test_input_shape=tuple(test_dataset.x.shape),
        test_target_shape=tuple(test_dataset.y.shape),
        train_idx=train_idx,
        val_idx=val_idx,
        exp_name=exp_name,
        weights_dir=weights_dir,
        checkpoints_dir=checkpoints_dir,
    )

    # Tensorboard
    tb_logger = TensorBoardLogger(
        save_dir=s.exp_dir / "tensorboard",
        name=exp_name,
        version="",
        default_hp_metric=False,
        # log_graph=True,
    )

    tb_logger.log_hyperparams(
        {
            "leadtime": leadtime,
            "network": s.net_name,
            "loss": s.loss_name,
            "learning_rate": s.init_learning_rate,
            "weight_decay": s.weight_decay,
            "batch_size": s.batch_size,
            "effective_batch_size": (
                s.batch_size * s.accumulate_grad_batches
            ),
            "normalization": s.normalization,
            "normalization_mode": s.normalization_mode,
            "seasonal_encoding": s.seasonal_encoding,
            "target_mode": s.target_mode,
        }
    )

    train_trainer = L.Trainer(
        max_epochs=s.max_epochs,
        accelerator=accelerator,
        devices=1,
        precision=s.trainer_precision,
        # gradient_clip_val=1.0,  # Recommended starting value (e.g., 0.5, 1.0, 5.0)
        # gradient_clip_algorithm="norm",  # "norm" for clipping by norm, "value" for clipping by value
        log_every_n_steps=1,
        logger=tb_logger,
        accumulate_grad_batches=s.accumulate_grad_batches,
        # callbacks=[],
        # enable_checkpointing=False,
        callbacks=init_callbacks(
            weights_folder_path=weights_dir,
            ckpt_folder_path=checkpoints_dir,
            patience=s.early_stopping_patience,
        ),
        # deterministic=True
        # num_sanity_val_steps=0,
    )

    resume_checkpoint: str | None = None

    if should_train and not force_retrain and last_checkpoint.exists():
        resume_checkpoint = str(last_checkpoint)

    if should_train:
        train_trainer.fit(
            model,
            datamodule=train_datamodule,
            ckpt_path=resume_checkpoint,
        )
    else:
        print(
            f"[green]Skipping training for leadtime {leadtime}: "
            f"best weights already exist.[/green]"
        )

    if not weights_file.exists():
        raise FileNotFoundError(
            f"Best model checkpoint was not created: {weights_file}"
        )

    model = type(model).load_from_checkpoint(
        weights_file,
        **net_kwargs,
    ).to(device)

    test_trainer = L.Trainer(
        accelerator=accelerator,
        devices=1,
        precision=s.trainer_precision,
        # gradient_clip_val=1.0,  # Recommended starting value (e.g., 0.5, 1.0, 5.0)
        # gradient_clip_algorithm="norm",  # "norm" for clipping by norm, "value" for clipping by value
        # deterministic=True
    )

    # Predict
    if should_test:
        _test(
            test_trainer=test_trainer,
            s=s,
            model=model,
            dataset=test_dataset,
            normalize_target=normalize_target,
            dataloader=test_dataloader,
            preds_store=test_store,
            an_clim=y_clim,
            log_monthly=log_monthly,
        )

        if val_test_dataloader is not None and val_dataset is not None:
            _test(
                test_trainer=test_trainer,
                s=s,
                model=model,
                dataset=val_dataset,
                normalize_target=normalize_target,
                dataloader=val_test_dataloader,
                preds_store=val_store,
                an_clim=y_clim,
                log_monthly=log_monthly,
            )

        _test(
            test_trainer=test_trainer,
            s=s,
            model=model,
            dataset=train_dataset,
            normalize_target=normalize_target,
            dataloader=train_test_dataloader,
            preds_store=train_store,
            an_clim=y_clim,
            log_monthly=log_monthly,
        )
    else:
        print(
            f"[green]Skipping testing for leadtime {leadtime}: "
            f"saved preds already exist.[/green]"
        )

    # Trainer clean-up
    train_trainer.strategy.teardown()
    test_trainer.strategy.teardown()

    for ds in (
        train_dataset.input_ds,
        train_dataset.target_ds,
        val_dataset.input_ds if val_dataset is not None else None,
        val_dataset.target_ds if val_dataset is not None else None,
        test_dataset.input_ds,
        test_dataset.target_ds,
    ):
        if ds is not None:
            try:
                ds.close()
            except Exception:
                pass

    try:
        tb_logger.finalize("success")
    except Exception:
        pass

    del train_trainer, test_trainer, model
    del train_datamodule
    del train_test_dataloader, val_test_dataloader, test_dataloader
    del train_dataset, val_dataset, test_dataset
    del normalize_input, normalize_target
    del latitudes, loss_kwargs, base_loss_params, net_kwargs
    del tb_logger

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    if (
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        torch.mps.empty_cache()

    return train_store, val_store, test_store


def make_regional_boxes(
    full_region: dict[str, tuple[float, float]] | None,
    lat_size: float = 30.0,
    lon_size: float = 30.0,
) -> list[dict[str, dict[str, tuple[float, float]]]]:
    if full_region is None:
        lat_min, lat_max = -90.0, 90.0
        lon_min, lon_max = -180.0, 180.0
    else:
        lat_min, lat_max = sorted(map(float, full_region["lat"]))
        lon_min, lon_max = sorted(map(float, full_region["lon"]))

    lat_edges = np.append(
        np.arange(lat_min, lat_max, lat_size),
        lat_max,
    )
    lon_edges = np.append(
        np.arange(lon_min, lon_max, lon_size),
        lon_max,
    )

    regions = []

    for lat0, lat1 in zip(
        lat_edges[:-1],
        lat_edges[1:],
        strict=True,
    ):
        for lon0, lon1 in zip(
            lon_edges[:-1],
            lon_edges[1:],
            strict=True,
        ):
            lat0 = float(lat0)
            lat1 = float(lat1)
            lon0 = float(lon0)
            lon1 = float(lon1)

            name = (
                f"lat_{lat0:+04.0f}_{lat1:+04.0f}_"
                f"lon_{lon0:+04.0f}_{lon1:+04.0f}"
            )

            regions.append(
                {
                    name: {
                        # Descending latitude.
                        "lat": (lat1, lat0),
                        "lon": (lon0, lon1),
                    }
                }
            )

    return regions


def experiment_stores(
    s: Settings,
    exp_name: str,
) -> tuple[Path, Path, Path]:
    exp_dir = s.exp_dir / exp_name

    return (
        exp_dir / "train_preds.zarr",
        exp_dir / "val_preds.zarr",
        exp_dir / "test_preds.zarr",
    )

def experiment_is_complete(
    s: Settings,
    exp_name: str,
    *,
    explicit_split: bool,
) -> bool:
    exp_dir = s.exp_dir / exp_name
    weights_file = exp_dir / "weights" / "weights.ckpt"

    train_store, val_store, test_store = experiment_stores(
        s,
        exp_name,
    )

    predictions_complete = (
        train_store.exists()
        and test_store.exists()
        and (
            not explicit_split
            or val_store.exists()
        )
    )

    return weights_file.exists() and predictions_complete


def train(
    var: str,
    region_name: str,
    region_location: dict[str, tuple[int | float, int | float]] | None,
) -> None:
    if var in {"mlotst", "ssh", "sss", "t20d"}:
        var_type_fc = "ocean"
        reanalysis_model = "oras5"
    else:
        var_type_fc = "atmo"
        reanalysis_model = "era5"

    dry_run = False
    force_retrain = False
    force_test = False
    interpolate_analysis = True
    log_monthly = False

    accelerator, device = resolve_accelerator_and_device()

    if accelerator == "gpu":
        torch.set_float32_matmul_precision("high")

    s = Settings(
        root_dir=Path("/Users/jacopodallaglio/ML/training/seasonal"),
        data_root_dir=None,
        exp_root_dir=None,
        plot_root_dir=None,
        extra_suffix_folder="",
        lead_period_offset=-1,
        var_file_fc=var,
        var_file_an=var,
        var_fc=var,
        var_an=var,
        model_fc=f"sps4_{var_type_fc}",
        model_an=reanalysis_model,
        leadtime_unit=LeadtimeUnit.MONTHS,
        # leadtimes=[3, 4, 5],
        leadtimes=[1, 2, 3, 4, 5, 6],
        separate_training_by_init_period=None,
        regional_training=False,
        regional_training_lat_size=60.0,
        regional_training_lon_size=30.0,
        region_name=region_name,
        region=region_location,
        train_start="1993-01-01",
        train_end="2014-12-01",
        val_start="2015-01-01",
        val_end="2020-12-01",
        test_start="2021-01-01",
        test_end="2022-12-01",
        target_mode="analysis",
        clim_period=ClimPeriod.MONTH,
        seed=42,
        channel_representation="variable",
        output_realizations="deterministic",
        split_strategy="explicit",
        shuffle_train_batch=True,
        normalization="full",
        normalization_mode="channel",
        seasonal_encoding=True, # automatically set to False if channel_representation="init_period"
        ensemble_encoding=True,
        net_name="SmaAt_UNet",
        loss_name="GeoMaskedMSELoss",
        target_scale_degrees=30.0, # only for GeoMaskedMSELowFreqLoss
        init_learning_rate=3e-4,
        weight_decay=1e-4,
        batch_size=32,
        max_epochs=100,
        target_realization_avg=False,
        fill_nan_value=0.0,
        torch_mask="target",
        training_norm="BatchNorm2d", # ignored for convnext (uses only LayerNorm)
        smaatunet_kwargs=dict(
            reduction_ratio=8,
            depth=3,
            kernels_per_layer=1,
            base_channels=16,
        ),
        convnext_kwargs=dict(
            depths=(2, 2, 4, 2),
            dims=(32, 64, 128, 256),
            drop_path_rate=0.05,
            layer_scale_init_value=1e-6,
        ),
        train_fraction=0.85,
        accumulate_grad_batches=1,
        early_stopping_patience=20,
        torch_workers=4,
        trainer_precision="bf16-mixed" if accelerator == "gpu" else "32-true",
    )
    s.make_dirs()
    s.save_config()

    if dry_run:
        return

    L.seed_everything(s.seed)

    train_pred_paths: list[PredictionRecord] = []
    val_pred_paths: list[PredictionRecord] = []
    test_pred_paths: list[PredictionRecord] = []

    if s.regional_training:
        region_boxes = make_regional_boxes(
            full_region=s.region,
            lat_size=s.regional_training_lat_size,
            lon_size=s.regional_training_lon_size,
        )
    else:
        region_boxes = [
            {
                s.region_name: s.region,
            }
        ]

    if s.separate_training_by_init_period is None:
        total_exps = len(region_boxes) * len(s.leadtimes)
    else:
        num_periods = 12 if s.separate_training_by_init_period==ClimPeriod.MONTH else 1
        total_exps = (
            len(region_boxes)
            * len(s.leadtimes)
            * num_periods
        )

    current_exp = 0

    for regional_entry in region_boxes:
        if len(regional_entry) != 1:
            raise ValueError(
                "Each regional entry must contain exactly one region"
            )

        regional_name, regional_location = next(
            iter(regional_entry.items())
        )

        for lt in s.leadtimes:
            explicit_split = s.split_strategy == "explicit"
            train_end = s.train_end if explicit_split else s.val_end

            def _make_datasets() -> LeadtimeDatasets:
                return make_train_test_datasets_for_leadtime(
                    forecast_ds_path=(
                        s.input_dir / f"{s.model_fc}_{s.var_fc}.zarr"
                    ),
                    analysis_ds_path=(
                        s.input_dir / f"{s.model_an}_{s.var_an}.zarr"
                    ),
                    leadtime=lt,
                    leadtime_unit=LeadtimeUnit(s.leadtime_unit),
                    train_start=s.train_start,
                    train_end=train_end,
                    val_start=s.val_start if explicit_split else None,
                    val_end=s.val_end if explicit_split else None,
                    test_start=s.test_start,
                    test_end=s.test_end,
                    target_mode=s.target_mode,
                    clim_period=s.clim_period,
                    forecast_vars=[s.var_fc],
                    analysis_vars=[s.var_an],
                    region=regional_location,
                    dataset_kwargs={
                        "target_realization_avg": s.target_realization_avg,
                        "channel_representation": s.channel_representation,
                        "init_period_dim": s.init_period_dim,
                        "output_realizations": s.output_realizations,
                        "torch_mask": s.torch_mask,
                        "fill_nan_value": s.fill_nan_value,
                    },
                    seasonal_encoding=(
                        s.seasonal_encoding
                        and s.channel_representation != "init_period"
                    ),
                    ensemble_encoding=s.ensemble_encoding,
                    interpolate_analysis=interpolate_analysis,
                    materialize=False,
                    separate_training_by_init_period=s.separate_training_by_init_period,
                )

            if s.separate_training_by_init_period is None:
                exp_name = f"exp_{lt}_{s.leadtime_unit.value}"

                if s.regional_training:
                    exp_name += f"_{regional_name}"

                current_exp += 1

                if (
                    not force_retrain
                    and not force_test
                    and experiment_is_complete(
                        s,
                        exp_name,
                        explicit_split=explicit_split,
                    )
                ):
                    train_store, val_store, test_store = experiment_stores(
                        s,
                        exp_name,
                    )

                    print(
                        f"[green]Skipping experiment "
                        f"{current_exp}/{total_exps}: {exp_name} "
                        f"is already complete.[/green]"
                    )

                    train_pred_paths.append(
                        (int(lt), None, regional_name, train_store)
                    )

                    if explicit_split:
                        val_pred_paths.append(
                            (int(lt), None, regional_name, val_store)
                        )

                    test_pred_paths.append(
                        (int(lt), None, regional_name, test_store)
                    )

                    continue

                dataset_d = _make_datasets()

                train_store, val_store, test_store = _core_train(
                    s=s,
                    exp_name=exp_name,
                    region_name=regional_name,
                    exp_ratio=(current_exp, total_exps),
                    train_dataset=dataset_d["train"],
                    val_dataset=dataset_d["val"],
                    test_dataset=dataset_d["test"],
                    x_clim=dataset_d["x_clim"],
                    y_clim=dataset_d["y_clim"],
                    force_retrain=force_retrain,
                    force_test=force_test,
                    dry_run=dry_run,
                    interpolate_analysis=interpolate_analysis,
                    device=device,
                    accelerator=accelerator,
                    leadtime=lt,
                    log_monthly=log_monthly,
                )

                train_pred_paths.append(
                    (int(lt), None, regional_name, train_store)
                )

                if dataset_d["val"] is not None:
                    val_pred_paths.append(
                        (int(lt), None, regional_name, val_store)
                    )

                test_pred_paths.append(
                    (int(lt), None, regional_name, test_store)
                )

            else:
                if s.separate_training_by_init_period != ClimPeriod.MONTH:
                    raise NotImplementedError(
                        "Currently only monthly separate training is supported."
                    )

                init_periods = [str(month) for month in range(1, 13)]
                pending_periods: set[str] = set()

                # First check every experiment without generating datasets.
                for init_period in init_periods:
                    exp_name = (
                        f"exp_{lt}_{s.leadtime_unit.value}_"
                        f"{s.separate_training_by_init_period.value}_"
                        f"{init_period}"
                    )

                    if s.regional_training:
                        exp_name += f"_{regional_name}"

                    current_exp += 1

                    complete = (
                        not force_retrain
                        and not force_test
                        and experiment_is_complete(
                            s,
                            exp_name,
                            explicit_split=explicit_split,
                        )
                    )

                    if complete:
                        train_store, val_store, test_store = experiment_stores(
                            s,
                            exp_name,
                        )

                        print(
                            f"[green]Skipping experiment "
                            f"{current_exp}/{total_exps}: {exp_name} "
                            f"is already complete.[/green]"
                        )

                        train_pred_paths.append(
                            (
                                int(lt),
                                init_period,
                                regional_name,
                                train_store,
                            )
                        )

                        if explicit_split:
                            val_pred_paths.append(
                                (
                                    int(lt),
                                    init_period,
                                    regional_name,
                                    val_store,
                                )
                            )

                        test_pred_paths.append(
                            (
                                int(lt),
                                init_period,
                                regional_name,
                                test_store,
                            )
                        )
                    else:
                        pending_periods.add(init_period)

                # Every month already exists: avoid all dataset work.
                if not pending_periods:
                    continue

                dataset_d = _make_datasets()

                train_by_period = dataset_d["train"]
                val_by_period = dataset_d["val"]
                test_by_period = dataset_d["test"]

                if not isinstance(train_by_period, dict):
                    raise TypeError(
                        "Expected train datasets divided by initialization period"
                    )

                if not isinstance(test_by_period, dict):
                    raise TypeError(
                        "Expected test datasets divided by initialization period"
                    )

                if (
                    val_by_period is not None
                    and not isinstance(val_by_period, dict)
                ):
                    raise TypeError(
                        "Expected validation datasets divided by "
                        "initialization period"
                    )

                available_periods = set(train_by_period)
                missing_periods = pending_periods - available_periods

                if missing_periods:
                    raise ValueError(
                        "Pending initialization periods are absent from the "
                        f"training data: {sorted(missing_periods, key=int)}"
                    )

                # current_exp has already advanced during the pre-check. Recover
                # each period's stable position in the global experiment sequence.
                first_period_exp = current_exp - len(init_periods) + 1

                for init_period in sorted(pending_periods, key=int):
                    if init_period not in test_by_period:
                        raise ValueError(
                            f"Initialization period {init_period!r} exists in "
                            "training but not in test data"
                        )

                    val_dataset = None

                    if val_by_period is not None:
                        if init_period not in val_by_period:
                            raise ValueError(
                                f"Initialization period {init_period!r} exists in "
                                "training but not in validation data"
                            )

                        val_dataset = val_by_period[init_period]

                    exp_name = (
                        f"exp_{lt}_{s.leadtime_unit.value}_"
                        f"{s.separate_training_by_init_period.value}_"
                        f"{init_period}"
                    )

                    if s.regional_training:
                        exp_name += f"_{regional_name}"

                    period_exp_number = (
                        first_period_exp + int(init_period) - 1
                    )

                    train_store, val_store, test_store = _core_train(
                        s=s,
                        exp_name=exp_name,
                        region_name=regional_name,
                        exp_ratio=(period_exp_number, total_exps),
                        train_dataset=train_by_period[init_period],
                        val_dataset=val_dataset,
                        test_dataset=test_by_period[init_period],
                        x_clim=dataset_d["x_clim"],
                        y_clim=dataset_d["y_clim"],
                        force_retrain=force_retrain,
                        force_test=force_test,
                        dry_run=dry_run,
                        interpolate_analysis=interpolate_analysis,
                        device=device,
                        accelerator=accelerator,
                        leadtime=lt,
                        log_monthly=log_monthly,
                    )

                    train_pred_paths.append(
                        (
                            int(lt),
                            init_period,
                            regional_name,
                            train_store,
                        )
                    )

                    if val_dataset is not None:
                        val_pred_paths.append(
                            (
                                int(lt),
                                init_period,
                                regional_name,
                                val_store,
                            )
                        )

                    test_pred_paths.append(
                        (
                            int(lt),
                            init_period,
                            regional_name,
                            test_store,
                        )
                    )

            # Clean-up after all leadtimes
            for clim_ds in (
                dataset_d["x_clim"],
                dataset_d["y_clim"],
            ):
                if clim_ds is not None:
                    try:
                        clim_ds.close()
                    except Exception:
                        pass

            del dataset_d

    # Combine leadtimes (always), regions and init periods if necessary
    _combine_predictions(
        s=s,
        data_type="train",
        pred_records=train_pred_paths,
    )

    if val_pred_paths:
        _combine_predictions(
            s=s,
            data_type="val",
            pred_records=val_pred_paths,
        )

    _combine_predictions(
        s=s,
        data_type="test",
        pred_records=test_pred_paths,
    )


def main():
    regions = {
        # "ConUS": {
        #     "lon": (-130, -60),
        #     "lat": (50, 25),
        # },
        # "Europe": {
        #     "lon": (-30, 60),
        #     "lat": (80, 30),
        # },
        # "Pacific": {
        #     "lon": (-200, -120),
        #     "lat": (30, -30),
        # },
        "World": None,
    }

    for region_name, region_location in regions.items():
        for var in [
            # Atmo
            "mslp",
            # "t2m",
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
        ]:
            print(f"Training for {var}")
            train(
                var=var,
                region_name=region_name,
                region_location=region_location,
            )

if __name__ == "__main__":
    main()
