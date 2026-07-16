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
    train: XarrayDataset
    val: XarrayDataset | None
    test: XarrayDataset
    x_clim: xr.Dataset | None
    y_clim: xr.Dataset | None

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
    prefix: str = "season",
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

    phase = (valid_times.month.to_numpy() - 1) / 12.0
    angle = 2.0 * np.pi * phase

    seasonal = xr.Dataset(
        {
            f"{prefix}_sin": (
                (time_dim,),
                np.sin(angle).astype(np.float32),
            ),
            f"{prefix}_cos": (
                (time_dim,),
                np.cos(angle).astype(np.float32),
            ),
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
    interpolate: bool = True,
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

    if interpolate:
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
    interpolate: bool = True,
    materialize: bool = False,
) -> tuple[xr.Dataset, xr.Dataset]:
    fc_period_ds, an_period_ds = extract_period_from_ds(
        fc_ds=fc_ds,
        an_ds=an_ds,
        start=start,
        end=end,
        leadtime=leadtime,
        leadtime_unit=leadtime_unit,
        interpolate=interpolate,
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
        if seasonal_encoding:
            fc_period_ds = seasonal_cycle_encoder(
                fc_period_ds,
                leadtime=leadtime,
                leadtime_unit=leadtime_unit,
            )
        return fc_period_ds, an_period_ds

    if target_mode == "residual":
        fc_base = (
            fc_period_ds.mean("realization")
            if "realization" in fc_period_ds.dims
            else fc_period_ds
        )
        if seasonal_encoding:
            fc_period_ds = seasonal_cycle_encoder(
                fc_period_ds,
                leadtime=leadtime,
                leadtime_unit=leadtime_unit,
            )
        return fc_period_ds, an_period_ds - fc_base

    if target_mode == "residual_realization":
        if seasonal_encoding:
            fc_period_ds = seasonal_cycle_encoder(
                fc_period_ds,
                leadtime=leadtime,
                leadtime_unit=leadtime_unit,
            )
        return fc_period_ds, an_period_ds - fc_period_ds

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
        if seasonal_encoding:
            fc_anom = seasonal_cycle_encoder(
                fc_anom,
                leadtime=leadtime,
                leadtime_unit=leadtime_unit,
            )
        return fc_anom, an_anom

    if target_mode == "anomaly_residual":
        fc_anom_base = (
            fc_anom.mean("realization")
            if "realization" in fc_anom.dims
            else fc_anom
        )
        if seasonal_encoding:
            fc_anom = seasonal_cycle_encoder(
                fc_anom,
                leadtime=leadtime,
                leadtime_unit=leadtime_unit,
            )
        return fc_anom, an_anom - fc_anom_base

    if target_mode == "anomaly_residual_realization":
        if seasonal_encoding:
            fc_anom = seasonal_cycle_encoder(
                fc_anom,
                leadtime=leadtime,
                leadtime_unit=leadtime_unit,
            )
        return fc_anom, an_anom - fc_anom

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
    interpolate: bool = True,
    materialize: bool = False,
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
            interpolate=interpolate,
        )

        fc_time_dim = fc_ds.earthml.guessed_dims.time
        an_time_dim = an_ds.earthml.guessed_dims.time

        fc_clim = calculate_climatology(fc_clim_ds, fc_time_dim, clim_period)
        an_clim = calculate_climatology(an_clim_ds, an_time_dim, clim_period)

    else:
        fc_clim = None
        an_clim = None

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
        interpolate=interpolate,
        materialize=materialize,
    )

    x_train, y_train = drop_zero_valid_target_samples(
        x_train,
        y_train,
        label="train",
    )

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
            interpolate=interpolate,
            materialize=materialize,
        )

        x_val, y_val = drop_zero_valid_target_samples(
            x_val,
            y_val,
            label="validation",
        )

        val_ds = XarrayDataset(x_val, y_val, **dataset_kwargs)

    else:
        val_ds = None

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
        interpolate=interpolate,
        materialize=materialize,
    )

    x_test, y_test = drop_zero_valid_target_samples(
        x_test,
        y_test,
        label="test",
    )

    train_ds = XarrayDataset(x_train, y_train, **dataset_kwargs)
    test_ds = XarrayDataset(x_test, y_test, **dataset_kwargs)

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
    accelerator: str,
    device,
    leadtime: int | float | str,
    normalization_name: str,
    n_channels: int,
    n_classes: int,
    force_retrain: bool,
    force_test: bool,
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
        "experiment.name": exp_name or s.output_name,
        "experiment.leadtime": f"{leadtime} {s.leadtime_unit.value}",
        "experiment.force_retrain": force_retrain,
        "experiment.force_test": force_test,
        "experiment.torch_workers": s.torch_workers,

        "data.forecast": f"{s.model_fc}/{s.var_fc}",
        "data.analysis": f"{s.model_an}/{s.var_an}",
        "data.region": f"{s.region_name} {s.region}",
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
        "split.train_fraction": s.train_fraction,
        "split.train_samples": len(train_idx) if train_idx is not None else None,
        "split.val_samples": len(val_idx) if val_idx is not None else None,
        "split.train_idx": f"{train_idx[0]} → {train_idx[-1]}" if train_idx is not None and len(train_idx) else None,
        "split.val_idx": f"{val_idx[0]} → {val_idx[-1]}" if val_idx is not None and len(val_idx) else None,

        "model.network": s.net_name,
        "model.loss": s.loss_name,
        "model.n_channels": n_channels,
        "model.n_classes": n_classes,
        "model.realization_as_channel": s.realization_as_channel,
        "model.output_realizations": s.output_realizations,
        "model.norm_layer": s.training_norm,
        "model.depth": s.depth,
        "model.base_channels": s.base_channels,
        "model.kernels_per_layer": s.kernels_per_layer,
        "model.reduction_ratio": s.reduction_ratio,

        "training.normalization": f"{normalization_name}(x), {normalization_name}(y)",
        "training.normalization_mode": s.normalization_mode,
        "training.seasonal_encoding": s.seasonal_encoding,
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

    console.print(Table(flat_recap, title="Training recap").table)


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
    force_test = True
    interpolate = True

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
        leadtimes=[1, 2, 3, 4, 5, 6],
        region_name=region_name,
        region=region_location,
        train_start="1993-01-01",
        train_end="2016-12-01",
        val_start="2017-01-01",
        val_end="2020-12-01",
        test_start="2021-01-01",
        test_end="2022-12-01",
        target_mode="analysis",
        clim_period=ClimPeriod.MONTH,
        seed=42,
        realization_as_channel=False,
        output_realizations="deterministic",
        split_strategy="time",
        normalization="full",
        normalization_mode="channel",
        seasonal_encoding=False,
        net_name="SmaAt_UNet",
        loss_name="MSELoss",
        init_learning_rate=3e-4,
        weight_decay=1e-4,
        batch_size=8,
        max_epochs=50,
        target_realization_avg=False,
        fill_nan_value=0.0,
        torch_mask="target",
        training_norm="GroupNorm",
        reduction_ratio=16,
        depth=5,
        kernels_per_layer=1,
        base_channels=32,
        depth=5,
        train_fraction=0.85,
        accumulate_grad_batches=4,
        early_stopping_patience=10,
        torch_workers=4,
        trainer_precision="bf16-mixed" if accelerator == "gpu" else "32-true",
    )
    s.make_dirs()
    s.save_config()

    if dry_run:
        return

    L.seed_everything(s.seed)

    train_pred_paths: list[tuple[int, Path]] = []
    val_pred_paths: list[tuple[int, Path]] = []
    test_pred_paths: list[tuple[int, Path]] = []
    for lt in s.leadtimes:
        exp_name = f"exp_{lt}_{s.leadtime_unit.value}"

        # Create exp dirs
        exp_dir = s.exp_dir / exp_name
        weights_dir = exp_dir / "weights"
        checkpoints_dir = exp_dir / "checkpoints"

        for d in (weights_dir, checkpoints_dir):
            d.mkdir(exist_ok=True, parents=True)

        weights_file = weights_dir / "weights.ckpt"
        last_checkpoint = checkpoints_dir / "last.ckpt"

        train_store = exp_dir / "train_preds.zarr"
        val_store = exp_dir / "val_preds.zarr"
        test_store = exp_dir / "test_preds.zarr"

        explicit_split = s.split_strategy == "explicit"

        training_complete = weights_file.exists()
        testing_complete = (
            train_store.exists()
            and test_store.exists()
            and (not explicit_split or val_store.exists())
        )

        should_train = force_retrain or not training_complete
        should_test = force_retrain or force_test or not testing_complete

        if force_retrain:
            print(f"[yellow]Force retrain enabled for leadtime {lt}.[/yellow]")

            weights_file.unlink(missing_ok=True)
            last_checkpoint.unlink(missing_ok=True)

        # Explicit:
        #   train dataset = training period only
        # Percentage:
        #   source dataset = full pre-test period, later split by the data module
        train_end = s.train_end if explicit_split else s.val_end

        dataset_d = make_train_test_datasets_for_leadtime(
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
            region=s.region,
            dataset_kwargs={
                "target_realization_avg": s.target_realization_avg,
                "realization_as_channel": s.realization_as_channel,
                "output_realizations": s.output_realizations,
                "torch_mask": s.torch_mask,
                "fill_nan_value": s.fill_nan_value,
            },
            seasonal_encoding=s.seasonal_encoding,
            interpolate=interpolate,
            materialize=False,
        )

        train_dataset = dataset_d["train"]
        val_dataset = dataset_d["val"]
        test_dataset = dataset_d["test"]

        # Normalize
        if s.normalization == "monthly":
            NormClass = MonthlyNormalize
        elif s.normalization == "full":
            NormClass = Normalize
        else:
            raise ValueError(f"normalization={s.normalization} not supported.")

        input_excluded_channels = (
            (-2, -1)
            if s.seasonal_encoding
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
        net_kwargs = dict(
            learning_rate=s.init_learning_rate,
            weight_decay=s.weight_decay,
            loss=s.loss_name,
            loss_params=base_loss_params,
            norm=s.training_norm,
            supervised=True,
            n_channels=n_channels,
            n_classes=n_classes,
            depth=s.depth,
            reduction_ratio=s.reduction_ratio,
            kernels_per_layer=s.kernels_per_layer,
            base_channels=s.base_channels,
        )

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
            split_strategy=s.split_strategy,
            batch_size=s.batch_size,
            seed=s.seed,
            num_workers=s.torch_workers,
        )

        if s.split_strategy == "explicit":
            train_idx, val_idx = None, None
        else:
            train_datamodule.setup("fit")
            train_idx = train_datamodule.train_indices
            val_idx = train_datamodule.val_indices

        # Test dataloader
        test_dataset = dataset_d["test"]

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
            num_workers=s.torch_workers,
            shuffle=False,
            pin_memory = (accelerator == "gpu"),
            persistent_workers=s.torch_workers > 0,
        )

        val_test_dataloader = None
        if val_dataset is not None:
            val_test_dataloader = DataLoader(
                val_dataset,
                batch_size=1,
                num_workers=s.torch_workers,
                shuffle=False,
                pin_memory = (accelerator == "gpu"),
                persistent_workers=s.torch_workers > 0,
            )

        train_test_dataloader = DataLoader(
            train_dataset,
            batch_size=1,
            num_workers=s.torch_workers,
            shuffle=False,
            pin_memory = (accelerator == "gpu"),
            persistent_workers=s.torch_workers > 0,
        )

        print_training_recap(
            settings=s,
            accelerator=accelerator,
            device=device,
            leadtime=lt,
            normalization_name=type(normalize_input).__name__,
            n_channels=n_channels,
            n_classes=n_classes,
            force_retrain=force_retrain,
            force_test=force_test,
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
            name=f"lead_{lt}",
            version="",
            default_hp_metric=False,
            # log_graph=True,
        )

        tb_logger.log_hyperparams(
            {
                "leadtime": lt,
                "network": s.net_name,
                "loss": s.loss_name,
                "learning_rate": s.init_learning_rate,
                "weight_decay": s.weight_decay,
                "batch_size": s.batch_size,
                "effective_batch_size": (
                    s.batch_size * s.accumulate_grad_batches
                ),
                "depth": s.depth,
                "base_channels": s.base_channels,
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
                f"[green]Skipping training for leadtime {lt}: "
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

        def _test(
            dataset: XarrayDataset,
            dataloader: DataLoader,
            preds_store: Path,
            an_clim: xr.Dataset | None = None,
        ):
            test_trainer.test(model, dataloaders=dataloader)

            preds_norm = model.test_preds.detach().float().cpu()
            del model.test_preds
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

        # Predict
        if should_test:
            _test(
                test_dataset,
                test_dataloader,
                test_store,
                dataset_d["y_clim"],
            )
            test_pred_paths.append((int(lt), test_store))

            if val_test_dataloader is not None and val_dataset is not None:
                _test(
                    val_dataset,
                    val_test_dataloader,
                    val_store,
                    dataset_d["y_clim"],
                )
                val_pred_paths.append((int(lt), val_store))

            _test(
                train_dataset,
                train_test_dataloader,
                train_store,
                dataset_d["y_clim"],
            )
            train_pred_paths.append((int(lt), train_store))
        else:
            print(
                f"[green]Skipping testing for leadtime {lt}: "
                f"saved preds already exist.[/green]"
            )
            train_pred_paths.append((int(lt), train_store))
            if val_dataset is not None:
                val_pred_paths.append((int(lt), val_store))
            test_pred_paths.append((int(lt), test_store))

        # Trainer clean-up
        train_trainer.strategy.teardown()
        test_trainer.strategy.teardown()

        del model, train_trainer, test_trainer
        del train_datamodule, train_test_dataloader, val_test_dataloader, test_dataloader
        del train_dataset, val_dataset, test_dataset, dataset_d

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Combine all leadtimes
    def _combine_leadtimes(
        data_type: Literal["train", "val", "test"],
        pred_paths: list[tuple[int, Path]],
    ):
        preds_ds_list = [
            add_or_set_leadtime(open_zarr(path), lt)
            for lt, path in pred_paths
        ]

        combined_preds = xr.concat(
            preds_ds_list,
            dim="leadtime",
            coords="minimal",
            compat="override",
            join="outer",
        )

        combined_preds = combined_preds.chunk(safe_chunk_spec(combined_preds))

        combined_preds["leadtime"].attrs.update({
            "long_name": "forecast lead time",
            "units": s.leadtime_unit.value,
        })

        combined_preds["time"].attrs.update({
            "long_name": "forecast initialization time",
            "standard_name": "forecast_reference_time",
        })

        save_zarr(
            combined_preds,
            s.output_dir / f"{data_type}_corrected.zarr",
            chunks={"leadtime": 1},
        )

        print(f"Final combined {data_type} preds dataset: {combined_preds.dims}")

        del preds_ds_list, combined_preds
        gc.collect()

    _combine_leadtimes("train", train_pred_paths)
    if val_pred_paths:
        _combine_leadtimes("val", val_pred_paths)
    _combine_leadtimes("test", test_pred_paths)


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
