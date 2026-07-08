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

from earthml import (
    LeadtimeUnit,
    ClimPeriod,
    TargetMode,
    XarrayDataset,
    build_loss,
    build_net,
    Normalize,
    MonthlyNormalize,
    SplitDataModule,
    Table,
    Settings,
    calculate_climatology,
    select_clim_for_time,
    open_zarr,
)


class LeadtimeDatasets(TypedDict):
    train: XarrayDataset
    test: XarrayDataset
    x_clim: xr.Dataset | None
    y_clim: xr.Dataset | None

console = Console()


def compute_leadtimes(
    leadtime_value: int | float,
    leadtime_unit: LeadtimeUnit,
):
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


def compute_valid_times(
    init_times,
    leadtime_value: int | float,
    leadtime_unit: LeadtimeUnit,
):
    init_times = pd.DatetimeIndex(init_times)

    leadtime = compute_leadtimes(leadtime_value, leadtime_unit)

    return init_times + leadtime


def make_leadtime_pair(
    forecast_ds_path: str | Path,
    analysis_ds_path: str | Path,
    leadtime: int | float,
    leadtime_unit: LeadtimeUnit,
    start: str,
    end: str,
    clim_start: str,
    clim_end: str,
    target_mode: TargetMode = "analysis",
    time_dim: str = "time",
    leadtime_dim: str = "leadtime",
    clim_period: ClimPeriod = ClimPeriod.MONTH,
    forecast_vars: Sequence | None = None,
    analysis_vars: Sequence | None = None,
    lat: Sequence | None = None,
    lon: Sequence | None = None,
    interpolate: bool = True,
):
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

    def _gen_fc_an_ds(
        fc_ds: xr.Dataset,
        an_ds: xr.Dataset,
        start: str,
        end: str,
        label: str = "",
    ) -> tuple[xr.Dataset, xr.Dataset]:
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
        an_ds = an_ds.rename({time_dim: time_dim})
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

        with ProgressBar():
            print(f"Materialize {label} input dataset")
            fc_ds = fc_ds.load()
            print(f"Materialize {label} target dataset")
            an_ds = an_ds.load()

        return fc_ds, an_ds

    fc_train_ds, an_train_ds = _gen_fc_an_ds(fc_ds, an_ds, start, end, "train")

    if target_mode == "analysis":
        return fc_train_ds, an_train_ds, None, None

    if target_mode == "residual":
        fc_train_ens_mean = fc_train_ds.mean("realization") if "realization" in fc_train_ds.dims else fc_train_ds
        res_train_ds = an_train_ds - fc_train_ens_mean
        return fc_train_ds, res_train_ds, None, None

    if target_mode in {"anomaly", "anomaly_residual"}:
        fc_clim_ds, an_clim_ds = _gen_fc_an_ds(
            fc_ds,
            an_ds,
            clim_start,
            clim_end,
            "clim",
        )

        fc_clim: xr.Dataset = calculate_climatology(fc_clim_ds, time_dim, clim_period)
        an_clim: xr.Dataset = calculate_climatology(an_clim_ds, time_dim, clim_period)

        fc_train_ds, fc_clim = xr.unify_chunks(fc_train_ds, fc_clim)
        an_train_ds, an_clim = xr.unify_chunks(an_train_ds, an_clim)

        fc_clim_for_train = select_clim_for_time(fc_clim, fc_train_ds[time_dim].values, clim_period)
        an_clim_for_train = select_clim_for_time(an_clim, an_train_ds[time_dim].values, clim_period)

        fc_train_anom = fc_train_ds - fc_clim_for_train
        an_train_anom = an_train_ds - an_clim_for_train

        if target_mode == "anomaly":
            return fc_train_anom, an_train_anom, fc_clim, an_clim

        fc_anom_mean = (
            fc_train_anom.mean("realization")
            if "realization" in fc_train_anom.dims
            else fc_train_anom
        )

        res_train_anom_ds = an_train_anom - fc_anom_mean
        return fc_train_anom, res_train_anom_ds, fc_clim, an_clim

    raise ValueError(f"Unsupported target_mode={target_mode!r}")


def make_train_test_datasets_for_leadtime(
    forecast_ds_path: str | Path,
    analysis_ds_path: str | Path,
    leadtime: int | float,
    leadtime_unit: LeadtimeUnit,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    target_mode: TargetMode = "analysis",
    clim_period: ClimPeriod = ClimPeriod.MONTH,
    forecast_vars: Sequence[str] | None = None,
    analysis_vars: Sequence[str] | None = None,
    region: dict | None = None,
    dataset_kwargs: dict | None = None,
    interpolate: bool = True,
) -> LeadtimeDatasets:
    dataset_kwargs = dataset_kwargs or {}

    x_train, y_train, x_clim, y_clim = make_leadtime_pair(
        forecast_ds_path=forecast_ds_path,
        analysis_ds_path=analysis_ds_path,
        leadtime=leadtime,
        leadtime_unit=leadtime_unit,
        start=train_start,
        end=train_end,
        clim_start=train_start,
        clim_end=train_end,
        target_mode=target_mode,
        clim_period=clim_period,
        forecast_vars=forecast_vars,
        analysis_vars=analysis_vars,
        lon=region["lon"] if region is not None else None,
        lat=region["lat"] if region is not None else None,
        interpolate=interpolate,
    )

    x_test, y_test, _, _ = make_leadtime_pair(
        forecast_ds_path=forecast_ds_path,
        analysis_ds_path=analysis_ds_path,
        leadtime=leadtime,
        leadtime_unit=leadtime_unit,
        start=test_start,
        end=test_end,
        clim_start=train_start,
        clim_end=train_end,
        target_mode=target_mode,
        clim_period=clim_period,
        forecast_vars=forecast_vars,
        analysis_vars=analysis_vars,
        lon=region["lon"] if region is not None else None,
        lat=region["lat"] if region is not None else None,
        interpolate=interpolate,
    )

    train_ds = XarrayDataset(x_train, y_train, **dataset_kwargs)
    test_ds = XarrayDataset(x_test, y_test, **dataset_kwargs)

    return {
        "train": train_ds,
        "test": test_ds,
        "x_clim": x_clim,
        "y_clim": y_clim,
    }


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
        # filename="checkpoint_{epoch:02d}-{val_loss:.2f}"
        filename="checkpoint",
    )
    callbacks.append(periodic_checkpoint_callback)

    # Model best weights, MUST be the last ModelCheckpoint. Can append from here since ModelCheckpoint will always be the last callbacks
    best_weights_callback = ModelCheckpoint(
        dirpath=weights_folder_path,
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename="weights"
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

    tdim = "time"
    rdim = "realization"
    ydim = "latitude"
    xdim = "longitude"

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
    input_rdim = input_ds.earthml.guessed_dims.realization
    if R_out > 1 and rdim is None:
        rdim = input_rdim or "realization"

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
    train_shape: tuple | None = None,
    target_shape: tuple | None = None,
    test_shape: tuple | None = None,
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
        "experiment.torch_workers": s.torch_workers,

        "data.forecast": f"{s.model_fc}/{s.var_fc}",
        "data.analysis": f"{s.model_an}/{s.var_an}",
        "data.region": f"{s.region_name} {s.region}",
        "data.train_period": f"{s.train_start} → {s.train_end}",
        "data.test_period": f"{s.test_start} → {s.test_end}",
        "data.train_x": train_shape,
        "data.train_y": target_shape,
        "data.test_x": test_shape,
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
        "model.base_channels": s.base_channels,
        "model.kernels_per_layer": s.kernels_per_layer,
        "model.reduction_ratio": s.reduction_ratio,

        "training.normalization": f"{normalization_name}(x), {normalization_name}(y)",
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
    interpolate = True

    accelerator, device = resolve_accelerator_and_device()

    if accelerator == "gpu":
        torch.set_float32_matmul_precision("high")

    s = Settings(
        root_dir=Path("/Users/jacopodallaglio/ML/training/seasonal"),
        data_root_dir=None,
        exp_root_dir=None,
        plot_root_dir=None,
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
        train_end="2020-12-01",
        test_start="2021-01-01",
        test_end="2022-12-01",
        target_mode="analysis",
        clim_period=ClimPeriod.MONTH,
        seed=42,
        realization_as_channel=False,
        output_realizations="deterministic",
        split_strategy="time",
        pretrain_norm="full",
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
        kernels_per_layer=1,
        base_channels=32,
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

    base_loss_params = {
        "loss": {},
        "net": {},
    }

    pred_paths: list[tuple[int, Path]] = []
    train_pred_paths: list[tuple[int, Path]] = []
    for lt in s.leadtimes:
        dataset_d = make_train_test_datasets_for_leadtime(
            forecast_ds_path=s.input_dir / f"{s.model_fc}_{s.var_fc}.zarr",
            analysis_ds_path=s.input_dir / f"{s.model_an}_{s.var_an}.zarr",
            leadtime=lt,
            leadtime_unit=LeadtimeUnit(s.leadtime_unit),
            train_start=s.train_start,
            train_end=s.train_end,
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
            interpolate=interpolate,
        )

        exp_name = f"exp_{lt}_{s.leadtime_unit}"

        # Create exp dirs
        exp_dir = s.exp_dir / exp_name
        weights_dir = exp_dir / "weights"
        checkpoints_dir = exp_dir / "checkpoints"

        for d in (weights_dir, checkpoints_dir):
            d.mkdir(exist_ok=True, parents=True)

        train_dataset = dataset_d["train"]

        # Normalize
        if s.pretrain_norm == "monthly":
            NormClass = MonthlyNormalize
        elif s.pretrain_norm == "full":
            NormClass = Normalize
        else:
            raise ValueError(f"pretrain_norm={s.pretrain_norm} not supported.")

        normalize_input  = NormClass().fit(train_dataset, dim='x')
        normalize_target = NormClass().fit(train_dataset, dim='y')
        train_dataset.transform_x = normalize_input
        train_dataset.transform_y = normalize_target

        # print("Train dataset input shape:", train_dataset.x.shape)
        # print("Train dataset target shape:", train_dataset.y.shape)

        n_channels = train_dataset.x.shape[1]
        n_classes = train_dataset.y.shape[1] if s.output_realizations=="deterministic" else train_dataset.x.shape[1]

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
            train_dataset,
            train_fraction=s.train_fraction,
            batch_size=s.batch_size,
            seed=s.seed,
            num_workers=s.torch_workers,
            split_strategy=s.split_strategy,
        )

        train_idx, val_idx = train_datamodule._get_indices()

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
            train_shape=tuple(train_dataset.x.shape),
            target_shape=tuple(train_dataset.y.shape),
            test_shape=tuple(test_dataset.x.shape),
            test_target_shape=tuple(test_dataset.y.shape),
            train_idx=train_idx,
            val_idx=val_idx,
            exp_name=exp_name,
            weights_dir=weights_dir,
            checkpoints_dir=checkpoints_dir,
        )

        train_trainer = L.Trainer(
            max_epochs=s.max_epochs,
            accelerator=accelerator,
            devices=1,
            precision=s.trainer_precision,
            # gradient_clip_val=1.0,           # Recommended starting value (e.g., 0.5, 1.0, 5.0)
            # gradient_clip_algorithm="norm",  # "norm" for clipping by norm, "value" for clipping by value
            log_every_n_steps=1,
            logger=None,
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

        # Find checkpoints
        ckpt_files = list(checkpoints_dir.glob("*.ckpt"))
        try:
            ckpt_path = max(ckpt_files, key=lambda p: p.stat().st_ctime) # get most recent
        except ValueError: # empty sequence
            ckpt_path = checkpoints_dir / "checkpoint.ckpt"

        # Train
        train_ckpt_path = None if force_retrain else (Path(ckpt_path) if Path(ckpt_path).exists() else None)
        train_trainer.fit(
            model,
            datamodule=train_datamodule,
            ckpt_path=train_ckpt_path,
        )

        # Load weights from latest checkpoints
        checkpoint = torch.load(ckpt_path, map_location=device)
        callbacks = checkpoint["callbacks"]
        last_key, last_callback = next(reversed(callbacks.items()))
        old_weights_file = Path(last_callback["best_model_path"])
        weights_file = weights_dir / old_weights_file.name
        weights = torch.load(weights_file, map_location=device)
        model.load_state_dict(weights['state_dict'])

        test_trainer = L.Trainer(
            accelerator=accelerator,
            devices=1,
            precision=s.trainer_precision,
            # gradient_clip_val=1.0,           # Recommended starting value (e.g., 0.5, 1.0, 5.0)
            # gradient_clip_algorithm="norm",  # "norm" for clipping by norm, "value" for clipping by value
            accumulate_grad_batches=s.accumulate_grad_batches,
            # deterministic=True
        )

        def _test(
            dataset: XarrayDataset,
            dataloader: DataLoader,
            zarr_file_name: str,
            an_clim: xr.Dataset | None = None,
            fc_clim: xr.Dataset | None = None,
        ) -> Path:
            test_trainer.test(model, dataloaders=dataloader)

            preds_norm = model.test_preds.detach().float().cpu()
            del model.test_preds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            months = dataset.months[: preds_norm.shape[0]]

            preds = normalize_target.inverse_tensor(preds_norm, months)
            preds_store = exp_dir / f"{zarr_file_name}.zarr"
            preds_ds = convert_to_xarray(preds, dataset, [s.var_an])

            # Reconstruct
            if s.target_mode == "residual":
                fc_base = dataset.input_ds[s.var_fc]
                if "realization" in fc_base.dims:
                    fc_base = fc_base.mean("realization")

                preds_ds = (preds_ds[s.var_an] + fc_base).to_dataset(name=s.var_an)

            elif s.target_mode == "anomaly":
                if an_clim is None:
                    raise ValueError("an_clim is required for target_mode='anomaly'.")

                an_clim_for_time = select_clim_for_time(
                    an_clim,
                    preds_ds.time.values,
                    s.clim_period,
                )

                preds_ds = (preds_ds[s.var_an] + an_clim_for_time[s.var_an]).to_dataset(
                    name=s.var_an
                )

            elif s.target_mode == "anomaly_residual":
                if an_clim is None or fc_clim is None:
                    raise ValueError(
                        "an_clim and fc_clim are required for residual anomaly reconstruction."
                    )

                an_clim_for_time = select_clim_for_time(
                    an_clim,
                    preds_ds.time.values,
                    s.clim_period,
                )

                fc_clim_for_time = select_clim_for_time(
                    fc_clim,
                    preds_ds.time.values,
                    s.clim_period,
                )

                fc_base = dataset.input_ds[s.var_fc]
                fc_anom = fc_base - fc_clim_for_time[s.var_fc]

                if "realization" in fc_anom.dims:
                    fc_anom = fc_anom.mean("realization")

                corrected = preds_ds[s.var_an] + fc_anom + an_clim_for_time[s.var_an]
                preds_ds = corrected.to_dataset(name=s.var_an)

            preds_ds.to_zarr(
                preds_store,
                mode="w",
                consolidated=False,
                zarr_format=2,
                align_chunks=True,
            )

            del preds_norm, preds, preds_ds
            if hasattr(model, "test_preds"):
                del model.test_preds

            return preds_store

        # Predict
        test_store = _test(test_dataset, test_dataloader, "test_preds", dataset_d["y_clim"], dataset_d["x_clim"])
        pred_paths.append((int(lt), test_store))

        # Predict on train period too, for climatology
        train_store = _test(train_dataset, train_test_dataloader, "train_preds", dataset_d["y_clim"], dataset_d["x_clim"])
        train_pred_paths.append((int(lt), train_store))

        # Trainer clean-up
        train_trainer.strategy.teardown()
        test_trainer.strategy.teardown()

        del model, train_trainer, test_trainer
        del train_datamodule, test_dataloader, train_test_dataloader
        del train_dataset, test_dataset, dataset_d

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Combine all leadtimes
    preds_ds_list = [
        add_or_set_leadtime(open_zarr(path), lt)
        for lt, path in pred_paths
    ]
    all_preds = xr.concat(
        preds_ds_list,
        dim="leadtime",
        coords="minimal",
        compat="override",
    )
    all_preds.to_zarr(
        s.output_dir / "test_corrected.zarr",
        mode="w",
        consolidated=False,
        zarr_format=2,
    )
    print(f"Final combined test preds dataset: {all_preds.dims}")

    del preds_ds_list, all_preds
    gc.collect()

    train_ds_list = [
        add_or_set_leadtime(open_zarr(path), lt)
        for lt, path in train_pred_paths
    ]
    all_train_preds = xr.concat(
        train_ds_list,
        dim="leadtime",
        coords="minimal",
        compat="override",
    )
    all_train_preds.to_zarr(
        s.output_dir / f"train_corrected.zarr",
        mode="w",
        consolidated=False,
        zarr_format=2,
    )
    print(f"Final combined train preds dataset: {all_train_preds.dims}")

    del train_ds_list, all_train_preds
    gc.collect()


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
