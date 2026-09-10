from copy import deepcopy
from pathlib import Path
from typing import Literal

import gc

import torch
from torch.utils.data import DataLoader
import lightning as L
import xarray as xr

from earthml import (
    Normalize,
    MonthlyNormalize,
    XarrayDataset,
    build_net,
    get_experiment_configs,
    open_zarr,
    safe_chunk_spec,
    save_zarr,
)

from train import (
    _test,
    add_or_set_leadtime,
    make_train_test_datasets_for_leadtime,
    resolve_accelerator_and_device,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

EXPERIMENTS_ROOT = Path(
    "/work/cmcc/jd19424/ML/MLBC/experiments/weather_atmo"
)

VARIABLES = [
    "t2m",
]

REGIONS = [
    "ConUS",
]

TEST_START = "2025-01-01"
TEST_END = "2025-10-10" # currently latest available day

WEIGHTS: Literal["best", "last"] = "best"

# Optional per-leadtime checkpoint override.
# Leave empty to use the selected experiment's best/last checkpoint.
CHECKPOINT_OVERRIDES: dict[int, Path] = {
    # 12: Path("/path/to/custom.ckpt"),
}

OVERWRITE = False
LOG_MONTHLY = False
INTERPOLATE_ANALYSIS = True


def dataset_kwargs_from_settings(s) -> dict:
    return {
        "target_realization_avg": s.target_realization_avg,
        "channel_representation": s.channel_representation,
        "init_period_dim": s.init_period_dim,
        "output_realizations": s.output_realizations,
        "torch_mask": s.torch_mask,
        "fill_nan_value": s.fill_nan_value,
    }


def make_normalizers(s, train_dataset: XarrayDataset):
    if s.normalization == "monthly":
        norm_class = MonthlyNormalize
    elif s.normalization == "full":
        norm_class = Normalize
    else:
        raise ValueError(
            f"Unsupported normalization={s.normalization!r}"
        )

    input_excluded_channels = (
        (-4, -3, -2, -1)
        if (
            s.seasonal_encoding
            and s.channel_representation != "init_period"
        )
        else None
    )

    normalize_input = norm_class(
        mode=s.normalization_mode,
        exclude_channels=input_excluded_channels,
    ).fit(
        train_dataset,
        dim="x",
    )

    # Keep identical to the current NOAA-copy training behavior.
    normalize_target = deepcopy(normalize_input)

    return normalize_input, normalize_target


def make_net_kwargs(
    s,
    train_dataset: XarrayDataset,
    region_name: str,
) -> dict:
    lat_dim = train_dataset.target_ds.earthml.guessed_dims.latitude

    latitudes = torch.as_tensor(
        train_dataset.target_ds[lat_dim].values,
        dtype=torch.float32,
    )

    grid_spacing = abs(
        float(
            train_dataset.target_ds[lat_dim]
            .diff(lat_dim)
            .median()
            .values
        )
    )

    loss_kwargs = dict(s.loss_kwargs)

    losses_with_latitudes = {
        "GeoMSELoss",
        "GeoMaskedMSELoss",
        "GeoMaskedMSEMultiScaleLoss",
        "SpatialCVaRMSELoss",
        "SpatialDegradationMSELoss",
    }

    if s.loss_name in losses_with_latitudes:
        loss_kwargs["latitudes"] = latitudes

    if "spatial_patch_size_degrees" in loss_kwargs:
        patch_degrees = float(
            loss_kwargs.pop("spatial_patch_size_degrees")
        )
        loss_kwargs["patch_size"] = max(
            1,
            round(patch_degrees / grid_spacing),
        )

    if s.loss_name == "GeoMaskedMSEMultiScaleLoss":
        scales_degrees = loss_kwargs.pop("scales_degrees")

        pool_kernel_sizes = []

        for scale_degrees in scales_degrees:
            kernel_size = max(
                3,
                round(float(scale_degrees) / grid_spacing),
            )

            if kernel_size % 2 == 0:
                kernel_size += 1

            pool_kernel_sizes.append(kernel_size)

        loss_kwargs["pool_kernel_sizes"] = tuple(
            pool_kernel_sizes
        )

    n_channels = train_dataset.x.shape[1]
    n_classes = train_dataset.y.shape[1]

    longitude_padding = (
        "circular"
        if region_name == "World"
        else "replicate"
    )

    common_net_kwargs = {
        "learning_rate": s.init_learning_rate,
        "weight_decay": s.weight_decay,
        "loss": s.loss_name,
        "loss_params": {
            "loss": loss_kwargs,
            "net": {},
        },
        "norm": s.training_norm,
        "supervised": True,
        "n_channels": n_channels,
        "n_classes": n_classes,
        "longitude_padding": longitude_padding,
        "zero_init_output": (
            s.target_mode
            in {
                "residual",
                "residual_realization",
                "anomaly_residual",
                "anomaly_residual_realization",
            }
        ),
    }

    return {
        **common_net_kwargs,
        **s.extra_net_kwargs,
    }


def resolve_checkpoint(
    s,
    leadtime: int,
) -> Path:
    override = CHECKPOINT_OVERRIDES.get(int(leadtime))

    if override is not None:
        checkpoint = override
    else:
        exp_name = f"exp_{leadtime}_{s.leadtime_unit}"
        exp_dir = s.exp_dir / exp_name

        if WEIGHTS == "best":
            checkpoint = exp_dir / "weights" / "weights.ckpt"
        else:
            checkpoint = exp_dir / "checkpoints" / "last.ckpt"

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint}"
        )

    return checkpoint


def combine_predictions(
    prediction_paths: list[tuple[int, Path]],
    output_path: Path,
) -> None:
    datasets = []

    try:
        for leadtime, path in prediction_paths:
            ds = open_zarr(path)
            ds = add_or_set_leadtime(ds, leadtime)
            datasets.append(ds)

        combined = xr.concat(
            datasets,
            dim="leadtime",
            coords="minimal",
            compat="override",
            join="outer",
        ).sortby("leadtime")

        combined = combined.chunk(
            safe_chunk_spec(combined)
        )

        save_zarr(
            combined,
            output_path,
            chunks={"leadtime": 1},
        )

        print(
            "Saved combined predictions:",
            output_path,
            dict(combined.sizes),
        )

    finally:
        for ds in datasets:
            try:
                ds.close()
            except Exception:
                pass


def infer_setting(s) -> None:
    if s.regional_training:
        raise NotImplementedError(
            "This inference script currently expects "
            "regional_training=False."
        )

    if s.separate_training_by_init_period is not None:
        raise NotImplementedError(
            "This inference script currently expects "
            "separate_training_by_init_period=None."
        )

    accelerator, device = resolve_accelerator_and_device()

    if accelerator == "gpu":
        torch.set_float32_matmul_precision("high")

    dataset_kwargs = dataset_kwargs_from_settings(s)

    output_dir = (
        s.exp_dir
        / "inference"
        / f"{TEST_START}_{TEST_END}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_output = output_dir / "test_corrected.zarr"

    if combined_output.exists() and not OVERWRITE:
        print(
            f"Skipping {s.output_name}: "
            f"{combined_output} already exists."
        )
        return

    prediction_paths: list[tuple[int, Path]] = []

    for leadtime in s.leadtimes:
        leadtime = int(leadtime)

        print("=" * 80)
        print(
            f"{s.output_name}: "
            f"leadtime={leadtime} {s.leadtime_unit}"
        )
        print(f"test period: {TEST_START} -> {TEST_END}")

        # Match the original training dataset extent exactly.
        explicit_split = s.split_strategy == "explicit"
        train_end = (
            s.train_end
            if explicit_split
            else s.val_end
        )

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
            train_end=train_end,
            val_start=None,
            val_end=None,
            test_start=TEST_START,
            test_end=TEST_END,
            target_mode=s.target_mode,
            clim_period=s.clim_period,
            forecast_vars=[s.var_fc],
            analysis_vars=[s.var_an],
            region=s.region,
            dataset_kwargs=dataset_kwargs,
            seasonal_encoding=(
                s.seasonal_encoding
                and s.channel_representation != "init_period"
            ),
            ensemble_encoding=s.ensemble_encoding,
            input_realization_avg=s.input_realization_avg,
            interpolate_analysis=INTERPOLATE_ANALYSIS,
            materialize=False,
            separate_training_by_init_period=None,
            defer_dataset_creation=False,
        )

        train_dataset = datasets["train"]
        test_dataset = datasets["test"]

        if not isinstance(train_dataset, XarrayDataset):
            raise TypeError("Expected XarrayDataset for training data")
        if not isinstance(test_dataset, XarrayDataset):
            raise TypeError("Expected XarrayDataset for test data")

        normalize_input, normalize_target = make_normalizers(
            s,
            train_dataset,
        )

        train_dataset.transform_x = normalize_input
        train_dataset.transform_y = normalize_target

        test_dataset.transform_x = normalize_input
        test_dataset.transform_y = normalize_target

        net_kwargs = make_net_kwargs(
            s,
            train_dataset,
            s.region_name,
        )

        checkpoint = resolve_checkpoint(
            s,
            leadtime,
        )

        print(f"weights: {checkpoint}")

        template_model = build_net(
            name=s.net_name,
            **net_kwargs,
        )

        model = type(template_model).load_from_checkpoint(
            checkpoint,
            strict=False,
            **net_kwargs,
        ).to(device)

        del template_model

        dataloader = DataLoader(
            test_dataset,
            batch_size=s.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=(accelerator == "gpu"),
            persistent_workers=False,
        )

        trainer = L.Trainer(
            accelerator=accelerator,
            devices=1,
            precision=s.trainer_precision,
            logger=False,
            enable_checkpointing=False,
        )

        lead_output = (
            output_dir
            / f"test_corrected_{leadtime}_{s.leadtime_unit}.zarr"
        )

        if lead_output.exists() and OVERWRITE:
            import shutil
            shutil.rmtree(lead_output)

        _test(
            test_trainer=trainer,
            s=s,
            model=model,
            dataset=test_dataset,
            normalize_target=normalize_target,
            dataloader=dataloader,
            preds_store=lead_output,
            an_clim=datasets["y_clim"],
            log_monthly=LOG_MONTHLY,
        )

        prediction_paths.append(
            (leadtime, lead_output)
        )

        trainer.strategy.teardown()

        for ds in (
            train_dataset.input_ds,
            train_dataset.target_ds,
            test_dataset.input_ds,
            test_dataset.target_ds,
            datasets["x_clim"],
            datasets["y_clim"],
        ):
            if ds is not None:
                try:
                    ds.close()
                except Exception:
                    pass

        del trainer
        del model
        del dataloader
        del train_dataset
        del test_dataset
        del normalize_input
        del normalize_target
        del datasets

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ):
            torch.mps.empty_cache()

    if combined_output.exists() and OVERWRITE:
        import shutil
        shutil.rmtree(combined_output)

    combine_predictions(
        prediction_paths,
        combined_output,
    )


def main() -> None:
    settings = get_experiment_configs(
        EXPERIMENTS_ROOT,
        var_fc=VARIABLES,
        region_name=REGIONS,
        net_name="SmaAt_UNet",
        # net_name="ConvNeXtTransformerUNet",
        # target_mode="anomaly_residual",
        extra_suffix_folder="NOAA_copy",
    )

    print(f"Found {len(settings)} matching experiment(s).")

    if not settings:
        raise RuntimeError("No matching experiments found.")

    for s in settings:
        infer_setting(s)


if __name__ == "__main__":
    main()
