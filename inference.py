from copy import deepcopy
from pathlib import Path
from typing import Literal

import gc

import numpy as np
import pandas as pd
import xarray as xr

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import lightning as L

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


# =============================================================================
# Configuration
# =============================================================================

exp_name = "weather_atmo"
# exp_name = "weather_atmo_one_year_test"
# exp_name = "weather_atmo_ablation_fixed_val"
# exp_name = "weather_atmo_short_zero_vs_replicate_padding"

EXPERIMENTS_ROOT = Path(
    "/Users/jacopodallaglio/ML/training/seasonal/experiments"
    # f"/work/cmcc/jd19424/ML/MLBC/experiments/{exp_name}"
)

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
    "ConUS",
    # "Europe",
    # "Pacific",
    # "World",
    # None,
]

# TEST_START = "2025-01-01T00:00:00"
# TEST_END = "2025-01-01T00:00:00"
# TEST_END = "2025-05-01T00:00:00"
# TEST_END = "2025-09-30T12:00:00"
# TEST_END = "2025-05-12" # corresponds to 264 samples
# TEST_END = "2025-10-10" # currently latest available day

# Feature map extraction inference
TEST_START = "2025-02-01T00:00:00"
# TEST_END = "2025-02-10T00:00:00"
TEST_END = "2025-03-31T00:00:00"

WEIGHTS: Literal["best", "last"] = "best"

# Optional per-leadtime checkpoint override.
# Leave empty to use the selected experiment's best/last checkpoint.
CHECKPOINT_OVERRIDES: dict[int, Path] = {
    # 12: Path("/path/to/custom.ckpt"),
}

OVERWRITE = True

LOG_MONTHLY = False

INTERPOLATE_ANALYSIS = False # weather

# -----------------------------------------------------------------------------
# Save feature maps config
# -----------------------------------------------------------------------------

FEATURE_MAP_TIMES: list[str] | None = [
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
    "2025-03-08T00:00:00", # B1 t2m 72h
    "2025-02-22T00:00:00", # B2 t2m 72h
    "2025-03-23T00:00:00", # B3 t2m 72h
    "2025-03-07T00:00:00", # B4 t2m 72h
    "2025-02-02T00:00:00", # B5 t2m 72h, near W4 in std-vs-mean scatter
    "2025-02-21T00:00:00", # B6 t2m 72h
    "2025-04-06T00:00:00", # B7 t2m 72h
    "2025-04-07T00:00:00", # B8 t2m 72h
    "2025-02-25T00:00:00", # B9 t2m 72h
    "2025-02-24T00:00:00", # B10 t2m 72h

    "2025-02-13T00:00:00", # W1 t2m 72h
    "2025-02-14T00:00:00", # W2 t2m 72h
    "2024-11-28T00:00:00", # W3 t2m 72h
    "2025-02-06T00:00:00", # W4 t2m 72h
    "2025-03-29T00:00:00", # W5 t2m 72h
    "2024-11-29T00:00:00", # W6 t2m 72h
    "2025-01-16T00:00:00", # W7 t2m 72h
    "2025-04-24T00:00:00", # W8 t2m 72h
    "2024-10-14T00:00:00", # W9 t2m 72h
    "2025-05-05T12:00:00", # W10 t2m 72h
]

SAVE_FEATURE_MAPS = True

# =============================================================================
# Feature map collector class
# =============================================================================

class FeatureMapCollector:
    def __init__(self, model):
        self.model = model

        self.handles: list = []
        self.features: dict[str, torch.Tensor] = {}

    def _make_hook(self, name: str):
        def hook(module, inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"Expected tensor output from {name}, "
                    f"got {type(output)}"
                )

            self.features[name] = (
                output
                .detach()
                .float()
                .cpu()
            )

        return hook

    def register(self):
        # ------------------------------------------------------
        # Initial encoder block
        # ------------------------------------------------------
        self.handles.append(
            self.model.inc.register_forward_hook(
                self._make_hook("encoder_raw_0")
            )
        )

        # ------------------------------------------------------
        # Downsampling encoder blocks
        # ------------------------------------------------------
        for level, module in enumerate(
            self.model.downs,
            start=1,
        ):
            self.handles.append(
                module.register_forward_hook(
                    self._make_hook(
                        f"encoder_raw_{level}"
                    )
                )
            )

        # ------------------------------------------------------
        # Attention-filtered encoder representations
        # ------------------------------------------------------
        for level, module in enumerate(
            self.model.encoder_cbam
        ):
            self.handles.append(
                module.register_forward_hook(
                    self._make_hook(
                        f"encoder_cbam_{level}"
                    )
                )
            )

        # ------------------------------------------------------
        # Decoder blocks
        # ------------------------------------------------------
        for level, module in enumerate(
            self.model.ups
        ):
            self.handles.append(
                module.register_forward_hook(
                    self._make_hook(
                        f"decoder_{level}"
                    )
                )
            )

        # ------------------------------------------------------
        # Network output
        # ------------------------------------------------------
        self.handles.append(
            self.model.outc.register_forward_hook(
                self._make_hook("output")
            )
        )

    def clear(self):
        self.features.clear()

    def remove(self):
        for handle in self.handles:
            handle.remove()

        self.handles.clear()

# =============================================================================
# Feature map generation utils
# =============================================================================

def get_sample_times(
    dataset: XarrayDataset,
) -> np.ndarray:
    time_dim = dataset.input_ds.earthml.guessed_dims.time

    if time_dim is None:
        raise ValueError(
            "Could not determine input time dimension."
        )

    init_times = dataset.input_ds[time_dim].values

    return np.repeat(
        init_times,
        dataset.samples_per_init,
    )


def extract_feature_maps(
    model,
    dataset: XarrayDataset,
    normalize_target,
    wanted_times: list[str],
    output_dir: Path,
    device,
    latitudes: torch.Tensor,
) -> None:
    sample_times = get_sample_times(dataset)

    requested_times = pd.to_datetime(
        wanted_times
    ).to_numpy(dtype="datetime64[ns]")

    sample_times_ns = np.asarray(
        sample_times,
        dtype="datetime64[ns]",
    )

    collector = FeatureMapCollector(model)
    collector.register()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    was_training = model.training
    model.eval()

    try:
        for requested_time in requested_times:
            matching_indices = np.flatnonzero(
                sample_times_ns == requested_time
            )

            if matching_indices.size == 0:
                print(
                    "WARNING: no test sample for "
                    f"{requested_time}"
                )
                continue

            for sample_within_init, sample_index in enumerate(
                matching_indices
            ):
                sample_index = int(sample_index)

                # Transformed input/target used by the network.
                x, y, _, _ = dataset[sample_index]

                # Add target mask
                target_mask = (
                    dataset.y_mask[sample_index]
                    .detach()
                    .cpu()
                    .bool()
                )

                # Raw physical forecast, before input normalization.
                raw_baseline = (
                    dataset.x[sample_index]
                    .detach()
                    .float()
                    .cpu()
                )

                # Keep only forecast channels corresponding to target.
                raw_baseline = raw_baseline[
                    : y.shape[0]
                ]

                # Put the physical FC into TARGET-normalized space.
                if isinstance(
                    normalize_target,
                    MonthlyNormalize,
                ):
                    month = int(
                        dataset.months[sample_index].item()
                    )

                    baseline_target_norm = normalize_target(
                        raw_baseline,
                        months=month,
                    )
                else:
                    baseline_target_norm = normalize_target(
                        raw_baseline,
                    )

                if x.ndim != 3:
                    raise ValueError(
                        "Expected input tensor (C,H,W), "
                        f"got {tuple(x.shape)}"
                    )

                collector.clear()

                x_batch = (
                    x
                    .unsqueeze(0)
                    .to(device)
                )

                with torch.inference_mode():
                    prediction = model(x_batch)

                # Remove batch dimension from everything.
                features = {
                    name: value[0].clone()
                    for name, value
                    in collector.features.items()
                }

                # Validation
                if target_mask.shape != y.shape:
                    raise ValueError(
                        f"Target mask/target shape mismatch: "
                        f"{tuple(target_mask.shape)} != "
                        f"{tuple(y.shape)}"
                    )

                if latitudes.ndim != 1:
                    raise ValueError(
                        f"Expected latitudes (H,), "
                        f"got {tuple(latitudes.shape)}"
                    )

                if latitudes.shape[0] != y.shape[-2]:
                    raise ValueError(
                        f"Latitude count {latitudes.shape[0]} "
                        f"does not match target height {y.shape[-2]}"
                    )

                # Prepare payload
                payload = {
                    "sample_index": sample_index,
                    "sample_within_init": (
                        sample_within_init
                    ),
                    "init_time": str(
                        pd.Timestamp(requested_time)
                    ),

                    "input": (
                        x.detach()
                        .float()
                        .cpu()
                    ),

                    "target": (
                        y.detach()
                        .float()
                        .cpu()
                    ),

                    "prediction": (
                        prediction[0]
                        .detach()
                        .float()
                        .cpu()
                    ),

                    "baseline_target_norm": (
                        baseline_target_norm
                        .detach()
                        .float()
                        .cpu()
                    ),

                    "target_mask": target_mask,

                    "latitudes": latitudes.detach().float().cpu(),

                    "features": features,
                }

                time_label = (
                    pd.Timestamp(requested_time)
                    .strftime("%Y%m%dT%H%M%S")
                )

                path = (
                    output_dir
                    / (
                        f"features_init_{time_label}"
                        f"_sample_{sample_within_init}.pt"
                    )
                )

                torch.save(
                    payload,
                    path,
                )

                print(
                    f"Saved feature maps: {path}"
                )

                for name, feature in features.items():
                    print(
                        f"  {name:20s} "
                        f"shape={tuple(feature.shape)} "
                        f"mean={feature.mean().item():+.4f} "
                        f"std={feature.std().item():.4f} "
                        f"min={feature.min().item():+.4f} "
                        f"max={feature.max().item():+.4f}"
                    )

    finally:
        collector.remove()

        if was_training:
            model.train()


def feature_spatial_coords(
    dataset: XarrayDataset,
    height: int,
    width: int,
):
    lat_dim = dataset.input_ds.earthml.guessed_dims.latitude
    lon_dim = dataset.input_ds.earthml.guessed_dims.longitude

    latitude = np.asarray(
        dataset.input_ds[lat_dim].values
    )

    longitude = np.asarray(
        dataset.input_ds[lon_dim].values
    )

    feature_latitude = np.linspace(
        latitude[0],
        latitude[-1],
        height,
    )

    feature_longitude = np.linspace(
        longitude[0],
        longitude[-1],
        width,
    )

    return (
        lat_dim,
        lon_dim,
        feature_latitude,
        feature_longitude,
    )

# =============================================================================
# Inference utils
# =============================================================================

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

    normalize_target = norm_class(
        mode=s.normalization_mode,
    ).fit(
        train_dataset,
        dim="y",
    )

    return normalize_input, normalize_target


def make_net_kwargs(
    s,
    train_dataset: XarrayDataset,
    region_name: str,
    latitudes: torch.Tensor,
) -> dict:
    lat_dim = train_dataset.target_ds.earthml.guessed_dims.latitude

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

        # Calculate latitudes for loss, net and metrics
        lat_dim = train_dataset.target_ds.earthml.guessed_dims.latitude
        latitudes = torch.as_tensor(
            train_dataset.target_ds[lat_dim].values,
            dtype=torch.float32,
        )

        net_kwargs = make_net_kwargs(
            s,
            train_dataset,
            s.region_name,
            latitudes=latitudes,
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

        # Save feature maps if requested
        if SAVE_FEATURE_MAPS and FEATURE_MAP_TIMES:
            feature_output_dir = (
                output_dir
                / "feature_maps"
                / f"leadtime_{leadtime}"
            )

            extract_feature_maps(
                model=model,
                dataset=test_dataset,
                normalize_target=normalize_target,
                wanted_times=FEATURE_MAP_TIMES,
                output_dir=feature_output_dir,
                device=device,
                latitudes=latitudes,
            )

        dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            # batch_size=s.batch_size,
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
            normalize_input=normalize_input,
            normalize_target=normalize_target,
            dataloader=dataloader,
            preds_store=lead_output,
            an_clim=datasets["y_clim"],
            log_monthly=LOG_MONTHLY,
            latitudes=latitudes,
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
        # extra_suffix_folder="264samples_randomsamples",
        extra_suffix_folder="",
        # extra_suffix_folder="264samples_consecutive",
        # extra_suffix_folder="NOAA_copy",
        train_subsamples=None,
        normalization_mode="channel",
    )

    print(f"Found {len(settings)} matching experiment(s).")

    if not settings:
        raise RuntimeError("No matching experiments found.")

    for s in settings:
        infer_setting(s)


if __name__ == "__main__":
    main()
