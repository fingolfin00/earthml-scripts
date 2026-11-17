from copy import deepcopy
from pathlib import Path
from typing import Literal

import gc

import numpy as np
import pandas as pd
import xarray as xr

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import lightning as L

from earthml import (
    Settings,
    Normalize,
    MonthlyNormalize,
    XarrayDataset,
    SplitDataModule,
    select_clim_for_time,
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

# Seasonal
EXP_NAME = "experiments"
# EXP_NAME = "experiment_new"

# Weather
# EXP_NAME = "weather_atmo"
# EXP_NAME = "weather_atmo_architecture"
# EXP_NAME = "weather_atmo_ablation_fixed_val"
# EXP_NAME = "weather_atmo_short_zero_vs_replicate_padding"

EXPERIMENTS_ROOT = Path(
    f"/Users/jacopodallaglio/ML/training/seasonal/{EXP_NAME}"
    # f"/work/cmcc/jd19424/ML/MLBC/experiments/{EXP_NAME}"
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

TIME_RANGE: tuple[str, str] | Literal[
    "train", "val", "test", "train_val", "full"
] = (
    "test"
    # ("2025-02-01T00:00:00", "2025-03-31T00:00:00")
)

WEIGHTS: Literal["best", "last"] = "best"

# Optional per-leadtime checkpoint override
# Leave empty to use the selected experiment's best/last checkpoint
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
    # "2025-03-08T00:00:00", # B1 t2m 72h
    # "2025-02-22T00:00:00", # B2 t2m 72h
]

SAVE_FEATURE_MAPS = True
SAVE_KERNEL_WEIGHTS = True  # One snapshot per loaded checkpoint

# Automatic selection of initialization times, per experiment and lead time
# Manual FEATURE_MAP_TIMES are included regardless of these settings
FEATURE_MAP_BEST_N = 20
FEATURE_MAP_WORST_N = 20

# bias ranks by reduction in absolute spatial bias; scc ranks by increasing atitude-weighted spatial correlation
# All metrics use PHYSICAL units
FEATURE_MAP_RANK_METRIC: Literal["rmse", "mae", "mse", "bias", "scc"] = "rmse"


# =============================================================================
# Feature map collector class
# =============================================================================

class FeatureMapCollector:
    def __init__(self, model):
        self.model = model

        self.handles: list = []
        self.features: dict[str, torch.Tensor] = {}
        self.attention: dict[str, dict[str, torch.Tensor]] = {}
        self._channel_logits: dict[str, torch.Tensor] = {}

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

        # Capture the actual CBAM gates
        # ChannelAttention calls its shared MLP exactly twice: once for spatial average pooling and once for spatial maximum pooling
        # The sum of those logits is passed through sigmoid
        for level, cbam in enumerate(self.model.encoder_cbam):
            if not (hasattr(cbam, "channel_att") and hasattr(cbam, "spatial_att")):
                continue  # nn.Identity() when CBAM is disabled

            name = f"encoder_cbam_{level}"

            def mlp_hook(module, inputs, output, name=name):
                previous = self._channel_logits.pop(name, None)
                if previous is None:
                    self._channel_logits[name] = output.detach()
                else:
                    gate = torch.sigmoid(previous + output.detach())
                    self.attention.setdefault(name, {})["channel"] = (
                        gate.float().cpu()
                    )

            # SpatialAttention uses conv -> norm -> sigmoid
            # Recording the normalization output gives the exact pre-sigmoid mask
            def spatial_hook(module, inputs, output, name=name):
                self.attention.setdefault(name, {})["spatial"] = torch.sigmoid(output.detach()).float().cpu()

            self.handles.append(cbam.channel_att.MLP.register_forward_hook(mlp_hook))
            self.handles.append(cbam.spatial_att.norm.register_forward_hook(spatial_hook))

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
        self.attention.clear()
        self._channel_logits.clear()

    def remove(self):
        for handle in self.handles:
            handle.remove()

        self.handles.clear()


def save_convolution_kernels(
    model: nn.Module,
    output_file: Path,
) -> None:
    """Save spatial and pointwise convolution weights once per checkpoint.

    Parameter names preserve their actual position in the architecture;
    arrays retain their PyTorch convolution dimensions (out,in/groups,H,W).
    """
    weights = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if name.endswith("weight") and parameter.ndim == 4
    }

    if not weights:
        raise RuntimeError("No 4D convolutional weights found")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weights": weights}, output_file)

    print(f"Saved {len(weights)} kernel tensors: {output_file}")


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


def target_sample_statistics(
    dataset: XarrayDataset,
    index: int,
    normalize_input: Normalize | MonthlyNormalize,
    normalize_target: Normalize | MonthlyNormalize,
) -> dict[str, torch.Tensor]:
    """Sample-level normalization kwargs, consistent with train.py:_test"""

    kwargs = {}
    if not normalize_target.uses_sample_stats:
        return kwargs

    x = dataset.x[index:index + 1].detach().float().cpu()
    mask = dataset.x_mask[index:index + 1].detach().cpu()

    mean, std = normalize_input.resolve_sample_mean_or_std(x, mask=mask)

    included = normalize_input.included_channels
    if included is None:
        raise ValueError("Input normalizer has no included_channels")

    positions = {channel: pos for pos, channel in enumerate(included)}
    selected = [positions[i] for i in dataset.target_to_input_channel_indices()]

    if normalize_target.mean_spec == "sample":
        kwargs["mean"] = mean[:, selected][0]

    if normalize_target.std_spec == "sample":
        kwargs["std"] = std[:, selected][0]

    return kwargs


def sample_climatology(
    clim_ds: xr.Dataset | None,
    var: str,
    init_time,
    clim_period,
    shape: tuple[int, int, int],
    sample_within_init: int,
    samples_per_init: int,
) -> torch.Tensor:
    """Return climatology as (C,H,W), using the sample's initialization time"""

    if clim_ds is None:
        raise ValueError(f"Climatology missing for {var!r} and anomaly target mode")

    selected = select_clim_for_time(
        clim_ds,
        np.asarray([np.datetime64(init_time, "ns")]),
        clim_period,
    )[var]

    lat_dim = selected.earthml.guessed_dims.latitude
    lon_dim = selected.earthml.guessed_dims.longitude
    time_dim = selected.earthml.guessed_dims.time

    if time_dim is not None and time_dim in selected.dims:
        if selected.sizes[time_dim] != 1:
            raise ValueError("Expected exactly one climatology time")
        selected = selected.isel({time_dim: 0}, drop=True)

    # For a physical ensemble with one sample per member, use matching climatological realization
    if "realization" in selected.dims:
        if samples_per_init > 1 and selected.sizes["realization"] == samples_per_init:
            selected = selected.isel(realization=sample_within_init, drop=True)
        else:
            selected = selected.mean("realization", skipna=True)

    other = [d for d in selected.dims if d not in (lat_dim, lon_dim)]

    if other:
        raise ValueError(f"Unexpected climatology dimensions for {var}: {other}")

    arr = torch.as_tensor(
        selected.transpose(lat_dim, lon_dim).to_numpy().copy(),
        dtype=torch.float32,
    ).unsqueeze(0)

    try:
        return torch.broadcast_to(arr, shape).clone()
    except RuntimeError as exc:
        raise ValueError(
            f"Climatology shape {tuple(arr.shape)} incompatible with target {shape}"
        ) from exc


def physical_fields_for_sample(
    dataset: XarrayDataset,
    sample_index: int,
    prediction_normalized: torch.Tensor,
    normalize_input: Normalize | MonthlyNormalize,
    normalize_target: Normalize | MonthlyNormalize,
    s: Settings,
    init_time: np.datetime64 | pd.Timestamp,
    fc_clim: xr.Dataset | None,
    an_clim: xr.Dataset | None,
) -> tuple[
    torch.Tensor,  # Physical forecast
    torch.Tensor,  # Corrected forecast
    torch.Tensor,  # Physical analysis
    torch.Tensor,  # Normalized target-space baseline
]:
    """
    Return physical (forecast, corrected, analysis, target-space baseline).
    Matches train.py's target definitions and _test reconstruction.
    The target-space baseline is also returned for the existing feature plots.
    """

    raw_target = dataset.y[sample_index].detach().float().cpu()
    raw_forecast = dataset.x[sample_index].detach().float().cpu()

    indices = list(dataset.target_to_input_channel_indices())

    fc = raw_forecast[indices]
    if fc.shape != raw_target.shape:
        raise ValueError(f"Forecast {fc.shape} and target {raw_target.shape} mismatch")

    kwargs = target_sample_statistics(dataset, sample_index, normalize_input, normalize_target)
    pred = normalize_target.inverse_tensor(
        prediction_normalized.detach().float().cpu(), **kwargs,
    ).detach().float().cpu()

    if s.target_mode in {"anomaly", "anomaly_residual", "anomaly_residual_realization"}:
        per_init_count = int(dataset.samples_per_init)

        member = sample_index % per_init_count
        shape = tuple(raw_target.shape)

        f_clim = sample_climatology(
            fc_clim, s.var_fc, init_time, s.clim_period,
            shape, member, per_init_count,
        )

        a_clim = sample_climatology(
            an_clim, s.var_an, init_time, s.clim_period,
            shape, member, per_init_count,
        )

    if s.target_mode == "analysis":
        baseline_t = fc
        baseline_phys, pred_phys, target_phys = fc, pred, raw_target

    elif s.target_mode in {"residual", "residual_realization"}:
        baseline_t = torch.zeros_like(raw_target)
        baseline_phys, pred_phys, target_phys = fc, fc + pred, fc + raw_target

    elif s.target_mode == "anomaly":
        baseline_t = fc + f_clim - a_clim
        baseline_phys = fc + f_clim
        pred_phys = pred + a_clim
        target_phys = raw_target + a_clim

    elif s.target_mode in {"anomaly_residual", "anomaly_residual_realization"}:
        baseline_t = f_clim - a_clim
        baseline_phys = fc + f_clim
        pred_phys = fc + a_clim + pred
        target_phys = fc + a_clim + raw_target

    else:
        raise ValueError(f"Unsupported target mode {s.target_mode!r}")

    baseline_norm = normalize_target(baseline_t, **kwargs).detach().float().cpu()

    return baseline_phys, pred_phys, target_phys, baseline_norm


def spatial_metric(field, reference, mask, latitudes, metric: str) -> float:
    field = field.float().cpu()
    reference = reference.float().cpu()
    valid = mask.bool().cpu() & torch.isfinite(field) & torch.isfinite(reference)

    weights = torch.cos(torch.deg2rad(latitudes.float().cpu())).clamp_min(0)[None, :, None]
    weights = torch.where(valid, weights, 0.0).expand_as(field)
    total = weights.sum()

    if not bool(total > 0):
        return float("nan")

    delta = torch.where(valid, field - reference, 0.0)
    bias = (weights * delta).sum() / total

    if metric == "bias":
        return float(bias.abs()) # Improvement means reduced bias magnitude
    if metric == "mae":
        return float((weights * delta.abs()).sum() / total)
    if metric == "mse":
        return float((weights * delta.square()).sum() / total)
    if metric == "rmse":
        return float(torch.sqrt((weights * delta.square()).sum() / total))
    if metric == "scc":
        # Area-weighted spatial pattern correlation, after subtracting each field's own spatial mean
        a = torch.where(valid, field, 0.0)
        b = torch.where(valid, reference, 0.0)

        ma = (weights * a).sum() / total
        mb = (weights * b).sum() / total

        ac = torch.where(valid, field - ma, 0.0)
        bc = torch.where(valid, reference - mb, 0.0)

        numerator = (weights * ac * bc).sum()
        denominator = torch.sqrt((weights * ac.square()).sum() *
                                 (weights * bc.square()).sum())
        if not bool(denominator > 1e-12):
            return float("nan")

        return float(numerator / denominator)

    raise ValueError(f"Unsupported FEATURE_MAP_RANK_METRIC: {metric!r}")


def select_feature_map_times(
    model: nn.Module,
    dataset: XarrayDataset,
    normalize_input: Normalize | MonthlyNormalize,
    normalize_target: Normalize | MonthlyNormalize,
    s: Settings,
    latitudes: torch.Tensor,
    device: torch.device,
    output_dir: Path,
    metric: Literal["rmse", "mae", "mse", "bias", "scc"],
    best_n: int,
    worst_n: int,
    fc_clim: xr.Dataset | None,
    an_clim: xr.Dataset | None,
) -> list[str]:
    """Rank forecast *initializations* by mean per-sample physical improvement"""

    if best_n < 0 or worst_n < 0:
        raise ValueError("Best/worst counts must be nonnegative")

    if best_n == 0 and worst_n == 0:
        return []

    if metric not in {"rmse", "mae", "mse", "bias", "scc"}:
        raise ValueError(f"Unknown comparison metric {metric!r}")

    times = pd.to_datetime(get_sample_times(dataset))
    if len(times) != len(dataset):
        raise ValueError("Sample/time mapping does not match dataset length")

    rows = []
    was_training = model.training
    model.eval()

    try:
        with torch.inference_mode():
            for index in range(len(dataset)):
                x, _, _, _ = dataset[index]
                pred = model(x.unsqueeze(0).to(device))[0]

                baseline, corrected, analysis, _ = physical_fields_for_sample(
                    dataset,
                    index,
                    pred,
                    normalize_input,
                    normalize_target,
                    s,
                    times[index],
                    fc_clim,
                    an_clim,
                )

                mask = dataset.y_mask[index].detach().cpu().bool()
                before = spatial_metric(baseline, analysis, mask, latitudes, metric)
                after = spatial_metric(corrected, analysis, mask, latitudes, metric)
                improvement = after - before if metric == "scc" else before - after

                rows.append({
                    "init_time": pd.Timestamp(times[index]),
                    "sample_index": index,
                    f"{metric}_baseline": before,
                    f"{metric}_corrected": after,
                    f"{metric}_improvement": improvement,
                })
    finally:
        if was_training:
            model.train()

    scores = pd.DataFrame(rows).dropna(subset=[f"{metric}_improvement"])
    if scores.empty:
        raise RuntimeError("No valid samples for ranking")

    ranked = (
        scores.groupby("init_time", as_index=False)
        .agg(
            n_samples=("sample_index", "size"),
            **{key: (key, "mean") for key in (f"{metric}_baseline", f"{metric}_corrected", f"{metric}_improvement")}
        )
        .sort_values(f"{metric}_improvement", ascending=False)
        .reset_index(drop=True)
    )

    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["selection"] = ""

    best = ranked.head(best_n) if best_n else ranked.iloc[:0]
    remaining = ranked.loc[~ranked["init_time"].isin(best["init_time"])]
    worst = remaining.tail(worst_n) if worst_n else remaining.iloc[:0]

    ranked.loc[ranked["init_time"].isin(best["init_time"]), "selection"] = "best"
    ranked.loc[ranked["init_time"].isin(worst["init_time"]), "selection"] = "worst"

    output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_dir / "feature_map_performance_samples.csv", index=False)
    ranked.to_csv(output_dir / "feature_map_performance_ranking.csv", index=False)

    print(f"Selected best/worst initializations by {metric} improvement (positive=better):")
    print(ranked.loc[ranked["selection"] != "", ["init_time", "selection", f"{metric}_improvement"]].to_string(index=False))

    return [str(t) for t in ranked.loc[ranked["selection"] != "", "init_time"]]


def extract_feature_maps(
    model: nn.Module,
    dataset: XarrayDataset,
    normalize_input: Normalize | MonthlyNormalize,
    normalize_target: Normalize | MonthlyNormalize,
    wanted_times: list[str],
    output_dir: Path,
    device: torch.device,
    latitudes: torch.Tensor,
    s: Settings,
    fc_clim: xr.Dataset | None,
    an_clim: xr.Dataset | None,
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
            matching_indices = np.flatnonzero(sample_times_ns == requested_time)

            if matching_indices.size == 0:
                print(f"WARNING: no test sample for {requested_time}")
                continue

            for sample_within_init, sample_index in enumerate(
                matching_indices
            ):
                sample_index = int(sample_index)

                # Transformed input/target used by the network
                x, y, _, _ = dataset[sample_index]

                # Add target mask
                target_mask = dataset.y_mask[sample_index].detach().cpu().bool()

                _, _, _, baseline_target_norm = physical_fields_for_sample(
                    dataset=dataset,
                    sample_index=sample_index,
                    prediction_normalized=y, # unused for baseline calculation
                    normalize_input=normalize_input,
                    normalize_target=normalize_target,
                    s=s,
                    init_time=requested_time,
                    fc_clim=fc_clim,
                    an_clim=an_clim,
                )

                if x.ndim != 3:
                    raise ValueError(f"Expected input tensor (C,H,W), got {tuple(x.shape)}")

                collector.clear()

                x_batch = x.unsqueeze(0).to(device)

                with torch.inference_mode():
                    prediction = model(x_batch)

                # Remove batch dimension from everything
                features = {
                    name: value[0].clone()
                    for name, value
                    in collector.features.items()
                }

                attention = {
                    name: {kind: tensor[0].clone() for kind, tensor in maps.items()}
                    for name, maps in collector.attention.items()
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
                    "sample_within_init": sample_within_init,
                    "init_time": str(pd.Timestamp(requested_time)),

                    "input": x.detach().float().cpu(),
                    "target": y.detach().float().cpu(),

                    "prediction": prediction[0].detach().float().cpu(),

                    "baseline_target_available": True,
                    "baseline_target_norm": baseline_target_norm.detach().float().cpu(),

                    "target_mask": target_mask,

                    "latitudes": latitudes.detach().float().cpu(),
                    "longitudes": torch.as_tensor(
                        dataset.target_ds[
                            dataset.target_ds.earthml.guessed_dims.longitude
                        ].values,
                        dtype=torch.float32,
                    ).cpu(),

                    "features": features,
                    "attention": attention,
                }

                time_label = pd.Timestamp(requested_time).strftime("%Y%m%dT%H%M%S")

                path = (
                    output_dir
                    / f"features_init_{time_label}_sample_{sample_within_init}.pt"
                )

                torch.save(
                    payload,
                    path,
                )

                print(f"Saved feature maps: {path}")

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
    # Keep in sync with train.py:train() dataset_kwargs
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


def make_normalizers(
    s: Settings,
    train_dataset: XarrayDataset,
    val_dataset: XarrayDataset | None,
):
    # Support only full normalization
    if s.normalization != "full":
        raise NotImplementedError("Monthly normalization has not yet been migrated in train.py")

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
        "latitudes": latitudes,
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
    s: Settings,
    leadtime: int,
) -> Path:
    override = CHECKPOINT_OVERRIDES.get(int(leadtime))

    if override is not None:
        checkpoint = override
    else:
        EXP_NAME = f"exp_{leadtime}_{s.leadtime_unit}"
        exp_dir = s.exp_dir / EXP_NAME

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


def infer_setting(s: Settings) -> None:
    if s.neural_ensemble:
        raise NotImplementedError("Select one neural member/checkpoint explicitly for ensemble experiments")

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

    if TIME_RANGE == "train":
        test_start, test_end = s.train_start, s.train_end
    elif TIME_RANGE == "val":
        test_start, test_end = s.val_start, s.val_end
    elif TIME_RANGE == "test":
        test_start, test_end = s.test_start, s.test_end
    elif TIME_RANGE == "train_val":
        test_start, test_end = s.train_start, s.val_end
    elif TIME_RANGE == "full":
        test_start, test_end = s.train_start, s.test_end
    elif isinstance(TIME_RANGE, tuple):
        test_start, test_end = TIME_RANGE
    else:
        raise ValueError(f"Invalid TIME_RANGE={TIME_RANGE!r}")

    output_dir = (
        s.exp_dir
        / "inference"
        / f"{test_start}_{test_end}"
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
        print(f"test period: {test_start} -> {test_end}")

        explicit_split = s.split_strategy == "explicit"
        train_end = (
            s.train_end
            if explicit_split
            else s.val_end
        )

        # Construct exactly the input channels used during training
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
            test_start=test_start,
            test_end=test_end,
            target_mode=s.target_mode,
            clim_period=s.clim_period,
            forecast_vars=forecast_vars,
            analysis_vars=[s.var_an],
            region=s.region,
            dataset_kwargs=dataset_kwargs,
            temporal_evolution_encoding=s.temporal_evolution_encoding,
            climatology_encoding=s.climatology_encoding,
            climatology_error_encoding=s.climatology_error_encoding,
            seasonal_encoding=s.seasonal_encoding and s.channel_representation != "init_period",
            ensemble_encoding=s.ensemble_encoding,
            spatial_encoding=s.spatial_encoding,
            orography_encoding=s.orography_encoding,
            orography_ds=orography_ds,
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
            datasets["val"],
        )

        train_dataset.sample_target_mean = s.sample_target_mean
        train_dataset.sample_target_std = s.sample_target_std
        test_dataset.sample_target_mean = "input"
        test_dataset.sample_target_std = "input"

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
            strict=True,
            **net_kwargs,
        ).to(device)

        del template_model

        # Save learned convolutional kernels once, not once per sample
        if SAVE_KERNEL_WEIGHTS:
            save_convolution_kernels(
                model,
                output_dir / "feature_maps" / f"leadtime_{leadtime}" / "kernels.pt",
            )

        # Save selected feature maps, using manual and/or automatically ranked times.
        if SAVE_FEATURE_MAPS:
            feature_output_dir = (
                output_dir
                / "feature_maps"
                / f"leadtime_{leadtime}"
            )
            wanted_times = list(FEATURE_MAP_TIMES or [])

            if FEATURE_MAP_BEST_N or FEATURE_MAP_WORST_N:
                wanted_times.extend(select_feature_map_times(
                    model=model,
                    dataset=test_dataset,
                    normalize_input=normalize_input,
                    normalize_target=normalize_target,
                    s=s,
                    latitudes=latitudes,
                    device=device,
                    output_dir=feature_output_dir,
                    metric=FEATURE_MAP_RANK_METRIC,
                    best_n=FEATURE_MAP_BEST_N,
                    worst_n=FEATURE_MAP_WORST_N,
                    fc_clim=datasets["x_clim"],
                    an_clim=datasets["y_clim"],
                ))
            wanted_times = list(dict.fromkeys(str(pd.Timestamp(t)) for t in wanted_times))

            if wanted_times:
                extract_feature_maps(
                    model=model,
                    dataset=test_dataset,
                    normalize_input=normalize_input,
                    normalize_target=normalize_target,
                    wanted_times=wanted_times,
                    output_dir=feature_output_dir,
                    device=device,
                    latitudes=latitudes,
                    s=s,
                    fc_clim=datasets["x_clim"],
                    an_clim=datasets["y_clim"],
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
            orography_ds,
            train_dataset.input_ds,
            train_dataset.target_ds,
            test_dataset.input_ds,
            test_dataset.target_ds,
            *( (datasets["val"].input_ds, datasets["val"].target_ds) if datasets["val"] is not None else () ),
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
        **FILTERS,
    )

    print(f"Found {len(settings)} matching experiment(s).")

    if not settings:
        raise RuntimeError("No matching experiments found.")

    for s in settings:
        infer_setting(s)


if __name__ == "__main__":
    main()
