from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from earthml import (
    get_experiment_configs, get_and_subset_datasets, ClimPeriod, LeadtimeUnit,
)
from earthml.metrics import (
    is_deterministic, is_probabilistic, get_metrics,
    build_metric_improvements, calculate_save_and_subset_climatologies,
    STD_REFERENCE_METRICS,
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

REGENERATE_PLOTS = True

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

LEADTIME = "6"
# LEADTIME = "72"

# Must match TIME_RANGE used by inference.py for this experiment
TIME_RANGE: tuple[str, str] | Literal[
    "train", "val", "test", "train_val", "full"
] = "test"
# TIME_RANGE = ("2025-02-01T00:00:00", "2025-03-31T00:00:00")

# -----------------------------------------------------------------------------
# Switches
# -----------------------------------------------------------------------------

ANALYZE_FEATURE_MAPS = False            # Spatial channel RMS/STD + individual channels
ANALYZE_CBAM = False                    # Spatial/channel gates and pre/post-CBAM features
ANALYZE_LAYER_STATISTICS = False        # Layer-level summaries and profiles
RUN_SAMPLE_COMPARISON = True            # Legacy pairwise similarity/difference analysis
PLOT_KERNEL_WEIGHTS = False             # Optional static weight analysis (not a sample map)
ANALYZE_CHANNEL_PARTICIPATION = False   # Channel-energy concentration versus forecast skill
PLOT_BATCH_SUMMARY_PANELS = False       # All ranked best/worst cases per layer and statistic
ANALYZE_SPATIAL_SUMMARY_SCATTERS = False # Spatial mean of channel RMS/STD vs physical improvement
PLOT_LAYER_PROFILE = False

# Reuse EarthML metric timeseries for each saved initialization.
ANALYZE_LIBRARY_METRICS = True

# -----------------------------------------------------------------------------
# Global settings
# -----------------------------------------------------------------------------

PLOT_GEO_COORDS = True  # Use saved lon/lat for spatial maps when available

# None = all saved feature layers or use e.g. ["encoder_raw_0", "encoder_cbam_2", "decoder_3"]
FEATURE_LAYERS: list[str] | None = None

FEATURE_SUMMARIES: tuple[Literal["rms", "std", "mean"], ...] = ("rms", "std")
FEATURE_CHANNEL_RANKING: Literal["rms", "std"] = "rms"
FEATURE_TOP_CHANNELS = 4

# Contact sheets: view every feature channel at each saved layer
PLOT_ALL_CHANNEL_MOSAICS = True
MOSAIC_COLUMNS = 8
MOSAIC_CHANNELS_PER_PAGE = 64  # Split very wide layers across pages
MOSAIC_SCALE: Literal["shared", "per_channel"] = "shared"
MOSAIC_QUANTILE = 0.995
MOSAIC_DPI = 180

# Batch summary panels: one figure per layer/statistic, B cases above W cases
# None uses FEATURE_LAYERS (or every saved layer when FEATURE_LAYERS is None)
PANEL_LAYERS: list[str] | None = None
PANEL_STATISTICS: tuple[Literal["rms", "std"], ...] = ("rms", "std")
PANEL_COLUMNS = 5                 # Number of cases per row
PANEL_ROWS_PER_PAGE: int | None = None  # None = all best/worst cases in one figure
PANEL_COLOR_QUANTILE = 0.995      # Shared across ALL samples/pages for each layer/statistic
PANEL_DPI = 180
PANEL_SHOW_GEO_TICKS = False      # Keep panels uncluttered; geo extent still used

# Scalar summaries of the same spatial RMS/STD maps used in the panels
SUMMARY_SCATTER_LAYERS: list[str] | None = None  # None -> PANEL_LAYERS / FEATURE_LAYERS
SUMMARY_SCATTER_STATISTICS: tuple[Literal["rms", "std"], ...] = ("rms", "std")
SUMMARY_SCATTER_AREA_WEIGHTED = False  # True -> cos(latitude) spatial average
SUMMARY_SCATTER_ANNOTATE = True
SUMMARY_SCATTER_FIGSIZE = (8, 6)

# -----------------------------------------------------------------------------
# EarthML metrics for all representation-vs-performance scatter plots
# -----------------------------------------------------------------------------

# Targets: "fc:<metric>", "mlfc:<metric>", "improvement:<metric>:<unit>"
# Additional target: "correction:predicted_rms" = RMSE(MLFC, FC)
# Units are the ImprovementUnit values supported by EarthML: Δ, %, normalized
SCATTER_Y_TARGETS = (
    "fc:rmse",
    "correction:predicted_rms",
    "mlfc:rmse",
    "improvement:rmse:Δ",
    "improvement:rmse:%",
    # "fc:mae",
    # "mlfc:mae",
    # "improvement:mae:Δ",
    # "improvement:scc:Δ",
)

METRIC_PERIOD_REFERENCE: Literal["init", "valid"] = "init"
METRIC_CLIM_PERIOD: ClimPeriod = ClimPeriod.MONTH
METRIC_CLIM_TIME_RANGE: Literal["train", "val", "test", "train_val", "full"] = "train_val"
METRIC_INTERPOLATE_ANALYSIS = False
METRIC_OROGRAPHY_PATH: Path | None = Path(
    "/work/cmcc/jd19424/ML/MLBC/data/orography/era5_orography.zarr"
)
# Set True for metrics requiring reference climatologies (e.g., anomaly skill).
METRIC_LOAD_CLIMATOLOGIES = False
METRIC_RECALCULATE_CLIMATOLOGY = False
# Select inference outputs saved at inference/<period>/test_corrected.zarr.
METRIC_INFERENCE_OUTPUT = True

ACTIVATION_ZERO_THRESHOLD = 1e-6

# -----------------------------------------------------------------------------
# Independent analysis
# -----------------------------------------------------------------------------

# The ranking CSV is created by inference.py, using FEATURE_MAP_RANK_METRIC in PHYSICAL units.
# Only samples already extracted by inference can be plotted
ANALYSIS_SELECTION: Literal["ranked", "manual", "both"] = "ranked"
ANALYSIS_BEST_N = 20
ANALYSIS_WORST_N = 20
ANALYSIS_MANUAL_TIMES: list[str] = [
    # "2025-03-08T00:00:00",
]

ANALYSIS_SAMPLE_WITHIN_INIT: int | None = 0
# None -> analyze every saved realization/sample for each initialization
# 0 -> use only features_init_*_sample_0.pt (usual single-sample case)

# -----------------------------------------------------------------------------
# Comparison
# -----------------------------------------------------------------------------

SAMPLE_SELECTION: Literal["manual", "ranked"] = "manual"
# "manual": SAMPLE_A and SAMPLE_B below
# "ranked": choose samples from inference's performance ranking CSV

RANKED_SAMPLE_A: Literal["best", "worst"] = "best"
RANKED_SAMPLE_B: Literal["best", "worst"] = "worst"
RANKED_INDEX_A = 5  # 1 = most improved for "best", least improved for "worst"
RANKED_INDEX_B = 5

SAMPLE_A = {"W3": "2025-02-02T00:00:00"}
SAMPLE_B = {"B6": "2025-02-13T00:00:00"}

SAMPLE_WITHIN_INIT_A = 0
SAMPLE_WITHIN_INIT_B = 0

SPATIAL_DIFFERENCE_LAYERS = [
    "encoder_raw_0",
    "encoder_cbam_2",
    "encoder_cbam_4",
    "decoder_0",
    "decoder_2",
    "decoder_3",
    "output",
]

# -----------------------------------------------------------------------------
# Participation channels (saved samples only)
# -----------------------------------------------------------------------------

PARTICIPATION_LAYERS = ["decoder_3"]  # or ["encoder_cbam_4", "decoder_3"]
PARTICIPATION_TOP_N = 5
# "fraction": N_eff / number of channels, useful across layers of different width
# "count": raw effective number of channels
PARTICIPATION_X_AXIS: Literal["count", "fraction"] = "count"
PARTICIPATION_ANNOTATE_POINTS = True
PARTICIPATION_FIGSIZE = (8, 6)


# =============================================================================
# Feature utilities
# =============================================================================

def summarize_feature_map(
    feature: torch.Tensor,
) -> dict[str, torch.Tensor | tuple[int, ...]]:
    """
    Summarize a feature tensor with shape (C,H,W)
    across channels.
    """
    if feature.ndim != 3:
        raise ValueError(
            f"Expected (C,H,W), got {tuple(feature.shape)}"
        )

    return {
        "shape": tuple(feature.shape),

        "min": feature.amin(dim=0),

        "max": feature.amax(dim=0),

        "mean": feature.mean(dim=0),

        "std": feature.std(
            dim=0,
            unbiased=False,
        ),

        "rms": torch.sqrt(
            torch.mean(
                feature**2,
                dim=0,
            )
        ),
    }


def feature_similarity(
    a: torch.Tensor,
    b: torch.Tensor,
) -> dict[str, float]:
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"{tuple(a.shape)} != {tuple(b.shape)}"
        )

    a_flat = a.flatten().float()
    b_flat = b.flatten().float()

    cosine = F.cosine_similarity(
        a_flat,
        b_flat,
        dim=0,
    )

    a_centered = a_flat - a_flat.mean()
    b_centered = b_flat - b_flat.mean()

    correlation = F.cosine_similarity(
        a_centered,
        b_centered,
        dim=0,
    )

    difference = a_flat - b_flat

    return {
        "cosine": cosine.item(),

        "correlation": correlation.item(),

        "mae": (
            difference
            .abs()
            .mean()
            .item()
        ),

        "rmse": torch.sqrt(
            torch.mean(
                difference**2
            )
        ).item(),

        "normalized_distance": (
            normalized_feature_distance(
                a,
                b,
            )
        ),
    }


def normalized_feature_distance(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:
    difference_rms = torch.sqrt(
        torch.mean(
            (a - b) ** 2
        )
    )

    reference_rms = 0.5 * (
        torch.sqrt(
            torch.mean(a**2)
        )
        + torch.sqrt(
            torch.mean(b**2)
        )
    )

    return (
        difference_rms
        / reference_rms.clamp_min(1e-12)
    ).item()


def channel_correlations(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Spatial Pearson correlation for each feature channel.

    Input:
        a, b: (C,H,W)

    Output:
        (C,)
    """
    if a.shape != b.shape:
        raise ValueError(
            f"{tuple(a.shape)} != {tuple(b.shape)}"
        )

    if a.ndim != 3:
        raise ValueError(
            f"Expected (C,H,W), got {tuple(a.shape)}"
        )

    a = a.flatten(1).float()
    b = b.flatten(1).float()

    a = a - a.mean(
        dim=1,
        keepdim=True,
    )

    b = b - b.mean(
        dim=1,
        keepdim=True,
    )

    numerator = torch.sum(
        a * b,
        dim=1,
    )

    denominator = (
        torch.linalg.vector_norm(
            a,
            dim=1,
        )
        * torch.linalg.vector_norm(
            b,
            dim=1,
        )
    )

    valid = denominator > 1e-12

    correlation = torch.full_like(
        numerator,
        torch.nan,
    )

    correlation[valid] = (
        numerator[valid]
        / denominator[valid]
    )

    return correlation


def feature_difference_rms(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Spatial RMS difference across channels.

    Input:
        (C,H,W)

    Output:
        (H,W)
    """
    return torch.sqrt(
        torch.mean(
            (a - b) ** 2,
            dim=0,
        )
    )


# =============================================================================
# Geographic plotting utilities
# =============================================================================

def _sample_plot_coords(
    sample: dict,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return approximate feature-grid cell edges (lon, lat).

    Coarser U-Net activations are represented across the same geographic
    domain as the full-resolution input. Their coordinates are approximate
    since pooling/upsampling changes effective receptive-field centers.
    """
    if not PLOT_GEO_COORDS or "longitudes" not in sample or "latitudes" not in sample:
        return None

    lon = np.asarray(sample["longitudes"], dtype=float).reshape(-1)
    lat = np.asarray(sample["latitudes"], dtype=float).reshape(-1)
    if min(len(lon), len(lat)) < 2 or not np.isfinite(lon).all() or not np.isfinite(lat).all():
        return None
    if not np.all(np.diff(lon) > 0) and not np.all(np.diff(lon) < 0):
        raise ValueError("Longitude coordinates must be monotonic for geographic plotting")
    if not np.all(np.diff(lat) > 0) and not np.all(np.diff(lat) < 0):
        raise ValueError("Latitude coordinates must be monotonic for geographic plotting")

    def edges(c: np.ndarray, n: int) -> np.ndarray:
        left = c[0] - (c[1] - c[0]) / 2
        right = c[-1] + (c[-1] - c[-2]) / 2
        return np.linspace(left, right, n + 1)

    h, w = shape
    return edges(lon, w), edges(lat, h)


def plot_spatial_field(
    ax,
    data: np.ndarray,
    sample: dict,
    **kwargs,
):
    """Plot (H,W) with geographic coordinates, or a labeled pixel fallback."""
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D spatial field, got {data.shape}")
    coords = _sample_plot_coords(sample, data.shape)
    if coords is not None:
        lon_edges, lat_edges = coords
        im = ax.pcolormesh(lon_edges, lat_edges, data, shading="flat", **kwargs)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_aspect("auto")
    else:
        im = ax.imshow(data, origin="lower", aspect="auto", **kwargs)
        ax.set_xlabel("Grid column (longitude index)")
        ax.set_ylabel("Grid row (latitude index)")
    return im


def feature_file_path(
    root: Path,
    time: str,
    sample_within_init: int,
) -> Path:
    time_label = (
        pd.Timestamp(time)
        .strftime("%Y%m%dT%H%M%S")
    )

    return (
        root
        / (
            f"features_init_{time_label}"
            f"_sample_{sample_within_init}.pt"
        )
    )


def load_feature_sample(
    root: Path,
    time: str,
    sample_within_init: int = 0,
) -> dict:
    path = feature_file_path(
        root,
        time,
        sample_within_init,
    )

    if not path.exists():
        raise FileNotFoundError(path)

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required_keys = {
        "input",
        "target",
        "prediction",
        "baseline_target_norm",
        "target_mask",
        "latitudes",
        "features",
    }

    missing = required_keys - payload.keys()

    if missing:
        raise KeyError(
            f"{path.name} is missing keys {sorted(missing)}. "
            "Regenerate feature files with the updated inference script."
        )

    print(
        f"Loaded {path.name}: "
        f"sample_index={payload['sample_index']}, "
        f"init={payload['init_time']}"
    )

    return payload


def compare_feature_layers(
    sample_a: dict,
    sample_b: dict,
) -> pd.DataFrame:
    features_a = sample_a["features"]
    features_b = sample_b["features"]

    if features_a.keys() != features_b.keys():
        raise ValueError(
            "Feature dictionaries contain "
            "different layers."
        )

    rows = []

    def add_row(
        name: str,
        a: torch.Tensor,
        b: torch.Tensor,
    ):
        stats = feature_similarity(
            a,
            b,
        )

        rows.append({
            "layer": name,
            "channels": a.shape[0],
            "height": a.shape[-2],
            "width": a.shape[-1],
            **stats,
        })

    add_row(
        "input",
        sample_a["input"],
        sample_b["input"],
    )

    add_row(
        "baseline",
        sample_a["baseline_target_norm"],
        sample_b["baseline_target_norm"],
    )

    for name in features_a:
        add_row(
            name,
            features_a[name],
            features_b[name],
        )

    add_row(
        "prediction",
        sample_a["prediction"],
        sample_b["prediction"],
    )

    add_row(
        "target",
        sample_a["target"],
        sample_b["target"],
    )

    return pd.DataFrame(rows)


def prediction_target_metrics(
    sample: dict,
) -> dict[str, float]:
    prediction = sample["prediction"].float()
    target = sample["target"].float()
    mask = sample["target_mask"].bool()
    latitudes = sample["latitudes"].float()

    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: "
            f"{tuple(prediction.shape)} != "
            f"{tuple(target.shape)}"
        )

    difference = prediction - target

    valid = (
        mask
        & torch.isfinite(prediction)
        & torch.isfinite(target)
    )

    mae = (
        difference[valid]
        .abs()
        .mean()
        .item()
    )

    rmse = geo_weighted_rmse(
        prediction,
        target,
        mask,
        latitudes,
    )

    p = prediction[valid]
    t = target[valid]

    p = p - p.mean()
    t = t - t.mean()

    denominator = (
        torch.linalg.vector_norm(p)
        * torch.linalg.vector_norm(t)
    )

    correlation = (
        (p * t).sum()
        / denominator.clamp_min(1e-12)
    ).item()

    return {
        "mae": mae,
        "rmse": rmse,
        "correlation": correlation,
    }


def plot_layer_similarity(
    comparison: pd.DataFrame,
    output_file: Path,
):
    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    x = np.arange(len(comparison))

    ax.plot(
        x,
        comparison["cosine"],
        marker="o",
        label="Cosine similarity",
    )

    ax.plot(
        x,
        comparison["correlation"],
        marker="o",
        label="Correlation",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        comparison["layer"],
        rotation=45,
        ha="right",
    )

    ax.set_ylabel("Similarity")
    ax.set_ylim(-1.0, 1.02)

    ax.axhline(
        0,
        linewidth=0.8,
    )

    ax.legend()

    ax.set_title(
        "Representation similarity through network"
    )

    fig.tight_layout()

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_layer_distance(
    comparison: pd.DataFrame,
    output_file: Path,
):
    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    x = np.arange(len(comparison))

    ax.plot(
        x,
        comparison["normalized_distance"],
        marker="o",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        comparison["layer"],
        rotation=45,
        ha="right",
    )

    ax.set_ylabel(
        "Normalized representation distance"
    )

    ax.set_title(
        f"Sample {list(SAMPLE_A.keys())[0]} - sample {list(SAMPLE_B.keys())[0]} representation distance through network"
    )

    fig.tight_layout()

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def build_encoder_cbam_comparison(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    level = 0

    while True:
        raw_name = f"encoder_raw_{level}"
        cbam_name = f"encoder_cbam_{level}"

        raw = comparison[
            comparison["layer"] == raw_name
        ]

        cbam = comparison[
            comparison["layer"] == cbam_name
        ]

        if raw.empty and cbam.empty:
            break

        if raw.empty or cbam.empty:
            raise RuntimeError(
                f"Incomplete encoder level {level}"
            )

        rows.append({
            "level": level,

            "raw_correlation": float(
                raw.iloc[0]["correlation"]
            ),

            "cbam_correlation": float(
                cbam.iloc[0]["correlation"]
            ),

            "raw_cosine": float(
                raw.iloc[0]["cosine"]
            ),

            "cbam_cosine": float(
                cbam.iloc[0]["cosine"]
            ),
        })

        level += 1

    return pd.DataFrame(rows)


def plot_raw_vs_cbam(
    encoder_df: pd.DataFrame,
    output_file: Path,
):
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        encoder_df["level"],
        encoder_df["raw_correlation"],
        marker="o",
        label="Raw encoder",
    )

    ax.plot(
        encoder_df["level"],
        encoder_df["cbam_correlation"],
        marker="o",
        label="After CBAM",
    )

    ax.set_xlabel("Encoder level")
    ax.set_ylabel(f"Sample {list(SAMPLE_A.keys())[0]} - sample {list(SAMPLE_B.keys())[0]} spatial-feature correlation")

    ax.set_xticks(
        encoder_df["level"]
    )

    ax.set_ylim(-1, 1.02)

    ax.legend()

    ax.set_title(
        "Effect of CBAM on sample distinguishability"
    )

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def collect_channel_correlations(
    sample_a: dict,
    sample_b: dict,
) -> dict[str, np.ndarray]:
    result = {}

    for name, a in sample_a["features"].items():
        b = sample_b["features"][name]

        corr = channel_correlations(
            a,
            b,
        )

        result[name] = (
            corr
            .detach()
            .cpu()
            .numpy()
        )

    return result

def plot_channel_correlation_distributions(
    correlations: dict[str, np.ndarray],
    output_file: Path,
):
    names = list(correlations)

    values = [
        correlations[name][
            np.isfinite(correlations[name])
        ]
        for name in names
    ]

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    ax.boxplot(
        values,
        tick_labels=names,
        showfliers=False,
    )

    ax.axhline(
        1,
        linewidth=0.8,
    )

    ax.axhline(
        0,
        linewidth=0.8,
    )

    ax.set_ylim(-1.05, 1.05)

    ax.set_ylabel(
        "Per-channel spatial correlation"
    )

    ax.set_xticklabels(
        names,
        rotation=45,
        ha="right",
    )

    ax.set_title(
        "Distribution of sample A - sample B channel similarities"
    )

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_feature_difference_map(
    a: torch.Tensor,
    b: torch.Tensor,
    layer_name: str,
    output_file: Path,
    sample: dict,
):
    difference = (
        feature_difference_rms(
            a,
            b,
        )
        .detach()
        .cpu()
        .numpy()
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    image = plot_spatial_field(ax, difference, sample)

    ax.set_title(
        f"{layer_name}: "
        "RMS difference between samples"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="RMS feature difference",
    )

    fig.tight_layout()

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_selected_feature_differences(
    sample_a: dict,
    sample_b: dict,
    layers: list[str],
    output_dir: Path,
):
    features_a = sample_a["features"]
    features_b = sample_b["features"]

    for layer in layers:
        if layer not in features_a:
            print(
                f"WARNING: layer not found: {layer}"
            )
            continue

        plot_feature_difference_map(
            features_a[layer],
            features_b[layer],
            layer,
            output_dir
            / f"difference_{layer}.png",
            sample_a,
        )


def plot_tensor_difference(
    a: torch.Tensor,
    b: torch.Tensor,
    title: str,
    output_file: Path,
    sample: dict,
):
    if a.shape != b.shape:
        raise ValueError(
            f"{tuple(a.shape)} != {tuple(b.shape)}"
        )

    difference = (
        a - b
    )

    if difference.ndim == 3:
        if difference.shape[0] == 1:
            difference = difference[0]
        else:
            difference = torch.sqrt(
                torch.mean(
                    difference**2,
                    dim=0,
                )
            )

    data = (
        difference
        .detach()
        .cpu()
        .numpy()
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    image = plot_spatial_field(ax, data, sample)

    ax.set_title(title)

    fig.colorbar(
        image,
        ax=ax,
    )

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_feature_channel_pair(
    a: torch.Tensor,
    b: torch.Tensor,
    channel: int,
    layer_name: str,
    correlation: float,
    output_file: Path,
    sample: dict,
):
    a_map = (
        a[channel]
        .detach()
        .cpu()
        .numpy()
    )

    b_map = (
        b[channel]
        .detach()
        .cpu()
        .numpy()
    )

    difference = (
        a_map - b_map
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4),
    )

    vmin = min(
        np.nanmin(a_map),
        np.nanmin(b_map),
    )

    vmax = max(
        np.nanmax(a_map),
        np.nanmax(b_map),
    )

    im0 = plot_spatial_field(axes[0], a_map, sample, vmin=vmin, vmax=vmax)

    axes[0].set_title(f"Sample {list(SAMPLE_A.keys())[0]}")

    plot_spatial_field(axes[1], b_map, sample, vmin=vmin, vmax=vmax)

    axes[1].set_title(f"Sample {list(SAMPLE_B.keys())[0]}")

    diff_limit = np.nanmax(
        np.abs(difference)
    )

    im2 = plot_spatial_field(axes[2], difference, sample, vmin=-diff_limit, vmax=diff_limit)

    axes[2].set_title("A - B")

    fig.colorbar(
        im0,
        ax=axes[:2],
        shrink=0.8,
        label="Activation",
    )

    fig.colorbar(
        im2,
        ax=axes[2],
        shrink=0.8,
        label="Activation difference",
    )

    fig.suptitle(
        f"{layer_name} channel {channel} "
        f"(corr={correlation:.3f})"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_discriminative_channels(
    sample_a: dict,
    sample_b: dict,
    layers: list[str],
    output_dir: Path,
    n_channels: int = 5,
):
    for layer in layers:
        a = sample_a["features"][layer]
        b = sample_b["features"][layer]

        selected = (
            most_discriminative_channels(
                a,
                b,
                n=n_channels,
            )
        )

        layer_dir = (
            output_dir
            / layer
        )

        layer_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for channel, correlation in selected:
            plot_feature_channel_pair(
                a,
                b,
                channel,
                layer,
                correlation,
                layer_dir
                / f"channel_{channel:04d}.png",
                sample_a,
            )


def most_discriminative_channels(
    a: torch.Tensor,
    b: torch.Tensor,
    n: int = 5,
) -> list[tuple[int, float]]:
    correlations = channel_correlations(
        a,
        b,
    )

    valid = torch.isfinite(
        correlations
    )

    indices = torch.arange(
        correlations.shape[0]
    )[valid]

    values = correlations[valid]

    order = torch.argsort(
        values
    )

    n = min(
        n,
        len(order),
    )

    return [
        (
            int(indices[i]),
            float(values[i]),
        )
        for i in order[:n]
    ]


def geo_weighted_rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    latitudes: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """
    Cosine-latitude weighted spatial RMSE.

    Expected:
        prediction: (C,H,W)
        target:     (C,H,W)
        mask:       (C,H,W)
        latitudes:  (H,)
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: "
            f"{tuple(prediction.shape)} != "
            f"{tuple(target.shape)}"
        )

    if mask.shape != target.shape:
        raise ValueError(
            f"Mask/target shape mismatch: "
            f"{tuple(mask.shape)} != "
            f"{tuple(target.shape)}"
        )

    if latitudes.ndim != 1:
        raise ValueError(
            f"Expected latitudes (H,), "
            f"got {tuple(latitudes.shape)}"
        )

    if latitudes.shape[0] != target.shape[-2]:
        raise ValueError(
            f"Latitude count {latitudes.shape[0]} "
            f"does not match field height "
            f"{target.shape[-2]}"
        )

    prediction = prediction.float()
    target = target.float()
    mask = mask.bool()
    latitudes = latitudes.float()

    # Area weighting for a regular lat/lon grid.
    latitude_weights = torch.cos(
        torch.deg2rad(latitudes)
    )

    # (H,) -> (1,H,1), broadcasts over channels/lon.
    weights = latitude_weights[
        None,
        :,
        None,
    ].expand_as(target)

    valid = (
        mask
        & torch.isfinite(prediction)
        & torch.isfinite(target)
    )

    weights = torch.where(
        valid,
        weights,
        torch.zeros_like(weights),
    )

    squared_error = torch.where(
        valid,
        (prediction - target) ** 2,
        torch.zeros_like(target),
    )

    weighted_mse = (
        (squared_error * weights).sum()
        / weights.sum().clamp_min(eps)
    )

    return torch.sqrt(
        weighted_mse
    ).item()


def normalized_rmse_improvement(
    sample: dict,
) -> dict[str, float]:
    target = sample["target"].float()
    prediction = sample["prediction"].float()

    baseline = (
        sample["baseline_target_norm"]
        .float()
    )

    mask = (
        sample["target_mask"]
        .bool()
    )

    latitudes = (
        sample["latitudes"]
        .float()
    )

    baseline_rmse = geo_weighted_rmse(
        baseline,
        target,
        mask,
        latitudes,
    )

    ml_rmse = geo_weighted_rmse(
        prediction,
        target,
        mask,
        latitudes,
    )

    improvement = (
        baseline_rmse
        - ml_rmse
    )

    relative_improvement = (
        improvement
        / max(baseline_rmse, 1e-12)
    )

    return {
        "baseline_rmse": baseline_rmse,
        "ml_rmse": ml_rmse,
        "rmse_improvement": improvement,
        "relative_improvement": (
            relative_improvement
        ),
    }


def channel_correlation_summary(
    correlations: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []

    for name, values in correlations.items():
        finite = values[
            np.isfinite(values)
        ]

        if finite.size == 0:
            rows.append({
                "layer": name,
                "n_channels": len(values),
                "n_valid_channels": 0,
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "min": np.nan,
                "q05": np.nan,
                "q25": np.nan,
                "q75": np.nan,
                "q95": np.nan,
                "max": np.nan,
            })
            continue

        rows.append({
            "layer": name,
            "n_channels": len(values),
            "n_valid_channels": len(finite),
            "mean": np.mean(finite),
            "median": np.median(finite),
            "std": np.std(finite),
            "min": np.min(finite),
            "q05": np.quantile(
                finite,
                0.05,
            ),
            "q25": np.quantile(
                finite,
                0.25,
            ),
            "q75": np.quantile(
                finite,
                0.75,
            ),
            "q95": np.quantile(
                finite,
                0.95,
            ),
            "max": np.max(finite),
        })

    return pd.DataFrame(rows)



# =============================================================================
# Kernel analysis
# =============================================================================

def plot_kernel_weights(
    kernel_file: Path,
    output_dir: Path,
    max_filters: int = 16,
) -> None:
    """Plot learned depthwise/spatial kernels and pointwise mixing matrices."""
    import math
    import re

    if not kernel_file.exists():
        print(f"WARNING: no kernel snapshot at {kernel_file}")
        return
    payload = torch.load(kernel_file, map_location="cpu", weights_only=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for name, weights in payload["weights"].items():
        w = weights.float().numpy()
        out_channels, in_channels, kh, kw = w.shape

        rows.append({
            "parameter": name, "output_filters": out_channels,
            "input_channels_per_group": in_channels,
            "kernel_height": kh, "kernel_width": kw,
            "mean": float(w.mean()), "std": float(w.std()),
            "rms": float(np.sqrt(np.mean(w**2))),
        })

        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)

        if kh > 1 or kw > 1:
            # Depthwise kernels normally have in_channels_per_group = 1
            # For other kernels show the first input-channel slice and label it
            count = min(max_filters, out_channels)
            cols = min(4, count)
            nrows = math.ceil(count / cols)

            fig, axes = plt.subplots(nrows, cols, figsize=(3 * cols, 3 * nrows), squeeze=False)

            vmax = max(float(np.max(np.abs(w[:count, 0]))), 1e-12)
            for j, ax in enumerate(axes.ravel()):
                if j < count:
                    ax.imshow(w[j, 0], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                    ax.set_title(f"filter {j}")
                ax.axis("off")

            fig.suptitle(f"{name}  {tuple(w.shape)}")
            fig.tight_layout()

            fig.savefig(output_dir / f"{safe_name}.png", dpi=180)
            plt.close(fig)
        else:
            # Pointwise kernels are CHANNEL-MIXING matrices, not spatial masks
            matrix = w[:, :, 0, 0]

            fig, ax = plt.subplots(figsize=(9, 6))

            vmax = max(float(np.max(np.abs(matrix))), 1e-12)

            im = ax.imshow(
                matrix,
                aspect="auto",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )

            ax.set_xlabel("Input channels")
            ax.set_ylabel("Output channels")
            ax.set_title(name)

            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(output_dir / f"{safe_name}.png", dpi=180)
            plt.close(fig)

    pd.DataFrame(rows).to_csv(output_dir / "kernel_statistics.csv", index=False)

# =============================================================================
# CBAM attention map analysis
# =============================================================================

def plot_attention_comparison(
    sample_a: dict,
    sample_b: dict,
    output_dir: Path,
) -> None:
    """Compare per-channel and spatial CBAM gates for two forecast samples."""
    import re

    att_a = sample_a.get("attention", {})
    att_b = sample_b.get("attention", {})

    if not att_a or not att_b:
        print("WARNING: missing CBAM masks. Regenerate samples with updated inference script.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in sorted(att_a.keys() & att_b.keys()):
        a = att_a[name]
        b = att_b[name]

        if "channel" not in a or "channel" not in b or "spatial" not in a or "spatial" not in b:
            print(f"WARNING: incomplete attention masks at {name}")
            continue

        ca = a["channel"].reshape(-1).numpy()
        cb = b["channel"].reshape(-1).numpy()
        sa = a["spatial"].squeeze().numpy()
        sb = b["spatial"].squeeze().numpy()

        if ca.shape != cb.shape or sa.shape != sb.shape:
            raise ValueError(f"Attention shape mismatch at {name}")

        rows.append({
            "level": name, "channels": ca.size,
            "channel_mean_A": ca.mean(), "channel_mean_B": cb.mean(),
            "channel_mean_absolute_difference": np.abs(ca - cb).mean(),
            "spatial_mean_A": sa.mean(), "spatial_mean_B": sb.mean(),
            "spatial_mean_absolute_difference": np.abs(sa - sb).mean(),
        })
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, data, title in zip(axes, (sa, sb, sa - sb), ("Spatial gate A", "Spatial gate B", "Gate A - B")):
            if title == "Gate A - B":
                im = plot_spatial_field(ax, data, sample_a, cmap="RdBu_r", vmin=-1, vmax=1)
            else:
                im = plot_spatial_field(ax, data, sample_a, cmap="viridis", vmin=0, vmax=1)
            ax.set_title(title)
            fig.colorbar(im, ax=ax, shrink=0.8)

        fig.suptitle(name)
        fig.tight_layout()
        fig.savefig(output_dir / f"{safe_name}_spatial.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 4))
        idx = np.arange(ca.size)

        ax.plot(idx, ca, linewidth=1, label="Sample A")
        ax.plot(idx, cb, linewidth=1, label="Sample B", alpha=0.8)

        ax.set_ylim(0, 1)
        ax.set_xlabel("Learned feature-channel index")
        ax.set_ylabel("Channel attention gate")
        ax.set_title(name)
        ax.legend()

        fig.tight_layout()
        fig.savefig(output_dir / f"{safe_name}_channel.png", dpi=200)
        plt.close(fig)

        pd.DataFrame({"channel": idx, "attention_A": ca, "attention_B": cb}).to_csv(
            output_dir / f"{safe_name}_channel.csv", index=False
        )

    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "attention_summary.csv", index=False)


def resolve_inference_range(s) -> tuple[str, str]:
    """Match the date strings and output-directory naming from inference.py."""
    ranges = {
        "train": (s.train_start, s.train_end),
        "val": (s.val_start, s.val_end),
        "test": (s.test_start, s.test_end),
        "train_val": (s.train_start, s.val_end),
        "full": (s.train_start, s.test_end),
    }
    if isinstance(TIME_RANGE, tuple):
        if len(TIME_RANGE) != 2:
            raise ValueError("TIME_RANGE tuple must contain start and end")
        return TIME_RANGE

    if TIME_RANGE not in ranges:
        raise ValueError(f"Invalid TIME_RANGE: {TIME_RANGE!r}")

    start, end = ranges[TIME_RANGE]
    if start is None or end is None:
        raise ValueError(f"Missing saved dates for TIME_RANGE={TIME_RANGE!r}")

    return str(start), str(end)


def choose_ranked_sample(
    feature_root_dir: Path,
    selection: Literal["best", "worst"],
    index: int,
) -> dict[str, str]:
    """Select the Nth best/worst initialization from inference ranking."""
    if index < 1:
        raise ValueError("Ranked sample index must be >= 1")

    csv_path = feature_root_dir / "feature_map_performance_ranking.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Ranking file missing: {csv_path}. "
            "Run inference with FEATURE_MAP_BEST_N / FEATURE_MAP_WORST_N enabled."
        )

    ranking = pd.read_csv(csv_path)
    required = {"init_time", "selection", "rank"}

    if not required.issubset(ranking.columns):
        raise ValueError(f"Ranking CSV missing columns: {required - set(ranking.columns)}")

    selected = ranking.loc[ranking["selection"] == selection].copy()
    if selected.empty:
        raise ValueError(f"No {selection!r} samples in {csv_path}")

    # The ranking CSV is sorted best-first
    # Reverse for worst so W1 is the strongest deterioration, not the least bad of the worst N
    selected = selected.sort_values("rank", ascending=(selection == "best"))
    if index > len(selected):
        raise ValueError(f"Requested {selection} #{index}, only {len(selected)} available")

    row = selected.iloc[index - 1]
    label = f"{'B' if selection == 'best' else 'W'}{index}"

    return {label: str(pd.Timestamp(row["init_time"]))}


def run_sample_comparison() -> None:
    # Comparison mode, the main workflow is single-sample
    global SAMPLE_A, SAMPLE_B

    settings = get_experiment_configs(
        EXPERIMENTS_ROOT,
        var_fc=VARIABLES,
        region_name=REGIONS,
        **FILTERS,
    )

    print(f"Found {len(settings)} matching experiment(s).")

    if not settings:
        raise RuntimeError("No matching experiments found.")

    manual_sample_a = SAMPLE_A.copy()
    manual_sample_b = SAMPLE_B.copy()

    for s in settings:
        inference_start, inference_end = resolve_inference_range(s)
        feature_root_dir = (
            s.exp_dir
            / "inference"
            / f"{inference_start}_{inference_end}"
            / "feature_maps"
            / f"leadtime_{LEADTIME}"
        )
        if not feature_root_dir.exists():
            raise FileNotFoundError(
                f"Feature directory not found: {feature_root_dir}. "
                "Check TIME_RANGE and LEADTIME against inference.py."
            )

        if SAMPLE_SELECTION == "ranked":
            SAMPLE_A = choose_ranked_sample(feature_root_dir, RANKED_SAMPLE_A, RANKED_INDEX_A)
            SAMPLE_B = choose_ranked_sample(feature_root_dir, RANKED_SAMPLE_B, RANKED_INDEX_B)
        elif SAMPLE_SELECTION == "manual":
            SAMPLE_A = manual_sample_a
            SAMPLE_B = manual_sample_b
        else:
            raise ValueError(f"Unknown SAMPLE_SELECTION: {SAMPLE_SELECTION!r}")

        label_a, time_a = next(iter(SAMPLE_A.items()))
        label_b, time_b = next(iter(SAMPLE_B.items()))

        print(f"Comparing {label_a}={time_a} with {label_b}={time_b}")

        out_dir = (
            s.plot_dir
            / "feature_map_analysis"
            / f"{inference_start}_{inference_end}"
            / f"leadtime_{LEADTIME}"
            / f"{label_a}_{label_b}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        sample_a = load_feature_sample(
            feature_root_dir,
            time_a,
            SAMPLE_WITHIN_INIT_A,
        )
        sample_b = load_feature_sample(
            feature_root_dir,
            time_b,
            SAMPLE_WITHIN_INIT_B,
        )

        if PLOT_GEO_COORDS and ("longitudes" not in sample_a or "longitudes" not in sample_b):
            print(
                "WARNING: sample files lack longitudes: showing grid indices. "
                "Re-extract feature maps with the updated inference script for lon/lat axes."
            )

        plot_kernel_weights(
            feature_root_dir / "kernels.pt",
            out_dir / "learned_kernels",
        )

        plot_attention_comparison(
            sample_a, sample_b,
            out_dir / "cbam_attention",
        )

        print()
        print(
            f"Compare:\n"
            f"  {list(SAMPLE_A.values())[0]} = {sample_a['init_time']}\n"
            f"  {list(SAMPLE_B.values())[0]} = {sample_b['init_time']}"
        )

        # ----------------------------------------------------------
        # Normalized performance
        # ----------------------------------------------------------

        norm_performance_a = normalized_rmse_improvement(
            sample_a
        )

        norm_performance_b = normalized_rmse_improvement(
            sample_b
        )

        norm_performance_df = pd.DataFrame([
            {
                "sample": str(list(SAMPLE_A.keys())[0]),
                "time": sample_a["init_time"],
                **norm_performance_a,
            },
            {
                "sample": str(list(SAMPLE_B.keys())[0]),
                "time": sample_b["init_time"],
                **norm_performance_b,
            },
        ])

        norm_performance_df.to_csv(
            out_dir
            / "normalized_rmse_improvement.csv",
            index=False,
        )

        print()
        print("=" * 80)
        print("NORMALIZED-SPACE RMSE IMPROVEMENT")
        print("=" * 80)
        print(
            norm_performance_df.to_string(
                index=False
            )
        )

        # ----------------------------------------------------------
        # Whole-representation similarity
        # ----------------------------------------------------------

        comparison = compare_feature_layers(
            sample_a,
            sample_b,
        )

        print()
        print("=" * 80)
        print("WHOLE-REPRESENTATION SIMILARITY")
        print("=" * 80)
        print(
            comparison.to_string(
                index=False
            )
        )

        comparison.to_csv(
            out_dir
            / "layer_similarity.csv",
            index=False,
        )

        plot_layer_similarity(
            comparison,
            out_dir
            / "layer_similarity.png",
        )

        plot_layer_distance(
            comparison,
            out_dir
            / "layer_normalized_distance.png",
        )

        # ----------------------------------------------------------
        # Raw encoder vs CBAM
        # ----------------------------------------------------------

        encoder_df = build_encoder_cbam_comparison(
            comparison
        )

        print()
        print("=" * 80)
        print("RAW ENCODER VS CBAM")
        print("=" * 80)
        print(
            encoder_df.to_string(
                index=False
            )
        )

        encoder_df.to_csv(
            out_dir
            / "encoder_raw_vs_cbam.csv",
            index=False,
        )

        plot_raw_vs_cbam(
            encoder_df,
            out_dir
            / "encoder_raw_vs_cbam.png",
        )

        # ----------------------------------------------------------
        # Channel-wise similarity
        # ----------------------------------------------------------

        channel_corr = collect_channel_correlations(
            sample_a,
            sample_b,
        )

        channel_summary = channel_correlation_summary(
            channel_corr
        )

        print()
        print("=" * 80)
        print("CHANNEL-WISE CORRELATION")
        print("=" * 80)
        print(
            channel_summary.to_string(
                index=False
            )
        )

        channel_summary.to_csv(
            out_dir
            / "channel_correlation_summary.csv",
            index=False,
        )

        plot_channel_correlation_distributions(
            channel_corr,
            out_dir
            / "channel_correlation_distributions.png",
        )

        # ----------------------------------------------------------
        # RMSE
        # ----------------------------------------------------------

        fit_a = prediction_target_metrics(
            sample_a
        )

        fit_b = prediction_target_metrics(
            sample_b
        )

        fit_df = pd.DataFrame([
            {
                "sample": "A",
                "time": sample_a["init_time"],
                **fit_a,
            },
            {
                "sample": "B",
                "time": sample_b["init_time"],
                **fit_b,
            },
        ])

        print()
        print("=" * 80)
        print("PREDICTION VS TARGET")
        print("=" * 80)
        print(
            fit_df.to_string(
                index=False
            )
        )

        fit_df.to_csv(
            out_dir
            / "prediction_vs_target.csv",
            index=False,
        )

        # ----------------------------------------------------------
        # Spatial difference
        # ----------------------------------------------------------

        plot_selected_feature_differences(
            sample_a,
            sample_b,
            SPATIAL_DIFFERENCE_LAYERS,
            out_dir / "spatial_differences",
        )

        plot_tensor_difference(
            sample_a["prediction"],
            sample_b["prediction"],
            "Prediction A - B",
            out_dir
            / "prediction_difference.png",
            sample_a,
        )

        plot_tensor_difference(
            sample_a["target"],
            sample_b["target"],
            "Target A - B",
            out_dir
            / "target_difference.png",
            sample_a,
        )

        # ----------------------------------------------------------
        # Most different channels
        # ----------------------------------------------------------

        for layer in [
            "encoder_cbam_2",
            "encoder_raw_4",
            "encoder_cbam_4",
            "decoder_3",
        ]:
            channels = most_discriminative_channels(
                sample_a["features"][layer],
                sample_b["features"][layer],
                n=5,
            )

            print(
                f"\n{layer}:"
            )

            for channel, correlation in channels:
                print(
                    f"  channel={channel:4d} "
                    f"correlation={correlation:+.4f}"
                )

        plot_discriminative_channels(
            sample_a,
            sample_b,
            layers=[
                "encoder_cbam_2",
                "encoder_raw_4",
                "encoder_cbam_4",
                "decoder_3",
            ],
            output_dir=(
                out_dir
                / "discriminative_channels"
            ),
            n_channels=5,
        )



# =============================================================================
# Independent single-sample analysis
# =============================================================================

def _save_spatial_map(
    data: torch.Tensor | np.ndarray,
    sample: dict,
    title: str,
    output_file: Path,
    *,
    cmap: str = "viridis",
    centered: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    arr = np.asarray(data.detach().cpu() if isinstance(data, torch.Tensor) else data, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D field for {title}, got {arr.shape}")

    finite = arr[np.isfinite(arr)]
    if centered and finite.size and vmin is None and vmax is None:
        limit = max(float(np.quantile(np.abs(finite), 0.99)), 1e-12)
        vmin, vmax = -limit, limit

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists() and not REGENERATE_PLOTS:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    image = plot_spatial_field(ax, arr, sample, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)

    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.tight_layout()

    fig.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _selected_feature_layers(sample: dict) -> list[str]:
    available = list(sample["features"])
    if FEATURE_LAYERS is None:
        return available

    missing = set(FEATURE_LAYERS) - set(available)
    if missing:
        print(f"WARNING: unavailable feature layers: {sorted(missing)}")

    return [name for name in FEATURE_LAYERS if name in sample["features"]]


def plot_all_channel_mosaics(
    feature: torch.Tensor,
    sample: dict,
    layer: str,
    output_dir: Path,
) -> None:
    """Paginated contact sheets of all (C,H,W) feature maps for a layer.

    Shared scaling (default) preserves differences in activation magnitude
    between channels. Per-channel scaling emphasizes spatial pattern instead.
    """
    import math

    if feature.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got {tuple(feature.shape)}")
    if MOSAIC_COLUMNS < 1 or MOSAIC_CHANNELS_PER_PAGE < 1:
        raise ValueError("Mosaic columns and channels per page must be positive")
    if not 0.5 < MOSAIC_QUANTILE <= 1:
        raise ValueError("MOSAIC_QUANTILE must be in (0.5, 1]")
    if MOSAIC_SCALE not in ("shared", "per_channel"):
        raise ValueError(f"Invalid MOSAIC_SCALE={MOSAIC_SCALE!r}")

    maps = feature.detach().float().cpu().numpy()
    n_channels = len(maps)
    finite = maps[np.isfinite(maps)]
    if finite.size == 0:
        print(f"WARNING: no finite activations in {layer}")
        return

    signed = bool(finite.min() < 0)
    cmap = "RdBu_r" if signed else "viridis"

    def limits(values: np.ndarray) -> tuple[float, float]:
        values = values[np.isfinite(values)]
        if not values.size:
            return (0.0, 1.0)
        if signed:
            vmax = max(float(np.quantile(np.abs(values), MOSAIC_QUANTILE)), 1e-12)
            return -vmax, vmax
        vmax = max(float(np.quantile(values, MOSAIC_QUANTILE)), 1e-12)
        return 0.0, vmax

    global_limits = limits(finite)
    page_size = MOSAIC_CHANNELS_PER_PAGE
    n_pages = math.ceil(n_channels / page_size)
    output_dir.mkdir(parents=True, exist_ok=True)

    for page in range(n_pages):
        start = page * page_size
        end = min(start + page_size, n_channels)

        path = output_dir / f"all_channels_page_{page + 1:02d}.png"
        if path.exists() and not REGENERATE_PLOTS:
            continue

        count = end - start
        cols = min(MOSAIC_COLUMNS, count)
        rows = math.ceil(count / cols)

        fig, axes = plt.subplots(
            rows, cols, figsize=(2.5 * cols + 1, 2.1 * rows + 1),
            squeeze=False, constrained_layout=True,
        )

        last_image = None
        for pos, ax in enumerate(axes.flat):
            channel = start + pos
            if channel >= end:
                ax.axis("off")
                continue
            vmin, vmax = (
                global_limits if MOSAIC_SCALE == "shared"
                else limits(maps[channel])
            )
            last_image = plot_spatial_field(
                ax, maps[channel], sample, cmap=cmap, vmin=vmin, vmax=vmax,
            )

            ax.set_title(f"ch {channel}", fontsize=9)
            # Geographic axes on every tile would overwhelm the mosaic
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(labelbottom=False, labelleft=False, length=0)

        if MOSAIC_SCALE == "shared" and last_image is not None:
            fig.colorbar(
                last_image,
                ax=axes.ravel().tolist(),
                shrink=0.7,
                label="Activation",
            )

        fig.suptitle(
            f"{layer}: channels {start}-{end - 1} / {n_channels - 1}"
            f"  ({MOSAIC_SCALE} color scale)", fontsize=13,
        )

        fig.savefig(path, dpi=MOSAIC_DPI, bbox_inches="tight")
        plt.close(fig)


def analyze_single_sample_feature_maps(sample: dict, output_dir: Path) -> None:
    for layer in _selected_feature_layers(sample):
        feature = sample["features"][layer].float()
        if feature.ndim != 3:
            print(f"WARNING: skipping non-(C,H,W) layer {layer}: {tuple(feature.shape)}")
            continue

        layer_dir = output_dir / layer
        if PLOT_ALL_CHANNEL_MOSAICS:
            plot_all_channel_mosaics(
                feature, sample, layer, layer_dir / "all_channels",
            )

        summary = summarize_feature_map(feature)
        for statistic in FEATURE_SUMMARIES:
            _save_spatial_map(
                summary[statistic], sample, f"{layer}: channel {statistic.upper()}",
                layer_dir / f"channel_{statistic}_map.png",
                cmap="RdBu_r" if statistic == "mean" else "viridis",
                centered=statistic == "mean",
            )

        # Rank channels within this sample
        # Channels are NOT selected by their # correlation to a different weather realization
        if FEATURE_TOP_CHANNELS > 0:
            flat = feature.flatten(1)
            scores = (
                flat.square().mean(dim=1).sqrt()
                if FEATURE_CHANNEL_RANKING == "rms"
                else flat.std(dim=1, unbiased=False)
            )

            indices = torch.argsort(scores, descending=True)[:FEATURE_TOP_CHANNELS]
            for channel in indices.tolist():
                _save_spatial_map(
                    feature[channel],
                    sample,
                    f"{layer} / channel {channel} ({FEATURE_CHANNEL_RANKING}={scores[channel]:.3g})",
                    layer_dir / "top_channels" / f"channel_{channel:04d}.png",
                    cmap="RdBu_r",
                    centered=True,
                )


def analyze_single_sample_cbam(sample: dict, output_dir: Path) -> None:
    attention = sample.get("attention", {})
    if not attention:
        print("WARNING: no CBAM gates saved for this sample")
        return

    rows = []
    for name, gates in sorted(attention.items()):
        raw_name = name.replace("encoder_cbam_", "encoder_raw_", 1)
        raw = sample["features"].get(raw_name)
        post = sample["features"].get(name)

        if raw is None or post is None:
            print(f"WARNING: missing raw/post activations for {name}")
            continue

        raw, post = raw.float(), post.float()
        if raw.shape != post.shape or raw.ndim != 3:
            print(f"WARNING: raw/post shape mismatch at {name}")
            continue

        directory = output_dir / name
        raw_rms = raw.square().mean(dim=0).sqrt()
        post_rms = post.square().mean(dim=0).sqrt()

        _save_spatial_map(raw_rms, sample, f"{name}: before CBAM / channel RMS", directory / "before_cbam_rms.png")
        _save_spatial_map(post_rms, sample, f"{name}: after CBAM / channel RMS", directory / "after_cbam_rms.png")
        _save_spatial_map(post_rms - raw_rms, sample, f"{name}: change in channel RMS", directory / "rms_change.png", cmap="RdBu_r", centered=True)

        row = {
            "layer": name, "channels": raw.shape[0],
            "raw_rms": raw.square().mean().sqrt().item(),
            "post_rms": post.square().mean().sqrt().item(),
            "raw_std": raw.std(unbiased=False).item(),
            "post_std": post.std(unbiased=False).item(),
        }

        if "spatial" in gates:
            spatial = gates["spatial"].float().squeeze()
            if spatial.ndim == 2:
                _save_spatial_map(spatial, sample, f"{name}: spatial CBAM gate", directory / "spatial_gate.png", vmin=0, vmax=1)
                row["spatial_gate_mean"] = spatial.mean().item()

        if "channel" in gates:
            channel = gates["channel"].float().reshape(-1)
            row["channel_gate_mean"] = channel.mean().item()
            if len(channel) == raw.shape[0]:
                frame = pd.DataFrame({
                    "channel": np.arange(len(channel)),
                    "gate": channel.numpy(),
                    "raw_rms": raw.flatten(1).square().mean(dim=1).sqrt().numpy(),
                    "post_rms": post.flatten(1).square().mean(dim=1).sqrt().numpy(),
                })

                directory.mkdir(parents=True, exist_ok=True)
                frame.to_csv(directory / "channel_gates.csv", index=False)
                if REGENERATE_PLOTS or not (directory / "channel_gates.png").exists():
                    fig, ax = plt.subplots(figsize=(11, 4))
                    ax.plot(frame["channel"], frame["gate"], linewidth=1)
                    ax.set(xlabel="Feature channel", ylabel="Channel gate", ylim=(0, 1), title=f"{name}: CBAM channel gates")

                    fig.tight_layout()
                    fig.savefig(directory / "channel_gates.png", dpi=220)
                    plt.close(fig)
            else:
                print(f"WARNING: channel gate shape mismatch at {name}")

        # Exact effective channel+spatial gate when shapes match. Reweighting factor
        if "channel" in gates and "spatial" in gates:
            channel = gates["channel"].float().reshape(-1)
            spatial = gates["spatial"].float().squeeze()

            if channel.numel() == raw.shape[0] and tuple(spatial.shape) == tuple(raw.shape[-2:]):
                weighted_gate = (channel[:, None, None] * spatial[None]).mean(dim=0)
                _save_spatial_map(weighted_gate, sample, f"{name}: mean combined CBAM gate", directory / "combined_gate_mean.png", vmin=0, vmax=1)

        rows.append(row)

    if rows:
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_dir / "cbam_effect_summary.csv", index=False)


def analyze_single_sample_layer_statistics(sample: dict, output_dir: Path) -> None:
    rows = []
    for name in _selected_feature_layers(sample):
        feature = sample["features"][name].float()
        if feature.ndim != 3:
            continue

        flat = feature.flatten(1)
        channel_rms = flat.square().mean(dim=1).sqrt()
        channel_std = flat.std(dim=1, unbiased=False)

        rows.append({
            "layer": name, "channels": feature.shape[0],
            "height": feature.shape[-2], "width": feature.shape[-1],
            "activation_mean": feature.mean().item(),
            "activation_std": feature.std(unbiased=False).item(),
            "activation_rms": feature.square().mean().sqrt().item(),
            "mean_channel_rms": channel_rms.mean().item(),
            "std_channel_rms": channel_rms.std(unbiased=False).item(),
            "mean_channel_spatial_std": channel_std.mean().item(),
            "near_zero_fraction": (feature.abs() <= ACTIVATION_ZERO_THRESHOLD).float().mean().item(),
        })

    if not rows:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "layer_activation_statistics.csv", index=False)

    if PLOT_LAYER_PROFILE and (REGENERATE_PLOTS or not (output_dir / "layer_activation_profile.png").exists()):
        fig, ax = plt.subplots(figsize=(max(12, 0.6 * len(df)), 5))
        x = np.arange(len(df))

        ax.plot(x, df["activation_rms"], marker="o", label="Activation RMS")
        ax.plot(x, df["activation_std"], marker="s", label="Activation STD")

        ax.set_xticks(x, df["layer"], rotation=60, ha="right")
        ax.set(ylabel="Activation magnitude (layer-specific units)", title="Representation evolution")

        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "layer_activation_profile.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(max(12, 0.6 * len(df)), 4))
        ax.plot(x, df["near_zero_fraction"], marker="o")
        ax.set_xticks(x, df["layer"], rotation=60, ha="right")
        ax.set(ylabel=f"Fraction |activation| <= {ACTIVATION_ZERO_THRESHOLD:g}", ylim=(0, 1), title="Near-zero activation fraction")

        fig.tight_layout()
        fig.savefig(output_dir / "layer_near_zero_fraction.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def select_independent_analysis_cases(feature_root: Path) -> list[dict]:
    """Choose best/worst saved initializations, plus optional manual times.

    Rank=1 means the strongest improvement. Worst cases are numbered starting
    at W1 for the MOST NEGATIVE improvement (largest global rank).
    """
    if ANALYSIS_SELECTION not in {"ranked", "manual", "both"}:
        raise ValueError(f"Invalid ANALYSIS_SELECTION={ANALYSIS_SELECTION!r}")
    if ANALYSIS_BEST_N < 0 or ANALYSIS_WORST_N < 0:
        raise ValueError("ANALYSIS_BEST_N and ANALYSIS_WORST_N must be nonnegative")

    cases: list[dict] = []
    seen: set[pd.Timestamp] = set()

    if ANALYSIS_SELECTION in {"ranked", "both"} and (ANALYSIS_BEST_N or ANALYSIS_WORST_N):
        path = feature_root / "feature_map_performance_ranking.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing inference ranking CSV: {path}. "
                "Run inference.py with FEATURE_MAP_BEST_N / FEATURE_MAP_WORST_N enabled."
            )

        ranking = pd.read_csv(path)
        required = {"init_time", "selection", "rank"}

        if not required.issubset(ranking.columns):
            raise ValueError(f"Ranking CSV missing columns: {required - set(ranking.columns)}")

        metrics = [col for col in ranking.columns if col.endswith("_improvement")]
        if len(metrics) != 1:
            raise ValueError(f"Expected one metric improvement column in {path}, got {metrics}")

        improvement_col = metrics[0]

        for selection, count, prefix, ascending in (
            ("best", ANALYSIS_BEST_N, "B", True),
            ("worst", ANALYSIS_WORST_N, "W", False),
        ):
            selected = ranking.loc[ranking["selection"] == selection].sort_values("rank", ascending=ascending)

            if len(selected) < count:
                print(
                    f"WARNING: requested {count} {selection} samples, "
                    f"but only {len(selected)} were selected/saved during inference"
                )

            for position, (_, row) in enumerate(selected.head(count).iterrows(), start=1):
                timestamp = pd.Timestamp(row["init_time"])

                if timestamp in seen:
                    continue

                seen.add(timestamp)
                cases.append({
                    "label": f"{prefix}{position}",
                    "init_time": str(timestamp),
                    "selection": selection,
                    "rank": int(row["rank"]),
                    "ranking_metric": improvement_col.removesuffix("_improvement"),
                    "improvement": float(row[improvement_col]),
                })

    if ANALYSIS_SELECTION in {"manual", "both"}:
        for position, init_time in enumerate(ANALYSIS_MANUAL_TIMES, start=1):
            timestamp = pd.Timestamp(init_time)
            if timestamp in seen:
                print(f"INFO: skipping manual time {timestamp}: already selected")
                continue

            seen.add(timestamp)
            cases.append({
                "label": f"M{position}",
                "init_time": str(timestamp),
                "selection": "manual",
                "rank": None,
                "ranking_metric": None,
                "improvement": None,
            })

    return cases


def saved_sample_indices(feature_root: Path, init_time: str) -> list[int]:
    """Resolve available .pt files without loading their full feature tensors."""
    prefix = pd.Timestamp(init_time).strftime("features_init_%Y%m%dT%H%M%S_sample_")
    files = sorted(feature_root.glob(f"{prefix}*.pt"))
    available = []

    for path in files:
        suffix = path.stem.removeprefix(prefix)
        if suffix.isdigit():
            available.append(int(suffix))

    if ANALYSIS_SAMPLE_WITHIN_INIT is None:
        return sorted(set(available))
    if ANALYSIS_SAMPLE_WITHIN_INIT < 0:
        raise ValueError("ANALYSIS_SAMPLE_WITHIN_INIT must be nonnegative or None")

    requested = ANALYSIS_SAMPLE_WITHIN_INIT

    return [requested] if requested in available else []


def channel_participation_statistics(
    feature: torch.Tensor,
    top_n: int = 5,
) -> dict[str, float | int]:
    """Spatial channel-energy participation for one saved (C,H,W) tensor.

    E_c = mean_{h,w}(F_c^2); p_c = E_c / sum(E_c),
    N_eff = 1 / sum_c(p_c^2).

    Values are descriptive for the *given sample*. N_eff is not a measure of
    causal contribution to the output, and does not adjust for correlations
    among channels.
    """
    if feature.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got {tuple(feature.shape)}")
    if top_n < 1:
        raise ValueError("PARTICIPATION_TOP_N must be positive")

    flat = feature.detach().to(device="cpu", dtype=torch.float64).flatten(1)
    valid = torch.isfinite(flat)
    counts = valid.sum(dim=1)
    energy = torch.where(valid, flat.square(), 0.0).sum(dim=1) / counts.clamp_min(1)
    if (counts == 0).any():
        raise ValueError("At least one feature channel has no finite pixels")

    total = energy.sum()
    n_channels = feature.shape[0]
    if not torch.isfinite(total) or float(total) <= 0:
        return {
            "channels": n_channels,
            "effective_channels": float("nan"),
            "effective_channel_fraction": float("nan"),
            "top_energy_fraction": float("nan"),
            "channel_energy_total": float(total),
        }

    shares = energy / total
    effective = float(shares.square().sum().reciprocal())
    top_share = float(shares.topk(min(top_n, n_channels)).values.sum())
    return {
        "channels": n_channels,
        "effective_channels": effective,
        "effective_channel_fraction": effective / n_channels,
        "top_energy_fraction": top_share,
        "channel_energy_total": float(total),
    }


def _scatter_metric_columns(targets: tuple[str, ...]) -> list[tuple[str, str]]:
    """Convert user targets into wide-table columns and axis labels."""
    result = []
    for spec in targets:
        parts = spec.split(":")

        if spec == "correction:predicted_rms":
            result.append(("correction_predicted_rms", "Predicted correction RMS (MLFC - FC)"))
        elif len(parts) == 2 and parts[0] in {"fc", "mlfc"} and parts[1]:
            result.append((f"{parts[0]}_{parts[1]}", f"{parts[0].upper()} {parts[1].upper()}"))
        elif (len(parts) == 3 and parts[0] == "improvement"
              and parts[1] and parts[2] in {"Δ", "%", "normalized"}):
            result.append((
                f"improvement_{parts[1]}_{parts[2]}",
                f"{parts[1].upper()} improvement ({parts[2]}; positive = better)"
            ))
        else:
            raise ValueError(f"Invalid SCATTER_Y_TARGETS entry: {spec!r}")

    return list(dict.fromkeys(result))


def plot_representation_metric_scatters(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    x_columns: list[tuple[str, str]],
    annotate: bool,
    figsize: tuple[float, float],
    filename_statistic: str | None = None,
) -> None:
    """Shared scatter backend: any feature statistic versus any EarthML metric"""
    if df.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    y_columns = _scatter_metric_columns(SCATTER_Y_TARGETS)

    for layer, sub in df.groupby("layer", sort=False):
        for xcol, xlabel in x_columns:
            if xcol not in sub:
                continue

            for ycol, ylabel in y_columns:
                if ycol not in sub:
                    print(f"WARNING: metric column {ycol!r} unavailable")
                    continue

                plot_df = sub.loc[:, ["init_time", "label", "selection", xcol, ycol]].copy()
                plot_df[xcol] = pd.to_numeric(plot_df[xcol], errors="coerce")
                plot_df[ycol] = pd.to_numeric(plot_df[ycol], errors="coerce")
                plot_df = plot_df.loc[np.isfinite(plot_df[xcol]) & np.isfinite(plot_df[ycol])]

                if plot_df.empty:
                    continue

                statistic_suffix = f"_{filename_statistic}" if filename_statistic else ""
                filename = f"{layer}_{xcol}{statistic_suffix}_vs_{ycol}.png"
                path = output_dir / filename
                if path.exists() and not REGENERATE_PLOTS:
                    continue

                fig, ax = plt.subplots(figsize=figsize)
                for selection, group in plot_df.groupby("selection", sort=False):
                    color = {"best": "tab:blue", "worst": "tab:red", "manual": "tab:gray"}.get(selection, "tab:gray")
                    ax.scatter(
                        group[xcol],
                        group[ycol],
                        s=38,
                        color=color,
                        alpha=0.85,
                        label=f"{selection} (n={len(group)})",
                    )

                    if annotate:
                        for _, row in group.iterrows():
                            ax.annotate(
                                str(row["label"]),
                                (row[xcol], row[ycol]),
                                xytext=(4, 4),
                                textcoords="offset points",
                                fontsize=8,
                            )

                rho, pval = float("nan"), float("nan")
                if (len(plot_df) >= 3 and plot_df[xcol].nunique() > 1
                    and plot_df[ycol].nunique() > 1):
                    corr = spearmanr(plot_df[xcol], plot_df[ycol])
                    rho, pval = float(corr.statistic), float(corr.pvalue)

                if ycol.startswith("improvement_"):
                    ax.axhline(
                        0,
                        color="black",
                        linestyle="--",
                        linewidth=0.9,
                        alpha=0.7,
                    )

                ax.set(
                    xlabel=xlabel,
                    ylabel=ylabel,
                    title=(
                        f"{layer}: {xlabel} vs {ylabel}\n"
                        f"Saved initializations n={len(plot_df)}"
                        f"; Spearman rho={rho:.3f}, p={pval:.3g}"
                    )
                )

                ax.legend(loc="best")

                fig.tight_layout()
                fig.savefig(path, dpi=220)
                plt.close(fig)

                print(f"Saved metric scatter: {path}")


def plot_participation_vs_improvement(df: pd.DataFrame, output_dir: Path) -> None:
    x = (
        "effective_channels"
        if PARTICIPATION_X_AXIS == "count"
        else "effective_channel_fraction"
    )

    plot_representation_metric_scatters(
        df,
        output_dir,
        x_columns=[
            (
                x, "Effective channels"
                if x == "effective_channels"
                else "Effective channel fraction"
            ),
            (
                "top_energy_fraction",
                f"Top-{PARTICIPATION_TOP_N} energy fraction"
            )
        ],
        annotate=PARTICIPATION_ANNOTATE_POINTS,
        figsize=PARTICIPATION_FIGSIZE,
    )


def save_participation_analysis(
    rows: list[dict],
    output_dir: Path,
    metric_df: pd.DataFrame,
) -> None:
    """Save individual-sample and per-init tables; plot against inference ranking"""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not rows:
        print("WARNING: No channel-participation measurements available")
        return

    sample_df = pd.DataFrame(rows)
    sample_df.to_csv(output_dir / "channel_participation_samples.csv", index=False)

    # Inference performance is one value per init_time, averaged over members
    grouped = sample_df.groupby(
        ["layer", "init_time"],
        as_index=False
    ).agg(
        label=("label", "first"),
        selection=("selection", "first"),
        ranking_metric=("ranking_metric", "first"),
        improvement=("improvement", "first"),
        n_saved_samples=("sample_within_init", "size"),
        channels=("channels", "first"),
        effective_channels=("effective_channels", "mean"),
        effective_channel_fraction=("effective_channel_fraction", "mean"),
        top_energy_fraction=("top_energy_fraction", "mean"),
        channel_energy_total=("channel_energy_total", "mean"),
    )

    grouped = merge_library_metrics(grouped, metric_df)
    grouped.to_csv(output_dir / "channel_participation_initializations.csv", index=False)

    plot_participation_vs_improvement(
        grouped,
        output_dir,
    )

    print(
        f"Channel participation: {len(sample_df)} saved samples, "
        f"{grouped['init_time'].nunique()} unique initializations, "
        f"{grouped['layer'].nunique()} layers"
    )


# =============================================================================
# Spatially averaged feature-summary scatter (independent of panel rendering)
# =============================================================================

def collect_spatial_summary_scalars(
    sample: dict,
    case: dict,
    sample_index: int,
    rows: list[dict],
) -> None:
    """Average the SAME across-channel RMS/STD maps visualized in batch panels.

    First reduce across channels at every grid point, then average over space.
    This differs from taking a global tensor STD or RMS in general.
    """
    layers = (
        SUMMARY_SCATTER_LAYERS
        if SUMMARY_SCATTER_LAYERS is not None
        else PANEL_LAYERS if PANEL_LAYERS is not None
        else _selected_feature_layers(sample)
    )

    for layer in layers:
        feature = sample["features"].get(layer)

        if feature is None:
            print(f"WARNING: summary scatter layer {layer!r} missing for {case['label']}")
            continue

        if feature.ndim != 3:
            raise ValueError(f"{layer}: expected (C,H,W), got {tuple(feature.shape)}")

        f = feature.detach().float().cpu()
        if not torch.isfinite(f).all():
            print(f"WARNING: non-finite activations for {case['label']} / {layer}; skipping")
            continue

        maps = {}
        if "rms" in SUMMARY_SCATTER_STATISTICS:
            maps["rms"] = f.square().mean(dim=0).sqrt()
        if "std" in SUMMARY_SCATTER_STATISTICS:
            maps["std"] = f.std(dim=0, unbiased=False)

        for stat, spatial_map in maps.items():
            if SUMMARY_SCATTER_AREA_WEIGHTED:
                latitude = sample.get("latitudes")
                if latitude is None:
                    raise ValueError("SUMMARY_SCATTER_AREA_WEIGHTED requires latitudes")

                lat = torch.as_tensor(latitude, dtype=torch.float32).reshape(-1)
                if lat.numel() < 2:
                    raise ValueError("Expected at least two latitude coordinates")

                # Map each downsampled layer's rows to the approximate lat centers
                lat = torch.linspace(lat[0].item(), lat[-1].item(), spatial_map.shape[0])
                weights = torch.cos(torch.deg2rad(lat)).clamp_min(0)[:, None]
                spatial_mean = (spatial_map * weights).sum() / (weights.sum() * spatial_map.shape[1]).clamp_min(1e-12)
            else:
                spatial_mean = spatial_map.mean()

            rows.append({
                "label": case["label"],
                "init_time": str(pd.Timestamp(case["init_time"])),
                "selection": case["selection"],
                "ranking_metric": case["ranking_metric"],
                "improvement": case["improvement"],
                "sample_within_init": sample_index,
                "layer": layer,
                "statistic": stat,
                "spatial_mean": float(spatial_mean),
                "n_channels": int(f.shape[0]),
            })


def save_spatial_summary_scatters(
    rows: list[dict],
    output_dir: Path,
    metric_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not rows:
        print("WARNING: no spatial summary scalar measurements")
        return

    samples = pd.DataFrame(rows)
    samples.to_csv(output_dir / "spatial_summary_samples.csv", index=False)

    initializations = samples.groupby(
        ["layer", "statistic", "init_time"],
        as_index=False,
    ).agg(
        label=("label", "first"),
        selection=("selection", "first"),
        ranking_metric=("ranking_metric", "first"),
        improvement=("improvement", "first"),
        spatial_mean=("spatial_mean", "mean"),
        n_samples=("sample_within_init", "size"),
        n_channels=("n_channels", "first"),
    )

    initializations = merge_library_metrics(initializations, metric_df)
    initializations.to_csv(output_dir / "spatial_summary_initializations.csv", index=False)

    for stat, stat_df in initializations.groupby("statistic", sort=False):
        weighted = " (cos-lat weighted)" if SUMMARY_SCATTER_AREA_WEIGHTED else ""
        plot_representation_metric_scatters(
            stat_df,
            output_dir,
            x_columns=[("spatial_mean", f"Spatial mean of channel {stat.upper()}{weighted}")],
            annotate=SUMMARY_SCATTER_ANNOTATE,
            figsize=SUMMARY_SCATTER_FIGSIZE,
            filename_statistic=stat,
        )

    print(
        f"Spatial summaries: {len(samples)} sample-layer rows, "
        f"{initializations['init_time'].nunique()} initializations"
    )


# =============================================================================
# Batch spatial summary panels (independent of A/B comparisons)
# =============================================================================

def collect_batch_panel_maps(
    sample: dict, case: dict, sample_index: int, store: dict,
) -> None:
    """Store only channel-summary 2D maps, not full feature tensors.

    Results are kept per realization; multiple realizations for the same
    initialization can therefore appear as separate tiles.
    """
    layers = (
        PANEL_LAYERS
        if PANEL_LAYERS is not None
        else _selected_feature_layers(sample)
    )

    for layer in layers:
        feature = sample["features"].get(layer)

        if feature is None:
            print(f"WARNING: panel layer {layer!r} is unavailable for {case['label']}")
            continue

        if feature.ndim != 3:
            print(f"WARNING: panel layer {layer!r} has invalid shape {tuple(feature.shape)}")
            continue

        wanted = set(PANEL_STATISTICS)

        values = feature.detach().float().cpu()

        maps = {}
        if "rms" in wanted:
            maps["rms"] = values.square().mean(dim=0).sqrt().numpy().copy()

        if "std" in wanted:
            maps["std"] = values.std(dim=0, unbiased=False).numpy().copy()

        for statistic, arr in maps.items():
            store.setdefault((layer, statistic), []).append({
                "selection": case["selection"],
                "label": case["label"],
                "init_time": str(pd.Timestamp(case["init_time"])),
                "improvement": case["improvement"],
                "ranking_metric": case["ranking_metric"],
                "sample_index": sample_index,
                "map": arr,
                "sample_coords": {
                    "longitudes": sample.get("longitudes"),
                    "latitudes": sample.get("latitudes"),
                },
            })


def plot_batch_summary_panels(
    store: dict,
    output_dir: Path,
) -> None:
    """Separate all-best and all-worst grids for each layer/statistic.

    A shared color range is computed across BOTH groups and all pages.
    Setting PANEL_ROWS_PER_PAGE=None puts all cases in each group in one figure.
    """
    if not store:
        print("WARNING: no batch summary maps available")
        return

    if PANEL_COLUMNS < 1:
        raise ValueError("PANEL_COLUMNS must be >= 1")

    if PANEL_ROWS_PER_PAGE is not None and PANEL_ROWS_PER_PAGE < 1:
        raise ValueError("PANEL_ROWS_PER_PAGE must be None or >= 1")

    if not 0 < PANEL_COLOR_QUANTILE <= 1:
        raise ValueError("PANEL_COLOR_QUANTILE must be in (0, 1]")

    if not set(PANEL_STATISTICS).issubset({"rms", "std"}):
        raise ValueError("PANEL_STATISTICS supports only 'rms' and 'std'")

    from math import ceil
    output_dir.mkdir(parents=True, exist_ok=True)
    for (layer, statistic), records in store.items():
        groups = {
            key: sorted(
                (r for r in records if r["selection"] == key),
                key=lambda r: (
                    int(r["label"][1:]) if r["label"][1:].isdigit() else 10**9,
                    r["sample_index"],
                ),
            )
            for key in ("best", "worst", "manual")
        }

        finite = [r["map"][np.isfinite(r["map"])] for r in records]
        finite = [arr for arr in finite if arr.size]

        if not finite:
            print(f"WARNING: non-finite maps for {layer} / {statistic}")
            continue

        values = np.concatenate(finite)
        vmax = float(np.quantile(values, PANEL_COLOR_QUANTILE))

        if not np.isfinite(vmax) or vmax <= 0:
            vmax = max(float(np.max(values)), 1e-12)

        metric = next(
            (
                r["ranking_metric"] for r in records
                if r["ranking_metric"]
            ),
            "metric",
        )

        for group in ("best", "worst", "manual"):
            selected = groups[group]
            if not selected:
                continue

            max_per_page = (
                PANEL_COLUMNS * PANEL_ROWS_PER_PAGE
                if PANEL_ROWS_PER_PAGE is not None
                else len(selected)
            )

            n_pages = ceil(len(selected) / max_per_page)
            for page in range(n_pages):
                page_records = selected[page * max_per_page:(page + 1) * max_per_page]
                n_rows = ceil(len(page_records) / PANEL_COLUMNS)
                suffix = f"_page_{page+1:02d}" if n_pages > 1 else ""

                path = output_dir / layer / f"{statistic}_{group}{suffix}.png"
                path.parent.mkdir(parents=True, exist_ok=True)

                if path.exists() and not REGENERATE_PLOTS:
                    continue

                fig, axes = plt.subplots(
                    n_rows, PANEL_COLUMNS,
                    figsize=(3.0 * PANEL_COLUMNS + 1.2, 3.0 * n_rows + 1.1),
                    squeeze=False,
                )

                last_im = None
                for idx, ax in enumerate(axes.flat):
                    if idx >= len(page_records):
                        ax.axis("off")
                        continue

                    record = page_records[idx]
                    last_im = plot_spatial_field(
                        ax,
                        record["map"],
                        record["sample_coords"],
                        cmap="viridis",
                        vmin=0,
                        vmax=vmax,
                    )

                    score = record["improvement"]
                    score_label = (
                        f" | Δ={score:+.3g}"
                        if score is not None and np.isfinite(score)
                        else ""
                    )

                    member = (
                        f" / s{record['sample_index']}"
                        if ANALYSIS_SAMPLE_WITHIN_INIT is None
                        else ""
                    )

                    date = pd.Timestamp(record["init_time"]).strftime("%Y-%m-%d")

                    ax.set_title(
                        f"{record['label']}{member} | {date}{score_label}",
                        fontsize=9,
                    )

                    if not PANEL_SHOW_GEO_TICKS:
                        ax.set_xlabel("")
                        ax.set_ylabel("")
                        ax.tick_params(labelbottom=False, labelleft=False, length=0)

                if last_im is not None:
                    colorbar_axis = fig.add_axes([0.905, 0.16, 0.017, 0.62])
                    fig.colorbar(
                        last_im,
                        cax=colorbar_axis,
                        label=f"Across-channel {statistic.upper()}",
                    )

                fig.suptitle(
                    (
                        f"{layer}: channel {statistic.upper()} | {group.capitalize()} "
                        f"cases ({len(selected)} total)"
                        + (f" (page {page+1}/{n_pages})" if n_pages > 1 else "")
                        + f"\nShared scale [0, {vmax:.3g}] across best and worst; "
                        f"Δ{metric} > 0 = improvement"
                    ),
                    fontsize=12,
                )
                fig.subplots_adjust(
                    left=0.055,
                    right=0.88,
                    bottom=0.045,
                    top=min(0.91, 1 - 0.55 / (3.0 * n_rows + 1.1)),
                    hspace=0.42,
                    wspace=0.10,
                )

                fig.savefig(path, dpi=PANEL_DPI, bbox_inches="tight")
                plt.close(fig)

                print(f"Saved panel: {path}")


# =============================================================================
# EarthML metric timeseries (shared between all feature scatter families)
# =============================================================================

def merge_library_metrics(
    features: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    if metrics.empty:
        return features

    left = features.copy()
    right = metrics.copy()
    left["init_time"] = pd.to_datetime(left["init_time"])
    right["init_time"] = pd.to_datetime(right["init_time"])

    return left.merge(
        right,
        on="init_time",
        how="left",
        validate="many_to_one",
    )


def _metric_time_series(da: xr.DataArray, name: str) -> pd.DataFrame:
    """Extract a unique initialization timeseries; refuse unintended aggregation"""

    if "leadtime" in da.dims:
        if da.sizes["leadtime"] != 1:
            raise ValueError(f"{name}: unexpected leadtime count {da.sizes['leadtime']}")
        da = da.squeeze("leadtime", drop=True)

    time_dim = da.earthml.guessed_dims.time
    if time_dim is None:
        if "time" in da.dims:
            time_dim = "time"
        else:
            raise ValueError(f"{name}: no time dimension: {da.dims}")

    non_time = [d for d in da.dims if d != time_dim]
    for d in non_time:
        if da.sizes[d] != 1:
            raise ValueError(
                f"{name}: unsupported non-singleton dimension {d!r} ({da.sizes[d]}). "
                "Select one period and ensemble-aggregated timeseries."
            )

    da = da.squeeze(non_time, drop=True) if non_time else da
    frame = pd.DataFrame({
        "init_time": pd.to_datetime(da[time_dim].values),
        name: pd.to_numeric(da.values, errors="coerce")
    })

    if frame["init_time"].duplicated().any():
        raise ValueError(f"Duplicate timestamps in {name}")

    return frame


def _resolve_metric_clim_range(s) -> tuple[str, str]:
    lookup = {
        "train": (s.train_start, s.train_end),
        "val": (s.val_start, s.val_end),
        "test": (s.test_start, s.test_end),
        "train_val": (s.train_start, s.val_end),
        "full": (s.train_start, s.test_end),
    }

    return lookup[METRIC_CLIM_TIME_RANGE]


def calculate_library_metric_table(
    s, inference_start: str, inference_end: str,
    init_times: list[str],
) -> pd.DataFrame:
    """Use the same metric functions as plot_metric_timeseries.py.

    Only the saved sample initialization timestamps are requested. The scatter
    metric is computed in physical space, not reconstructed normalized space.
    """
    if not ANALYZE_LIBRARY_METRICS or not init_times or not (ANALYZE_SPATIAL_SUMMARY_SCATTERS or ANALYZE_CHANNEL_PARTICIPATION):
        return pd.DataFrame()
    if not SCATTER_Y_TARGETS:
        raise ValueError("Configure at least one SCATTER_Y_TARGETS entry")

    _scatter_metric_columns(SCATTER_Y_TARGETS)

    # "correction:predicted_rms"
    requested_metrics = list(dict.fromkeys(
        spec.split(":")[1] for spec in SCATTER_Y_TARGETS
        if not spec.startswith("correction:")
    ))

    # Normalized improvement can require additional reference metrics
    extra_metrics = [
        STD_REFERENCE_METRICS[m] for m in requested_metrics
        if m in STD_REFERENCE_METRICS and any(spec == f"improvement:{m}:normalized" for spec in SCATTER_Y_TARGETS)
    ]

    metrics = list(dict.fromkeys(requested_metrics + extra_metrics))
    deterministic = [m for m in metrics if is_deterministic(m)]
    probabilistic = [m for m in metrics if is_probabilistic(m)]

    unsupported = [m for m in metrics if m not in deterministic and m not in probabilistic]
    if unsupported:
        raise ValueError(f"Unsupported EarthML metric(s): {unsupported}")

    result = pd.DataFrame({"init_time": pd.to_datetime(init_times)}).drop_duplicates().sort_values("init_time")
    range_start, range_end = resolve_inference_range(s)

    mlfc_path = (
        s.exp_dir
        / "inference"
        / f"{inference_start}_{inference_end}"
        / "test_corrected.zarr"
    ) if METRIC_INFERENCE_OUTPUT else None

    if mlfc_path is not None and not mlfc_path.exists():
        raise FileNotFoundError(
            f"Missing physical corrected forecast: {mlfc_path}. "
            "Set METRIC_INFERENCE_OUTPUT=False to use experiment's standard MLFC."
        )

    lat_lon = list(s.region.values()) if s.region is not None else [None, None]

    fc, an, mlfc = get_and_subset_datasets(
        s,
        leadtime_units=LeadtimeUnit(s.leadtime_unit),
        lat_range=lat_lon[0],
        lon_range=lat_lon[1],
        time_range=(range_start, range_end),
        interpolate=METRIC_INTERPOLATE_ANALYSIS,
        mlfc_path=mlfc_path,
    )

    if mlfc is None:
        raise ValueError("MLFC dataset not available for metric evaluation")

    # Select leadtime and timestamps explicitly
    requested_lead = int(LEADTIME)
    lead_dim = fc.earthml.guessed_dims.leadtime
    if lead_dim is None:
        raise ValueError("Could not determine leadtime coordinate")

    frames = {}
    times = pd.DatetimeIndex(result["init_time"])

    for model, ds in (("fc", fc), ("an", an), ("mlfc", mlfc)):
        if lead_dim in ds.dims or lead_dim in ds.coords:
            ds = ds.sel({lead_dim: [requested_lead]})

        time_dim = ds.earthml.guessed_dims.time or "time"
        available = pd.DatetimeIndex(ds[time_dim].values)
        missing = times.difference(available)

        if len(missing):
            raise ValueError(
                f"{model}: missing {len(missing)} saved feature "
                f"initializations, first={missing[0]}"
            )

        frames[model] = ds.sel({time_dim: times})

    clim = {"fc": None, "mlfc": None, "an": None}
    if METRIC_LOAD_CLIMATOLOGIES:
        clim_fc, clim_an, clim_mlfc = calculate_save_and_subset_climatologies(
            s,
            leadtime_units=LeadtimeUnit(s.leadtime_unit),
            force=METRIC_RECALCULATE_CLIMATOLOGY,
            clim_period=METRIC_CLIM_PERIOD,
            rolling_window=None,
            rolling_center=True,
            rolling_min_periods=1,
            lat_range=lat_lon[0],
            lon_range=lat_lon[1],
            time_range=_resolve_metric_clim_range(s),
            time_start=None,
            interpolate=METRIC_INTERPOLATE_ANALYSIS,
            engine="zarr",
            build_analysis=True,
            coord_rename_fc=None,
            coord_rename_an=None,
        )
        clim = {"fc": clim_fc, "an": clim_an, "mlfc": clim_mlfc}
        for model in ("fc", "an", "mlfc"):
            if clim[model] is not None:
                dim = clim[model].earthml.guessed_dims.leadtime
                if dim is not None:
                    clim[model] = clim[model].sel({dim: [requested_lead]})

    period_dim = f"{METRIC_PERIOD_REFERENCE}_{METRIC_CLIM_PERIOD.value}"
    by_model = {}
    for model in ("fc", "mlfc"):
        pieces = []
        for selected, realization_agg in ((deterministic, True), (probabilistic, False)):
            if not selected:
                continue

            pieces.append(get_metrics(
                an=frames["an"],
                fc=frames[model],
                var=s.var_fc,
                metric_kind="timeseries",
                leadtime_agg="single",
                realization_agg=realization_agg,
                an_clim=clim["an"],
                fc_clim=clim[model],
                orography_path=METRIC_OROGRAPHY_PATH,
                metrics=selected,
                leadtime_windows=s.seasonal_leadtime_windows,
                leadtime_agg_coord="leadtime",
                clim_period=METRIC_CLIM_PERIOD,
                period_reference=METRIC_PERIOD_REFERENCE,
                period_dim=period_dim,
                periods_requested=["all"],
                leadtime_unit=s.leadtime_unit,
                align=False,
                fair_correction=False,
            ))

        by_model[model] = xr.merge(pieces)

        for metric in requested_metrics:
            col = f"{model}_{metric}"
            result = result.merge(
                _metric_time_series(by_model[model][metric], col),
                on="init_time",
                how="left",
                validate="one_to_one",
            )

    # Predicted correction magnitude: RMS(MLFC - FC) in physical units
    if "correction:predicted_rms" in SCATTER_Y_TARGETS:
        fc_reference = frames["fc"][[s.var_fc]].rename({s.var_fc: s.var_an})
        realization_dim = fc_reference.earthml.guessed_dims.realization

        if realization_dim is not None and realization_dim in fc_reference.dims:
            fc_reference = fc_reference.mean(realization_dim, skipna=True)

        correction_metrics = get_metrics(
            an=fc_reference,
            fc=frames["mlfc"],
            var=s.var_fc,
            metric_kind="timeseries",
            leadtime_agg="single",
            realization_agg=True,
            an_clim=None,
            fc_clim=None,
            orography_path=METRIC_OROGRAPHY_PATH,
            metrics=["rmse"],
            leadtime_windows=s.seasonal_leadtime_windows,
            leadtime_agg_coord="leadtime",
            clim_period=METRIC_CLIM_PERIOD,
            period_reference=METRIC_PERIOD_REFERENCE,
            period_dim=period_dim,
            periods_requested=["all"],
            leadtime_unit=s.leadtime_unit,
            align=False,
            fair_correction=False,
        )

        result = result.merge(
            _metric_time_series(correction_metrics["rmse"], "correction_predicted_rms"),
            on="init_time",
            how="left",
            validate="one_to_one",
        )

    for metric in requested_metrics:
        units = tuple(dict.fromkeys(
            spec.split(":")[2] for spec in SCATTER_Y_TARGETS
            if spec.startswith(f"improvement:{metric}:")
        ))

        if not units:
            continue

        improvements = build_metric_improvements(
            by_model["fc"],
            by_model["mlfc"],
            metric=metric,
            baseline_model="fc",
            target_model="mlfc",
            improvement_units=units,
        )

        # Unit identification
        suffix_to_unit = {
            "_percentage": "%",
            "_difference": "Δ",
            "_normalized": "normalized",
        }

        found_units = set()
        for name, da in improvements.items():
            unit = next(
                (
                    unit for suffix, unit in suffix_to_unit.items()
                    if str(name).endswith(suffix)
                ),
                None,
            )

            if unit is None:
                raise ValueError(f"Cannot identify improvement unit for {name!r}")
            if unit not in units:
                raise ValueError(f"Unexpected improvement unit {unit!r} for {metric}")
            if unit in found_units:
                raise ValueError(f"Duplicate improvement unit {unit!r} for {metric}")

            found_units.add(unit)
            col = f"improvement_{metric}_{unit}"

            result = result.merge(
                _metric_time_series(da, col),
                on="init_time",
                how="left",
                validate="one_to_one",
            )

        if found_units != set(units):
            raise ValueError(f"Missing units for {metric}: {set(units) - found_units}")

    return result


def main() -> None:
    settings = get_experiment_configs(
        EXPERIMENTS_ROOT,
        var_fc=VARIABLES,
        region_name=REGIONS,
        **FILTERS,
    )

    if not settings:
        raise RuntimeError("No matching experiments found")

    print(f"Found {len(settings)} matching experiment(s).")

    for s in settings:
        start, end = resolve_inference_range(s)
        feature_root = (
            s.exp_dir
            / "inference"
            / f"{start}_{end}"
            / "feature_maps"
            / f"leadtime_{LEADTIME}"
        )
        if not feature_root.exists():
            raise FileNotFoundError(feature_root)

        base_output = (
            s.plot_dir
            / "feature_map_analysis"
            / f"{start}_{end}"
            / f"leadtime_{LEADTIME}"
        )

        cases = select_independent_analysis_cases(feature_root)
        print(
            f"Selected {len(cases)} independent initializations "
            f"({ANALYSIS_SELECTION=}, best={ANALYSIS_BEST_N}, worst={ANALYSIS_WORST_N})."
        )

        metric_df = calculate_library_metric_table(
            s,
            start,
            end,
            [case["init_time"] for case in cases],
        )

        if not metric_df.empty:
            base_output.mkdir(parents=True, exist_ok=True)
            metric_df.to_csv(base_output / "earthml_metric_timeseries.csv", index=False)
            print(f"Computed EarthML metric values for {len(metric_df)} selected initializations")

        manifest = []
        participation_rows = []
        batch_panel_maps = {}
        spatial_summary_rows = []
        for case in cases:
            label, init_time = case["label"], case["init_time"]
            indices = saved_sample_indices(feature_root, init_time)

            if not indices:
                print(
                    f"WARNING: no extracted feature .pt file for {label} / {init_time}; "
                    "adjust ANALYSIS_SAMPLE_WITHIN_INIT or run inference extraction"
                )
                continue

            for index in indices:
                sample = load_feature_sample(feature_root, init_time, index)
                output_dir = (
                    base_output
                    / "single_samples"
                    / f"{label}_{pd.Timestamp(init_time):%Y%m%dT%H%M%S}_sample_{index}"
                )

                print(f"Analyzing {label} / {init_time} / sample_{index} -> {output_dir}")

                if PLOT_GEO_COORDS and "longitudes" not in sample:
                    print("WARNING: longitude data missing; using grid indices")
                if ANALYZE_SPATIAL_SUMMARY_SCATTERS:
                    collect_spatial_summary_scalars(sample, case, index, spatial_summary_rows)
                if PLOT_BATCH_SUMMARY_PANELS:
                    collect_batch_panel_maps(sample, case, index, batch_panel_maps)
                if ANALYZE_FEATURE_MAPS:
                    analyze_single_sample_feature_maps(sample, output_dir / "feature_maps")
                if ANALYZE_CBAM:
                    analyze_single_sample_cbam(sample, output_dir / "cbam")
                if ANALYZE_LAYER_STATISTICS:
                    analyze_single_sample_layer_statistics(sample, output_dir / "layer_statistics")
                if ANALYZE_CHANNEL_PARTICIPATION:
                    for layer in PARTICIPATION_LAYERS:
                        feature = sample["features"].get(layer)
                        if feature is None:
                            print(f"WARNING: missing participation layer {layer!r} in {label}")
                            continue

                        participation_rows.append({
                            "label": label,
                            "init_time": str(pd.Timestamp(init_time)),
                            "selection": case["selection"],
                            "ranking_metric": case["ranking_metric"],
                            "improvement": case["improvement"],
                            "sample_within_init": index,
                            "layer": layer,
                            **channel_participation_statistics(feature, top_n=PARTICIPATION_TOP_N),
                        })

                manifest.append({
                    **case,
                    "sample_within_init": index,
                    "output_dir": str(output_dir)
                })

                del sample

        base_output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            manifest,
            columns=[
                "label",
                "init_time",
                "selection",
                "rank",
                "ranking_metric",
                "improvement",
                "sample_within_init",
                "output_dir",
            ]
        ).to_csv(
            base_output / "independent_analysis_manifest.csv",
            index=False,
        )

        print(f"Completed {len(manifest)} independent sample analyses")

        if ANALYZE_CHANNEL_PARTICIPATION:
            save_participation_analysis(participation_rows, base_output / "channel_participation", metric_df)
        if ANALYZE_SPATIAL_SUMMARY_SCATTERS:
            save_spatial_summary_scatters(spatial_summary_rows, base_output / "spatial_summary_scatter", metric_df)
        if PLOT_BATCH_SUMMARY_PANELS:
            plot_batch_summary_panels(batch_panel_maps, base_output / "batch_summary_panels")
        if PLOT_KERNEL_WEIGHTS:
            plot_kernel_weights(feature_root / "kernels.pt", base_output / "learned_kernels")

    # Pairwise comparisons remain a completely separate opt-in path
    if RUN_SAMPLE_COMPARISON:
        run_sample_comparison()


if __name__ == "__main__":
    main()
