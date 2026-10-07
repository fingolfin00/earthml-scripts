from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from earthml import get_experiment_configs

# =============================================================================
# Configuration
# =============================================================================

# exp_name = "weather_atmo"
exp_name = "weather_atmo_one_year_test"
# exp_name = "weather_atmo_ablation_fixed_val"
# exp_name = "weather_atmo_short_zero_vs_replicate_padding"

EXPERIMENTS_ROOT = Path(
    f"/work/cmcc/jd19424/ML/MLBC/experiments/{exp_name}"
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
    "ConUS",
    # "Europe",
    # "Pacific",
    # "World",
    # None,
]

LEADTIME = "72"

INFERENCE_START = "2025-02-01T00:00:00"
# INFERENCE_END = "2025-02-10T00:00:00"
INFERENCE_END = "2025-03-31T00:00:00"

# SAMPLE_A = {"B5": "2025-02-02T00:00:00"} # B5 t2m 72h, near W4 in std-vs-mean scatter
SAMPLE_A = {"W1": "2025-02-13T00:00:00"} # W1 t2m 72h
# SAMPLE_B = {"W4": "2025-02-06T00:00:00"} # W3 / W4 t2m 72h
SAMPLE_B = {"W2": "2025-02-14T00:00:00"} # W2 t2m 72h
# SAMPLE_B = {"B1": "2025-03-08T00:00:00"} # B1 t2m 72h
# SAMPLE_B = {"B2": "2025-02-22T00:00:00"} # B2 t2m 72h

# Ten-month test 01-09 2025, channel norm
# SAMPLE_A = {"B1": "2025-03-08T00:00:00"} # B1 t2m 72h
# SAMPLE_A = {"B2": "2025-02-22T00:00:00"} # B2 t2m 72h
# SAMPLE_A = {"B3": "2025-03-23T00:00:00"} # B3 t2m 72h
# SAMPLE_A = {"B4": "2025-02-02T00:00:00"} # B4 t2m 72h, near W3 in stb-vs-mean scatter
# SAMPLE_A = {"B5": "2025-03-07T00:00:00"} # B5 t2m 72h
# SAMPLE_A = {"B6": "2025-02-21T00:00:00"} # B6 t2m 72h
# SAMPLE_A = {"B7": "2025-04-06T00:00:00"} # B7 t2m 72h
# SAMPLE_A = {"B8": "2025-02-01T00:00:00"} # B8 t2m 72h
# SAMPLE_A = {"B9": "2025-04-07T00:00:00"} # B9 t2m 72h
# SAMPLE_A = {"B10": "2025-03-06T00:00:00"} # B10 t2m 72h

# SAMPLE_B = {"W1": "2025-02-13T00:00:00"} # W1 t2m 72h
# SAMPLE_B = {"W2": "2025-01-03T00:00:00"} # W2 t2m 72h
# SAMPLE_B = {"W3": "2025-02-06T00:00:00"} # W3 t2m 72h
# SAMPLE_B = {"W4": "2025-02-18T00:00:00"} # W4 t2m 72h
# SAMPLE_B = {"W5": "2025-02-16T00:00:00"} # W5 t2m 72h
# SAMPLE_B = {"W6": "2025-01-01T12:00:00"} # W6 t2m 72h
# SAMPLE_B = {"W7": "2025-02-12T12:00:00"} # W7 t2m 72h
# SAMPLE_B = {"W8": "2025-04-24T00:00:00"} # W8 t2m 72h
# SAMPLE_B = {"W9": "2025-03-29T00:00:00"} # W9 t2m 72h
# SAMPLE_B = {"W10": "2025-05-25T12:00:00"} # W10 t2m 72h

# One year test, channel norm
# SAMPLE_A = {"B1": "2025-03-08T00:00:00"} # B1 t2m 72h
# SAMPLE_A = {"B2": "2025-02-22T00:00:00"} # B2 t2m 72h
# SAMPLE_A = {"B3": "2025-03-23T00:00:00"} # B3 t2m 72h
# SAMPLE_A = {"B4": "2025-03-07T00:00:00"} # B4 t2m 72h
# SAMPLE_A = {"B5": "2025-02-02T00:00:00"} # B5 t2m 72h, near W4 in std-vs-mean scatter
# SAMPLE_A = {"B6": "2025-02-21T00:00:00"} # B6 t2m 72h
# SAMPLE_A = {"B7": "2025-04-06T00:00:00"} # B7 t2m 72h
# SAMPLE_A = {"B8": "2025-04-07T00:00:00"} # B8 t2m 72h
# SAMPLE_A = {"B9": "2025-02-25T00:00:00"} # B9 t2m 72h
# SAMPLE_A = {"B10": "2025-02-24T00:00:00"} # B10 t2m 72h

# SAMPLE_B = {"W1": "2025-02-13T00:00:00"} # W1 t2m 72h
# SAMPLE_B = {"W2": "2025-02-14T00:00:00"} # W2 t2m 72h
# SAMPLE_B = {"W3": "2024-11-28T00:00:00"} # W3 t2m 72h
# SAMPLE_B = {"W4": "2025-02-06T00:00:00"} # W4 t2m 72h
# SAMPLE_B = {"W5": "2025-03-29T00:00:00"} # W5 t2m 72h
# SAMPLE_B = {"W6": "2024-11-29T00:00:00"} # W6 t2m 72h
# SAMPLE_B = {"W7": "2025-01-16T00:00:00"} # W7 t2m 72h
# SAMPLE_B = {"W8": "2025-04-24T00:00:00"} # W8 t2m 72h
# SAMPLE_B = {"W9": "2024-10-14T00:00:00"} # W9 t2m 72h
# SAMPLE_B = {"W10": "2025-05-05T12:00:00"} # W10 t2m 72h

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

    image = ax.imshow(
        difference,
        origin="lower",
        aspect="auto",
    )

    ax.set_title(
        f"{layer_name}: "
        "RMS difference between samples"
    )

    ax.set_xlabel("Feature-grid x")
    ax.set_ylabel("Feature-grid y")

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
        )


def plot_tensor_difference(
    a: torch.Tensor,
    b: torch.Tensor,
    title: str,
    output_file: Path,
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

    image = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
    )

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

    im0 = axes[0].imshow(
        a_map,
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    axes[0].set_title(f"Sample {list(SAMPLE_A.keys())[0]}")

    axes[1].imshow(
        b_map,
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    axes[1].set_title(f"Sample {list(SAMPLE_B.keys())[0]}")

    diff_limit = np.nanmax(
        np.abs(difference)
    )

    im2 = axes[2].imshow(
        difference,
        origin="lower",
        aspect="auto",
        vmin=-diff_limit,
        vmax=diff_limit,
    )

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


def main():
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
        feature_root_dir = (
            s.exp_dir
            / "inference"
            / f"{INFERENCE_START}_{INFERENCE_END}"
            / "feature_maps"
            / f"leadtime_{LEADTIME}"
        )
        out_dir = (
            s.plot_dir
            / "feature_map_analysis"
            / f"{list(SAMPLE_A.keys())[0]}_{list(SAMPLE_B.keys())[0]}"
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        sample_a = load_feature_sample(
            feature_root_dir,
            list(SAMPLE_A.values())[0],
            SAMPLE_WITHIN_INIT_A,
        )

        sample_b = load_feature_sample(
            feature_root_dir,
            list(SAMPLE_B.values())[0],
            SAMPLE_WITHIN_INIT_B,
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

        encoder_df = (
            build_encoder_cbam_comparison(
                comparison
            )
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

        channel_corr = (
            collect_channel_correlations(
                sample_a,
                sample_b,
            )
        )

        channel_summary = (
            channel_correlation_summary(
                channel_corr
            )
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
        )

        plot_tensor_difference(
            sample_a["target"],
            sample_b["target"],
            "Target A - B",
            out_dir
            / "target_difference.png",
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


if __name__ == "__main__":
    main()
