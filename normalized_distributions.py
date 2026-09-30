from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from scipy.stats import gaussian_kde

import torch

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from earthml import (
    MonthlyNormalize,
    Normalize,
    XarrayDataset,
    get_experiment_configs,
)

from train import make_train_test_datasets_for_leadtime


# ============================================================================
# Configuration
# ============================================================================

EXP_NAME = "weather_atmo"

EXPERIMENTS_ROOT = Path(
    f"/work/cmcc/jd19424/ML/MLBC/experiments/{EXP_NAME}"
)

OUTPUT_ROOT = Path(
    f"/work/cmcc/jd19424/ML/MLBC/plots/{EXP_NAME}"
)

# Change these to select the experiment(s) you want
FILTERS = dict(
    var_fc="t2m",
    var_an="t2m",
    region_name="ConUS",
)

# Optional subset of leadtimes. None -> all experiment leadtimes
LEADTIMES: list[int] | None = [72]

# Whether to include extrema in the CSV
COMPUTE_EXTREMA = True

# Enable monthly plots
PLOT_MONTHLY = True

# Distribution histogram or KDE
DISTRIBUTION_PLOT: Literal["hist", "kde", "both"] = "both"

# Histogram bins
BINS = 50

# Consistent split colors everywhere
SPLIT_COLORS = {
    "train": "C0",
    "val": "C1",
    "test": "C2",
}

# Initialization times to highlight
HIGHLIGHT_TIMES = [
    # t2m 72h worse rmse
    ("2025-02-13T00:00:00", "red", -0.545447),
    ("2025-01-03T00:00:00", "red", -0.339955),
    ("2025-02-06T00:00:00", "red", -0.314875),
    ("2025-02-18T00:00:00", "red", -0.259523),
    ("2025-02-16T00:00:00", "red", -0.194165),
    ("2025-01-01T12:00:00", "red", -0.190058),
    ("2025-02-12T12:00:00", "red", -0.16384),
    ("2025-04-24T00:00:00", "red", -0.14856),
    ("2025-03-29T00:00:00", "red", -0.142354),
    ("2025-05-25T12:00:00", "red", -0.13341),

    # t2m 72h best rmse
    ("2025-03-08T00:00:00", "blue", 1.2332),
    ("2025-02-22T00:00:00", "blue", 0.979847),
    ("2025-02-23T00:00:00", "blue", 0.977976),
    ("2025-02-02T00:00:00", "blue", 0.928439),
    ("2025-03-07T00:00:00", "blue", 0.917336),
    ("2025-02-21T00:00:00", "blue", 0.799746),
    ("2025-04-06T00:00:00", "blue", 0.796945),
    ("2025-02-01T00:00:00", "blue", 0.785558),
    ("2025-04-07T00:00:00", "blue", 0.751719),
    ("2025-03-06T00:00:00", "blue", 0.694854),
]


# ============================================================================
# Normalization
# ============================================================================

def make_input_normalizer(
    s,
    train_dataset: XarrayDataset,
):
    """
    Reproduce the input normalization used during training.

    IMPORTANT:
    the statistics are fitted ONLY on the training dataset.
    """
    if s.normalization == "monthly":
        norm_class = MonthlyNormalize
    elif s.normalization == "full":
        norm_class = Normalize
    else:
        raise ValueError(
            f"Unsupported normalization={s.normalization!r}"
        )

    # These channels are encodings and should not be normalized.
    n_excluded_channels = 0

    if (
        s.seasonal_encoding
        and s.channel_representation != "init_period"
    ):
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


# ============================================================================
# Statistics
# ============================================================================

def masked_field_stats(
    x: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Calculate spatial statistics independently for every channel.

    Parameters
    ----------
    x
        Normalized field with shape (C, H, W).

    mask
        Valid-data mask with shape (C, H, W).

    Returns
    -------
    mean, std, minimum, maximum
        Tensors with shape (C,).
    """
    if x.ndim != 3:
        raise ValueError(
            f"Expected x with shape (C,H,W), got {tuple(x.shape)}"
        )

    if mask.ndim != 3:
        raise ValueError(
            f"Expected mask with shape (C,H,W), got {tuple(mask.shape)}"
        )

    # x may contain extra encoder channels whereas the returned mask
    # corresponds to the physical data channels.
    n_channels = mask.shape[0]

    if x.shape[0] < n_channels:
        raise ValueError(
            f"x has fewer channels than mask: "
            f"{tuple(x.shape)} vs {tuple(mask.shape)}"
        )

    x = x[:n_channels]

    if x.shape != mask.shape:
        raise ValueError(
            f"x/mask shape mismatch after channel selection: "
            f"{tuple(x.shape)} != {tuple(mask.shape)}"
        )

    means = torch.full(
        (n_channels,),
        torch.nan,
        dtype=torch.float64,
    )
    stds = torch.full_like(means, torch.nan)
    minima = torch.full_like(means, torch.nan)
    maxima = torch.full_like(means, torch.nan)

    for channel in range(n_channels):
        valid = mask[channel].bool()

        values = x[channel][valid].double()

        if values.numel() == 0:
            continue

        means[channel] = values.mean()

        # Population std: consistent with the normalization convention.
        stds[channel] = values.std(unbiased=False)

        minima[channel] = values.min()
        maxima[channel] = values.max()

    return means, stds, minima, maxima


def collect_normalized_field_stats(
    dataset: XarrayDataset,
    *,
    split: str,
    leadtime: int,
) -> pd.DataFrame:
    """
    Collect per-field statistics from samples AFTER dataset normalization.

    Calling dataset[idx] invokes transform_x, so x here is exactly the
    normalized tensor produced for the neural-network input pipeline.
    """
    rows = []

    sample_times = get_sample_times(dataset)

    for idx in range(len(dataset)):
        x, _, mask, month = dataset[idx]

        x = x.detach().cpu()
        mask = mask.detach().cpu().bool()

        mean, std, minimum, maximum = masked_field_stats(
            x,
            mask,
        )

        time = pd.Timestamp(sample_times[idx])

        for channel in range(len(mean)):
            row = {
                "split": split,
                "leadtime": leadtime,
                "sample": idx,
                "time": time,
                "channel": channel,
                "month": int(month),
                "mean": float(mean[channel]),
                "std": float(std[channel]),
            }

            if COMPUTE_EXTREMA:
                row["min"] = float(minimum[channel])
                row["max"] = float(maximum[channel])

            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# Plotting
# ============================================================================

def common_bin_edges(
    groups: list[pd.Series],
    bins: int,
) -> np.ndarray:
    """
    Generate common histogram bins so train/val/test distributions are
    directly comparable.
    """
    values = np.concatenate([
        s.dropna().to_numpy()
        for s in groups
        if not s.dropna().empty
    ])

    if values.size == 0:
        return np.linspace(0.0, 1.0, bins + 1)

    lo = values.min()
    hi = values.max()

    if np.isclose(lo, hi):
        pad = max(abs(lo) * 0.05, 1e-6)
        lo -= pad
        hi += pad

    return np.linspace(lo, hi, bins + 1)


def plot_distribution(
    stats: pd.DataFrame,
    *,
    quantity: str,
    channel: int,
    output_path: Path,
    plot_type: Literal["hist", "kde", "both"] = "both",
    title_suffix: str = "",
) -> None:
    """
    Plot train / validation / test distributions.

    plot_type:
        "hist" -> histogram only
        "kde"  -> KDE only
        "both" -> histogram + KDE
    """
    if plot_type not in {"hist", "kde", "both"}:
        raise ValueError(
            f"Unsupported plot_type={plot_type!r}. "
            "Expected 'hist', 'kde', or 'both'."
        )

    channel_df = stats[
        stats["channel"] == channel
    ]

    available_splits = [
        split
        for split in ("train", "val", "test")
        if split in channel_df["split"].unique()
    ]

    groups = [
        channel_df.loc[
            channel_df["split"] == split,
            quantity,
        ].dropna().to_numpy()
        for split in available_splits
    ]

    # Common range for both histograms and KDE curves.
    all_values = np.concatenate(
        [
            values
            for values in groups
            if len(values) > 0
        ]
    )

    if all_values.size == 0:
        return

    lo = all_values.min()
    hi = all_values.max()

    if np.isclose(lo, hi):
        padding = max(abs(lo) * 0.05, 1e-6)
    else:
        padding = 0.05 * (hi - lo)

    lo -= padding
    hi += padding

    edges = np.linspace(
        lo,
        hi,
        BINS + 1,
    )

    kde_x = np.linspace(
        lo,
        hi,
        500,
    )

    fig, ax = plt.subplots(
        figsize=(8, 5),
    )

    for split, values in zip(
        available_splits,
        groups,
        strict=True,
    ):
        if len(values) == 0:
            continue

        label = f"{split} (n={len(values)})"
        color = SPLIT_COLORS[split]

        # ----------------------------------------------------------
        # Histogram
        # ----------------------------------------------------------

        if plot_type in {"hist", "both"}:
            ax.hist(
                values,
                bins=edges,
                histtype="step",
                linewidth=1.5,
                density=True,
                alpha=0.6 if plot_type == "both" else 1.0,
                color=color,
                label=label if plot_type == "hist" else None,
            )

        # ----------------------------------------------------------
        # KDE
        # ----------------------------------------------------------

        if (
            plot_type in {"kde", "both"}
            and len(values) > 1
            and not np.isclose(np.std(values), 0.0)
        ):
            kde = gaussian_kde(values)

            ax.plot(
                kde_x,
                kde(kde_x),
                linewidth=2.0,
                color=color,
                label=label,
            )

    # --------------------------------------------------------------
    # Reference lines
    # --------------------------------------------------------------

    if quantity == "mean":
        ax.axvline(
            0.0,
            linestyle="--",
            linewidth=1.0,
        )

    elif quantity == "std":
        ax.axvline(
            1.0,
            linestyle="--",
            linewidth=1.0,
        )

    ax.set_xlabel(
        f"Spatial {quantity} of normalized field"
    )
    ax.set_ylabel("Probability density")

    ax.set_title(
        f"Channel {channel}: normalized field {quantity}"
        f"{title_suffix}"
    )

    ax.legend()

    annotate_highlight_times_distribution(
        ax,
        stats,
        quantity=quantity,
        channel=channel,
        highlight_times=HIGHLIGHT_TIMES,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


def plot_mean_vs_std(
    stats: pd.DataFrame,
    *,
    channel: int,
    output_path: Path,
    title_suffix: str = "",
) -> None:
    channel_df = stats[
        stats["channel"] == channel
    ]

    fig, ax = plt.subplots(
        figsize=(7, 6),
    )

    for split in ("train", "val", "test"):
        df = channel_df[
            channel_df["split"] == split
        ]

        if df.empty:
            continue

        ax.scatter(
            df["mean"],
            df["std"],
            s=10,
            alpha=0.35,
            color=SPLIT_COLORS[split],
            label=split,
        )

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xlabel("Spatial mean")
    ax.set_ylabel("Spatial standard deviation")

    ax.set_title(
        f"Channel {channel}: normalized input fields"
        f"{title_suffix}"
    )

    ax.legend()

    annotate_highlight_times_scatter(
        ax,
        stats,
        channel=channel,
        highlight_times=HIGHLIGHT_TIMES,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


def plot_monthly_stats(
    stats: pd.DataFrame,
    output_dir: Path,
) -> None:
    monthly_output = output_dir / "monthly"

    monthly_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    channels = sorted(
        stats["channel"].unique()
    )

    months = sorted(
        stats["month"].dropna().unique()
    )

    for month in months:
        month = int(month)

        month_stats = stats[
            stats["month"] == month
        ]

        if month_stats.empty:
            continue

        month_name = MONTH_NAMES[month]

        month_dir = (
            monthly_output
            / f"{month:02d}_{month_name.lower()}"
        )

        month_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for channel in channels:
            channel_stats = month_stats[
                month_stats["channel"] == channel
            ]

            if channel_stats.empty:
                continue

            title_suffix = f" — {month_name}"

            plot_distribution(
                month_stats,
                quantity="mean",
                channel=channel,
                plot_type=DISTRIBUTION_PLOT,
                title_suffix=title_suffix,
                output_path=(
                    month_dir
                    / f"channel_{channel}_mean_distribution.png"
                ),
            )

            plot_distribution(
                month_stats,
                quantity="std",
                channel=channel,
                plot_type=DISTRIBUTION_PLOT,
                title_suffix=title_suffix,
                output_path=(
                    month_dir
                    / f"channel_{channel}_std_distribution.png"
                ),
            )

            plot_mean_vs_std(
                month_stats,
                channel=channel,
                title_suffix=title_suffix,
                output_path=(
                    month_dir
                    / f"channel_{channel}_mean_vs_std.png"
                ),
            )

def plot_stats(
    stats: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    channels = sorted(
        stats["channel"].unique()
    )

    # ----------------------------------------------------------
    # Full-period plots
    # ----------------------------------------------------------

    for channel in channels:
        plot_distribution(
            stats,
            quantity="mean",
            channel=channel,
            plot_type=DISTRIBUTION_PLOT,
            output_path=(
                output_dir
                / f"channel_{channel}_mean_distribution.png"
            ),
        )

        plot_distribution(
            stats,
            quantity="std",
            channel=channel,
            plot_type=DISTRIBUTION_PLOT,
            output_path=(
                output_dir
                / f"channel_{channel}_std_distribution.png"
            ),
        )

        plot_mean_vs_std(
            stats,
            channel=channel,
            output_path=(
                output_dir
                / f"channel_{channel}_mean_vs_std.png"
            ),
        )

    # ----------------------------------------------------------
    # Monthly plots
    # ----------------------------------------------------------

    if PLOT_MONTHLY:
        plot_monthly_stats(
            stats,
            output_dir,
        )

# ============================================================================
# Annotation
# ============================================================================

def annotate_highlight_times_distribution(
    ax,
    stats: pd.DataFrame,
    *,
    quantity: str,
    channel: int,
    highlight_times: list[tuple[str, str, float]],
) -> None:
    channel_df = stats[
        stats["channel"] == channel
    ].copy()

    channel_df["time"] = pd.to_datetime(
        channel_df["time"]
    )

    for requested_time, base_color, shade_value in highlight_times:
        requested = pd.Timestamp(requested_time)

        matches = channel_df[
            channel_df["time"] == requested
        ]

        for _, row in matches.iterrows():
            value = row[quantity]

            cmap = mcolors.LinearSegmentedColormap.from_list(
                "highlight_shade",
                ["white", base_color],
            )

            line_color = cmap(
                np.clip(abs(shade_value), 0.0, 1.0)
            )

            ax.axvline(
                value,
                color=line_color,
                linestyle="-",
                linewidth=1.5,
                alpha=0.9,
            )

            ax.annotate(
                requested.strftime("%Y-%m-%d %H:%M"),
                xy=(value, 1.0),
                xycoords=(
                    "data",
                    "axes fraction",
                ),
                xytext=(4, -4),
                textcoords="offset points",
                rotation=90,
                va="top",
                ha="left",
                color="black",
                fontsize=8,
            )

def annotate_highlight_times_scatter(
    ax,
    stats: pd.DataFrame,
    *,
    channel: int,
    highlight_times: list[tuple[str, str, float]],
) -> None:
    channel_df = stats[
        stats["channel"] == channel
    ].copy()

    channel_df["time"] = pd.to_datetime(
        channel_df["time"]
    )

    for requested_time, base_color, shade_value in highlight_times:
        requested = pd.Timestamp(requested_time)

        matches = channel_df[
            channel_df["time"] == requested
        ]

        for _, row in matches.iterrows():
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "highlight_shade",
                ["white", base_color],
            )

            ann_color = cmap(
                np.clip(abs(shade_value), 0.0, 1.0)
            )

            ax.scatter(
                row["mean"],
                row["std"],
                s=70,
                color=ann_color,
                edgecolor="black",
                linewidth=1.0,
                zorder=10,
            )

            ax.annotate(
                requested.strftime("%Y-%m-%d %H:%M"),
                xy=(
                    row["mean"],
                    row["std"],
                ),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
                color="black",
            )

# ============================================================================
# Summary
# ============================================================================

def print_summary(
    stats: pd.DataFrame,
) -> None:
    summary = (
        stats
        .groupby(
            [
                "leadtime",
                "channel",
                "split",
            ]
        )[
            [
                "mean",
                "std",
            ]
        ]
        .agg(
            [
                "mean",
                "std",
                "median",
                "min",
                "max",
            ]
        )
    )

    print()
    print(summary.to_string())
    print()


# ============================================================================
# Dataset construction
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


def process_leadtime(
    s,
    leadtime: int,
) -> pd.DataFrame:
    print("=" * 80)
    print(
        f"{s.output_name}: "
        f"leadtime={leadtime} {s.leadtime_unit}"
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

    if not isinstance(
        train_dataset,
        XarrayDataset,
    ):
        raise TypeError(
            "Expected XarrayDataset for training data"
        )

    if (
        val_dataset is not None
        and not isinstance(val_dataset, XarrayDataset)
    ):
        raise TypeError(
            "Expected XarrayDataset for validation data"
        )

    if not isinstance(
        test_dataset,
        XarrayDataset,
    ):
        raise TypeError(
            "Expected XarrayDataset for test data"
        )

    # ----------------------------------------------------------------
    # Fit exactly once using TRAINING data.
    # ----------------------------------------------------------------

    normalize_input = make_input_normalizer(
        s,
        train_dataset,
    )

    # ----------------------------------------------------------------
    # Apply the SAME train statistics to every split.
    # ----------------------------------------------------------------

    train_dataset.transform_x = normalize_input

    if val_dataset is not None:
        val_dataset.transform_x = normalize_input

    test_dataset.transform_x = normalize_input

    # We do not need transform_y because this diagnostic only looks
    # at the network input.
    train_dataset.transform_y = None

    if val_dataset is not None:
        val_dataset.transform_y = None

    test_dataset.transform_y = None

    # ----------------------------------------------------------------
    # Collect stats.
    # ----------------------------------------------------------------

    frames = [
        collect_normalized_field_stats(
            train_dataset,
            split="train",
            leadtime=leadtime,
        )
    ]

    if val_dataset is not None:
        frames.append(
            collect_normalized_field_stats(
                val_dataset,
                split="val",
                leadtime=leadtime,
            )
        )

    frames.append(
        collect_normalized_field_stats(
            test_dataset,
            split="test",
            leadtime=leadtime,
        )
    )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def get_sample_times(
    dataset: XarrayDataset,
) -> np.ndarray:
    """
    Return one initialization time for every dataset sample.

    samples_per_init handles cases where realizations are represented
    as independent samples.
    """
    time_dim = dataset.input_ds.earthml.guessed_dims.time

    if time_dim is None:
        raise ValueError(
            "Could not determine input time dimension."
        )

    init_times = dataset.input_ds[time_dim].values

    sample_times = np.repeat(
        init_times,
        dataset.samples_per_init,
    )

    if len(sample_times) != len(dataset):
        raise RuntimeError(
            "Cannot align initialization times with dataset samples: "
            f"{len(sample_times)=}, {len(dataset)=}"
        )

    return sample_times

# ============================================================================
# Main
# ============================================================================

def main() -> None:
    settings = get_experiment_configs(
        EXPERIMENTS_ROOT,
        **FILTERS,
    )

    if not settings:
        raise RuntimeError(
            "No matching experiments found."
        )

    print(
        f"Found {len(settings)} matching experiment(s)."
    )

    for s in settings:
        experiment_output = (
            OUTPUT_ROOT
            / s.output_name
            / "normalized_input_diagnostics"
        )

        experiment_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        leadtimes = (
            [int(x) for x in s.leadtimes]
            if LEADTIMES is None
            else LEADTIMES
        )

        all_stats = []

        for leadtime in leadtimes:
            stats = process_leadtime(
                s,
                leadtime,
            )

            all_stats.append(stats)

            leadtime_output = (
                experiment_output
                / f"leadtime_{leadtime}"
            )

            leadtime_output.mkdir(
                parents=True,
                exist_ok=True,
            )

            stats.to_csv(
                leadtime_output / "field_stats.csv",
                index=False,
            )

            plot_stats(
                stats,
                leadtime_output,
            )

            print_summary(stats)

        all_stats = pd.concat(
            all_stats,
            ignore_index=True,
        )

        all_stats.to_csv(
            experiment_output / "field_stats_all_leadtimes.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
