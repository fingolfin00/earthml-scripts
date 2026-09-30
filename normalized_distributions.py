from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from scipy.stats import gaussian_kde
import torch

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from earthml import (
    ClimPeriod,
    LeadtimeUnit,
    MonthlyNormalize,
    Normalize,
    XarrayDataset,
    get_experiment_configs,
    get_and_subset_datasets,
)

from earthml.metrics import (
    get_metrics,
    build_metric_improvements,
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

INTERPOLATE_ANALYSIS = False # weather

# ----------------------------------------------------------------------------
# Experiment selection
# ----------------------------------------------------------------------------

FILTERS = dict(
    var_fc="t2m",
    var_an="t2m",
    region_name="ConUS",
)

# None -> all experiment leadtimes
LEADTIMES: list[int] | None = [72]

# ----------------------------------------------------------------------------
# Field statistics
# ----------------------------------------------------------------------------

COMPUTE_EXTREMA = True

# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

PLOT_MONTHLY = False

DISTRIBUTION_PLOT: Literal[
    "hist",
    "kde",
    "both",
] = "kde"

BINS = 50

KDE_POINTS = 500

# Splits shown in the distribution and scatter plots.
# Any combination of: "train", "val", "test".
PLOT_SPLITS: tuple[
    Literal["train", "val", "test"],
    ...
] = (
    # "train",
    # "val",
    "test",
)

# Number of best/worst performance timesteps to annotate.
# Performance is currently available for the test split only.
ANNOTATE_BEST_TIMESTEPS = 4
ANNOTATE_WORST_TIMESTEPS = 4
ANNOTATION_FONTSIZE = 5

ANNOTATION_DATE_FORMAT = "%Y-%m-%d:%H:%M"

SPLIT_COLORS = {
    "train": "C0",
    "val": "C1",
    "test": "C2",
}

# ----------------------------------------------------------------------------
# Performance coloring
# ----------------------------------------------------------------------------

COLOR_BY_PERFORMANCE = True
PERFORMANCE_METRIC = "rmse"
PERFORMANCE_IMPROVEMENT_UNIT: Literal[
    "Δ",
    "%",
] = "Δ"

# Positive improvement = corrected forecast is better.
#
# RdBu:
#   negative -> red
#   zero     -> white
#   positive -> blue
PERFORMANCE_CMAP = "RdBu"

# Robust symmetric color range:
# e.g. 0.98 ignores the outermost 2% when setting vmax.
# Values outside the range are clipped to the endpoint colors.
PERFORMANCE_COLOR_QUANTILE = 0.90
PERFORMANCE_FILL_ALPHA = 0.95

# Scale factor applied to the automatic local-performance bandwidth.
# 1.0 is a good starting point.
PERFORMANCE_BANDWIDTH_SCALE = 1.0

# Scatter settings
PERFORMANCE_SCATTER_SIZE = 14
PERFORMANCE_SCATTER_ALPHA = 0.80

# Non-performance-colored scatter points use the same neutral appearance
# regardless of whether they belong to train, validation, or test.
SCATTER_COLOR = "0.45"
SCATTER_SIZE = 12
SCATTER_ALPHA = 0.35

# ============================================================================
# Month names
# ============================================================================

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

# ============================================================================
# Normalization
# ============================================================================

def make_input_normalizer(
    s,
    train_dataset: XarrayDataset,
):
    """
    Reproduce input normalization used during training.
    The normalization statistics are fitted only on training data.
    """
    if s.normalization == "monthly":
        norm_class = MonthlyNormalize
    elif s.normalization == "full":
        norm_class = Normalize
    else:
        raise ValueError(
            f"Unsupported normalization={s.normalization!r}"
        )

    # Encoding channels should not be normalized
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
    }

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
# Normalized field statistics
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
    Calculate spatial statistics independently for each physical channel.
    Parameters
    ----------
    x
        Normalized field with shape (C, H, W).
    mask
        Valid-data mask with shape (C, H, W).
    Returns
    -------
    mean, std, minimum, maximum
        Each has shape (C,).
    """
    if x.ndim != 3:
        raise ValueError(
            f"Expected x with shape (C,H,W), got {tuple(x.shape)}"
        )
    if mask.ndim != 3:
        raise ValueError(
            f"Expected mask with shape (C,H,W), got {tuple(mask.shape)}"
        )

    n_channels = mask.shape[0]

    if x.shape[0] < n_channels:
        raise ValueError(
            "x has fewer channels than mask: "
            f"{tuple(x.shape)} vs {tuple(mask.shape)}"
        )
    # Remove any additional encoder channels for these diagnostics.
    x = x[:n_channels]
    if x.shape != mask.shape:
        raise ValueError(
            "x/mask shape mismatch after channel selection: "
            f"{tuple(x.shape)} != {tuple(mask.shape)}"
        )
    means = torch.full(
        (n_channels,),
        torch.nan,
        dtype=torch.float64,
    )
    stds = torch.full_like(
        means,
        torch.nan,
    )
    minima = torch.full_like(
        means,
        torch.nan,
    )
    maxima = torch.full_like(
        means,
        torch.nan,
    )
    for channel in range(n_channels):
        valid = mask[channel].bool()
        values = x[channel][valid].double()
        if values.numel() == 0:
            continue
        means[channel] = values.mean()
        # Population std, consistent with the normalizer.
        stds[channel] = values.std(
            unbiased=False,
        )
        minima[channel] = values.min()
        maxima[channel] = values.max()
    return (
        means,
        stds,
        minima,
        maxima,
    )


def collect_normalized_field_stats(
    dataset: XarrayDataset,
    *,
    split: str,
    leadtime: int,
) -> pd.DataFrame:
    """
    Collect field statistics after input normalization.
    dataset[idx] invokes transform_x, so x here is the actual tensor
    immediately before the NN machinery.
    """
    rows = []
    sample_times = get_sample_times(
        dataset
    )
    for idx in range(len(dataset)):
        x, y, mask, month = dataset[idx]

        x = x.detach().cpu()
        y = y.detach().cpu()

        mask = (
            mask
            .detach()
            .cpu()
            .bool()
        )

        (
            mean,
            std,
            minimum,
            maximum,
        ) = masked_field_stats(
            x,
            mask,
        )

        (
            target_mean,
            target_std,
            _,
            _,
        ) = masked_field_stats(
            y,
            mask,
        )
        x = x.detach().cpu()
        mask = (
            mask
            .detach()
            .cpu()
            .bool()
        )
        (
            mean,
            std,
            minimum,
            maximum,
        ) = masked_field_stats(
            x,
            mask,
        )
        time = pd.Timestamp(
            sample_times[idx]
        )
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
                row["min"] = float(
                    minimum[channel]
                )
                row["max"] = float(
                    maximum[channel]
                )
            rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# Metric performance
# ============================================================================

def get_metric_improvement_timeseries(
    s,
    *,
    leadtime: int,
    metric: str,
    improvement_unit: Literal["Δ", "%"],
) -> pd.DataFrame:
    """
    Calculate FC -> MLFC metric improvement for every test initialization.
    The time coordinate is already initialization time, so NO lead-time
    shift is performed.
    Positive improvement always means MLFC is better.
    """
    time_range = (
        s.test_start,
        s.test_end,
    )
    leadtime_units = (
        s.leadtime_unit
        if isinstance(s.leadtime_unit, LeadtimeUnit)
        else LeadtimeUnit(s.leadtime_unit)
    )
    clim_period = (
        s.clim_period
        if isinstance(s.clim_period, ClimPeriod)
        else ClimPeriod(s.clim_period)
    )
    lat_lon = (
        list(s.region.values())
        if s.region is not None
        else [None, None]
    )
    valid_lat_range = lat_lon[0]
    valid_lon_range = lat_lon[1]
    fc, an, mlfc = get_and_subset_datasets(
        s,
        leadtime_units=leadtime_units,
        lat_range=valid_lat_range,
        lon_range=valid_lon_range,
        time_range=time_range,
        interpolate=INTERPOLATE_ANALYSIS,
        mlfc_path=None,
    )
    if mlfc is None:
        raise ValueError(
            "MLFC dataset is required to calculate "
            f"{metric!r} improvement."
        )
    leadtime_dim = (
        fc.earthml.guessed_dims.leadtime
    )
    if leadtime_dim is None:
        raise ValueError(
            "Could not determine leadtime dimension."
        )
    # Ensure all three datasets use only the requested lead.
    fc = fc.sel({
        leadtime_dim: [leadtime]
    })
    an = an.sel({
        leadtime_dim: [leadtime]
    })
    mlfc = mlfc.sel({
        leadtime_dim: [leadtime]
    })

    # ------------------------------------------------------------
    # Baseline forecast metric
    # ------------------------------------------------------------

    fc_metrics = get_metrics(
        an=an,
        fc=fc,
        var=s.var_fc,
        metric_kind="timeseries",
        leadtime_agg="single",
        realization_agg=True,
        an_clim=None,
        fc_clim=None,
        orography_path=None,
        metrics=[metric],
        leadtime_windows=s.seasonal_leadtime_windows,
        leadtime_agg_coord="leadtime",
        clim_period=clim_period,
        period_reference="init",
        period_dim=f"init_{clim_period.value}",
        periods_requested=["all"],
        leadtime_unit=leadtime_units,
        align=False,
        fair_correction=False,
    )

    # ------------------------------------------------------------
    # Corrected forecast metric
    # ------------------------------------------------------------

    mlfc_metrics = get_metrics(
        an=an,
        fc=mlfc,
        var=s.var_fc,
        metric_kind="timeseries",
        leadtime_agg="single",
        realization_agg=True,
        an_clim=None,
        fc_clim=None,
        orography_path=None,
        metrics=[metric],
        leadtime_windows=s.seasonal_leadtime_windows,
        leadtime_agg_coord="leadtime",
        clim_period=clim_period,
        period_reference="init",
        period_dim=f"init_{clim_period.value}",
        periods_requested=["all"],
        leadtime_unit=leadtime_units,
        align=False,
        fair_correction=False,
    )

    # ------------------------------------------------------------
    # Improvement
    # ------------------------------------------------------------

    improvements = build_metric_improvements(
        fc_metrics,
        mlfc_metrics,
        metric=metric,
        baseline_model="fc",
        target_model="mlfc",
        improvement_units=(
            improvement_unit,
        ),
    )
    if len(improvements) != 1:
        raise RuntimeError(
            "Expected exactly one improvement DataArray, "
            f"got {list(improvements)}"
        )
    comparison_model, improvement_da = next(
        iter(improvements.items())
    )
    print(
        f"Using performance metric: "
        f"{comparison_model}"
    )

    # Select requested lead
    if (
        "leadtime" in improvement_da.dims
        or "leadtime" in improvement_da.coords
    ):
        improvement_da = improvement_da.sel(
            leadtime=leadtime
        )

    # Remove singleton dimensions such as period="all".
    improvement_da = improvement_da.squeeze(
        drop=True
    )
    time_dim = (
        improvement_da
        .earthml
        .guessed_dims
        .time
    )
    if time_dim is None:
        raise ValueError(
            "Could not determine time dimension "
            "of metric improvement."
        )

    # After selecting the lead and period, this should be a
    # one-dimensional timeseries.
    extra_dims = [
        dim
        for dim in improvement_da.dims
        if dim != time_dim
    ]
    if extra_dims:
        raise ValueError(
            "Metric improvement still contains unexpected "
            f"dimensions: {improvement_da.dims}"
        )
    metric_df = pd.DataFrame({
        "time": pd.to_datetime(
            improvement_da[
                time_dim
            ].values
        ),
        "metric_improvement": (
            improvement_da.values
        ),
    })
    metric_df["metric_improvement"] = (
        pd.to_numeric(
            metric_df["metric_improvement"],
            errors="coerce",
        )
    )
    metric_df = (
        metric_df
        .drop_duplicates(
            subset="time"
        )
        .sort_values("time")
        .reset_index(drop=True)
    )
    return metric_df


def add_metric_performance(
    stats: pd.DataFrame,
    metric_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add metric improvement to the test rows.
    Train/validation rows remain NaN because performance is calculated
    over the configured test interval.
    """
    stats = stats.copy()
    stats["time"] = pd.to_datetime(
        stats["time"]
    )
    metric_df = metric_df.copy()
    metric_df["time"] = pd.to_datetime(
        metric_df["time"]
    )
    # Only attach performance to test samples.
    test_stats = stats[
        stats["split"] == "test"
    ].merge(
        metric_df,
        on="time",
        how="left",
        validate="many_to_one",
    )
    other_stats = stats[
        stats["split"] != "test"
    ].copy()
    other_stats[
        "metric_improvement"
    ] = np.nan
    result = pd.concat(
        [
            other_stats,
            test_stats,
        ],
        ignore_index=True,
    )
    # Check alignment.
    test_rows = result[
        result["split"] == "test"
    ]
    n_missing = int(
        test_rows[
            "metric_improvement"
        ].isna().sum()
    )
    if n_missing:
        print(
            f"WARNING: {n_missing}/{len(test_rows)} "
            "test rows have no matching metric improvement."
        )
    return result

# ============================================================================
# Performance coloring
# ============================================================================

def make_performance_norm(
    stats: pd.DataFrame,
) -> mcolors.TwoSlopeNorm | None:
    """
    Build a robust symmetric normalization centered at zero.
    """
    if (
        "metric_improvement"
        not in stats.columns
    ):
        return None
    values = (
        stats.loc[
            stats["split"] == "test",
            "metric_improvement",
        ]
        .dropna()
        .to_numpy(dtype=float)
    )
    values = values[
        np.isfinite(values)
    ]
    if values.size == 0:
        return None
    abs_values = np.abs(values)
    vmax = float(
        np.quantile(
            abs_values,
            PERFORMANCE_COLOR_QUANTILE,
        )
    )
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(
            np.nanmax(abs_values)
        )
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return mcolors.TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0.0,
        vmax=vmax,
    )


def local_performance_on_grid(
    values: np.ndarray,
    performance: np.ndarray,
    x_grid: np.ndarray,
) -> np.ndarray:
    """
    Smooth local average performance as a function of distribution x.
    Uses Gaussian distance weighting in field-statistic space.
    """
    valid = (
        np.isfinite(values)
        & np.isfinite(performance)
    )
    values = values[valid]
    performance = performance[valid]
    if len(values) < 2:
        return np.full(
            x_grid.shape,
            np.nan,
            dtype=float,
        )
    std = np.std(
        values,
        ddof=0,
    )
    if not np.isfinite(std) or std <= 0:
        return np.full(
            x_grid.shape,
            np.nan,
            dtype=float,
        )

    # Scott-like 1-D bandwidth
    bandwidth = (
        std
        * len(values) ** (-1.0 / 5.0)
        * PERFORMANCE_BANDWIDTH_SCALE
    )
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        return np.full(
            x_grid.shape,
            np.nan,
            dtype=float,
        )
    distance = (
        x_grid[:, None]
        - values[None, :]
    ) / bandwidth
    weights = np.exp(
        -0.5 * distance**2
    )
    denominator = weights.sum(
        axis=1
    )
    numerator = (
        weights
        * performance[None, :]
    ).sum(
        axis=1
    )
    result = np.full(
        x_grid.shape,
        np.nan,
        dtype=float,
    )
    valid_denominator = (
        denominator > 1e-12
    )
    result[
        valid_denominator
    ] = (
        numerator[
            valid_denominator
        ]
        / denominator[
            valid_denominator
        ]
    )
    return result


def performance_color(
    value,
    *,
    cmap,
    norm,
):
    normalized = np.clip(
        norm(value),
        0.0,
        1.0,
    )
    return cmap(normalized)


def fill_density_by_performance(
    ax,
    *,
    x_grid: np.ndarray,
    density: np.ndarray,
    stats: pd.DataFrame,
    quantity: str,
    cmap,
    norm: mcolors.Normalize,
) -> None:
    """
    Fill the area below a test density using local average performance.
    """
    if "metric_improvement" not in stats.columns:
        return
    test_df = stats[
        stats["split"] == "test"
    ]
    if test_df.empty:
        return
    values = test_df[
        quantity
    ].to_numpy(
        dtype=float
    )
    performance = test_df[
        "metric_improvement"
    ].to_numpy(
        dtype=float
    )
    local_performance = (
        local_performance_on_grid(
            values,
            performance,
            x_grid,
        )
    )
    for i in range(
        len(x_grid) - 1
    ):
        perf = (
            local_performance[i]
            + local_performance[i + 1]
        ) / 2
        if not np.isfinite(perf):
            continue
        ax.fill_between(
            x_grid[i:i + 2],
            0.0,
            density[i:i + 2],
            color=performance_color(
                perf,
                cmap=cmap,
                norm=norm,
            ),
            alpha=PERFORMANCE_FILL_ALPHA,
            linewidth=0,
            zorder=1,
        )

# ============================================================================
# Performance extrema
# ============================================================================

def get_performance_extrema(
    stats: pd.DataFrame,
    *,
    n_best: int,
    n_worst: int,
) -> pd.DataFrame:
    """
    Return unique best/worst timesteps according to metric improvement.

    Positive improvement means the corrected forecast is better, so the
    largest values are the best timesteps and the smallest values are the
    worst timesteps.

    Performance is currently attached only to test rows. If one initialization
    corresponds to multiple samples, it still appears only once here.
    """
    columns = [
        "time",
        "metric_improvement",
        "extreme",
        "rank",
    ]
    if "metric_improvement" not in stats.columns:
        return pd.DataFrame(columns=columns)
    perf = (
        stats.loc[
            stats["metric_improvement"].notna(),
            [
                "time",
                "metric_improvement",
            ],
        ]
        .drop_duplicates(subset="time")
        .sort_values("metric_improvement")
        .reset_index(drop=True)
    )
    if perf.empty:
        return pd.DataFrame(columns=columns)
    n_best = max(int(n_best), 0)
    n_worst = max(int(n_worst), 0)
    frames = []
    if n_worst > 0:
        worst = perf.head(n_worst).copy()
        worst["extreme"] = "worst"
        worst["rank"] = np.arange(1, len(worst) + 1)
        frames.append(worst)
    if n_best > 0:
        best = (
            perf.tail(n_best)
            .sort_values(
                "metric_improvement",
                ascending=False,
            )
            .copy()
        )
        best["extreme"] = "best"
        best["rank"] = np.arange(1, len(best) + 1)
        frames.append(best)
    if not frames:
        return pd.DataFrame(columns=columns)
    return (
        pd.concat(
            frames,
            ignore_index=True,
        )
        .drop_duplicates(subset="time")
        .reset_index(drop=True)
    )


def format_extreme_annotation(row: pd.Series) -> str:
    """Compact label used by both distribution and scatter annotations."""
    prefix = (
        "B"
        if row["extreme"] == "best"
        else "W"
    )
    time = pd.Timestamp(row["time"])
    return (
        f"{prefix}{int(row['rank'])} "
        f"{time.strftime(ANNOTATION_DATE_FORMAT)}"
    )

# ============================================================================
# Plotting helpers
# ============================================================================

def distribution_range(
    groups: list[np.ndarray],
) -> tuple[float, float]:
    arrays = [
        values
        for values in groups
        if len(values) > 0
    ]
    if not arrays:
        return 0.0, 1.0
    all_values = np.concatenate(
        arrays
    )
    lo = float(
        np.nanmin(all_values)
    )
    hi = float(
        np.nanmax(all_values)
    )
    if np.isclose(lo, hi):
        padding = max(
            abs(lo) * 0.05,
            1e-6,
        )
    else:
        padding = 0.05 * (
            hi - lo
        )
    return (
        lo - padding,
        hi + padding,
    )


def histogram_density_on_grid(
    values: np.ndarray,
    *,
    edges: np.ndarray,
    x_grid: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the histogram density as a piecewise-constant function
    on x_grid.
    Used as the performance-fill envelope in histogram-only mode.
    """
    density, _ = np.histogram(
        values,
        bins=edges,
        density=True,
    )
    indices = np.searchsorted(
        edges,
        x_grid,
        side="right",
    ) - 1
    result = np.zeros_like(
        x_grid,
        dtype=float,
    )
    valid = (
        (indices >= 0)
        & (indices < len(density))
    )
    result[valid] = density[
        indices[valid]
    ]
    return result


def add_performance_colorbar(
    fig,
    ax,
    *,
    cmap,
    norm,
) -> None:
    sm = plt.cm.ScalarMappable(
        norm=norm,
        cmap=cmap,
    )
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=ax,
        pad=0.02,
    )
    unit = (
        PERFORMANCE_IMPROVEMENT_UNIT
    )
    cbar.set_label(
        f"{PERFORMANCE_METRIC.upper()} "
        f"improvement ({unit})"
    )

def annotate_nonoverlapping(
    ax,
    text: str,
    *,
    xy: tuple[float, float],
    occupied: list,
    rotation: float = 0.0,
) -> None:
    """
    Annotate a point using a leader line while attempting to avoid
    overlap with previously placed annotations.
    """
    fig = ax.figure

    # Need a renderer for label bounding-box calculations.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    x, _ = xy
    x_center = np.mean(ax.get_xlim())

    # Prefer putting labels outward from the plot center.
    direction = -1 if x < x_center else 1

    candidate_offsets = [
        (direction * 8, 8),
        (-direction * 8, 8),
        (direction * 12, 16),
        (-direction * 12, 16),
        (direction * 18, 24),
        (-direction * 18, 24),
        (direction * 24, 32),
        (-direction * 24, 32),
        (direction * 32, 40),
        (-direction * 32, 40),
        (direction * 40, 50),
        (-direction * 40, 50),
    ]

    axes_bbox = ax.get_window_extent(renderer)

    annotation = None

    for x_offset, y_offset in candidate_offsets:
        candidate = ax.annotate(
            text,
            xy=xy,
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha="left" if x_offset > 0 else "right",
            va="bottom",
            fontsize=ANNOTATION_FONTSIZE,
            rotation=rotation,
            color="black",
            bbox={
                "boxstyle": "round,pad=0.10",
                "facecolor": "white",
                "edgecolor": "0.5",
                "linewidth": 0.3,
                "alpha": 0.75,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": "0.25",
                "linewidth": 0.6,
                "shrinkA": 1,
                "shrinkB": 2,
            },
            annotation_clip=True,
            zorder=8,
        )

        bbox = candidate.get_window_extent(
            renderer
        ).expanded(
            1.05,
            1.10,
        )

        inside_axes = (
            bbox.x0 >= axes_bbox.x0
            and bbox.x1 <= axes_bbox.x1
            and bbox.y0 >= axes_bbox.y0
            and bbox.y1 <= axes_bbox.y1
        )

        overlaps = any(
            bbox.overlaps(existing)
            for existing in occupied
        )

        if inside_axes and not overlaps:
            annotation = candidate
            occupied.append(bbox)
            break

        candidate.remove()

    # If every candidate overlaps, keep the last-resort position.
    if annotation is None:
        x_offset, y_offset = candidate_offsets[-1]

        annotation = ax.annotate(
            text,
            xy=xy,
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha="left" if x_offset > 0 else "right",
            va="bottom",
            fontsize=ANNOTATION_FONTSIZE,
            rotation=rotation,
            color="black",
            bbox={
                "boxstyle": "round,pad=0.10",
                "facecolor": "white",
                "edgecolor": "0.5",
                "linewidth": 0.3,
                "alpha": 0.75,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": "0.25",
                "linewidth": 0.6,
                "shrinkA": 1,
                "shrinkB": 2,
            },
            annotation_clip=True,
            zorder=8,
        )

        occupied.append(
            annotation.get_window_extent(renderer).expanded(
                1.05,
                1.10,
            )
        )

# ============================================================================
# Distribution plot
# ============================================================================

def plot_distribution(
    stats: pd.DataFrame,
    *,
    quantity: str,
    channel: int,
    output_path: Path,
    plot_type: Literal[
        "hist",
        "kde",
        "both",
    ] = "both",
    title_suffix: str = "",
) -> None:
    if plot_type not in {
        "hist",
        "kde",
        "both",
    }:
        raise ValueError(
            f"Unsupported plot_type={plot_type!r}"
        )
    channel_df = stats[
        stats["channel"] == channel
    ]
    available_splits = [
        split
        for split in PLOT_SPLITS
        if split in channel_df["split"].unique()
    ]
    if not available_splits:
        return
    groups = {
        split: (
            channel_df.loc[
                channel_df["split"] == split,
                quantity,
            ]
            .dropna()
            .to_numpy(dtype=float)
        )
        for split in available_splits
    }
    lo, hi = distribution_range(
        list(groups.values())
    )
    edges = np.linspace(
        lo,
        hi,
        BINS + 1,
    )
    x_grid = np.linspace(
        lo,
        hi,
        KDE_POINTS,
    )
    fig, ax = plt.subplots(
        figsize=(8, 5),
    )
    # ------------------------------------------------------------
    # Performance color setup
    # ------------------------------------------------------------
    performance_norm = (
        make_performance_norm(
            channel_df
        )
        if COLOR_BY_PERFORMANCE
        else None
    )
    performance_cmap = plt.get_cmap(
        PERFORMANCE_CMAP
    )
    test_density_for_fill = None
    # ------------------------------------------------------------
    # Plot each selected split
    # ------------------------------------------------------------
    for split in available_splits:
        values = groups[split]
        if len(values) == 0:
            continue
        color = SPLIT_COLORS[split]
        median = float(
            np.median(values)
        )
        label = (
            f"{split} "
            f"(n={len(values)}, "
            f"median={median:.3f})"
        )
        # --------------------------------------------------------
        # Histogram
        # --------------------------------------------------------
        if plot_type in {
            "hist",
            "both",
        }:
            ax.hist(
                values,
                bins=edges,
                histtype="step",
                linewidth=1.5,
                density=True,
                alpha=(
                    0.7
                    if plot_type == "both"
                    else 1.0
                ),
                color=color,
                label=(
                    label
                    if plot_type == "hist"
                    else None
                ),
                zorder=4,
            )
        # --------------------------------------------------------
        # KDE
        # --------------------------------------------------------
        kde_density = None
        if (
            plot_type in {
                "kde",
                "both",
            }
            and len(values) > 1
            and not np.isclose(
                np.std(values),
                0.0,
            )
        ):
            kde = gaussian_kde(values)
            kde_density = kde(
                x_grid
            )
            ax.plot(
                x_grid,
                kde_density,
                linewidth=2.0,
                color=color,
                label=label,
                zorder=5,
            )
        # Plot split median.
        ax.axvline(
            median,
            color=color,
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
            zorder=3,
        )
        # --------------------------------------------------------
        # Save test distribution envelope for performance fill
        # --------------------------------------------------------
        if split == "test":
            if kde_density is not None:
                test_density_for_fill = kde_density
            elif plot_type == "hist":
                test_density_for_fill = (
                    histogram_density_on_grid(
                        values,
                        edges=edges,
                        x_grid=x_grid,
                    )
                )
    # ------------------------------------------------------------
    # Performance-colored fill
    # ------------------------------------------------------------
    if (
        "test" in available_splits
        and COLOR_BY_PERFORMANCE
        and performance_norm is not None
        and test_density_for_fill is not None
    ):
        fill_density_by_performance(
            ax,
            x_grid=x_grid,
            density=test_density_for_fill,
            stats=channel_df,
            quantity=quantity,
            cmap=performance_cmap,
            norm=performance_norm,
        )
        add_performance_colorbar(
            fig,
            ax,
            cmap=performance_cmap,
            norm=performance_norm,
        )

    # ------------------------------------------------------------
    # Best/worst test-timestep annotations
    # ------------------------------------------------------------

    if (
        "test" in available_splits
        and test_density_for_fill is not None
    ):
        extrema = get_performance_extrema(
            channel_df,
            n_best=ANNOTATE_BEST_TIMESTEPS,
            n_worst=ANNOTATE_WORST_TIMESTEPS,
        )

        occupied_annotations = []

        for _, row in extrema.iterrows():
            time = pd.Timestamp(row["time"])

            timestep_rows = channel_df[
                (channel_df["split"] == "test")
                & (channel_df["time"] == time)
            ]

            if timestep_rows.empty:
                continue

            x = float(
                timestep_rows[quantity].mean()
            )

            y = float(
                np.interp(
                    x,
                    x_grid,
                    test_density_for_fill,
                )
            )

            if not (
                np.isfinite(x)
                and np.isfinite(y)
            ):
                continue

            perf = float(
                row["metric_improvement"]
            )

            marker_color = performance_color(
                perf,
                cmap=performance_cmap,
                norm=performance_norm,
            )

            ax.plot(
                x,
                y,
                marker="o",
                markersize=4.0,
                markerfacecolor=marker_color,
                markeredgecolor="black",
                markeredgewidth=0.6,
                zorder=7,
            )

            annotate_nonoverlapping(
                ax,
                format_extreme_annotation(row),
                xy=(x, y),
                occupied=occupied_annotations,
                rotation=45,
            )

    # ------------------------------------------------------------
    # Reference lines
    # ------------------------------------------------------------

    if quantity == "mean":
        ax.axvline(
            0.0,
            linestyle="--",
            linewidth=1.0,
            color="black",
            alpha=0.7,
        )
    
    elif quantity == "std":
        ax.axvline(
            1.0,
            linestyle="--",
            linewidth=1.0,
            color="black",
            alpha=0.7,
        )

    # ------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------

    ax.set_xlabel(
        f"Spatial {quantity} "
        "of normalized field"
    )
    ax.set_ylabel(
        "Probability density"
    )

    ax.set_title(
        f"Channel {channel}: "
        f"normalized field {quantity}"
        f"{title_suffix}"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)

# ============================================================================
# Mean vs std
# ============================================================================

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

    plot_df = channel_df[
        channel_df["split"].isin(
            PLOT_SPLITS
        )
    ]

    if plot_df.empty:
        return

    fig, ax = plt.subplots(
        figsize=(7, 6),
    )

    performance_norm = (
        make_performance_norm(
            channel_df
        )
        if COLOR_BY_PERFORMANCE
        else None
    )

    performance_cmap = plt.get_cmap(
        PERFORMANCE_CMAP
    )

    plotted_performance = False

    # ------------------------------------------------------------
    # Selected splits
    # ------------------------------------------------------------

    for split in PLOT_SPLITS:
        df = plot_df[
            plot_df["split"] == split
        ]

        if df.empty:
            continue

        has_performance = (
            COLOR_BY_PERFORMANCE
            and performance_norm is not None
            and "metric_improvement" in df.columns
        )

        if has_performance:
            valid_perf = (
                df["metric_improvement"]
                .notna()
            )
        else:
            valid_perf = pd.Series(
                False,
                index=df.index,
                dtype=bool,
            )

        # --------------------------------------------------------
        # Points without performance data
        # Same neutral style for train/val/test.
        # --------------------------------------------------------

        ordinary = df[
            ~valid_perf
        ]

        if not ordinary.empty:
            ax.scatter(
                ordinary["mean"],
                ordinary["std"],
                s=SCATTER_SIZE,
                alpha=SCATTER_ALPHA,
                color=SCATTER_COLOR,
                label=split,
                zorder=2,
            )

        # --------------------------------------------------------
        # Points with performance data
        # Color encodes performance, not split membership.
        # --------------------------------------------------------

        performance_df = df[
            valid_perf
        ]

        if not performance_df.empty:
            perf_values = performance_df[
                "metric_improvement"
            ].to_numpy(dtype=float)

            point_colors = performance_color(
                perf_values,
                cmap=performance_cmap,
                norm=performance_norm,
            )

            ax.scatter(
                performance_df["mean"],
                performance_df["std"],
                c=point_colors,
                s=PERFORMANCE_SCATTER_SIZE,
                alpha=PERFORMANCE_SCATTER_ALPHA,
                label=split,
                zorder=4,
            )
            plotted_performance = True

    if plotted_performance:
        add_performance_colorbar(
            fig,
            ax,
            cmap=performance_cmap,
            norm=performance_norm,
        )

    # ------------------------------------------------------------
    # Best/worst test-timestep annotations
    # ------------------------------------------------------------

    if "test" in PLOT_SPLITS:
        extrema = get_performance_extrema(
            channel_df,
            n_best=ANNOTATE_BEST_TIMESTEPS,
            n_worst=ANNOTATE_WORST_TIMESTEPS,
        )

        occupied_annotations = []

        for _, row in extrema.iterrows():
            time = pd.Timestamp(row["time"])

            timestep_rows = channel_df[
                (channel_df["split"] == "test")
                & (channel_df["time"] == time)
            ]

            if timestep_rows.empty:
                continue

            x = float(
                timestep_rows["mean"].mean()
            )
            y = float(
                timestep_rows["std"].mean()
            )

            if not (
                np.isfinite(x)
                and np.isfinite(y)
            ):
                continue

            # Thin black ring around annotated points.
            ax.scatter(
                [x],
                [y],
                s=30,
                facecolors="none",
                edgecolors="black",
                linewidths=0.8,
                zorder=7,
            )

            annotate_nonoverlapping(
                ax,
                format_extreme_annotation(row),
                xy=(x, y),
                occupied=occupied_annotations,
            )

    # ------------------------------------------------------------
    # References
    # ------------------------------------------------------------

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        color="black",
        alpha=0.7,
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
        color="black",
        alpha=0.7,
    )

    ax.set_xlabel(
        "Spatial mean"
    )

    ax.set_ylabel(
        "Spatial standard deviation"
    )

    ax.set_title(
        f"Channel {channel}: "
        "normalized input fields"
        f"{title_suffix}"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)

# ============================================================================
# Monthly plots
# ============================================================================

def plot_monthly_stats(
    stats: pd.DataFrame,
    output_dir: Path,
) -> None:
    monthly_output = (
        output_dir
        / "monthly"
    )
    monthly_output.mkdir(
        parents=True,
        exist_ok=True,
    )
    channels = sorted(
        stats[
            "channel"
        ].unique()
    )
    months = sorted(
        stats[
            "month"
        ]
        .dropna()
        .unique()
    )
    for month_value in months:
        month = int(
            month_value
        )
        month_stats = stats[
            stats["month"] == month
        ]
        if month_stats.empty:
            continue
        month_name = MONTH_NAMES[
            month
        ]
        month_dir = (
            monthly_output
            / (
                f"{month:02d}_"
                f"{month_name.lower()}"
            )
        )
        month_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        title_suffix = (
            f" — {month_name}"
        )
        for channel in channels:
            channel_stats = (
                month_stats[
                    month_stats[
                        "channel"
                    ] == channel
                ]
            )
            if channel_stats.empty:
                continue
            plot_distribution(
                month_stats,
                quantity="mean",
                channel=channel,
                plot_type=(
                    DISTRIBUTION_PLOT
                ),
                title_suffix=(
                    title_suffix
                ),
                output_path=(
                    month_dir
                    / (
                        f"channel_{channel}_"
                        "mean_distribution.png"
                    )
                ),
            )
            plot_distribution(
                month_stats,
                quantity="std",
                channel=channel,
                plot_type=(
                    DISTRIBUTION_PLOT
                ),
                title_suffix=(
                    title_suffix
                ),
                output_path=(
                    month_dir
                    / (
                        f"channel_{channel}_"
                        "std_distribution.png"
                    )
                ),
            )
            plot_mean_vs_std(
                month_stats,
                channel=channel,
                title_suffix=(
                    title_suffix
                ),
                output_path=(
                    month_dir
                    / (
                        f"channel_{channel}_"
                        "mean_vs_std.png"
                    )
                ),
            )

# ============================================================================
# All plots
# ============================================================================

def plot_stats(
    stats: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    channels = sorted(
        stats[
            "channel"
        ].unique()
    )
    # ------------------------------------------------------------
    # Full-period plots
    # ------------------------------------------------------------
    for channel in channels:
        plot_distribution(
            stats,
            quantity="mean",
            channel=channel,
            plot_type=(
                DISTRIBUTION_PLOT
            ),
            output_path=(
                output_dir
                / (
                    f"channel_{channel}_"
                    "mean_distribution.png"
                )
            ),
        )
        plot_distribution(
            stats,
            quantity="std",
            channel=channel,
            plot_type=(
                DISTRIBUTION_PLOT
            ),
            output_path=(
                output_dir
                / (
                    f"channel_{channel}_"
                    "std_distribution.png"
                )
            ),
        )
        plot_mean_vs_std(
            stats,
            channel=channel,
            output_path=(
                output_dir
                / (
                    f"channel_{channel}_"
                    "mean_vs_std.png"
                )
            ),
        )
    # ------------------------------------------------------------
    # Monthly
    # ------------------------------------------------------------
    if PLOT_MONTHLY:
        plot_monthly_stats(
            stats,
            output_dir,
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
    print(
        summary.to_string()
    )
    print()
    if (
        "metric_improvement"
        in stats.columns
    ):
        test_perf = (
            stats.loc[
                stats["split"] == "test",
                [
                    "time",
                    "metric_improvement",
                ],
            ]
            .dropna()
            .drop_duplicates(
                subset="time"
            )
        )
        if not test_perf.empty:
            print(
                f"{PERFORMANCE_METRIC.upper()} "
                f"improvement "
                f"({PERFORMANCE_IMPROVEMENT_UNIT}):"
            )
            print(
                test_perf[
                    "metric_improvement"
                ]
                .describe()
                .to_string()
            )
            print()

# ============================================================================
# Build normalized diagnostics for one leadtime
# ============================================================================

def process_leadtime(
    s,
    leadtime: int,
) -> pd.DataFrame:
    print(
        "=" * 80
    )
    print(
        f"{s.output_name}: "
        f"leadtime={leadtime} "
        f"{s.leadtime_unit}"
    )
    datasets = (
        make_train_test_datasets_for_leadtime(
            forecast_ds_path=(
                s.input_dir
                / (
                    f"{s.model_fc}_"
                    f"{s.var_fc}.zarr"
                )
            ),
            analysis_ds_path=(
                s.input_dir
                / (
                    f"{s.model_an}_"
                    f"{s.var_an}.zarr"
                )
            ),
            leadtime=leadtime,
            leadtime_unit=(
                s.leadtime_unit
            ),
            train_start=s.train_start,
            train_end=s.train_end,
            val_start=s.val_start,
            val_end=s.val_end,
            test_start=s.test_start,
            test_end=s.test_end,
            target_mode=s.target_mode,
            clim_period=s.clim_period,
            forecast_vars=[
                s.var_fc
            ],
            analysis_vars=[
                s.var_an
            ],
            region=s.region,
            dataset_kwargs=(
                dataset_kwargs_from_settings(
                    s
                )
            ),
            seasonal_encoding=(
                s.seasonal_encoding
                and (
                    s.channel_representation
                    != "init_period"
                )
            ),
            ensemble_encoding=(
                s.ensemble_encoding
            ),
            input_realization_avg=(
                s.input_realization_avg
            ),
            interpolate_analysis=False,
            materialize=False,
            separate_training_by_init_period=None,
            defer_dataset_creation=False,
        )
    )
    train_dataset = datasets[
        "train"
    ]
    val_dataset = datasets[
        "val"
    ]
    test_dataset = datasets[
        "test"
    ]
    if not isinstance(
        train_dataset,
        XarrayDataset,
    ):
        raise TypeError(
            "Expected XarrayDataset "
            "for training data"
        )
    if (
        val_dataset is not None
        and not isinstance(
            val_dataset,
            XarrayDataset,
        )
    ):
        raise TypeError(
            "Expected XarrayDataset "
            "for validation data"
        )
    if not isinstance(
        test_dataset,
        XarrayDataset,
    ):
        raise TypeError(
            "Expected XarrayDataset "
            "for test data"
        )
    # ------------------------------------------------------------
    # Fit on TRAIN only
    # ------------------------------------------------------------
    normalize_input = (
        make_input_normalizer(
            s,
            train_dataset,
        )
    )
    # ------------------------------------------------------------
    # Apply same transform to every split
    # ------------------------------------------------------------
    train_dataset.transform_x = (
        normalize_input
    )
    if val_dataset is not None:
        val_dataset.transform_x = (
            normalize_input
        )
    test_dataset.transform_x = (
        normalize_input
    )
    # Target transformation not needed.
    train_dataset.transform_y = None
    if val_dataset is not None:
        val_dataset.transform_y = None
    test_dataset.transform_y = None
    # ------------------------------------------------------------
    # Normalized-field statistics
    # ------------------------------------------------------------
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
    stats = pd.concat(
        frames,
        ignore_index=True,
    )
    # ------------------------------------------------------------
    # Automatic metric performance
    # ------------------------------------------------------------
    if COLOR_BY_PERFORMANCE:
        print(
            f"Calculating "
            f"{PERFORMANCE_METRIC} "
            f"improvement..."
        )
        metric_df = (
            get_metric_improvement_timeseries(
                s,
                leadtime=leadtime,
                metric=PERFORMANCE_METRIC,
                improvement_unit=(
                    PERFORMANCE_IMPROVEMENT_UNIT
                ),
            )
        )
        stats = add_metric_performance(
            stats,
            metric_df,
        )
    return stats

# ============================================================================
# Main
# ============================================================================

def main() -> None:
    settings = (
        get_experiment_configs(
            EXPERIMENTS_ROOT,
            **FILTERS,
        )
    )
    if not settings:
        raise RuntimeError(
            "No matching experiments found."
        )
    print(
        f"Found {len(settings)} "
        "matching experiment(s)."
    )
    for s in settings:
        experiment_output = (
            OUTPUT_ROOT
            / s.output_name
            / "normalized"
            / "input_diagnostics"
        )
        experiment_output.mkdir(
            parents=True,
            exist_ok=True,
        )
        leadtimes = (
            [
                int(x)
                for x in s.leadtimes
            ]
            if LEADTIMES is None
            else LEADTIMES
        )
        all_stats = []
        for leadtime in leadtimes:
            stats = process_leadtime(
                s,
                leadtime,
            )
            all_stats.append(
                stats
            )
            leadtime_output = (
                experiment_output
                / f"leadtime_{leadtime}"
            )
            leadtime_output.mkdir(
                parents=True,
                exist_ok=True,
            )
            stats.to_csv(
                leadtime_output
                / "field_stats.csv",
                index=False,
            )
            plot_stats(
                stats,
                leadtime_output,
            )
            print_summary(
                stats
            )
        all_stats = pd.concat(
            all_stats,
            ignore_index=True,
        )
        all_stats.to_csv(
            experiment_output
            / "field_stats_all_leadtimes.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
