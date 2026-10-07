from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import pandas as pd
import xarray as xr

from scipy.stats import gaussian_kde, spearmanr

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tqdm.auto import tqdm

from earthml import (
    Settings,
    LeadtimeUnit,
    XarrayDataset,
    get_experiment_configs,
)

from earthml.metrics import (
    ImprovementUnit,
    PeriodReference,    
)

from train import make_train_test_datasets_for_leadtime
from normalized_scatter import (
    make_normalizer,
    get_leadtime_xarray_datasets,
    get_normalized_datasets,
    subset_diagnostic_dataset_region,
    build_fields,
    dataset_kwargs_from_settings,
    diagnostic_region_label,
    get_metric_improvement_timeseries,
    get_performance_extrema,
    format_extrema_label,
    format_extrema_annotation,
)

# ============================================================================
# Configuration
# ============================================================================

# EXP_NAME = "weather_atmo"
EXP_NAME = "weather_atmo_one_year_test"

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
# Diagnostic spatial region
# ----------------------------------------------------------------------------

# Optional box used consistently for normalized-input statistics, spectral
# diagnostics, FC/AN/MLFC diagnostic fields, and performance (e.g. RMSE
# improvement). None keeps the corresponding full experiment dimension.
# Best/worst rankings therefore refer to performance inside this box.
DISTRIBUTION_LAT_RANGE: tuple[float, float] | None = None
DISTRIBUTION_LON_RANGE: tuple[float, float] | None = None

# DISTRIBUTION_LAT_RANGE = (30, 45)
# DISTRIBUTION_LON_RANGE = (-108, -90)

# ----------------------------------------------------------------------------
# Field statistics
# ----------------------------------------------------------------------------

DISTRIBUTION_FIELD: Literal[
    "input",
    "target",
    "corrected",
    "fc_error",
    "mlfc_error",
    "ideal_correction",
    "ml_correction",
] = "input"

NORMALIZED_DISTRIBUTION = True

# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

PLOT_MONTHLY = False

DISTRIBUTION_PLOT: Literal[
    "hist",
    "kde",
    "both",
] = "kde"

BINS = 250
KDE_POINTS = 500

SPLIT_RANGE: Literal["train", "val", "test"] = (
    # "train"
    # "val"
    "test"
)

SPLIT_COLORS = {
    "train": "C0",
    "val": "C1",
    "test": "C2",
}

LEGEND_LOCATION = "upper right"

ANNOTATE_BEST_TIMESTEPS = 10
ANNOTATE_WORST_TIMESTEPS = 10

# SELECTED_SAMPLES_TO_ANNOTATE: set[str] | None = None
SELECTED_SAMPLES_TO_ANNOTATE: set[str] | None = {
    "B5",
    "W4",
}

ANNOTATION_FONTSIZE = 5
ANNOTATION_DATE_FORMAT = "%Y-%m-%d:%H:%M"

# ----------------------------------------------------------------------------
# Performance and coloring
# ----------------------------------------------------------------------------

COLOR_BY_PERFORMANCE = True

PERFORMANCE_METRIC = "rmse"
PERFORMANCE_METRICS_NORMALIZED = False

PERFORMANCE_IMPROVEMENT_UNIT: ImprovementUnit = "Δ"
PERFORMANCE_REFERENCE_TIME_PERIOD: PeriodReference = "init"

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
# Build field-distribution statistics for one leadtime
# ============================================================================

def get_leadtime_stats(
    s: Settings,
    leadtime: int,
) -> pd.DataFrame:
    split = SPLIT_RANGE

    print(
        "=" * 80
    )

    print(
        f"{s.output_name}\n"
        f"leadtime={leadtime} "
        f"{s.leadtime_unit}, "
        f"split={split}"
    )
    print(
        "Selected spatial region: "
        f"lat={DISTRIBUTION_LAT_RANGE}, lon={DISTRIBUTION_LON_RANGE}"
    )

    # ------------------------------------------------------------
    # Generate datasets
    # ------------------------------------------------------------

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
        leadtime_unit=LeadtimeUnit(s.leadtime_unit),
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
        spatial_encoding=s.spatial_encoding,
        interpolate_analysis=INTERPOLATE_ANALYSIS,
        materialize=False,
    )

    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    test_dataset = datasets["test"]

    if not isinstance(train_dataset, XarrayDataset):
        raise NotImplementedError(
            "Train dataset must be an XarrayDataset"
        )

    if (
        val_dataset is not None
        and not isinstance(val_dataset, XarrayDataset)
    ):
        raise NotImplementedError(
            "Validation dataset must be an XarrayDataset or None"
        )

    if not isinstance(test_dataset, XarrayDataset):
        raise NotImplementedError(
            "Test dataset must be an XarrayDataset"
        )

    normalize_input = None
    normalize_target = None

    space = (
        "normalized"
        if NORMALIZED_DISTRIBUTION
        else "physical"
    )

    if NORMALIZED_DISTRIBUTION:
        normalize_input = make_normalizer(
            s,
            train_dataset,
            dim="x",
        )

        normalize_target = make_normalizer(
            s,
            train_dataset,
            dim="y",
        )

    frames = []
    if split=="val" and val_dataset is None:
        raise ValueError(f"Split {split} not avalable for split strategy {s.split_strategy}")

    fc, an, mlfc = get_leadtime_xarray_datasets(
        s=s,
        time_range=split,
        leadtime=leadtime,
    )

    if mlfc is None:
        raise ValueError("MLFC dataset is required to calculate metrics improvement")

    if NORMALIZED_DISTRIBUTION:
        assert normalize_input is not None
        assert normalize_target is not None

        distro_fc, distro_an, distro_mlfc = get_normalized_datasets(
            s=s,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            normalize_input=normalize_input,
            normalize_target=normalize_target,
            fc=fc,
            mlfc=mlfc,
            time_range=split,
        )
    else:
        distro_fc = fc
        distro_an = an
        distro_mlfc = mlfc

    # Subset
    distro_fc = subset_diagnostic_dataset_region(distro_fc, DISTRIBUTION_LAT_RANGE, DISTRIBUTION_LON_RANGE)
    distro_an = subset_diagnostic_dataset_region(distro_an, DISTRIBUTION_LAT_RANGE, DISTRIBUTION_LON_RANGE)
    distro_mlfc = subset_diagnostic_dataset_region(distro_mlfc, DISTRIBUTION_LAT_RANGE, DISTRIBUTION_LON_RANGE)

    fields = build_fields(
        baseline=distro_fc[s.var_fc],
        target=distro_an[s.var_an],
        corrected=distro_mlfc[s.var_fc],
    )

    # ------------------------------------------------------------
    # Field and reference (target) statistics
    # ------------------------------------------------------------

    frames.append(
        collect_field_stats(
            field=fields[DISTRIBUTION_FIELD],
            field_name=DISTRIBUTION_FIELD,
            reference_field=fields["target"],
            space=space,
            split=split,
            leadtime=leadtime,
        )
    )

    stats = pd.concat(
        frames,
        ignore_index=True,
    )

    # ------------------------------------------------------------
    # Metric performance
    # ------------------------------------------------------------

    if COLOR_BY_PERFORMANCE:
        fc, an, mlfc = get_leadtime_xarray_datasets(
            s=s,
            time_range=split,
            leadtime=leadtime,
        )

        if mlfc is None:
            raise ValueError(
                "MLFC dataset is required to calculate performance."
            )

        print(
            "Calculating "
            f"{'normalized ' if PERFORMANCE_METRICS_NORMALIZED else ''}"
            f"{PERFORMANCE_METRIC} improvement in split={split}..."
        )

        metric_df = get_metric_improvement_timeseries(
            s=s,
            baseline=fc,
            target=an,
            corrected=mlfc,
            leadtime=leadtime,
            metric=PERFORMANCE_METRIC,
            improvement_unit=PERFORMANCE_IMPROVEMENT_UNIT,
            period_reference=PERFORMANCE_REFERENCE_TIME_PERIOD,
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

    distribution_space = (
        "normalized"
        if NORMALIZED_DISTRIBUTION
        else "physical"
    )

    for s in settings:
        experiment_output = (
            OUTPUT_ROOT
            / s.output_name
            / "distribution"
            / distribution_space
            / DISTRIBUTION_FIELD
            / diagnostic_region_label()
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
            stats = get_leadtime_stats(
                s,
                leadtime,
            )

            all_stats.append(
                stats
            )

            leadtime_output = (
                experiment_output
                / f"leadtime_{leadtime}"
                / DISTRIBUTION_FIELD
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

            print_summary(stats)

        all_stats = pd.concat(
            all_stats,
            ignore_index=True,
        )

        all_stats.to_csv(
            experiment_output
            / "stats_all_leadtimes.csv",
            index=False,
        )

# ============================================================================
# Field statistics
# ============================================================================

class FieldStats(TypedDict):
    min: xr.DataArray
    max: xr.DataArray
    mean: xr.DataArray
    std: xr.DataArray
    split_min: float
    split_max: float
    split_mean: float
    split_std: float

def collect_field_stats(
    field: xr.DataArray,
    field_name: str,
    *,
    reference_field: xr.DataArray | None = None,
    space: Literal["normalized", "physical"],
    split: str,
    leadtime: int,
) -> pd.DataFrame:
    time_dim = field.earthml.guessed_dims.time
    lat_dim = field.earthml.guessed_dims.latitude
    lon_dim = field.earthml.guessed_dims.longitude

    if time_dim is None:
        raise ValueError(
            "Could not determine time dimension of field."
        )

    if lat_dim is None or lon_dim is None:
        raise ValueError(
            "Could not determine spatial dimensions of field."
        )

    field = field.squeeze(drop=True)

    extra_dims = [
        dim
        for dim in field.dims
        if dim not in {
            time_dim,
            lat_dim,
            lon_dim,
        }
    ]

    if extra_dims:
        raise ValueError(
            "Field contains unsupported dimensions after "
            f"alignment: {field.dims}. "
            f"Extra dimensions: {extra_dims}"
        )

    spatial_dims = (
        lat_dim,
        lon_dim,
    )

    def compute_stats(
        da: xr.DataArray,
    ) -> FieldStats:
        da = da.squeeze(drop=True)

        return {
            "min": da.min(
                dim=spatial_dims,
                skipna=True,
            ),
            "max": da.max(
                dim=spatial_dims,
                skipna=True,
            ),
            "mean": da.mean(
                dim=spatial_dims,
                skipna=True,
            ),
            "std": da.std(
                dim=spatial_dims,
                skipna=True,
                ddof=0,
            ),
            "split_min": float(
                da.min(skipna=True).values
            ),
            "split_max": float(
                da.max(skipna=True).values
            ),
            "split_mean": float(
                da.mean(skipna=True).values
            ),
            "split_std": float(
                da.std(
                    skipna=True,
                    ddof=0,
                ).values
            ),
        }

    field_stats = compute_stats(field)

    reference_stats = (
        compute_stats(reference_field)
        if reference_field is not None
        else None
    )

    times = pd.to_datetime(
        field[time_dim].values
    )

    frame = pd.DataFrame({
        "space": space,
        "field": field_name,
        "split": split,
        "leadtime": leadtime,
        "sample": np.arange(len(times)),
        "time": times,
        "channel": 0,
        "month": times.month,

        # Per-sample field stats within split
        "min": np.asarray(
            field_stats["min"].values
        ),
        "max": np.asarray(
            field_stats["max"].values
        ),
        "mean": np.asarray(
            field_stats["mean"].values
        ),
        "std": np.asarray(
            field_stats["std"].values
        ),

        # Whole field stats within split
        "split_min": field_stats["split_min"],
        "split_max": field_stats["split_max"],
        "split_mean": field_stats["split_mean"],
        "split_std": field_stats["split_std"],

        # Per-sample target stats within split
        "target_min": (
            np.asarray(reference_stats["min"].values)
            if reference_stats is not None
            else np.nan
        ),
        "target_max": (
            np.asarray(reference_stats["max"].values)
            if reference_stats is not None
            else np.nan
        ),
        "target_mean": (
            np.asarray(reference_stats["mean"].values)
            if reference_stats is not None
            else np.nan
        ),
        "target_std": (
            np.asarray(reference_stats["std"].values)
            if reference_stats is not None
            else np.nan
        ),

        # Whole target stats within split
        "target_split_min": (
            reference_stats["split_min"]
            if reference_stats is not None
            else np.nan
        ),
        "target_split_max": (
            reference_stats["split_max"]
            if reference_stats is not None
            else np.nan
        ),
        "target_split_mean": (
            reference_stats["split_mean"]
            if reference_stats is not None
            else np.nan
        ),
        "target_split_std": (
            reference_stats["split_std"]
            if reference_stats is not None
            else np.nan
        ),
    })

    return frame


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

    # Only attach performance to test samples
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

    # Check alignment
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

    unit = PERFORMANCE_IMPROVEMENT_UNIT

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
    plot_type: Literal["hist", "kde", "both"] = "both",
    title_suffix: str = "",
    points_to_annotate: pd.DataFrame | None = None,
) -> None:
    if plot_type not in {"hist", "kde", "both"}:
        raise ValueError(
            f"Unsupported plot_type={plot_type!r}"
        )

    channel_df = stats[
        stats["channel"] == channel
    ]

    split = SPLIT_RANGE

    groups = {
        split: (
            channel_df.loc[
                channel_df["split"] == split,
                quantity,
            ]
            .dropna()
            .to_numpy(dtype=float)
        )
    }

    lo, hi = distribution_range(
        list(groups.values())
    )

    edges = np.linspace(
        lo,
        hi,
        BINS + 1,
    )

    n_density_points = max(
        KDE_POINTS,
        4 * len(groups[split]),
    )

    x_grid = np.linspace(
        lo,
        hi,
        n_density_points,
    )

    fig, ax = plt.subplots(
        figsize=(8, 5),
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

    split_density_for_annotations = None

    # ------------------------------------------------------------
    # Plot selected split
    # ------------------------------------------------------------

    values = groups[split]
    n_values = len(values)
    if n_values == 0:
        raise ValueError(f"Empty split {split}")

    color = SPLIT_COLORS[split]

    median = float(
        np.median(values)
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
            label=f"{split} histogram",
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
        and n_values > 1
        and not np.isclose(
            np.std(values),
            0.0,
        )
    ):
        kde = gaussian_kde(values)
        kde_density = kde(x_grid)

        ax.plot(
            x_grid,
            kde_density,
            linewidth=2.0,
            color=color,
            label=f"{split} distribution",
            zorder=5,
        )

    if kde_density is not None:
        split_density_for_annotations = kde_density
    elif plot_type == "hist":
        split_density_for_annotations = (
            histogram_density_on_grid(
                values,
                edges=edges,
                x_grid=x_grid,
            )
        )

    # ------------------------------------------------------------
    # Performance-colored density bands + correlation
    # ------------------------------------------------------------

    split_test_perf = pd.DataFrame()

    if "metric_improvement" in channel_df.columns:
        split_test_perf = (
            channel_df.loc[
                channel_df["split"] == split,
                [quantity, "metric_improvement"],
            ]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .copy()
        )

    if (
        not split_test_perf.empty
        and performance_norm is not None
        and split_density_for_annotations is not None
    ):
        sample_values = split_test_perf[
            quantity
        ].to_numpy(dtype=float)

        sample_performance = split_test_perf[
            "metric_improvement"
        ].to_numpy(dtype=float)

        # --------------------------------------------------------
        # Sort samples by x coordinate
        # --------------------------------------------------------

        order = np.argsort(sample_values)

        sample_values = sample_values[order]
        sample_performance = sample_performance[order]

        # --------------------------------------------------------
        # Deal with samples having exactly the same x value
        #
        # Multiple samples at the same x coordinate cannot occupy
        # separate horizontal bands. Average their performance.
        # --------------------------------------------------------

        unique_values, inverse = np.unique(
            sample_values,
            return_inverse=True,
        )

        unique_performance = np.empty(
            len(unique_values),
            dtype=float,
        )

        for i in range(len(unique_values)):
            unique_performance[i] = np.mean(
                sample_performance[
                    inverse == i
                ]
            )

        # --------------------------------------------------------
        # Construct sample-centered band edges
        # --------------------------------------------------------

        if len(unique_values) == 1:
            # Degenerate case: one unique sample value.
            # Give it a small band around its x position.
            width = (
                hi - lo
            ) / max(KDE_POINTS, 1)

            sample_edges = np.array([
                unique_values[0] - 0.5 * width,
                unique_values[0] + 0.5 * width,
            ])

        else:
            midpoints = (
                unique_values[:-1]
                + unique_values[1:]
            ) / 2.0

            sample_edges = np.empty(
                len(unique_values) + 1,
                dtype=float,
            )

            sample_edges[1:-1] = midpoints

            # Extend first and last bands symmetrically.
            sample_edges[0] = (
                unique_values[0]
                - (
                    midpoints[0]
                    - unique_values[0]
                )
            )

            sample_edges[-1] = (
                unique_values[-1]
                + (
                    unique_values[-1]
                    - midpoints[-1]
                )
            )

            # Do not color outside the plotted distribution range.
            sample_edges[0] = max(
                sample_edges[0],
                lo,
            )
            sample_edges[-1] = min(
                sample_edges[-1],
                hi,
            )

        # --------------------------------------------------------
        # Color density under curve
        # --------------------------------------------------------

        for i, mean_performance in enumerate(
            unique_performance
        ):
            x0 = sample_edges[i]
            x1 = sample_edges[i + 1]

            band_mask = (
                (x_grid >= x0)
                & (x_grid <= x1)
            )

            if not np.any(band_mask):
                continue

            ax.fill_between(
                x_grid[band_mask],
                0.0,
                split_density_for_annotations[
                    band_mask
                ],
                color=performance_color(
                    mean_performance,
                    cmap=performance_cmap,
                    norm=performance_norm,
                ),
                alpha=0.95,
                linewidth=0.0,
                zorder=1,
            )

        add_performance_colorbar(
            fig,
            ax,
            cmap=performance_cmap,
            norm=performance_norm,
        )

    # ------------------------------------------------------------
    # Correlation
    # ------------------------------------------------------------

    correlation_text = ""

    if len(split_test_perf) >= 2:
        x = split_test_perf[
            quantity
        ].to_numpy(dtype=float)

        y = split_test_perf[
            "metric_improvement"
        ].to_numpy(dtype=float)

        rho, p_value = spearmanr(
            x,
            y,
        )

        if np.isfinite(rho):
            correlation_text = (
                rf"Spearman $\rho$ "
                rf"({quantity}-"
                rf"{PERFORMANCE_IMPROVEMENT_UNIT}"
                rf"{PERFORMANCE_METRIC.upper()}) "
                rf"= {rho:.3f}"
            )

            if np.isfinite(p_value):
                correlation_text += (
                    f" (p={p_value:.3g})"
                )

    # ------------------------------------------------------------
    # Best/worst timestep annotations
    # ------------------------------------------------------------

    if (
        points_to_annotate is not None
        and split_density_for_annotations is not None
    ):
        occupied_annotations = []

        for _, row in points_to_annotate.iterrows():
            time = pd.Timestamp(row["time"])

            timestep_rows = channel_df[
                (channel_df["split"] == split)
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
                    split_density_for_annotations,
                )
            )

            if not (
                np.isfinite(x)
                and np.isfinite(y)
            ):
                continue

            if performance_norm is not None:
                perf = float(
                    row["metric_improvement"]
                )
                marker_color = performance_color(
                    perf,
                    cmap=performance_cmap,
                    norm=performance_norm,
                )
            else:
                marker_color = "0.45"

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
                format_extrema_annotation(row),
                xy=(x, y),
                occupied=occupied_annotations,
            )

    # ------------------------------------------------------------
    # Reference lines
    # ------------------------------------------------------------

    split_df = channel_df[
        channel_df["split"] == split
    ]

    if split_df.empty:
        raise ValueError(f"Empty split {split}")

    # Plot split input median
    ax.axvline(
        median,
        color=color,
        linestyle=":",
        linewidth=1.2,
        alpha=1,
        zorder=3,
    )

    target_split_quantity = float(
        split_df[
            f"target_split_{quantity}"
        ].iloc[0]
    )

    ax.axvline(
        target_split_quantity,
        linestyle="--",
        linewidth=1.0,
        color=SPLIT_COLORS[split],
        alpha=0.8,
    )

    reference_ticks = [target_split_quantity]
    reference_labels = [
        f"whole-split\n{target_split_quantity:.3f}"
    ]

    mean_target_split_quantity = np.nan

    if quantity == "std":
        mean_target_split_quantity = float(
            split_df[
                f"target_{quantity}"
            ].mean()
        )

        ax.axvline(
            mean_target_split_quantity,
            linestyle="-",
            linewidth=1.0,
            color=SPLIT_COLORS[split],
            alpha=0.8,
        )

        reference_ticks.append(
            mean_target_split_quantity
        )
        reference_labels.append(
            f"mean per-sample\n{mean_target_split_quantity:.3f}"
        )

    # ------------------------------------------------------------
    # Top reference axis
    # ------------------------------------------------------------

    reference_ticks = [
        median,
        target_split_quantity,
    ]

    reference_labels = [
        f"input\nmedian\n{median:.3f}",
        f"target\nmean\n{target_split_quantity:.3f}",
    ]

    if quantity == "std":
        reference_ticks.append(
            mean_target_split_quantity
        )
        reference_labels.append(
            f"avg\nps-target\n{mean_target_split_quantity:.3f}"
        )

    reference_ax = ax.secondary_xaxis("top")

    reference_ax.set_xticks(
        reference_ticks
    )

    reference_ax.set_xticklabels(
        reference_labels,
        fontsize=5,
    )

    reference_ax.tick_params(
        axis="x",
        length=4,
    )

    # ------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------

    space = (
        "normalized"
        if NORMALIZED_DISTRIBUTION
        else "physical"
    )

    ax.set_xlabel(f"Spatial {quantity} of {space} {DISTRIBUTION_FIELD}")
    ax.set_ylabel("Probability density")

    ax.set_title(
        f"{space} {DISTRIBUTION_FIELD} distribution "
        f"{quantity}"
        f"{title_suffix} · "
        f"n={n_values}\n"
        f"{correlation_text}"
    )

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
    points_to_annotate: pd.DataFrame | None = None,
) -> None:
    split = SPLIT_RANGE

    channel_df = stats[
        stats["channel"] == channel
    ]

    plot_df = channel_df[
        channel_df["split"].isin(
            [split]
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

    df = plot_df[
        plot_df["split"] == split
    ]

    if df.empty:
        raise ValueError(f"Empty split {split}")

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
    # Points with performance data, color encodes performance
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

    if points_to_annotate is not None:
        occupied_annotations = []

        for _, row in points_to_annotate.iterrows():
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

            # Thin black ring around annotated points
            ax.scatter(
                [x],
                [y],
                s=PERFORMANCE_SCATTER_SIZE*2,
                facecolors="none",
                edgecolors="black",
                linewidths=0.8,
                zorder=7,
            )

            annotate_nonoverlapping(
                ax,
                format_extrema_annotation(row),
                xy=(x, y),
                occupied=occupied_annotations,
            )

    # ------------------------------------------------------------
    # Reference values
    # ------------------------------------------------------------

    split_df = plot_df[
        plot_df["split"] == split
    ]

    if split_df.empty:
        raise ValueError(f"Empty split {split}")

    # Mean of input single sample's std within split
    mean_input_std = float(
        split_df["std"].mean()
    )

    # Whole input split mean and std
    input_mean = float(
        split_df["split_mean"].iloc[0]
    )
    input_std = float(
        split_df["split_std"].iloc[0]
    )

    # Mean of target single sample's std within split
    mean_target_std = float(
        split_df["target_std"].mean()
    )

    # Whole target split mean and std
    target_mean = float(
        split_df["target_split_mean"].iloc[0]
    )
    target_std = float(
        split_df["target_split_std"].iloc[0]
    )

    # ------------------------------------------------------------
    # Reference points
    # ------------------------------------------------------------

    references = (
        (
            "avg target",
            target_mean,
            mean_target_std,
            "--",
            SPLIT_COLORS[split],
        ),
        (
            "target",
            target_mean,
            target_std,
            "-",
            SPLIT_COLORS[split],
        ),
        (
            "avg input",
            input_mean,
            mean_input_std,
            "-.",
            SPLIT_COLORS[split],
        ),
        (
            "input",
            input_mean,
            input_std,
            ":",
            SPLIT_COLORS[split],
        ),
    )

    occupied = []

    for (
        label,
        mean_value,
        std_value,
        linestyle,
        color,
    ) in references:

        ax.axvline(
            mean_value,
            linestyle=linestyle,
            linewidth=1.0,
            color=color,
            alpha=0.7,
            zorder=2,
        )

        ax.axhline(
            std_value,
            linestyle=linestyle,
            linewidth=1.0,
            color=color,
            alpha=0.7,
            zorder=2,
        )

        ax.scatter(
            [mean_value],
            [std_value],
            marker="o",
            s=PERFORMANCE_SCATTER_SIZE*2,
            edgecolors="black",
            color=color,
            zorder=8,
        )

        annotate_nonoverlapping(
            ax,
            (
                f"{label}\n"
                f"mean={mean_value:.3f}\n"
                f"std={std_value:.3f}"
            ),
            xy=(
                mean_value,
                std_value,
            ),
            occupied=occupied,
        )

    # ------------------------------------------------------------
    # Labels and title
    # ------------------------------------------------------------

    space = (
        "normalized"
        if NORMALIZED_DISTRIBUTION
        else "physical"
    )

    ax.set_xlabel(f"Spatial mean")
    ax.set_ylabel(f"Spatial standard deviation")

    ax.set_title(
        f"{space} {DISTRIBUTION_FIELD} field: "
        "mean vs standard deviation"
        f"{title_suffix}"
    )

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
    points_to_annotate: pd.DataFrame | None = None,
) -> None:
    monthly_output = output_dir / "monthly"

    monthly_output.mkdir(parents=True, exist_ok=True)

    channels = sorted(stats["channel"].unique())

    months = sorted(stats["month"].dropna().unique())

    for month_value in months:
        month = int(month_value)
        month_stats = stats[
            stats["month"] == month
        ]

        if month_stats.empty:
            continue

        month_name = MONTH_NAMES[month]

        month_dir = (
            monthly_output
            / (
                f"{month:02d}_"
                f"{month_name.lower()}"
            )
        )

        month_dir.mkdir(parents=True, exist_ok=True)

        title_suffix = f" — {month_name}"

        for channel in channels:
            channel_stats = (
                month_stats[
                    month_stats["channel"] == channel
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
                        f"channel_{channel}"
                        f"_{PERFORMANCE_METRIC}_color_encoding"
                        "_mean_distribution.png"
                    )
                ),
                points_to_annotate=points_to_annotate,
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
                        f"channel_{channel}"
                        f"_{PERFORMANCE_METRIC}_color_encoding"
                        "_std_distribution.png"
                    )
                ),
                points_to_annotate=points_to_annotate,
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
                        f"channel_{channel}"
                        f"_{PERFORMANCE_METRIC}_color_encoding"
                        "_mean_vs_std.png"
                    )
                ),
                points_to_annotate=points_to_annotate,
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

    # Calculate and filter extrema
    extrema = get_performance_extrema(
        stats,
        n_best=ANNOTATE_BEST_TIMESTEPS,
        n_worst=ANNOTATE_WORST_TIMESTEPS,
    )

    if SELECTED_SAMPLES_TO_ANNOTATE is not None:
        extrema = extrema[
            extrema.apply(
                format_extrema_label,
                axis=1,
            ).isin(SELECTED_SAMPLES_TO_ANNOTATE)
        ]

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
                    f"channel_{channel}"
                    f"_{PERFORMANCE_METRIC}_color_encoding"
                    "_mean_distribution.png"
                )
            ),
            points_to_annotate=extrema,
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
                    f"channel_{channel}"
                    f"_{PERFORMANCE_METRIC}_color_encoding"
                    "_std_distribution.png"
                )
            ),
            points_to_annotate=extrema,
        )

        plot_mean_vs_std(
            stats,
            channel=channel,
            output_path=(
                output_dir
                / (
                    f"channel_{channel}"
                    f"_{PERFORMANCE_METRIC}_color_encoding"
                    "_mean_vs_std.png"
                )
            ),
            points_to_annotate=extrema,
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


if __name__ == "__main__":
    main()
