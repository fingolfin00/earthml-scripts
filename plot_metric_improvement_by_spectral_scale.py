from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize as MPLNormalize

import numpy as np
import pandas as pd

from earthml import get_experiment_configs
from earthml.plots import safe_label

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

# Must match the split used by the radial-spectrum script.
PLOT_SPLIT: Literal["train", "val", "test"] = "test"

# Must match the optional spatial subset used when generating the spectra.
LAT_RANGE: tuple[float, float] | None = None
LON_RANGE: tuple[float, float] | None = None

# Example:
# LAT_RANGE = (30, 45)
# LON_RANGE = (-108, -90)

# ----------------------------------------------------------------------------
# Spectral field / channel
# ----------------------------------------------------------------------------

SPECTRAL_FIELD: Literal[
    "input",
    "target",
    "corrected",
    "fc_error",
    "mlfc_error",
    "ideal_correction",
    "ml_correction",
] = "mlfc_error"

CHANNEL = 0

SPECTRAL_FIELD_LABELS = {
    "input": "FC input",
    "target": "AN target",
    "corrected": "ML corrected FC",
    "fc_error": "FC - AN",
    "mlfc_error": "MLFC - AN",
    "ideal_correction": "ideal correction (AN - FC)",
    "ml_correction": "ML correction (MLFC - FC)",
}

# ----------------------------------------------------------------------------
# Characteristic spectral scale
# ----------------------------------------------------------------------------

ScaleMetrics = Literal[
    "peak_wavenumber",
    "power_weighted_wavenumber",
]

SCALE_METRICS: tuple[ScaleMetrics, ...] = (
    # "peak_wavenumber",
    "power_weighted_wavenumber",
)

SCALE_METRIC_LABELS = {
    "peak_wavenumber": "Peak normalized radial wavenumber",
    "power_weighted_wavenumber": "Power-weighted normalized radial wavenumber",
}

# ----------------------------------------------------------------------------
# Plot appearance
# ----------------------------------------------------------------------------

PLOT_FIGSIZE = (14, 6)
PLOT_CMAP = "jet"
DPI = 250

WAVENUMBER_COLOR_RANGE: tuple[float, float] | None = None # infer from the data
# WAVENUMBER_COLOR_RANGE: tuple[float, float] | None = (0.0, 1.0) # fixed scale

# Draw points on top of the filled metric-improvement curve, colored with the
# exact characteristic wavenumber of each initialization
PLOT_COLORED_POINTS = True
POINT_SIZE = 18

REGENERATE_PLOTS = True

# ============================================================================
# Input CSVs
# ============================================================================

def load_spectral_and_metric_data(
    s,
    leadtime: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Load per-initialization radial spectra and metric improvement."""

    source_dir = (
        OUTPUT_ROOT
        / s.output_name
        / "normalized"
        / "radial_spectra"
        / PLOT_SPLIT
        / f"leadtime_{safe_label(leadtime)}"
        / f"lat_{safe_label(LAT_RANGE)}_lon_{safe_label(LON_RANGE)}"
    )

    spectra_path = source_dir / "radial_power_spectra_by_time.csv"
    metric_path = source_dir / "metric_improvement.csv"

    if not spectra_path.exists():
        raise FileNotFoundError(
            "Radial-spectrum CSV not found. Run the radial-spectrum script first: "
            f"{spectra_path}"
        )

    if not metric_path.exists():
        raise FileNotFoundError(
            "Metric-improvement CSV not found. Run the radial-spectrum script with "
            f"COMPUTE_PERFORMANCE=True first: {metric_path}"
        )

    spectra = pd.read_csv(
        spectra_path,
        parse_dates=["time"],
    )

    metric = pd.read_csv(
        metric_path,
        parse_dates=["time"],
    )

    required_spectral_columns = {
        "time",
        "field",
        "channel",
        "wavenumber",
        "power",
    }
    missing = required_spectral_columns - set(spectra.columns)
    if missing:
        raise ValueError(
            f"{spectra_path} is missing required columns: {sorted(missing)}"
        )

    required_metric_columns = {
        "time",
        "metric_improvement",
    }
    missing = required_metric_columns - set(metric.columns)
    if missing:
        raise ValueError(
            f"{metric_path} is missing required columns: {sorted(missing)}"
        )

    return spectra, metric, source_dir

# ============================================================================
# Characteristic spatial scale
# ============================================================================

def summarize_characteristic_wavenumbers(
    spectra: pd.DataFrame,
    *,
    field_name: str,
    channel: int,
) -> pd.DataFrame:
    """
    Calculate characteristic wavenumbers for each initialization time.

    peak_wavenumber:
        Wavenumber of the radial bin with maximum power.

    power_weighted_wavenumber:
        Spectral centroid sum(k P(k)) / sum(P(k)).
    """

    selected = spectra[
        (spectra["field"] == field_name)
        & (spectra["channel"] == channel)
    ].copy()

    if selected.empty:
        available_fields = sorted(
            spectra["field"].dropna().astype(str).unique()
        )
        available_channels = sorted(
            int(x)
            for x in spectra.loc[
                spectra["field"] == field_name,
                "channel",
            ].dropna().unique()
        )

        raise ValueError(
            f"No spectra found for field={field_name!r}, channel={channel}. "
            f"Available fields={available_fields}; "
            f"channels for this field={available_channels}."
        )

    rows: list[dict] = []

    for time, group in selected.groupby("time", sort=True):
        k = pd.to_numeric(
            group["wavenumber"],
            errors="coerce",
        ).to_numpy(dtype=float)

        power = pd.to_numeric(
            group["power"],
            errors="coerce",
        ).to_numpy(dtype=float)

        valid = np.isfinite(k) & np.isfinite(power)
        k = k[valid]
        power = power[valid]

        if k.size == 0:
            rows.append({
                "time": pd.Timestamp(time),
                "peak_wavenumber": np.nan,
                "peak_power": np.nan,
                "power_weighted_wavenumber": np.nan,
                "total_spectral_power": np.nan,
            })
            continue

        peak_index = int(np.argmax(power))
        peak_wavenumber = float(k[peak_index])
        peak_power = float(power[peak_index])

        total_power = float(np.sum(power))
        if np.isfinite(total_power) and total_power > 0:
            centroid = float(
                np.sum(k * power) / total_power
            )
        else:
            centroid = np.nan

        rows.append({
            "time": pd.Timestamp(time),
            "peak_wavenumber": peak_wavenumber,
            "peak_power": peak_power,
            "power_weighted_wavenumber": centroid,
            "total_spectral_power": total_power,
        })

    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def merge_metric_and_spectral_scale(
    metric: pd.DataFrame,
    characteristic: pd.DataFrame,
) -> pd.DataFrame:
    """Align performance and spectral-scale diagnostics by initialization time."""

    metric = (
        metric[["time", "metric_improvement"]]
        .drop_duplicates(subset="time")
        .sort_values("time")
    )

    characteristic = (
        characteristic
        .drop_duplicates(subset="time")
        .sort_values("time")
    )

    merged = metric.merge(
        characteristic,
        on="time",
        how="inner",
        validate="one_to_one",
    )

    merged = merged.replace([np.inf, -np.inf], np.nan)

    if merged.empty:
        raise ValueError(
            "Metric-improvement and spectral-scale CSVs have no overlapping times."
        )

    return merged.sort_values("time").reset_index(drop=True)

# ============================================================================
# Plotting
# ============================================================================

def color_normalization(
    values: np.ndarray,
) -> MPLNormalize:
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        raise ValueError("No finite characteristic wavenumbers available for plotting.")

    if WAVENUMBER_COLOR_RANGE is None:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
    else:
        vmin, vmax = (
            float(WAVENUMBER_COLOR_RANGE[0]),
            float(WAVENUMBER_COLOR_RANGE[1]),
        )

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError(
            f"Invalid wavenumber color range: {(vmin, vmax)}"
        )

    if vmax <= vmin:
        center = 0.5 * (vmin + vmax)
        vmin = center - 0.5
        vmax = center + 0.5

    return MPLNormalize(vmin=vmin, vmax=vmax)


def plot_metric_improvement_by_scale(
    df: pd.DataFrame,
    *,
    s,
    leadtime: int,
    scale_metric: ScaleMetrics,
    output_path: Path,
) -> None:
    """
    Plot metric improvement with the area to zero colored by spectral scale.

    Each interval between neighboring initialization times is filled with the
    color corresponding to the mean characteristic wavenumber of its endpoints.
    Individual points are optionally overlaid using their exact timestep value.
    """

    plot_df = (
        df[["time", "metric_improvement", scale_metric]]
        .dropna()
        .sort_values("time")
        .reset_index(drop=True)
    )

    if len(plot_df) < 2:
        raise ValueError(
            f"Need at least two finite timesteps to plot {scale_metric}, "
            f"got {len(plot_df)}."
        )

    times = pd.to_datetime(plot_df["time"])
    improvement = plot_df["metric_improvement"].to_numpy(dtype=float)
    characteristic_k = plot_df[scale_metric].to_numpy(dtype=float)

    norm = color_normalization(characteristic_k)
    cmap = plt.get_cmap(PLOT_CMAP)

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

    # Fill one trapezoid per neighboring pair. Using the mean endpoint
    # wavenumber gives each interval one representative spectral-scale color.
    for index in range(len(plot_df) - 1):
        x = times.iloc[index:index + 2]
        y = improvement[index:index + 2]

        interval_k = float(
            np.nanmean(characteristic_k[index:index + 2])
        )

        ax.fill_between(
            x,
            0.0,
            y,
            color=cmap(norm(interval_k)),
            linewidth=0,
            alpha=0.95,
            zorder=1,
        )

    # Metric-improvement envelope.
    ax.plot(
        times,
        improvement,
        color="black",
        linewidth=1.1,
        zorder=3,
    )

    if PLOT_COLORED_POINTS:
        ax.scatter(
            times,
            improvement,
            c=characteristic_k,
            cmap=cmap,
            norm=norm,
            s=POINT_SIZE,
            edgecolors="black",
            linewidths=0.25,
            zorder=4,
        )

    ax.axhline(
        0.0,
        color="black",
        linewidth=0.8,
        alpha=0.8,
        zorder=2,
    )

    lead_unit = getattr(
        s.leadtime_unit,
        "value",
        s.leadtime_unit,
    )

    field_label = SPECTRAL_FIELD_LABELS.get(
        SPECTRAL_FIELD,
        SPECTRAL_FIELD,
    )

    ax.set_xlabel("Initialization time")
    ax.set_ylabel("Metric improvement")
    ax.set_title(
        f"Metric improvement colored by spectral scale of {field_label}"
        f" · channel {CHANNEL} · lead {leadtime} {lead_unit}"
    )

    sm = ScalarMappable(
        norm=norm,
        cmap=cmap,
    )
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=ax,
        pad=0.015,
    )
    cbar.set_label(SCALE_METRIC_LABELS[scale_metric])

    fig.autofmt_xdate()
    fig.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

# ============================================================================
# One experiment / leadtime
# ============================================================================

def process_leadtime(
    s,
    leadtime: int,
) -> None:
    print("=" * 80)
    print(
        f"{s.output_name}: leadtime={leadtime} "
        f"{getattr(s.leadtime_unit, 'value', s.leadtime_unit)}"
    )

    spectra, metric, source_dir = load_spectral_and_metric_data(
        s,
        leadtime,
    )

    characteristic = summarize_characteristic_wavenumbers(
        spectra,
        field_name=SPECTRAL_FIELD,
        channel=CHANNEL,
    )

    merged = merge_metric_and_spectral_scale(
        metric,
        characteristic,
    )

    output_dir = (
        source_dir
        / "metric_improvement_by_spectral_scale"
        / SPECTRAL_FIELD
        / f"channel_{CHANNEL}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_path = output_dir / "metric_improvement_spectral_scale.csv"
    merged.to_csv(
        merged_path,
        index=False,
    )
    print(f"Saving merged diagnostics {merged_path}")

    for scale_metric in SCALE_METRICS:
        if scale_metric not in SCALE_METRIC_LABELS:
            raise ValueError(
                f"Unsupported scale metric: {scale_metric!r}"
            )

        output_path = output_dir / f"{scale_metric}.png"

        if output_path.exists() and not REGENERATE_PLOTS:
            continue

        print(f"Saving {scale_metric} plot {output_path}")

        plot_metric_improvement_by_scale(
            merged,
            s=s,
            leadtime=leadtime,
            scale_metric=scale_metric,
            output_path=output_path,
        )

# ============================================================================
# Main
# ============================================================================

def main() -> None:
    settings = get_experiment_configs(
        EXPERIMENTS_ROOT,
        **FILTERS,
    )

    if not settings:
        raise RuntimeError("No matching experiments found.")

    print(f"Found {len(settings)} matching experiment(s).")

    for s in settings:
        leadtimes = (
            [int(x) for x in s.leadtimes]
            if LEADTIMES is None
            else LEADTIMES
        )

        missing_leadtimes = (
            set(leadtimes)
            - set(int(x) for x in s.leadtimes)
        )

        if missing_leadtimes:
            raise ValueError(
                f"Requested leadtimes {sorted(missing_leadtimes)} are not "
                f"available for {s.output_name}."
            )

        for leadtime in leadtimes:
            process_leadtime(
                s,
                int(leadtime),
            )


if __name__ == "__main__":
    main()
