from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console()

from dask.diagnostics.progress import ProgressBar

import logging
import warnings

logging.getLogger("cfgrib").setLevel(logging.ERROR)
logging.getLogger("eccodes").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="ecCodes.*")


VAR_FILTERS = {
    "thetao": {"shortName": "thetao"},
    "uovo": {"shortName": "uovo"},
    "so": {"shortName": "so"},
}

REGION = "ATLN"
# REGION = "PACEQ"
PRODUCT_FREQ = "6h"

forecast_dir = Path(f"/work/cmcc/jd19424/ML/MLBC/mercator/historical_forecasts/{REGION}/")

out_dir = Path(f"/work/cmcc/jd19424/ML/MLBC/mercator/data/{REGION}/")
out_dir.mkdir(exist_ok=True, parents=True)

start = pd.Timestamp("2022-11-24 00:00")
# start = pd.Timestamp("2025-10-01 00:00")
end = pd.Timestamp("2026-03-31 00:00")

init_freq = "1D"
lead_days = [0, 1, 2, 3, 4 ,5 ,6 ,7, 9]
forecast_valid_time_tolerance_days = 1

variables = [
    "thetao",
    "uovo",
    "so",
]

save_freq = "MS"  # yearly. Use "MS" for monthly, "QS" quarterly.


def directory_size(path: Path) -> int:
    return sum(
        f.stat().st_size
        for f in path.rglob("*")
        if f.is_file()
    )

def human_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


def forecast_file(
    init_time: pd.Timestamp,
    var: str,
    lead: int,
) -> Path:
    valid_time = init_time + pd.Timedelta(days=lead)

    init_yyyymmdd = init_time.strftime("%Y%m%d")
    valid_yyyymmdd = valid_time.strftime("%Y%m%d")

    name = (
        f"ext-glo12_rg_{PRODUCT_FREQ}-i_"
        f"{valid_yyyymmdd}-{valid_yyyymmdd}"
        f"_3D-{var}_fcst_R{init_yyyymmdd}.nc"
    )

    return (
        forecast_dir
        / init_time.strftime("%Y")
        / init_time.strftime("%m")
        / name
    )

def open_nc_var(
    path: Path,
    var: str,
) -> xr.Dataset | None:
    ds = xr.open_dataset(
        path,
        engine="netcdf4", # netcdf4 h5netcdf
        chunks={},
        # chunks={"time": 1, "latitude": 300, "longitude": 600},
    )

    if len(ds.data_vars) == 0:
        return None

    if var not in ds:
        if len(ds.data_vars) == 1:
            old = next(iter(ds.data_vars))
            ds = ds.rename({old: var})
        else:
            raise RuntimeError(f"Expected one variable in {path}, got {list(ds.data_vars)}")

    return ds[[var]]


def clean_var(ds: xr.Dataset, var: str) -> xr.Dataset:
    ds = ds[[var]]

    if "longitude" in ds.coords:
        lon = ds.longitude
        new_lon = ((lon + 180) % 360) - 180

        ds = ds.assign_coords(longitude=new_lon)
        ds = ds.sortby("longitude")

        nlon = ds.sizes["longitude"]
        canonical_lon = np.linspace(-180, 180, nlon, endpoint=False)

        ds = ds.assign_coords(longitude=canonical_lon)

    ds.attrs = {}
    ds[var].attrs = {}

    keep_coords = {"time", "depth", "latitude", "longitude"}
    drop_coords = [c for c in ds.coords if c not in keep_coords]

    return ds.drop_vars(drop_coords, errors="ignore")


def load_forecast_var(
    init_time: pd.Timestamp,
    lead: int,
    var: str,
) -> xr.Dataset | None:
    forecast_path = forecast_file(init_time, var, lead)

    if not forecast_path.exists():
        console.print("missing forecast:", forecast_path)
        return None

    try:
        ds = open_nc_var(forecast_path, var)
    except Exception as e:
        console.print("bad forecast:", forecast_path, e)
        return None

    if ds is None:
        return None

    target_date = init_time + pd.Timedelta(days=lead)

    ds = clean_var(ds, var)

    # keep the 4 native 6-hourly times from the file
    ds = ds.expand_dims(
        init_time=[init_time],
        leadtime=[lead],
    )

    return ds


def load_analysis_var(
    valid_time: pd.Timestamp,
    var: str,
) -> xr.Dataset | None:
    analysis_path = forecast_file(valid_time, var, 0)

    if not analysis_path.exists():
        console.print("missing analysis:", valid_time)
        return None

    try:
        ds = open_nc_var(analysis_path, var)
    except Exception as e:
        console.print("bad analysis:", analysis_path, e)
        return None

    if ds is None:
        console.print("analysis missing var:", var, valid_time)
        return None

    ds = clean_var(ds, var)

    return ds


def assert_same_lat_lon(
    reference: xr.Dataset,
    candidate: xr.Dataset,
    label: str
):
    for coord in ["latitude", "longitude"]:
        if coord not in reference.coords or coord not in candidate.coords:
            raise RuntimeError(f"{label}: missing coordinate {coord}")

        if reference.sizes[coord] != candidate.sizes[coord]:
            raise RuntimeError(
                f"{label}: {coord} size mismatch: "
                f"{reference.sizes[coord]} != {candidate.sizes[coord]}"
            )

        if not reference[coord].identical(candidate[coord]):
            diff = float(abs(reference[coord] - candidate[coord]).max())

            raise RuntimeError(
                f"{label}: {coord} values differ; refusing to merge. "
                f"max_abs_diff={diff:.6g}, "
                f"ref first/last={float(reference[coord][0]):.3f}/{float(reference[coord][-1]):.3f}, "
                f"candidate first/last={float(candidate[coord][0]):.3f}/{float(candidate[coord][-1]):.3f}"
            )


def time_windows(start: pd.Timestamp, end: pd.Timestamp, freq: str):
    starts = pd.date_range(start.normalize(), end, freq=freq)

    if len(starts) == 0 or starts[0] > start:
        starts = starts.insert(0, start)

    for i, window_start in enumerate(starts):
        window_start = max(pd.Timestamp(window_start), start)

        if i + 1 < len(starts):
            window_end = min(pd.Timestamp(starts[i + 1]) - pd.Timedelta(seconds=1), end)
        else:
            window_end = end

        if window_start <= window_end:
            yield window_start, window_end


def main():
    for var in variables:
        for chunk_start, chunk_end in time_windows(start, end, save_freq):
            init_times = pd.date_range(chunk_start, chunk_end, freq=init_freq)

            requested_range = (
                f"{chunk_start:%Y-%m-%d %H:%M} -> "
                f"{chunk_end:%Y-%m-%d %H:%M}"
            )

            console.print(f"\n=== {var} | requested: {requested_range} ===")

            forecast_parts = []
            analysis_parts = {}

            tasks = [
                (init_time, lead)
                for init_time in init_times
                for lead in lead_days
            ]

            total_files = len(tasks)

            period = f"{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}"

            fc_path = out_dir / f"forecast_{var}_{period}.zarr"
            an_path = out_dir / f"analysis_{var}_{period}.zarr"

            if fc_path.exists() and an_path.exists():
                continue

            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.fields[var]}[/]"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("[cyan]{task.completed}/{task.total} files[/]"),
                TimeElapsedColumn(),
                TextColumn("remaining:"),
                TimeRemainingColumn(),
                console=console,
            )

            with progress:
                task_id = progress.add_task(
                    "loading",
                    total=total_files,
                    var=var,
                )

                missing_forecast_paths = []
                for init_time, lead in tasks:
                    valid_date = (init_time + pd.Timedelta(days=lead)).normalize()

                    if not fc_path.exists():
                        fc = load_forecast_var(init_time, lead, var)
                        if fc is not None:
                            forecast_parts.append(fc)
                        else:
                            missing_forecast_paths.append((init_time, lead, forecast_file(init_time, var, lead)))

                    if not an_path.exists():
                        if valid_date not in analysis_parts:
                            an = load_analysis_var(valid_date, var)
                            if an is not None:
                                analysis_parts[valid_date] = an

                    progress.advance(task_id)

            if forecast_parts:
                ref = forecast_parts[0]

                for i, ds in enumerate(forecast_parts[1:], start=1):
                    assert_same_lat_lon(ref, ds, f"forecast {var} piece {i}")

                console.print("[cyan]Combining forecast datasets...[/]")
                fc_all = xr.combine_by_coords(
                    forecast_parts,
                    combine_attrs="override",
                    join="outer",
                    coords="minimal",
                    compat="override",
                )

                console.print("[cyan]Reindexing...[/]")
                fc_all = fc_all.reindex(
                    init_time=init_times,
                    leadtime=lead_days,
                )

                # coverage = (
                #     fc_all[var]
                #     .notnull()
                #     .any(dim=("latitude", "longitude"))
                #     .sum()
                #     .item()
                # )
                coverage = len(forecast_parts)

                expected = len(init_times) * len(lead_days)

                console.print(
                    f"coverage={coverage}/{expected} "
                    f"({100*coverage/expected:.1f}%)"
                )

                fc_all = fc_all.sortby(["init_time", "leadtime"])
                fc_all = fc_all.chunk({
                    "init_time": 1,
                    "leadtime": 1,
                    "time": 4,
                    "latitude": 300,
                    "longitude": 600,
                })


                with ProgressBar():
                    fc_all.to_zarr(
                        fc_path,
                        mode="w",
                        zarr_format=2,
                        consolidated=False,
                    )
                console.print("saved:", fc_path)
                console.print(fc_all.coords)
                console.print(
                    f"{var}: shape={fc_all[var].shape}, "
                    f"size={human_size(directory_size(fc_path))}"
                )

                console.print(f"missing forecast files/read failures: {len(missing_forecast_paths)}")

                missing_by_init = {}
                for init_time, lead, path in missing_forecast_paths:
                    missing_by_init.setdefault(init_time, []).append(lead)

                for init_time, leads in list(missing_by_init.items())[:20]:
                    console.print(f"{init_time}: missing leads {leads}")

            if analysis_parts:
                analysis_list = [analysis_parts[t] for t in sorted(analysis_parts)]

                ref = analysis_list[0]

                for t in sorted(analysis_parts):
                    ds = analysis_parts[t]
                    assert_same_lat_lon(ref, ds, f"analysis {var} time={t}")

                console.print("[cyan]Concatenating analysis datasets...[/]")
                an_all = xr.concat(
                    analysis_list,
                    dim="time",
                    combine_attrs="override",
                    join="exact",
                )

                expected_valid_times = sorted({
                    init_time + pd.Timedelta(days=lead)
                    for init_time in init_times
                    for lead in lead_days
                })

                console.print("[cyan]Reindexing...[/]")
                an_all = an_all.reindex(
                    time=expected_valid_times,
                )

                # coverage = (
                #     an_all[var]
                #     .notnull()
                #     .any(dim=("latitude", "longitude"))
                #     .sum()
                #     .item()
                # )
                expected_valid_times = sorted({
                    init_time + pd.Timedelta(days=lead)
                    for init_time in init_times
                    for lead in lead_days
                })

                expected = len(expected_valid_times)
                coverage = len(analysis_parts)

                console.print(
                    f"coverage={coverage}/{expected} "
                    f"({100*coverage/expected:.1f}%)"
                )

                an_all = an_all.chunk({
                    "time": 1,
                    "latitude": 300,
                    "longitude": 600,
                })

                with ProgressBar():
                    an_all.to_zarr(
                        an_path,
                        mode="w",
                        zarr_format=2,
                        consolidated=False,
                    )
                console.print("saved:", an_path)
                console.print(an_all.coords)
                console.print(
                    f"{var}: shape={an_all[var].shape}, "
                    f"size={human_size(directory_size(an_path))}"
                )

                expected_valid_times = sorted({
                    init_time + pd.Timedelta(days=lead)
                    for init_time in init_times
                    for lead in lead_days
                })

                actual_valid_times = {
                    pd.Timestamp(t)
                    for t in an_all.time.values
                }

                console.print(
                    f"analysis missing times: "
                    f"{len(set(expected_valid_times) - actual_valid_times)}"
                )

if __name__ == "__main__":
    main()
