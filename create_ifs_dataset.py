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
    "sd": {"shortName": "sd"},
    "lsm": {"shortName": "lsm"},
    "tcc": {"shortName": "tcc"},
    "sp": {"shortName": "sp"},
    "msl": {"shortName": "msl"},
    "v10": {"shortName": "10v"},
    "u10": {"shortName": "10u"},
    "t2m": {"shortName": "2t"},
    "d2m": {"shortName": "2d"},
    "siconc": {"shortName": "ci"},
    "rsn": {"shortName": "rsn"},
}


forecast_dir = Path("/data/inputs/METOCEAN/rolling/model/atmos/ECMWF/IFS_010/1.0forecast/1h/grib/")
analysis_dir = Path("/data/inputs/METOCEAN/historical/model/atmos/ECMWF/IFS_010/analysis/6h/grib/")

out_dir = Path("./ifs_data")
out_dir.mkdir(exist_ok=True)

start = pd.Timestamp("2019-10-14 00:00")
# start = pd.Timestamp("2025-10-01 00:00")
end = pd.Timestamp("2025-10-10 00:00")

init_freq = "12h"
lead_hours = [12, 24, 36, 48, 60, 72,] # 84 only possible with 24h freq
forecast_valid_time_tolerance_hours = 1

variables = [
    # "sd",
    # "lsm",
    "tcc",
    "msl",
    "v10",
    "u10",
    "t2m",
    "d2m",
    # "sp",
    # "siconc",
    # "rsn",
]

save_freq = "YS"  # yearly. Use "MS" for monthly, "QS" quarterly.


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


def forecast_file(init_time: pd.Timestamp, lead: int) -> list[tuple[Path, int]]:
    target_valid_time = init_time + pd.Timedelta(hours=lead)

    candidates = []
    
    offset_order = [0, -forecast_valid_time_tolerance_hours, forecast_valid_time_tolerance_hours]

    for offset in offset_order:
        valid_time = target_valid_time + pd.Timedelta(hours=offset)

        init_mmddhh = init_time.strftime("%m%d%H")
        valid_mmddhh = valid_time.strftime("%m%d%H")

        name = f"JLS{init_mmddhh}00{valid_mmddhh}001"
        path = forecast_dir / init_time.strftime("%Y%m%d") / name

        candidates.append((path, offset))

    return candidates


def analysis_files(valid_time: pd.Timestamp) -> list[Path]:
    mmddhh = valid_time.strftime("%m%d%H")
    month_dir = analysis_dir / valid_time.strftime("%Y") / valid_time.strftime("%m")

    return [
        month_dir / f"JLD{mmddhh}00{mmddhh}001",
        month_dir / f"JLD{mmddhh}00{mmddhh}011",
    ]

def open_grib_var(path: Path, var: str) -> xr.Dataset | None:
    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "indexpath": "",
            "filter_by_keys": VAR_FILTERS[var],
        },
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

    keep_coords = {"latitude", "longitude"}
    drop_coords = [c for c in ds.coords if c not in keep_coords]

    return ds.drop_vars(drop_coords, errors="ignore")


def load_forecast_var(init_time: pd.Timestamp, lead: int, var: str) -> xr.Dataset | None:
    candidates = forecast_file(init_time, lead)

    selected_path = None
    selected_offset = None

    for path, offset in candidates:
        if path.exists():
            selected_path = path
            selected_offset = offset
            break

    if selected_path is None:
        console.print(f"\nmissing forecast lt={lead}:", candidates[forecast_valid_time_tolerance_hours][0])
        return None

    try:
        ds = open_grib_var(selected_path, var)
    except Exception as e:
        console.print("bad forecast:", selected_path, e)
        return None

    if ds is None:
        console.print("forecast missing var:", var, selected_path)
        return None

    target_forecast_time = init_time + pd.Timedelta(hours=lead)
    actual_forecast_time = target_forecast_time + pd.Timedelta(hours=selected_offset)

    ds = clean_var(ds, var)

    ds = ds.expand_dims(
        time=[init_time],
        leadtime=[lead],
    )

    ds = ds.assign_coords(
        forecast_time=(("time", "leadtime"), [[target_forecast_time]]),
        actual_forecast_time=(("time", "leadtime"), [[actual_forecast_time]]),
        forecast_time_offset_hours=(("time", "leadtime"), [[selected_offset]]),
    )

    return ds


def load_analysis_var(valid_time: pd.Timestamp, var: str) -> xr.Dataset | None:
    existing = [p for p in analysis_files(valid_time) if p.exists()]

    if not existing:
        console.print("missing analysis:", valid_time)
        return None

    pieces = []

    for path in existing:
        try:
            ds = open_grib_var(path, var)
        except Exception as e:
            console.print("bad analysis:", path, e)
            continue

        if ds is None:
            continue

        pieces.append(clean_var(ds, var))

    if not pieces:
        console.print("analysis missing var:", var, valid_time)
        return None

    ds = xr.merge(pieces, compat="override")
    ds = ds.expand_dims(time=[valid_time])

    return ds


def assert_same_lat_lon(reference: xr.Dataset, candidate: xr.Dataset, label: str):
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
    init_times = pd.date_range(start, end, freq=init_freq)

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
                for lead in lead_hours
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
                    valid_time = init_time + pd.Timedelta(hours=lead)

                    if not fc_path.exists():
                        fc = load_forecast_var(init_time, lead, var)
                        if fc is not None:
                            forecast_parts.append(fc)
                        else:
                            missing_forecast_paths.append((init_time, lead, forecast_file(init_time, lead)))

                    if not an_path.exists():
                        if valid_time not in analysis_parts:
                            an = load_analysis_var(valid_time, var)
                            if an is not None:
                                analysis_parts[valid_time] = an

                    progress.advance(task_id)

            if forecast_parts:
                ref = forecast_parts[0]

                for i, ds in enumerate(forecast_parts[1:], start=1):
                    assert_same_lat_lon(ref, ds, f"forecast {var} piece {i}")

                fc_all = xr.combine_by_coords(
                    forecast_parts,
                    combine_attrs="override",
                    join="outer",
                    coords="minimal",
                    compat="override",
                )

                fc_all = fc_all.reindex(
                    time=init_times,
                    leadtime=lead_hours,
                )

                # coverage = (
                #     fc_all[var]
                #     .notnull()
                #     .any(dim=("latitude", "longitude"))
                #     .sum()
                #     .item()
                # )
                coverage = len(forecast_parts)

                expected = len(init_times) * len(lead_hours)

                console.print(
                    f"coverage={coverage}/{expected} "
                    f"({100*coverage/expected:.1f}%)"
                )

                fc_all = fc_all.sortby(["time", "leadtime"])
                fc_all = fc_all.chunk({
                    "time": 512,
                    **{
                        d: -1
                        for d in fc_all.dims
                        if d != "time"
                    }
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

                an_all = xr.concat(
                    analysis_list,
                    dim="time",
                    combine_attrs="override",
                    join="exact",
                )

                expected_valid_times = sorted({
                    init_time + pd.Timedelta(hours=lead)
                    for init_time in init_times
                    for lead in lead_hours
                })

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
                    init_time + pd.Timedelta(hours=lead)
                    for init_time in init_times
                    for lead in lead_hours
                })

                expected = len(expected_valid_times)
                coverage = len(analysis_parts)

                console.print(
                    f"coverage={coverage}/{expected} "
                    f"({100*coverage/expected:.1f}%)"
                )

                an_all = an_all.chunk({
                    "time": 512,
                    **{
                        d: -1
                        for d in fc_all.dims
                        if d != "time"
                    }
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
                    init_time + pd.Timedelta(hours=lead)
                    for init_time in init_times
                    for lead in lead_hours
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
