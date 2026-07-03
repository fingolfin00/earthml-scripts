#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import xarray as xr
from dask.diagnostics.progress import ProgressBar

DATA_DIR = Path("/work/cmcc/jd19424/ML/MLBC/data/weather_atmo/raw")
OUT_DIR = Path("/work/cmcc/jd19424/ML/MLBC/data/weather_atmo/input")

TYPE = "analysis"
VAR = "v10"
VAR_OUT = "v10"

paths = sorted(DATA_DIR.glob(f"{TYPE}_{VAR}_*.zarr"))

print(f"Found {len(paths)} datasets")
for p in paths:
    print(p.name)

def preprocess(ds):
    if "time" in ds:
        ds = ds.sortby("time")
        _, index = np.unique(ds["time"].values, return_index=True)
        ds = ds.isel(time=sorted(index))
    return ds

ds = xr.open_mfdataset(
    [str(p) for p in paths],
    engine="zarr",
    combine="nested",
    concat_dim="time",
    preprocess=preprocess,
    chunks="auto",
    parallel=True,
    consolidated=False,
)

ds = ds.sortby("time")

_, index = np.unique(ds["time"].values, return_index=True)
ds = ds.isel(time=sorted(index))

if VAR_OUT != VAR:
    ds = ds.rename({VAR: VAR_OUT})

out_path = OUT_DIR / f"{TYPE}_{VAR_OUT}.zarr"

with ProgressBar():
    ds.to_zarr(
        out_path,
        mode="w",
        zarr_format=2,
        consolidated=False,
    )

ds.close()

print(f"Written: {out_path}")

