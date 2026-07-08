import shutil
from pathlib import Path

import xarray as xr
from dask.diagnostics import ProgressBar

from numcodecs import Blosc


INPUT_DIR = Path("/work/cmcc/jd19424/ML/MLBC/data/weather_atmo/input")

variables = [
    "tcc",
    "msl",
    "v10",
    "u10",
    "t2m",
    "d2m",
]

types = [
    "forecast",
    "analysis",
]

compressor = Blosc(
    cname="zstd",     # or "lz4"
    clevel=3,         # 1-5 is usually a good tradeoff
    shuffle=Blosc.BITSHUFFLE,
)

for typ in types:
    for var in variables:
        print(f"\n{'='*80}")
        print(f"{typ} - {var}")
        print(f"{'='*80}")

        path = INPUT_DIR / f"{typ}_{var}.zarr"

        if not path.exists():
            print("Missing, skipping.")
            continue

        tmp = INPUT_DIR / f"{typ}_{var}_rechunked.zarr"
        backup = INPUT_DIR / f"{typ}_{var}_old_chunks.zarr"

        ds = xr.open_zarr(path, consolidated=False)

        print("Old chunks:")
        print(ds.chunks)

        chunks = {
            "time": 512,
            "latitude": -1,
            "longitude": -1,
        }

        if "leadtime" in ds.dims:
            chunks["leadtime"] = -1

        ds = ds.chunk(chunks)

        print("New chunks:")
        print(ds.chunks)

        if tmp.exists():
            shutil.rmtree(tmp)

        encoding = {
            var: {"compressor": compressor}
            for var in ds.data_vars
        }

        with ProgressBar():
            ds.to_zarr(
                path,
                mode="w",
                encoding=encoding,
                zarr_format=2,
                consolidated=False,
            )

        if backup.exists():
            shutil.rmtree(backup)

        shutil.move(path, backup)
        shutil.move(tmp, path)

        print("Done")
