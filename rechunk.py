import shutil
from pathlib import Path

import xarray as xr
from dask.diagnostics.progress import ProgressBar

from numcodecs import Blosc


INPUT_DIR = Path("/Users/jacopodallaglio/ML/training/seasonal/data/input")

variables = [
    # Atmo
    # "tcc",
    "mslp",
    "v10",
    "u10",
    "t2m",
    "d2m",
    "sst",
    "tprate",
    # Ocean
    # "mlotst",
    # "sss",
    # "ssh",
]

types = [
    "era5",
    # "oras5",
    "sps4_atmo",
    # "sps4_ocean",
    # "forecast",
    # "analysis",
]

compressor = Blosc(
    cname="zstd",
    clevel=3,
    shuffle=Blosc.BITSHUFFLE,
)

for typ in types:
    for var in variables:
        print(f"\n{'=' * 80}")
        print(f"{typ} - {var}")
        print(f"{'=' * 80}")

        path = INPUT_DIR / f"{typ}_{var}.zarr"

        if not path.exists():
            print("Missing, skipping.")
            continue

        tmp = INPUT_DIR / f"{typ}_{var}_rechunked.zarr"
        backup = INPUT_DIR / f"{typ}_{var}_old_chunks.zarr"

        if tmp.exists():
            shutil.rmtree(tmp)

        ds = xr.open_zarr(path, consolidated=False)

        if not ds.data_vars:
            print(f"WARNING: {path} has no data variables. Skipping.")
            continue

        print("Old chunks:")
        print(ds.chunks)

        chunks = {
            "time": 64 if "realization" not in ds.dims else 16,
            "latitude": -1,
            "longitude": -1,
        }

        if "leadtime" in ds.dims:
            chunks["leadtime"] = -1

        if "realization" in ds.dims:
            chunks["realization"] = -1

        chunks = {k: v for k, v in chunks.items() if k in ds.dims}

        ds = ds.chunk(chunks)

        print("New chunks:")
        print(ds.chunks)

        for name in ds.variables:
            ds[name].encoding.pop("chunks", None)
            ds[name].encoding.pop("preferred_chunks", None)

        encoding = {
            name: {"compressor": compressor}
            for name in ds.data_vars
        }

        print(f"Writing temporary store: {tmp}")

        with ProgressBar():
            ds.to_zarr(
                tmp,
                mode="w",
                encoding=encoding,
                zarr_format=2,
                consolidated=False,
            )

        check = xr.open_zarr(tmp, consolidated=False)

        if not check.data_vars:
            raise RuntimeError(f"Temporary store has no data variables: {tmp}")

        if "time" not in check.dims:
            raise RuntimeError(f"Temporary store has no time dimension: {tmp}")

        print("Validated temporary store:")
        print(check)

        if backup.exists():
            shutil.rmtree(backup)

        print(f"Moving original to backup: {backup}")
        shutil.move(path, backup)

        print(f"Moving temporary to final path: {path}")
        shutil.move(tmp, path)

        print("Done")
