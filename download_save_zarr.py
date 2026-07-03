from typing import Any

import shutil, zipfile, re
from pathlib import Path

import cdsapi

import pandas as pd
import xarray as xr
import earthml

from dask.diagnostics.progress import ProgressBar

from rich import print

from earthml import Settings


# =============================================================================
# Configuration
# =============================================================================

s = Settings()

OUTDIR_ROOT = Path("/Users/jacopodallaglio/ML/training/seasonal/data/download")
OUTDIR_SPS4 = OUTDIR_ROOT / "sps4_seasonal"
OUTDIR_ERA5 = OUTDIR_ROOT / "era5_monthly"
OUTDIR_ORAS5 = OUTDIR_ROOT / "oras5_monthly"

for path in [OUTDIR_SPS4, OUTDIR_ERA5, OUTDIR_ORAS5]:
    path.mkdir(parents=True, exist_ok=True)


YEARS_SPS4 = [str(y) for y in range(1993, 2023)]
YEARS_ERA5 = [str(y) for y in range(1993, 2024)]
YEARS_ORAS5_CONSOLIDATED = [str(y) for y in range(1993, 2015)]
YEARS_ORAS5_OPERATIONAL = [str(y) for y in range(2015, 2024)]

MONTHS = [f"{m:02d}" for m in range(1, 13)]
LEADTIME_MONTHS = [str(i) for i in range(1, 7)]

GLOBAL_AREA = [90, -180, -90, 180]


VARIABLES_ERA5 = {
    "msl": "mean_sea_level_pressure",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "d2m": "2m_dewpoint_temperature",
    "t2m": "2m_temperature",
    "sst": "sea_surface_temperature",
    "tp": "total_precipitation",
}

VARIABLES_ORAS5 = {
    "sss": "sea_surface_salinity",
    "t20d": "depth_of_20_c_isotherm",
    "ssh": "sea_surface_height",
    "mld1": "mixed_layer_depth_0_01",  # verify this name
}

VARIABLES_SPS4_ATMO = {
    "msl": "Mean sea level pressure",
    "u10": "10 metre U wind component",
    "v10": "10 metre V wind component",
    "d2m": "2 metre dewpoint temperature",
    "t2m": "2 metre temperature",
    "sst": "Sea surface temperature",
    "tprate": "Time-mean total precipitation rate",
}

VARIABLES_SPS4_OCEAN = {
    "sos": "sea_surface_salinity",
    "t20d": "depth_of_20_c_isotherm",
    "zos": "sea_surface_height_above_geoid",
    "mlotst": "mixed_layer_depth_0_01",  # verify this name
}

CANONICAL_VARS = {
    # atmosphere
    "msl": "mslp",
    "u10": "u10",
    "v10": "v10",
    "d2m": "d2m",
    "t2m": "t2m",
    "sst": "sst",
    "tp": "tprate",
    "tprate": "tprate",

    # ocean
    "sos": "sss",
    "sss": "sss",
    "sosaline": "sss",
    "t20d": "t20d",
    "so20chgt": "t20d",
    "zos": "ssh",
    "ssh": "ssh",
    "sossheig": "ssh",
    "mlotst": "mlotst",
    "mld1": "mlotst",
    "mlotstheta001": "mlotst",
    "somxl010": "mlotst",
}

OUTDIR_ZARR = Path("/Users/jacopodallaglio/ML/training/seasonal/data/input")
OUTDIR_ZARR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Download
# =============================================================================

REALIZATION_RE = re.compile(r"_r(\d+)i\d+p\d+\.nc$")


def get_realization_from_name(name: str) -> int:
    match = REALIZATION_RE.search(name)
    if match is None:
        raise ValueError(f"Could not parse realization from filename: {name}")
    return int(match.group(1))


def validate_netcdf(path: Path) -> None:
    try:
        with xr.open_dataset(path, decode_timedelta=False):
            pass
    except Exception as exc:
        print(f"[red]Invalid NetCDF file:[/red] {path}")
        print(path.read_bytes()[:500])
        raise exc


def extract_or_combine_netcdf(path: Path, target: Path) -> Path:
    if not zipfile.is_zipfile(path):
        return path

    extract_dir = target.parent / f".{target.name}.extract"

    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(path) as zf:
        nc_names = sorted(name for name in zf.namelist() if name.endswith(".nc"))

        if not nc_names:
            raise ValueError(f"No NetCDF files found in ZIP: {path}")

        extracted_files = []
        for name in nc_names:
            zf.extract(name, extract_dir)
            extracted_files.append(extract_dir / Path(name).name)

    try:
        # ORAS5 / ERA-style ZIP: no realization pattern
        if not all(REALIZATION_RE.search(p.name) for p in extracted_files):
            if len(extracted_files) == 1:
                return extracted_files[0]

            combined_tmp = target.parent / f".{target.stem}.combined.nc"

            if combined_tmp.exists():
                if combined_tmp.is_dir():
                    shutil.rmtree(combined_tmp)
                else:
                    combined_tmp.unlink()

            ds = xr.open_mfdataset(
                sorted(extracted_files),
                combine="by_coords",
                decode_times=True,
                decode_timedelta=False,
            )

            ds.load().to_netcdf(str(combined_tmp), mode="w", engine="netcdf4")
            ds.close()

            if combined_tmp.is_dir():
                raise IsADirectoryError(f"to_netcdf created a directory: {combined_tmp}")

            return combined_tmp

        # SPS ensemble ZIP
        combined_tmp = target.with_name(target.name + ".combined_tmp.nc")

        if combined_tmp.exists():
            if combined_tmp.is_dir():
                shutil.rmtree(combined_tmp)
            else:
                combined_tmp.unlink()

        datasets = []

        for file in sorted(extracted_files, key=lambda p: get_realization_from_name(p.name)):
            realization = get_realization_from_name(file.name)

            with xr.open_dataset(file, decode_times=True, decode_timedelta=False) as ds:
                ds = ds.load()
                ds = ds.expand_dims(realization=[realization])
                datasets.append(ds)

        combined = xr.concat(
            datasets,
            dim="realization",
            coords="minimal",
            compat="override",
        )

        combined.to_netcdf(combined_tmp, mode="w")
        combined.close()

        return combined_tmp

    finally:
        path.unlink(missing_ok=True)
        shutil.rmtree(extract_dir, ignore_errors=True)


CLIENT = cdsapi.Client()

def download_file(
    dataset: str,
    request: dict[str, Any],
    target: Path,
    *,
    validate: bool = True,
    overwrite: bool = False,
) -> None:
    if target.exists() and not overwrite:
        if target.is_file():
            validate_netcdf(target)
            print(f"[yellow]Already exists:[/yellow] {target}")
            return

        print(f"[yellow]Removing broken directory:[/yellow] {target}")
        shutil.rmtree(target)

    tmp = target.with_suffix(target.suffix + ".tmp")

    if tmp.exists():
        tmp.unlink()

    if target.exists():
        target.unlink()

    print(f"[cyan]Downloading:[/cyan] {target}")
    CLIENT.retrieve(dataset, request, str(tmp))

    downloaded = extract_or_combine_netcdf(tmp, target)

    print(f"DEBUG downloaded={downloaded}")
    print(f"DEBUG is_file={downloaded.is_file()} is_dir={downloaded.is_dir()}")

    if downloaded.is_dir():
        raise IsADirectoryError(f"Expected NetCDF file, got directory: {downloaded}")

    if validate:
        validate_netcdf(downloaded)

    if target.exists() and target.is_dir():
        print(f"[yellow]Removing broken target directory:[/yellow] {target}")
        shutil.rmtree(target)

    if downloaded != target:
        downloaded.rename(target)

    print(f"[green]Saved:[/green] {target}")


# =============================================================================
# Request builders
# =============================================================================

def sps4_atmo_request(year: str, month: str) -> dict[str, Any]:
    return {
        "originating_centre": "cmcc",
        "system": "4",
        "variable": list(VARIABLES_SPS4_ATMO.values()),
        "product_type": ["monthly_mean"],
        "year": [year],
        "month": [month],
        "leadtime_month": LEADTIME_MONTHS,
        "area": GLOBAL_AREA,
        "data_format": "netcdf",
    }


def sps4_ocean_request(long_var: str, year: str, month: str) -> dict[str, Any]:
    return {
        "originating_centre": "cmcc",
        "system": "4",
        "variable": [long_var],
        "year": [year],
        "month": [month],
        "forecast_type": ["hindcast"],
        "data_format": "netcdf",
    }


def era5_request(long_var: str, year: str) -> dict[str, Any]:
    return {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [long_var],
        "year": [year],
        "month": MONTHS,
        "time": ["00:00"],
        "area": GLOBAL_AREA,
        "data_format": "netcdf",
    }


def oras5_request(long_var: str, year: str, product_type: str) -> dict[str, Any]:
    return {
        "product_type": [product_type],
        "vertical_resolution": ["single_level"],
        "variable": [long_var],
        "year": [year],
        "month": MONTHS,
        "time": ["00:00"],
        "data_format": "netcdf",
    }


# =============================================================================
# Download routines
# =============================================================================

def download_sps4_atmosphere() -> None:
    dataset = "seasonal-monthly-single-levels"

    for year in YEARS_SPS4:
        for month in MONTHS:
            target = OUTDIR_SPS4 / f"cmcc_sps4_atmo_{year}_{month}.nc"
            request = sps4_atmo_request(year, month)
            download_file(dataset, request, target)


def download_sps4_ocean() -> None:
    dataset = "seasonal-monthly-ocean"

    for short_var, long_var in VARIABLES_SPS4_OCEAN.items():
        for year in YEARS_SPS4:
            for month in MONTHS:
                target = OUTDIR_SPS4 / f"cmcc_sps4_ocean_{short_var}_{year}_{month}.nc"
                request = sps4_ocean_request(long_var, year, month)
                download_file(dataset, request, target)


def download_era5_monthly() -> None:
    dataset = "reanalysis-era5-single-levels-monthly-means"

    for short_var, long_var in VARIABLES_ERA5.items():
        for year in YEARS_ERA5:
            target = OUTDIR_ERA5 / f"era5_{short_var}_monthly_{year}.nc"
            request = era5_request(long_var, year)
            download_file(dataset, request, target)


def download_oras5_monthly() -> None:
    dataset = "reanalysis-oras5"

    jobs = [
        ("consolidated", YEARS_ORAS5_CONSOLIDATED),
        ("operational", YEARS_ORAS5_OPERATIONAL),
    ]

    for product_type, years in jobs:
        for short_var, long_var in VARIABLES_ORAS5.items():
            for year in years:
                target = OUTDIR_ORAS5 / f"oras5_{short_var}_monthly_{year}.nc"
                request = oras5_request(long_var, year, product_type)
                download_file(dataset, request, target)



# =============================================================================
# Save Zarr stores
# =============================================================================

TEMPERATURE_VARS = {"t2m", "d2m", "sst"}


def celsius_to_kelvin(da: xr.DataArray) -> xr.DataArray:
    units = str(da.attrs.get("units", "")).lower().replace(" ", "")

    if units in {"c", "degc", "degree_celsius", "degrees_celsius", "celsius"}:
        da = da + 273.15
        da.attrs["units"] = "K"

    return da


def tp_to_tprate(da: xr.DataArray) -> xr.DataArray:
    units = str(da.attrs.get("units", "")).lower().replace(" ", "")

    # Already a rate
    if units in {"m/s", "ms-1", "kgm-2s-1", "kg/m2/s"}:
        da.attrs["units"] = "m s-1"
        return da

    # Monthly total precipitation depth -> monthly mean precipitation rate
    if "time" not in da.coords:
        raise ValueError("Cannot convert tp to tprate without a time coordinate.")

    seconds_in_month = da["time"].dt.days_in_month * 24 * 60 * 60
    da = da / seconds_in_month
    da.attrs["units"] = "m s-1"
    da.attrs["standard_name"] = "precipitation_flux"
    da.attrs["long_name"] = "Time-mean total precipitation rate"

    return da


def standardize_final_variable(ds: xr.Dataset, source_var: str) -> xr.Dataset:
    canonical_var = CANONICAL_VARS.get(source_var, source_var)

    if source_var not in ds:
        alias = next(
            (
                str(ds_var)
                for ds_var in ds.data_vars
                if CANONICAL_VARS.get(str(ds_var), str(ds_var)) == canonical_var
            ),
            None,
        )

        if alias is None:
            raise KeyError(
                f"{source_var} not found in dataset. "
                f"Available data vars: {list(ds.data_vars)}"
            )

        source_var = alias

    da = ds[source_var]

    if canonical_var in TEMPERATURE_VARS:
        da = celsius_to_kelvin(da)

    if source_var == "tp":
        da = tp_to_tprate(da)
    elif source_var == "tprate":
        da.attrs["units"] = "m s-1"

    return da.to_dataset(name=canonical_var)


def fix_lon(ds: xr.Dataset) -> xr.Dataset:
    """Convert longitude from 0/360 to -180/180 and sort."""
    if "longitude" not in ds.coords:
        return ds

    lon = ((ds.longitude + 180) % 360) - 180
    ds = ds.assign_coords(longitude=lon)
    return ds.sortby("longitude")


def clean_time_names(ds: xr.Dataset) -> xr.Dataset:
    """Use a common time coordinate name where possible."""
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    return ds


def remove_existing_zarr(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)


def save_zarr(
    ds: xr.Dataset,
    target: Path,
    *,
    chunks: dict | None = None,
    overwrite: bool = True,
) -> None:
    """Save dataset to a consolidated Zarr store."""
    remove_existing_zarr(target, overwrite=overwrite)

    if chunks is not None:
        ds = ds.chunk(chunks)

    print(f"[cyan]Saving Zarr:[/cyan] {target}")

    ds.to_zarr(
        target,
        mode="w",
        consolidated=True,
        zarr_format=2,
    )

    print(f"[green]Saved:[/green] {target}")


def group_files_by_variable(
    files: list[Path],
    variables: dict[str, str],
    *,
    prefix: str,
    suffix: str,
) -> dict[str, list[Path]]:
    """
    Group files by short variable name using exact filename structure.

    Example:
        era5_t2m_monthly_1993.nc
        oras5_ssh_monthly_2010.nc
    """
    grouped = {short_var: [] for short_var in variables}

    for file in files:
        name = file.name

        for short_var in variables:
            expected_start = f"{prefix}_{short_var}_{suffix}_"
            if name.startswith(expected_start):
                grouped[short_var].append(file)

    return {k: sorted(v) for k, v in grouped.items()}


def open_reanalysis_variable(files: list[Path], short_var: str) -> xr.Dataset:
    if not files:
        raise FileNotFoundError("No files found for this variable.")

    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        decode_times=True,
    )

    ds = fix_lon(ds)
    ds = clean_time_names(ds)
    ds = standardize_final_variable(ds, short_var)

    return ds


# =============================================================================
# Reanalysis: ERA5 and ORAS5
# =============================================================================

def save_reanalysis_zarrs(
    *,
    name: str,
    files: dict[str, list[Path]],
    outdir: Path,
    chunks: dict | None = None,
) -> dict[str, xr.Dataset]:
    datasets = {}

    for short_var, var_files in files.items():
        print(f"[bold]Processing {name.upper()} {short_var}[/bold]")

        an = open_reanalysis_variable(var_files, short_var)

        an = an.earthml.normalize_dims_and_coords()

        an = an.assign_coords(time=an.indexes["time"].to_period("M").to_timestamp())

        canonical_var = CANONICAL_VARS.get(short_var, short_var)
        target = outdir / f"{name}_{canonical_var}.zarr"
        save_zarr(an, target, chunks=chunks)

        datasets[short_var] = an

    return datasets

REANALYSIS_CHUNKS = {
    "time": 12,
    "latitude": 180,
    "longitude": 360,
}

def save_era5_to_zarr():
    era5_files_list = sorted(OUTDIR_ERA5.glob("era5_*_monthly_*.nc"))

    era5_files = group_files_by_variable(
        era5_files_list,
        VARIABLES_ERA5,
        prefix="era5",
        suffix="monthly",
    )

    _ = save_reanalysis_zarrs(
        name="era5",
        files=era5_files,
        outdir=OUTDIR_ZARR,
        chunks=REANALYSIS_CHUNKS,
    )

def save_oras5_to_zarr():
    oras5_files_list = sorted(OUTDIR_ORAS5.glob("oras5_*_monthly_*.nc"))

    oras5_files = group_files_by_variable(
        oras5_files_list,
        VARIABLES_ORAS5,
        prefix="oras5",
        suffix="monthly",
    )

    _ = save_reanalysis_zarrs(
        name="oras5",
        files=oras5_files,
        outdir=OUTDIR_ZARR,
        chunks=REANALYSIS_CHUNKS,
    )


# =============================================================================
# Forecast helpers
# =============================================================================

def standardize_forecast_dims(da: xr.DataArray) -> xr.DataArray:
    rename_map = {}

    if "forecastMonth" in da.dims:
        rename_map["forecastMonth"] = "leadtime"

    if "number" in da.dims:
        rename_map["number"] = "realization"

    if "member" in da.dims:
        rename_map["member"] = "realization"

    if rename_map:
        da = da.rename(rename_map)

    return da


def forecast_var_to_time(
    fc: xr.Dataset,
    var: str,
) -> xr.Dataset:
    """
    Convert one forecast file and one variable into a time-based forecast dataset.

    Output dimensions are typically:
        time, realization, leadtime, latitude, longitude

    Coordinates:
        time        = forecast initialization date
        leadtime    = forecast month
    """
    if "forecast_reference_time" in fc.coords:
        init_time = pd.to_datetime(fc["forecast_reference_time"].values[0])
    elif "reftime" in fc.coords:
        init_time = pd.to_datetime(fc["reftime"].values)
    else:
        raise ValueError("Missing forecast_reference_time/reftime coordinate.")

    da = fc[var]

    if "forecast_reference_time" in da.dims:
        da = da.squeeze("forecast_reference_time", drop=True)

    da = standardize_forecast_dims(da)

    if "leadtime" not in da.dims:
        raise ValueError(f"{var} has no leadtime/forecastMonth dimension.")

    if "time" in da.dims:
        da = da.squeeze("time", drop=True)

    if "time" in da.coords:
        da = da.drop_vars("time", errors="ignore")

    da = da.expand_dims(time=[init_time])

    ds = da.to_dataset(name=var)
    ds = standardize_final_variable(ds, var)
    ds = ds.assign_coords(time=ds.indexes["time"].to_period("M").to_timestamp())

    return ds


def save_forecast_variable_zarr(
    *,
    files: list[Path],
    var: str,
    target: Path,
    chunks: dict | None = None,
) -> xr.Dataset:
    samples: list[xr.Dataset] = []

    for _, file in enumerate(files):
        print(f"Processing {file.name} | {var}")

        with xr.open_dataset(file, decode_times=True, decode_timedelta=False) as fc:
            fc = fc.earthml.normalize_dims_and_coords()
            fc = fix_lon(fc)

            if var not in fc:
                alias = next(
                    (
                        str(fc_var)
                        for fc_var in fc.data_vars
                        if CANONICAL_VARS.get(str(fc_var)) == var
                    ),
                    None,
                )

                if alias is None:
                    print(f"[yellow]Skipping {file.name}: {var} not found[/yellow]")
                    continue

                fc = fc.rename({alias: var})

            sample = forecast_var_to_time(fc, var)
            samples.append(sample)

    if not samples:
        raise ValueError(f"No forecast samples found for variable {var}")

    with ProgressBar():
        ds_out = xr.concat(
            samples,
            dim="time",
            coords="minimal",
            compat="override",
            join="inner",
        ).compute()

    save_zarr(ds_out, target, chunks=chunks)

    return ds_out


# =============================================================================
# SPS4
# =============================================================================
SPS4_FORECAST_CHUNKS = {
    "time": 1,
    "realization": -1,
    "leadtime": -1,
    "latitude": 180,
    "longitude": 360,
}

def save_sps4_atmo_to_zarr():
    sps4_atmo_files = sorted(OUTDIR_SPS4.glob("cmcc_sps4_atmo_*.nc"))

    sps4_atmo = {}

    for short_var in VARIABLES_SPS4_ATMO:
        canonical_var = CANONICAL_VARS.get(short_var, short_var)
        target = OUTDIR_ZARR / f"sps4_atmo_{canonical_var}.zarr"

        sps4_atmo[short_var] = save_forecast_variable_zarr(
            files=sps4_atmo_files,
            var=short_var,
            target=target,
            chunks=SPS4_FORECAST_CHUNKS,
        )


def sps4_ocean_files_for_variable(short_var: str) -> list[Path]:
    return sorted(OUTDIR_SPS4.glob(f"cmcc_sps4_ocean_{short_var}_*.nc"))


def save_sps4_ocean_to_zarr():
    sps4_ocean = {}

    for short_var in VARIABLES_SPS4_OCEAN:
        canonical_var = CANONICAL_VARS.get(short_var, short_var)
        target = OUTDIR_ZARR / f"sps4_ocean_{canonical_var}.zarr"

        sps4_ocean[short_var] = save_forecast_variable_zarr(
            files=sps4_ocean_files_for_variable(short_var),
            var=short_var,
            target=target,
            chunks=SPS4_FORECAST_CHUNKS,
        )

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    # download_sps4_atmosphere()
    # save_sps4_atmo_to_zarr()

    # download_sps4_ocean()
    # save_sps4_ocean_to_zarr()

    # download_era5_monthly()
    # save_era5_to_zarr()

    download_oras5_monthly()
    save_oras5_to_zarr()

if __name__ == "__main__":
    main()
