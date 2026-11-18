"""
Stage 2: night lights.
For each month of one year, downloads the NASA Black Marble monthly tiles
(VNP46A3) that cover the study countries and adds up the light inside each
locality. Run once per year, so the years can run in parallel:
  python -m pipeline.lights --year 2023

The Black Marble monthly product fills cloudy pixels with values from earlier
years (quality flag 2). Those pixels are removed, because a filled-in value
would hide exactly the decline this project looks for.

For each locality and month it records four sums, which can simply be added
across tiles when a locality crosses a tile edge:
  ntl_sum    total radiance of the clear pixels
  valid_px   number of clear pixels
  land_px    number of pixels with any reading (the locality's size in pixels)
  num_sum    total number of nightly observations behind the clear pixels
  lit_sum    total radiance of clear pixels at or above the lit threshold (0.5)
  lit_px     number of those lit pixels
  lit2_sum, lit2_px   the same with the stricter threshold (2.0)

Needs a NASA Earthdata token in the BLACKMARBLE_TOKEN environment variable.

Writes one file per month, so a failed run keeps the months it finished:
  data/interim/lights/lights_<year>_<month>.csv
"""

import argparse
import datetime
import os
import time
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401  (adds the .rio accessor used below)
from exactextract import exact_extract

from pipeline.common import downloads, interim, load_config

DATA_FIELDS = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields"
TILE_DEGREES = 10          # each Black Marble tile covers 10 x 10 degrees


def read_scaled(dataset):
    """Read an HDF5 dataset as floats, with fill values set to missing and the scale applied."""
    data = dataset[:].astype("float64")
    fill = dataset.attrs.get("_FillValue")
    if fill is not None:
        data[data == np.asarray(fill).flat[0]] = np.nan
    scale = np.asarray(dataset.attrs.get("scale_factor", 1)).flat[0]
    offset = np.asarray(dataset.attrs.get("add_offset", dataset.attrs.get("offset", 0))).flat[0]
    return data * scale + offset


def tile_bounds(h, v):
    """Corner coordinates of tile h, v: tile h00v00 starts at 180 W, 90 N."""
    west = -180 + TILE_DEGREES * h
    north = 90 - TILE_DEGREES * v
    return west, north - TILE_DEGREES, west + TILE_DEGREES, north


def to_raster(array, bounds):
    """Wrap a 2-D array as a georeferenced raster that exactextract can read."""
    west, south, east, north = bounds
    rows, cols = array.shape
    x = west + (np.arange(cols) + 0.5) * (east - west) / cols
    y = north - (np.arange(rows) + 0.5) * (north - south) / rows
    da = xr.DataArray(array.astype("float32"), coords={"y": y, "x": x}, dims=("y", "x"))
    return da.rio.write_crs("EPSG:4326").rio.write_nodata(np.nan)


def process_tile(path, localities, variable, drop_flags, lit=0.5, lit_strict=2.0):
    """Add up light, clear pixels, all pixels, and observations for each locality in one tile."""
    with h5py.File(path, "r") as f:
        fields = f[DATA_FIELDS]
        h = int(f.attrs["HorizontalTileNumber"])
        v = int(f.attrs["VerticalTileNumber"])
        radiance = read_scaled(fields[variable])
        quality = fields[f"{variable}_Quality"][:]
        observations = read_scaled(fields[f"{variable}_Num"])

    bounds = tile_bounds(h, v)
    west, south, east, north = bounds
    inside = localities.cx[west:east, south:north]
    if inside.empty:
        return pd.DataFrame()

    has_reading = quality != 255
    clear = has_reading & ~np.isin(quality, drop_flags) & ~np.isnan(radiance)

    layers = {
        "ntl_sum": np.where(clear, radiance, np.nan),
        "valid_px": np.where(clear, 1.0, np.nan),
        "land_px": np.where(has_reading, 1.0, np.nan),
        "num_sum": np.where(clear, observations, np.nan),
        "lit_sum": np.where(clear & (radiance >= lit), radiance, np.nan),
        "lit_px": np.where(clear & (radiance >= lit), 1.0, np.nan),
        "lit2_sum": np.where(clear & (radiance >= lit_strict), radiance, np.nan),
        "lit2_px": np.where(clear & (radiance >= lit_strict), 1.0, np.nan),
    }
    result = inside[["locality_id"]].reset_index(drop=True)
    for name, array in layers.items():
        stats = exact_extract(to_raster(array, bounds), inside, ["sum"], output="pandas")
        result[name] = stats["sum"].fillna(0).to_numpy()
    return result


TOKEN_MESSAGE = ("NASA rejected the Earthdata token. Log in once at "
                 "https://ladsweb.modaps.eosdis.nasa.gov with your Earthdata account, then "
                 "generate a new token at https://urs.earthdata.nasa.gov and update the "
                 "NASA_EARTHDATA_TOKEN secret.")


def download_month(downloader, outline, product, month_start):
    """
    Download one month of tiles. Returns the file paths, or None if the month
    has to be skipped. The NASA archive sometimes answers "not found" (404) for
    files it does have, so a 404 is tried three times before the month is skipped.
    """
    for attempt in range(1, 4):
        try:
            return downloader.download(gdf=outline, product_id=product,
                                       date_range=[month_start], skip_if_exists=True)
        except Exception as e:
            message = str(e)
            if "token" in message.lower() or "401" in message or "403" in message:
                raise RuntimeError(TOKEN_MESSAGE) from e   # affects every month: stop now
            if "404" in message:
                if attempt < 3:
                    print(f"   {month_start:%Y-%m}: NASA answered 404, retrying in {60 * attempt} s")
                    time.sleep(60 * attempt)
                    continue
                print(f"   {month_start:%Y-%m}: skipped, NASA answered 404 three times")
                return None
            if isinstance(e, ValueError):   # blackmarblepy: a tile is missing for this month
                print(f"   {month_start:%Y-%m}: skipped, tiles missing on the NASA server")
                return None
            raise


def process_month(cfg, localities, outline, month_start, token):
    """Download one month of tiles, add them up per locality, then delete the tiles."""
    from blackmarble.download import BlackMarbleDownloader
    from blackmarble.types import Product

    bm = cfg["sources"]["black_marble"]
    folder = downloads(cfg, f"blackmarble/{month_start:%Y%m}")
    downloader = BlackMarbleDownloader(token, folder)
    paths = download_month(downloader, outline, Product(bm["product"]), month_start)
    if not paths:
        print(f"   {month_start:%Y-%m}: no tiles available")
        return pd.DataFrame()

    parts = [process_tile(p, localities, bm["variable"], bm["drop_quality_flags"],
                          bm["lit_threshold"], bm["lit_threshold_strict"]) for p in paths]
    month = pd.concat(parts, ignore_index=True).groupby("locality_id", as_index=False).sum()
    month["year"], month["month"] = month_start.year, month_start.month
    for p in paths:                       # free the disk space before the next month
        Path(p).unlink(missing_ok=True)
    print(f"   {month_start:%Y-%m}: {len(paths)} tiles, {len(month)} localities")
    return month


def run(cfg, year, force=False):
    folder_out = interim(cfg, "lights")
    folder_out.mkdir(parents=True, exist_ok=True)
    todo = [m for m in range(1, 13)
            if force or not (folder_out / f"lights_{year}_{m:02d}.csv").exists()]
    if not todo:
        print(f"lights {year}: already built, skipping (use --force to rebuild)")
        return
    token = os.environ.get("BLACKMARBLE_TOKEN")
    if not token:
        raise RuntimeError("Set the BLACKMARBLE_TOKEN environment variable to a NASA Earthdata token")

    localities = gpd.read_file(interim(cfg, "localities.gpkg"))
    # A simplified outline of all study countries, used only to choose which tiles to download
    outline = gpd.GeoDataFrame(geometry=[localities.union_all().simplify(0.05)], crs="EPSG:4326")
    columns = ["locality_id", "year", "month", "ntl_sum", "valid_px", "land_px", "num_sum",
               "lit_sum", "lit_px", "lit2_sum", "lit2_px"]
    saved = 0
    for m in todo:
        month = process_month(cfg, localities, outline, datetime.date(year, m, 1), token)
        if not month.empty:
            month[columns].to_csv(folder_out / f"lights_{year}_{m:02d}.csv", index=False)
            saved += 1
    print(f"lights {year}: {saved} of {len(todo)} months saved")
    if saved == 0:
        raise RuntimeError(f"No Black Marble data could be processed for {year}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(load_config(), args.year, force=args.force)
