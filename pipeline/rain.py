"""
Stage 4: rainfall.
Downloads CHIRPS monthly rainfall for Africa and averages it over each locality,
from the start of the "normal" period (1991) to the end of the study period.

Writes one file per year, so a failed run keeps the years it finished:
  data/interim/rain/rain_<year>.csv  (locality_id, year, month, rain_mm)
"""

import argparse
import gzip
import shutil

import geopandas as gpd
import pandas as pd
from exactextract import exact_extract

from pipeline.common import download, downloads, interim, load_config


def run(cfg, force=False):
    folder_out = interim(cfg, "rain")
    folder_out.mkdir(parents=True, exist_ok=True)
    localities = gpd.read_file(interim(cfg, "localities.gpkg"))[["locality_id", "geometry"]]
    first = cfg["sources"]["chirps"]["normal_years"][0]
    last = cfg["years"]["end"]
    folder = downloads(cfg, "chirps")
    for year in range(first, last + 1):
        out = folder_out / f"rain_{year}.csv"
        if out.exists() and not force:
            continue                      # saved by an earlier run
        parts = []
        for month in range(1, 13):
            url = cfg["sources"]["chirps"]["url"].format(year=year, month=month)
            gz = download(url, folder / f"{year}{month:02d}.tif.gz")
            if gz is None:
                print(f"   {year}-{month:02d}: not published, skipped")
                continue
            tif = folder / f"{year}{month:02d}.tif"
            with gzip.open(gz, "rb") as src, open(tif, "wb") as dst:
                shutil.copyfileobj(src, dst)
            stats = exact_extract(str(tif), localities, ["mean"], output="pandas")
            parts.append(pd.DataFrame({"locality_id": localities["locality_id"],
                                       "year": year, "month": month,
                                       "rain_mm": stats["mean"].round(2)}))
            gz.unlink()
            tif.unlink()
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(out, index=False)
        print(f"   {year}: {len(parts)} months saved")
    print("rain: done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(load_config(), force=args.force)
