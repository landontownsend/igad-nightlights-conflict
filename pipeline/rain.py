"""
Stage 4: rainfall.
Downloads CHIRPS monthly rainfall for Africa and averages it over each locality,
from the start of the "normal" period (1991) to the end of the study period.

Writes: data/interim/rain_monthly.csv  (locality_id, year, month, rain_mm)
"""

import argparse
import gzip
import shutil

import geopandas as gpd
import pandas as pd
from exactextract import exact_extract

from pipeline.common import download, downloads, interim, load_config


def run(cfg, force=False):
    out = interim(cfg, "rain_monthly.csv")
    if out.exists() and not force:
        print("rain: already built, skipping (use --force to rebuild)")
        return

    localities = gpd.read_file(interim(cfg, "localities.gpkg"))[["locality_id", "geometry"]]
    first = cfg["sources"]["chirps"]["normal_years"][0]
    last = cfg["years"]["end"]
    folder = downloads(cfg, "chirps")
    parts = []
    for year in range(first, last + 1):
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
        print(f"   {year} done")
    pd.concat(parts, ignore_index=True).to_csv(out, index=False)
    print(f"rain: {len(parts)} months saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(load_config(), force=args.force)
