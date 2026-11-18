"""
Stage 5: population.
Downloads WorldPop yearly population (1 km) for each country and adds it up
within each locality. WorldPop's consistent series ends in 2020, so the build
step reuses the 2020 figure for later years.

Writes: data/interim/population.csv  (locality_id, year, population)
"""

import argparse

import geopandas as gpd
import pandas as pd
from exactextract import exact_extract

from pipeline.common import country_codes, download, downloads, interim, load_config


def run(cfg, force=False):
    out = interim(cfg, "population.csv")
    if out.exists() and not force:
        print("population: already built, skipping (use --force to rebuild)")
        return

    localities = gpd.read_file(interim(cfg, "localities.gpkg"))[["locality_id", "country", "geometry"]]
    last = cfg["sources"]["worldpop"]["last_year"]
    folder = downloads(cfg, "worldpop")
    parts = []
    for iso3 in country_codes(cfg):
        country = localities[localities["country"] == iso3]
        for year in range(cfg["years"]["start"], last + 1):
            url = cfg["sources"]["worldpop"]["url"].format(year=year, ISO=iso3, iso=iso3.lower())
            tif = download(url, folder / f"{iso3}_{year}.tif")
            if tif is None:
                print(f"   {iso3} {year}: not found, skipped")
                continue
            stats = exact_extract(str(tif), country, ["sum"], output="pandas")
            parts.append(pd.DataFrame({"locality_id": country["locality_id"].to_numpy(),
                                       "year": year, "population": stats["sum"].round(0)}))
            tif.unlink()
        print(f"   {iso3} done")
    pd.concat(parts, ignore_index=True).to_csv(out, index=False)
    print("population: saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(load_config(), force=args.force)
