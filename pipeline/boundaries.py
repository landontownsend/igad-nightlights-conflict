"""
Stage 1: boundaries.
Downloads the second-level administrative areas (localities, districts, or
zones) for each study country from geoBoundaries, and finds each locality's
neighbors.

Writes:
  data/interim/localities.gpkg   one polygon per locality
  data/interim/localities.csv    the same without geometry, plus area and centroid
  data/interim/neighbors.csv     one row per pair of localities that share a border
"""

import argparse
import warnings

import geopandas as gpd
import pandas as pd
import requests

from pipeline.common import country_codes, interim, load_config


def fetch_country(cfg, iso3):
    """Download one country's boundaries. Uses ADM2, or ADM1 if ADM2 does not exist."""
    api = cfg["sources"]["boundaries"]["api"]
    for level in [cfg["sources"]["boundaries"]["level"], "ADM1"]:
        r = requests.get(f"{api}/{iso3}/{level}/", timeout=60)
        if r.status_code == 200:
            meta = r.json()
            if isinstance(meta, list):          # the API sometimes wraps the answer in a list
                meta = meta[0]
            # The simplified outline keeps the files small; at the 500 m scale of the
            # night-light pixels the difference from the full outline is negligible
            url = meta.get("simplifiedGeometryGeoJSON") or meta["gjDownloadURL"]
            gdf = gpd.read_file(url)
            print(f"   {iso3}: {len(gdf)} areas at {level}")
            return gdf, level
        print(f"   {iso3}: no {level} boundaries (HTTP {r.status_code})")
    raise RuntimeError(f"No boundaries found for {iso3}")


def find_neighbors(localities):
    """
    Two localities are neighbors if their borders touch. Borders from different
    sources rarely line up exactly, so each polygon is widened by about 500
    meters before checking, which also links localities across country borders.
    """
    widened = localities[["locality_id", "geometry"]].copy()
    with warnings.catch_warnings():          # a buffer in degrees is fine for this purpose
        warnings.simplefilter("ignore", UserWarning)
        widened["geometry"] = widened.geometry.buffer(0.005)
    pairs = gpd.sjoin(widened, localities[["locality_id", "geometry"]],
                      predicate="intersects", how="inner")
    pairs = pairs[pairs["locality_id_left"] != pairs["locality_id_right"]]
    return (pairs[["locality_id_left", "locality_id_right"]]
            .rename(columns={"locality_id_left": "locality_id",
                             "locality_id_right": "neighbor_id"})
            .drop_duplicates()
            .sort_values(["locality_id", "neighbor_id"])
            .reset_index(drop=True))


def run(cfg, force=False):
    out = interim(cfg, "localities.gpkg")
    if out.exists() and not force:
        print("boundaries: already built, skipping (use --force to rebuild)")
        return

    parts = []
    for iso3 in country_codes(cfg):
        gdf, level = fetch_country(cfg, iso3)
        gdf = gdf.to_crs("EPSG:4326")
        gdf["country"] = iso3
        gdf["level"] = level
        # geoBoundaries gives each area a permanent ID (shapeID) and a name (shapeName)
        gdf["locality_id"] = gdf["shapeID"].astype(str)
        gdf["locality_name"] = gdf["shapeName"].astype(str)
        parts.append(gdf[["locality_id", "locality_name", "country", "level", "geometry"]])

    localities = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")
    localities["geometry"] = localities.geometry.make_valid()
    if localities["locality_id"].duplicated().any():
        raise ValueError("Duplicate locality IDs in the boundary files")

    # Area and centroid, calculated in an equal-area projection
    equal_area = localities.to_crs("ESRI:102022")      # Africa Albers Equal Area
    localities["area_km2"] = (equal_area.area / 1e6).round(1)
    centroids = equal_area.centroid.to_crs("EPSG:4326")
    localities["lon"] = centroids.x.round(4)
    localities["lat"] = centroids.y.round(4)

    neighbors = find_neighbors(localities)

    localities.to_file(out, driver="GPKG")
    localities.drop(columns="geometry").to_csv(interim(cfg, "localities.csv"), index=False)
    neighbors.to_csv(interim(cfg, "neighbors.csv"), index=False)
    print(f"boundaries: {len(localities)} localities, {len(neighbors)} neighbor pairs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(load_config(), force=args.force)
