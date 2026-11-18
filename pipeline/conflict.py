"""
Stage 3: conflict events.
Downloads the UCDP Georeferenced Event Dataset (GED), plus UCDP's monthly
Candidate Events for months the latest GED release does not cover yet, and
places every event in its locality.

The download links change with each release, so they are read from the UCDP
downloads page rather than written into the code.

Writes:
  data/interim/conflict_events.csv   one row per event in a study locality
  data/interim/conflict_sources.json which files were used and the last date covered
"""

import argparse
import io
import json
import re
import zipfile
from urllib.parse import urljoin

import geopandas as gpd
import pandas as pd
import requests

from pipeline.common import interim, load_config

COLUMNS = ["id", "type_of_violence", "country", "latitude", "longitude",
           "where_prec", "date_start", "best"]


def find_links(page_url):
    """Return every link on the UCDP downloads page, as full URLs."""
    html = requests.get(page_url, timeout=60).text
    return [urljoin(page_url, href) for href in re.findall(r'href="([^"]+)"', html)]


def read_table(url):
    """Read a UCDP CSV file, or the CSV inside a ZIP file, keeping only the columns needed."""
    content = requests.get(url, timeout=600).content
    if url.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
            content = z.read(name)
    table = pd.read_csv(io.BytesIO(content), low_memory=False)
    return table[[c for c in COLUMNS if c in table.columns]]


def latest_ged(links):
    """The newest GED release, from links such as .../ged/ged251-csv.zip (version 25.1)."""
    versions = []
    for link in links:
        m = re.search(r"ged(\d{2,3})-csv\.zip$", link)
        if m:
            versions.append((int(m.group(1)), link))
    if not versions:
        raise RuntimeError("Could not find a GED download link on the UCDP downloads page")
    return max(versions)[1]


def candidate_links(links):
    """Links to UCDP Candidate Events files (CSV or ZIP)."""
    return sorted({l for l in links if "candidate" in l.lower()
                   and l.lower().endswith((".csv", ".zip"))})


def run(cfg, force=False):
    out = interim(cfg, "conflict_events.csv")
    if out.exists() and not force:
        print("conflict: already built, skipping (use --force to rebuild)")
        return

    links = find_links(cfg["sources"]["ucdp"]["downloads_page"])
    ged_url = latest_ged(links)
    print(f"   GED: {ged_url}")
    events = read_table(ged_url)
    events["source"] = "GED"
    events["date_start"] = pd.to_datetime(events["date_start"])
    ged_end = events["date_start"].max()
    used = {"ged": ged_url, "ged_last_date": str(ged_end.date()), "candidate": []}

    # Add Candidate Events for months after the GED release ends
    study_end = pd.Timestamp(f"{cfg['years']['end']}-12-31")
    if ged_end < study_end:
        for url in candidate_links(links):
            try:
                extra = read_table(url)
            except Exception as e:                    # one bad file should not stop the run
                print(f"   skipped {url}: {e}")
                continue
            extra["date_start"] = pd.to_datetime(extra["date_start"], errors="coerce")
            extra = extra[extra["date_start"] > ged_end]
            if len(extra):
                extra["source"] = "Candidate"
                events = pd.concat([events, extra], ignore_index=True)
                used["candidate"].append(url)
        events = events.drop_duplicates(subset="id", keep="first")
    used["last_date"] = str(events["date_start"].max().date())
    print(f"   events cover up to {used['last_date']}")

    # Keep events in the study countries that are located precisely enough
    names = {n for c in cfg["countries"] for n in c["ucdp_names"]}
    max_prec = cfg["sources"]["ucdp"]["max_location_precision"]
    events = events[events["country"].isin(names) & (events["where_prec"] <= max_prec)]

    # Place each event in its locality
    localities = gpd.read_file(interim(cfg, "localities.gpkg"))[["locality_id", "country", "geometry"]]
    points = gpd.GeoDataFrame(events, geometry=gpd.points_from_xy(events.longitude, events.latitude),
                              crs="EPSG:4326")
    placed = gpd.sjoin(points, localities.rename(columns={"country": "locality_country"}),
                       predicate="within", how="inner")
    placed = placed.drop_duplicates(subset="id")      # an event on a shared border counts once
    print(f"   {len(placed)} of {len(points)} events placed in a locality")

    placed["year"] = placed["date_start"].dt.year
    placed["month"] = placed["date_start"].dt.month
    keep = ["id", "source", "date_start", "year", "month", "type_of_violence", "best",
            "where_prec", "locality_id", "locality_country"]
    placed[keep].sort_values(["date_start", "id"]).to_csv(out, index=False)
    used["events_placed"] = int(len(placed))
    with open(interim(cfg, "conflict_sources.json"), "w") as f:
        json.dump(used, f, indent=2)
    print(f"conflict: {len(placed)} events saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(load_config(), force=args.force)
