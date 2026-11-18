"""
Tests that run without the internet: each stage's logic on small, made-up data.
Run from the repository root:  pytest
"""

import copy

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from pipeline import build
from pipeline.boundaries import find_neighbors
from pipeline.common import load_config
from pipeline.conflict import candidate_links, latest_ged
from pipeline.lights import DATA_FIELDS, process_tile, tile_bounds


# -----------------------------------------------------------------------------
# Lights
# -----------------------------------------------------------------------------

def test_tile_bounds():
    # Tile h21v07 covers 30 to 40 E and 10 to 20 N
    assert tile_bounds(21, 7) == (30, 10, 40, 20)


def test_process_tile_drops_filled_pixels(tmp_path):
    path = tmp_path / "VNP46A3.A2023001.h21v07.002.h5"
    n = 2400
    quality = np.zeros((n, n), dtype="uint8")
    quality[:, :1200] = 2                      # west half filled in from past years
    with h5py.File(path, "w") as f:
        f.attrs["HorizontalTileNumber"] = np.bytes_(b"21")
        f.attrs["VerticalTileNumber"] = np.bytes_(b"07")
        g = f.create_group(DATA_FIELDS)
        rad = g.create_dataset("NearNadir_Composite_Snow_Free", data=np.full((n, n), 20, dtype="uint16"))
        rad.attrs["scale_factor"] = 0.1        # stored 20 means 2.0 nW/cm2/sr
        g.create_dataset("NearNadir_Composite_Snow_Free_Quality", data=quality)
        g.create_dataset("NearNadir_Composite_Snow_Free_Num", data=np.full((n, n), 4, dtype="uint16"))
    localities = gpd.GeoDataFrame({"locality_id": ["east", "west"]},
                                  geometry=[box(36, 12, 37, 13), box(31, 12, 32, 13)], crs="EPSG:4326")
    out = process_tile(path, localities, "NearNadir_Composite_Snow_Free", [2, 255]).set_index("locality_id")
    pixels = 240 * 240                         # one degree is 240 pixels across
    assert out.loc["east", "valid_px"] == pytest.approx(pixels)
    assert out.loc["east", "ntl_sum"] == pytest.approx(2.0 * pixels)
    assert out.loc["west", "valid_px"] == 0
    assert out.loc["west", "land_px"] == pytest.approx(pixels)


# -----------------------------------------------------------------------------
# Boundaries and conflict links
# -----------------------------------------------------------------------------

def test_neighbors_are_two_way_and_exclude_distant_areas():
    localities = gpd.GeoDataFrame({"locality_id": ["a", "b", "c"]},
                                  geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(5, 5, 6, 6)],
                                  crs="EPSG:4326")
    pairs = set(map(tuple, find_neighbors(localities).to_numpy()))
    assert pairs == {("a", "b"), ("b", "a")}


def test_ucdp_links_pick_newest_ged_and_candidates():
    links = ["https://ucdp.uu.se/downloads/ged/ged241-csv.zip",
             "https://ucdp.uu.se/downloads/ged/ged251-csv.zip",
             "https://ucdp.uu.se/downloads/candidateged/GEDEvent_v25_0_9.csv",
             "https://ucdp.uu.se/downloads/other.pdf"]
    assert latest_ged(links).endswith("ged251-csv.zip")
    assert candidate_links(links) == ["https://ucdp.uu.se/downloads/candidateged/GEDEvent_v25_0_9.csv"]


# -----------------------------------------------------------------------------
# Build: a tiny made-up world with three localities
# -----------------------------------------------------------------------------

@pytest.fixture()
def tiny(tmp_path):
    """
    Localities A and B (country X) are neighbors; C (country Y) is on its own.
    - A's lights are steady at 10, then fall to 5 from June 2023 on.
    - B has a battle with 30 deaths in December 2023 (a new outbreak) and 40 in January 2024.
    - C has one small one-sided attack in March 2020.
    """
    cfg = copy.deepcopy(load_config())
    cfg["paths"]["interim"] = str(tmp_path / "interim")
    cfg["paths"]["processed"] = str(tmp_path / "processed")
    (tmp_path / "interim" / "lights").mkdir(parents=True)

    pd.DataFrame({"locality_id": ["A", "B", "C"], "locality_name": ["A", "B", "C"],
                  "country": ["X", "X", "Y"]}).to_csv(tmp_path / "interim/localities.csv", index=False)
    pd.DataFrame({"locality_id": ["A", "B"], "neighbor_id": ["B", "A"]}).to_csv(
        tmp_path / "interim/neighbors.csv", index=False)

    for year in range(2012, 2026):
        rows = []
        for lid in ["A", "B", "C"]:
            for month in range(1, 13):
                light = 5.0 if (lid == "A" and (year, month) >= (2023, 6)) else 10.0
                rows.append({"locality_id": lid, "year": year, "month": month,
                             "ntl_sum": light * 100, "valid_px": 100, "land_px": 100, "num_sum": 600})
        pd.DataFrame(rows).to_csv(tmp_path / f"interim/lights/lights_{year}.csv", index=False)

    pd.DataFrame([
        {"id": 1, "year": 2023, "month": 12, "type_of_violence": 1, "best": 30, "locality_id": "B"},
        {"id": 2, "year": 2024, "month": 1, "type_of_violence": 1, "best": 40, "locality_id": "B"},
        {"id": 3, "year": 2020, "month": 3, "type_of_violence": 3, "best": 2, "locality_id": "C"},
    ]).to_csv(tmp_path / "interim/conflict_events.csv", index=False)

    rain = [{"locality_id": lid, "year": y, "month": m, "rain_mm": 50.0}
            for lid in "ABC" for y in range(1991, 2026) for m in range(1, 13)]
    pd.DataFrame(rain).to_csv(tmp_path / "interim/rain_monthly.csv", index=False)
    pop = [{"locality_id": lid, "year": y, "population": 1000} for lid in "ABC" for y in range(2015, 2021)]
    pd.DataFrame(pop).to_csv(tmp_path / "interim/population.csv", index=False)

    build.run(cfg)
    return pd.read_csv(tmp_path / "processed/igad_locality_month.csv")


def row(df, lid, year, month):
    return df[(df.locality_id == lid) & (df.year == year) & (df.month == month)].iloc[0]


def test_shape(tiny):
    assert len(tiny) == 3 * 132                       # 3 localities x 132 months


def test_outbreak_only_after_peace(tiny):
    assert row(tiny, "B", 2023, 12).new_outbreak == 1
    assert row(tiny, "B", 2024, 1).new_outbreak == 0  # fighting continued, not new
    assert row(tiny, "B", 2024, 1).at_risk == 0
    assert tiny.new_outbreak.sum() == 1


def test_lights_change_uses_previous_months_only(tiny):
    # A's lights halve from June 2023. In June the previous 3 months were normal.
    assert row(tiny, "A", 2023, 6).light_change_3m == 0
    # In September 2023, June to August were all at half the usual level
    assert row(tiny, "A", 2023, 9).light_change_3m == -50
    assert row(tiny, "A", 2023, 7).sudden_drop == 1   # the June fall is in the past 6 months
    assert row(tiny, "A", 2023, 9).months_declining == 3


def test_nearby_and_national_deaths(tiny):
    # In February 2024, A's neighbor B had 70 battle deaths over the previous 3 months
    assert row(tiny, "A", 2024, 2).nearby_deaths_3m == 70
    assert row(tiny, "A", 2024, 2).national_deaths_3m == 70
    assert row(tiny, "C", 2024, 2).nearby_deaths_3m == 0


def test_months_since_violence_counts_all_types_and_caps(tiny):
    assert row(tiny, "C", 2020, 4).months_since_violence == 1   # one-sided attack in March 2020
    assert row(tiny, "C", 2020, 3).months_since_violence == 60  # nothing before it: capped
    assert row(tiny, "B", 2024, 1).months_since_violence == 1


def test_controls(tiny):
    assert row(tiny, "A", 2024, 5).rain_anomaly == 0             # rain equals the normal
    assert row(tiny, "A", 2024, 5).population == 1000            # 2020 figure reused
    assert row(tiny, "A", 2016, 1).prewar_light == 10
    assert row(tiny, "A", 2016, 1).clear_nights == 6
