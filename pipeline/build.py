"""
Stage 6: build the dataset.
Combines every interim file into one row per locality per month, from January
2015 to December 2025, with the outcome measured in that month and every
predictor measured from earlier months only.

Most calculations below work on tables with one row per locality and one
column per month (a "grid"), because "the previous three months" is then just
the three columns to the left.

Writes:
  data/processed/igad_locality_month.csv      the dataset
  data/processed/igad_locality_month.parquet  the same, in a compact format
  data/processed/summary.json                 counts and missing-value shares
"""

import argparse
import glob
import json
import warnings

import numpy as np
import pandas as pd
from scipy import sparse

from pipeline.common import interim, load_config, processed


# -----------------------------------------------------------------------------
# Helpers for grids (rows = localities, columns = months)
# -----------------------------------------------------------------------------

def to_grid(long, value, ids, months):
    """Turn a long table (locality_id, month_index, value) into a grid; gaps are missing (NaN)."""
    wide = long.pivot_table(index="locality_id", columns="month_index", values=value, aggfunc="sum")
    return wide.reindex(index=ids, columns=months).to_numpy(dtype="float64")


def window_sum(grid, k):
    """
    For each month t, the sum over the k previous months (t-k to t-1), treating
    missing as zero, plus how many of those months were present.
    Months with fewer than k earlier months get NaN.
    """
    present = ~np.isnan(grid)
    values = np.where(present, grid, 0.0)
    zeros = np.zeros((grid.shape[0], 1))
    cum_values = np.hstack([zeros, np.cumsum(values, axis=1)])
    cum_present = np.hstack([zeros, np.cumsum(present, axis=1)])
    total = np.full(grid.shape, np.nan)
    count = np.full(grid.shape, np.nan)
    total[:, k:] = cum_values[:, k:-1] - cum_values[:, :-k - 1]
    count[:, k:] = cum_present[:, k:-1] - cum_present[:, :-k - 1]
    return total, count


def lag(grid, k=1):
    """The value from k months earlier (NaN for the first k months)."""
    out = np.full(grid.shape, np.nan)
    out[:, k:] = grid[:, :-k]
    return out


def month_index(year, month):
    """Months counted from year 0, so that consecutive months differ by 1."""
    return np.asarray(year) * 12 + np.asarray(month) - 1


# -----------------------------------------------------------------------------
# Night lights
# -----------------------------------------------------------------------------

def light_features(cfg, ids, months):
    d = cfg["definitions"]
    files = sorted(glob.glob(str(interim(cfg, "lights")) + "/lights_*.csv"))
    lights = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    lights["month_index"] = month_index(lights["year"], lights["month"])

    # Average radiance of the clear pixels; missing if under half the locality was clear
    share = lights["valid_px"] / lights["land_px"]
    ok = (lights["valid_px"] > 0) & (share >= cfg["sources"]["black_marble"]["min_valid_share"])
    lights["light"] = np.where(ok, lights["ntl_sum"] / lights["valid_px"], np.nan)
    lights["clear_nights"] = np.where(lights["valid_px"] > 0,
                                      lights["num_sum"] / lights["valid_px"], np.nan)
    light = to_grid(lights, "light", ids, months)
    clear = to_grid(lights, "clear_nights", ids, months)

    # "Usual level": the same calendar month in the previous 3 years (at least 2 of them)
    past = np.stack([lag(light, 12 * y) for y in range(1, d["usual_level_years"] + 1)])
    n_past = np.sum(~np.isnan(past), axis=0)
    with np.errstate(invalid="ignore"):
        usual = np.where(n_past >= 2, np.nanmean(past, axis=0), np.nan)

    # Percent change over the previous 3 and 6 months, compared with the usual level
    both = ~np.isnan(light) & ~np.isnan(usual)
    features = {}
    for k, min_months in [(3, 2), (6, 4)]:
        actual, n = window_sum(np.where(both, light, np.nan), k)
        expected, _ = window_sum(np.where(both, usual, np.nan), k)
        with np.errstate(divide="ignore", invalid="ignore"):
            change = (actual / expected - 1) * 100
        valid = (n >= min_months) & (expected > d["light_floor"] * k)
        features[f"light_change_{k}m"] = np.where(valid, change, np.nan)

    # Sudden drop: light fell by more than 30 percent from one month to the next,
    # in any of the previous 6 months
    previous = lag(light)
    with np.errstate(invalid="ignore"):
        dropped = (light < (1 - d["sudden_drop_share"]) * previous) & (previous > d["light_floor"])
    drops, _ = window_sum(np.where(dropped, 1.0, 0.0), 6)
    features["sudden_drop"] = np.where(np.isnan(drops), np.nan, (drops > 0).astype(float))

    # Consecutive months below the usual level, up to last month (a missing month ends the run)
    with np.errstate(invalid="ignore"):
        below = (light < usual) & both
    streak = np.zeros(light.shape)
    for t in range(1, light.shape[1]):
        streak[:, t] = (streak[:, t - 1] + 1) * below[:, t]
    features["months_declining"] = lag(streak)

    features["clear_nights"] = lag(clear)

    # Average light before the study period, which separates cities from rural areas
    first, last = cfg["years"]["prewar_light"]
    cols = [i for i, m in enumerate(months) if first <= m // 12 <= last]
    with np.errstate(invalid="ignore"):
        features["prewar_light"] = np.repeat(np.nanmean(light[:, cols], axis=1, keepdims=True),
                                             len(months), axis=1)
    return features


# -----------------------------------------------------------------------------
# Conflict
# -----------------------------------------------------------------------------

def conflict_features(cfg, localities, neighbors, months):
    d = cfg["definitions"]
    ids = localities["locality_id"].tolist()
    events = pd.read_csv(interim(cfg, "conflict_events.csv"))
    events["month_index"] = month_index(events["year"], events["month"])

    # Battle deaths: state-based and non-state violence only (one-sided violence
    # against civilians is not a battle)
    battles = events[events["type_of_violence"].isin(d["battle_violence_types"])]
    deaths = to_grid(battles, "best", ids, months)
    deaths = np.nan_to_num(deaths, nan=0.0)          # no recorded event means no deaths

    features = {"battle_deaths": deaths}
    prior12, _ = window_sum(deaths, d["peaceful_months"])
    at_risk = prior12 == 0
    features["at_risk"] = np.where(np.isnan(prior12), np.nan, at_risk.astype(float))
    features["new_outbreak"] = np.where(np.isnan(prior12), np.nan,
                                        (at_risk & (deaths >= d["outbreak_deaths"])).astype(float))

    # Months since the last violent event of any type, using the full UCDP history
    cap = d["months_since_cap"]
    since = np.full((len(ids), len(months)), float(cap))
    by_locality = events.groupby("locality_id")["month_index"].apply(lambda s: np.sort(s.unique()))
    month_array = np.asarray(months)
    for row, lid in enumerate(ids):
        if lid in by_locality.index:
            history = by_locality[lid]
            pos = np.searchsorted(history, month_array, side="left") - 1   # last event before t
            has = pos >= 0
            gap = np.where(has, month_array - history[np.clip(pos, 0, None)], cap)
            since[row] = np.minimum(gap, cap)
    features["months_since_violence"] = since

    # Battle deaths over the previous 3 months: in neighboring localities, and
    # elsewhere in the same country
    own3, _ = window_sum(deaths, 3)
    position = {lid: i for i, lid in enumerate(ids)}
    pairs = neighbors[neighbors["locality_id"].isin(position) & neighbors["neighbor_id"].isin(position)]
    adjacency = sparse.csr_matrix((np.ones(len(pairs)),
                                   (pairs["locality_id"].map(position), pairs["neighbor_id"].map(position))),
                                  shape=(len(ids), len(ids)))
    features["nearby_deaths_3m"] = adjacency @ np.nan_to_num(own3) + np.where(np.isnan(own3), np.nan, 0)

    national = np.zeros(own3.shape)
    for iso3, group in localities.groupby("country"):
        rows = group.index.to_numpy()
        national[rows] = np.nansum(own3[rows], axis=0) - own3[rows]
    features["national_deaths_3m"] = np.where(np.isnan(own3), np.nan, national)
    return features


# -----------------------------------------------------------------------------
# Rainfall and population
# -----------------------------------------------------------------------------

def rain_anomaly(cfg, ids, months):
    """Rainfall over the previous 3 months, as a percent above or below normal for those months."""
    files = sorted(glob.glob(str(interim(cfg, "rain")) + "/rain_*.csv"))
    rain = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    rain["month_index"] = month_index(rain["year"], rain["month"])
    first = int(rain["month_index"].min())
    all_months = list(range(first, max(months) + 1))
    grid = to_grid(rain, "rain_mm", ids, all_months)

    # Normal rainfall for each calendar month, averaged over the normal years
    lo, hi = cfg["sources"]["chirps"]["normal_years"]
    calendar = np.array([m % 12 for m in all_months])
    years = np.array([m // 12 for m in all_months])
    in_normal = (years >= lo) & (years <= hi)
    normal = np.full(grid.shape, np.nan)
    for cm in range(12):
        cols = calendar == cm
        with np.errstate(invalid="ignore"):
            normal[:, cols] = np.nanmean(grid[:, cols & in_normal], axis=1, keepdims=True)

    actual, n = window_sum(grid, 3)
    expected, _ = window_sum(normal, 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        anomaly = np.where((n == 3) & (expected >= 1.0), (actual / expected - 1) * 100, np.nan)
    keep = [all_months.index(m) for m in months]
    return anomaly[:, keep]


def population_grid(cfg, ids, months):
    pop = pd.read_csv(interim(cfg, "population.csv"))
    last = cfg["sources"]["worldpop"]["last_year"]
    wide = pop.pivot_table(index="locality_id", columns="year", values="population").reindex(ids)
    years = [min(m // 12, last) for m in months]
    return np.column_stack([wide[y].to_numpy() if y in wide.columns else np.full(len(ids), np.nan)
                            for y in years])


# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------

def check(df, n_localities, n_months):
    problems = []
    if len(df) != n_localities * n_months:
        problems.append(f"expected {n_localities * n_months} rows, found {len(df)}")
    if df.duplicated(["locality_id", "year", "month"]).any():
        problems.append("duplicate locality-months")
    if ((df["new_outbreak"] == 1) & (df["at_risk"] != 1)).any():
        problems.append("an outbreak in a locality that had fighting in the previous 12 months")
    if (df["battle_deaths"] < 0).any():
        problems.append("negative battle deaths")
    if not df["months_since_violence"].between(1, 60).all():
        problems.append("months_since_violence outside 1 to 60")
    if (df["prewar_light"] < 0).any():
        problems.append("negative light")
    if problems:
        raise ValueError("Dataset checks failed: " + "; ".join(problems))


# -----------------------------------------------------------------------------
# Put it together
# -----------------------------------------------------------------------------

COLUMN_ORDER = [
    # identifiers
    "country", "locality_id", "locality_name", "year", "month",
    # outcome, and the two columns it is built from
    "new_outbreak", "at_risk", "battle_deaths",
    # night-light predictors
    "light_change_3m", "light_change_6m", "sudden_drop", "months_declining",
    # baseline predictors
    "months_since_violence", "nearby_deaths_3m", "national_deaths_3m",
    # controls
    "population", "prewar_light", "rain_anomaly", "clear_nights",
]


def run(cfg):
    localities = pd.read_csv(interim(cfg, "localities.csv"))
    neighbors = pd.read_csv(interim(cfg, "neighbors.csv"))
    ids = localities["locality_id"].tolist()

    # Work on every month from the first year of night lights, then keep the study period
    months = list(range(month_index(cfg["years"]["lights_start"], 1),
                        month_index(cfg["years"]["end"], 12) + 1))
    grids = {}
    grids.update(light_features(cfg, ids, months))
    grids.update(conflict_features(cfg, localities, neighbors, months))
    grids["rain_anomaly"] = rain_anomaly(cfg, ids, months)
    grids["population"] = population_grid(cfg, ids, months)

    study = [i for i, m in enumerate(months) if m >= month_index(cfg["years"]["start"], 1)]
    study_months = [months[i] for i in study]
    df = pd.DataFrame({
        "locality_id": np.repeat(ids, len(study)),
        "year": np.tile([m // 12 for m in study_months], len(ids)),
        "month": np.tile([m % 12 + 1 for m in study_months], len(ids)),
    })
    for name, grid in grids.items():
        df[name] = grid[:, study].reshape(-1)
    df = df.merge(localities[["locality_id", "locality_name", "country"]], on="locality_id")

    integer_cols = ["new_outbreak", "at_risk", "sudden_drop", "months_declining",
                    "months_since_violence", "battle_deaths", "nearby_deaths_3m",
                    "national_deaths_3m", "population"]
    for col in integer_cols:
        df[col] = df[col].round().astype("Int64")
    for col in ["light_change_3m", "light_change_6m", "rain_anomaly"]:
        df[col] = df[col].round(1)
    for col in ["prewar_light", "clear_nights"]:
        df[col] = df[col].round(3)
    df = df[COLUMN_ORDER].sort_values(["country", "locality_id", "year", "month"]).reset_index(drop=True)

    check(df, len(ids), len(study))
    df.to_csv(processed(cfg, "igad_locality_month.csv"), index=False)
    df.to_parquet(processed(cfg, "igad_locality_month.parquet"), index=False)

    at_risk = df[df["at_risk"] == 1]
    summary = {
        "rows": int(len(df)),
        "localities": len(ids),
        "localities_by_country": localities["country"].value_counts().to_dict(),
        "months": len(study),
        "new_outbreaks": int(df["new_outbreak"].sum()),
        "new_outbreaks_by_country": df.groupby("country")["new_outbreak"].sum().astype(int).to_dict(),
        "new_outbreaks_by_year": df.groupby("year")["new_outbreak"].sum().astype(int).to_dict(),
        "at_risk_rows": int(len(at_risk)),
        "outbreak_rate_among_at_risk_rows": round(float(at_risk["new_outbreak"].mean()), 5),
        "missing_share": {c: round(float(df[c].isna().mean()), 3) for c in COLUMN_ORDER if df[c].isna().any()},
    }
    with open(processed(cfg, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=int)
    print(json.dumps(summary, indent=2, default=int))


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    warnings.filterwarnings("ignore", message="Mean of empty slice")   # all-missing rows are expected
    run(load_config())
