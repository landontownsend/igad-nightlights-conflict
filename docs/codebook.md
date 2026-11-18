# Codebook: `data/processed/igad_locality_month.csv`

**One row = one locality in one month**, January 2015 to December 2025 (132 months).

**Timing rule:** the outcome is measured in the row's own month; every predictor uses earlier months only.

**At-risk rule:** only rows with `at_risk = 1` (no battle deaths in the previous 12 months) can have a new outbreak, so the models train and test on those rows only.

## Identifiers

| Column | Meaning | Type |
|---|---|---|
| `country` | ISO3 code of the country | Text (SDN, SSD, ETH, SOM, KEN, UGA, DJI) |
| `locality_id` | Permanent geoBoundaries ID of the second-level area | Text |
| `locality_name` | Name of the area | Text |
| `year`, `month` | Calendar month of the row | Integers |

## Outcome

| Column | Meaning | Type | Source |
|---|---|---|---|
| `new_outbreak` (DV) | 1 if this month has at least 25 battle deaths and the previous 12 months had none; otherwise 0 | 0 or 1 | UCDP |
| `at_risk` | 1 if the previous 12 months had no battle deaths | 0 or 1 | UCDP |
| `battle_deaths` | Battle deaths this month (UCDP best estimate, state-based and non-state violence). Same month: never a predictor | Count | UCDP |

## Night-light predictors (main IVs)

**How light is measured.** Light is the total radiance of a locality's lit pixels, scaled up to the whole locality when clouds hid part of it. In rural East Africa this total jumps from month to month, because fires, lightning and thin cloud can light up a dark area for a single month. A plain percent change turned those jumps into changes of thousands of percent. So the change measures compare **medians**, which one unusual month cannot move, and use a **log ratio** with a constant of 10 added to both sides, so that small changes in very dark localities stay close to zero. How to read a log ratio: 0 means light is at its usual level, -0.69 means half the usual level, +0.69 means double, and -0.1 means about 10 percent lower.

"Usual level" means the same calendar month in the previous three years (at least two of them with a clear reading).

| Column | Meaning | Type | Source |
|---|---|---|---|
| `light_change_3m` | Light in the previous 3 months compared with the same 3 months in the previous 3 years: log of (median recent light + 10) / (median past light + 10). At least 2 recent and 6 past months needed | Log ratio | Black Marble |
| `light_change_6m` | The same over the previous 6 months (at least 4 recent and 12 past months needed) | Log ratio | Black Marble |
| `sudden_drop` | 1 if light in the previous 3 months was typically less than half its usual level (`light_change_3m` below -0.69) | 0 or 1 | Black Marble |
| `months_declining` | Consecutive months, up to last month, with light below its usual level for that calendar month (the median of the previous 3 years; a month without a clear reading ends the run). Counted only where the usual level is at least 10 | Count | Black Marble |

## Baseline predictors

| Column | Meaning | Type | Source |
|---|---|---|---|
| `months_since_violence` | Months since the last violent event of any type (including one-sided violence), using UCDP's full history; capped at 60 | Count, 1 to 60 | UCDP |
| `nearby_deaths_3m` | Battle deaths in neighboring localities (shared border) over the previous 3 months | Count | UCDP |
| `national_deaths_3m` | Battle deaths elsewhere in the same country over the previous 3 months | Count | UCDP |

## Controls

| Column | Meaning | Type | Source |
|---|---|---|---|
| `population` | Estimated population (2021 onward reuses 2020) | People | WorldPop |
| `prewar_light` | Average monthly lit total in 2012 to 2014, which separates cities from rural areas | Sum of nW/cm²/sr over lit pixels | Black Marble |
| `rain_anomaly` | Rainfall over the previous 3 months compared with the 1991 to 2020 normal for those months | Percent | CHIRPS |
| `clear_nights` | Average number of nightly observations behind last month's clear pixels | Count | Black Marble |

## Missing values

A value is missing when the data needed to calculate it is missing: for example, a light measure when too few months had a clear reading, or `rain_anomaly` in desert localities where normal rainfall is under 1 mm. `data/processed/summary.json` lists the share of missing values in each column.
