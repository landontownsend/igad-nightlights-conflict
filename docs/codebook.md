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

**How light is measured.** Empty land is not perfectly dark in the satellite readings: it shows small random values. In dark rural localities that noise is most of the total, and comparing one month of noise with another produced changes of thousands of percent. So the measure counts only lit pixels (0.5 nW/cm²/sr or brighter), adds up their light, and scales the total up to the whole locality when clouds hid part of it. The percent changes are calculated only where the usual lit total is at least 10 (about five lit pixels); in localities with no measurable lights they are missing.

"Usual level" means the same calendar month in the previous three years (at least two of them with a clear reading).

| Column | Meaning | Type | Source |
|---|---|---|---|
| `light_change_3m` | Percent change in light over the previous 3 months compared with the usual level for those months (at least 2 of the 3 months needed) | Percent | Black Marble |
| `light_change_6m` | The same over the previous 6 months (at least 4 of the 6 needed) | Percent | Black Marble |
| `sudden_drop` | 1 if light fell by more than 30 percent from one month to the next in any of the previous 6 months | 0 or 1 | Black Marble |
| `months_declining` | Consecutive months, up to last month, with light below the usual level (a month without a clear reading ends the run) | Count | Black Marble |
| `light_change_3m_strict` | `light_change_3m` counting only pixels at 2.0 nW/cm²/sr or brighter, to check that results do not depend on the threshold | Percent | Black Marble |

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
