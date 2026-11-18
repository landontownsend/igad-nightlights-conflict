# When the Lights Go Out: Night Lights and Conflict in East Africa

[![Tests](https://github.com/landontownsend/igad-nightlights-conflict/actions/workflows/tests.yml/badge.svg)](https://github.com/landontownsend/igad-nightlights-conflict/actions/workflows/tests.yml)

Does a decline in a locality's night lights help predict where a new armed conflict will break out in the following months, beyond what past violence in and around that locality already predicts?

This repository builds the dataset for that question: one row per locality per month, January 2015 to December 2025, for the seven IGAD member states in the study (Sudan, South Sudan, Ethiopia, Somalia, Kenya, Uganda, and Djibouti). It combines NASA Black Marble night lights, UCDP conflict events, CHIRPS rainfall, and WorldPop population, summarized to each second-level administrative area.

## The dataset at a glance

| Group | Columns |
|---|---|
| Identifiers | `country`, `locality_id`, `locality_name`, `year`, `month` |
| Outcome (DV) | `new_outbreak`, plus `at_risk` and `battle_deaths`, which it is built from |
| Night-light predictors | `light_change_3m`, `light_change_6m`, `sudden_drop`, `months_declining` (and `light_change_3m_strict`, a check) |
| Baseline predictors | `months_since_violence`, `nearby_deaths_3m`, `national_deaths_3m` |
| Controls | `population`, `prewar_light`, `rain_anomaly`, `clear_nights` |

Every predictor is measured from earlier months only. Full definitions are in [docs/codebook.md](docs/codebook.md).

## How it is built

```
geoBoundaries ──► localities + neighbors ─┐
NASA Black Marble (monthly tiles) ────────┤
UCDP events (GED + Candidate) ────────────┼──► one row per locality per month ──► checks ──► data/processed/
CHIRPS rainfall ──────────────────────────┤
WorldPop population ──────────────────────┘
```

| Stage | Script | Output in `data/interim/` |
|---|---|---|
| Boundaries | `pipeline/boundaries.py` | `localities.gpkg`, `localities.csv`, `neighbors.csv` |
| Night lights | `pipeline/lights.py` | `lights/lights_<year>_<month>.csv` (one file per month, 2012 to 2025) |
| Conflict | `pipeline/conflict.py` | `conflict_events.csv`, `conflict_sources.json` |
| Rainfall | `pipeline/rain.py` | `rain/rain_<year>.csv` (1991 to 2025) |
| Population | `pipeline/population.py` | `population.csv` |
| Build | `pipeline/build.py` | `data/processed/igad_locality_month.csv` (and `.parquet`, `summary.json`) |

All settings, including every variable definition, are in `config.yaml`.

### Running it

The night-light tiles are tens of gigabytes, so the two large stages run in Google Colab, not on GitHub Actions: open [notebooks/process_lights_colab.ipynb](https://colab.research.google.com/github/landontownsend/igad-nightlights-conflict/blob/data-build/notebooks/process_lights_colab.ipynb), add a free [NASA Earthdata](https://urs.earthdata.nasa.gov) token as the Colab secret `NASA_EARTHDATA_TOKEN`, and click *Run all*. It saves each month to Google Drive, so a disconnect loses nothing, and it takes about two to three hours. Its results go in `data/interim/lights/` and `data/interim/rain/`.

The other stages are small and run anywhere. The **Build dataset** workflow on GitHub Actions only builds the dataset from the saved results and runs the tests (about one minute).

To run a stage locally (from the repository root):

```bash
pip install -r requirements.txt
python -m pipeline.boundaries
BLACKMARBLE_TOKEN=<your token> python -m pipeline.lights --year 2023
python -m pipeline.conflict
python -m pipeline.rain
python -m pipeline.population
python -m pipeline.build
pytest                      # tests on made-up data, no internet needed
```

## Decisions worth knowing

- **Filled-in pixels are removed.** Black Marble's monthly product fills cloudy pixels with values from earlier years (quality flag 2). Those would hide the very declines this project looks for, so they are dropped, and a month counts only if at least half of a locality had a clear reading.
- **Only lit pixels count.** Empty land shows faint random values in the satellite readings, and in dark rural localities that noise produced changes of thousands of percent. Light is therefore the total of pixels at 0.5 nW/cm²/sr or brighter, and percent changes are left missing where a locality has no measurable lights.
- **Battle deaths exclude one-sided violence.** The outcome uses UCDP's state-based and non-state violence (types 1 and 2). One-sided violence against civilians still counts toward `months_since_violence`.
- **New outbreaks only.** A month is an outbreak if it has at least 25 battle deaths and the previous 12 months had none. Rows that follow fighting are kept but marked `at_risk = 0`, and the models use only `at_risk = 1` rows.
- **Population after 2020** reuses WorldPop's 2020 estimate, the last year of its consistent series.

## Not yet included

- Displacement (IOM Displacement Tracking Matrix). Its survey rounds differ by country and are not monthly, so it will be added once a consistent rule for carrying values forward is settled.
- Removing light from fires and gas flares (NASA FIRMS). The quality filtering above removes the largest distortions; this is a planned refinement.

## Data sources

- Román, M. O., et al. (2018). NASA's Black Marble nighttime lights product suite. *Remote Sensing of Environment*, 210, 113-143. Product VNP46A3, Collection 2, accessed with [blackmarblepy](https://github.com/worldbank/blackmarblepy).
- Sundberg, R., and Melander, E. (2013). Introducing the UCDP Georeferenced Event Dataset. *Journal of Peace Research*, 50(4), 523-532.
- Funk, C., et al. (2015). The climate hazards infrared precipitation with stations (CHIRPS). *Scientific Data*, 2, 150066.
- WorldPop. Global 1 km population estimates, 2000 to 2020.
- Runfola, D., et al. (2020). geoBoundaries: A global database of political administrative boundaries. *PLoS ONE*, 15(4), e0231866.

Each data source keeps its own license and citation requirements. The code is released under the MIT License.

## Author

Landon Townsend, M.S. Applied Political Analytics, University of Maryland.
