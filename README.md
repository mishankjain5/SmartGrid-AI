# SmartGrid-AI

Day-ahead forecasting and battery dispatch optimisation for the German
electricity market.

A household with rooftop PV, a heat pump, an EV and a battery has to decide each
hour whether to store energy, export it, or buy from the grid. Making that
decision well needs three forecasts — solar generation, consumption, and price —
and an optimiser that turns them into a schedule.

> **Status: step 0 of 9.** Project skeleton only. No data, models or results yet.

## Pipeline

```
APIs / CSV
   │  ingestion (Python)
   ▼
BigQuery: raw
   │  transformation (SQLMesh)        clean → preprocess → features
   ▼
BigQuery: staging → marts
   │  modelling (Python, scikit-learn)
   ▼
forecasts → dashboard → battery optimiser
```

SQL handles cleaning and feature engineering; Python handles ingestion and
modelling. BigQuery is the only warehouse, so there is one SQL dialect.

## Data sources

| Source | Provides |
|---|---|
| [Energy-Charts](https://api.energy-charts.info) (Fraunhofer ISE) | Day-ahead prices, generation by type, installed capacity, and the TSO's own day-ahead forecasts |
| [Open-Meteo](https://open-meteo.com/en/docs/historical-forecast-api) | Archived weather forecasts — temperature and direct/diffuse/shortwave radiation |
| [Open Power System Data](https://open-power-system-data.org) | Behind-the-meter household metering: PV, heat pump, EV |

Weather comes from archived *forecasts* rather than reanalysis, so features
reflect what was knowable when a forecast had to be issued.
`public_power_forecast` supplies the operational benchmark that models are
scored against.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -e ".[dev]"

cp .env.example .env            # then set GCP_PROJECT
```

BigQuery access requires a Google Cloud project and application default
credentials:

```bash
gcloud auth application-default login
```

## Development

```bash
pytest
ruff check .
```

## Layout

```
src/smartgrid/
  config.py        paths, market constants, environment settings
  sources/         one module per data source
  warehouse/       BigQuery load and query helpers
transform/         SQLMesh project
notebooks/         exploratory analysis
tests/
data/              local cache, gitignored
```

## Roadmap

- [x] 0 — Project skeleton
- [ ] 1 — Google Cloud project and authentication
- [ ] 2 — Ingestion from the three sources
- [ ] 3 — Load into BigQuery
- [ ] 4 — Cleaning and preprocessing in SQLMesh, with audits
- [ ] 5 — Feature engineering
- [ ] 6 — Exploratory analysis
- [ ] 7 — Forecasting: solar, load, price
- [ ] 8 — Dashboard
- [ ] 9 — Battery dispatch optimiser

## Licence

MIT for project code. Each data source carries its own terms.
