"""Fetch every source and load it into the BigQuery `raw` dataset.

    python -m smartgrid.ingest              # all tables
    python -m smartgrid.ingest price weather
    python -m smartgrid.ingest --refresh    # bypass the local response cache

Source responses are cached under data/raw, so re-running is cheap and only the
BigQuery load repeats.
"""

import argparse
from collections.abc import Callable
from datetime import date

import pandas as pd

from smartgrid.config import DATA_START, data_end
from smartgrid.sources import energy_charts, open_meteo, opsd
from smartgrid.warehouse.load import load_dataframe

#: Day-ahead forecasts published by the transmission operators. Used as the
#: benchmark that this project's own forecasts are scored against.
FORECAST_TYPES = ("solar", "wind_onshore", "wind_offshore", "load")


def build_price(start: date, end: date, refresh: bool) -> pd.DataFrame:
    return energy_charts.fetch_price(start, end, refresh=refresh)


def build_public_power(start: date, end: date, refresh: bool) -> pd.DataFrame:
    return energy_charts.fetch_public_power(start, end, refresh=refresh)


def build_day_ahead_forecast(start: date, end: date, refresh: bool) -> pd.DataFrame:
    frames = [
        energy_charts.fetch_day_ahead_forecast(
            production_type, start, end, refresh=refresh
        )
        for production_type in FORECAST_TYPES
    ]
    return pd.concat(frames, ignore_index=True)


def build_installed_power(start: date, end: date, refresh: bool) -> pd.DataFrame:
    return energy_charts.fetch_installed_power(refresh=refresh)


def build_weather(start: date, end: date, refresh: bool) -> pd.DataFrame:
    return open_meteo.fetch_weather(start, end, refresh=refresh)


def build_household(start: date, end: date, refresh: bool) -> pd.DataFrame:
    # OPSD covers 2014-2019 and is independent of the grid data window.
    return opsd.load(refresh=refresh)


#: table name -> builder
TABLES: dict[str, Callable[[date, date, bool], pd.DataFrame]] = {
    "price": build_price,
    "public_power": build_public_power,
    "day_ahead_forecast": build_day_ahead_forecast,
    "installed_power": build_installed_power,
    "weather": build_weather,
    "household": build_household,
}


def ingest(
    tables: list[str] | None = None,
    *,
    start: date | None = None,
    end: date | None = None,
    refresh: bool = False,
) -> dict[str, int]:
    """Build and load the named tables. Returns rows loaded per table."""
    selected = tables or list(TABLES)
    unknown = sorted(set(selected) - set(TABLES))
    if unknown:
        raise ValueError(f"unknown tables: {unknown}. Available: {sorted(TABLES)}")

    start = start or DATA_START
    end = end or data_end()

    loaded = {}
    for name in selected:
        print(f"{name}: fetching...", flush=True)
        frame = TABLES[name](start, end, refresh)

        print(f"{name}: loading {len(frame):,} rows", flush=True)
        loaded[name] = load_dataframe(frame, name)
        print(f"{name}: {loaded[name]:,} rows in BigQuery", flush=True)

    return loaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tables", nargs="*", choices=[*TABLES, []], default=[])
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument(
        "--refresh", action="store_true", help="re-fetch instead of using the cache"
    )
    args = parser.parse_args(argv)

    loaded = ingest(
        args.tables or None, start=args.start, end=args.end, refresh=args.refresh
    )

    print("\nloaded:")
    for name, rows in loaded.items():
        print(f"  {name:<20} {rows:>10,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
