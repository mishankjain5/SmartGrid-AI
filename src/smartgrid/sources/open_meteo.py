"""Open-Meteo historical forecast client.

https://open-meteo.com/en/docs/historical-forecast-api — public, no API key.

This uses the *historical forecast* archive, which replays what the weather
models actually predicted at the time. It is deliberately not the ERA5 archive
(`archive-api.open-meteo.com`), which is reanalysis: a reconstruction assembled
after the fact and therefore not available to a forecaster at the gate.

Two lead times are fetched for every variable:

* the plain name — the archive's default, stitched from the earliest hours of
  successive model runs, so a very short lead;
* the `_previous_day1` suffix — the forecast issued a day before the valid time,
  which is the lead a day-ahead gate actually has.

The day-ahead columns are the legitimate features. The short-lead columns are
kept so the cost of forecast lead time can be measured rather than assumed: for
midday on 2024-06-11 the short-lead archive says 793 W/m² where the day-ahead
forecast said 659.

Coverage begins 2022-01-01. A multi-year range returns in a single hourly
response, so no pagination is needed.
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from smartgrid.config import RAW_DATA_DIR

BASE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
CACHE_DIR = RAW_DATA_DIR / "open_meteo"
TIMEOUT_SECONDS = 300

ARCHIVE_START = date(2022, 1, 1)

#: Area-weighted centroid of Germany. The grid series this is joined against are
#: national aggregates, so a single representative point matches their scale.
GERMANY_LATITUDE = 51.16
GERMANY_LONGITUDE = 10.45

#: API variable name -> column name in the returned frame.
VARIABLES = {
    "temperature_2m": "temperature_c",
    "shortwave_radiation": "ghi_wm2",
    "direct_radiation": "direct_radiation_wm2",
    "diffuse_radiation": "diffuse_radiation_wm2",
    "cloud_cover": "cloud_cover_pct",
}

#: Suffix requesting the forecast issued a day before the valid time.
DAY_AHEAD_SUFFIX = "_previous_day1"

#: Same variables at day-ahead lead. `ghi_wm2` becomes `ghi_wm2_day_ahead`.
DAY_AHEAD_VARIABLES = {
    f"{name}{DAY_AHEAD_SUFFIX}": f"{column}_day_ahead"
    for name, column in VARIABLES.items()
}

ALL_VARIABLES = VARIABLES | DAY_AHEAD_VARIABLES


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.json"


def _request(cache_name: str, refresh: bool = False, **params) -> dict:
    cached = _cache_path(cache_name)
    if cached.exists() and not refresh:
        return json.loads(cached.read_text(encoding="utf-8"))

    response = requests.get(BASE_URL, params=params, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()

    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def fetch_weather(
    start: date | str,
    end: date | str,
    *,
    latitude: float = GERMANY_LATITUDE,
    longitude: float = GERMANY_LONGITUDE,
    refresh: bool = False,
) -> pd.DataFrame:
    """Hourly archived weather forecasts for a point.

    Returns temperature, global/direct/diffuse irradiance and cloud cover at two
    lead times: the archive default (short lead) and `*_day_ahead` (issued a day
    before the valid time).
    """
    if pd.Timestamp(start).date() < ARCHIVE_START:
        raise ValueError(
            f"the forecast archive starts {ARCHIVE_START}; got start={start}"
        )

    payload = _request(
        cache_name=f"weather_{latitude}_{longitude}_{start}_{end}",
        refresh=refresh,
        latitude=latitude,
        longitude=longitude,
        start_date=str(start),
        end_date=str(end),
        hourly=",".join(ALL_VARIABLES),
        timezone="UTC",
    )

    hourly = payload["hourly"]
    missing = [name for name in ALL_VARIABLES if name not in hourly]
    if missing:
        raise KeyError(f"response is missing variables: {missing}")

    frame = pd.DataFrame(
        {"utc_timestamp": pd.to_datetime(hourly["time"], utc=True)}
        | {
            column: pd.to_numeric(hourly[name], errors="coerce")
            for name, column in ALL_VARIABLES.items()
        }
    )

    return frame.sort_values("utc_timestamp", ignore_index=True)
