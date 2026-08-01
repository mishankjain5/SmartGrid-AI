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
kept so the cost of forecast lead time can be measured rather than assumed.

Weather is fetched at several points across Germany rather than one. National
solar output is the sum of installations spread over 800 km, and conditions
diverge sharply: on 2025-06-10 at midday the archive gives 84 W/m² near Hamburg,
483 near the centre and 854 near Munich. A single centroid sees only the middle
number and cannot distinguish an overcast north from a clear south.

Day-ahead *radiation* is only archived from 2024-01-19. Temperature at the same
lead reaches back to 2022, but radiation does not, so the usable window for solar
modelling starts in 2024.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from smartgrid.config import RAW_DATA_DIR

BASE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
CACHE_DIR = RAW_DATA_DIR / "open_meteo"

# One location over the full window takes around three minutes to generate.
# Asking for all nine in a single request makes the server do them sequentially,
# which exceeds any sane timeout, so each location is requested separately.
#
# Concurrency is low because the free tier meters by data volume rather than
# request count: six parallel multi-year requests returns 429. Three is enough to
# cut wall time substantially while staying inside the quota.
TIMEOUT_SECONDS = 600
MAX_CONCURRENT_REQUESTS = 3
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = 20

#: The archive as a whole begins here; day-ahead radiation begins later.
ARCHIVE_START = date(2022, 1, 1)
DAY_AHEAD_RADIATION_START = date(2024, 1, 19)

#: Points spanning Germany, named for the nearest city. Chosen to cover the
#: north coast, the industrial west, the east and the southern solar belt rather
#: than to sample uniformly: the model learns its own weighting across them.
LOCATIONS: dict[str, tuple[float, float]] = {
    "hamburg": (53.55, 9.99),
    "rostock": (54.09, 12.14),
    "hanover": (52.37, 9.73),
    "berlin": (52.52, 13.40),
    "cologne": (50.94, 6.96),
    "kassel": (51.31, 9.49),
    "nuremberg": (49.45, 11.08),
    "stuttgart": (48.78, 9.18),
    "munich": (48.14, 11.58),
}

#: API variable name -> column name.
VARIABLES = {
    "temperature_2m": "temperature_c",
    "shortwave_radiation": "ghi_wm2",
    "direct_radiation": "direct_radiation_wm2",
    "diffuse_radiation": "diffuse_radiation_wm2",
    "cloud_cover": "cloud_cover_pct",
}

DAY_AHEAD_SUFFIX = "_previous_day1"

DAY_AHEAD_VARIABLES = {
    f"{name}{DAY_AHEAD_SUFFIX}": f"{column}_day_ahead"
    for name, column in VARIABLES.items()
}

ALL_VARIABLES = VARIABLES | DAY_AHEAD_VARIABLES


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.json"


def _request(cache_name: str, refresh: bool = False, **params) -> dict:
    """GET one location, caching the body.

    Retries on 429. The free tier's quota refills over time, so a rejected
    request usually succeeds shortly afterwards; failing the whole ingest for it
    would throw away the locations that already downloaded.
    """
    cached = _cache_path(cache_name)
    if cached.exists() and not refresh:
        return json.loads(cached.read_text(encoding="utf-8"))

    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = requests.get(BASE_URL, params=params, timeout=TIMEOUT_SECONDS)

        if response.status_code == 429:
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(
                    f"{cache_name}: still rate limited after {MAX_ATTEMPTS} attempts. "
                    "Lower MAX_CONCURRENT_REQUESTS or narrow the date window."
                )

            delay = BACKOFF_SECONDS * 2 ** (attempt - 1)
            print(
                f"  {cache_name}: rate limited, retrying in {delay}s "
                f"(attempt {attempt}/{MAX_ATTEMPTS})",
                flush=True,
            )
            time.sleep(delay)
            continue

        response.raise_for_status()
        payload = response.json()

        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    raise AssertionError("unreachable")


def _fetch_location(
    name: str,
    latitude: float,
    longitude: float,
    start: date | str,
    end: date | str,
    refresh: bool,
) -> dict:
    return _request(
        cache_name=f"weather_{name}_{start}_{end}",
        refresh=refresh,
        latitude=latitude,
        longitude=longitude,
        start_date=str(start),
        end_date=str(end),
        hourly=",".join(ALL_VARIABLES),
        timezone="UTC",
    )


def fetch_weather(
    start: date | str,
    end: date | str,
    *,
    locations: dict[str, tuple[float, float]] | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Hourly archived weather forecasts, in long form.

    One row per timestamp and location, carrying every variable at both lead
    times.
    """
    if pd.Timestamp(start).date() < ARCHIVE_START:
        raise ValueError(f"the archive starts {ARCHIVE_START}; got start={start}")

    locations = locations or LOCATIONS
    names = list(locations)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as pool:
        payload = list(
            pool.map(
                lambda name: _fetch_location(
                    name, *locations[name], start, end, refresh
                ),
                names,
            )
        )

    if len(payload) != len(names):
        raise ValueError(
            f"requested {len(names)} locations but got {len(payload)} responses"
        )

    frames = []
    for name, location in zip(names, payload, strict=True):
        hourly = location["hourly"]
        missing = [v for v in ALL_VARIABLES if v not in hourly]
        if missing:
            raise KeyError(f"response for {name} is missing variables: {missing}")

        frames.append(
            pd.DataFrame(
                {
                    "utc_timestamp": pd.to_datetime(hourly["time"], utc=True),
                    "location": name,
                    "latitude": location["latitude"],
                    "longitude": location["longitude"],
                }
                | {
                    column: pd.to_numeric(hourly[api_name], errors="coerce")
                    for api_name, column in ALL_VARIABLES.items()
                }
            )
        )

    return pd.concat(frames, ignore_index=True).sort_values(
        ["location", "utc_timestamp"], ignore_index=True
    )
