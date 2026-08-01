"""Energy-Charts API client (Fraunhofer ISE).

https://api.energy-charts.info — public, no API key.

Data is CC BY 4.0, sourced from Bundesnetzagentur | SMARD.de.

The API returns an entire multi-year range in a single response, so no
pagination is needed. Responses are cached to disk because they are large and
unchanging for historical windows.

Series resolution is not constant: German day-ahead products moved from hourly
to quarter-hourly on 2025-10-01, so a range spanning that date returns mixed
step sizes. Values are returned here at native resolution; resampling belongs in
the transformation layer.
"""

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from smartgrid.config import RAW_DATA_DIR

BASE_URL = "https://api.energy-charts.info"
CACHE_DIR = RAW_DATA_DIR / "energy_charts"
TIMEOUT_SECONDS = 300

BIDDING_ZONE = "DE-LU"
COUNTRY = "de"

#: Quarter-hourly day-ahead products began on this date.
QUARTER_HOURLY_FROM = pd.Timestamp("2025-10-01", tz="Europe/Berlin")


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.json"


#: Passed as `cache_name` by callers that must never read or write the cache,
#: such as anything fetching prices for hours still in the future.
NO_CACHE = "__uncached__"


def _request(path: str, cache_name: str, refresh: bool = False, **params) -> dict:
    """GET an endpoint, caching the JSON body under `cache_name`."""
    cached = _cache_path(cache_name)
    if cache_name != NO_CACHE and cached.exists() and not refresh:
        return json.loads(cached.read_text(encoding="utf-8"))

    response = requests.get(f"{BASE_URL}{path}", params=params, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()

    if cache_name != NO_CACHE:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _timestamps(payload: dict) -> pd.DatetimeIndex:
    if "unix_seconds" not in payload:
        raise KeyError(f"response has no unix_seconds; keys were {list(payload)}")
    return pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)


def fetch_price(
    start: date | str, end: date | str, *, refresh: bool = False
) -> pd.DataFrame:
    """Day-ahead spot price for the DE-LU bidding zone, EUR/MWh."""
    payload = _request(
        "/price",
        cache_name=f"price_{start}_{end}",
        refresh=refresh,
        bzn=BIDDING_ZONE,
        start=str(start),
        end=str(end),
    )

    return pd.DataFrame(
        {
            "utc_timestamp": _timestamps(payload),
            "price_eur_mwh": pd.to_numeric(payload["price"], errors="coerce"),
        }
    )


def fetch_upcoming_price(*, days_ahead: int = 2) -> pd.DataFrame:
    """Day-ahead prices including tomorrow, once the auction has cleared.

    Tomorrow's prices are *published*, not predicted: the auction closes at 12:00
    and results appear around 12:45 market time. Before that the response covers
    today only, which is why callers check how far the series actually reaches
    rather than assuming tomorrow is present.

    Never cached — the point is the newest publication.
    """
    today = date.today()
    payload = _request(
        "/price",
        cache_name=NO_CACHE,
        refresh=True,
        bzn=BIDDING_ZONE,
        start=str(today - timedelta(days=1)),
        end=str(today + timedelta(days=days_ahead)),
    )

    return pd.DataFrame(
        {
            "utc_timestamp": _timestamps(payload),
            "price_eur_mwh": pd.to_numeric(payload["price"], errors="coerce"),
        }
    ).dropna(subset=["price_eur_mwh"])


def fetch_public_power(
    start: date | str, end: date | str, *, refresh: bool = False
) -> pd.DataFrame:
    """Net public electricity production by production type, MW.

    Returned in long form: one row per timestamp and production type.
    """
    payload = _request(
        "/public_power",
        cache_name=f"public_power_{start}_{end}",
        refresh=refresh,
        country=COUNTRY,
        start=str(start),
        end=str(end),
    )

    timestamps = _timestamps(payload)
    frames = [
        pd.DataFrame(
            {
                "utc_timestamp": timestamps,
                "production_type": series["name"],
                "power_mw": pd.to_numeric(series["data"], errors="coerce"),
            }
        )
        for series in payload["production_types"]
    ]

    return pd.concat(frames, ignore_index=True)


def fetch_day_ahead_forecast(
    production_type: str,
    start: date | str,
    end: date | str,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """The transmission operators' own published day-ahead forecast.

    `production_type` is one of solar, wind_onshore, wind_offshore, load. Used as
    the benchmark that this project's models are scored against.
    """
    payload = _request(
        "/public_power_forecast",
        cache_name=f"forecast_{production_type}_{start}_{end}",
        refresh=refresh,
        country=COUNTRY,
        production_type=production_type,
        forecast_type="day-ahead",
        start=str(start),
        end=str(end),
    )

    return pd.DataFrame(
        {
            "utc_timestamp": _timestamps(payload),
            "production_type": production_type,
            "forecast_mw": pd.to_numeric(payload["forecast_values"], errors="coerce"),
        }
    )


def fetch_installed_power(*, refresh: bool = False) -> pd.DataFrame:
    """Installed generation capacity by type, MW, monthly.

    Solar output must be normalised by capacity: installed PV grows steadily, so
    raw MW carries a deployment trend unrelated to weather.
    """
    payload = _request(
        "/installed_power",
        cache_name="installed_power_monthly",
        refresh=refresh,
        country=COUNTRY,
        time_step="monthly",
    )

    # `time` is month labels such as "01.2022" rather than unix seconds.
    period = pd.to_datetime(payload["time"], format="%m.%Y", utc=True)

    frames = [
        pd.DataFrame(
            {
                "month": period,
                "production_type": series["name"],
                # Reported in GW.
                "capacity_mw": pd.to_numeric(series["data"], errors="coerce") * 1000,
            }
        )
        for series in payload["production_types"]
    ]

    return pd.concat(frames, ignore_index=True)
