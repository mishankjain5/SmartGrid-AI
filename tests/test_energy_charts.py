"""Energy-Charts client tests.

Parsing is checked against fixtures shaped like real responses. Tests that call
the live API are marked `network` and can be deselected with `-m "not network"`.
"""

import json

import pandas as pd
import pytest

from smartgrid.sources import energy_charts as ec

PRICE_PAYLOAD = {
    "license_info": "CC BY 4.0",
    "unix_seconds": [1640995200, 1640998800, 1641002400],
    "price": [50.1, -3.2, 61.0],
    "unit": "EUR / MWh",
}

POWER_PAYLOAD = {
    "unix_seconds": [1640995200, 1640998800],
    "production_types": [
        {"name": "Solar", "data": [0.0, 120.5]},
        {"name": "Wind onshore", "data": [8000.0, 7500.25]},
    ],
}

FORECAST_PAYLOAD = {
    "unix_seconds": [1640995200, 1640998800],
    "forecast_values": [0.0, 118.0],
    "production_type": "solar",
    "forecast_type": "day-ahead",
}

CAPACITY_PAYLOAD = {
    "time": ["01.2022", "02.2022"],
    "production_types": [
        {"name": "Solar DC", "data": [59.0, 59.5]},
        {"name": "Wind onshore", "data": [56.1, None]},
    ],
}


@pytest.fixture
def stub_request(monkeypatch):
    """Replace the HTTP layer with a fixture payload."""

    def install(payload):
        monkeypatch.setattr(ec, "_request", lambda *a, **k: payload)

    return install


def test_price_is_parsed_with_utc_timestamps(stub_request):
    stub_request(PRICE_PAYLOAD)
    frame = ec.fetch_price("2022-01-01", "2022-01-02")

    assert list(frame.columns) == ["utc_timestamp", "price_eur_mwh"]
    assert len(frame) == 3
    assert str(frame["utc_timestamp"].dt.tz) == "UTC"
    assert frame["utc_timestamp"].iloc[0] == pd.Timestamp("2022-01-01", tz="UTC")


def test_negative_prices_are_preserved(stub_request):
    stub_request(PRICE_PAYLOAD)
    frame = ec.fetch_price("2022-01-01", "2022-01-02")
    assert (frame["price_eur_mwh"] < 0).sum() == 1


def test_public_power_is_returned_in_long_form(stub_request):
    stub_request(POWER_PAYLOAD)
    frame = ec.fetch_public_power("2022-01-01", "2022-01-02")

    assert list(frame.columns) == ["utc_timestamp", "production_type", "power_mw"]
    assert len(frame) == 4
    assert set(frame["production_type"]) == {"Solar", "Wind onshore"}

    solar = frame[frame["production_type"] == "Solar"]
    assert solar["power_mw"].tolist() == [0.0, 120.5]


def test_day_ahead_forecast_carries_its_production_type(stub_request):
    stub_request(FORECAST_PAYLOAD)
    frame = ec.fetch_day_ahead_forecast("solar", "2022-01-01", "2022-01-02")

    assert list(frame.columns) == ["utc_timestamp", "production_type", "forecast_mw"]
    assert set(frame["production_type"]) == {"solar"}


def test_installed_power_parses_months_and_converts_to_mw(stub_request):
    stub_request(CAPACITY_PAYLOAD)
    frame = ec.fetch_installed_power()

    assert frame["month"].iloc[0] == pd.Timestamp("2022-01-01", tz="UTC")
    solar = frame[frame["production_type"] == "Solar DC"]
    # Reported in GW; 59.0 GW is 59,000 MW.
    assert solar["capacity_mw"].tolist() == [59000.0, 59500.0]


def test_missing_values_survive_as_nan(stub_request):
    stub_request(CAPACITY_PAYLOAD)
    frame = ec.fetch_installed_power()
    wind = frame[frame["production_type"] == "Wind onshore"]
    assert wind["capacity_mw"].isna().sum() == 1


def test_unexpected_payload_shape_is_reported(stub_request):
    stub_request({"unexpected": []})
    with pytest.raises(KeyError, match="unix_seconds"):
        ec.fetch_price("2022-01-01", "2022-01-02")


# --- caching ----------------------------------------------------------------


def test_response_is_cached_and_reused(monkeypatch, tmp_path):
    calls = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return PRICE_PAYLOAD

    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        return Response()

    monkeypatch.setattr(ec, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(ec.requests, "get", fake_get)

    first = ec._request("/price", cache_name="sample", start="a", end="b")
    second = ec._request("/price", cache_name="sample", start="a", end="b")

    assert first == second == PRICE_PAYLOAD
    assert len(calls) == 1, "second call must be served from disk"
    assert json.loads((tmp_path / "sample.json").read_text()) == PRICE_PAYLOAD


def test_refresh_bypasses_the_cache(monkeypatch, tmp_path):
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return PRICE_PAYLOAD

    monkeypatch.setattr(ec, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        ec.requests, "get", lambda *a, **k: (calls.append(1), Response())[1]
    )

    ec._request("/price", cache_name="sample")
    ec._request("/price", cache_name="sample", refresh=True)

    assert len(calls) == 2


# --- live API ---------------------------------------------------------------


@pytest.mark.network
def test_live_price_request_returns_data(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "CACHE_DIR", tmp_path)
    frame = ec.fetch_price("2026-01-01", "2026-01-03")

    assert len(frame) > 24
    assert frame["utc_timestamp"].is_monotonic_increasing
    assert frame["price_eur_mwh"].notna().any()
