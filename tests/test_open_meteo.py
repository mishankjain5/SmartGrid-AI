"""Open-Meteo client tests."""

import pandas as pd
import pytest

from smartgrid.sources import open_meteo as om

WEATHER_PAYLOAD = {
    "latitude": 51.16,
    "longitude": 10.46,
    "timezone": "GMT",
    "hourly_units": {"temperature_2m": "°C", "shortwave_radiation": "W/m²"},
    "hourly": {
        "time": ["2022-01-01T00:00", "2022-01-01T01:00", "2022-01-01T02:00"],
        "temperature_2m": [11.3, 11.3, 11.1],
        "shortwave_radiation": [0.0, 0.0, 45.5],
        "direct_radiation": [0.0, 0.0, 30.0],
        "diffuse_radiation": [0.0, 0.0, 15.5],
        "cloud_cover": [100, 88, 42],
    },
}


@pytest.fixture
def stub_request(monkeypatch):
    monkeypatch.setattr(om, "_request", lambda *a, **k: WEATHER_PAYLOAD)


def test_columns_are_renamed_to_units_bearing_names(stub_request):
    frame = om.fetch_weather("2022-01-01", "2022-01-02")
    assert list(frame.columns) == [
        "utc_timestamp",
        "temperature_c",
        "ghi_wm2",
        "direct_radiation_wm2",
        "diffuse_radiation_wm2",
        "cloud_cover_pct",
    ]


def test_timestamps_are_utc_aware(stub_request):
    frame = om.fetch_weather("2022-01-01", "2022-01-02")
    assert str(frame["utc_timestamp"].dt.tz) == "UTC"
    assert frame["utc_timestamp"].iloc[0] == pd.Timestamp("2022-01-01 00:00", tz="UTC")
    assert frame["utc_timestamp"].is_monotonic_increasing


def test_values_are_numeric(stub_request):
    frame = om.fetch_weather("2022-01-01", "2022-01-02")
    assert frame["ghi_wm2"].tolist() == [0.0, 0.0, 45.5]
    assert frame["temperature_c"].dtype.kind == "f"


def test_requests_before_the_archive_start_are_refused():
    with pytest.raises(ValueError, match="archive starts"):
        om.fetch_weather("2021-12-31", "2022-01-05")


def test_missing_variables_are_reported(monkeypatch):
    trimmed = {"hourly": {"time": ["2022-01-01T00:00"], "temperature_2m": [1.0]}}
    monkeypatch.setattr(om, "_request", lambda *a, **k: trimmed)

    with pytest.raises(KeyError, match="missing variables"):
        om.fetch_weather("2022-01-01", "2022-01-02")


def test_response_is_cached(monkeypatch, tmp_path):
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return WEATHER_PAYLOAD

    monkeypatch.setattr(om, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        om.requests, "get", lambda *a, **k: (calls.append(1), Response())[1]
    )

    om._request("sample", latitude=1, longitude=2)
    om._request("sample", latitude=1, longitude=2)
    assert len(calls) == 1


@pytest.mark.network
def test_live_request_returns_hourly_data(monkeypatch, tmp_path):
    monkeypatch.setattr(om, "CACHE_DIR", tmp_path)
    frame = om.fetch_weather("2024-06-01", "2024-06-03")

    assert len(frame) == 72
    assert frame["utc_timestamp"].diff().dropna().eq(pd.Timedelta(1, "h")).all()
    # Irradiance must be zero at night and positive at midday somewhere in range.
    assert frame["ghi_wm2"].min() == 0
    assert frame["ghi_wm2"].max() > 100
