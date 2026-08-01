"""Open-Meteo client tests."""

import pandas as pd
import pytest

from smartgrid.sources import open_meteo as om


def _payload() -> dict:
    hourly = {
        "time": ["2022-01-01T00:00", "2022-01-01T01:00", "2022-01-01T02:00"],
        "temperature_2m": [11.3, 11.3, 11.1],
        "shortwave_radiation": [0.0, 0.0, 45.5],
        "direct_radiation": [0.0, 0.0, 30.0],
        "diffuse_radiation": [0.0, 0.0, 15.5],
        "cloud_cover": [100, 88, 42],
    }
    # The day-ahead lead is a different forecast, so deliberately different values.
    hourly |= {
        "temperature_2m_previous_day1": [10.9, 10.8, 10.5],
        "shortwave_radiation_previous_day1": [0.0, 0.0, 38.0],
        "direct_radiation_previous_day1": [0.0, 0.0, 24.0],
        "diffuse_radiation_previous_day1": [0.0, 0.0, 14.0],
        "cloud_cover_previous_day1": [100, 95, 55],
    }
    return {"latitude": 51.16, "longitude": 10.46, "timezone": "GMT", "hourly": hourly}


WEATHER_PAYLOAD = _payload()


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
        "temperature_c_day_ahead",
        "ghi_wm2_day_ahead",
        "direct_radiation_wm2_day_ahead",
        "diffuse_radiation_wm2_day_ahead",
        "cloud_cover_pct_day_ahead",
    ]


def test_day_ahead_lead_is_a_distinct_forecast(stub_request):
    """The point of fetching both: they are different values, not aliases."""
    frame = om.fetch_weather("2022-01-01", "2022-01-02")
    assert frame["ghi_wm2"].tolist() != frame["ghi_wm2_day_ahead"].tolist()
    assert frame["ghi_wm2_day_ahead"].iloc[2] == 38.0


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


@pytest.mark.network
def test_live_day_ahead_lead_differs_from_short_lead(monkeypatch, tmp_path):
    """Confirms against the live API that the two leads are genuinely different."""
    monkeypatch.setattr(om, "CACHE_DIR", tmp_path)
    frame = om.fetch_weather("2024-06-10", "2024-06-12")

    both = frame.dropna(subset=["ghi_wm2", "ghi_wm2_day_ahead"])
    assert len(both) > 48
    assert not both["ghi_wm2"].equals(both["ghi_wm2_day_ahead"])

    daylight = both[both["ghi_wm2"] > 50]
    disagreement = (daylight["ghi_wm2"] - daylight["ghi_wm2_day_ahead"]).abs().mean()
    assert disagreement > 10, "a day of lead time should cost real accuracy"
