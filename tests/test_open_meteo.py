"""Open-Meteo client tests."""

import pandas as pd
import pytest

from smartgrid.sources import open_meteo as om

TIMES = ["2024-06-10T00:00", "2024-06-10T01:00", "2024-06-10T02:00"]


def _location(latitude: float, longitude: float, ghi: list[float]) -> dict:
    """A response for one coordinate, shaped like the real thing."""
    hourly = {"time": TIMES}
    for api_name in om.VARIABLES:
        hourly[api_name] = ghi if "radiation" in api_name else [10.0, 10.5, 11.0]
    for api_name in om.DAY_AHEAD_VARIABLES:
        # The day-ahead lead is a different forecast, so different values.
        base = ghi if "radiation" in api_name else [9.0, 9.5, 10.0]
        hourly[api_name] = [v * 0.8 for v in base]

    return {"latitude": latitude, "longitude": longitude, "hourly": hourly}


PAYLOAD = [
    _location(53.55, 9.99, [0.0, 0.0, 80.0]),
    _location(48.14, 11.58, [0.0, 0.0, 400.0]),
]

TWO_POINTS = {"hamburg": (53.55, 9.99), "munich": (48.14, 11.58)}


@pytest.fixture
def stub_request(monkeypatch):
    """Serve each location its own response, keyed by cache name."""
    by_name = dict(zip(TWO_POINTS, PAYLOAD, strict=True))

    def fake(cache_name: str, refresh: bool = False, **params):
        for name in by_name:
            if f"_{name}_" in cache_name:
                return by_name[name]
        raise AssertionError(f"unexpected cache name {cache_name}")

    monkeypatch.setattr(om, "_request", fake)


def test_long_form_has_one_row_per_location_and_hour(stub_request):
    frame = om.fetch_weather("2024-06-10", "2024-06-11", locations=TWO_POINTS)

    assert len(frame) == len(TIMES) * len(TWO_POINTS)
    assert set(frame["location"]) == set(TWO_POINTS)
    assert frame.groupby("location").size().eq(len(TIMES)).all()


def test_locations_are_labelled_and_carry_coordinates(stub_request):
    frame = om.fetch_weather("2024-06-10", "2024-06-11", locations=TWO_POINTS)
    hamburg = frame[frame["location"] == "hamburg"].iloc[0]

    assert hamburg["latitude"] == 53.55
    assert hamburg["longitude"] == 9.99


def test_locations_differ_from_one_another(stub_request):
    """The whole reason for fetching several points."""
    frame = om.fetch_weather("2024-06-10", "2024-06-11", locations=TWO_POINTS)
    peak = frame.groupby("location")["ghi_wm2"].max()

    assert peak["munich"] > peak["hamburg"]


def test_both_lead_times_are_returned(stub_request):
    frame = om.fetch_weather("2024-06-10", "2024-06-11", locations=TWO_POINTS)

    for column in ("ghi_wm2", "temperature_c", "cloud_cover_pct"):
        assert column in frame.columns
        assert f"{column}_day_ahead" in frame.columns

    munich = frame[frame["location"] == "munich"]
    assert munich["ghi_wm2"].max() != munich["ghi_wm2_day_ahead"].max()


def test_timestamps_are_utc_aware_and_sorted(stub_request):
    frame = om.fetch_weather("2024-06-10", "2024-06-11", locations=TWO_POINTS)
    hamburg = frame[frame["location"] == "hamburg"]

    assert str(frame["utc_timestamp"].dt.tz) == "UTC"
    assert hamburg["utc_timestamp"].is_monotonic_increasing


def test_requests_before_the_archive_start_are_refused():
    with pytest.raises(ValueError, match="archive starts"):
        om.fetch_weather("2021-12-31", "2022-01-05")


def test_each_location_is_requested_separately(monkeypatch):
    """One combined request makes the server work sequentially and time out."""
    seen = []

    def record(cache_name: str, refresh: bool = False, **params):
        seen.append((cache_name, params["latitude"], params["longitude"]))
        return PAYLOAD[0]

    monkeypatch.setattr(om, "_request", record)
    om.fetch_weather("2024-06-10", "2024-06-11", locations=TWO_POINTS)

    assert len(seen) == len(TWO_POINTS)
    assert len({lat for _, lat, _ in seen}) == len(TWO_POINTS)
    # Cache keys name the location, so refreshing one leaves the others alone.
    assert all("hamburg" in c or "munich" in c for c, _, _ in seen)


def test_missing_variables_are_reported(monkeypatch):
    trimmed = {"latitude": 1, "longitude": 2,
               "hourly": {"time": TIMES, "temperature_2m": [1, 2, 3]}}
    monkeypatch.setattr(om, "_request", lambda *a, **k: trimmed)

    with pytest.raises(KeyError, match="missing variables"):
        om.fetch_weather("2024-06-10", "2024-06-11", locations={"one": (1.0, 2.0)})


def test_default_locations_span_the_country():
    latitudes = [lat for lat, _ in om.LOCATIONS.values()]
    longitudes = [lon for _, lon in om.LOCATIONS.values()]

    assert len(om.LOCATIONS) >= 8
    assert max(latitudes) - min(latitudes) > 5, "north to south"
    assert max(longitudes) - min(longitudes) > 5, "west to east"


def test_rate_limiting_is_retried_not_fatal(monkeypatch, tmp_path):
    """A 429 must not throw away the locations that already downloaded."""
    attempts = []

    class Response:
        def __init__(self, status):
            self.status_code = status

        def raise_for_status(self):
            if self.status_code >= 400:
                raise AssertionError("should not reach raise_for_status on retry")

        def json(self):
            return PAYLOAD[0]

    def flaky(*args, **kwargs):
        attempts.append(1)
        return Response(429 if len(attempts) < 3 else 200)

    monkeypatch.setattr(om, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(om, "BACKOFF_SECONDS", 0)
    monkeypatch.setattr(om.time, "sleep", lambda _: None)
    monkeypatch.setattr(om.requests, "get", flaky)

    assert om._request("weather_hamburg_a_b") == PAYLOAD[0]
    assert len(attempts) == 3


def test_persistent_rate_limiting_is_reported(monkeypatch, tmp_path):
    class Response:
        status_code = 429

        def raise_for_status(self):
            raise AssertionError("a persistent 429 should raise before this")

    monkeypatch.setattr(om, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(om, "BACKOFF_SECONDS", 0)
    monkeypatch.setattr(om.time, "sleep", lambda _: None)
    monkeypatch.setattr(om.requests, "get", lambda *a, **k: Response())

    with pytest.raises(RuntimeError, match="still rate limited"):
        om._request("weather_hamburg_a_b")


def test_response_is_cached_per_location(monkeypatch, tmp_path):
    calls = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return PAYLOAD[0]

    monkeypatch.setattr(om, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        om.requests, "get", lambda *a, **k: (calls.append(1), Response())[1]
    )

    om._request("weather_hamburg_a_b")
    om._request("weather_hamburg_a_b")
    om._request("weather_munich_a_b")

    assert len(calls) == 2, "each location caches independently"
    assert {p.name for p in tmp_path.iterdir()} == {
        "weather_hamburg_a_b.json",
        "weather_munich_a_b.json",
    }


@pytest.mark.network
def test_live_multi_point_request(monkeypatch, tmp_path):
    monkeypatch.setattr(om, "CACHE_DIR", tmp_path)
    frame = om.fetch_weather("2025-06-10", "2025-06-12")

    assert set(frame["location"]) == set(om.LOCATIONS)
    assert len(frame) == 72 * len(om.LOCATIONS)

    # At a single sunny midday hour the country should not look uniform.
    midday = frame[frame["utc_timestamp"] == pd.Timestamp("2025-06-10 11:00", tz="UTC")]
    spread = midday["ghi_wm2_day_ahead"].max() - midday["ghi_wm2_day_ahead"].min()
    assert spread > 100, "regional variation is the reason for multiple points"
