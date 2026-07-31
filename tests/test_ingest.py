"""Ingestion orchestration tests.

The BigQuery load is stubbed; what is checked here is table selection, the date
window, and that each builder produces the expected shape.
"""

from datetime import date

import pandas as pd
import pytest

from smartgrid import ingest
from smartgrid.config import DATA_START


@pytest.fixture
def captured_loads(monkeypatch):
    """Replace the BigQuery load with a recorder."""
    loads: dict[str, pd.DataFrame] = {}

    def fake_load(frame, table, **kwargs):
        loads[table] = frame
        return len(frame)

    monkeypatch.setattr(ingest, "load_dataframe", fake_load)
    return loads


@pytest.fixture
def stub_sources(monkeypatch):
    """Every builder returns a one-row frame tagged with its arguments."""
    calls = []

    def builder(name):
        def build(start, end, refresh):
            calls.append((name, start, end, refresh))
            return pd.DataFrame({"utc_timestamp": [pd.Timestamp("2022-01-01", tz="UTC")]})

        return build

    monkeypatch.setattr(ingest, "TABLES", {n: builder(n) for n in ingest.TABLES})
    return calls


def test_all_tables_load_by_default(captured_loads, stub_sources):
    loaded = ingest.ingest()
    assert set(loaded) == {
        "price",
        "public_power",
        "day_ahead_forecast",
        "installed_power",
        "weather",
        "household",
    }
    assert set(captured_loads) == set(loaded)


def test_a_subset_can_be_selected(captured_loads, stub_sources):
    loaded = ingest.ingest(["price", "weather"])
    assert set(loaded) == {"price", "weather"}
    assert "household" not in captured_loads


def test_unknown_table_is_rejected(captured_loads, stub_sources):
    with pytest.raises(ValueError, match="unknown tables"):
        ingest.ingest(["not_a_table"])


def test_default_window_starts_at_the_archive_boundary(captured_loads, stub_sources):
    ingest.ingest(["price"])
    _, start, end, _ = stub_sources[0]

    assert start == DATA_START
    assert end < date.today(), "end is the last complete day, not today"


def test_explicit_window_is_passed_through(captured_loads, stub_sources):
    ingest.ingest(["price"], start=date(2023, 5, 1), end=date(2023, 5, 3))
    _, start, end, _ = stub_sources[0]

    assert (start, end) == (date(2023, 5, 1), date(2023, 5, 3))


def test_refresh_flag_reaches_the_builders(captured_loads, stub_sources):
    ingest.ingest(["price"], refresh=True)
    assert stub_sources[0][3] is True


# --- builders ---------------------------------------------------------------


def test_day_ahead_forecast_concatenates_every_production_type(monkeypatch):
    requested = []

    def fake_fetch(production_type, start, end, refresh=False):
        requested.append(production_type)
        return pd.DataFrame(
            {
                "utc_timestamp": [pd.Timestamp("2022-01-01", tz="UTC")],
                "production_type": [production_type],
                "forecast_mw": [1.0],
            }
        )

    monkeypatch.setattr(ingest.energy_charts, "fetch_day_ahead_forecast", fake_fetch)

    frame = ingest.build_day_ahead_forecast(date(2022, 1, 1), date(2022, 1, 2), False)

    assert requested == list(ingest.FORECAST_TYPES)
    assert len(frame) == len(ingest.FORECAST_TYPES)
    assert set(frame["production_type"]) == set(ingest.FORECAST_TYPES)


def test_cli_rejects_an_unknown_table():
    with pytest.raises(SystemExit):
        ingest.main(["nonsense"])


def test_cli_parses_a_date_window(captured_loads, stub_sources):
    assert ingest.main(["price", "--start", "2023-01-01", "--end", "2023-01-05"]) == 0
    _, start, end, _ = stub_sources[0]
    assert (start, end) == (date(2023, 1, 1), date(2023, 1, 5))
