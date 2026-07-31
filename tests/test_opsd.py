"""OPSD household client tests."""

import pandas as pd
import pytest

from smartgrid.sources import opsd

CSV = (
    "utc_timestamp,cet_cest_timestamp,DE_KN_residential4_pv,DE_KN_industrial3_machine_1\n"
    "2017-06-05T00:00:00Z,2017-06-05T02:00:00+0200,100.5,20.0\n"
    "2017-06-05T01:00:00Z,2017-06-05T03:00:00+0200,100.9,\n"
    "2017-06-05T02:00:00Z,2017-06-05T04:00:00+0200,101.7,21.5\n"
)


@pytest.fixture
def cached_csv(monkeypatch, tmp_path):
    path = tmp_path / "household.csv"
    path.write_text(CSV, encoding="utf-8")
    monkeypatch.setattr(opsd, "CACHE_PATH", path)
    return path


def test_channel_columns_ignores_metadata(cached_csv):
    wide = pd.read_csv(cached_csv)
    assert opsd.channel_columns(wide) == [
        "DE_KN_residential4_pv",
        "DE_KN_industrial3_machine_1",
    ]


def test_load_returns_long_form(cached_csv):
    frame = opsd.load()
    assert set(frame.columns) == {
        "utc_timestamp",
        "channel",
        "meter_kwh",
        "building",
        "building_type",
        "device",
    }
    # Five readings, not six: one cell is blank.
    assert len(frame) == 5


def test_channel_names_are_parsed(cached_csv):
    frame = opsd.load()
    pv = frame[frame["channel"] == "DE_KN_residential4_pv"].iloc[0]

    assert pv["building"] == "residential4"
    assert pv["building_type"] == "residential"
    assert pv["device"] == "pv"


def test_multi_word_devices_are_kept_intact(cached_csv):
    frame = opsd.load()
    machine = frame[frame["building_type"] == "industrial"].iloc[0]
    assert machine["building"] == "industrial3"
    assert machine["device"] == "machine_1"


def test_meter_readings_are_returned_as_stored(cached_csv):
    """Values are cumulative; differencing happens downstream."""
    frame = opsd.load()
    pv = frame[frame["channel"] == "DE_KN_residential4_pv"]

    assert pv["meter_kwh"].tolist() == [100.5, 100.9, 101.7]
    assert pv["meter_kwh"].is_monotonic_increasing


def test_missing_readings_are_dropped_not_zero_filled(cached_csv):
    frame = opsd.load()
    machine = frame[frame["channel"] == "DE_KN_industrial3_machine_1"]
    assert len(machine) == 2
    assert machine["meter_kwh"].notna().all()


def test_download_skips_when_cached(monkeypatch, cached_csv):
    monkeypatch.setattr(
        opsd.requests, "get", lambda *a, **k: pytest.fail("should not download")
    )
    assert opsd.download() == cached_csv


@pytest.mark.network
def test_live_download_and_load(monkeypatch, tmp_path):
    monkeypatch.setattr(opsd, "CACHE_PATH", tmp_path / "household.csv")
    frame = opsd.load()

    assert len(frame) > 500_000
    assert frame["building_type"].isin({"residential", "industrial", "public"}).all()
    assert frame["utc_timestamp"].min().year == 2014
