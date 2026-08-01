"""Tests for persisting and scoring forecasts."""

import pandas as pd
import pytest

from smartgrid.modelling import store


@pytest.fixture
def prediction() -> pd.DataFrame:
    index = pd.date_range("2026-08-02", periods=24, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "utc_timestamp": index,
            "predicted_capacity_factor": 0.1,
            "predicted_mw": 11500.0,
            "ghi_mean_forecast": 300.0,
            "issued_at": pd.Timestamp("2026-08-01 13:00", tz="UTC"),
            "target_date": pd.Timestamp("2026-08-02").date(),
        }
    )


def test_saving_appends_rather_than_replaces(monkeypatch, prediction):
    """Re-running for a day must not erase the earlier call.

    Two forecasts for one day, issued at different times, are a revision history.
    Overwriting would hide that the forecast changed.
    """
    captured = {}

    def fake_load(frame, table, *, dataset, replace):
        captured["replace"] = replace
        captured["rows"] = len(frame)
        captured["columns"] = set(frame.columns)
        return len(frame)

    monkeypatch.setattr(store, "load_dataframe", fake_load)
    store.save_predictions(prediction, model_name="gradient_boosting")

    assert captured["replace"] is False
    assert captured["rows"] == 24
    assert "model_name" in captured["columns"]


def test_saving_requires_the_columns_scoring_needs(monkeypatch, prediction):
    monkeypatch.setattr(store, "load_dataframe", lambda *a, **k: 0)

    with pytest.raises(KeyError, match="missing columns"):
        store.save_predictions(
            prediction.drop(columns=["predicted_capacity_factor"]),
            model_name="gradient_boosting",
        )


# --- scoring -----------------------------------------------------------------


def _scored(**overrides) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "utc_timestamp": pd.date_range("2026-08-02 08:00", periods=6, freq="h", tz="UTC"),
            "target_date": pd.Timestamp("2026-08-02").date(),
            "predicted_capacity_factor": [0.10, 0.20, 0.30, 0.25, 0.15, 0.05],
            "actual_capacity_factor": [0.12, 0.18, 0.28, 0.27, 0.14, 0.06],
            "tso_capacity_factor": [0.11, 0.19, 0.31, 0.24, 0.16, 0.04],
            "ghi_mean_forecast": [100.0, 400.0, 700.0, 500.0, 200.0, 0.0],
        }
    )
    return base.assign(**overrides)


def test_scoring_reports_both_our_forecast_and_the_benchmark():
    summary = store.summarise_scores(_scored())

    assert set(summary.index) == {"our forecast", "TSO forecast"}
    assert (summary["mae"] > 0).all()


def test_scoring_excludes_night():
    """The last hour has zero forecast irradiance and must not be scored."""
    summary = store.summarise_scores(_scored())
    assert summary.loc["our forecast", "hours"] == 5


def test_scoring_without_a_benchmark_still_reports_our_error():
    """The operators' forecast may be absent for very recent hours."""
    summary = store.summarise_scores(_scored(tso_capacity_factor=float("nan")))

    assert set(summary.index) == {"our forecast"}
    assert summary.loc["our forecast", "mae"] > 0


def test_nothing_to_score_is_explained():
    with pytest.raises(ValueError, match="no saved forecast"):
        store.summarise_scores(pd.DataFrame())


def test_forecasts_with_no_daylight_are_reported():
    dark = _scored(ghi_mean_forecast=0.0)
    with pytest.raises(ValueError, match="no daylight hours"):
        store.summarise_scores(dark)
