"""Tests for the forward prediction path.

The backtest replays history; this code forecasts hours that have not happened.
The risk it carries is different: features must be built exactly as the training
mart builds them, and nothing may reach forward in time.
"""

import numpy as np
import pandas as pd
import pytest

from smartgrid.modelling import dataset, predict
from smartgrid.optimisation import Battery, plan_day

BERLIN = "Europe/Berlin"


@pytest.fixture
def history() -> pd.DataFrame:
    """Observed generation up to and including today."""
    index = pd.date_range("2026-06-01", "2026-08-01 21:00", freq="h", tz="UTC")
    local = index.tz_convert(BERLIN)
    daylight = np.clip(np.sin((local.hour.to_numpy() - 6) / 12 * np.pi), 0, None)
    rng = np.random.default_rng(0)

    return pd.DataFrame(
        {
            "utc_timestamp": index,
            dataset.TARGET: daylight * 0.4 * rng.uniform(0.5, 1.0, len(index)),
            "solar_ac_mw": 115_000.0,
        }
    )


@pytest.fixture
def forecast() -> pd.DataFrame:
    """Forward weather for the next three days, at all nine locations.

    Starts a day early on purpose: a Berlin day begins at 22:00 UTC the previous
    day in summer, so a forecast starting at UTC midnight would leave the first
    two hours of the target day uncovered.
    """
    index = pd.date_range("2026-08-01", periods=96, freq="h", tz="UTC")
    local = index.tz_convert(BERLIN)
    daylight = np.clip(np.sin((local.hour.to_numpy() - 6) / 12 * np.pi), 0, None)

    frames = []
    for offset, name in enumerate(f.removeprefix("ghi_") for f in dataset.LOCATION_FEATURES):
        frames.append(
            pd.DataFrame(
                {
                    "utc_timestamp": index,
                    "location": name,
                    "ghi_wm2": daylight * (700 + offset * 20),
                    "direct_radiation_wm2": daylight * 500,
                    "diffuse_radiation_wm2": daylight * 200,
                    "temperature_c": 20.0,
                    "cloud_cover_pct": 30.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# --- which day is being forecast --------------------------------------------


@pytest.mark.parametrize("hour", [0, 9, 14, 23])
def test_the_target_is_always_tomorrow(hour):
    """The noon auction clears tomorrow, so tomorrow is what can be planned.

    The day after is not schedulable: its prices do not exist until tomorrow's
    auction runs.
    """
    now = pd.Timestamp(f"2026-08-01 {hour:02d}:00", tz=BERLIN)
    assert predict.target_day(now).date() == pd.Timestamp("2026-08-02").date()


def test_prices_are_only_expected_after_the_auction_publishes():
    assert not predict.prices_expected(pd.Timestamp("2026-08-01 09:00", tz=BERLIN))
    assert not predict.prices_expected(pd.Timestamp("2026-08-01 12:30", tz=BERLIN))
    assert predict.prices_expected(pd.Timestamp("2026-08-01 14:00", tz=BERLIN))


# --- feature construction ----------------------------------------------------


@pytest.fixture
def features(history, forecast) -> pd.DataFrame:
    day = pd.Timestamp("2026-08-02", tz=BERLIN)
    return predict.build_prediction_features(history, forecast, day=day)


def test_every_training_feature_is_produced(features):
    """A missing column would fail at predict time; a renamed one would not."""
    missing = set(dataset.FEATURES) - set(features.columns)
    assert not missing, sorted(missing)


def test_one_row_per_hour_of_the_target_day(features):
    assert len(features) == 24


def test_lags_are_taken_from_observed_history(features, history):
    """Not from the forecast, and not from hours after the gate."""
    observed = history.set_index("utc_timestamp")[dataset.TARGET]

    for lag in predict.LAG_HOURS:
        column = f"capacity_factor_lag_{lag}h"
        for timestamp in features.index[:6]:
            expected = observed.get(timestamp - pd.Timedelta(lag, "h"))
            if pd.notna(expected):
                assert features.loc[timestamp, column] == pytest.approx(expected)


def test_no_feature_reads_beyond_the_gate(features, history):
    """The gate is noon the day before; the shortest lag is 48 hours."""
    day = features.index.min()
    gate = predict.gate_for(day)
    earliest_lag_source = day - pd.Timedelta(min(predict.LAG_HOURS), "h")

    assert earliest_lag_source <= gate


def test_day_of_week_uses_the_warehouse_convention(features):
    """BigQuery numbers Sunday as 1; pandas numbers Monday as 0."""
    # 2026-08-02 is a Sunday.
    assert set(features["day_of_week"]) == {1}


def test_the_spread_feature_reflects_disagreement(features):
    assert (features["ghi_spread"] >= 0).all()
    assert features["ghi_spread"].max() > 0, "locations were given different values"


def test_a_forecast_that_stops_short_is_reported(history, forecast):
    far_off = pd.Timestamp("2026-09-01", tz=BERLIN)
    with pytest.raises(ValueError, match="does not cover"):
        predict.build_prediction_features(history, forecast, day=far_off)


def test_missing_locations_are_reported(history, forecast):
    thin = forecast[forecast["location"] != "munich"]
    with pytest.raises(KeyError, match="missing locations"):
        predict.build_prediction_features(
            history, thin, day=pd.Timestamp("2026-08-02", tz=BERLIN)
        )


# --- the recommendation ------------------------------------------------------


@pytest.fixture
def prediction(features) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "utc_timestamp": features.index,
            "predicted_capacity_factor": features["ghi_mean"].to_numpy() / 2000,
            "target_date": pd.Timestamp("2026-08-02").date(),
        }
    )


@pytest.fixture
def prices(features) -> pd.Series:
    cheap_then_dear = np.array([20.0] * 12 + [140.0] * 12)
    return pd.Series(cheap_then_dear, index=features.index, name="price_eur_mwh")


def test_a_plan_covers_every_hour(prediction, prices):
    plan = plan_day(prediction, prices, battery=Battery())
    assert len(plan.hours) == 24
    assert set(plan.hours["local_hour"]) == set(range(24))


def test_the_plan_charges_cheap_and_discharges_dear(prediction, prices):
    plan = plan_day(prediction, prices, battery=Battery())

    charge_hours = plan.hours.loc[plan.hours["charge_kw"] > 0.01, "price_eur_mwh"]
    discharge_hours = plan.hours.loc[plan.hours["discharge_kw"] > 0.01, "price_eur_mwh"]

    assert charge_hours.max() < discharge_hours.min()


def test_the_advice_is_actionable(prediction, prices):
    advice = plan_day(prediction, prices, battery=Battery()).advice()
    joined = " ".join(advice)

    assert "Charge" in joined
    assert "Discharge" in joined
    assert "EUR" in joined


def test_benefit_combines_battery_and_solar(prediction, prices):
    plan = plan_day(prediction, prices, battery=Battery())

    assert plan.revenue_eur > 0
    assert plan.solar_value_eur > 0
    assert plan.total_benefit_eur == pytest.approx(
        plan.revenue_eur + plan.solar_value_eur
    )


def test_a_bigger_array_is_worth_more(prediction, prices):
    small = plan_day(prediction, prices, system_kwp=5.0)
    large = plan_day(prediction, prices, system_kwp=20.0)

    assert large.solar_value_eur > small.solar_value_eur


def test_planning_without_a_full_day_of_prices_is_refused(prediction, prices):
    with pytest.raises(ValueError, match="24 priced hours"):
        plan_day(prediction, prices.iloc[:10])
