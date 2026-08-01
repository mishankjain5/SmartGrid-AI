"""Household consumption tests."""

import numpy as np
import pandas as pd
import pytest

from smartgrid.modelling import household


@pytest.fixture
def metered() -> pd.DataFrame:
    """A year of hourly consumption with a realistic daily shape and heavy noise.

    The noise matters: a single home's hourly draw is dominated by when someone
    happens to run an appliance, which is why the fitted profile beats a flat
    assumption by so little on real data.
    """
    index = pd.date_range("2017-01-01", periods=24 * 365, freq="h", tz="Europe/Berlin")
    rng = np.random.default_rng(0)

    hour = index.hour.to_numpy()
    morning = np.exp(-(((hour - 7) / 2.0) ** 2))
    evening = np.exp(-(((hour - 19) / 2.5) ** 2))
    shape = 0.3 + 0.9 * morning + 1.1 * evening

    # Weekends flatten and shift later.
    weekend = np.isin(index.dayofweek.to_numpy(), (5, 6))
    shape = np.where(weekend, shape * 0.85 + 0.15, shape)

    return pd.DataFrame(
        {
            "utc_timestamp": index.tz_convert("UTC"),
            "local_datetime": index.tz_localize(None),
            "building": "residential4",
            household.TARGET: np.clip(shape + rng.normal(0, 0.4, len(index)), 0, None),
        }
    )


def test_the_profile_covers_every_hour_of_the_week(metered):
    profile = household.fit_profile(metered)
    assert len(profile.by_hour_of_week) == 168


def test_it_recovers_the_daily_shape(metered):
    profile = household.fit_profile(metered)
    day = profile.for_day(pd.Timestamp("2026-08-04", tz="Europe/Berlin"))  # a Tuesday

    local_hours = day.index.tz_convert("Europe/Berlin").hour
    overnight = day[np.isin(local_hours, (2, 3, 4))].mean()
    peak = day[np.isin(local_hours, (18, 19, 20))].mean()

    assert peak > overnight * 1.5, "the evening peak should survive the averaging"


def test_weekends_differ_from_weekdays(metered):
    profile = household.fit_profile(metered)

    tuesday = profile.for_day(pd.Timestamp("2026-08-04", tz="Europe/Berlin")).sum()
    sunday = profile.for_day(pd.Timestamp("2026-08-02", tz="Europe/Berlin")).sum()

    assert tuesday != sunday, "a week-shaped profile must distinguish the two"


def test_a_day_is_twenty_four_hours(metered):
    profile = household.fit_profile(metered)
    day = profile.for_day(pd.Timestamp("2026-08-02", tz="Europe/Berlin"))

    assert len(day) == 24
    assert str(day.index.tz) == "UTC"


def test_daily_total_matches_the_observed_level(metered):
    profile = household.fit_profile(metered)
    observed = metered[household.TARGET].mean() * 24

    assert profile.daily_kwh == pytest.approx(observed, rel=0.02)


def test_scoring_compares_against_a_flat_assumption(metered):
    """The honest framing: a profile replaces a constant, so beat the constant."""
    profile = household.fit_profile(metered)
    scores = household.score_profile(metered, profile)

    assert scores["hours"] == len(metered)
    assert scores["profile_mae_kwh"] <= scores["flat_mae_kwh"]
    assert scores["improvement"] >= 0


def test_an_empty_panel_is_reported():
    empty = pd.DataFrame(
        {"local_datetime": [], "utc_timestamp": [], household.TARGET: []}
    )
    with pytest.raises(ValueError, match="no consumption data"):
        household.fit_profile(empty)


def test_the_feature_list_excludes_the_target():
    assert household.TARGET not in household.FEATURES
