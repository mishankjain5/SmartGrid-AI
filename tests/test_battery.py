"""Battery dispatch tests.

The optimiser is a linear program, so its behaviour is checkable against
properties that must hold for any correct solution rather than against
remembered numbers.
"""

import numpy as np
import pandas as pd
import pytest

from smartgrid.optimisation import Battery, annualise, optimise_day, optimise_series

#: Cheap overnight, dear in the evening — the shape arbitrage exists to exploit.
CHEAP_THEN_DEAR = np.array([10.0] * 12 + [100.0] * 12)


@pytest.fixture
def battery() -> Battery:
    return Battery(energy_mwh=0.010, power_mw=0.005, round_trip_efficiency=0.90)


def test_it_charges_when_cheap_and_discharges_when_dear(battery):
    schedule = optimise_day(CHEAP_THEN_DEAR, battery)

    assert schedule.charge_mw[:12].sum() > 0
    assert schedule.discharge_mw[:12].sum() == pytest.approx(0, abs=1e-9)
    assert schedule.discharge_mw[12:].sum() > 0
    assert schedule.revenue_eur > 0


def test_power_limit_is_respected(battery):
    schedule = optimise_day(CHEAP_THEN_DEAR, battery)

    assert schedule.charge_mw.max() <= battery.power_mw + 1e-9
    assert schedule.discharge_mw.max() <= battery.power_mw + 1e-9


def test_energy_capacity_is_respected(battery):
    schedule = optimise_day(CHEAP_THEN_DEAR, battery)

    assert schedule.state_of_charge_mwh.min() >= -1e-9
    assert schedule.state_of_charge_mwh.max() <= battery.energy_mwh + 1e-9


def test_the_day_ends_where_it_started(battery):
    """Otherwise revenue could come from draining the battery and never refilling."""
    schedule = optimise_day(CHEAP_THEN_DEAR, battery)
    assert schedule.state_of_charge_mwh[-1] == pytest.approx(battery.energy_mwh / 2, abs=1e-9)


def test_flat_prices_yield_no_revenue(battery):
    """With nothing to arbitrage, round-trip losses make trading a loss."""
    schedule = optimise_day(np.full(24, 50.0), battery)
    assert schedule.revenue_eur == pytest.approx(0, abs=1e-9)


def test_a_wider_spread_earns_more(battery):
    narrow = optimise_day(np.array([40.0] * 12 + [60.0] * 12), battery)
    wide = optimise_day(np.array([10.0] * 12 + [200.0] * 12), battery)

    assert wide.revenue_eur > narrow.revenue_eur


def test_round_trip_losses_reduce_revenue():
    lossless = Battery(energy_mwh=0.01, power_mw=0.005, round_trip_efficiency=1.0)
    lossy = Battery(energy_mwh=0.01, power_mw=0.005, round_trip_efficiency=0.80)

    assert (
        optimise_day(CHEAP_THEN_DEAR, lossless).revenue_eur
        > optimise_day(CHEAP_THEN_DEAR, lossy).revenue_eur
    )


def test_negative_prices_are_exploited(battery):
    """Being paid to charge is an opportunity, not an error."""
    prices = np.array([-50.0] * 6 + [30.0] * 18)
    schedule = optimise_day(prices, battery)

    assert schedule.charge_mw[:6].sum() > 0
    assert schedule.revenue_eur > 0


@pytest.mark.parametrize(
    "prices",
    [
        np.array([-50.0] * 6 + [30.0] * 18),
        np.array([-5.0] * 24),
        np.concatenate([np.full(12, -3.0), np.full(12, 180.0)]),
    ],
    ids=["negative_then_positive", "all_negative", "negative_then_peak"],
)
def test_it_never_charges_and_discharges_at_once(battery, prices):
    """A real inverter cannot do both, and negative prices tempt the solver to try.

    Being paid to consume makes circulating energy through the battery profitable
    on paper: the round-trip losses are the "consumption" being paid for. The
    shared inverter limit forbids it.
    """
    schedule = optimise_day(prices, battery)
    both = (schedule.charge_mw > 1e-9) & (schedule.discharge_mw > 1e-9)

    assert not both.any(), f"simultaneous operation in hours {np.flatnonzero(both)}"


def test_combined_throughput_respects_the_inverter(battery):
    prices = np.concatenate([np.full(12, -3.0), np.full(12, 180.0)])
    schedule = optimise_day(prices, battery)

    combined = schedule.charge_mw + schedule.discharge_mw
    assert combined.max() <= battery.power_mw + 1e-9


def test_nan_prices_are_refused(battery):
    prices = CHEAP_THEN_DEAR.copy()
    prices[5] = np.nan

    with pytest.raises(ValueError, match="NaN"):
        optimise_day(prices, battery)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"energy_mwh": 0},
        {"power_mw": -1},
        {"round_trip_efficiency": 0},
        {"round_trip_efficiency": 1.5},
    ],
)
def test_invalid_batteries_are_refused(kwargs):
    with pytest.raises(ValueError):
        Battery(**kwargs)


# --- series -----------------------------------------------------------------


@pytest.fixture
def prices() -> pd.Series:
    """Ten whole days, aligned to Berlin midnight.

    Days are grouped in market time, so a UTC-aligned series would start and end
    mid-day and yield nine complete days rather than ten.
    """
    index = pd.date_range(
        "2025-01-01", periods=24 * 10, freq="h", tz="Europe/Berlin"
    ).tz_convert("UTC")
    return pd.Series(np.tile(CHEAP_THEN_DEAR, 10), index=index)


def test_series_dispatch_returns_one_row_per_day(prices, battery):
    daily = optimise_series(prices, battery)
    assert len(daily) == 10
    assert (daily["revenue_eur"] > 0).all()


def test_partial_days_are_skipped_not_padded(prices, battery):
    truncated = prices.iloc[:-5]
    daily = optimise_series(truncated, battery)
    assert len(daily) == 9, "the incomplete final day is dropped"


def test_planning_on_a_forecast_settles_at_actual_prices(prices, battery):
    """Perfect foresight is an upper bound; planning on anything else earns less."""
    perfect = optimise_series(prices, battery)

    # A forecast that inverts the daily shape schedules exactly wrongly.
    misleading = pd.Series(prices.to_numpy()[::-1], index=prices.index)
    misled = optimise_series(prices, battery, decision_prices=misleading)

    assert misled["revenue_eur"].sum() < perfect["revenue_eur"].sum()


def test_annualise_reports_cycles_against_capacity(prices, battery):
    summary = annualise(optimise_series(prices, battery), battery)

    assert summary["days"] == 10
    assert summary["annual_eur"] == pytest.approx(summary["mean_daily_eur"] * 365.25)
    # One charge and one discharge of the full battery is one cycle.
    assert 0 < summary["cycles_per_day"] <= 4
    assert summary["cycles_per_year"] == pytest.approx(summary["cycles_per_day"] * 365.25)


def test_annualise_refuses_an_empty_result(battery):
    with pytest.raises(ValueError, match="no days"):
        annualise(pd.DataFrame(), battery)