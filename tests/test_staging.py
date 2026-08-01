"""Checks across staging models.

Per-model invariants — no duplicate timestamps, unbroken hourly grid, no partial
hours — are enforced by SQLMesh audits and run on every `sqlmesh plan`. What is
left here is what a single-model audit cannot see: agreement between models, and
the outcome of the household meter conversion.

Requires the staging models to be built; skipped without credentials.
"""

import pytest

from smartgrid.config import get_settings
from smartgrid.warehouse import query

pytestmark = pytest.mark.skipif(
    not get_settings().bigquery_available,
    reason="GCP_PROJECT not set",
)

PROJECT = get_settings().gcp_project

HOURLY_MODELS = ("stg_price", "stg_weather", "stg_generation", "stg_day_ahead_forecast")


@pytest.fixture(scope="module")
def coverage():
    unions = "\nUNION ALL ".join(
        f"SELECT '{model}' AS model, COUNT(*) AS n, "
        f"MIN(utc_timestamp) AS first_ts, MAX(utc_timestamp) AS last_ts "
        f"FROM `{PROJECT}.staging.{model}`"
        for model in HOURLY_MODELS
    )
    return query(unions).set_index("model")


def test_every_hourly_model_covers_a_comparable_span(coverage):
    counts = coverage["n"]
    assert counts.min() > 40_000

    # Sources may be refreshed independently, so the newest table can extend a
    # day or two past the others, and the two APIs disagree about whether date
    # bounds are local or UTC. Anything beyond that is a truncated load rather
    # than a boundary difference. Joinability is asserted separately below.
    assert counts.max() - counts.min() <= 72


def test_models_overlap_enough_to_join(coverage):
    latest_start = coverage["first_ts"].max()
    earliest_end = coverage["last_ts"].min()
    span_days = (earliest_end - latest_start).days

    assert span_days > 1600, "expected roughly four and a half years of overlap"


def test_price_is_resampled_to_a_single_resolution():
    """The source switches from hourly to quarter-hourly on 2025-10-01."""
    frame = query(
        f"""
        SELECT source_intervals, COUNT(*) AS n
        FROM `{PROJECT}.staging.stg_price`
        GROUP BY 1 ORDER BY 1
        """
    )
    assert set(frame["source_intervals"]) <= {1, 4}
    # Both eras are present, so the resampling is actually doing work.
    assert len(frame) == 2


def test_negative_prices_survive_cleaning():
    negatives = query(
        f"SELECT COUNTIF(price_eur_mwh < 0) AS n FROM `{PROJECT}.staging.stg_price`"
    ).loc[0, "n"]
    assert negatives > 500, "negative prices are a real market outcome"


def test_solar_generation_is_zero_at_night_and_positive_at_midday():
    frame = query(
        f"""
        SELECT
          AVG(IF(EXTRACT(HOUR FROM utc_timestamp) = 2, solar_mw, NULL)) AS night_mw,
          AVG(IF(EXTRACT(HOUR FROM utc_timestamp) = 11, solar_mw, NULL)) AS midday_mw
        FROM `{PROJECT}.staging.stg_generation`
        """
    )
    assert frame.loc[0, "night_mw"] < 100
    assert frame.loc[0, "midday_mw"] > 5000


def test_capacity_reports_ac_below_dc():
    """AC is the inverter rating and the correct denominator for a capacity factor."""
    frame = query(
        f"""
        SELECT solar_ac_mw, solar_dc_mw
        FROM `{PROJECT}.staging.stg_capacity`
        WHERE solar_ac_mw IS NOT NULL AND solar_dc_mw IS NOT NULL
        ORDER BY month DESC LIMIT 1
        """
    )
    assert 0 < frame.loc[0, "solar_ac_mw"] < frame.loc[0, "solar_dc_mw"]


# --- household meter conversion ---------------------------------------------


@pytest.fixture(scope="module")
def household():
    return query(
        f"""
        SELECT
          COUNT(*) AS rows_total,
          COUNTIF(energy_kwh IS NOT NULL) AS with_energy,
          COUNTIF(is_implausible) AS flagged,
          MAX(meter_kwh) AS max_cumulative,
          MAX(IF(NOT is_implausible, energy_kwh, NULL)) AS max_hourly_kept,
          MIN(energy_kwh) AS min_hourly
        FROM `{PROJECT}.staging.stg_household`
        """
    ).loc[0]


def test_cumulative_readings_became_hourly_energy(household):
    # Meters run into the hundreds of thousands; an hour of use does not.
    assert household["max_cumulative"] > 100_000
    assert household["max_hourly_kept"] < 250


def test_no_negative_energy(household):
    """A decrease means a meter reset, which must yield NULL rather than a negative."""
    assert household["min_hourly"] >= 0


def test_gaps_yield_null_rather_than_a_fabricated_value(household):
    assert household["with_energy"] < household["rows_total"]


def test_meter_replacements_are_flagged_but_preserved(household):
    assert household["flagged"] > 0

    kept = query(
        f"""
        SELECT MAX(energy_kwh) AS worst
        FROM `{PROJECT}.staging.stg_household`
        WHERE is_implausible
        """
    ).loc[0, "worst"]
    assert kept > household["max_hourly_kept"], "flagged values are kept, not clipped"


def test_flagging_scales_per_channel():
    """A fixed threshold would delete real industrial load."""
    frame = query(
        f"""
        SELECT building_type, MAX(IF(NOT is_implausible, energy_kwh, NULL)) AS kept
        FROM `{PROJECT}.staging.stg_household`
        GROUP BY 1
        """
    ).set_index("building_type")

    assert frame.loc["industrial", "kept"] > 100
    assert frame.loc["residential", "kept"] < frame.loc["industrial", "kept"]
