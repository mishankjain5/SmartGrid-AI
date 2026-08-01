"""Checks on the solar feature mart.

Per-row invariants — gate before target, horizon in range, capacity factor
physical — are SQLMesh audits and run on every build. These cover the properties
that need a query to express.
"""

import pytest

from smartgrid.config import get_settings
from smartgrid.warehouse import query

pytestmark = pytest.mark.skipif(
    not get_settings().bigquery_available,
    reason="GCP_PROJECT not set",
)

PROJECT = get_settings().gcp_project
MART = f"`{PROJECT}.marts.mart_solar_features`"


def test_the_join_does_not_silently_drop_generation_hours():
    """The guard that matters.

    The mart inner-joins generation against weather. If a source fell behind,
    training hours would disappear with no error — the table would simply be
    shorter. Comparing row counts between staging models catches that only
    indirectly; this checks the join itself.

    Scoped to the period weather covers. Weather deliberately starts later than
    generation: Open-Meteo archives day-ahead radiation only from 2024-01-19, so
    earlier hours could never produce a usable feature row. Within that window,
    every generation hour must survive.
    """
    unmatched = query(
        f"""
        WITH covered AS (
          SELECT MIN(utc_timestamp) AS first_ts, MAX(utc_timestamp) AS last_ts
          FROM `{PROJECT}.staging.stg_weather_national`
        )
        SELECT COUNT(*) AS n
        FROM `{PROJECT}.staging.stg_generation` AS g
        CROSS JOIN covered AS c
        LEFT JOIN `{PROJECT}.staging.stg_weather_national` AS w USING (utc_timestamp)
        WHERE w.utc_timestamp IS NULL
          AND g.utc_timestamp BETWEEN c.first_ts AND c.last_ts
        """
    ).loc[0, "n"]

    assert unmatched == 0, (
        f"{unmatched} generation hours inside the weather window have no weather; "
        "a source has fallen behind"
    )


def test_the_mart_covers_the_whole_weather_window():
    """A shrinking mart is the symptom a row-count tolerance would miss."""
    frame = query(
        f"""
        SELECT
          (SELECT COUNT(*) FROM {MART}) AS mart_rows,
          (SELECT COUNT(*) FROM `{PROJECT}.staging.stg_weather_national`) AS weather_rows
        """
    ).loc[0]

    # Weather may run a little past generation at the tail; nothing more.
    assert frame["mart_rows"] >= frame["weather_rows"] - 48


def test_horizons_span_the_day_ahead_window():
    frame = query(
        f"SELECT horizon_hours, COUNT(*) AS n FROM {MART} GROUP BY 1 ORDER BY 1"
    ).set_index("horizon_hours")["n"]

    assert frame.index.min() == 12
    assert frame.index.max() == 36
    # Every ordinary horizon appears on most days of the ~2.5 year window.
    assert frame.loc[12:35].min() > 900


def test_the_thirty_six_hour_horizon_is_rare_and_seasonal():
    """One 25-hour day a year, at the autumn clock change."""
    frame = query(
        f"""
        SELECT EXTRACT(MONTH FROM local_datetime) AS month_of_year, COUNT(*) AS n
        FROM {MART} WHERE horizon_hours = 36 GROUP BY 1
        """
    )
    assert set(frame["month_of_year"]) == {10}
    assert frame["n"].sum() < 10


def test_capacity_factor_is_bounded_and_complete():
    frame = query(
        f"""
        SELECT MIN(solar_capacity_factor) AS lo,
               MAX(solar_capacity_factor) AS hi,
               COUNTIF(solar_capacity_factor IS NULL) AS missing
        FROM {MART}
        """
    ).loc[0]

    assert frame["lo"] == 0
    assert 0.3 < frame["hi"] < 1.0
    assert frame["missing"] == 0


def test_normalising_removes_the_deployment_trend():
    """Capacity keeps climbing; the capacity factor should not follow it.

    Stated as a ratio between the two rather than as absolute thresholds, so the
    check survives a change to the date window. Over 2024-2026 capacity rises
    about 32% while the capacity factor moves about 20% — and in the opposite
    direction, since 2024 and 2025 were duller than 2026.
    """
    frame = query(
        f"""
        SELECT EXTRACT(YEAR FROM local_datetime) AS yr,
               AVG(solar_ac_mw) AS capacity,
               AVG(solar_capacity_factor) AS cf
        FROM {MART} GROUP BY 1 ORDER BY 1
        """
    ).set_index("yr")

    capacity_growth = frame["capacity"].iloc[-1] / frame["capacity"].iloc[0] - 1
    cf_drift = frame["cf"].max() / frame["cf"].min() - 1

    assert capacity_growth > 0.2, "capacity should be visibly trending"
    assert cf_drift < capacity_growth, "the normalised target must be the steadier of the two"


def test_weather_in_the_mart_covers_multiple_locations():
    """A single point cannot represent output spread over 800 km."""
    columns = set(
        query(
            f"""
            SELECT column_name
            FROM `{PROJECT}.marts.INFORMATION_SCHEMA.COLUMNS`
            WHERE table_name = 'mart_solar_features'
            """
        )["column_name"]
    )

    per_location = {c for c in columns if c.startswith("ghi_")} - {
        "ghi_mean", "ghi_min", "ghi_max", "ghi_stddev", "ghi_spread"
    }
    assert len(per_location) >= 8, sorted(per_location)
    assert {"ghi_stddev", "ghi_spread"} <= columns


def test_regional_weather_actually_diverges():
    """If locations never disagreed, sampling several would be pointless."""
    frame = query(
        f"""
        SELECT AVG(ghi_spread) AS mean_spread, MAX(ghi_spread) AS max_spread
        FROM {MART} WHERE ghi_max > 100
        """
    ).loc[0]

    assert frame["mean_spread"] > 50
    assert frame["max_spread"] > 400


def test_lags_are_whole_days_beyond_the_safe_bound():
    columns = set(
        query(
            f"""
            SELECT column_name
            FROM `{PROJECT}.marts.INFORMATION_SCHEMA.COLUMNS`
            WHERE table_name = 'mart_solar_features' AND column_name LIKE '%lag%'
            """
        )["column_name"]
    )

    assert columns == {
        "capacity_factor_lag_48h",
        "capacity_factor_lag_72h",
        "capacity_factor_lag_168h",
    }
    # 24h would be leaky for late target hours; 36h is the bound, 48h the first
    # whole day clearing it.
    assert not any("lag_24h" in c for c in columns)


def test_lag_48h_matches_the_value_two_days_earlier():
    mismatches = query(
        f"""
        WITH joined AS (
          SELECT a.utc_timestamp, a.capacity_factor_lag_48h, b.solar_capacity_factor AS actual
          FROM {MART} AS a
          JOIN {MART} AS b
            ON b.utc_timestamp = TIMESTAMP_SUB(a.utc_timestamp, INTERVAL 48 HOUR)
        )
        SELECT COUNT(*) AS n FROM joined
        WHERE ABS(capacity_factor_lag_48h - actual) > 1e-9
        """
    ).loc[0, "n"]

    assert mismatches == 0


def test_tso_benchmark_beats_a_naive_lag():
    """Establishes the bar: the operators' forecast is much better than persistence."""
    frame = query(
        f"""
        SELECT AVG(ABS(tso_solar_forecast_capacity_factor - solar_capacity_factor)) AS tso,
               AVG(ABS(capacity_factor_lag_48h - solar_capacity_factor)) AS lag48
        FROM {MART}
        WHERE ghi_max > 0
          AND tso_solar_forecast_capacity_factor IS NOT NULL
          AND capacity_factor_lag_48h IS NOT NULL
        """
    ).loc[0]

    assert frame["tso"] < frame["lag48"] / 2, "the benchmark should be a real bar"
