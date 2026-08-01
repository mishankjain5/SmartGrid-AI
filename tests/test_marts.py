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
    shorter. Comparing row counts between staging models only catches that
    indirectly; this checks the join itself.

    One unmatched hour is expected and permanent: Energy-Charts reports from
    Berlin midnight (23:00 UTC the previous day) while Open-Meteo starts at
    00:00 UTC, so the first hour of the series has no weather.
    """
    unmatched = query(
        f"""
        SELECT COUNT(*) AS n
        FROM `{PROJECT}.staging.stg_generation` AS g
        LEFT JOIN `{PROJECT}.staging.stg_weather` AS w USING (utc_timestamp)
        WHERE w.utc_timestamp IS NULL
        """
    ).loc[0, "n"]

    assert unmatched <= 2, (
        f"{unmatched} generation hours have no weather; a source has fallen behind"
    )

    retained = query(
        f"""
        SELECT (SELECT COUNT(*) FROM {MART}) AS mart,
               (SELECT COUNT(*) FROM `{PROJECT}.staging.stg_generation`) AS generation
        """
    ).loc[0]
    assert retained["mart"] >= retained["generation"] - 2


def test_horizons_span_the_day_ahead_window():
    frame = query(
        f"SELECT horizon_hours, COUNT(*) AS n FROM {MART} GROUP BY 1 ORDER BY 1"
    ).set_index("horizon_hours")["n"]

    assert frame.index.min() == 12
    assert frame.index.max() == 36
    # Every ordinary horizon appears on most days of the window.
    assert frame.loc[12:35].min() > 1500


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
    """Capacity nearly doubled; the capacity factor should not follow it."""
    frame = query(
        f"""
        SELECT EXTRACT(YEAR FROM local_datetime) AS yr,
               AVG(solar_ac_mw) AS capacity,
               AVG(solar_mw) AS generation,
               AVG(solar_capacity_factor) AS cf
        FROM {MART} GROUP BY 1 ORDER BY 1
        """
    ).set_index("yr")

    growth_in_capacity = frame["capacity"].iloc[-1] / frame["capacity"].iloc[0]
    spread_in_cf = frame["cf"].max() / frame["cf"].min()

    assert growth_in_capacity > 1.7
    assert spread_in_cf < 1.4, "the normalised target should be far more stable"


def test_only_day_ahead_weather_reaches_the_mart():
    """Short-lead weather exists in staging and must not appear here."""
    columns = set(
        query(
            f"""
            SELECT column_name
            FROM `{PROJECT}.marts.INFORMATION_SCHEMA.COLUMNS`
            WHERE table_name = 'mart_solar_features'
            """
        )["column_name"]
    )

    weather = {c for c in columns if "radiation" in c or "cloud" in c or "ghi" in c}
    assert weather, "expected weather features"
    assert all(c.endswith("_day_ahead") for c in weather), sorted(weather)


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
        WHERE ghi_wm2_day_ahead > 0
          AND tso_solar_forecast_capacity_factor IS NOT NULL
          AND capacity_factor_lag_48h IS NOT NULL
        """
    ).loc[0]

    assert frame["tso"] < frame["lag48"] / 2, "the benchmark should be a real bar"
