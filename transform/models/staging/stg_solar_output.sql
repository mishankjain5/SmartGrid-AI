MODEL (
  name staging.stg_solar_output,
  kind FULL,
  description 'Measured solar output as a capacity factor, with capacity carried forward.',
  audits (
    assert_no_duplicate_timestamps,
    assert_hourly_grid_is_unbroken
  )
);

/*
  The single definition of "how much of installed capacity was actually
  generating". The feature mart, the live scoring path and the lag features all
  need this, and computing it in three places invites them to disagree.

  Capacity is published monthly and lags: generation for the current month
  arrives before the capacity figure for it does. An exact month join therefore
  leaves the newest days with no denominator, silently dropping exactly the rows
  a live forecast depends on. The last known value is carried forward instead —
  installed capacity moves about 1% a month and only upward, so a stale figure
  understates growth slightly while a missing one loses the day entirely.
*/
WITH joined AS (
  SELECT
    g.utc_timestamp,
    DATETIME(g.utc_timestamp, 'Europe/Berlin') AS local_datetime,
    g.solar_mw,
    c.solar_ac_mw AS reported_capacity_mw
  FROM staging.stg_generation AS g
  LEFT JOIN staging.stg_capacity AS c
    ON DATE_TRUNC(DATE(DATETIME(g.utc_timestamp, 'Europe/Berlin')), MONTH)
       = DATE(c.month)
),

filled AS (
  SELECT
    *,
    LAST_VALUE(reported_capacity_mw IGNORE NULLS) OVER (
      ORDER BY utc_timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS solar_ac_mw
  FROM joined
)

SELECT
  utc_timestamp,
  local_datetime,
  solar_mw,
  solar_ac_mw,
  SAFE_DIVIDE(solar_mw, solar_ac_mw) AS solar_capacity_factor,
  reported_capacity_mw IS NULL AS capacity_carried_forward
FROM filled
