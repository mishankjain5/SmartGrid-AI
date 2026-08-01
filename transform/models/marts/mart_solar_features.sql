MODEL (
  name marts.mart_solar_features,
  kind FULL,
  description 'Day-ahead solar forecasting features. Every column is available at the 12:00 gate.',
  audits (
    assert_no_duplicate_timestamps,
    assert_hourly_grid_is_unbroken,
    assert_gate_precedes_target,
    assert_horizon_within_day_ahead_window,
    assert_capacity_factor_is_physical
  )
);

/*
  TARGET. solar_capacity_factor, not raw MW. Installed capacity grew from roughly
  55 GW to 116 GW across this window, so raw generation carries a deployment
  trend that has nothing to do with weather. Capacity is monthly and AC, the
  inverter rating, because generation is metered at the grid.

  GATE. The German day-ahead auction closes at 12:00 Berlin time on day D and
  clears all 24 hours of D+1 at once. Every hour of the target day therefore
  shares one gate, and every feature must be computable at it. Horizons run 12 to
  36 hours.

  LAGS. A 24-hour lag is not safe. For the 00:00 target it points to 00:00 on day
  D, before the gate; for the 23:00 target it points to 23:00 on day D, eleven
  hours after the gate closed. The same nominal feature is legitimate early in the
  day and leaky late in it. Lags here are whole days at 48 hours and beyond, which
  clears the 36-hour bound and keeps the hour of day aligned.

  WEATHER. Only the `_day_ahead` columns are used, which were issued a day before
  the valid time. The short-lead columns in stg_weather are deliberately absent.

  BENCHMARK. tso_solar_forecast_mw is what the transmission operators published
  for the same hour. It is carried as a comparison, not a feature.
*/
WITH hours AS (
  SELECT
    g.utc_timestamp,
    DATETIME(g.utc_timestamp, 'Europe/Berlin') AS local_datetime,
    g.solar_mw,
    w.ghi_wm2_day_ahead,
    w.direct_radiation_wm2_day_ahead,
    w.diffuse_radiation_wm2_day_ahead,
    w.total_radiation_wm2_day_ahead,
    w.temperature_c_day_ahead,
    w.cloud_cover_pct_day_ahead,
    f.solar_forecast_mw AS tso_solar_forecast_mw
  FROM staging.stg_generation AS g
  INNER JOIN staging.stg_weather AS w USING (utc_timestamp)
  LEFT JOIN staging.stg_day_ahead_forecast AS f USING (utc_timestamp)
),

with_capacity AS (
  SELECT
    h.*,
    c.solar_ac_mw
  FROM hours AS h
  LEFT JOIN staging.stg_capacity AS c
    ON DATE_TRUNC(DATE(h.local_datetime), MONTH) = DATE(c.month)
),

/*
  The gate as an absolute instant. Built by taking noon on the day before the
  target's local date and converting that local wall-clock time back to UTC, so
  the clock changes land on the right instant rather than drifting by an hour.
*/
gated AS (
  SELECT
    *,
    TIMESTAMP(
      DATETIME(DATE_SUB(DATE(local_datetime), INTERVAL 1 DAY), TIME '12:00:00'),
      'Europe/Berlin'
    ) AS gate_utc
  FROM with_capacity
)

SELECT
  utc_timestamp,
  local_datetime,
  gate_utc,
  TIMESTAMP_DIFF(utc_timestamp, gate_utc, HOUR) AS horizon_hours,

  -- Target
  solar_mw,
  solar_ac_mw,
  SAFE_DIVIDE(solar_mw, solar_ac_mw) AS solar_capacity_factor,

  -- Calendar
  EXTRACT(HOUR FROM local_datetime) AS hour_of_day,
  EXTRACT(DAYOFWEEK FROM local_datetime) AS day_of_week,
  EXTRACT(DAYOFYEAR FROM local_datetime) AS day_of_year,
  EXTRACT(MONTH FROM local_datetime) AS month_of_year,

  -- Cyclical encodings so 23:00 sits beside 00:00 rather than 23 apart
  SIN(2 * ACOS(-1) * EXTRACT(HOUR FROM local_datetime) / 24) AS hour_sin,
  COS(2 * ACOS(-1) * EXTRACT(HOUR FROM local_datetime) / 24) AS hour_cos,
  SIN(2 * ACOS(-1) * EXTRACT(DAYOFYEAR FROM local_datetime) / 365.25) AS day_of_year_sin,
  COS(2 * ACOS(-1) * EXTRACT(DAYOFYEAR FROM local_datetime) / 365.25) AS day_of_year_cos,

  -- Weather at day-ahead lead
  ghi_wm2_day_ahead,
  direct_radiation_wm2_day_ahead,
  diffuse_radiation_wm2_day_ahead,
  total_radiation_wm2_day_ahead,
  temperature_c_day_ahead,
  cloud_cover_pct_day_ahead,

  -- Gate-safe lags of the target, in whole days
  LAG(SAFE_DIVIDE(solar_mw, solar_ac_mw), 48) OVER w AS capacity_factor_lag_48h,
  LAG(SAFE_DIVIDE(solar_mw, solar_ac_mw), 72) OVER w AS capacity_factor_lag_72h,
  LAG(SAFE_DIVIDE(solar_mw, solar_ac_mw), 168) OVER w AS capacity_factor_lag_168h,

  -- Rolling means ending at the last hour that clears the gate for every target
  -- hour of the day. 48 rather than 36 keeps the window on a day boundary.
  AVG(SAFE_DIVIDE(solar_mw, solar_ac_mw)) OVER (
    ORDER BY utc_timestamp ROWS BETWEEN 215 PRECEDING AND 48 PRECEDING
  ) AS capacity_factor_mean_7d,
  AVG(SAFE_DIVIDE(solar_mw, solar_ac_mw)) OVER (
    ORDER BY utc_timestamp ROWS BETWEEN 767 PRECEDING AND 48 PRECEDING
  ) AS capacity_factor_mean_30d,

  -- Benchmark, not a feature
  tso_solar_forecast_mw,
  SAFE_DIVIDE(tso_solar_forecast_mw, solar_ac_mw) AS tso_solar_forecast_capacity_factor

FROM gated
WINDOW w AS (ORDER BY utc_timestamp)
