MODEL (
  name marts.mart_household_features,
  kind FULL,
  description 'Household consumption with calendar features, for load profile modelling.'
);

/*
  Deliberately calendar-only, and the reason is a coverage gap rather than a
  modelling choice: the OPSD household panel ends in 2018 while the weather
  archive used elsewhere in this project begins in 2024. There is no overlap to
  join on.

  So this cannot capture temperature-driven demand, which for a house with a
  heat pump is a large part of winter load. What it does capture is the daily and
  weekly rhythm — when people wake, cook and go out — which is what a standard
  load profile encodes and is enough to plan a battery against.

  The limitation is worth stating rather than hiding: a January forecast from
  this model will miss a cold snap entirely.

  Lags respect the same 48-hour gate bound as the solar features, so a profile
  built here could be produced at the same moment a day-ahead schedule is.
*/
SELECT
  utc_timestamp,
  local_datetime,
  building,
  building_type,
  has_pv,

  consumption_kwh,
  grid_import_kwh,
  pv_kwh,
  heat_pump_kwh,
  ev_kwh,

  EXTRACT(HOUR FROM local_datetime) AS hour_of_day,
  EXTRACT(DAYOFWEEK FROM local_datetime) AS day_of_week,
  EXTRACT(DAYOFYEAR FROM local_datetime) AS day_of_year,
  EXTRACT(MONTH FROM local_datetime) AS month_of_year,
  EXTRACT(DAYOFWEEK FROM local_datetime) IN (1, 7) AS is_weekend,

  SIN(2 * ACOS(-1) * EXTRACT(HOUR FROM local_datetime) / 24) AS hour_sin,
  COS(2 * ACOS(-1) * EXTRACT(HOUR FROM local_datetime) / 24) AS hour_cos,
  SIN(2 * ACOS(-1) * EXTRACT(DAYOFYEAR FROM local_datetime) / 365.25) AS day_of_year_sin,
  COS(2 * ACOS(-1) * EXTRACT(DAYOFYEAR FROM local_datetime) / 365.25) AS day_of_year_cos,

  LAG(consumption_kwh, 48) OVER w AS consumption_lag_48h,
  LAG(consumption_kwh, 168) OVER w AS consumption_lag_168h,
  AVG(consumption_kwh) OVER (
    PARTITION BY building
    ORDER BY utc_timestamp ROWS BETWEEN 215 PRECEDING AND 48 PRECEDING
  ) AS consumption_mean_7d

FROM staging.stg_household_load
WHERE consumption_kwh IS NOT NULL
  AND consumption_kwh >= 0
WINDOW w AS (PARTITION BY building ORDER BY utc_timestamp)
