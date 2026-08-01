MODEL (
  name staging.stg_day_ahead_forecast,
  kind FULL,
  description "The transmission operators' published day-ahead forecasts, hourly.",
  audits (
    assert_no_duplicate_timestamps,
    assert_hourly_grid_is_unbroken,
    assert_no_partial_hours
  )
);

/*
  These are operational forecasts published before delivery, not values computed
  after the fact. They are the benchmark this project's own models are scored
  against: beating the transmission operators' forecast is a meaningful claim,
  where beating a seasonal-naive baseline is not.

  Resampled to hourly on the same basis as stg_generation so the two align.
*/
SELECT
  TIMESTAMP_TRUNC(utc_timestamp, HOUR) AS utc_timestamp,
  AVG(IF(production_type = 'solar', forecast_mw, NULL)) AS solar_forecast_mw,
  AVG(IF(production_type = 'load', forecast_mw, NULL)) AS load_forecast_mw,
  AVG(IF(production_type = 'wind_onshore', forecast_mw, NULL)) AS wind_onshore_forecast_mw,
  AVG(IF(production_type = 'wind_offshore', forecast_mw, NULL)) AS wind_offshore_forecast_mw,
  COUNTIF(production_type = 'solar') AS source_intervals
FROM raw.day_ahead_forecast
GROUP BY 1
-- Ingestion runs to the current day, so the final hour is often incomplete.
-- Dropped until all its intervals exist; see stg_price for the reasoning.
HAVING COUNTIF(production_type = 'solar') IN (1, 4)
