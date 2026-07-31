MODEL (
  name staging.stg_price,
  kind FULL,
  description 'German day-ahead price at a single hourly resolution.',
  audits (
    assert_no_duplicate_timestamps,
    assert_hourly_grid_is_unbroken,
    assert_no_partial_hours
  )
);

/*
  The source resolution is not constant. German day-ahead products moved from
  hourly to quarter-hourly on 2025-10-01, so the raw series is 60-minute before
  that date and 15-minute after it. Joining that against hourly weather without
  resampling would silently produce four rows per hour for recent years.

  Averaging is the correct aggregation: the hourly price is the mean of its
  quarter-hourly constituents. `source_intervals` records how many intervals
  each hour was built from, which the audits check.

  Negative prices are kept. They occur in roughly 5% of hours and are a real
  market outcome, not an error.
*/
SELECT
  TIMESTAMP_TRUNC(utc_timestamp, HOUR) AS utc_timestamp,
  AVG(price_eur_mwh) AS price_eur_mwh,
  COUNT(*) AS source_intervals
FROM raw.price
WHERE price_eur_mwh IS NOT NULL
GROUP BY 1
