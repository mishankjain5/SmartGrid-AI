MODEL (
  name staging.stg_generation,
  kind FULL,
  description 'Hourly German generation and load, pivoted from long form.',
  audits (
    assert_no_duplicate_timestamps,
    assert_hourly_grid_is_unbroken,
    assert_no_partial_hours
  )
);

/*
  raw.public_power is quarter-hourly for the whole period and long-form, one row
  per timestamp and production type. This averages to hourly and pivots the
  series the project uses into columns.

  Averaging MW over the hour gives mean power, which for an hourly interval is
  numerically equal to the energy in MWh.

  `Load` is carried here rather than in a separate model because it arrives in
  the same source table on the same grid.
*/
SELECT
  TIMESTAMP_TRUNC(utc_timestamp, HOUR) AS utc_timestamp,
  AVG(IF(production_type = 'Solar', power_mw, NULL)) AS solar_mw,
  AVG(IF(production_type = 'Wind onshore', power_mw, NULL)) AS wind_onshore_mw,
  AVG(IF(production_type = 'Wind offshore', power_mw, NULL)) AS wind_offshore_mw,
  AVG(IF(production_type = 'Load', power_mw, NULL)) AS load_mw,
  AVG(IF(production_type = 'Residual load', power_mw, NULL)) AS residual_load_mw,
  COUNTIF(production_type = 'Solar') AS source_intervals
FROM raw.public_power
GROUP BY 1
