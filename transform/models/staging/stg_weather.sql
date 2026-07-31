MODEL (
  name staging.stg_weather,
  kind FULL,
  description 'Archived hourly weather forecasts for the German centroid.',
  audits (
    assert_no_duplicate_timestamps,
    assert_hourly_grid_is_unbroken
  )
);

/*
  From the Open-Meteo historical *forecast* archive, not reanalysis: these are
  the values the weather models predicted at the time, so they were available to
  a forecaster before the gate closed. Already hourly and gap-free, so this is a
  typed passthrough.

  total_radiation_wm2 is derived rather than taken from shortwave_radiation so
  that the direct and diffuse components always sum to the total used downstream.
*/
SELECT
  utc_timestamp,
  temperature_c,
  ghi_wm2,
  direct_radiation_wm2,
  diffuse_radiation_wm2,
  direct_radiation_wm2 + diffuse_radiation_wm2 AS total_radiation_wm2,
  cloud_cover_pct
FROM raw.weather
