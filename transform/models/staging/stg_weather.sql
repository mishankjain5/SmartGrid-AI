MODEL (
  name staging.stg_weather,
  kind FULL,
  description 'Archived hourly weather forecasts for the German centroid, at two lead times.',
  audits (
    assert_no_duplicate_timestamps,
    assert_hourly_grid_is_unbroken
  )
);

/*
  From the Open-Meteo historical *forecast* archive, not reanalysis: these are
  values the weather models predicted at the time, not a reconstruction assembled
  afterwards. Already hourly and gap-free, so this is a typed passthrough.

  Two lead times are carried, and the distinction decides whether a downstream
  model is honest:

  * `*_day_ahead` — issued a day before the valid time. This is the lead a
    forecast gated at 12:00 the previous day actually has, and these are the only
    weather columns a day-ahead model may use.
  * the unsuffixed columns — the archive's default, stitched from the earliest
    hours of successive model runs, so a very short lead. Using them in a
    day-ahead model would be leakage. They are kept solely to quantify what the
    lead time costs.

  total_radiation is derived so direct and diffuse always sum to the total used
  downstream, rather than relying on GHI agreeing with its own components.
*/
SELECT
  utc_timestamp,

  -- Day-ahead lead: legitimate features.
  temperature_c_day_ahead,
  ghi_wm2_day_ahead,
  direct_radiation_wm2_day_ahead,
  diffuse_radiation_wm2_day_ahead,
  direct_radiation_wm2_day_ahead
    + diffuse_radiation_wm2_day_ahead AS total_radiation_wm2_day_ahead,
  cloud_cover_pct_day_ahead,

  -- Short lead: comparison only, never a model input.
  temperature_c,
  ghi_wm2,
  direct_radiation_wm2,
  diffuse_radiation_wm2,
  direct_radiation_wm2 + diffuse_radiation_wm2 AS total_radiation_wm2,
  cloud_cover_pct
FROM raw.weather
