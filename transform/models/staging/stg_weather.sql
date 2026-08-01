MODEL (
  name staging.stg_weather,
  kind FULL,
  description 'Archived hourly weather forecasts at nine German locations, two lead times.'
);

/*
  From the Open-Meteo historical *forecast* archive, not reanalysis: values the
  weather models predicted at the time, not a reconstruction assembled
  afterwards.

  Long form, one row per hour and location. Two lead times are carried, and the
  distinction decides whether a downstream model is honest:

  * `*_day_ahead` — issued a day before the valid time. This is the lead a
    forecast gated at noon the previous day actually has, and the only weather a
    day-ahead model may use.
  * the unsuffixed columns — the archive default, stitched from the earliest
    hours of successive model runs, so a very short lead. Using them in a
    day-ahead model would be leakage. Kept only to quantify what lead time costs.

  Day-ahead radiation is archived from 2024-01-19 onward; earlier rows carry the
  short lead only.
*/
SELECT
  utc_timestamp,
  location,
  latitude,
  longitude,

  -- Day-ahead lead: legitimate features.
  temperature_c_day_ahead,
  ghi_wm2_day_ahead,
  direct_radiation_wm2_day_ahead,
  diffuse_radiation_wm2_day_ahead,
  cloud_cover_pct_day_ahead,

  -- Short lead: comparison only, never a model input.
  temperature_c,
  ghi_wm2,
  direct_radiation_wm2,
  diffuse_radiation_wm2,
  cloud_cover_pct
FROM raw.weather
