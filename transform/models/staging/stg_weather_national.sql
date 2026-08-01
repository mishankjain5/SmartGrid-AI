MODEL (
  name staging.stg_weather_national,
  kind FULL,
  description 'Hourly weather across Germany: per-location irradiance plus national aggregates.',
  audits (
    assert_no_duplicate_timestamps,
    assert_hourly_grid_is_unbroken
  )
);

/*
  One row per hour, collapsing the nine locations two ways.

  The aggregates carry most of the value. Averaging nine points estimates
  national conditions far better than any single point: replacing one centroid
  reading with these columns cut model error by roughly 40%. `ghi_mean` and
  `direct_radiation_mean` are the strongest weather features by permutation
  importance.

  Per-location columns are kept as well, but they earn less than expected —
  removing them costs about 4% of accuracy. Conditions do diverge sharply within
  an hour (on 2025-06-10 at midday the archive gives 84 W/m² near Hamburg against
  854 near Munich), yet knowing which end was dark adds little once the average
  is right. They are retained because 4% is not nothing and they cost nothing at
  inference, not because regional structure turned out to be the main effect.

  Only the day-ahead lead is exposed. The short-lead columns stay in stg_weather.
*/
SELECT
  utc_timestamp,

  -- Per location, so regional structure survives.
  MAX(IF(location = 'hamburg', ghi_wm2_day_ahead, NULL)) AS ghi_hamburg,
  MAX(IF(location = 'rostock', ghi_wm2_day_ahead, NULL)) AS ghi_rostock,
  MAX(IF(location = 'hanover', ghi_wm2_day_ahead, NULL)) AS ghi_hanover,
  MAX(IF(location = 'berlin', ghi_wm2_day_ahead, NULL)) AS ghi_berlin,
  MAX(IF(location = 'cologne', ghi_wm2_day_ahead, NULL)) AS ghi_cologne,
  MAX(IF(location = 'kassel', ghi_wm2_day_ahead, NULL)) AS ghi_kassel,
  MAX(IF(location = 'nuremberg', ghi_wm2_day_ahead, NULL)) AS ghi_nuremberg,
  MAX(IF(location = 'stuttgart', ghi_wm2_day_ahead, NULL)) AS ghi_stuttgart,
  MAX(IF(location = 'munich', ghi_wm2_day_ahead, NULL)) AS ghi_munich,

  -- National aggregates.
  AVG(ghi_wm2_day_ahead) AS ghi_mean,
  MIN(ghi_wm2_day_ahead) AS ghi_min,
  MAX(ghi_wm2_day_ahead) AS ghi_max,
  STDDEV_POP(ghi_wm2_day_ahead) AS ghi_stddev,
  MAX(ghi_wm2_day_ahead) - MIN(ghi_wm2_day_ahead) AS ghi_spread,

  AVG(direct_radiation_wm2_day_ahead) AS direct_radiation_mean,
  AVG(diffuse_radiation_wm2_day_ahead) AS diffuse_radiation_mean,
  AVG(temperature_c_day_ahead) AS temperature_mean,
  AVG(cloud_cover_pct_day_ahead) AS cloud_cover_mean,
  STDDEV_POP(cloud_cover_pct_day_ahead) AS cloud_cover_stddev,

  COUNT(ghi_wm2_day_ahead) AS locations_reporting
FROM staging.stg_weather
GROUP BY utc_timestamp