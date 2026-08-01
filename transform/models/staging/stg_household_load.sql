MODEL (
  name staging.stg_household_load,
  kind FULL,
  description 'Total hourly consumption per household, reconstructed from meter channels.'
);

/*
  Grid import alone is not consumption. A house with panels covers part of its
  own demand directly, and that self-consumed energy never crosses the meter.
  Total demand is therefore:

      consumption = grid_import + pv_generation - grid_export

  For a house without panels the last two terms are absent and grid import is
  the whole story.

  This matters for battery planning. Sizing a battery against grid import alone
  understates midday demand — exactly when a PV system is covering it — and so
  understates how much a battery could shift.

  Buildings are kept separate rather than averaged: they differ by a factor of
  two in daily consumption, and a mean over a panel of eleven would describe
  none of them.
*/
WITH per_device AS (
  SELECT
    utc_timestamp,
    building,
    building_type,
    MAX(IF(device = 'grid_import', energy_kwh, NULL)) AS grid_import_kwh,
    MAX(IF(device = 'grid_export', energy_kwh, NULL)) AS grid_export_kwh,
    MAX(IF(device = 'pv', energy_kwh, NULL)) AS pv_kwh,
    MAX(IF(device = 'heat_pump', energy_kwh, NULL)) AS heat_pump_kwh,
    MAX(IF(device = 'ev', energy_kwh, NULL)) AS ev_kwh
  FROM staging.stg_household
  WHERE NOT is_implausible
  GROUP BY utc_timestamp, building, building_type
)

SELECT
  utc_timestamp,
  DATETIME(utc_timestamp, 'Europe/Berlin') AS local_datetime,
  building,
  building_type,
  grid_import_kwh,
  grid_export_kwh,
  pv_kwh,
  heat_pump_kwh,
  ev_kwh,
  grid_import_kwh
    + COALESCE(pv_kwh, 0)
    - COALESCE(grid_export_kwh, 0) AS consumption_kwh,
  pv_kwh IS NOT NULL AS has_pv
FROM per_device
WHERE grid_import_kwh IS NOT NULL
