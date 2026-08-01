-- Audits specific to day-ahead feature tables.

AUDIT (
  name assert_gate_precedes_target
);
-- The rule everything rests on: the forecast is issued before the hour it
-- predicts. A row failing this means a feature could see its own answer.
SELECT
  utc_timestamp,
  gate_utc,
  horizon_hours
FROM @this_model
WHERE gate_utc >= utc_timestamp;


AUDIT (
  name assert_horizon_within_day_ahead_window
);
-- 12 hours at the earliest, 36 at the latest. 36 rather than 35 because the
-- autumn clock change makes one day of the year 25 hours long, stretching the
-- gap between a noon gate and the final hour of the next day.
SELECT
  utc_timestamp,
  horizon_hours
FROM @this_model
WHERE horizon_hours < 12
   OR horizon_hours > 36;


AUDIT (
  name assert_capacity_factor_is_physical
);
-- Generation divided by inverter capacity cannot exceed 1, and never goes
-- negative. A breach means the capacity join picked the wrong month or the wrong
-- rating (DC instead of AC).
SELECT
  utc_timestamp,
  solar_capacity_factor
FROM @this_model
WHERE solar_capacity_factor < 0
   OR solar_capacity_factor > 1;
