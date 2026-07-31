-- Audits fail the run when their query returns rows, so each selects the
-- offending records rather than a pass/fail flag.

AUDIT (
  name assert_no_duplicate_timestamps
);
SELECT
  utc_timestamp,
  COUNT(*) AS occurrences
FROM @this_model
GROUP BY utc_timestamp
HAVING COUNT(*) > 1;


AUDIT (
  name assert_hourly_grid_is_unbroken
);
-- Downstream lag features index by position, which only equals a time-based lag
-- when the grid has no holes.
WITH steps AS (
  SELECT
    utc_timestamp,
    TIMESTAMP_DIFF(
      utc_timestamp,
      LAG(utc_timestamp) OVER (ORDER BY utc_timestamp),
      MINUTE
    ) AS step_minutes
  FROM @this_model
)
SELECT *
FROM steps
WHERE step_minutes IS NOT NULL
  AND step_minutes != 60;


AUDIT (
  name assert_no_partial_hours
);
-- Resampling to hourly averages the sub-hourly intervals inside each hour. An
-- hour built from fewer intervals than its neighbours is a truncated boundary
-- or a source gap, not a normal value.
SELECT
  utc_timestamp,
  source_intervals
FROM @this_model
WHERE source_intervals NOT IN (1, 4);
