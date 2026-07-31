MODEL (
  name staging.stg_household,
  kind FULL,
  description 'Household channels with cumulative meter readings converted to hourly energy.'
);

/*
  Source values are running meter readings in kWh, not energy used during the
  hour. DE_KN_residential4_grid_import climbs from 0.05 to 10,247 across the
  file and never decreases. Modelling it as stored fits a monotonic ramp.

  Differencing is only valid when the previous reading is exactly one hour
  earlier. Buildings joined and left the panel at different times, so consecutive
  non-null readings can be weeks apart, and differencing across that would place
  weeks of consumption in a single hour. A decrease means the meter was reset or
  replaced. Both cases yield NULL.

  A third case evades both guards: a meter replaced in place jumps forward
  between two adjacent readings. Those are flagged against the channel's own
  scale, because 200 kWh in an hour is impossible for a house and unremarkable
  for an industrial site. Flagged rows keep their original value; clipping would
  hide the problem from whoever looks next.
*/
WITH readings AS (
  SELECT
    utc_timestamp,
    channel,
    building,
    building_type,
    device,
    meter_kwh,
    LAG(meter_kwh) OVER w AS previous_kwh,
    LAG(utc_timestamp) OVER w AS previous_timestamp
  FROM raw.household
  WINDOW w AS (PARTITION BY channel ORDER BY utc_timestamp)
),

differenced AS (
  SELECT
    * EXCEPT (previous_kwh, previous_timestamp),
    IF(
      TIMESTAMP_DIFF(utc_timestamp, previous_timestamp, MINUTE) = 60
        AND meter_kwh >= previous_kwh,
      meter_kwh - previous_kwh,
      NULL
    ) AS energy_kwh
  FROM readings
),

channel_scale AS (
  SELECT
    channel,
    APPROX_QUANTILES(energy_kwh, 100)[OFFSET(99)] AS p99_kwh
  FROM differenced
  WHERE energy_kwh IS NOT NULL
  GROUP BY channel
)

SELECT
  d.utc_timestamp,
  d.channel,
  d.building,
  d.building_type,
  d.device,
  d.meter_kwh,
  d.energy_kwh,
  COALESCE(d.energy_kwh > GREATEST(50 * s.p99_kwh, 1.0), FALSE) AS is_implausible
FROM differenced AS d
LEFT JOIN channel_scale AS s
  USING (channel)
