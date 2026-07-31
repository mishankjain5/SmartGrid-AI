MODEL (
  name staging.stg_capacity,
  kind FULL,
  description 'Installed generation capacity by month, MW.'
);

/*
  Monthly, matching the source. It is deliberately not expanded to hourly:
  capacity is only reported monthly, and fabricating hourly rows would imply a
  precision the data does not have. Downstream models join on the month.

  Solar is reported twice. `Solar DC` is the panel rating; `Solar AC` is the
  inverter rating, which is what can actually reach the grid — currently about
  116 GW against 127 GW DC. Generation is metered at the grid, so solar_ac_mw is
  the correct denominator when normalising output to a capacity factor.
*/
SELECT
  month,
  MAX(IF(production_type = 'Solar AC', capacity_mw, NULL)) AS solar_ac_mw,
  MAX(IF(production_type = 'Solar DC', capacity_mw, NULL)) AS solar_dc_mw,
  MAX(IF(production_type = 'Wind onshore', capacity_mw, NULL)) AS wind_onshore_mw,
  MAX(IF(production_type = 'Wind offshore', capacity_mw, NULL)) AS wind_offshore_mw,
  MAX(IF(production_type = 'Battery storage (power)', capacity_mw, NULL)) AS battery_power_mw,
  MAX(IF(production_type = 'Battery storage (capacity)', capacity_mw, NULL)) AS battery_energy_mwh
FROM raw.installed_power
GROUP BY month
