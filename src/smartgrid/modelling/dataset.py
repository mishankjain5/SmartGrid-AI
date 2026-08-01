"""Loading the solar feature table and selecting model inputs.

The mart already enforces gate discipline: every feature in it was computable at
12:00 the day before the hour it describes, and the SQLMesh audits fail the build
if that stops being true. This module does not re-derive any of that. It selects
columns and hands over a frame.

Feature columns are listed explicitly rather than inferred from the schema. A
`SELECT *` approach would silently absorb any column added later, including the
actuals and the benchmark, and produce a model that looks excellent.
"""

import pandas as pd

from smartgrid.config import get_settings
from smartgrid.warehouse import query

__all__ = [
    "BENCHMARK",
    "FEATURES",
    "TARGET",
    "is_daylight",
    "load_features",
    "load_recent_generation",
    "modelling_frame",
]

TARGET = "solar_capacity_factor"

#: Irradiance at each of the nine sampled locations. Worth about 4% of accuracy
#: over the national aggregates alone — useful but not the main effect. Most of
#: the value of sampling several points comes from the averages below, which
#: estimate national conditions far better than any single reading.
LOCATION_FEATURES = [
    "ghi_hamburg",
    "ghi_rostock",
    "ghi_hanover",
    "ghi_berlin",
    "ghi_cologne",
    "ghi_kassel",
    "ghi_nuremberg",
    "ghi_stuttgart",
    "ghi_munich",
]

#: The averages here are the strongest weather features by permutation
#: importance. ghi_stddev and ghi_spread describe how uniform the country is,
#: which the per-location columns imply but do not state directly.
NATIONAL_WEATHER_FEATURES = [
    "ghi_mean",
    "ghi_min",
    "ghi_max",
    "ghi_stddev",
    "ghi_spread",
    "direct_radiation_mean",
    "diffuse_radiation_mean",
    "temperature_mean",
    "cloud_cover_mean",
    "cloud_cover_stddev",
]

WEATHER_FEATURES = LOCATION_FEATURES + NATIONAL_WEATHER_FEATURES

CALENDAR_FEATURES = [
    "hour_of_day",
    "day_of_week",
    "day_of_year",
    "month_of_year",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]

HISTORY_FEATURES = [
    "capacity_factor_lag_48h",
    "capacity_factor_lag_72h",
    "capacity_factor_lag_168h",
    "capacity_factor_mean_7d",
    "capacity_factor_mean_30d",
]

FEATURES = WEATHER_FEATURES + CALENDAR_FEATURES + HISTORY_FEATURES

#: Present in the mart, never model inputs.
#:   solar_mw, solar_ac_mw  -- the target's own components
#:   tso_*                  -- the benchmark being competed against
#:   gate_utc, horizon_hours, timestamps -- bookkeeping
EXCLUDED = [
    "solar_mw",
    "solar_ac_mw",
    "tso_solar_forecast_mw",
    "tso_solar_forecast_capacity_factor",
    "gate_utc",
    "horizon_hours",
    "utc_timestamp",
    "local_datetime",
]

BENCHMARK = "tso_solar_forecast_capacity_factor"


def load_features() -> pd.DataFrame:
    """Read the mart, ordered by time and ready to split into folds."""
    project = get_settings().require_project()

    frame = query(
        f"""
        SELECT *
        FROM `{project}.marts.mart_solar_features`
        ORDER BY utc_timestamp
        """
    )
    frame["local_datetime"] = pd.to_datetime(frame["local_datetime"])
    return frame


def load_recent_generation(days: int = 30) -> pd.DataFrame:
    """Observed capacity factor straight from staging, bypassing the mart.

    The lag features need generation history, not weather. Reading them from the
    feature mart would couple them to weather coverage, because the mart inner
    joins the two — so a weather table a day behind would block a forecast whose
    inputs are all present. This reads generation and capacity directly.
    """
    project = get_settings().require_project()

    return query(
        f"""
        SELECT utc_timestamp, {TARGET}, solar_ac_mw
        FROM `{project}.staging.stg_solar_output`
        WHERE utc_timestamp >= TIMESTAMP_SUB(
                CURRENT_TIMESTAMP(), INTERVAL {days} DAY)
          AND {TARGET} IS NOT NULL
        ORDER BY utc_timestamp
        """
    )


def modelling_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that cannot be trained or scored on.

    The first week has no lag features, and the most recent hours may have no
    benchmark yet.
    """
    required = [TARGET, *FEATURES]
    return frame.dropna(subset=required).reset_index(drop=True)


def is_daylight(frame: pd.DataFrame) -> pd.Series:
    """Hours with forecast irradiance anywhere in the country.

    Solar output is exactly zero at night and every method predicts that
    correctly, so scoring across all hours averages in thousands of free correct
    answers and flatters everything equally.

    Uses the maximum across locations rather than the mean: sunrise reaches the
    east before the west, and an hour lit anywhere is an hour the grid sees
    generation in.
    """
    return frame["ghi_max"] > 0