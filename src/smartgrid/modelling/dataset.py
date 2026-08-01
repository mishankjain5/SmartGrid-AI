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

TARGET = "solar_capacity_factor"

WEATHER_FEATURES = [
    "ghi_wm2_day_ahead",
    "direct_radiation_wm2_day_ahead",
    "diffuse_radiation_wm2_day_ahead",
    "total_radiation_wm2_day_ahead",
    "temperature_c_day_ahead",
    "cloud_cover_pct_day_ahead",
]

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


def modelling_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that cannot be trained or scored on.

    The first week has no lag features, and the most recent hours may have no
    benchmark yet.
    """
    required = [TARGET, *FEATURES]
    return frame.dropna(subset=required).reset_index(drop=True)


def is_daylight(frame: pd.DataFrame) -> pd.Series:
    """Hours with any forecast irradiance.

    Solar output is exactly zero at night and every method predicts that
    correctly, so scoring across all hours averages in thousands of free correct
    answers and flatters everything equally.
    """
    return frame["ghi_wm2_day_ahead"] > 0