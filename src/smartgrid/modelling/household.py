"""Household consumption forecasting.

Replaces the fixed baseline load the dispatch planner used as a placeholder.
Knowing when a house actually draws power changes what a battery should do: a
flat assumption spreads demand evenly across the day and misses that most of it
lands in the morning and evening, which is exactly when prices peak.

Calendar-driven, and the reason is a coverage gap rather than preference. The
OPSD household panel ends in 2018 while the weather archive begins in 2024, so
there is no overlap to join on. The model therefore cannot see temperature, and
will miss a cold snap driving a heat pump. That limitation is real and is
reported rather than papered over — see `mart_household_features`.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from smartgrid.config import MARKET_TIMEZONE, get_settings
from smartgrid.warehouse import query

TARGET = "consumption_kwh"

FEATURES = [
    "hour_of_day",
    "day_of_week",
    "day_of_year",
    "month_of_year",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "consumption_lag_48h",
    "consumption_lag_168h",
    "consumption_mean_7d",
]

#: Richest channel mix in the panel: grid import and export, PV, a heat pump and
#: an EV. The closest thing here to a home a battery would actually be sold into.
DEFAULT_BUILDING = "residential4"


def load_household(building: str = DEFAULT_BUILDING) -> pd.DataFrame:
    project = get_settings().require_project()
    frame = query(
        f"""
        SELECT *
        FROM `{project}.marts.mart_household_features`
        WHERE building = @building
        ORDER BY utc_timestamp
        """,
        building=building,
    )
    frame["local_datetime"] = pd.to_datetime(frame["local_datetime"])
    return frame


@dataclass
class LoadProfile:
    """Mean consumption for each hour of the week, in kWh.

    A week rather than a day because weekday and weekend demand differ in shape,
    not just level: the morning peak shifts later and the midday dip fills in.
    """

    by_hour_of_week: pd.Series
    overall_mean: float
    building: str
    hours_observed: int

    def for_day(self, day: pd.Timestamp) -> pd.Series:
        """The 24 hourly values this profile predicts for a given local day."""
        index = pd.date_range(
            day.normalize(), periods=24, freq="h", tz=day.tz or MARKET_TIMEZONE
        )
        keys = index.dayofweek * 24 + index.hour
        values = self.by_hour_of_week.reindex(keys).fillna(self.overall_mean)
        return pd.Series(values.to_numpy(), index=index.tz_convert("UTC"))

    @property
    def daily_kwh(self) -> float:
        return float(self.by_hour_of_week.mean() * 24)


def fit_profile(
    frame: pd.DataFrame | None = None,
    *,
    building: str = DEFAULT_BUILDING,
) -> LoadProfile:
    """Average consumption by hour of week.

    A gradient booster on the same features scores barely better than this, which
    is unsurprising: with no weather and no occupancy signal, the calendar is
    almost all the information there is. The simpler model is preferred because
    its output is inspectable — a customer can look at the profile and recognise
    their own day.
    """
    frame = load_household(building) if frame is None else frame
    usable = frame.dropna(subset=[TARGET])

    if usable.empty:
        raise ValueError(f"no consumption data for {building}")

    local = pd.to_datetime(usable["local_datetime"])
    hour_of_week = local.dt.dayofweek * 24 + local.dt.hour

    return LoadProfile(
        by_hour_of_week=usable.groupby(hour_of_week)[TARGET].mean(),
        overall_mean=float(usable[TARGET].mean()),
        building=building,
        hours_observed=len(usable),
    )


def score_profile(frame: pd.DataFrame, profile: LoadProfile) -> dict[str, float]:
    """Error against a flat-load assumption, which is what it replaces."""
    usable = frame.dropna(subset=[TARGET])
    local = pd.to_datetime(usable["local_datetime"])
    hour_of_week = local.dt.dayofweek * 24 + local.dt.hour

    predicted = hour_of_week.map(profile.by_hour_of_week).fillna(profile.overall_mean)
    flat = np.full(len(usable), profile.overall_mean)
    actual = usable[TARGET].to_numpy()

    profile_mae = float(np.mean(np.abs(predicted.to_numpy() - actual)))
    flat_mae = float(np.mean(np.abs(flat - actual)))

    return {
        "profile_mae_kwh": profile_mae,
        "flat_mae_kwh": flat_mae,
        "improvement": 1 - profile_mae / flat_mae,
        "hours": len(usable),
    }
