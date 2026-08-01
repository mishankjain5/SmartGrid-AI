"""Forecasting tomorrow, rather than replaying yesterday.

The backtest in `backtest.py` measures whether the approach works. This produces
the actual output: a solar forecast for hours that have not happened yet.

Features for a future hour come from three places, and only one of them is a
forecast:

* weather — Open-Meteo's forward forecast, the genuinely uncertain part;
* lags and rolling means — actual generation up to now. The gate is noon today
  for all of tomorrow, and the shortest lag is 48 hours, so every one of these is
  already observed;
* calendar and capacity — known.

The features are built the same way the training mart builds them. If they were
constructed differently here, the model would be scored on one definition and
applied to another, and the gate discipline enforced upstream would count for
nothing.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from smartgrid.config import MARKET_TIMEZONE
from smartgrid.modelling.dataset import (
    FEATURES,
    LOCATION_FEATURES,
    TARGET,
    load_features,
    load_recent_generation,
    modelling_frame,
)
from smartgrid.modelling.models import Forecaster, GradientBoosting
from smartgrid.sources import open_meteo

#: Lags the mart builds, in hours. All clear the 36-hour gate bound.
LAG_HOURS = (48, 72, 168)
ROLLING_WINDOWS = {"capacity_factor_mean_7d": 168, "capacity_factor_mean_30d": 720}


def market_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(UTC)).tz_convert(MARKET_TIMEZONE)


def target_day(reference: pd.Timestamp | None = None) -> pd.Timestamp:
    """The day a forecast issued now would cover: always tomorrow.

    The auction closing at noon today clears all 24 hours of tomorrow, and the
    results publish around 12:45. So from early afternoon, tomorrow's prices are
    known and tomorrow is the day to plan.

    The day after tomorrow is not the answer: its prices will not exist until
    tomorrow's auction runs, so a schedule for it could not be priced.
    """
    now = reference or market_now()
    return (now + pd.Timedelta(1, "D")).normalize()


def prices_expected(reference: pd.Timestamp | None = None) -> bool:
    """Whether tomorrow's auction results should have published yet."""
    now = reference or market_now()
    return now.hour >= 13


def _weather_features(forecast: pd.DataFrame) -> pd.DataFrame:
    """Collapse long-form forecasts to the columns stg_weather_national produces."""
    per_location = forecast.pivot_table(
        index="utc_timestamp", columns="location", values="ghi_wm2", aggfunc="mean"
    )
    per_location.columns = [f"ghi_{name}" for name in per_location.columns]

    missing = set(LOCATION_FEATURES) - set(per_location.columns)
    if missing:
        raise KeyError(f"forecast is missing locations: {sorted(missing)}")

    grouped = forecast.groupby("utc_timestamp")
    aggregates = pd.DataFrame(
        {
            "ghi_mean": grouped["ghi_wm2"].mean(),
            "ghi_min": grouped["ghi_wm2"].min(),
            "ghi_max": grouped["ghi_wm2"].max(),
            "ghi_stddev": grouped["ghi_wm2"].std(ddof=0),
            "direct_radiation_mean": grouped["direct_radiation_wm2"].mean(),
            "diffuse_radiation_mean": grouped["diffuse_radiation_wm2"].mean(),
            "temperature_mean": grouped["temperature_c"].mean(),
            "cloud_cover_mean": grouped["cloud_cover_pct"].mean(),
            "cloud_cover_stddev": grouped["cloud_cover_pct"].std(ddof=0),
        }
    )
    aggregates["ghi_spread"] = aggregates["ghi_max"] - aggregates["ghi_min"]

    return per_location[LOCATION_FEATURES].join(aggregates)


def _calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    local = index.tz_convert(MARKET_TIMEZONE)
    hour = local.hour.to_numpy()
    day_of_year = local.dayofyear.to_numpy()

    return pd.DataFrame(
        {
            "hour_of_day": hour,
            # BigQuery's DAYOFWEEK is 1-7 starting Sunday; pandas is 0-6 starting
            # Monday. Matched here so the model sees the same encoding it trained on.
            "day_of_week": (local.dayofweek.to_numpy() + 1) % 7 + 1,
            "day_of_year": day_of_year,
            "month_of_year": local.month.to_numpy(),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "day_of_year_sin": np.sin(2 * np.pi * day_of_year / 365.25),
            "day_of_year_cos": np.cos(2 * np.pi * day_of_year / 365.25),
        },
        index=index,
    )


def _history_features(history: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Lags and rolling means of observed generation.

    Every value here is already measured. The shortest lag is 48 hours and the
    forecast covers at most the day after tomorrow, so nothing reaches forward.
    """
    observed = history.set_index("utc_timestamp")[TARGET].sort_index()

    built = pd.DataFrame(index=index)
    for lag in LAG_HOURS:
        built[f"capacity_factor_lag_{lag}h"] = observed.reindex(
            index - pd.Timedelta(lag, "h")
        ).to_numpy()

    # Rolling windows end 48 hours back, matching the mart's gate-safe offset.
    for column, window in ROLLING_WINDOWS.items():
        rolled = observed.rolling(f"{window}h").mean()
        built[column] = rolled.reindex(
            index - pd.Timedelta(48, "h"), method="ffill"
        ).to_numpy()

    return built


def build_prediction_features(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    day: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Assemble one row per hour of the target day."""
    day = day or target_day()

    start = day.tz_convert("UTC")
    index = pd.date_range(start, periods=24, freq="h", tz="UTC")

    weather = _weather_features(forecast).reindex(index)
    if weather["ghi_mean"].isna().all():
        raise ValueError(
            f"the weather forecast does not cover {day.date()}; "
            "request more forecast_days"
        )

    frame = pd.concat(
        [weather, _calendar_features(index), _history_features(history, index)],
        axis=1,
    )

    # Capacity comes from the most recent month reported.
    frame["solar_ac_mw"] = history["solar_ac_mw"].iloc[-1]
    frame.index.name = "utc_timestamp"
    return frame


def predict_day(
    *,
    day: pd.Timestamp | None = None,
    model: Forecaster | None = None,
    forecast_days: int = 3,
) -> pd.DataFrame:
    """Train on all history and forecast one day of solar output.

    Returns one row per hour with the predicted capacity factor and the megawatts
    it implies at current installed capacity.
    """
    day = day or target_day()

    # Training data comes from the mart and may lag by a day or two without
    # harm. The lags must be current, so they come from staging directly — see
    # load_recent_generation.
    history = modelling_frame(load_features())
    recent = load_recent_generation()
    forecast = open_meteo.fetch_forecast(days=forecast_days)

    features = build_prediction_features(recent, forecast, day=day)
    usable = features.dropna(subset=FEATURES)

    if usable.empty:
        # The binding constraint is the *shortest* lag: it reaches closest to the
        # target day and so needs the most recent data.
        latest = recent["utc_timestamp"].max()
        needed = day.tz_convert("UTC") - pd.Timedelta(min(LAG_HOURS), "h")
        raise ValueError(
            f"no hour of {day.date()} has a complete feature set. "
            f"Generation history ends {latest:%Y-%m-%d %H:%M} but lags need data "
            f"through {needed:%Y-%m-%d %H:%M}. Run: python -m smartgrid.ingest "
            f"price public_power day_ahead_forecast --refresh"
        )

    fitted = (model or GradientBoosting()).fit(history)
    predicted = fitted.predict(usable)

    return pd.DataFrame(
        {
            "utc_timestamp": usable.index,
            "local_datetime": usable.index.tz_convert(MARKET_TIMEZONE).tz_localize(None),
            "predicted_capacity_factor": predicted.to_numpy(),
            "predicted_mw": predicted.to_numpy() * usable["solar_ac_mw"].to_numpy(),
            "ghi_mean_forecast": usable["ghi_mean"].to_numpy(),
            "issued_at": pd.Timestamp.now(tz="UTC"),
            "target_date": day.date(),
        }
    ).reset_index(drop=True)


def gate_for(day: pd.Timestamp) -> pd.Timestamp:
    """Noon market time on the day before, as an instant."""
    return (day.normalize() - timedelta(days=1) + pd.Timedelta(12, "h")).tz_convert("UTC")
