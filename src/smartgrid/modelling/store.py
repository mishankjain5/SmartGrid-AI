"""Persisting forecasts, so they can be scored against what actually happened.

A backtest measures the approach. It cannot catch a bug that only appears going
forward — a stale weather fetch, a feature built differently at predict time
than at train time, a model quietly falling back to its fallback. Those show up
only when yesterday's saved forecast is compared with today's measurement.

Predictions are appended, never replaced. Re-running for the same day writes a
second row with a later `issued_at`, so revisions stay visible instead of
overwriting the earlier call.
"""

import pandas as pd

from smartgrid.config import get_settings
from smartgrid.warehouse import query
from smartgrid.warehouse.load import load_dataframe

DATASET = "marts"
TABLE = "solar_predictions"


def save_predictions(prediction: pd.DataFrame, *, model_name: str) -> int:
    """Append a forecast. Returns the table's total row count."""
    required = {"utc_timestamp", "predicted_capacity_factor", "target_date"}
    missing = required - set(prediction.columns)
    if missing:
        raise KeyError(f"prediction is missing columns: {sorted(missing)}")

    record = prediction.copy()
    record["model_name"] = model_name
    record["target_date"] = pd.to_datetime(record["target_date"]).dt.date

    return load_dataframe(record, TABLE, dataset=DATASET, replace=False)


def load_predictions() -> pd.DataFrame:
    """Every forecast ever saved, newest issue first."""
    project = get_settings().require_project()
    return query(
        f"""
        SELECT *
        FROM `{project}.{DATASET}.{TABLE}`
        ORDER BY issued_at DESC, utc_timestamp
        """
    )


def score_predictions(*, latest_only: bool = True) -> pd.DataFrame:
    """Compare saved forecasts with measured output, hour by hour.

    Args:
        latest_only: keep only the most recent forecast for each target day.
            Re-running produces revisions, and scoring all of them would weight
            heavily-revised days more than others.

    Returns one row per scored hour, with the operators' forecast alongside for
    comparison on identical hours.
    """
    project = get_settings().require_project()

    ranking = "QUALIFY ROW_NUMBER() OVER w = 1" if latest_only else ""
    window = (
        "WINDOW w AS (PARTITION BY p.utc_timestamp ORDER BY p.issued_at DESC)"
        if latest_only
        else ""
    )

    # Joined against staging rather than the feature mart. The mart inner joins
    # weather, and the historical weather archive lags a day or two behind
    # generation — so scoring through the mart would be blocked from measuring
    # exactly the days that just finished. Scoring needs measured output and the
    # benchmark, neither of which involves weather.
    return query(
        f"""
        SELECT
          p.utc_timestamp,
          p.target_date,
          p.issued_at,
          p.model_name,
          p.predicted_capacity_factor,
          p.ghi_mean_forecast,
          s.solar_capacity_factor AS actual_capacity_factor,
          SAFE_DIVIDE(f.solar_forecast_mw, s.solar_ac_mw) AS tso_capacity_factor
        FROM `{project}.{DATASET}.{TABLE}` AS p
        JOIN `{project}.staging.stg_solar_output` AS s
          USING (utc_timestamp)
        LEFT JOIN `{project}.staging.stg_day_ahead_forecast` AS f
          USING (utc_timestamp)
        WHERE s.solar_capacity_factor IS NOT NULL
        {ranking}
        {window}
        ORDER BY p.utc_timestamp
        """
    )


def summarise_scores(scored: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute error for the saved forecasts and the benchmark.

    Daylight hours only, matching how the backtest reports: output is zero at
    night and every method gets that right for free.
    """
    if scored.empty:
        raise ValueError(
            "no saved forecast has a measured outcome yet. Predictions can only "
            "be scored once the day they cover has passed and been ingested."
        )

    # Daylight is taken from the forecast's own irradiance rather than a measured
    # column, so the filter needs nothing beyond what was already stored.
    daylight = scored[scored["ghi_mean_forecast"] > 0]
    if daylight.empty:
        raise ValueError("no daylight hours among the scored predictions")

    actual = daylight["actual_capacity_factor"]
    rows = [
        {
            "source": "our forecast",
            "mae": float((daylight["predicted_capacity_factor"] - actual).abs().mean()),
            "hours": len(daylight),
        }
    ]

    benchmark = daylight.dropna(subset=["tso_capacity_factor"])
    if not benchmark.empty:
        rows.append(
            {
                "source": "TSO forecast",
                "mae": float(
                    (
                        benchmark["tso_capacity_factor"]
                        - benchmark["actual_capacity_factor"]
                    )
                    .abs()
                    .mean()
                ),
                "hours": len(benchmark),
            }
        )

    return pd.DataFrame(rows).set_index("source")
