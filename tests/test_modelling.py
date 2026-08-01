"""Modelling tests.

The forecasters and the backtest are exercised on a synthetic frame shaped like
the mart, so these run without credentials. Cloud-backed checks on the real
feature table are marked and skipped when GCP_PROJECT is unset.
"""

import numpy as np
import pandas as pd
import pytest

from smartgrid.config import get_settings
from smartgrid.modelling import backtest, dataset, models

requires_cloud = pytest.mark.skipif(
    not get_settings().bigquery_available, reason="GCP_PROJECT not set"
)


@pytest.fixture
def frame() -> pd.DataFrame:
    """Two years of hourly data with a learnable solar shape."""
    index = pd.date_range("2024-01-01", "2025-12-31 23:00", freq="h", tz="UTC")
    local = index.tz_convert("Europe/Berlin")
    rng = np.random.default_rng(0)

    hour = local.hour.to_numpy()
    day = local.dayofyear.to_numpy()
    daylight = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
    seasonal = 0.6 + 0.4 * np.sin((day - 80) / 365.25 * 2 * np.pi)
    cloud = rng.uniform(0.3, 1.0, len(index))

    ghi = daylight * seasonal * cloud * 900
    target = daylight * seasonal * cloud * 0.45

    # Each location gets its own cloud draw, so the spread features are real.
    per_location = {
        name: ghi * rng.uniform(0.7, 1.15, len(index))
        for name in dataset.LOCATION_FEATURES
    }
    location_matrix = np.column_stack(list(per_location.values()))

    built = pd.DataFrame(
        {
            "utc_timestamp": index,
            "local_datetime": local.tz_localize(None),
            dataset.TARGET: target,
        }
        | per_location
        | {
            "ghi_mean": location_matrix.mean(axis=1),
            "ghi_min": location_matrix.min(axis=1),
            "ghi_max": location_matrix.max(axis=1),
            "ghi_stddev": location_matrix.std(axis=1),
            "ghi_spread": location_matrix.max(axis=1) - location_matrix.min(axis=1),
            "direct_radiation_mean": ghi * 0.7,
            "diffuse_radiation_mean": ghi * 0.3,
            "temperature_mean": 10 + 10 * np.sin(day / 365.25 * 2 * np.pi),
            "cloud_cover_mean": (1 - cloud) * 100,
            "cloud_cover_stddev": rng.uniform(0, 20, len(index)),
            "hour_of_day": hour,
            "day_of_week": local.dayofweek.to_numpy(),
            "day_of_year": day,
            "month_of_year": local.month.to_numpy(),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "day_of_year_sin": np.sin(2 * np.pi * day / 365.25),
            "day_of_year_cos": np.cos(2 * np.pi * day / 365.25),
            # A benchmark slightly better than the models will manage.
            dataset.BENCHMARK: target + rng.normal(0, 0.004, len(index)),
        }
    )

    series = built[dataset.TARGET]
    built["capacity_factor_lag_48h"] = series.shift(48)
    built["capacity_factor_lag_72h"] = series.shift(72)
    built["capacity_factor_lag_168h"] = series.shift(168)
    built["capacity_factor_mean_7d"] = series.shift(48).rolling(168).mean()
    built["capacity_factor_mean_30d"] = series.shift(48).rolling(720).mean()

    return dataset.modelling_frame(built)


# --- dataset ----------------------------------------------------------------


def test_the_target_and_benchmark_are_never_features():
    assert dataset.TARGET not in dataset.FEATURES
    assert dataset.BENCHMARK not in dataset.FEATURES
    for column in dataset.EXCLUDED:
        assert column not in dataset.FEATURES


def test_weather_features_cover_the_country_not_one_point():
    """National output is spread over 800 km; a single point cannot represent it."""
    assert len(dataset.LOCATION_FEATURES) >= 8
    assert set(dataset.LOCATION_FEATURES) <= set(dataset.FEATURES)

    # The spread across locations is itself informative and must be carried.
    assert {"ghi_stddev", "ghi_spread"} <= set(dataset.FEATURES)


def test_no_short_lead_weather_is_selected():
    """stg_weather still holds short-lead columns; none may reach the model."""
    forbidden = [f for f in dataset.FEATURES if f.endswith(("_wm2", "_pct", "_c"))]
    assert not forbidden, forbidden


def test_no_lag_shorter_than_the_safe_bound():
    lags = [f for f in dataset.FEATURES if "lag_" in f]
    hours = [int(f.split("lag_")[1].rstrip("h")) for f in lags]
    assert hours and min(hours) >= 48


def test_modelling_frame_drops_rows_without_features(frame):
    assert frame[[dataset.TARGET, *dataset.FEATURES]].notna().all().all()


def test_daylight_filter_excludes_night(frame):
    lit = dataset.is_daylight(frame)
    assert lit.any() and not lit.all()
    assert (frame.loc[~lit, "ghi_max"] == 0).all()


# --- forecasters ------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        models.Persistence(),
        models.Climatology(),
        models.LinearIrradiance(),
        models.GradientBoosting(max_iter=25),
    ],
    ids=lambda m: m.name,
)
def test_every_model_fits_and_predicts_in_range(frame, model):
    train = frame[frame["utc_timestamp"] < "2025-07-01"]
    test = frame[frame["utc_timestamp"] >= "2025-07-01"]

    predicted = model.fit(train).predict(test)

    assert len(predicted) == len(test)
    assert predicted.notna().all()
    assert predicted.between(0, 1).all(), "capacity factor is bounded"


def test_predictions_are_clipped_to_physical_range(frame):
    class Rogue(models.Forecaster):
        name = "rogue"

        def _fit(self, train):
            pass

        def _predict(self, test):
            return np.full(len(test), 5.0)

    predicted = Rogue().fit(frame).predict(frame)
    assert (predicted == 1.0).all()


def test_default_models_are_uniquely_named():
    built = models.default_models()
    assert len(built) == 4
    assert set(built) == {m.name for m in built.values()}


# --- backtest ---------------------------------------------------------------


@pytest.fixture
def result(frame):
    return backtest.run_backtest(
        frame,
        {"persistence_48h": models.Persistence(), "climatology": models.Climatology()},
        initial_train_days=200,
        test_days=60,
        verbose=False,
    )


def test_folds_are_chronological_and_do_not_overlap(result):
    for earlier, later in zip(result.folds, result.folds[1:], strict=False):
        assert earlier.test_end <= later.test_start


def test_training_stops_before_every_test_window(result):
    """The gate guarantee, re-derived from the folds actually used."""
    for fold in result.folds:
        assert fold.train_end < fold.test_start


def test_summary_scores_every_model_and_the_benchmark(result):
    summary = result.summary()
    assert set(summary.index) == {"persistence_48h", "climatology", "TSO forecast"}
    assert summary.loc["TSO forecast", "skill_vs_tso"] == 0.0


def test_scoring_uses_daylight_hours_only(result):
    assert len(result.daylight_only()) < len(result.predictions)


def test_a_series_shorter_than_the_warmup_is_refused(frame):
    with pytest.raises(ValueError, match="no folds"):
        backtest.run_backtest(
            frame[frame["utc_timestamp"] < "2024-03-01"],
            {"persistence_48h": models.Persistence()},
            initial_train_days=365,
            verbose=False,
        )


# --- against the real feature table -----------------------------------------


@requires_cloud
def test_real_features_are_all_present_in_the_mart():
    frame = dataset.load_features().head(10)
    missing = [c for c in [*dataset.FEATURES, dataset.TARGET] if c not in frame.columns]
    assert not missing
