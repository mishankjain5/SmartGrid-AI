"""Forecasters, from trivial baselines to gradient boosting.

All share one interface — `fit(train)` then `predict(test)` on frames from the
mart — so the backtest runs any of them without special cases and each is judged
on identical footing.

The baselines are not filler. A model that cannot beat persistence has learned
nothing, and an error metric without that comparison hides it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartgrid.modelling.dataset import FEATURES, TARGET


class Forecaster:
    """Base class. Subclasses implement `_fit` and `_predict`."""

    name = "forecaster"

    def fit(self, train: pd.DataFrame) -> Forecaster:
        self._fit(train)
        return self

    def predict(self, test: pd.DataFrame) -> pd.Series:
        predicted = np.asarray(self._predict(test), dtype=float)
        # Capacity factor is bounded; a model has no business predicting outside it.
        return pd.Series(np.clip(predicted, 0.0, 1.0), index=test.index)

    def _fit(self, train: pd.DataFrame) -> None:
        raise NotImplementedError

    def _predict(self, test: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class Persistence(Forecaster):
    """Repeat the value from 48 hours earlier.

    Cannot use 24 hours: for a 23:00 target that reaches past the noon gate. 48
    is the first whole day clearing the 36-hour bound, so the hour of day still
    lines up.
    """

    name = "persistence_48h"

    def _fit(self, train: pd.DataFrame) -> None:
        self.fallback = float(train[TARGET].mean())

    def _predict(self, test: pd.DataFrame) -> pd.Series:
        return test["capacity_factor_lag_48h"].fillna(self.fallback)


class Climatology(Forecaster):
    """Average output for each (week of year, hour) seen in training.

    Knows the calendar and nothing else, so the gap between this and a real model
    is the value of knowing about weather.
    """

    name = "climatology"

    def _fit(self, train: pd.DataFrame) -> None:
        self.fallback = float(train[TARGET].mean())
        self.by_hour = train.groupby("hour_of_day")[TARGET].mean()
        self.by_week_hour = (
            train.assign(week=((train["day_of_year"] - 1) // 7).astype(int))
            .groupby(["week", "hour_of_day"])[TARGET]
            .mean()
        )

    def _predict(self, test: pd.DataFrame) -> pd.Series:
        week = ((test["day_of_year"] - 1) // 7).astype(int)
        keys = pd.MultiIndex.from_arrays([week, test["hour_of_day"]])

        seasonal = pd.Series(self.by_week_hour.reindex(keys).to_numpy(), index=test.index)
        hourly = test["hour_of_day"].map(self.by_hour)
        return seasonal.fillna(hourly).fillna(self.fallback)


class LinearIrradiance(Forecaster):
    """Ridge regression on the full feature set.

    A linear reference: irradiance and capacity factor are close to
    proportional, so this establishes how much of the problem is linear before
    any tree is fitted.
    """

    name = "ridge"

    def _fit(self, train: pd.DataFrame) -> None:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        self.model.fit(train[FEATURES], train[TARGET])

    def _predict(self, test: pd.DataFrame) -> pd.Series:
        return self.model.predict(test[FEATURES])


class GradientBoosting(Forecaster):
    """Histogram-based gradient boosted trees.

    scikit-learn's implementation rather than LightGBM or XGBoost: same algorithm
    family, no external native library, and it handles NaN natively.
    """

    name = "gradient_boosting"

    def __init__(self, **params):
        self.params = {
            "max_iter": 400,
            "learning_rate": 0.05,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 40,
            "l2_regularization": 1.0,
            "random_state": 0,
            **params,
        }

    def _fit(self, train: pd.DataFrame) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.model = HistGradientBoostingRegressor(**self.params)
        self.model.fit(train[FEATURES], train[TARGET])

    def _predict(self, test: pd.DataFrame) -> pd.Series:
        return self.model.predict(test[FEATURES])


def default_models() -> dict[str, Forecaster]:
    return {
        model.name: model
        for model in (Persistence(), Climatology(), LinearIrradiance(), GradientBoosting())
    }