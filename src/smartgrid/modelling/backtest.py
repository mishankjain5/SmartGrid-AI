"""Walk-forward backtesting.

A random train/test split is invalid here: it lets a model learn from the future.
A single chronological split tests one moment, which across four years of
weather-driven data is close to luck.

Rolling origin replays history instead. Train on everything up to a point,
forecast the next window, step forward, refit, repeat. That mirrors how the model
would actually have been operated and yields many independent windows to average
over.

Training data is truncated at each fold's gate rather than at the start of its
test window. Those differ by 12 to 36 hours, and that gap is exactly the
information a deployed model would not have had.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from smartgrid.modelling.dataset import BENCHMARK, TARGET, is_daylight
from smartgrid.modelling.models import Forecaster


@dataclass(frozen=True)
class Fold:
    index: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __str__(self) -> str:
        return (
            f"fold {self.index}: train to {self.train_end.date()} "
            f"-> test {self.test_start.date()}..{self.test_end.date()}"
        )


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    folds: list[Fold] = field(default_factory=list)

    @property
    def model_names(self) -> list[str]:
        reserved = {"fold", "utc_timestamp", "actual", "daylight", BENCHMARK}
        return [c for c in self.predictions.columns if c not in reserved]

    def daylight_only(self) -> pd.DataFrame:
        return self.predictions[self.predictions["daylight"]]

    def summary(self) -> pd.DataFrame:
        """Error per model, scored on daylight hours, against the benchmark."""
        scored = self.daylight_only()
        actual = scored["actual"]

        benchmark_mae = float(np.mean(np.abs(scored[BENCHMARK] - actual)))

        rows = []
        for name in [*self.model_names, BENCHMARK]:
            error = scored[name] - actual
            rows.append(
                {
                    "model": "TSO forecast" if name == BENCHMARK else name,
                    "mae": float(np.mean(np.abs(error))),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "bias": float(np.mean(error)),
                    "skill_vs_tso": 1 - float(np.mean(np.abs(error))) / benchmark_mae,
                }
            )

        return pd.DataFrame(rows).set_index("model").sort_values("mae")


def iter_folds(
    timestamps: pd.Series,
    *,
    initial_train_days: int = 365,
    test_days: int = 30,
) -> Iterator[Fold]:
    """Rolling-origin folds over a time-ordered series."""
    start, end = timestamps.min(), timestamps.max()
    test_start = start + pd.Timedelta(initial_train_days, "D")

    index = 0
    while test_start < end:
        test_end = min(test_start + pd.Timedelta(test_days, "D"), end)

        # Training stops at the gate for the first target hour: noon on the
        # preceding day, never later.
        train_end = test_start.tz_convert("Europe/Berlin").normalize()
        train_end = (train_end - pd.Timedelta(1, "D") + pd.Timedelta(12, "h")).tz_convert("UTC")

        yield Fold(index=index, train_end=train_end, test_start=test_start, test_end=test_end)

        test_start = test_end
        index += 1


def run_backtest(
    frame: pd.DataFrame,
    models: dict[str, Forecaster],
    *,
    initial_train_days: int = 365,
    test_days: int = 30,
    verbose: bool = True,
) -> BacktestResult:
    """Replay history fold by fold, refitting every model on each."""
    frame = frame.sort_values("utc_timestamp").reset_index(drop=True)

    collected = []
    folds = []

    for fold in iter_folds(
        frame["utc_timestamp"],
        initial_train_days=initial_train_days,
        test_days=test_days,
    ):
        train = frame[frame["utc_timestamp"] <= fold.train_end]
        test = frame[
            (frame["utc_timestamp"] >= fold.test_start)
            & (frame["utc_timestamp"] < fold.test_end)
        ]

        if train.empty or test.empty:
            continue

        predictions = pd.DataFrame(
            {
                "fold": fold.index,
                "utc_timestamp": test["utc_timestamp"],
                "actual": test[TARGET],
                "daylight": is_daylight(test),
                BENCHMARK: test[BENCHMARK],
            },
            index=test.index,
        )

        for name, model in models.items():
            predictions[name] = model.fit(train).predict(test)

        collected.append(predictions)
        folds.append(fold)

        if verbose:
            print(f"  {fold}  train={len(train):,} test={len(test):,}", flush=True)

    if not collected:
        raise ValueError("no folds produced; the series is shorter than the warm-up")

    return BacktestResult(
        predictions=pd.concat(collected).dropna(subset=["actual", BENCHMARK]),
        folds=folds,
    )