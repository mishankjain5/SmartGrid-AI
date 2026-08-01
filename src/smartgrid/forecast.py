"""Run the solar forecast backtest.

    python -m smartgrid.forecast
    python -m smartgrid.forecast --test-days 60 --quiet

Scores every model on daylight hours against the transmission operators'
published day-ahead forecast.
"""

import argparse

import pandas as pd

from smartgrid.modelling import (
    default_models,
    load_features,
    modelling_frame,
    run_backtest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-train-days", type=int, default=365)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--quiet", action="store_true", help="suppress per-fold output")
    args = parser.parse_args(argv)

    pd.set_option("display.width", 200)

    print("loading features...", flush=True)
    frame = modelling_frame(load_features())
    print(
        f"{len(frame):,} usable hours, "
        f"{frame['utc_timestamp'].min().date()} to {frame['utc_timestamp'].max().date()}\n"
    )

    result = run_backtest(
        frame,
        default_models(),
        initial_train_days=args.initial_train_days,
        test_days=args.test_days,
        verbose=not args.quiet,
    )

    scored = result.daylight_only()
    print(
        f"\n{len(result.folds)} folds, {len(scored):,} daylight hours scored\n"
    )
    print(result.summary().round(4).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())