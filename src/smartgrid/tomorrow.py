"""Forecast tomorrow and say what to do about it.

    python -m smartgrid.tomorrow
    python -m smartgrid.tomorrow --system-kwp 15 --energy-kwh 20

Trains on all available history, forecasts the target day's solar output, pairs
it with the published day-ahead prices, and prints the battery schedule and the
expected saving.
"""

import argparse

import pandas as pd

from smartgrid.config import MARKET_TIMEZONE
from smartgrid.modelling.predict import predict_day, target_day
from smartgrid.optimisation.battery import Battery
from smartgrid.optimisation.plan import plan_day
from smartgrid.sources import energy_charts


def load_published_prices() -> pd.Series:
    """Prices for the coming days, as published by the auction."""
    frame = energy_charts.fetch_upcoming_price()
    hourly = (
        frame.set_index("utc_timestamp")["price_eur_mwh"]
        .resample("h")
        .mean()
        .dropna()
    )
    return hourly


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-kwp", type=float, default=10.0)
    parser.add_argument("--energy-kwh", type=float, default=10.0)
    parser.add_argument("--power-kw", type=float, default=5.0)
    parser.add_argument("--round-trip", type=float, default=0.90)
    parser.add_argument(
        "--day",
        type=pd.Timestamp,
        help="target a specific day instead of tomorrow, e.g. 2026-08-01",
    )
    args = parser.parse_args(argv)

    day = (
        args.day.tz_localize(MARKET_TIMEZONE) if args.day is not None else target_day()
    )
    print(f"Forecasting {day.date()}\n")

    prediction = predict_day(day=day)
    prices = load_published_prices()

    covered = prices.loc[prices.index >= day.tz_convert("UTC")]
    if len(covered) < 24:
        print(
            f"Only {len(covered)} priced hours available for {day.date()}. "
            "The day-ahead auction clears around 12:45 market time; "
            "before then, tomorrow's prices are not yet published."
        )
        return 1

    plan = plan_day(
        prediction,
        prices,
        battery=Battery(
            energy_mwh=args.energy_kwh / 1000,
            power_mw=args.power_kw / 1000,
            round_trip_efficiency=args.round_trip,
        ),
        system_kwp=args.system_kwp,
    )

    print(
        f"{args.system_kwp:.0f} kWp array, "
        f"{args.energy_kwh:.0f} kWh / {args.power_kw:.0f} kW battery\n"
    )
    print(plan.hours.round(2).to_string(index=False))

    print("\nRecommendation:")
    for line in plan.advice():
        print(f"  - {line}")

    total_solar = plan.hours["predicted_solar_kw"].sum()
    print(f"\nPredicted generation: {total_solar:.1f} kWh over the day.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
