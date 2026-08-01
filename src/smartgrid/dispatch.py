"""Run the battery dispatch backtest.

    python -m smartgrid.dispatch
    python -m smartgrid.dispatch --energy-kwh 20 --power-kw 10

Reports what a battery would have earned trading against German day-ahead
prices, under perfect foresight and under a realistic price estimate.
"""

import argparse

import pandas as pd

from smartgrid.config import get_settings
from smartgrid.optimisation import Battery, annualise, optimise_series
from smartgrid.warehouse import query

#: A day-ahead schedule has to be planned before prices are known. Using the
#: previous day's shape is the honest naive estimate, and the gap to perfect
#: foresight is what a price forecast would be worth.
NAIVE_LAG_HOURS = 24


def load_prices() -> pd.Series:
    project = get_settings().require_project()
    frame = query(
        f"""
        SELECT utc_timestamp, price_eur_mwh
        FROM `{project}.staging.stg_price`
        WHERE price_eur_mwh IS NOT NULL
        ORDER BY utc_timestamp
        """
    )
    return frame.set_index("utc_timestamp")["price_eur_mwh"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy-kwh", type=float, default=10.0)
    parser.add_argument("--power-kw", type=float, default=5.0)
    parser.add_argument("--round-trip", type=float, default=0.90)
    args = parser.parse_args(argv)

    battery = Battery(
        energy_mwh=args.energy_kwh / 1000,
        power_mw=args.power_kw / 1000,
        round_trip_efficiency=args.round_trip,
    )

    prices = load_prices()
    print(
        f"{len(prices):,} hourly prices, "
        f"{prices.index.min().date()} to {prices.index.max().date()}"
    )
    print(
        f"battery: {args.energy_kwh:.0f} kWh, {args.power_kw:.0f} kW, "
        f"{args.round_trip:.0%} round-trip\n"
    )

    print("solving perfect foresight...", flush=True)
    perfect = annualise(optimise_series(prices, battery), battery)

    print("solving with previous-day prices...", flush=True)
    naive = annualise(
        optimise_series(
            prices, battery, decision_prices=prices.shift(NAIVE_LAG_HOURS)
        ),
        battery,
    )

    summary = pd.DataFrame(
        {"perfect foresight": perfect, "previous-day prices": naive}
    ).T
    summary["capture_rate"] = summary["annual_eur"] / perfect["annual_eur"]

    print()
    print(summary.round(2).to_string())
    print(
        f"\nA price forecast is worth up to "
        f"EUR {perfect['annual_eur'] - naive['annual_eur']:.0f}/year here."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())