"""Battery dispatch as a linear program.

Given a price series and a battery, decide when to charge and when to discharge.
The objective is revenue: buy cheap, sell dear, subject to power limits, energy
capacity and round-trip losses.

Solved one day at a time, matching the day-ahead market: all 24 hours clear at
once at noon the day before, so a day is the natural planning horizon. Each day
starts and ends at the same state of charge, which stops the optimiser from
manufacturing revenue by simply draining the battery and never refilling it.

Formulated as an optimisation rather than a heuristic because the optimum is
exactly computable here, and a rule of thumb ("charge in the cheapest four
hours") would leave money on the table without making it obvious how much.

Mixed-integer rather than purely linear: one binary per hour records which way
the inverter is running. A pure LP charges and discharges simultaneously
whenever prices go negative, which is profitable on paper and impossible in
practice.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

HOURS_PER_DAY = 24


@dataclass(frozen=True)
class Battery:
    """A battery, in MWh and MW.

    Defaults describe a typical German home system: 10 kWh usable, 5 kW inverter,
    90% round-trip. Efficiency is split evenly between charging and discharging,
    so each leg is sqrt(0.9).
    """

    energy_mwh: float = 0.010
    power_mw: float = 0.005
    round_trip_efficiency: float = 0.90

    @property
    def leg_efficiency(self) -> float:
        return float(np.sqrt(self.round_trip_efficiency))

    def __post_init__(self):
        if self.energy_mwh <= 0 or self.power_mw <= 0:
            raise ValueError("energy and power must be positive")
        if not 0 < self.round_trip_efficiency <= 1:
            raise ValueError("round-trip efficiency must be in (0, 1]")


@dataclass
class Schedule:
    """One day of dispatch."""

    charge_mw: np.ndarray
    discharge_mw: np.ndarray
    state_of_charge_mwh: np.ndarray
    revenue_eur: float

    @property
    def energy_cycled_mwh(self) -> float:
        return float(self.discharge_mw.sum())


def optimise_day(prices_eur_mwh: np.ndarray, battery: Battery) -> Schedule:
    """Optimal dispatch for one day.

    Args:
        prices_eur_mwh: one price per hour.
        battery: the system being dispatched.

    Variables are [charge per hour, discharge per hour, state of charge per hour],
    so 3N in total. The state of charge is carried as an explicit variable rather
    than derived, which keeps the energy balance a linear equality.
    """
    prices = np.asarray(prices_eur_mwh, dtype=float)
    if np.isnan(prices).any():
        raise ValueError("prices contain NaN")

    n = len(prices)
    efficiency = battery.leg_efficiency
    power, capacity = battery.power_mw, battery.energy_mwh

    # Variables: charge, discharge, state of charge, and a binary per hour saying
    # which direction the inverter is running.
    width = 4 * n
    charge_at = lambda t: t  # noqa: E731
    discharge_at = lambda t: n + t  # noqa: E731
    soc_at = lambda t: 2 * n + t  # noqa: E731
    mode_at = lambda t: 3 * n + t  # noqa: E731

    # Maximise revenue, so minimise its negative. Discharging earns the price,
    # charging pays it. The binaries carry no cost.
    objective = np.concatenate([prices, -prices, np.zeros(2 * n)])

    rows, lower, upper = [], [], []

    # Energy balance: soc[t] - soc[t-1] - eff * charge[t] + discharge[t] / eff = 0
    initial = capacity / 2
    for t in range(n):
        row = np.zeros(width)
        row[charge_at(t)] = -efficiency
        row[discharge_at(t)] = 1.0 / efficiency
        row[soc_at(t)] = 1.0
        if t > 0:
            row[soc_at(t - 1)] = -1.0
        rows.append(row)
        bound = initial if t == 0 else 0.0
        lower.append(bound)
        upper.append(bound)

    # Finish where the day started, so revenue cannot come from net depletion.
    closing = np.zeros(width)
    closing[soc_at(n - 1)] = 1.0
    rows.append(closing)
    lower.append(initial)
    upper.append(initial)

    # One inverter, one direction at a time. Without this the solver charges and
    # discharges in the same hour whenever prices are negative: being paid to
    # consume makes it profitable to circulate energy and burn it as round-trip
    # losses. Economically rational, physically impossible, and it would destroy
    # the battery. Bounding only the sum is not enough — the solver simply splits
    # the inverter's rating between the two directions.
    for t in range(n):
        charging = np.zeros(width)
        charging[charge_at(t)] = 1.0
        charging[mode_at(t)] = -power
        rows.append(charging)
        lower.append(-np.inf)
        upper.append(0.0)

        discharging = np.zeros(width)
        discharging[discharge_at(t)] = 1.0
        discharging[mode_at(t)] = power
        rows.append(discharging)
        lower.append(-np.inf)
        upper.append(power)

    result = milp(
        c=objective,
        constraints=LinearConstraint(np.vstack(rows), lower, upper),
        integrality=np.concatenate([np.zeros(3 * n), np.ones(n)]),
        bounds=Bounds(
            lb=np.zeros(width),
            ub=np.concatenate(
                [
                    np.full(n, power),
                    np.full(n, power),
                    np.full(n, capacity),
                    np.ones(n),
                ]
            ),
        ),
    )

    if not result.success:
        raise RuntimeError(f"dispatch did not solve: {result.message}")

    charge = result.x[:n]
    discharge = result.x[n : 2 * n]

    return Schedule(
        charge_mw=charge,
        discharge_mw=discharge,
        # Bounded slice: the tail of the solution holds the mode binaries.
        state_of_charge_mwh=result.x[2 * n : 3 * n],
        revenue_eur=float(np.dot(prices, discharge - charge)),
    )


def optimise_series(
    prices: pd.Series,
    battery: Battery,
    *,
    decision_prices: pd.Series | None = None,
) -> pd.DataFrame:
    """Dispatch a whole price series, one day at a time.

    Args:
        prices: actual prices, used to settle revenue.
        battery: the system being dispatched.
        decision_prices: the prices the schedule is *planned* against. Defaults
            to `prices`, which is perfect foresight and therefore an upper bound.
            Passing a forecast instead gives what a real operator would capture:
            the plan is made on the forecast, the money is settled at the actual.

    Returns one row per day.
    """
    planning = prices if decision_prices is None else decision_prices

    frame = pd.DataFrame({"actual": prices, "planning": planning}).dropna()
    frame["day"] = frame.index.tz_convert("Europe/Berlin").date

    rows = []
    for day, group in frame.groupby("day"):
        if len(group) != HOURS_PER_DAY:
            # Clock-change days and truncated ends are skipped rather than
            # padded, which would invent prices.
            continue

        schedule = optimise_day(group["planning"].to_numpy(), battery)
        settled = float(
            np.dot(group["actual"].to_numpy(), schedule.discharge_mw - schedule.charge_mw)
        )

        rows.append(
            {
                "day": day,
                "revenue_eur": settled,
                "planned_revenue_eur": schedule.revenue_eur,
                "energy_cycled_mwh": schedule.energy_cycled_mwh,
                "price_spread_eur_mwh": float(
                    group["actual"].max() - group["actual"].min()
                ),
            }
        )

    return pd.DataFrame(rows).set_index("day")


def annualise(daily: pd.DataFrame, battery: Battery) -> dict[str, float]:
    """Scale observed daily revenue to a yearly figure.

    Cycles are reported because they are the cost side of this trade. Every cycle
    consumes warranty life, and a schedule earning a few more euros by cycling
    twice as hard is not obviously better. Degradation is not modelled here, so
    the cycle count is the number to check against a warranty.
    """
    if daily.empty:
        raise ValueError("no days were dispatched")

    mean_daily = float(daily["revenue_eur"].mean())
    cycles_per_day = float(daily["energy_cycled_mwh"].mean()) / battery.energy_mwh

    return {
        "days": len(daily),
        "total_eur": float(daily["revenue_eur"].sum()),
        "mean_daily_eur": mean_daily,
        "annual_eur": mean_daily * 365.25,
        "cycles_per_day": cycles_per_day,
        "cycles_per_year": cycles_per_day * 365.25,
    }