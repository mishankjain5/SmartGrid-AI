"""Turning a forecast into a recommendation.

A solar forecast on its own is a number. What a customer wants is an answer to
"what should my battery do tomorrow, and what does it save me?"

Tomorrow's prices are *published*, not predicted: the day-ahead auction clears at
noon and results appear shortly after. So the uncertain input is solar output,
which is what the model supplies. The battery schedule follows from both.

The saving is stated against doing nothing — leaving the battery idle and buying
at whatever the price happens to be. That is the honest comparison, because it is
what the customer does today.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from smartgrid.config import MARKET_TIMEZONE
from smartgrid.optimisation.battery import Battery, optimise_day

#: Fallback draw in MW, used only when no measured load profile is supplied.
#: Roughly 9.6 kWh a day — about half what the metered household in the OPSD
#: panel actually uses, which is why a profile is preferred.
TYPICAL_HOUSEHOLD_LOAD_MW = 0.0004


@dataclass
class DayPlan:
    """One day's recommendation."""

    target_date: pd.Timestamp
    hours: pd.DataFrame
    battery: Battery
    revenue_eur: float
    solar_value_eur: float

    @property
    def total_benefit_eur(self) -> float:
        return self.revenue_eur + self.solar_value_eur

    @property
    def cheapest_hours(self) -> list[int]:
        cheap = self.hours.nsmallest(3, "price_eur_mwh")
        return sorted(cheap["local_hour"].tolist())

    @property
    def dearest_hours(self) -> list[int]:
        dear = self.hours.nlargest(3, "price_eur_mwh")
        return sorted(dear["local_hour"].tolist())

    def advice(self) -> list[str]:
        """Plain statements a customer could act on."""
        charge = self.hours[self.hours["charge_kw"] > 0.01]
        discharge = self.hours[self.hours["discharge_kw"] > 0.01]

        lines = [
            f"Cheapest power at {_join_hours(self.cheapest_hours)}; "
            f"dearest at {_join_hours(self.dearest_hours)}.",
        ]

        if not charge.empty:
            lines.append(
                f"Charge from the grid around {_join_hours(charge['local_hour'].tolist())}."
            )
        if not discharge.empty:
            lines.append(
                f"Discharge around {_join_hours(discharge['local_hour'].tolist())}."
            )

        solar_peak = self.hours.loc[self.hours["predicted_solar_kw"].idxmax()]
        lines.append(
            f"Solar peaks near {int(solar_peak['local_hour']):02d}:00 at "
            f"{solar_peak['predicted_solar_kw']:.1f} kW."
        )
        lines.append(
            f"Expected benefit on {self.target_date.date()}: "
            f"EUR {self.total_benefit_eur:.2f} "
            f"({self.revenue_eur:.2f} from the battery, "
            f"{self.solar_value_eur:.2f} from solar offsetting purchases)."
        )
        return lines


def _join_hours(hours: list[int]) -> str:
    return ", ".join(f"{int(h):02d}:00" for h in sorted(set(hours)))


def plan_day(
    prediction: pd.DataFrame,
    prices: pd.Series,
    *,
    battery: Battery | None = None,
    system_kwp: float = 10.0,
    load_kwh: pd.Series | None = None,
    household_load_mw: float = TYPICAL_HOUSEHOLD_LOAD_MW,
) -> DayPlan:
    """Build tomorrow's recommendation.

    Args:
        prediction: output of `predict_day`, one row per hour.
        prices: published day-ahead prices, indexed by UTC timestamp.
        battery: the customer's system.
        system_kwp: the customer's PV array. The national capacity factor is
            applied to it, which assumes their roof behaves like the national
            fleet — reasonable for a rough figure, wrong in detail for any one
            roof's orientation and shading.
        load_kwh: forecast consumption per hour, indexed by UTC timestamp. From
            `household.LoadProfile.for_day`. Falls back to a flat draw when
            absent, which understates a real home's usage by roughly half.
        household_load_mw: the flat fallback, used only when `load_kwh` is None.
    """
    battery = battery or Battery()

    frame = prediction.set_index("utc_timestamp").join(
        prices.rename("price_eur_mwh"), how="inner"
    )
    if len(frame) < 24:
        raise ValueError(
            f"need 24 priced hours to plan a day, got {len(frame)}. "
            "Tomorrow's auction may not have cleared yet."
        )

    schedule = optimise_day(frame["price_eur_mwh"].to_numpy(), battery)

    # The customer's array, scaled from the national capacity factor.
    solar_mw = frame["predicted_capacity_factor"].to_numpy() * (system_kwp / 1000)

    if load_kwh is not None:
        # kWh in an hour is kW, and MW is a thousandth of that.
        load_mw = load_kwh.reindex(frame.index).to_numpy() / 1000
        load_mw = np.where(np.isnan(load_mw), household_load_mw, load_mw)
    else:
        load_mw = np.full(len(frame), household_load_mw)

    # Solar covers demand first; only the surplus is exported. Self-consumption
    # is worth the price that would otherwise have been paid for it.
    self_consumed_mw = np.minimum(solar_mw, load_mw)
    solar_value = float(np.dot(frame["price_eur_mwh"].to_numpy(), self_consumed_mw))

    local = frame.index.tz_convert(MARKET_TIMEZONE)

    # The solver returns values like -1e-17 where it means zero. Rounding to the
    # watt keeps "-0.00" out of a table a customer reads.
    def clean(values: np.ndarray) -> np.ndarray:
        return np.round(np.where(np.abs(values) < 1e-9, 0.0, values), 6)

    hours = pd.DataFrame(
        {
            "local_hour": local.hour,
            "price_eur_mwh": frame["price_eur_mwh"].to_numpy(),
            "predicted_solar_kw": clean(solar_mw * 1000),
            "expected_load_kw": clean(load_mw * 1000),
            "charge_kw": clean(schedule.charge_mw * 1000),
            "discharge_kw": clean(schedule.discharge_mw * 1000),
            "state_of_charge_kwh": clean(schedule.state_of_charge_mwh * 1000),
        },
        index=frame.index,
    )

    return DayPlan(
        target_date=pd.Timestamp(prediction["target_date"].iloc[0]),
        hours=hours,
        battery=battery,
        revenue_eur=schedule.revenue_eur,
        solar_value_eur=solar_value,
    )
