"""Streamlit dashboard over the BigQuery marts.

    streamlit run src/smartgrid/dashboard.py

Three views: what the grid did, how the solar forecast performs against the
transmission operators' own forecast, and what a battery would earn trading
against day-ahead prices.

Queries are cached, so moving the date range does not re-bill BigQuery for data
already fetched.
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from smartgrid.config import get_settings
from smartgrid.optimisation import Battery, annualise, optimise_series
from smartgrid.viz import style
from smartgrid.warehouse import query

CACHE_SECONDS = 3600


@st.cache_data(ttl=CACHE_SECONDS)
def load_solar() -> pd.DataFrame:
    project = get_settings().require_project()
    frame = query(
        f"""
        SELECT utc_timestamp, local_datetime, hour_of_day,
               solar_capacity_factor, solar_mw, solar_ac_mw,
               tso_solar_forecast_capacity_factor, ghi_mean, ghi_max, ghi_spread
        FROM `{project}.marts.mart_solar_features`
        ORDER BY utc_timestamp
        """
    )
    frame["local_datetime"] = pd.to_datetime(frame["local_datetime"])
    return frame


@st.cache_data(ttl=CACHE_SECONDS)
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


@st.cache_data(ttl=CACHE_SECONDS)
def dispatch(energy_kwh: float, power_kw: float, round_trip: float) -> dict:
    battery = Battery(
        energy_mwh=energy_kwh / 1000,
        power_mw=power_kw / 1000,
        round_trip_efficiency=round_trip,
    )
    prices = load_prices()

    perfect = annualise(optimise_series(prices, battery), battery)
    naive = annualise(
        optimise_series(prices, battery, decision_prices=prices.shift(24)), battery
    )
    return {"perfect": perfect, "naive": naive}


def solar_view(solar: pd.DataFrame) -> None:
    st.subheader("Solar forecast against the operators' own")

    daylight = solar[
        (solar["ghi_max"] > 0) & solar["tso_solar_forecast_capacity_factor"].notna()
    ].copy()
    daylight["tso_error"] = (
        daylight["tso_solar_forecast_capacity_factor"] - daylight["solar_capacity_factor"]
    ).abs()

    left, middle, right = st.columns(3)
    left.metric("Daylight hours", f"{len(daylight):,}")
    middle.metric("TSO forecast MAE", f"{daylight['tso_error'].mean():.4f}")
    right.metric(
        "Peak capacity factor", f"{solar['solar_capacity_factor'].max():.2f}"
    )
    st.caption(
        "Capacity-factor units. Night is excluded: output is exactly zero then and "
        "every method predicts it correctly, which flatters all of them equally."
    )

    recent = solar[solar["local_datetime"] >= solar["local_datetime"].max() - pd.Timedelta(14, "D")]

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(recent["local_datetime"], recent["solar_capacity_factor"],
            color=style.BLUE, label="actual")
    ax.plot(recent["local_datetime"], recent["tso_solar_forecast_capacity_factor"],
            color=style.ORANGE, linewidth=1.2, label="TSO forecast")
    ax.set_ylabel("capacity factor")
    ax.legend(loc="upper left")
    style.titled(ax, "Last two weeks")
    st.pyplot(fig)

    by_hour = daylight.groupby("hour_of_day")["tso_error"].mean()
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(by_hour.index, by_hour.to_numpy(), color=style.BLUE, marker="o")
    ax.set_xlabel("hour of day (Europe/Berlin)")
    ax.set_ylabel("mean absolute error")
    ax.set_xticks(range(0, 24, 3))
    style.titled(ax, "Forecast error is hardest around midday")
    st.pyplot(fig)


def price_view(prices: pd.Series) -> None:
    st.subheader("Day-ahead prices")

    negative = (prices < 0).mean()
    local_hour = prices.index.tz_convert("Europe/Berlin").hour
    shape = prices.groupby(local_hour).mean()

    left, middle, right = st.columns(3)
    left.metric("Hours priced", f"{len(prices):,}")
    middle.metric("Negative hours", f"{negative:.1%}")
    right.metric("Daily spread", f"{shape.max() - shape.min():.0f} EUR/MWh")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].hist(prices, bins=120, color=style.BLUE)
    axes[0].axvline(0, color=style.CRITICAL, linewidth=1)
    axes[0].set_xlim(-100, 400)
    axes[0].set_xlabel("EUR/MWh")
    style.titled(axes[0], "Distribution")

    axes[1].plot(shape.index, shape.to_numpy(), color=style.BLUE, marker="o")
    axes[1].set_xlabel("hour of day")
    axes[1].set_xticks(range(0, 24, 3))
    style.titled(axes[1], "Daily shape")
    st.pyplot(fig)

    st.caption(
        "The midday dip is solar pushing prices down; the two peaks are why a "
        "battery can cycle twice a day."
    )


def battery_view() -> None:
    st.subheader("Battery dispatch")

    left, middle, right = st.columns(3)
    energy_kwh = left.slider("Capacity (kWh)", 5.0, 50.0, 10.0, step=5.0)
    power_kw = middle.slider("Power (kW)", 2.5, 25.0, 5.0, step=2.5)
    round_trip = right.slider("Round-trip efficiency", 0.70, 1.00, 0.90, step=0.05)

    with st.spinner("Solving dispatch..."):
        result = dispatch(energy_kwh, power_kw, round_trip)

    perfect, naive = result["perfect"], result["naive"]
    capture = naive["annual_eur"] / perfect["annual_eur"]

    left, middle, right = st.columns(3)
    left.metric("Perfect foresight", f"EUR {perfect['annual_eur']:.0f}/yr")
    middle.metric(
        "Previous-day prices",
        f"EUR {naive['annual_eur']:.0f}/yr",
        delta=f"{capture - 1:.0%} vs perfect",
    )
    right.metric("Cycles per year", f"{naive['cycles_per_year']:.0f}")

    st.caption(
        f"Perfect foresight is an upper bound, not an achievable result. The gap "
        f"of EUR {perfect['annual_eur'] - naive['annual_eur']:.0f}/year is what a "
        f"price forecast would be worth. Degradation is not modelled, so check the "
        f"cycle count against a warranty before believing the revenue."
    )


def main() -> None:
    st.set_page_config(page_title="SmartGrid-AI", layout="wide")
    style.apply_style()

    st.title("SmartGrid-AI")
    st.caption(
        "German day-ahead solar forecasting and battery dispatch, built on "
        "Energy-Charts, Open-Meteo and BigQuery."
    )

    if not get_settings().bigquery_available:
        st.error("GCP_PROJECT is not set. Copy .env.example to .env and fill it in.")
        return

    solar_tab, price_tab, battery_tab = st.tabs(["Solar", "Prices", "Battery"])

    with solar_tab:
        solar_view(load_solar())
    with price_tab:
        price_view(load_prices())
    with battery_tab:
        battery_view()


main()