"""Streamlit dashboard.

    streamlit run src/smartgrid/dashboard.py

Leads with tomorrow: the solar forecast, the battery schedule it implies, and
what that is worth. Accuracy and market context sit behind it, because they
justify the forecast rather than being the point of it.

Every expensive step is cached. Training the model and solving 24 hours of
dispatch takes seconds; the historical backtest takes a minute or two.
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from smartgrid.config import MARKET_TIMEZONE, get_settings
from smartgrid.modelling import default_models, load_features, modelling_frame
from smartgrid.modelling.backtest import run_backtest
from smartgrid.modelling.predict import predict_day, prices_expected, target_day
from smartgrid.optimisation import Battery, plan_day
from smartgrid.tomorrow import load_published_prices
from smartgrid.viz import style
from smartgrid.warehouse import query

FORECAST_TTL = 1800
HISTORY_TTL = 3600


@st.cache_data(ttl=FORECAST_TTL, show_spinner=False)
def cached_forecast(day_iso: str) -> pd.DataFrame:
    return predict_day(day=pd.Timestamp(day_iso).tz_localize(MARKET_TIMEZONE))


@st.cache_data(ttl=FORECAST_TTL, show_spinner=False)
def cached_prices() -> pd.Series:
    return load_published_prices()


@st.cache_data(ttl=HISTORY_TTL, show_spinner=False)
def cached_history() -> pd.DataFrame:
    project = get_settings().require_project()
    frame = query(
        f"""
        SELECT utc_timestamp, local_datetime, hour_of_day, month_of_year,
               solar_capacity_factor, tso_solar_forecast_capacity_factor, ghi_max
        FROM `{project}.marts.mart_solar_features`
        ORDER BY utc_timestamp
        """
    )
    frame["local_datetime"] = pd.to_datetime(frame["local_datetime"])
    return frame


@st.cache_data(ttl=HISTORY_TTL, show_spinner=False)
def cached_backtest() -> pd.DataFrame:
    frame = modelling_frame(load_features())
    result = run_backtest(frame, default_models(), test_days=60, verbose=False)
    return result.summary()


# --- tomorrow ---------------------------------------------------------------


def tomorrow_view(battery: Battery, system_kwp: float) -> None:
    day = target_day()
    st.subheader(f"Forecast for {day.strftime('%A %d %B %Y')}")

    with st.spinner("Training on all history and forecasting..."):
        prediction = cached_forecast(str(day.date()))
        prices = cached_prices()

    generation_kwh = float(
        (prediction["predicted_capacity_factor"] * system_kwp).sum()
    )
    covered = prices.loc[prices.index >= day.tz_convert("UTC")]

    if len(covered) < 24:
        st.warning(
            f"Only {len(covered)} of 24 hours are priced for {day.date()}. "
            + (
                "The auction has cleared but the feed has not caught up yet."
                if prices_expected()
                else "The day-ahead auction clears at 12:00 and publishes around "
                "12:45 market time."
            )
        )
        st.metric("Predicted generation", f"{generation_kwh:.1f} kWh")
        _solar_chart(prediction, system_kwp)
        return

    plan = plan_day(prediction, prices, battery=battery, system_kwp=system_kwp)

    left, middle, right = st.columns(3)
    left.metric("Predicted generation", f"{generation_kwh:.1f} kWh")
    middle.metric("Expected benefit", f"EUR {plan.total_benefit_eur:.2f}")
    right.metric(
        "Price range",
        f"{plan.hours['price_eur_mwh'].min():.0f}-"
        f"{plan.hours['price_eur_mwh'].max():.0f} EUR/MWh",
    )

    st.markdown("**What to do**")
    for line in plan.advice():
        st.markdown(f"- {line}")

    _plan_chart(plan)

    with st.expander("Hour by hour"):
        table = plan.hours.reset_index(drop=True)
        st.dataframe(table.round(2), width="stretch")


def _solar_chart(prediction: pd.DataFrame, system_kwp: float) -> None:
    local = pd.to_datetime(prediction["local_datetime"])
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.fill_between(
        local,
        prediction["predicted_capacity_factor"] * system_kwp,
        color=style.BLUE,
        alpha=0.25,
        linewidth=0,
    )
    ax.plot(local, prediction["predicted_capacity_factor"] * system_kwp, color=style.BLUE)
    ax.set_ylabel("kW")
    style.titled(ax, "Predicted output")
    st.pyplot(fig)


def _plan_chart(plan) -> None:
    hours = plan.hours
    local = hours.index.tz_convert(MARKET_TIMEZONE)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    axes[0].fill_between(local, hours["predicted_solar_kw"], color=style.BLUE,
                         alpha=0.25, linewidth=0)
    axes[0].plot(local, hours["predicted_solar_kw"], color=style.BLUE)
    axes[0].set_ylabel("solar kW")
    style.titled(axes[0], "Predicted generation")

    axes[1].plot(local, hours["price_eur_mwh"], color=style.ORANGE)
    axes[1].axhline(0, color=style.AXIS, linewidth=0.8)
    axes[1].set_ylabel("EUR/MWh")
    style.titled(axes[1], "Published day-ahead price")

    axes[2].bar(local, hours["charge_kw"], width=0.03, color=style.GREEN, label="charge")
    axes[2].bar(local, -hours["discharge_kw"], width=0.03, color=style.CRITICAL,
                label="discharge")
    axes[2].axhline(0, color=style.AXIS, linewidth=0.8)
    axes[2].set_ylabel("kW")
    axes[2].legend(loc="upper left")
    style.titled(axes[2], "Recommended battery schedule")

    st.pyplot(fig)


# --- accuracy ---------------------------------------------------------------


def accuracy_view() -> None:
    st.subheader("Does the forecast work?")
    st.caption(
        "Walk-forward backtest: train on history up to a point, forecast the next "
        "window, step forward, refit. Scored on daylight hours against the "
        "forecast the transmission operators actually published."
    )

    with st.spinner("Replaying history..."):
        summary = cached_backtest()

    best = summary.index[0]
    ours = summary.loc["gradient_boosting"]
    tso = summary.loc["TSO forecast"]

    left, middle, right = st.columns(3)
    left.metric("Our model, MAE", f"{ours['mae']:.4f}")
    middle.metric("Operators' forecast, MAE", f"{tso['mae']:.4f}")
    right.metric("Ratio", f"{ours['mae'] / tso['mae']:.1f}x")

    st.dataframe(summary.round(4), width="stretch")
    st.caption(
        f"Lowest error: {best}. Capacity-factor units. The operators have "
        "plant-level registry data, live telemetry and multi-model ensembles; "
        "this uses free public data only."
    )


# --- market -----------------------------------------------------------------


def market_view(history: pd.DataFrame) -> None:
    st.subheader("Market context")

    pivot = history.pivot_table(
        index="month_of_year", columns="hour_of_day",
        values="solar_capacity_factor", aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(10, 3.8))
    mesh = ax.pcolormesh(pivot.columns, pivot.index, pivot.to_numpy(),
                         cmap=style.sequential_cmap(), shading="nearest")
    ax.set_xlabel("hour of day (Europe/Berlin)")
    ax.set_yticks(range(1, 13))
    ax.invert_yaxis()
    ax.grid(False)
    style.titled(ax, "Solar output by month and hour", "Mean capacity factor")
    bar = fig.colorbar(mesh, ax=ax, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=9)
    st.pyplot(fig)

    prices = cached_prices()
    local_hour = prices.index.tz_convert(MARKET_TIMEZONE).hour
    shape = prices.groupby(local_hour).mean()

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(shape.index, shape.to_numpy(), color=style.BLUE, marker="o")
    ax.set_xlabel("hour of day")
    ax.set_xticks(range(0, 24, 3))
    ax.set_ylabel("EUR/MWh")
    style.titled(ax, "Recent price shape",
                 "Midday dip from solar, evening peak — the spread a battery trades")
    st.pyplot(fig)


def main() -> None:
    st.set_page_config(page_title="SmartGrid-AI", layout="wide")
    style.apply_style()

    st.title("SmartGrid-AI")
    st.caption(
        "Day-ahead solar forecasting and battery dispatch for German households."
    )

    if not get_settings().bigquery_available:
        st.error("GCP_PROJECT is not set. Copy .env.example to .env and fill it in.")
        return

    with st.sidebar:
        st.header("System")
        system_kwp = st.slider("PV array (kWp)", 2.0, 30.0, 10.0, step=1.0)
        energy_kwh = st.slider("Battery capacity (kWh)", 5.0, 50.0, 10.0, step=5.0)
        power_kw = st.slider("Battery power (kW)", 2.5, 25.0, 5.0, step=2.5)
        round_trip = st.slider("Round-trip efficiency", 0.70, 1.00, 0.90, step=0.05)
        st.caption(
            "The national capacity factor is applied to your array, which assumes "
            "it behaves like the German fleet on average."
        )

    battery = Battery(
        energy_mwh=energy_kwh / 1000,
        power_mw=power_kw / 1000,
        round_trip_efficiency=round_trip,
    )

    tomorrow_tab, accuracy_tab, market_tab = st.tabs(
        ["Tomorrow", "Accuracy", "Market"]
    )

    with tomorrow_tab:
        tomorrow_view(battery, system_kwp)
    with accuracy_tab:
        accuracy_view()
    with market_tab:
        market_view(cached_history())


main()
