"""Open Power System Data household client.

https://data.open-power-system-data.org/household_data/2020-04-15/

Behind-the-meter metering from eleven German buildings, several with rooftop PV,
a heat pump, an EV or a battery. OPSD stopped publishing in 2020, so this covers
2014-12 to 2019-05 and is used for the household side of the project only; grid
data comes from Energy-Charts.

Values are cumulative kWh meter readings, not energy consumed during the hour.
They are returned here as stored; differencing belongs in the transformation
layer, where a gap or meter reset can be handled explicitly.
"""

from pathlib import Path

import pandas as pd
import requests

from smartgrid.config import RAW_DATA_DIR

URL = (
    "https://data.open-power-system-data.org/household_data/2020-04-15/"
    "household_data_60min_singleindex.csv"
)
CACHE_PATH = RAW_DATA_DIR / "opsd" / "household_data_60min_singleindex.csv"
TIMEOUT_SECONDS = 300
CHUNK_BYTES = 1024 * 1024

CHANNEL_PREFIX = "DE_KN_"


def download(*, refresh: bool = False) -> Path:
    """Fetch the CSV to the local cache. Roughly 15 MB."""
    if CACHE_PATH.exists() and not refresh:
        return CACHE_PATH

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = CACHE_PATH.with_suffix(CACHE_PATH.suffix + ".part")

    with requests.get(URL, timeout=TIMEOUT_SECONDS, stream=True) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                handle.write(chunk)

    # Rename only on success so an interrupted download is never mistaken for a
    # complete one by the check above.
    partial.replace(CACHE_PATH)
    return CACHE_PATH


def channel_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith(CHANNEL_PREFIX)]


def load(*, refresh: bool = False) -> pd.DataFrame:
    """Load the household CSV in long form.

    Returns one row per timestamp and channel, with the building, building type
    and device parsed out of the channel name.
    """
    path = download(refresh=refresh)

    wide = pd.read_csv(path, parse_dates=["utc_timestamp"])
    channels = channel_columns(wide)

    long_form = wide.melt(
        id_vars="utc_timestamp",
        value_vars=channels,
        var_name="channel",
        value_name="meter_kwh",
    ).dropna(subset=["meter_kwh"])

    # Channel names are DE_KN_<building><n>_<device>, e.g. DE_KN_residential4_pv.
    parts = long_form["channel"].str.extract(
        rf"^{CHANNEL_PREFIX}(?P<building>(?P<building_type>[a-z]+)\d*)_(?P<device>.+)$"
    )

    return pd.concat([long_form, parts], axis=1).sort_values(
        ["channel", "utc_timestamp"], ignore_index=True
    )
