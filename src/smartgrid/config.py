"""Project paths and environment settings."""

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TRANSFORM_DIR = PROJECT_ROOT / "transform"

MARKET_TIMEZONE = "Europe/Berlin"

# The German day-ahead auction closes at 12:00 market time and clears all 24
# hours of the following day.
GATE_HOUR = 12
LEAD_DAYS = 1

# Bounded by the Open-Meteo forecast archive, which begins 2022-01-01. Grid data
# reaches further back but cannot be paired with a weather forecast before this.
DATA_START = date(2022, 1, 1)

# Weather is fetched from later than the rest. Open-Meteo archives day-ahead
# *radiation* only from 2024-01-19 — temperature at that lead goes back to 2022,
# radiation does not — so any honest day-ahead solar model starts in 2024
# regardless. Fetching earlier weather spends API quota on rows the feature table
# discards, and the free tier meters by data volume.
WEATHER_START = date(2024, 1, 1)

BIGQUERY_RAW_DATASET = "raw"


def data_end() -> date:
    """How far forward to request data.

    Today, not yesterday. A day-ahead forecast needs generation history within 48
    hours of the target, and stopping at yesterday leaves the shortest lag short
    by a few hours. Sources return what exists, so requesting today simply yields
    a partial final day; the staging models drop any incomplete trailing hour.
    """
    return date.today()

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Runtime configuration read from the environment."""

    gcp_project: str | None
    gcp_location: str

    @property
    def bigquery_available(self) -> bool:
        return bool(self.gcp_project)

    def require_project(self) -> str:
        if not self.gcp_project:
            raise RuntimeError(
                "GCP_PROJECT is not set. Copy .env.example to .env and fill it in."
            )
        return self.gcp_project


def get_settings() -> Settings:
    return Settings(
        gcp_project=os.environ.get("GCP_PROJECT") or None,
        gcp_location=os.environ.get("GCP_LOCATION", "EU"),
    )


def ensure_data_dirs() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
