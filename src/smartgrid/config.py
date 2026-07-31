"""Project paths and environment settings."""

import os
from dataclasses import dataclass
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
