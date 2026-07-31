import pytest

from smartgrid import config


def test_project_root_points_at_the_repository():
    assert (config.PROJECT_ROOT / "pyproject.toml").is_file()
    assert (config.PROJECT_ROOT / "src" / "smartgrid").is_dir()


def test_data_paths_are_inside_the_project():
    for path in (config.DATA_DIR, config.RAW_DATA_DIR, config.NOTEBOOKS_DIR):
        assert config.PROJECT_ROOT in path.parents


def test_ensure_data_dirs_is_idempotent():
    config.ensure_data_dirs()
    config.ensure_data_dirs()
    assert config.RAW_DATA_DIR.is_dir()


def test_market_constants_match_the_day_ahead_auction():
    assert config.MARKET_TIMEZONE == "Europe/Berlin"
    assert config.GATE_HOUR == 12
    assert config.LEAD_DAYS == 1


def test_settings_read_the_environment(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT", "example-project")
    monkeypatch.setenv("GCP_LOCATION", "US")

    settings = config.get_settings()
    assert settings.gcp_project == "example-project"
    assert settings.gcp_location == "US"
    assert settings.bigquery_available
    assert settings.require_project() == "example-project"


def test_location_defaults_to_eu(monkeypatch):
    monkeypatch.delenv("GCP_LOCATION", raising=False)
    assert config.get_settings().gcp_location == "EU"


def test_missing_project_raises_with_a_usable_message(monkeypatch):
    monkeypatch.delenv("GCP_PROJECT", raising=False)

    settings = config.get_settings()
    assert not settings.bigquery_available
    with pytest.raises(RuntimeError, match="GCP_PROJECT"):
        settings.require_project()
