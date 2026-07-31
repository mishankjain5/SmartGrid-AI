"""Warehouse helper tests.

Tests that reach BigQuery are marked `cloud` and skipped unless GCP_PROJECT is
set, so the suite passes on a fresh clone without credentials.
"""

import pandas as pd
import pytest

from smartgrid.config import get_settings
from smartgrid.warehouse import client

requires_cloud = pytest.mark.skipif(
    not get_settings().bigquery_available,
    reason="GCP_PROJECT not set",
)


def test_references_are_fully_qualified(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT", "example-project")
    assert client.dataset_reference("raw") == "example-project.raw"
    assert client.table_reference("raw", "price") == "example-project.raw.price"


def test_query_parameters_cover_the_common_types():
    cases = {"a": True, "b": 1, "c": 1.5, "d": "x", "e": pd.Timestamp("2026-01-01")}
    expected = ["BOOL", "INT64", "FLOAT64", "STRING", "TIMESTAMP"]

    built = [client._query_parameter(k, v).type_ for k, v in cases.items()]
    assert built == expected


def test_unsupported_parameter_type_is_rejected():
    with pytest.raises(TypeError, match="unsupported query parameter"):
        client._query_parameter("bad", {"not": "scalar"})


def test_client_requires_a_project(monkeypatch):
    monkeypatch.delenv("GCP_PROJECT", raising=False)
    client.get_client.cache_clear()

    with pytest.raises(RuntimeError, match="GCP_PROJECT"):
        client.get_client()

    client.get_client.cache_clear()


@requires_cloud
def test_query_round_trip():
    frame = client.query("SELECT @n AS n, 'ok' AS status", n=7)
    assert frame.loc[0, "n"] == 7
    assert frame.loc[0, "status"] == "ok"


@requires_cloud
def test_dataset_can_be_created_and_detected():
    name = "smartgrid_selftest"
    try:
        assert client.ensure_dataset(name).endswith(f".{name}")
        assert client.dataset_exists(name)
    finally:
        client.get_client().delete_dataset(
            client.dataset_reference(name), delete_contents=True, not_found_ok=True
        )

    assert not client.dataset_exists(name)
