"""BigQuery client and dataset helpers.

Authentication uses Application Default Credentials:

    gcloud auth application-default login
"""

from functools import lru_cache

import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

from smartgrid.config import get_settings


@lru_cache(maxsize=1)
def get_client() -> bigquery.Client:
    """Client bound to the configured project and location.

    Cached because constructing one resolves credentials, which is slow enough
    to matter when called per operation.
    """
    settings = get_settings()
    return bigquery.Client(
        project=settings.require_project(),
        location=settings.gcp_location,
    )


def dataset_reference(dataset: str) -> str:
    return f"{get_settings().require_project()}.{dataset}"


def table_reference(dataset: str, table: str) -> str:
    return f"{dataset_reference(dataset)}.{table}"


def dataset_exists(dataset: str) -> bool:
    try:
        get_client().get_dataset(dataset_reference(dataset))
    except NotFound:
        return False
    return True


def ensure_dataset(dataset: str) -> str:
    """Create the dataset if absent. Returns its fully qualified name."""
    settings = get_settings()
    reference = bigquery.Dataset(dataset_reference(dataset))
    reference.location = settings.gcp_location

    get_client().create_dataset(reference, exists_ok=True)
    return dataset_reference(dataset)


def query(sql: str, **parameters) -> pd.DataFrame:
    """Run a query and return the result as a DataFrame.

    Keyword arguments become named query parameters, so values are never
    interpolated into SQL text.
    """
    job_config = None
    if parameters:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                _query_parameter(name, value) for name, value in parameters.items()
            ]
        )

    return get_client().query(sql, job_config=job_config).result().to_dataframe()


def _query_parameter(name: str, value) -> bigquery.ScalarQueryParameter:
    types = {bool: "BOOL", int: "INT64", float: "FLOAT64", str: "STRING"}
    for python_type, bigquery_type in types.items():
        if isinstance(value, python_type):
            return bigquery.ScalarQueryParameter(name, bigquery_type, value)

    if isinstance(value, pd.Timestamp):
        return bigquery.ScalarQueryParameter(name, "TIMESTAMP", value)

    raise TypeError(f"unsupported query parameter type for {name!r}: {type(value)}")
