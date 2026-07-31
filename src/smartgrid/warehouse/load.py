"""Loading DataFrames into BigQuery.

`load_table_from_dataframe` serialises via Parquet, so the dtypes already
resolved in Python carry through instead of BigQuery re-inferring them from
text. That matters for timestamps and for numeric columns that are sparse at the
start of a series.

Tables are replaced rather than appended: `raw` is a faithful rebuild from
source, so a reload should be idempotent.
"""

import pandas as pd
from google.cloud import bigquery

from smartgrid.config import BIGQUERY_RAW_DATASET
from smartgrid.warehouse.client import ensure_dataset, get_client, table_reference


def load_dataframe(
    frame: pd.DataFrame,
    table: str,
    *,
    dataset: str = BIGQUERY_RAW_DATASET,
    replace: bool = True,
) -> int:
    """Load a frame into `dataset.table`. Returns the resulting row count."""
    if frame.empty:
        raise ValueError(f"refusing to load an empty frame into {dataset}.{table}")

    ensure_dataset(dataset)
    destination = table_reference(dataset, table)

    job_config = bigquery.LoadJobConfig(
        write_disposition=(
            bigquery.WriteDisposition.WRITE_TRUNCATE
            if replace
            else bigquery.WriteDisposition.WRITE_APPEND
        ),
    )

    client = get_client()
    client.load_table_from_dataframe(
        frame, destination, job_config=job_config
    ).result()

    return client.get_table(destination).num_rows


def table_row_count(table: str, *, dataset: str = BIGQUERY_RAW_DATASET) -> int:
    return get_client().get_table(table_reference(dataset, table)).num_rows
