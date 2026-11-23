"""Weekly VN30 financial statement ingestion, VAS parsing, and dbt Gold mart refresh DAG."""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

DEFAULT_ARGS = {
    "owner": "cfi-data-engineering",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="dag_vn30_metrics_sync",
    description="Periodic VN30 financial statement sync -> VAS Silver parser -> dbt Gold marts",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 11, 1),
    schedule="0 3 * * 1",
    catchup=False,
    max_active_runs=1,
    tags=["vn30", "vas", "dbt"],
) as dag:
    ingest_vn30_financials = BashOperator(
        task_id="ingest_vn30_financials",
        bash_command="cd /opt/airflow && python -m src.cli ingest --vn",
    )

    process_vn30_facts = BashOperator(
        task_id="process_vn30_facts",
        bash_command="cd /opt/airflow && python -m src.cli process --vn",
    )

    dbt_build_vn30_gold = BashOperator(
        task_id="dbt_build_vn30_gold",
        bash_command="cd /opt/airflow/transform && dbt build --target docker",
    )

    ingest_vn30_financials >> process_vn30_facts >> dbt_build_vn30_gold
