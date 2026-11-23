"""Monthly SEC EDGAR ingestion, Hive-partitioned text chunking, Qdrant sync, and dbt Gold build DAG."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "cfi-data-engineering",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def inspect_dead_letter_manifest() -> dict[str, int]:
    """Audit text pipeline dead-letter manifest after Hive chunking without blocking healthy tickers."""
    manifest_path = Path("/opt/airflow/data/dead_letter/text_pipeline_dead_letter.json")
    if not manifest_path.exists():
        logger.info("No dead-letter manifest found; 0 failed filings.")
        return {"failed_count": 0}

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    failed_count = int(payload.get("failed_count", 0))
    if failed_count > 0:
        logger.warning(
            "Text pipeline recorded %d failed filing(s) in dead-letter manifest: %s",
            failed_count,
            [e.get("s3_key") for e in payload.get("entries", [])],
        )
    else:
        logger.info("Dead-letter manifest exists with 0 failed filings.")
    return {"failed_count": failed_count}


with DAG(
    dag_id="dag_sec_monthly_ingestion",
    description="Monthly SEC EDGAR incremental ingestion -> Hive text chunking -> Qdrant sync -> dbt Gold marts",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 11, 1),
    schedule="0 2 1 * *",
    catchup=False,
    max_active_runs=1,
    tags=["sec", "lakehouse", "qdrant", "dbt"],
) as dag:
    ingest_sec_filings = BashOperator(
        task_id="ingest_sec_filings",
        bash_command="cd /opt/airflow && python -m src.cli ingest --all --limit 4",
    )

    process_sec_facts = BashOperator(
        task_id="process_sec_facts",
        bash_command="cd /opt/airflow && python -m src.cli process --facts",
    )

    process_sec_text_hive = BashOperator(
        task_id="process_sec_text_hive",
        bash_command="cd /opt/airflow && python -m src.cli process --text",
    )

    check_dead_letter = PythonOperator(
        task_id="check_dead_letter_manifest",
        python_callable=inspect_dead_letter_manifest,
    )

    sync_qdrant_vectors = BashOperator(
        task_id="sync_qdrant_vectors",
        bash_command="cd /opt/airflow && python -m src.cli index --sync --verify",
    )

    dbt_build_gold_marts = BashOperator(
        task_id="dbt_build_gold_marts",
        bash_command="cd /opt/airflow/transform && dbt build --target docker",
    )

    ingest_sec_filings >> [process_sec_facts, process_sec_text_hive]
    process_sec_text_hive >> check_dead_letter >> sync_qdrant_vectors
    process_sec_facts >> dbt_build_gold_marts
    sync_qdrant_vectors >> dbt_build_gold_marts
