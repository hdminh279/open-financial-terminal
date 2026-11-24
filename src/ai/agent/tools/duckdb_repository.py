"""DuckDB repository for querying standardized Gold Parquet financial marts."""

import logging
import os
import re
import threading
from typing import Any

import duckdb

from src.ai.agent.tools.financial_sql_tool import (
    RATIO_METRICS,
    STATEMENT_METRICS,
    FinancialQueryInput,
)

logger = logging.getLogger(__name__)

_READ_ONLY_SQL_RE = re.compile(r"^\s*(?:SELECT|WITH|DESCRIBE|SHOW)\b", re.IGNORECASE)
_MUTATING_SQL_RE = re.compile(
    r"(?:;|\b(?:DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|REPLACE|TRUNCATE|COPY|ATTACH|DETACH|CALL|LOAD|INSTALL)\b)",
    re.IGNORECASE,
)


def _sql_literal(value: str) -> str:
    """Validate and quote a SQL string literal for DuckDB DDL statements."""
    if "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError("SQL configuration literal must not contain NUL or newline characters.")
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


class DatabaseExecutionError(Exception):
    """Raised when an underlying database query or storage connection fails."""

    pass


class DuckDBFinancialRepository:
    """Repository managing DuckDB connections and parameterized queries over Gold Parquet marts."""

    def __init__(
        self,
        statements_path: str | None = None,
        ratios_path: str | None = None,
        connection: duckdb.DuckDBPyConnection | None = None,
    ):
        """Initialize repository with optional custom paths and injected connection.

        Args:
            statements_path: Path to fct_financial_statements (S3 URL or local Parquet path).
            ratios_path: Path to fct_financial_ratios (S3 URL or local Parquet path).
            connection: Optional existing DuckDB connection for testing.
        """
        default_stmt, default_ratio = self.resolve_default_paths()
        self.statements_path = statements_path or default_stmt
        self.ratios_path = ratios_path or default_ratio

        self.conn = connection or duckdb.connect(":memory:")
        self._lock = threading.Lock()
        self._schema_cache: dict[str, set[str]] = {}

        # Automatically configure S3 httpfs if either path targets an S3 bucket
        if self._is_s3_path(self.statements_path) or self._is_s3_path(self.ratios_path):
            self._configure_s3()
            if connection is None:
                # Lock runtime config so DuckDB S3 credentials/endpoints cannot be altered
                self.conn.execute("SET lock_configuration = true;")

    @staticmethod
    def _execute_and_fetch(
        executor: Any,
        sql: str,
        params: list[Any] | None,
    ) -> tuple[list[str], list[tuple[Any, ...]]]:
        """Execute SQL on a connection or cursor and materialize columns and rows."""
        res = executor.execute(sql, params) if params is not None else executor.execute(sql)
        columns = [desc[0] for desc in (res.description or [])]
        return columns, res.fetchall()

    def _run_cursor_query(
        self,
        sql: str,
        params: list[Any] | None = None,
    ) -> tuple[list[str], list[tuple[Any, ...]]]:
        """Run a read-only query using a dedicated DuckDB cursor for concurrent thread safety."""
        if isinstance(self.conn, duckdb.DuckDBPyConnection):
            cursor = self.conn.cursor()
            try:
                return self._execute_and_fetch(cursor, sql, params)
            finally:
                cursor.close()

        # Injected mock connections in unit tests stub execute() directly on the connection
        with self._lock:
            return self._execute_and_fetch(self.conn, sql, params)

    def _has_column(self, parquet_path: str, column_name: str) -> bool:
        """Check whether a Parquet dataset contains a given column (cached per path)."""
        with self._lock:
            cached = self._schema_cache.get(parquet_path)
        if cached is None:
            try:
                _, rows = self._run_cursor_query(
                    "DESCRIBE SELECT * FROM read_parquet(?)",
                    [parquet_path],
                )
                cols = {str(r[0]).lower() for r in rows}
            except (duckdb.Error, OSError):
                cols = set()
            with self._lock:
                self._schema_cache[parquet_path] = cols
                cached = cols
        return column_name.lower() in cached

    def execute_fetchall(self, sql: str, params: list[Any] | None = None) -> list[tuple[Any, ...]]:
        """Execute a parameterized read-only query concurrently via a per-query cursor."""
        stripped = (sql or "").strip()
        if not _READ_ONLY_SQL_RE.match(stripped) or _MUTATING_SQL_RE.search(stripped):
            raise DatabaseExecutionError("Write or multi-statement SQL operations are blocked in read-only mode.")
        _, rows = self._run_cursor_query(sql, params)
        return rows

    @staticmethod
    def resolve_default_paths() -> tuple[str, str]:
        """Resolve Parquet file locations based on execution environment.

        Rules:
        1. Explicit path in env vars (GOLD_STATEMENTS_PARQUET_PATH) takes precedence.
        2. Test runs with USE_TEST_FIXTURES='true' use local Parquet test fixtures.
        3. All operational runs (both Docker and local host) connect directly to
           the authoritative MinIO S3 Gold Lakehouse.
        """
        use_fixtures = os.getenv("USE_TEST_FIXTURES", "").lower() in ("true", "1", "yes")

        if use_fixtures:
            default_stmt = "tests/fixtures/gold/fct_financial_statements.parquet"
            default_ratio = "tests/fixtures/gold/fct_financial_ratios.parquet"
        else:
            default_stmt = "s3://sec-gold/marts/fct_financial_statements.parquet"
            default_ratio = "s3://sec-gold/marts/fct_financial_ratios.parquet"

        statements_path = os.getenv("GOLD_STATEMENTS_PARQUET_PATH", default_stmt)
        ratios_path = os.getenv("GOLD_RATIOS_PARQUET_PATH", default_ratio)
        return statements_path, ratios_path

    @staticmethod
    def _is_s3_path(path: str) -> bool:
        return path.startswith("s3://") or path.startswith("gs://") or path.startswith("gcs://")

    def _configure_s3(self) -> None:
        """Configure DuckDB extension for S3-compatible MinIO or Google Cloud Storage access."""
        is_gcs = (
            self.statements_path.startswith("gs://")
            or self.statements_path.startswith("gcs://")
            or self.ratios_path.startswith("gs://")
            or self.ratios_path.startswith("gcs://")
        )
        if is_gcs:
            try:
                self.conn.execute("LOAD gcs;")
            except duckdb.Error:
                self.conn.execute("INSTALL gcs FROM community; LOAD gcs;")
            # On GCP (Cloud Run), credential_chain automatically uses Application Default Credentials
            try:
                self.conn.execute("CREATE SECRET IF NOT EXISTS (TYPE gcp, PROVIDER credential_chain);")
            except duckdb.Error as exc:
                logger.debug("DuckDB GCS credential_chain setup notice: %s", exc)
            return

        app_env = os.getenv("APP_ENV", "").lower()
        if app_env in ("docker", "prod", "production"):
            raw_endpoint = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
        else:
            raw_endpoint = os.getenv("S3_ENDPOINT_URL_LOCAL", "http://localhost:9000")

        if not raw_endpoint:
            raw_endpoint = "http://localhost:9000"

        use_ssl = raw_endpoint.startswith("https://") or os.getenv("S3_USE_SSL", "").lower() in ("true", "1")
        s3_endpoint = raw_endpoint.replace("http://", "").replace("https://", "").rstrip("/")
        if app_env in ("prod", "production"):
            access_key = (os.getenv("S3_ACCESS_KEY") or "").strip()
            secret_key = (os.getenv("S3_SECRET_KEY") or "").strip()
            if not access_key or not secret_key:
                raise ValueError("S3_ACCESS_KEY and S3_SECRET_KEY are required when APP_ENV is production.")
        else:
            access_key = os.getenv("S3_ACCESS_KEY") or "minioadmin"
            secret_key = os.getenv("S3_SECRET_KEY") or "minioadmin"
        s3_region = (os.getenv("S3_REGION") or "").strip()

        try:
            self.conn.execute("LOAD httpfs;")
        except duckdb.Error:
            self.conn.execute("INSTALL httpfs; LOAD httpfs;")

        # DuckDB DDL rejects '?' placeholders; quote all string options via _sql_literal().
        secret_options = [
            "TYPE S3",
            f"KEY_ID {_sql_literal(access_key)}",
            f"SECRET {_sql_literal(secret_key)}",
            f"ENDPOINT {_sql_literal(s3_endpoint)}",
            "URL_STYLE 'path'",
            f"USE_SSL {'true' if use_ssl else 'false'}",
        ]
        if s3_region:
            secret_options.append(f"REGION {_sql_literal(s3_region)}")
        self.conn.execute(f"CREATE OR REPLACE SECRET s3_lakehouse ({', '.join(secret_options)});")

    def query_metric(self, query: FinancialQueryInput) -> list[dict[str, Any]]:
        """Execute parameterized query across multiple tickers, metrics, years, and periods.

        Args:
            query: Validated FinancialQueryInput payload (supporting batch lists).

        Returns:
            List of raw dictionaries matching the target metrics and filters.
        """
        # Partition requested metrics into statements and ratios
        stmt_metrics = [m for m in query.metrics if m in STATEMENT_METRICS]
        ratio_metrics = [m for m in query.metrics if m in RATIO_METRICS]

        queries: list[str] = []
        params: list[Any] = []

        ticker_placeholders = ", ".join(["?"] * len(query.tickers))
        year_filter = ""
        year_params: list[Any] = []
        if query.fiscal_years:
            year_placeholders = ", ".join(["?"] * len(query.fiscal_years))
            year_filter = f" AND fiscal_year IN ({year_placeholders})"
            year_params = list(query.fiscal_years)

        period_filter = ""
        period_params: list[Any] = []
        if query.fiscal_periods:
            period_placeholders = ", ".join(["?"] * len(query.fiscal_periods))
            period_filter = f" AND fiscal_period IN ({period_placeholders})"
            period_params = list(query.fiscal_periods)

        type_filter = ""
        type_params: list[Any] = []
        if query.period_type is not None:
            type_filter = " AND period_type = ?"
            type_params = [query.period_type]

        # 1. Build queries for Statement metrics
        stmt_currency_expr = (
            "currency" if self._has_column(self.statements_path, "currency") else "CAST(NULL AS VARCHAR) AS currency"
        )
        for m in stmt_metrics:
            subquery = f"""
                SELECT
                    statement_key AS record_key,
                    ticker,
                    cik,
                    {stmt_currency_expr},
                    fiscal_year,
                    fiscal_period,
                    period_type,
                    start_date,
                    end_date,
                    ? AS metric_name,
                    CAST({m} AS DECIMAL(28, 2)) AS metric_value,
                    is_derived,
                    derivation_method,
                    source_accession_number,
                    source_filed_date
                FROM read_parquet(?)
                WHERE ticker IN ({ticker_placeholders})
                {year_filter}
                {period_filter}
                {type_filter}
            """
            queries.append(subquery)
            params.extend([m, self.statements_path] + query.tickers + year_params + period_params + type_params)

        # 2. Build queries for Ratio metrics
        ratio_currency_expr = (
            "currency" if self._has_column(self.ratios_path, "currency") else "CAST(NULL AS VARCHAR) AS currency"
        )
        ratio_acc_expr = (
            "source_accession_number"
            if self._has_column(self.ratios_path, "source_accession_number")
            else "CAST(NULL AS VARCHAR) AS source_accession_number"
        )
        ratio_filed_expr = (
            "source_filed_date"
            if self._has_column(self.ratios_path, "source_filed_date")
            else "CAST(NULL AS DATE) AS source_filed_date"
        )
        for m in ratio_metrics:
            subquery = f"""
                SELECT
                    ratio_key AS record_key,
                    ticker,
                    cik,
                    {ratio_currency_expr},
                    fiscal_year,
                    fiscal_period,
                    period_type,
                    start_date,
                    end_date,
                    ? AS metric_name,
                    CAST({m} AS DECIMAL(28, 4)) AS metric_value,
                    false AS is_derived,
                    NULL AS derivation_method,
                    {ratio_acc_expr},
                    {ratio_filed_expr}
                FROM read_parquet(?)
                WHERE ticker IN ({ticker_placeholders})
                {year_filter}
                {period_filter}
                {type_filter}
            """
            queries.append(subquery)
            params.extend([m, self.ratios_path] + query.tickers + year_params + period_params + type_params)

        if not queries:
            return []

        full_sql = (
            " UNION ALL ".join(queries)
            + """
            ORDER BY
                ticker ASC,
                fiscal_year ASC,
                CASE fiscal_period
                    WHEN 'Q1' THEN 1
                    WHEN 'Q2' THEN 2
                    WHEN 'Q3' THEN 3
                    WHEN 'Q4' THEN 4
                    WHEN 'FY' THEN 5
                    ELSE 6
                END ASC,
                metric_name ASC
        """
        )

        try:
            columns, rows = self._run_cursor_query(full_sql, params)
        except (duckdb.Error, OSError) as exc:
            raise DatabaseExecutionError(f"Database query execution failed: {exc}") from exc

        return [dict(zip(columns, row, strict=False)) for row in rows]
