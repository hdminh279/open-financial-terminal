"""Unified Command-Line Interface for Corporate Filing Intelligence.

Provides entrypoints for orchestrating SEC data ingestion pipelines,
checking data lakehouse integrity, and managing Bronze/Silver/Gold layers.
"""

import argparse
import logging
import os
import sys
from typing import List, Optional

from dotenv import load_dotenv

from src.data.ingestion.fetch_facts import FactsIngestionPipeline
from src.data.ingestion.fetch_filings import FilingsIngestionPipeline
from src.data.ingestion.rate_limiter import RateLimiter
from src.data.storage.s3_storage import S3Storage
from src.data.ingestion.sec_client import SECClient

logger = logging.getLogger("corporate_filing_cli")


def build_parser() -> argparse.ArgumentParser:
    """Build and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="corporate-filing",
        description="Corporate Filing Intelligence: Lakehouse Ingestion & Verification CLI",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging level.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Ingest Subcommand
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest SEC submissions, XBRL facts, and 10-K / 10-Q HTML filings into MinIO Bronze.",
    )
    group = ingest_parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all",
        action="store_true",
        help="Run end-to-end ingestion (both company facts and filings).",
    )
    group.add_argument(
        "--facts",
        action="store_true",
        help="Ingest only submissions and XBRL company facts.",
    )
    group.add_argument(
        "--filings",
        action="store_true",
        help="Ingest only 10-K and 10-Q HTML filings.",
    )
    group.add_argument(
        "--vn",
        "--vn30",
        dest="vn",
        action="store_true",
        help="Ingest VN30 financial statements (Income Statement, Balance Sheet, Cash Flow) into MinIO Bronze.",
    )

    ingest_parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Optional single company ticker for targeted ingestion (e.g. FPT, AAPL).",
    )

    ingest_parser.add_argument(
        "--config",
        type=str,
        default="configs/companies.yaml",
        help="Path to companies configuration YAML (default: configs/companies.yaml).",
    )
    ingest_parser.add_argument(
        "--force",
        action="store_true",
        help="Force download and overwrite existing objects in S3.",
    )
    ingest_parser.add_argument(
        "--limit",
        type=int,
        default=4,
        help="Maximum number of filings to ingest per form type (default: 4, set 0 or use --no-limit for all).",
    )
    ingest_parser.add_argument(
        "--forms",
        nargs="+",
        default=["10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "20-F/A"],
        help="Target filing forms (default: 10-K 10-Q 10-K/A 10-Q/A 20-F 20-F/A).",
    )
    ingest_parser.add_argument(
        "--min-year",
        type=int,
        default=2009,
        help="Earliest filing/report year to ingest (default: 2009).",
    )
    ingest_parser.add_argument(
        "--no-limit",
        action="store_true",
        help="Ingest all available historical filings without limit.",
    )

    # 2. Verify Subcommand
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify SHA-256 integrity of objects in MinIO Lakehouse storage.",
    )
    verify_parser.add_argument(
        "--bucket",
        type=str,
        default="sec-bronze",
        help="Bucket name to audit (default: sec-bronze).",
    )
    verify_parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional prefix filter for object keys.",
    )

    # 3. Index Subcommand
    index_parser = subparsers.add_parser(
        "index",
        help="Synchronize text chunks from Gold Parquet into Qdrant Vector DB with dense & BM25 vectors.",
    )
    index_parser.add_argument(
        "--sync",
        action="store_true",
        help="Run incremental sync (upsert new/updated, purge stale points).",
    )
    index_parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate Qdrant collection before indexing.",
    )
    index_parser.add_argument(
        "--verify",
        action="store_true",
        help="Run count and hash integrity checks between Parquet and Qdrant.",
    )
    index_parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Create a Qdrant snapshot after synchronization.",
    )
    index_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding generation and upserting (default: 64).",
    )
    index_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of chunks to process (useful for testing).",
    )
    index_parser.add_argument(
        "--parquet-path",
        type=str,
        default=None,
        help="Optional local path override for text_chunks.parquet.",
    )

    # 4. Coverage Subcommand
    coverage_parser = subparsers.add_parser(
        "coverage",
        help="Run ExtractionCoverageReporter audit across all Bronze HTML filings.",
    )
    coverage_parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Filter audit by company ticker (e.g. AAPL).",
    )
    coverage_parser.add_argument(
        "--output-file",
        type=str,
        default="docs/reports/EXTRACTION_COVERAGE_REPORT.md",
        help="Path to output markdown report file.",
    )

    # 5. Process Subcommand
    process_parser = subparsers.add_parser(
        "process",
        help="Transform raw Bronze artifacts into Silver observations and Gold text chunks.",
    )
    p_group = process_parser.add_mutually_exclusive_group(required=True)
    p_group.add_argument(
        "--all",
        action="store_true",
        help="Run end-to-end processing (both facts to Silver and text chunking to Gold).",
    )
    p_group.add_argument(
        "--facts",
        action="store_true",
        help="Process raw XBRL company facts into Silver Parquet observations.",
    )
    p_group.add_argument(
        "--text",
        action="store_true",
        help="Process raw HTML filings into Silver sections and Gold Parquet chunks.",
    )
    p_group.add_argument(
        "--vn",
        "--vn30",
        dest="vn",
        action="store_true",
        help="Process raw VN30 statements from Bronze into Silver Parquet and quarantine.",
    )
    process_parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Filter text processing by company ticker (e.g. AAPL).",
    )
    process_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing and overwrite existing objects in S3.",
    )
    process_parser.add_argument(
        "--config",
        type=str,
        default="configs/companies.yaml",
        help="Path to companies configuration YAML (default: configs/companies.yaml).",
    )
    process_parser.add_argument(
        "--taxonomy",
        type=str,
        default="configs/xbrl_taxonomy_map.yaml",
        help="Path to taxonomy whitelist YAML (default: configs/xbrl_taxonomy_map.yaml).",
    )

    return parser


def run_ingest(args: argparse.Namespace) -> int:
    """Execute ingestion pipeline according to provided arguments."""
    load_dotenv()

    # Route VN30 financial statement ingestion
    if getattr(args, "vn", False):
        from src.data.ingestion.vn_ingestion import VNFinancialClient, VNIngestionPipeline

        s3_storage = S3Storage()
        s3_storage.ensure_buckets_exist()

        vn_config = getattr(args, "config", "configs/companies_vn.yaml")
        if vn_config == "configs/companies.yaml":
            vn_config = "configs/companies_vn.yaml"

        target_tickers = [args.ticker.upper()] if getattr(args, "ticker", None) else None

        logger.info("Routing to VNIngestionPipeline (config='%s', target=%s)...", vn_config, target_tickers)
        with VNFinancialClient() as vn_client:
            pipeline = VNIngestionPipeline(client=vn_client, storage=s3_storage)
            summary = pipeline.run(config_path=vn_config, target_tickers=target_tickers, force=args.force)
            logger.info("Finished VN30 ingestion: %s", summary)
        return 0

    user_agent = os.getenv("SEC_USER_AGENT")
    if not user_agent:
        logger.error("Missing SEC_USER_AGENT in environment. Please set it in .env file.")
        return 1

    effective_limit = None if (getattr(args, "no_limit", False) or args.limit == 0) else args.limit
    forms = getattr(args, "forms", ["10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "20-F/A"])
    min_year = getattr(args, "min_year", 2009)

    rate_limiter = RateLimiter(requests_per_second=10.0)
    with SECClient(user_agent=user_agent, rate_limiter=rate_limiter) as sec_client:
        s3_storage = S3Storage()
        s3_storage.ensure_buckets_exist()

        do_facts = args.all or args.facts
        do_filings = args.all or args.filings

        if do_facts:
            logger.info("Starting company submissions and facts ingestion from %s...", args.config)
            facts_pipeline = FactsIngestionPipeline(sec_client=sec_client, s3_storage=s3_storage)
            facts_pipeline.run_from_config(config_path=args.config, force=args.force)
            logger.info("Finished facts ingestion successfully.")

        if do_filings:
            limit_desc = f"{effective_limit}/form" if effective_limit else "all available"
            logger.info("Starting filings ingestion (forms=%s, limit=%s, min_year=%s)...", forms, limit_desc, min_year)
            filings_pipeline = FilingsIngestionPipeline(sec_client=sec_client, s3_storage=s3_storage)
            filings_pipeline.run_from_config(
                config_path=args.config,
                forms=forms,
                limit_per_form=effective_limit,
                min_year=min_year,
                force=args.force,
            )
            logger.info("Finished filings ingestion successfully.")

    return 0


def run_index(args: argparse.Namespace) -> int:
    """Execute corpus synchronization and indexing into Qdrant."""
    from src.ai.vectordb.sync_engine import CorpusSyncEngine

    sync_engine = CorpusSyncEngine()

    if args.sync or args.recreate:
        logger.info("Starting corpus synchronization into Qdrant...")
        summary = sync_engine.sync(
            local_path=args.parquet_path,
            recreate=args.recreate,
            limit=args.limit,
            batch_size=args.batch_size,
            create_snapshot_after=args.snapshot,
        )
        logger.info(
            "Sync completed: status=%s, upserted=%d, stale_purged=%d, total_in_db=%d",
            summary["status"],
            summary["upserted_count"],
            summary["stale_purged_count"],
            summary["qdrant_points_count"],
        )

    if args.verify or (not args.sync and not args.recreate):
        logger.info("Running Parquet vs Qdrant integrity audit...")
        report = sync_engine.verify_integrity(local_path=args.parquet_path, sample_size=50)
        logger.info("Integrity check result: status=%s (audited=%d samples)", report["status"], report["sampled_audited"])
        if report["status"] != "PASSED":
            logger.error("Corpus integrity check FAILED: %s", report["discrepancies"])
            return 1

    return 0


def run_verify(args: argparse.Namespace) -> int:
    """Audit and verify SHA-256 integrity of stored objects in S3."""
    s3_storage = S3Storage()
    logger.info("Listing objects in bucket '%s' with prefix '%s'...", args.bucket, args.prefix)
    keys = s3_storage.list_objects(bucket=args.bucket, prefix=args.prefix)

    # Filter out companion metadata files to verify primary payloads
    primary_keys = [k for k in keys if not k.endswith("_metadata.json")]
    if not primary_keys:
        logger.warning("No primary objects found in bucket '%s' matching prefix '%s'.", args.bucket, args.prefix)
        return 0

    logger.info("Auditing SHA-256 integrity for %d objects...", len(primary_keys))
    passed = 0
    failed = 0

    for key in primary_keys:
        is_valid = s3_storage.verify_integrity(bucket=args.bucket, key=key)
        if is_valid:
            passed += 1
        else:
            logger.error("Integrity check FAILED: s3://%s/%s", args.bucket, key)
            failed += 1

    logger.info(
        "Verification complete: %d passed, %d failed out of %d audited.",
        passed,
        failed,
        len(primary_keys),
    )
    return 1 if failed > 0 else 0


def run_coverage(args: argparse.Namespace) -> int:
    """Execute CoverageReporter quality gate audit on all Bronze HTML filings."""
    load_dotenv()
    from src.data.processing.coverage_reporter import CoverageReporter
    from src.data.processing.section_extractor import HTMLSectionExtractor

    s3_storage = S3Storage()
    bronze_bucket = os.getenv("S3_BRONZE_BUCKET", "sec-bronze")

    prefix = "raw/filings/"
    if getattr(args, "ticker", None):
        prefix = f"raw/filings/{args.ticker.upper()}/"

    filing_keys = [
        k for k in s3_storage.list_objects(bronze_bucket, prefix=prefix)
        if (k.endswith(".htm") or k.endswith(".html")) and not k.endswith("_metadata.json")
    ]
    if not filing_keys:
        logger.warning("No HTML filings found in s3://%s/%s", bronze_bucket, prefix)
        return 0

    logger.info("Found %d HTML filings to audit for extraction coverage...", len(filing_keys))
    extractor = HTMLSectionExtractor()
    reporter = CoverageReporter()
    summaries = []

    for idx, key in enumerate(filing_keys, 1):
        try:
            html_bytes = s3_storage.download_bytes(bronze_bucket, key)
            html_content = html_bytes.decode("utf-8", errors="replace")

            parts = key.split("/")
            ticker = parts[2] if len(parts) >= 3 else "UNKNOWN"
            filename = parts[-1]
            # Key format: {form}_{date}_{accNum}.htm -> extract form
            form_part = filename.split("_")[0].replace("_", "/")
            if "10-K_A" in filename:
                form_part = "10-K/A"
            elif "10-Q_A" in filename:
                form_part = "10-Q/A"

            acc_num = filename.rsplit(".", 1)[0].split("_")[-1]

            extraction_report = extractor.extract(
                html_content=html_content,
                form=form_part,
                ticker=ticker,
                accession_number=acc_num,
            )
            summary = reporter.evaluate_filing(extraction_report)
            summaries.append(summary)
            if idx % 10 == 0 or idx == len(filing_keys):
                logger.info("Audited %d/%d filings for coverage...", idx, len(filing_keys))
        except Exception as exc:
            logger.error("Failed auditing %s: %s", key, exc)

    report_md = reporter.generate_markdown_report(summaries)
    print("\n" + report_md + "\n")

    out_file = getattr(args, "output_file", None)
    if out_file:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(report_md)
        logger.info("Saved extraction coverage report to %s", out_file)

    return 0


def run_process(args: argparse.Namespace) -> int:
    """Execute data transformation pipelines (Bronze -> Silver facts & Silver/Gold text chunks)."""
    load_dotenv()
    do_vn = getattr(args, "vn", False)
    do_facts = args.all or args.facts
    do_text = args.all or args.text

    s3_storage = S3Storage()
    s3_storage.ensure_buckets_exist()

    if do_vn:
        from src.data.processing.vn_facts_parser import VNFactsParser
        logger.info("Starting VNFactsParser (Bronze VN statements -> Silver Parquet observations)...")
        parser = VNFactsParser(
            s3_storage=s3_storage,
            companies_vn_path=getattr(args, "config", "configs/companies_vn.yaml"),
        )
        target = [args.ticker.upper()] if getattr(args, "ticker", None) else None
        stats = parser.run(target_tickers=target)
        logger.info("VN Facts processing complete: %s", stats)
        return 0

    if do_facts:
        from src.data.processing.facts_parser import FactsParser
        logger.info("Starting FactsParser (Bronze XBRL facts -> Silver Parquet observations)...")
        parser = FactsParser(
            s3_storage=s3_storage,
            taxonomy_path=getattr(args, "taxonomy", "configs/xbrl_taxonomy_map.yaml"),
            companies_path=getattr(args, "config", "configs/companies.yaml"),
        )
        stats = parser.run()
        logger.info("Facts processing complete: %s", stats)

    if do_text:
        from src.data.pipelines.text_pipeline import FilingsTextPipeline
        from src.core.financial.company_registry import get_supported_tickers
        logger.info("Starting FilingsTextPipeline (Bronze HTML -> Silver sections -> Gold chunks)...")
        pipeline = FilingsTextPipeline(s3_storage=s3_storage)
        target_tickers = [args.ticker.upper()] if getattr(args, "ticker", None) else sorted(list(get_supported_tickers()))
        summary = pipeline.process_all(tickers=target_tickers, force=args.force)
        report_md = pipeline.format_verification_report(summary)
        print("\n" + report_md + "\n")
        logger.info(
            "Text processing complete: newly_processed=%d, chunks_in_gold=%d",
            summary.get("filings_newly_processed", 0),
            summary.get("total_chunks_in_gold", 0),
        )

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.command == "ingest":
        return run_ingest(args)
    elif args.command == "verify":
        return run_verify(args)
    elif args.command == "index":
        return run_index(args)
    elif args.command == "coverage":
        return run_coverage(args)
    elif args.command == "process":
        return run_process(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())


"""
#Check audit SHA-256
docker compose run --rm app uv run python -m src.cli verify

Run ingestion
docker compose run --rm app uv run python -m src.cli ingest --all
"""
