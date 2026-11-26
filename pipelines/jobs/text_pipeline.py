"""End-to-End Text Pipeline for SEC EDGAR Filings.

Orchestrates parsing raw HTML filings from Bronze S3, storing structured Silver sections
and tables, performing structure-aware chunking, and writing strongly-typed Parquet
chunks to Gold S3. Generates multi-company verification and audit reports.
"""

import argparse
import gc
import io
import json
import logging
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from dotenv import load_dotenv

from backend.core.storage.s3_storage import S3Storage
from pipelines.chunking.chunking_engine import StructureAwareChunkingEngine, TextChunk
from pipelines.processing.section_extractor import (
    ExtractionReport,
    HTMLSectionExtractor,
    SectionStatus,
)

load_dotenv()

logger = logging.getLogger(__name__)

# Strongly typed PyArrow schema for Gold chunks analytical mart
CHUNK_PARQUET_SCHEMA = pa.schema(
    [
        ("chunk_id", pa.string()),
        ("cik", pa.string()),
        ("ticker", pa.string()),
        ("accession_number", pa.string()),
        ("form", pa.string()),
        ("fiscal_year", pa.int64()),
        ("filed_date", pa.string()),
        ("period_end", pa.string()),
        ("part", pa.string()),
        ("item", pa.string()),
        ("subsection", pa.string()),
        ("content_type", pa.string()),
        ("chunk_index", pa.int64()),
        ("token_count", pa.int64()),
        ("text", pa.string()),
        ("source_url", pa.string()),
        ("source_content_hash", pa.string()),
    ]
)


class FilingsTextPipeline:
    """Orchestrates end-to-end HTML parsing, section extraction, and Parquet chunking."""

    def __init__(
        self,
        s3_storage: S3Storage | None = None,
        section_extractor: HTMLSectionExtractor | None = None,
        chunking_engine: StructureAwareChunkingEngine | None = None,
        bronze_bucket: str = "sec-bronze",
        silver_bucket: str = "sec-silver",
        gold_bucket: str = "sec-gold",
        dead_letter_dir: str | Path = "data/dead_letter",
    ) -> None:
        """Initialize the text pipeline with storage adapter and engines."""
        self.s3 = s3_storage or S3Storage()
        self.extractor = section_extractor or HTMLSectionExtractor()
        self.chunker = chunking_engine or StructureAwareChunkingEngine()
        self.bronze_bucket = bronze_bucket
        self.silver_bucket = silver_bucket
        self.gold_bucket = gold_bucket
        self.dead_letter_dir = Path(dead_letter_dir)

    def process_filing(
        self,
        html_s3_key: str,
        force: bool = False,
    ) -> tuple[list[TextChunk], dict[str, Any]]:
        """Process a single HTML filing from Bronze to Silver and Gold chunks.

        Args:
            html_s3_key: S3 object key of the raw HTML filing in Bronze bucket.
            force: If True, overwrite existing Silver/Gold records.

        Returns:
            Tuple of (list of generated TextChunks, filing extraction summary dict).
        """
        logger.info("Processing filing: s3://%s/%s", self.bronze_bucket, html_s3_key)

        # 1. Download companion metadata JSON if present
        meta_key = html_s3_key.rsplit(".", 1)[0] + "_metadata.json"
        metadata: dict[str, Any] = {}
        if self.s3.object_exists(self.bronze_bucket, meta_key):
            try:
                meta_bytes = self.s3.download_bytes(self.bronze_bucket, meta_key)
                metadata = json.loads(meta_bytes.decode("utf-8"))
            except Exception as exc:
                logger.warning("Failed to parse metadata from %s: %s", meta_key, exc)

        # Fallback metadata from filename structure: {form}_{period}_{accNum}.htm or {form}_A_{period}_{accNum}.htm
        filename = html_s3_key.rsplit("/", 1)[-1]
        parts = filename.replace(".htm", "").replace(".html", "").split("_")
        acc_fallback = parts[-1] if len(parts) >= 3 else ""
        period_fallback = parts[-2] if len(parts) >= 3 else ""
        form_fallback = "_".join(parts[:-2]).replace("_A", "/A") if len(parts) >= 3 else "10-K"

        ticker = metadata.get("ticker") or html_s3_key.split("/")[-2]
        cik = str(metadata.get("cik") or "").zfill(10)
        form = metadata.get("form") or form_fallback
        filed_date = metadata.get("filing_date") or period_fallback
        period_end = metadata.get("report_date") or period_fallback
        accession_number = metadata.get("accession_number") or acc_fallback
        source_url = metadata.get("source_url") or ""

        # Derive fiscal year integer
        try:
            fiscal_year = int(period_end[:4]) if period_end else datetime.now().year
        except ValueError:
            fiscal_year = datetime.now().year

        # 2. Download raw HTML bytes
        html_bytes = self.s3.download_bytes(self.bronze_bucket, html_s3_key)
        html_content = html_bytes.decode("utf-8", errors="replace")

        # 4. Save Silver Layer artifacts
        silver_prefix = f"sections/{ticker}/{accession_number}"
        report_s3_key = f"{silver_prefix}/_extraction_report.json"

        # Check if Silver extraction report already exists and can be reused
        if not force and self.s3.object_exists(self.silver_bucket, report_s3_key):
            try:
                cached_report_bytes = self.s3.download_bytes(self.silver_bucket, report_s3_key)
                report_dict = json.loads(cached_report_bytes.decode("utf-8"))
                report = ExtractionReport(**report_dict)
                logger.info("Reusing existing Silver extraction report for %s [%s]", ticker, accession_number)
            except Exception as exc:
                logger.warning("Failed to load cached Silver report for %s: %s. Re-extracting.", accession_number, exc)
                report = self.extractor.extract(
                    html_content=html_content,
                    form=form,
                    ticker=ticker,
                    accession_number=accession_number,
                )
        else:
            report = self.extractor.extract(
                html_content=html_content,
                form=form,
                ticker=ticker,
                accession_number=accession_number,
            )

        report_bytes = json.dumps(report.model_dump(), indent=2).encode("utf-8")
        self.s3.upload_bytes(
            self.silver_bucket,
            report_s3_key,
            report_bytes,
            content_type="application/json",
        )

        extracted_sections = [
            s for s in report.sections if s.status in (SectionStatus.EXTRACTED, SectionStatus.NON_STANDARD)
        ]

        for sec in extracted_sections:
            clean_item_tag = re.sub(r"[^A-Za-z0-9_]+", "_", sec.item).strip("_")
            # Save raw narrative text
            txt_key = f"{silver_prefix}/{clean_item_tag}.txt"
            self.s3.upload_bytes(
                self.silver_bucket,
                txt_key,
                sec.raw_text.encode("utf-8"),
                content_type="text/plain",
            )
            # Save structured nodes JSON (headings, lists, tables with Markdown)
            nodes_key = f"{silver_prefix}/{clean_item_tag}_nodes.json"
            nodes_data = [node.model_dump() for node in sec.nodes]
            self.s3.upload_bytes(
                self.silver_bucket,
                nodes_key,
                json.dumps(nodes_data, indent=2).encode("utf-8"),
                content_type="application/json",
            )

        # 5. Generate Gold Chunks using StructureAwareChunkingEngine
        chunks: list[TextChunk] = []
        chunk_offset = 0
        for sec in extracted_sections:
            sec_chunks = self.chunker.chunk_section(
                section=sec,
                cik=cik,
                ticker=ticker,
                accession_number=accession_number,
                form=form,
                fiscal_year=fiscal_year,
                filed_date=filed_date,
                period_end=period_end,
                source_url=source_url,
                start_chunk_index=chunk_offset,
            )
            chunks.extend(sec_chunks)
            chunk_offset += len(sec_chunks)

        filing_summary = {
            "ticker": ticker,
            "accession_number": accession_number,
            "form": form,
            "fiscal_year": fiscal_year,
            "toc_found": report.toc_found,
            "sections_extracted": len(extracted_sections),
            "tables_extracted": report.diagnostics.get("total_tables_extracted", 0),
            "total_chunks": len(chunks),
            "unidentified_sections": report.unidentified_sections,
        }

        return chunks, filing_summary

    @staticmethod
    def hive_partition_key(ticker: str) -> str:
        """Return canonical Hive-partitioned Gold Parquet key for a single ticker."""
        return f"chunks/ticker={ticker.strip().upper()}/chunks.parquet"

    def _write_dead_letter_manifest(self, entries: list[dict[str, Any]]) -> str | None:
        """Persist failed-filing dead-letter manifest to local disk and MinIO Silver."""
        if not entries:
            return None

        manifest = {
            "generated_at": datetime.now(UTC).isoformat(),
            "failed_count": len(entries),
            "entries": entries,
        }
        payload_bytes = json.dumps(manifest, indent=2).encode("utf-8")
        s3_key = "dead_letter/text_pipeline_dead_letter.json"

        try:
            self.dead_letter_dir.mkdir(parents=True, exist_ok=True)
            local_file = self.dead_letter_dir / "text_pipeline_dead_letter.json"
            local_file.write_bytes(payload_bytes)
        except OSError as exc:
            logger.warning("Failed to write local dead-letter manifest: %s", exc)

        try:
            self.s3.upload_bytes(
                self.silver_bucket,
                s3_key,
                payload_bytes,
                content_type="application/json",
            )
            return f"s3://{self.silver_bucket}/{s3_key}"
        except Exception as exc:
            logger.warning("Failed to upload dead-letter manifest to MinIO: %s", exc)
            return str(self.dead_letter_dir / "text_pipeline_dead_letter.json")

    @staticmethod
    def _new_company_stat() -> dict[str, Any]:
        return {
            "filings_count": 0,
            "10K_count": 0,
            "10Q_count": 0,
            "20F_count": 0,
            "total_sections": 0,
            "total_tables": 0,
            "total_chunks": 0,
            "abnormal_sections": set(),
        }

    def process_all(
        self,
        tickers: list[str] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Process HTML filings ticker-by-ticker into Hive-partitioned Gold Parquet.

        Streams each filing batch into a per-ticker ParquetWriter to keep memory bounded
        at O(1 filing) and isolates malformed filings in a dead-letter manifest.
        """
        all_keys = self.s3.list_objects(self.bronze_bucket, prefix="raw/filings/")
        html_keys = [k for k in all_keys if k.endswith(".htm") or k.endswith(".html")]

        if tickers:
            ticker_set = {t.upper() for t in tickers}
            html_keys = [k for k in html_keys if k.split("/")[-2].upper() in ticker_set]

        html_keys.sort()
        logger.info("Discovered %d HTML filing objects across target tickers.", len(html_keys))

        keys_by_ticker: dict[str, list[str]] = {}
        for key in html_keys:
            tk = key.split("/")[-2].upper()
            keys_by_ticker.setdefault(tk, []).append(key)

        skipped_count = 0
        newly_processed_count = 0
        new_chunks_created = 0
        total_chunks_in_gold = 0
        company_stats: dict[str, dict[str, Any]] = {}
        sample_records: list[dict[str, Any]] = []
        dead_letter_entries: list[dict[str, Any]] = []
        written_partitions: list[str] = []

        for ticker in sorted(keys_by_ticker.keys()):
            ticker_keys = keys_by_ticker[ticker]
            partition_key = self.hive_partition_key(ticker)
            existing_table: pa.Table | None = None
            existing_accessions: set[str] = set()

            if not force and self.s3.object_exists(self.gold_bucket, partition_key):
                try:
                    part_bytes = self.s3.download_bytes(self.gold_bucket, partition_key)
                    existing_table = pq.read_table(io.BytesIO(part_bytes))
                    existing_accessions = set(existing_table["accession_number"].to_pylist())
                except Exception as exc:
                    logger.warning("Failed to read existing partition %s (%s). Rebuilding ticker.", partition_key, exc)
                    existing_table = None
                    existing_accessions = set()

            keys_to_process: list[str] = []
            for key in ticker_keys:
                filename = key.rsplit("/", 1)[-1]
                parts = filename.replace(".htm", "").replace(".html", "").split("_")
                acc_num = parts[-1] if len(parts) >= 3 else ""
                if not force and acc_num and acc_num in existing_accessions:
                    skipped_count += 1
                else:
                    keys_to_process.append(key)

            stat = company_stats.setdefault(ticker, self._new_company_stat())

            if not keys_to_process:
                if existing_table is not None:
                    total_chunks_in_gold += existing_table.num_rows
                    stat["total_chunks"] += existing_table.num_rows
                    seen_accs: set[str] = set()
                    for row in existing_table.select(["accession_number", "form"]).to_pylist():
                        acc = row.get("accession_number")
                        if acc and acc not in seen_accs:
                            seen_accs.add(acc)
                            stat["filings_count"] += 1
                            form_u = str(row.get("form") or "").upper()
                            if "10-K" in form_u:
                                stat["10K_count"] += 1
                            elif "20-F" in form_u:
                                stat["20F_count"] += 1
                            else:
                                stat["10Q_count"] += 1
                    if len(sample_records) < 3 and existing_table.num_rows > 0:
                        needed = 3 - len(sample_records)
                        sample_records.extend(existing_table.slice(0, needed).to_pylist())
                continue

            reprocessed_accs: set[str] = set()
            ticker_new_chunks = 0
            ticker_total_rows = 0

            with tempfile.SpooledTemporaryFile(max_size=16 * 1024 * 1024) as spooled_buf:
                writer = pq.ParquetWriter(spooled_buf, CHUNK_PARQUET_SCHEMA, compression="snappy")
                try:
                    for key in keys_to_process:
                        try:
                            chunks, summary = self.process_filing(key, force=force)
                            newly_processed_count += 1
                            reprocessed_accs.add(summary["accession_number"])

                            stat["filings_count"] += 1
                            form_upper = summary["form"].upper()
                            if "10-K" in form_upper:
                                stat["10K_count"] += 1
                            elif "20-F" in form_upper:
                                stat["20F_count"] += 1
                            else:
                                stat["10Q_count"] += 1
                            stat["total_sections"] += summary["sections_extracted"]
                            stat["total_tables"] += summary["tables_extracted"]
                            stat["total_chunks"] += summary["total_chunks"]
                            for un_sec in summary["unidentified_sections"]:
                                stat["abnormal_sections"].add(un_sec)

                            if chunks:
                                records = [chunk.model_dump() for chunk in chunks]
                                if len(sample_records) < 3:
                                    sample_records.extend(records[: 3 - len(sample_records)])
                                batch_table = pa.Table.from_pylist(records, schema=CHUNK_PARQUET_SCHEMA)
                                writer.write_table(batch_table)
                                ticker_new_chunks += len(records)
                                ticker_total_rows += len(records)
                                del records, batch_table
                            del chunks
                        except Exception as exc:
                            logger.error("Error processing filing %s: %s", key, exc, exc_info=True)
                            dead_letter_entries.append(
                                {
                                    "s3_key": key,
                                    "ticker": ticker,
                                    "error_type": type(exc).__name__,
                                    "error": str(exc),
                                    "timestamp": datetime.now(UTC).isoformat(),
                                }
                            )

                    if existing_table is not None and existing_table.num_rows > 0:
                        retained_table = existing_table
                        if reprocessed_accs:
                            mask = pc.invert(
                                pc.is_in(
                                    existing_table["accession_number"],
                                    value_set=pa.array(list(reprocessed_accs), type=pa.string()),
                                )
                            )
                            retained_table = existing_table.filter(mask)
                        if retained_table.num_rows > 0:
                            writer.write_table(retained_table.cast(CHUNK_PARQUET_SCHEMA))
                            ticker_total_rows += retained_table.num_rows
                            stat["total_chunks"] += retained_table.num_rows
                finally:
                    writer.close()

                if ticker_total_rows > 0:
                    spooled_buf.seek(0)
                    parquet_bytes = spooled_buf.read()
                    self.s3.upload_bytes(
                        self.gold_bucket,
                        partition_key,
                        parquet_bytes,
                        content_type="application/vnd.apache.parquet",
                        metadata={"ticker": ticker, "total_chunks": str(ticker_total_rows)},
                    )
                    written_partitions.append(partition_key)
                    logger.info(
                        "Checkpointed Hive partition s3://%s/%s (%d chunks, %d new).",
                        self.gold_bucket,
                        partition_key,
                        ticker_total_rows,
                        ticker_new_chunks,
                    )

            new_chunks_created += ticker_new_chunks
            total_chunks_in_gold += ticker_total_rows
            del existing_table
            gc.collect()

        dead_letter_manifest_uri = self._write_dead_letter_manifest(dead_letter_entries)

        for s in company_stats.values():
            s["abnormal_sections"] = list(s["abnormal_sections"])

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "incremental": not force,
            "hive_partitioned": True,
            "partitions_written": written_partitions,
            "filings_skipped_up_to_date": skipped_count,
            "filings_newly_processed": newly_processed_count,
            "new_chunks_created": new_chunks_created,
            "total_chunks_in_gold": total_chunks_in_gold,
            "dead_letter_count": len(dead_letter_entries),
            "dead_letter_manifest": dead_letter_manifest_uri,
            "dead_letter_entries": dead_letter_entries,
            "company_stats": company_stats,
            "sample_chunks": sample_records,
        }

    @staticmethod
    def format_verification_report(summary: dict[str, Any]) -> str:
        """Format aggregated summary dictionary into a clean Markdown audit report."""
        lines = [
            "# TEXT PIPELINE & CHUNKING VERIFICATION REPORT",
            f"**Timestamp:** {summary.get('timestamp')}",
            f"**Execution Mode:** {'Incremental' if summary.get('incremental') else 'Full Build'}",
            f"**Filings Up-to-Date (Skipped):** {summary.get('filings_skipped_up_to_date', 0)}",
            f"**Filings Newly Processed:** {summary.get('filings_newly_processed', 0)}",
            f"**New Chunks Created:** {summary.get('new_chunks_created', 0)}",
            f"**Total Chunks in Gold Lakehouse:** {summary.get('total_chunks_in_gold', 0)}",
            "",
            "## 1. Company-Level Statistics",
            "",
            "| Ticker | Filings (10-K / 10-Q) | Sections Detected | Tables Extracted | Chunks Created | Unidentified / Missing Items |",
            "| :--- | :---: | :---: | :---: | :---: | :--- |",
        ]

        comp_stats = summary.get("company_stats", {})
        for ticker, st in sorted(comp_stats.items()):
            filing_str = f"{st['filings_count']} ({st['10K_count']}K / {st['10Q_count']}Q)"
            missing_str = ", ".join(st.get("abnormal_sections", [])) or "None (100% Extracted)"
            lines.append(
                f"| **{ticker}** | {filing_str} | {st['total_sections']} | "
                f"{st['total_tables']} | {st['total_chunks']} | `{missing_str}` |"
            )

        lines.extend(
            [
                "",
                "## 2. Sample Chunk Verification (First 3 Chunks)",
                "",
            ]
        )

        for i, sample in enumerate(summary.get("sample_chunks", [])):
            lines.extend(
                [
                    f"### Sample {i + 1}: Chunk ID `{sample.get('chunk_id')[:12]}...`",
                    f"- **Ticker / Form / FY**: {sample.get('ticker')} | {sample.get('form')} | FY{sample.get('fiscal_year')}",
                    f"- **Section / Subsection**: {sample.get('item')} | {sample.get('subsection') or 'Root'}",
                    f"- **Type / Tokens**: `{sample.get('content_type')}` | {sample.get('token_count')} tokens",
                    f"- **Source URL**: [{sample.get('source_url')}]({sample.get('source_url')})",
                    "```text",
                    sample.get("text", "")[:280] + ("..." if len(sample.get("text", "")) > 280 else ""),
                    "```",
                    "",
                ]
            )

        return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="SEC Filings HTML Text & Chunking Pipeline")
    parser.add_argument(
        "--ticker",
        type=str,
        help="Optional ticker filter (e.g., AAPL, MSFT, GOOGL)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing of filings",
    )
    args = parser.parse_args()

    pipeline = FilingsTextPipeline()
    from backend.core.financial.company_registry import get_supported_tickers

    target_tickers = [args.ticker.upper()] if args.ticker else sorted(list(get_supported_tickers()))

    result = pipeline.process_all(tickers=target_tickers, force=args.force)
    report_md = pipeline.format_verification_report(result)
    print("\n" + report_md)
