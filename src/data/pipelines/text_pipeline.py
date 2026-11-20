"""End-to-End Text Pipeline for SEC EDGAR Filings.

Orchestrates parsing raw HTML filings from Bronze S3, storing structured Silver sections
and tables, performing structure-aware chunking, and writing strongly-typed Parquet
chunks to Gold S3. Generates multi-company verification and audit reports.
"""

import argparse
import io
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.chunking.chunking_engine import StructureAwareChunkingEngine, TextChunk
from src.data.processing.section_extractor import (
    ExtractedSection,
    ExtractionReport,
    HTMLSectionExtractor,
    SectionStatus,
)
from src.data.storage.s3_storage import S3Storage

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# Strongly typed PyArrow schema for Gold chunks analytical mart
CHUNK_PARQUET_SCHEMA = pa.schema([
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
])


class FilingsTextPipeline:
    """Orchestrates end-to-end HTML parsing, section extraction, and Parquet chunking."""

    def __init__(
        self,
        s3_storage: Optional[S3Storage] = None,
        section_extractor: Optional[HTMLSectionExtractor] = None,
        chunking_engine: Optional[StructureAwareChunkingEngine] = None,
        bronze_bucket: str = "sec-bronze",
        silver_bucket: str = "sec-silver",
        gold_bucket: str = "sec-gold",
    ) -> None:
        """Initialize the text pipeline with storage adapter and engines."""
        self.s3 = s3_storage or S3Storage()
        self.extractor = section_extractor or HTMLSectionExtractor()
        self.chunker = chunking_engine or StructureAwareChunkingEngine()
        self.bronze_bucket = bronze_bucket
        self.silver_bucket = silver_bucket
        self.gold_bucket = gold_bucket

    def process_filing(
        self,
        html_s3_key: str,
        force: bool = False,
    ) -> Tuple[List[TextChunk], Dict[str, Any]]:
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
        metadata: Dict[str, Any] = {}
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
            s for s in report.sections
            if s.status in (SectionStatus.EXTRACTED, SectionStatus.NON_STANDARD)
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
        chunks: List[TextChunk] = []
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

    def process_all(
        self,
        tickers: Optional[List[str]] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Process all HTML filings across companies, aggregate chunks, and upload Parquet.

        Args:
            tickers: Optional list of tickers to filter (e.g., ['AAPL', 'MSFT', 'GOOGL']).
            force: If True, re-process filings.

        Returns:
            Dictionary containing aggregated statistics and verification summaries.
        """
        all_keys = self.s3.list_objects(self.bronze_bucket, prefix="raw/filings/")
        html_keys = [k for k in all_keys if k.endswith(".htm")]

        if tickers:
            ticker_set = {t.upper() for t in tickers}
            html_keys = [k for k in html_keys if k.split("/")[-2].upper() in ticker_set]

        html_keys.sort()
        logger.info("Discovered %d HTML filing objects across target tickers.", len(html_keys))

        gold_key = "chunks/text_chunks.parquet"
        existing_records: List[Dict[str, Any]] = []
        existing_accessions: set[str] = set()

        # Step 1: Incremental check - inspect existing Gold Parquet mart
        if not force and self.s3.object_exists(self.gold_bucket, gold_key):
            try:
                gold_bytes = self.s3.download_bytes(self.gold_bucket, gold_key)
                existing_table = pq.read_table(io.BytesIO(gold_bytes))
                existing_records = existing_table.to_pylist()
                existing_accessions = set(existing_table["accession_number"].to_pylist())
                logger.info(
                    "Loaded existing Gold analytical mart with %d chunks across %d filings.",
                    len(existing_records),
                    len(existing_accessions),
                )
            except Exception as exc:
                logger.warning("Failed to read existing Gold Parquet (%s). Will re-build.", exc)
                existing_records = []
                existing_accessions = set()

        # Step 2: Separate up-to-date filings from new/changed filings
        keys_to_process: List[str] = []
        skipped_count = 0

        for key in html_keys:
            filename = key.rsplit("/", 1)[-1]
            parts = filename.replace(".htm", "").replace(".html", "").split("_")
            acc_num = parts[-1] if len(parts) >= 3 else ""

            if not force and acc_num and acc_num in existing_accessions:
                skipped_count += 1
            else:
                keys_to_process.append(key)

        logger.info(
            "Incremental check: %d filings already up-to-date (skipped), %d filings to process.",
            skipped_count,
            len(keys_to_process),
        )

        all_new_chunks: List[TextChunk] = []
        filing_summaries: List[Dict[str, Any]] = []
        company_stats: Dict[str, Dict[str, Any]] = {}

        # Step 3: Process only the new or modified filings
        for key in keys_to_process:
            try:
                chunks, summary = self.process_filing(key, force=force)
                all_new_chunks.extend(chunks)
                filing_summaries.append(summary)

                ticker = summary["ticker"]
                if ticker not in company_stats:
                    company_stats[ticker] = {
                        "filings_count": 0,
                        "10K_count": 0,
                        "10Q_count": 0,
                        "total_sections": 0,
                        "total_tables": 0,
                        "total_chunks": 0,
                        "abnormal_sections": set(),
                    }

                stat = company_stats[ticker]
                stat["filings_count"] += 1
                if summary["form"] == "10-K":
                    stat["10K_count"] += 1
                else:
                    stat["10Q_count"] += 1
                stat["total_sections"] += summary["sections_extracted"]
                stat["total_tables"] += summary["tables_extracted"]
                stat["total_chunks"] += summary["total_chunks"]
                for un_sec in summary["unidentified_sections"]:
                    stat["abnormal_sections"].add(un_sec)

            except Exception as exc:
                logger.error("Error processing filing %s: %s", key, exc, exc_info=True)

        # Step 4: Incremental Merge into Gold Parquet
        if keys_to_process and (all_new_chunks or existing_records):
            reprocessed_accs = {s["accession_number"] for s in filing_summaries}
            retained_records = [
                r for r in existing_records
                if r.get("accession_number") not in reprocessed_accs
            ]
            new_chunk_records = [chunk.model_dump() for chunk in all_new_chunks]
            final_records = retained_records + new_chunk_records

            table = pa.Table.from_pylist(final_records, schema=CHUNK_PARQUET_SCHEMA)
            buf = pa.BufferOutputStream()
            pq.write_table(table, buf, compression="snappy")
            parquet_bytes = buf.getvalue().to_pybytes()

            self.s3.upload_bytes(
                self.gold_bucket,
                gold_key,
                parquet_bytes,
                content_type="application/vnd.apache.parquet",
                metadata={"total_chunks": str(len(final_records))},
            )
            logger.info(
                "Incremental update complete: merged %d new chunks with %d existing chunks (Total: %d chunks) to s3://%s/%s",
                len(new_chunk_records),
                len(retained_records),
                len(final_records),
                self.gold_bucket,
                gold_key,
            )
            total_chunks_in_gold = len(final_records)
            sample_records = final_records[:3]
        else:
            total_chunks_in_gold = len(existing_records)
            sample_records = existing_records[:3]

        # Reconstruct company stats if incremental no-op
        if not keys_to_process and existing_records:
            for r in existing_records:
                tk = r.get("ticker", "UNKNOWN")
                if tk not in company_stats:
                    company_stats[tk] = {
                        "filings_count": 0,
                        "10K_count": 0,
                        "10Q_count": 0,
                        "total_sections": 0,
                        "total_tables": 0,
                        "total_chunks": 0,
                        "abnormal_sections": set(),
                    }
                company_stats[tk]["total_chunks"] += 1
            # Fill filing counts from existing unique accessions
            for r in existing_records:
                tk = r.get("ticker", "UNKNOWN")
            # Distinct filings per ticker
            seen_accs: set[str] = set()
            for r in existing_records:
                acc = r.get("accession_number")
                tk = r.get("ticker", "UNKNOWN")
                if acc and acc not in seen_accs:
                    seen_accs.add(acc)
                    company_stats[tk]["filings_count"] += 1
                    if r.get("form") == "10-K":
                        company_stats[tk]["10K_count"] += 1
                    else:
                        company_stats[tk]["10Q_count"] += 1

        # Format company stats for JSON report
        for t, s in company_stats.items():
            s["abnormal_sections"] = list(s["abnormal_sections"])

        overall_summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "incremental": True,
            "filings_skipped_up_to_date": skipped_count,
            "filings_newly_processed": len(keys_to_process),
            "new_chunks_created": len(all_new_chunks),
            "total_chunks_in_gold": total_chunks_in_gold,
            "company_stats": company_stats,
            "sample_chunks": sample_records,
        }

        return overall_summary

    @staticmethod
    def format_verification_report(summary: Dict[str, Any]) -> str:
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

        lines.extend([
            "",
            "## 2. Sample Chunk Verification (First 3 Chunks)",
            "",
        ])

        for i, sample in enumerate(summary.get("sample_chunks", [])):
            lines.extend([
                f"### Sample {i+1}: Chunk ID `{sample.get('chunk_id')[:12]}...`",
                f"- **Ticker / Form / FY**: {sample.get('ticker')} | {sample.get('form')} | FY{sample.get('fiscal_year')}",
                f"- **Section / Subsection**: {sample.get('item')} | {sample.get('subsection') or 'Root'}",
                f"- **Type / Tokens**: `{sample.get('content_type')}` | {sample.get('token_count')} tokens",
                f"- **Source URL**: [{sample.get('source_url')}]({sample.get('source_url')})",
                "```text",
                sample.get("text", "")[:280] + ("..." if len(sample.get("text", "")) > 280 else ""),
                "```",
                "",
            ])

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
    from src.core.financial.company_registry import get_supported_tickers
    target_tickers = [args.ticker.upper()] if args.ticker else sorted(list(get_supported_tickers()))

    result = pipeline.process_all(tickers=target_tickers, force=args.force)
    report_md = pipeline.format_verification_report(result)
    print("\n" + report_md)
