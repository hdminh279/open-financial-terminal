"""Module for resolving SEC Form 10-K, 10-Q, 10-K/A, and 10-Q/A filings from submissions metadata.

Parses raw SEC submissions JSON stored in MinIO Bronze layer or provided as dictionaries,
traverses both recent and paginated historical submission files, filters for target filing forms,
builds authoritative SEC EDGAR archive URLs, and determines deterministic destination keys for MinIO storage.
"""

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set

from src.data.storage.s3_storage import S3Storage

if TYPE_CHECKING:
    from src.data.ingestion.sec_client import SECClient

logger = logging.getLogger(__name__)

SEC_ARCHIVE_BASE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_no_hyphens}/{primary_doc}"
DEFAULT_TARGET_FORMS: Sequence[str] = ("10-K", "10-Q", "10-K/A", "10-Q/A")


@dataclass(frozen=True)
class ResolvedFiling:
    """Represents a resolved SEC filing with metadata and retrieval endpoints."""

    ticker: str
    cik: str
    form: str
    filing_date: str
    report_date: str
    accession_number: str
    primary_document: str
    sec_url: str
    s3_key: str


class FilingResolver:
    """Resolves target filings from SEC submissions payloads."""

    def __init__(
        self,
        s3_storage: Optional[S3Storage] = None,
        bronze_bucket: str = "sec-bronze",
        sec_client: Optional[Any] = None,
    ) -> None:
        """Initialize the resolver with an optional S3 storage adapter and SEC client.

        Args:
            s3_storage: Adapter for S3 operations. Optional for unit testing pure parsing logic.
            bronze_bucket: Name of the Bronze bucket containing raw submissions.
            sec_client: Optional SECClient for fetching paginated historical submission files on-demand.
        """
        self.s3_storage = s3_storage
        self.bronze_bucket = bronze_bucket
        self.sec_client = sec_client

    @staticmethod
    def build_sec_archive_url(cik: str, accession_number: str, primary_document: str) -> str:
        """Construct the official SEC EDGAR archive download URL.

        Args:
            cik: Central Index Key (with or without leading zeros).
            accession_number: SEC accession number (e.g. '0000320193-23-000106').
            primary_document: Name of primary document (e.g. 'aapl-20230930.htm').

        Returns:
            Authoritative HTTPS URL to the raw filing document on SEC EDGAR.
        """
        cik_int = str(int(cik.strip()))
        accession_no_hyphens = accession_number.strip().replace("-", "")
        primary_doc_clean = primary_document.strip()

        return SEC_ARCHIVE_BASE_URL.format(
            cik_int=cik_int,
            accession_no_hyphens=accession_no_hyphens,
            primary_doc=primary_doc_clean,
        )

    @staticmethod
    def build_s3_key(ticker: str, form: str, report_date: str, accession_number: str) -> str:
        """Build deterministic Bronze S3 key for storing raw filing HTML.

        Format: raw/filings/{ticker}/{form}_{report_date}_{accession_number}.htm
        """
        ticker_clean = ticker.strip().upper()
        form_clean = form.strip().upper().replace("/", "_")
        report_clean = report_date.strip()
        accession_clean = accession_number.strip()

        return f"raw/filings/{ticker_clean}/{form_clean}_{report_clean}_{accession_clean}.htm"

    @classmethod
    def _parse_entries_block(
        cls,
        block: Dict[str, Any],
        cik: str,
        ticker: str,
        target_forms_set: Set[str],
        min_year: Optional[int] = None,
    ) -> List[ResolvedFiling]:
        """Extract and validate ResolvedFiling entries from a raw filing array block."""
        form_list: List[str] = block.get("form", [])
        accession_list: List[str] = block.get("accessionNumber", [])
        filing_date_list: List[str] = block.get("filingDate", [])
        report_date_list: List[str] = block.get("reportDate", [])
        primary_doc_list: List[str] = block.get("primaryDocument", [])

        total = min(
            len(form_list),
            len(accession_list),
            len(filing_date_list),
            len(report_date_list),
            len(primary_doc_list),
        )
        items: List[ResolvedFiling] = []

        for i in range(total):
            form_clean = str(form_list[i]).strip().upper()
            if form_clean in target_forms_set:
                report_date = str(report_date_list[i] or "").strip()
                filing_date = str(filing_date_list[i] or "").strip()

                # Temporal filter against min_year
                ref_date = report_date or filing_date
                if min_year is not None and ref_date:
                    try:
                        filing_year = int(ref_date[:4])
                        if filing_year < min_year:
                            continue
                    except (ValueError, TypeError):
                        pass

                accession = str(accession_list[i]).strip()
                primary_doc = str(primary_doc_list[i] or "").strip()
                if not primary_doc:
                    continue

                sec_url = cls.build_sec_archive_url(
                    cik=cik,
                    accession_number=accession,
                    primary_document=primary_doc,
                )
                s3_key = cls.build_s3_key(
                    ticker=ticker,
                    form=form_clean,
                    report_date=report_date,
                    accession_number=accession,
                )
                items.append(
                    ResolvedFiling(
                        ticker=ticker,
                        cik=cik,
                        form=form_clean,
                        filing_date=filing_date,
                        report_date=report_date,
                        accession_number=accession,
                        primary_document=primary_doc,
                        sec_url=sec_url,
                        s3_key=s3_key,
                    )
                )

        return items

    def resolve_from_submissions_dict(
        self,
        submissions_data: Dict[str, Any],
        target_forms: Sequence[str] = DEFAULT_TARGET_FORMS,
        limit_per_form: Optional[int] = None,
        min_year: Optional[int] = 2009,
        additional_blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ResolvedFiling]:
        """Parse submissions payload and extract target filings up to limit_per_form.

        Args:
            submissions_data: Deserialized SEC submissions JSON dictionary.
            target_forms: Target forms to extract (default: ('10-K', '10-Q', '10-K/A', '10-Q/A')).
            limit_per_form: Maximum filings per form type (default: None for all historical).
            min_year: Earliest year to extract (default: 2009).
            additional_blocks: Optional historical submission blocks loaded from paginated files.

        Returns:
            List of ResolvedFiling instances ordered chronologically (newest first).
        """
        cik_raw = submissions_data.get("cik", "")
        cik = str(cik_raw).zfill(10)
        tickers = submissions_data.get("tickers", [])
        ticker = tickers[0].upper() if tickers else "UNKNOWN"

        target_forms_set = {f.upper() for f in target_forms}
        filings_block = submissions_data.get("filings", {})
        recent = filings_block.get("recent", {})

        all_resolved = self._parse_entries_block(
            block=recent,
            cik=cik,
            ticker=ticker,
            target_forms_set=target_forms_set,
            min_year=min_year,
        )

        if additional_blocks:
            for hist_block in additional_blocks:
                hist_items = self._parse_entries_block(
                    block=hist_block,
                    cik=cik,
                    ticker=ticker,
                    target_forms_set=target_forms_set,
                    min_year=min_year,
                )
                all_resolved.extend(hist_items)

        # Sort chronologically newest first
        all_resolved.sort(
            key=lambda x: (x.report_date or x.filing_date, x.filing_date),
            reverse=True,
        )

        # Apply limit_per_form if specified and positive
        if limit_per_form is not None and limit_per_form > 0:
            counts_per_form: Dict[str, int] = defaultdict(int)
            filtered: List[ResolvedFiling] = []
            for item in all_resolved:
                if counts_per_form[item.form] < limit_per_form:
                    filtered.append(item)
                    counts_per_form[item.form] += 1
            all_resolved = filtered

        # Deduplicate by accession number while preserving chronological order
        seen_accessions: Set[str] = set()
        deduped: List[ResolvedFiling] = []
        counts: Dict[str, int] = defaultdict(int)
        for item in all_resolved:
            if item.accession_number not in seen_accessions:
                seen_accessions.add(item.accession_number)
                deduped.append(item)
                counts[item.form] += 1

        logger.info(
            "Resolved %d filings for %s (CIK: %s): %s",
            len(deduped),
            ticker,
            cik,
            dict(counts),
        )
        return deduped

    def resolve_from_s3(
        self,
        ticker: str,
        cik: str,
        target_forms: Sequence[str] = DEFAULT_TARGET_FORMS,
        limit_per_form: Optional[int] = None,
        min_year: Optional[int] = 2009,
    ) -> List[ResolvedFiling]:
        """Fetch submissions JSON from S3 Bronze bucket and resolve target filings.

        Also traverses and fetches any historical paginated submission files
        referenced in the 'files' array to support full historical queries (2009-present).

        Args:
            ticker: Stock ticker symbol (e.g. 'AAPL').
            cik: SEC CIK number.
            target_forms: Target forms to resolve.
            limit_per_form: Maximum filings per form type (default: None for all).
            min_year: Earliest year to extract (default: 2009).

        Returns:
            List of ResolvedFiling objects.
        """
        if not self.s3_storage:
            raise ValueError("S3Storage instance must be provided to resolve from S3.")

        cik_10 = str(cik).strip().zfill(10)
        ticker_clean = ticker.strip().upper()
        s3_key = f"raw/submissions/{ticker_clean}_{cik_10}.json"

        logger.info("Reading submissions from s3://%s/%s", self.bronze_bucket, s3_key)
        raw_bytes = self.s3_storage.download_bytes(bucket=self.bronze_bucket, key=s3_key)
        submissions_dict = json.loads(raw_bytes.decode("utf-8"))

        # Inspect and load additional historical submissions files
        additional_blocks: List[Dict[str, Any]] = []
        hist_files = submissions_dict.get("filings", {}).get("files", [])

        for f_meta in hist_files:
            file_name = f_meta.get("name", "")
            filing_to = f_meta.get("filingTo", "")
            # Check if this historical file could contain filings >= min_year
            if min_year is not None and filing_to and filing_to[:4] < str(min_year):
                continue

            hist_s3_key = f"raw/submissions/{file_name}"
            hist_bytes: Optional[bytes] = None

            if self.s3_storage.object_exists(bucket=self.bronze_bucket, key=hist_s3_key):
                hist_bytes = self.s3_storage.download_bytes(bucket=self.bronze_bucket, key=hist_s3_key)
            elif self.sec_client is not None:
                hist_url = f"https://data.sec.gov/submissions/{file_name}"
                logger.info("Fetching historical submissions file: %s", hist_url)
                resp = self.sec_client.get(hist_url)
                hist_bytes = resp.content
                file_hash = hashlib.sha256(hist_bytes).hexdigest()
                self.s3_storage.upload_bytes(
                    bucket=self.bronze_bucket,
                    key=hist_s3_key,
                    data=hist_bytes,
                    content_type="application/json",
                    metadata={"sha256": file_hash},
                )

            if hist_bytes:
                try:
                    hist_data = json.loads(hist_bytes.decode("utf-8"))
                    additional_blocks.append(hist_data)
                except Exception as exc:
                    logger.warning("Failed parsing historical file %s: %s", file_name, exc)

        return self.resolve_from_submissions_dict(
            submissions_data=submissions_dict,
            target_forms=target_forms,
            limit_per_form=limit_per_form,
            min_year=min_year,
            additional_blocks=additional_blocks,
        )
