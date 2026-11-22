"""Extraction Coverage Audit and Quality Reporting Engine.

Audits extracted filing sections against SEC canonical taxonomy items, classifies
omissions (Incorporated by Reference, Not Applicable, Extracted, Missing Anomaly),
and generates audit reports in Markdown and JSON without graphical icons.
"""

import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.data.processing.section_extractor import (
    CANONICAL_10K_ITEMS,
    CANONICAL_10Q_ITEMS,
    ExtractionReport,
    SectionStatus,
)

logger = logging.getLogger(__name__)


class ItemClassification(str, Enum):
    """Categorization of an Item's status in a filing."""

    EXTRACTED = "EXTRACTED"  # Parsed and populated with text
    INCORPORATED_BY_REFERENCE = "INCORPORATED_BY_REFERENCE"  # Part III items deferred to DEF 14A
    NOT_APPLICABLE = "NOT_APPLICABLE"  # E.g. Mine Safety for tech companies
    MISSING_ANOMALY = "MISSING_ANOMALY"  # Expected standard item missing


# Standard known tech company omissions in Form 10-K
KNOWN_NOT_APPLICABLE_10K = {"ITEM 4"}  # Mine Safety Disclosures
KNOWN_PROXY_INCORPORATED_10K = {"ITEM 10", "ITEM 11", "ITEM 12", "ITEM 13", "ITEM 14"}  # Deferred to DEF 14A


class ItemCoverageDetail(BaseModel):
    """Detailed audit status for an individual filing item."""

    item_key: str
    canonical_title: str
    status: ItemClassification
    char_count: int = 0
    table_count: int = 0
    notes: str = ""


class FilingCoverageSummary(BaseModel):
    """Filing-level coverage report."""

    ticker: str
    form: str
    accession_number: str | None = None
    total_expected_items: int = 0
    total_extracted_items: int = 0
    coverage_ratio: float = 0.0
    items: list[ItemCoverageDetail] = Field(default_factory=list)
    has_item_8_notes: bool = False
    has_rd_disclosure: bool = False


class CoverageReporter:
    """Evaluates extraction completeness across SEC EDGAR HTML primary documents."""

    def evaluate_filing(self, report: ExtractionReport) -> FilingCoverageSummary:
        """Evaluate a single filing ExtractionReport against canonical expectations.

        Args:
            report: Populated ExtractionReport from HTMLSectionExtractor.

        Returns:
            FilingCoverageSummary with item-by-item classification.
        """
        form = (report.form or "10-K").upper()
        ticker = (report.ticker or "UNKNOWN").upper()

        # Strict filter: ignore NOT_FOUND placeholders and empty sections
        valid_extracted_sections = [
            s
            for s in report.sections
            if s.status in (SectionStatus.EXTRACTED, SectionStatus.NON_STANDARD)
            and (s.char_count > 0 or len(s.raw_text.strip()) > 0)
        ]

        details: list[ItemCoverageDetail] = []
        extracted_count = 0
        has_item_8_notes = False
        has_rd_disclosure = False

        # Scan for notes and R&D in sections
        for sec in valid_extracted_sections:
            sec_text_lower = sec.raw_text.lower()
            if ("item 8" in sec.item.lower() or "financial statement" in sec.title.lower()) and (
                "note" in sec_text_lower or len(sec.raw_text) > 5_000
            ):
                has_item_8_notes = True
            if any(k in sec_text_lower for k in ["research and development", "r&d"]):
                has_rd_disclosure = True

        if "10-K" in form:
            extracted_map = {s.item.upper(): s for s in valid_extracted_sections}
            total_expected = len(CANONICAL_10K_ITEMS)

            for item_key, title in CANONICAL_10K_ITEMS.items():
                if item_key in extracted_map:
                    sec = extracted_map[item_key]
                    extracted_count += 1
                    details.append(
                        ItemCoverageDetail(
                            item_key=item_key,
                            canonical_title=title,
                            status=ItemClassification.EXTRACTED,
                            char_count=sec.char_count,
                            table_count=sec.table_count,
                            notes="Successfully extracted",
                        )
                    )
                elif item_key in KNOWN_PROXY_INCORPORATED_10K:
                    details.append(
                        ItemCoverageDetail(
                            item_key=item_key,
                            canonical_title=title,
                            status=ItemClassification.INCORPORATED_BY_REFERENCE,
                            notes="Omitted in 10-K; incorporated by reference from Definitive Proxy Statement (DEF 14A).",
                        )
                    )
                elif item_key in KNOWN_NOT_APPLICABLE_10K:
                    details.append(
                        ItemCoverageDetail(
                            item_key=item_key,
                            canonical_title=title,
                            status=ItemClassification.NOT_APPLICABLE,
                            notes="Not applicable to technology hardware and cloud companies.",
                        )
                    )
                else:
                    details.append(
                        ItemCoverageDetail(
                            item_key=item_key,
                            canonical_title=title,
                            status=ItemClassification.MISSING_ANOMALY,
                            notes="Item expected but not located in filing HTML.",
                        )
                    )
        else:
            # Form 10-Q: Build composite keys f"{part} {item}" to avoid Part I vs Part II collision
            extracted_10q_map: dict[str, Any] = {}
            for s in valid_extracted_sections:
                p = (s.part or "PART I").upper()
                composite_key = f"{p} {s.item.upper()}"
                extracted_10q_map[composite_key] = s

            expected_10q_items: dict[str, str] = {}
            for part, items in CANONICAL_10Q_ITEMS.items():
                for it_k, it_name in items.items():
                    composite = f"{part.upper()} {it_k.upper()}"
                    expected_10q_items[composite] = f"{part} {it_name}"

            total_expected = len(expected_10q_items)

            for composite_key, title in expected_10q_items.items():
                if composite_key in extracted_10q_map:
                    sec = extracted_10q_map[composite_key]
                    extracted_count += 1
                    details.append(
                        ItemCoverageDetail(
                            item_key=composite_key,
                            canonical_title=title,
                            status=ItemClassification.EXTRACTED,
                            char_count=sec.char_count,
                            table_count=sec.table_count,
                            notes="Successfully extracted",
                        )
                    )
                else:
                    details.append(
                        ItemCoverageDetail(
                            item_key=composite_key,
                            canonical_title=title,
                            status=ItemClassification.MISSING_ANOMALY,
                            notes="10-Q section missing from filing HTML.",
                        )
                    )

        coverage_ratio = round(extracted_count / total_expected, 4) if total_expected > 0 else 0.0

        return FilingCoverageSummary(
            ticker=ticker,
            form=form,
            accession_number=report.accession_number,
            total_expected_items=total_expected,
            total_extracted_items=extracted_count,
            coverage_ratio=coverage_ratio,
            items=details,
            has_item_8_notes=has_item_8_notes,
            has_rd_disclosure=has_rd_disclosure,
        )

    def generate_markdown_report(self, summaries: list[FilingCoverageSummary]) -> str:
        """Format a list of FilingCoverageSummary objects into an executive Markdown report."""
        lines = [
            "# SEC EDGAR Filing Section Extraction Coverage Report",
            "",
            "| Ticker | Form | Accession Number | Total Expected | Extracted | Coverage Ratio | Item 8 Notes | R&D Disclosed |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for s in summaries:
            item8_status = "[YES]" if s.has_item_8_notes else "[NO]"
            rd_status = "[YES]" if s.has_rd_disclosure else "[NO]"
            lines.append(
                f"| {s.ticker} | {s.form} | `{s.accession_number or 'N/A'}` | "
                f"{s.total_expected_items} | {s.total_extracted_items} | {s.coverage_ratio * 100:.1f}% | "
                f"{item8_status} | {rd_status} |"
            )
        return "\n".join(lines)
