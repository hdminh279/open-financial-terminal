"""Unit tests for Extraction Coverage Audit & Item 8 Note Disclosures."""

from bs4 import BeautifulSoup

from pipelines.processing.coverage_reporter import (
    CoverageReporter,
    ItemClassification,
)
from pipelines.processing.section_extractor import (
    ExtractedSection,
    ExtractionReport,
    HTMLSectionExtractor,
    SectionStatus,
)


def test_coverage_reporter_ignores_not_found_placeholders() -> None:
    """Regression test: NOT_FOUND placeholder sections must not inflate extracted count."""
    report = ExtractionReport(
        ticker="AAPL",
        form="10-K",
        accession_number="0000320193-23-000106",
        sections=[
            ExtractedSection(
                item="ITEM 1",
                title="Business",
                status=SectionStatus.EXTRACTED,
                raw_text="Apple designs and manufactures personal devices.",
                char_count=50,
            ),
            # Placeholder created by section_extractor for unlocated items
            ExtractedSection(
                item="ITEM 2",
                title="Properties",
                status=SectionStatus.NOT_FOUND,
                raw_text="",
                char_count=0,
            ),
            ExtractedSection(
                item="ITEM 3",
                title="Legal Proceedings",
                status=SectionStatus.NOT_FOUND,
                raw_text="",
                char_count=0,
            ),
        ],
    )

    reporter = CoverageReporter()
    summary = reporter.evaluate_filing(report)

    # Only ITEM 1 should be counted as extracted
    assert summary.total_extracted_items == 1
    item_map = {it.item_key: it for it in summary.items}
    assert item_map["ITEM 1"].status == ItemClassification.EXTRACTED
    assert item_map["ITEM 2"].status == ItemClassification.MISSING_ANOMALY
    assert item_map["ITEM 3"].status == ItemClassification.MISSING_ANOMALY


def test_coverage_reporter_10q_composite_parts() -> None:
    """Regression test: Form 10-Q Part I Item 1 must not satisfy Part II Item 1."""
    report = ExtractionReport(
        ticker="MSFT",
        form="10-Q",
        accession_number="0000063276-23-000001",
        sections=[
            ExtractedSection(
                part="PART I",
                item="ITEM 1",
                title="Financial Statements",
                status=SectionStatus.EXTRACTED,
                raw_text="Consolidated Balance Sheets...",
                char_count=5000,
            )
        ],
    )

    reporter = CoverageReporter()
    summary = reporter.evaluate_filing(report)

    item_map = {it.item_key: it for it in summary.items}
    assert item_map["PART I ITEM 1"].status == ItemClassification.EXTRACTED
    assert item_map["PART II ITEM 1"].status == ItemClassification.MISSING_ANOMALY


def test_note_disclosure_heading_vs_prose() -> None:
    """Regression test: Item 8 Note headings should be detected while long prose is preserved."""
    extractor = HTMLSectionExtractor()

    # Valid heading: short, bold/heading tag, no terminal period
    soup_heading = BeautifulSoup("<h3>Note 1 – Summary of Significant Accounting Policies</h3>", "html.parser")
    tag_h = soup_heading.find("h3")
    res_h = extractor._is_note_heading(tag_h, tag_h.get_text())
    assert res_h is not None
    assert res_h[0] == "1"

    # Prose sentence: long or ending with period -> must return None
    soup_prose = BeautifulSoup(
        "<p>Note 1 to our consolidated financial statements describes our revenue recognition policy.</p>",
        "html.parser",
    )
    tag_p = soup_prose.find("p")
    res_p = extractor._is_note_heading(tag_p, tag_p.get_text())
    assert res_p is None


def test_coverage_markdown_report_generation() -> None:
    """Verify CoverageReporter renders a formatted executive Markdown summary table without icons."""
    report = ExtractionReport(
        ticker="GOOGL",
        form="10-K",
        accession_number="0001652044-24-000022",
        sections=[
            ExtractedSection(
                item="ITEM 8",
                title="Financial Statements",
                status=SectionStatus.EXTRACTED,
                raw_text="Research and development costs were $39.5B in 2022, $45.4B in 2023, $49.3B in 2024.",
                char_count=20000,
            )
        ],
    )
    reporter = CoverageReporter()
    summary = reporter.evaluate_filing(report)
    md = reporter.generate_markdown_report([summary])

    assert "| GOOGL | 10-K |" in md
    assert "[YES]" in md
    # Ensure no emojis in output
    assert "✅" not in md
    assert "❌" not in md
