"""Unit tests for HTMLSectionExtractor supporting Form 10-K, 10-Q, and Form 20-F/A."""

import pytest
from src.data.processing.section_extractor import (
    HTMLSectionExtractor,
    CANONICAL_10K_ITEMS,
    CANONICAL_10Q_ITEMS,
    CANONICAL_20F_ITEMS,
    SectionStatus,
)


@pytest.fixture
def extractor() -> HTMLSectionExtractor:
    """Fixture providing HTMLSectionExtractor instance."""
    return HTMLSectionExtractor()


class TestHTMLSectionExtractorForm20F:
    """Validate section extraction logic for Form 20-F and 20-F/A (Foreign Private Issuers)."""

    def test_form_20f_canonical_items_mapping(self, extractor: HTMLSectionExtractor):
        """Verify that Form 20-F items map to correct canonical foreign issuer disclosures."""
        html_doc = """
        <html>
        <body>
            <div><p>PART I</p></div>
            <div><h1>ITEM 3. KEY INFORMATION</h1><p>General economic and market conditions overview.</p></div>
            <div><h2>ITEM 3.D. RISK FACTORS</h2><p>Risks related to semiconductor supply chain and geopolitics.</p></div>
            <div><h1>ITEM 4. INFORMATION ON THE COMPANY</h1><p>TSMC was established in 1987 in Hsinchu Science Park.</p></div>
            <div><h1>ITEM 4A. UNRESOLVED STAFF COMMENTS</h1><p>None.</p></div>
            <div><h1>ITEM 5. OPERATING AND FINANCIAL REVIEW AND PROSPECTS</h1><p>Net revenue increased 24.9% year-over-year.</p></div>
            <div><h1>ITEM 8. FINANCIAL INFORMATION</h1>
                <p>Consolidated Financial Statements and Supplementary Data.</p>
                <h3>Note 1. General Information</h3>
                <p>Taiwan Semiconductor Manufacturing Company Limited details.</p>
            </div>
            <div><p>PART II</p></div>
            <div><h1>ITEM 15. CONTROLS AND PROCEDURES</h1><p>Evaluation of disclosure controls.</p></div>
            <div><h1>ITEM 16K. CYBERSECURITY</h1><p>Cybersecurity risk management and strategy.</p></div>
        </body>
        </html>
        """
        report = extractor.extract(
            html_content=html_doc,
            form="20-F",
            ticker="TSM",
            accession_number="0001065280-24-000010",
        )

        extracted_items = {s.item: s for s in report.sections if s.status == SectionStatus.EXTRACTED}

        # 1. Item 3: Key Information
        assert "ITEM 3" in extracted_items
        assert extracted_items["ITEM 3"].title == "Key Information"

        # 2. Item 3.D: Risk Factors
        assert "ITEM 3D" in extracted_items
        assert extracted_items["ITEM 3D"].title == "Risk Factors"
        assert "Risks related to semiconductor" in extracted_items["ITEM 3D"].raw_text

        # 3. Item 4: Information on the Company (NOT Controls and Procedures)
        assert "ITEM 4" in extracted_items
        assert extracted_items["ITEM 4"].title == "Information on the Company"
        assert "TSMC was established" in extracted_items["ITEM 4"].raw_text

        # 4. Item 4A: Unresolved Staff Comments
        assert "ITEM 4A" in extracted_items
        assert extracted_items["ITEM 4A"].title == "Unresolved Staff Comments"

        # 5. Item 5: Operating and Financial Review and Prospects (NOT Other Information)
        assert "ITEM 5" in extracted_items
        assert extracted_items["ITEM 5"].title == "Operating and Financial Review and Prospects"
        assert "Net revenue increased" in extracted_items["ITEM 5"].raw_text

        # 6. Item 8: Financial Information and Note disclosure
        assert "ITEM 8" in extracted_items
        assert extracted_items["ITEM 8"].title == "Financial Information"

        # Check financial note parsed within Item 8
        non_standard_items = {s.item: s for s in report.sections if s.status == SectionStatus.NON_STANDARD}
        assert "ITEM_8_NOTE_1" in non_standard_items

        # 7. Item 15 & Item 16K
        assert "ITEM 15" in extracted_items
        assert extracted_items["ITEM 15"].title == "Controls and Procedures"

        assert "ITEM 16K" in extracted_items
        assert extracted_items["ITEM 16K"].title == "Cybersecurity"

    def test_form_20f_missing_canonical_items_are_reported(self, extractor: HTMLSectionExtractor):
        """Verify that absent Form 20-F canonical items are flagged as NOT_FOUND."""
        html_doc = """
        <html>
        <body>
            <div><h1>ITEM 3. KEY INFORMATION</h1><p>Content.</p></div>
        </body>
        </html>
        """
        report = extractor.extract(html_content=html_doc, form="20-F")
        not_found_items = {s.item for s in report.sections if s.status == SectionStatus.NOT_FOUND}

        # Form 20-F canonical items like ITEM 4, ITEM 5, ITEM 8 must be flagged
        assert "ITEM 4" in not_found_items
        assert "ITEM 5" in not_found_items
        assert "ITEM 8" in not_found_items
        assert "ITEM 4" in report.unidentified_sections


class TestHTMLSectionExtractorForm10KAnd10Q:
    """Ensure backward compatibility for Form 10-K and 10-Q parsing."""

    def test_form_10k_canonical_mapping(self, extractor: HTMLSectionExtractor):
        """Verify Form 10-K correctly maps Item 1, Item 1A, and Item 7."""
        html_doc = """
        <html>
        <body>
            <div><h1>ITEM 1. BUSINESS</h1><p>Apple designs smartphones.</p></div>
            <div><h1>ITEM 1A. RISK FACTORS</h1><p>Global economic conditions.</p></div>
            <div><h1>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS</h1><p>Revenue overview.</p></div>
        </body>
        </html>
        """
        report = extractor.extract(html_content=html_doc, form="10-K", ticker="AAPL")
        extracted_items = {s.item: s for s in report.sections if s.status == SectionStatus.EXTRACTED}

        assert "ITEM 1" in extracted_items
        assert extracted_items["ITEM 1"].title == "Business"
        assert "ITEM 1A" in extracted_items
        assert extracted_items["ITEM 1A"].title == "Risk Factors"
        assert "ITEM 7" in extracted_items
        assert "Management's Discussion" in extracted_items["ITEM 7"].title

    def test_form_10q_canonical_mapping(self, extractor: HTMLSectionExtractor):
        """Verify Form 10-Q correctly maps Part I and Part II items."""
        html_doc = """
        <html>
        <body>
            <div><p>PART I</p></div>
            <div><h1>ITEM 1. FINANCIAL STATEMENTS</h1><p>Condensed consolidated statements.</p></div>
            <div><h1>ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS</h1><p>Three months ended March 31.</p></div>
            <div><p>PART II</p></div>
            <div><h1>ITEM 1A. RISK FACTORS</h1><p>Quarterly risk updates.</p></div>
        </body>
        </html>
        """
        report = extractor.extract(html_content=html_doc, form="10-Q", ticker="MSFT")
        extracted_items = {s.item: s for s in report.sections if s.status == SectionStatus.EXTRACTED}

        assert "ITEM 1" in extracted_items
        assert extracted_items["ITEM 1"].title == "Financial Statements"
        assert "ITEM 2" in extracted_items
        assert "Management's Discussion" in extracted_items["ITEM 2"].title
        assert "ITEM 1A" in extracted_items
        assert extracted_items["ITEM 1A"].title == "Risk Factors"
