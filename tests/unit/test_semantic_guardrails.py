"""Unit tests for 4-Layer Financial Semantic Guardrails."""

from src.core.financial.semantic_guardrails import (
    FinancialOntology,
    FinancialSemanticValidator,
    SegmentType,
)


def test_financial_ontology_segment_disambiguation() -> None:
    """Verify Layer 2: Disambiguation between Geographic and Product segments."""
    ontology = FinancialOntology()

    # 1. Product line recognition
    aapl_prod = ontology.identify_segments("Revenues for iPhone and Mac in 2023", ticker="AAPL")
    assert any(s.name == "iPhone" and s.segment_type == SegmentType.PRODUCT_LINE for s in aapl_prod)
    assert any(s.name == "Mac" and s.segment_type == SegmentType.PRODUCT_LINE for s in aapl_prod)

    # 2. Geographic segment recognition
    aapl_geo = ontology.identify_segments("Revenues in Greater China and Europe", ticker="AAPL")
    assert any(s.name == "Greater China" and s.segment_type == SegmentType.GEOGRAPHIC for s in aapl_geo)
    assert any(s.name == "Europe" and s.segment_type == SegmentType.GEOGRAPHIC for s in aapl_geo)

    # 3. Disambiguation detects mixed geographic and product query
    mixed_check = ontology.disambiguate_query("Compare iPhone revenue and Americas revenue for Apple")
    assert mixed_check["has_segment_conflict"] is True
    assert "iPhone" in mixed_check["product_lines"]
    assert "Americas" in mixed_check["geographic_segments"]


def test_ontology_does_not_match_english_pronoun_us() -> None:
    """Regression test: English pronoun 'us' must not trigger false positive United States segment."""
    ontology = FinancialOntology()
    prose = "Contact us for further inquiries regarding Microsoft cloud services."
    matched = ontology.identify_segments(prose, ticker="MSFT")
    # Should not match 'United States' purely from the word 'us'
    assert not any(s.name == "United States" for s in matched)

    # Exact uppercase 'US' or 'United States' should match
    us_matched = ontology.identify_segments("Revenues in the United States reached $100B", ticker="MSFT")
    assert any(s.name == "United States" for s in us_matched)


def test_deterministic_margin_validation() -> None:
    """Verify Layer 4: Mathematical margin validation formula with percentage point tolerance."""
    validator = FinancialSemanticValidator()

    # Case 1: Valid Apple FY2023 operating margin (114,301 / 383,285 = 29.82%)
    is_valid, calc_pct, msg = validator.validate_margin(
        revenue=383_285_000_000,
        income=114_301_000_000,
        reported_margin=29.82,
        tolerance_pct=1.5,
    )
    assert is_valid is True
    assert calc_pct == 29.82

    # Case 2: Rounding within 1.5 percentage points (Reported 30.0% vs expected 29.82% -> diff 0.18%)
    is_valid_round, _, _ = validator.validate_margin(
        revenue=383_285_000_000,
        income=114_301_000_000,
        reported_margin=30.0,
        tolerance_pct=1.5,
    )
    assert is_valid_round is True

    # Case 3: Hallucinated margin (Reporting 45% when reality is 29.82%)
    is_invalid, calc_pct2, msg2 = validator.validate_margin(
        revenue=383_285_000_000,
        income=114_301_000_000,
        reported_margin=45.0,
        tolerance_pct=1.5,
    )
    assert is_invalid is False
    assert "Margin mismatch" in msg2


def test_response_structure_guardrail() -> None:
    """Verify Layer 3: Structure compliance for official disclosures vs analysis."""
    validator = FinancialSemanticValidator()

    compliant_answer = """
### [Official SEC Disclosures]
| Fiscal Period | Revenue |
| :--- | :---: |
| FY2023 | $383.29B [1] |

### [Analysis & Commentary]
Apple's operating margin reflects sustained pricing power across product lines.
"""
    result = validator.audit_response_structure(compliant_answer)
    assert result["is_structure_compliant"] is True

    # Single-paragraph prose without headers must be rejected
    non_compliant_answer = "Apple had FY2023 revenues of $383 billion and strong profit."
    bad_result = validator.audit_response_structure(non_compliant_answer)
    assert bad_result["is_structure_compliant"] is False
