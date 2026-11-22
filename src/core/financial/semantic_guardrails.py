"""Financial Semantic Guardrails and Segment Ontology Engine.

Provides 4-layer financial guardrails:
1. Metric Registry GAAP vs Non-GAAP metadata validation.
2. Strict disambiguation between Geographic Segments and Product Line categories.
3. Evidence-based response structure enforcement (Official Disclosures vs Analysis).
4. Deterministic mathematical checks (Operating/Net Margin formula and unit consistency).
"""

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class SegmentType(str, Enum):
    """Authoritative semantic type of corporate reporting segment."""

    GEOGRAPHIC = "geographic"  # Regional markets: Americas, Europe, Greater China, APAC
    PRODUCT_LINE = "product_line"  # Product/service lines: iPhone, Mac, Azure, Google Cloud
    REPORTABLE_OPERATING = "operating"  # Official operating segments under ASC 280
    CONSOLIDATED = "consolidated"  # Company-wide total


@dataclass(frozen=True)
class SegmentMetadata:
    """Immutable domain representation of a company segment."""

    ticker: str
    name: str
    segment_type: SegmentType
    aliases: list[str] = field(default_factory=list)
    parent_segment: str | None = None
    note_reference: str = "Item 8 Note - Segment Information"


class FinancialOntology:
    """Ontology repository for corporate segments and disaggregated product lines."""

    DEFAULT_CONFIG_PATH = "configs/segments.yaml"

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH) -> None:
        """Initialize company segment definitions from YAML configuration."""
        self.config_path = config_path
        self._segments: dict[str, list[SegmentMetadata]] = {}
        self._load_segments()

    def _load_segments(self) -> None:
        """Load and index segments from external configuration file (SSOT)."""
        if not os.path.exists(self.config_path):
            logger.warning("Segments config file not found: %s", self.config_path)
            return

        with open(self.config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        raw_segments = data.get("segments", {})
        for ticker, items in raw_segments.items():
            self._segments[ticker] = []
            for item in items:
                seg_type_str = item.get("type", "product_line")
                try:
                    seg_type = SegmentType(seg_type_str)
                except ValueError:
                    seg_type = SegmentType.PRODUCT_LINE

                metadata = SegmentMetadata(
                    ticker=ticker,
                    name=item.get("name", ""),
                    segment_type=seg_type,
                    aliases=item.get("aliases", []),
                    parent_segment=item.get("parent_segment"),
                    note_reference=item.get("note_reference", "Item 8 Note - Segment Information"),
                )
                self._segments[ticker].append(metadata)

    def identify_segments(self, text: str, ticker: str | None = None) -> list[SegmentMetadata]:
        """Identify mentioned corporate segments and their semantic classification.

        Args:
            text: Query string or text excerpt.
            ticker: Optional target ticker to scope search and prevent cross-company false positives.

        Returns:
            List of matched SegmentMetadata instances.
        """
        if not text:
            return []
        text_lower = text.lower()
        matched: list[SegmentMetadata] = []

        tickers_to_check = [ticker] if ticker and ticker in self._segments else list(self._segments.keys())
        for t in tickers_to_check:
            for seg in self._segments[t]:
                all_names = [seg.name.lower()] + [a.lower() for a in seg.aliases]
                matched_alias = False
                for alias in all_names:
                    # Require boundary and minimum 3 chars unless exact match for "u.s."
                    if (len(alias) >= 3 or alias == "u.s.") and re.search(rf"\b{re.escape(alias)}\b", text_lower):
                        matched_alias = True
                        break

                # Separate case-sensitive check for uppercase 'US'
                if not matched_alias and seg.name == "United States" and re.search(r"\bUS\b", text):
                    matched_alias = True

                if matched_alias and seg not in matched:
                    matched.append(seg)
        return matched

    def disambiguate_query(self, query: str, ticker: str | None = None) -> dict[str, Any]:
        """Detect whether query mixes Geographic and Product Line categories.

        Args:
            query: User natural language inquiry.
            ticker: Optional ticker symbol.

        Returns:
            Dictionary with disambiguation flags and guidance.
        """
        matched = self.identify_segments(query, ticker)
        geo = [s for s in matched if s.segment_type == SegmentType.GEOGRAPHIC]
        prod = [s for s in matched if s.segment_type in (SegmentType.PRODUCT_LINE, SegmentType.REPORTABLE_OPERATING)]

        has_conflict = bool(geo and prod)
        return {
            "has_segment_conflict": has_conflict,
            "geographic_segments": [g.name for g in geo],
            "product_lines": [p.name for p in prod],
            "guidance": (
                "Notice: Geographic segment reporting (ASC 280) is distinct from product-line disclosures. "
                "Both dimensions are resolved and reported separately."
                if has_conflict
                else ""
            ),
        }


class FinancialSemanticValidator:
    """Deterministic mathematical and unit consistency validator for financial metrics."""

    @staticmethod
    def validate_margin(
        revenue: float,
        income: float,
        reported_margin: float,
        tolerance_pct: float = 1.5,
    ) -> tuple[bool, float, str]:
        """Validate whether reported margin matches income / revenue within tolerance.

        Args:
            revenue: Total revenue denominator.
            income: Operating income or Net income numerator.
            reported_margin: Margin in percentage (e.g. 29.8 for 29.8% or 0.298).
            tolerance_pct: Permissible margin difference in percentage points (default 1.5 pp).

        Returns:
            Tuple of (is_valid, calculated_percentage, audit_message).
        """
        if revenue <= 0:
            return False, 0.0, "Invalid revenue denominator (<= 0)."

        expected_ratio = income / revenue
        # Normalize reported_margin if passed as ratio (0.298 -> 29.8, -0.05 -> -5.0)
        norm_reported_pct = reported_margin * 100 if abs(reported_margin) <= 1.0 else reported_margin
        expected_pct = expected_ratio * 100

        diff_pct = abs(expected_pct - norm_reported_pct)
        is_valid = diff_pct <= tolerance_pct

        msg = (
            f"Margin valid: calculated={expected_pct:.2f}%, reported={norm_reported_pct:.2f}% (diff: {diff_pct:.2f}% <= {tolerance_pct}%)"
            if is_valid
            else f"Margin mismatch: expected {expected_pct:.2f}% based on {income}/{revenue}, but reported {norm_reported_pct:.2f}% (diff: {diff_pct:.2f}% > {tolerance_pct}%)"
        )
        return is_valid, round(expected_pct, 2), msg

    @staticmethod
    def normalize_monetary_unit(val: float, raw_unit: str) -> float:
        """Normalize raw numerical scalar based on magnitude unit string."""
        unit_lower = (raw_unit or "").lower()
        if "billion" in unit_lower or " b" in unit_lower:
            return val * 1_000_000_000
        if "million" in unit_lower or " m" in unit_lower:
            return val * 1_000_000
        return val

    @staticmethod
    def audit_response_structure(answer: str) -> dict[str, Any]:
        """Verify Layer 3 requirement: response contains both official disclosure and analysis sections.

        Enforces strict header or table structure, rejecting arbitrary narrative prose.
        """
        has_official_header = bool(
            re.search(
                r"###?\s*\[?(?:Official (?:SEC )?Disclosures?|Company Disclosures|Factual Disclosures)\]?",
                answer,
                re.IGNORECASE,
            )
        )
        has_table = bool(re.search(r"\|[^\n]+\|\n\|[\s:-]+\|", answer))
        has_official = has_official_header or has_table

        has_analysis_header = bool(
            re.search(
                r"###?\s*\[?(?:Analysis|Commentary|Interpretation)\]?",
                answer,
                re.IGNORECASE,
            )
        )

        return {
            "has_official_section": has_official,
            "has_analysis_section": has_analysis_header,
            "is_structure_compliant": has_official and has_analysis_header,
        }
