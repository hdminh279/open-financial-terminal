"""Fiscal year parsing and time horizon expansion utilities.

Provides centralized, deterministic parsing of fiscal year ranges from
user query strings and query plans without duplicated heuristics.
"""

import re
from typing import List, Optional, Sequence


def extract_years_from_text(text: str) -> List[int]:
    """Extract distinct 4-digit calendar/fiscal years sorted chronologically."""
    if not text:
        return []
    matches = re.findall(r"\b(19\d\d|20\d\d)\b", text)
    return sorted(list(dict.fromkeys(int(m) for m in matches)))


def expand_year_range(
    years: Sequence[int],
    max_expansion_span: int = 30,
) -> List[int]:
    """Expand a 2-year boundary pair [start, end] into a full inclusive sequence."""
    if len(years) == 2 and years[0] < years[1]:
        if years[1] - years[0] <= max_expansion_span:
            return list(range(years[0], years[1] + 1))
    return list(years)


def parse_fiscal_years(
    text: str,
    explicit_years: Optional[Sequence[int]] = None,
    max_expansion_span: int = 30,
) -> List[int]:
    """Parse and expand fiscal years from explicit input or natural language query text.

    Args:
        text: Query text string to inspect if explicit_years is not provided.
        explicit_years: Optional pre-parsed or planned fiscal years.
        max_expansion_span: Maximum span in years to auto-expand (default 30).

    Returns:
        Sorted list of integer fiscal years.
    """
    if explicit_years:
        valid_explicit = [int(y) for y in explicit_years if isinstance(y, int) and 1900 <= y <= 2100]
        if len(valid_explicit) == 2 and valid_explicit[0] < valid_explicit[1]:
            return expand_year_range(valid_explicit, max_expansion_span=max_expansion_span)
        return valid_explicit

    found = extract_years_from_text(text)
    if len(found) >= 2 and min(found) < max(found):
        span = max(found) - min(found)
        if span <= max_expansion_span:
            return list(range(min(found), max(found) + 1))
    return found
