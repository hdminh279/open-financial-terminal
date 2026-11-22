"""Fiscal year parsing and time horizon expansion utilities.

Provides centralized, deterministic parsing of fiscal year ranges from
user query strings and query plans without duplicated heuristics.
"""

import re
from collections.abc import Sequence
from datetime import date, datetime


def _coerce_date(value: date | datetime | str | None) -> date | None:
    """Convert date, datetime, or ISO date string into a date object."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError:
            return None
    return None


def dominant_calendar_year(
    start_date: date | datetime | str | None,
    end_date: date | datetime | str | None,
) -> tuple[int | None, float]:
    """Compute the calendar year containing the largest share of days in [start_date, end_date].

    Args:
        start_date: Inclusive period start date (None for point-in-time instant balances).
        end_date: Inclusive period end date.

    Returns:
        Tuple of (dominant_year, share_ratio) where share_ratio is in [0.0, 1.0].
        Returns (None, 0.0) when dates are missing or inverted.
    """
    start = _coerce_date(start_date)
    end = _coerce_date(end_date)
    if start is None or end is None or end < start:
        return None, 0.0

    total_days = (end - start).days + 1
    best_year: int | None = None
    best_days = -1

    for yr in range(start.year, end.year + 1):
        seg_start = max(start, date(yr, 1, 1))
        seg_end = min(end, date(yr, 12, 31))
        days_in_year = (seg_end - seg_start).days + 1
        if days_in_year > best_days:
            best_days = days_in_year
            best_year = yr

    return best_year, best_days / total_days


def is_shifted_fiscal_period(
    fiscal_year: int,
    start_date: date | datetime | str | None,
    end_date: date | datetime | str | None,
    threshold: float = 0.75,
) -> tuple[bool, int | None]:
    """Determine whether a reporting period falls predominantly in a different calendar year.

    For example, NVIDIA's fiscal year ends in late January, so FY2027 Q1 runs from
    late January 2026 to late April 2026 (100% inside calendar year 2026), whereas
    Microsoft's June 30 fiscal year splits ~50/50 and does not trigger a shift warning.

    Args:
        fiscal_year: Authoritative SEC/VAS fiscal year label.
        start_date: Inclusive start date of the duration period.
        end_date: Inclusive end date of the duration period.
        threshold: Minimum share of period days required in the non-matching calendar year.

    Returns:
        Tuple of (is_shifted, dominant_calendar_year).
    """
    dom_year, share = dominant_calendar_year(start_date, end_date)
    if dom_year is None:
        return False, None
    return (dom_year != int(fiscal_year) and share >= threshold), dom_year


def extract_years_from_text(text: str) -> list[int]:
    """Extract distinct 4-digit calendar/fiscal years sorted chronologically."""
    if not text:
        return []
    matches = re.findall(r"\b(19\d\d|20\d\d)\b", text)
    return sorted(list(dict.fromkeys(int(m) for m in matches)))


def expand_year_range(
    years: Sequence[int],
    max_expansion_span: int = 30,
) -> list[int]:
    """Expand a 2-year boundary pair [start, end] into a full inclusive sequence."""
    if len(years) == 2 and years[0] < years[1] and years[1] - years[0] <= max_expansion_span:
        return list(range(years[0], years[1] + 1))
    return list(years)


def parse_fiscal_years(
    text: str,
    explicit_years: Sequence[int] | None = None,
    max_expansion_span: int = 30,
) -> list[int]:
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
