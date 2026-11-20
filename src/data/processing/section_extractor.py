"""HTML Parsing & Section Extraction Engine for SEC EDGAR Filings (Form 10-K & 10-Q).

Handles DOM parsing, Table of Contents (TOC) isolation, bookmark/anchor resolution,
split-span heading detection fallback, structural content extraction (paragraphs, headings,
lists, tables), and non-standard narrative items (such as Executive Officers).
"""

from enum import Enum
import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple
from bs4 import BeautifulSoup, Comment, Tag
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SectionStatus(str, Enum):
    """Extraction status flag for each document section."""
    EXTRACTED = "EXTRACTED"
    NOT_FOUND = "NOT_FOUND"
    UNCERTAIN_BOUNDARY = "UNCERTAIN_BOUNDARY"
    NON_STANDARD = "NON_STANDARD"


class ContentNodeType(str, Enum):
    """Semantic type of content node within a section."""
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST = "list"
    TABLE = "table"


class ExtractedContentNode(BaseModel):
    """Atomic structural unit extracted from filing HTML."""
    node_type: ContentNodeType
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExtractedSection(BaseModel):
    """Complete representation of a parsed Part/Item section."""
    part: Optional[str] = None
    item: str
    title: str
    status: SectionStatus = SectionStatus.EXTRACTED
    is_standard: bool = True
    start_anchor: Optional[str] = None
    nodes: List[ExtractedContentNode] = Field(default_factory=list)
    raw_text: str = ""
    table_count: int = 0
    char_count: int = 0


class ExtractionReport(BaseModel):
    """Summary report of filing section extraction."""
    ticker: Optional[str] = None
    form: Optional[str] = None
    accession_number: Optional[str] = None
    sections: List[ExtractedSection] = Field(default_factory=list)
    toc_found: bool = False
    toc_anchors_count: int = 0
    unidentified_sections: List[str] = Field(default_factory=list)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


CANONICAL_10K_ITEMS: Dict[str, str] = {
    "ITEM 1": "Business",
    "ITEM 1A": "Risk Factors",
    "ITEM 1B": "Unresolved Staff Comments",
    "ITEM 1C": "Cybersecurity",
    "ITEM 2": "Properties",
    "ITEM 3": "Legal Proceedings",
    "ITEM 4": "Mine Safety Disclosures",
    "ITEM 5": "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities",
    "ITEM 6": "Reserved / Selected Financial Data",
    "ITEM 7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "ITEM 7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "ITEM 8": "Financial Statements and Supplementary Data",
    "ITEM 9": "Changes in and Disagreements With Accountants on Accounting and Financial Disclosure",
    "ITEM 9A": "Controls and Procedures",
    "ITEM 9B": "Other Information",
    "ITEM 9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "ITEM 10": "Directors, Executive Officers and Corporate Governance",
    "ITEM 11": "Executive Compensation",
    "ITEM 12": "Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters",
    "ITEM 13": "Certain Relationships and Related Transactions, and Director Independence",
    "ITEM 14": "Principal Accountant Fees and Services",
    "ITEM 15": "Exhibits, Financial Statement Schedules",
    "ITEM 16": "Form 10-K Summary",
}

CANONICAL_10Q_ITEMS: Dict[str, Dict[str, str]] = {
    "PART I": {
        "ITEM 1": "Financial Statements",
        "ITEM 2": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
        "ITEM 3": "Quantitative and Qualitative Disclosures About Market Risk",
        "ITEM 4": "Controls and Procedures",
    },
    "PART II": {
        "ITEM 1": "Legal Proceedings",
        "ITEM 1A": "Risk Factors",
        "ITEM 2": "Unregistered Sales of Equity Securities and Use of Proceeds",
        "ITEM 3": "Defaults Upon Senior Securities",
        "ITEM 4": "Mine Safety Disclosures",
        "ITEM 5": "Other Information",
        "ITEM 6": "Exhibits",
    },
}

PART_REGEX = re.compile(r"^\s*PART\s+([IVX]+)\b", re.IGNORECASE)
ITEM_REGEX = re.compile(r"^\s*ITEM\s+([0-9]{1,2}[A-Z]?)[.\s–—\-]*(.*?)$", re.IGNORECASE)
EXECUTIVE_OFFICERS_REGEX = re.compile(
    r"(?:INFORMATION\s+ABOUT\s+(?:OUR\s+)?EXECUTIVE\s+OFFICERS|EXECUTIVE\s+OFFICERS\s+OF\s+THE\s+REGISTRANT)",
    re.IGNORECASE,
)
BLOCK_TAGS = {"p", "div", "table", "ul", "ol", "dl", "h1", "h2", "h3", "h4", "h5", "h6"}


class HTMLSectionExtractor:
    """Production-grade HTML Parser and Section Extractor for SEC EDGAR Filings."""

    def __init__(self) -> None:
        """Initialize the section extractor."""
        pass

    def sanitize_dom(self, soup: BeautifulSoup) -> None:
        """Strip non-visible tags, scripts, styles, and hidden XBRL or CSS elements.

        Preserves narrative disclosures inside inline XBRL tags such as <ix:nonNumeric>.
        """
        for tag in soup.find_all(["script", "style", "meta", "link", "noscript"]):
            tag.decompose()

        for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
            comment.extract()

        for tag in soup.find_all(["ix:hidden", "ix:header"]):
            tag.decompose()

        hidden_style_pattern = re.compile(
            r"(?:display\s*:\s*none|visibility\s*:\s*hidden)", re.IGNORECASE
        )
        for tag in soup.find_all(style=hidden_style_pattern):
            tag.decompose()

    def identify_toc_elements(self, soup: BeautifulSoup) -> Tuple[Set[Tag], Dict[str, str]]:
        """Identify Table of Contents containers and extract anchor links mapping items to IDs.

        Returns:
            Tuple of (set of TOC container tags, dict mapping canonical item keys to anchor IDs).
        """
        toc_elements: Set[Tag] = set()
        anchor_map: Dict[str, str] = {}

        toc_anchors = soup.find_all("a", href=re.compile(r"^#.+"))
        for a in toc_anchors:
            href = a.get("href", "")
            target_id = href.lstrip("#")
            text = self._normalize_text(a.get_text(" ", strip=True))
            m_item = ITEM_REGEX.match(text)
            if m_item:
                item_key = f"ITEM {m_item.group(1).upper()}"
                anchor_map[item_key] = target_id
                parent = a.find_parent(["table", "div"])
                if parent:
                    toc_elements.add(parent)
            elif EXECUTIVE_OFFICERS_REGEX.search(text):
                anchor_map["EXECUTIVE_OFFICERS"] = target_id
                parent = a.find_parent(["table", "div"])
                if parent:
                    toc_elements.add(parent)

        for table in soup.find_all("table"):
            table_text = self._normalize_text(table.get_text(" ", strip=True))
            matches = re.findall(r"\bItem\s+[0-9]{1,2}[A-Z]?", table_text, re.IGNORECASE)
            if len(matches) >= 3:
                toc_elements.add(table)

        return toc_elements, anchor_map

    def _normalize_text(self, text: str) -> str:
        """Normalize Unicode characters, non-breaking spaces, and collapse redundant whitespace."""
        if not text:
            return ""
        norm = unicodedata.normalize("NFKC", text)
        return re.sub(r"\s+", " ", norm).strip()

    def _extract_leaf_blocks(self, element: Tag, toc_elements: Set[Tag]) -> List[Tag]:
        """Recursively collect leaf block-level elements, skipping TOC containers."""
        if element in toc_elements:
            return []

        blocks: List[Tag] = []
        for child in element.children:
            if not isinstance(child, Tag):
                continue
            if child in toc_elements:
                continue

            tag_name = child.name.lower()
            if tag_name in ("table", "ul", "ol", "dl", "h1", "h2", "h3", "h4", "h5", "h6"):
                blocks.append(child)
            elif tag_name == "p":
                blocks.append(child)
            elif tag_name == "div":
                has_nested_block = child.find(lambda t: t != child and t.name.lower() in BLOCK_TAGS) is not None
                if has_nested_block:
                    blocks.extend(self._extract_leaf_blocks(child, toc_elements))
                else:
                    if child.get_text(strip=True):
                        blocks.append(child)
            else:
                blocks.extend(self._extract_leaf_blocks(child, toc_elements))

        return blocks

    def _parse_content_node(self, tag: Tag) -> Optional[ExtractedContentNode]:
        """Convert a BeautifulSoup tag into an ExtractedContentNode preserving structure."""
        tag_name = tag.name.lower()
        if tag_name == "table":
            headers: List[str] = []
            rows: List[List[str]] = []
            for tr in tag.find_all("tr"):
                th_cells = [self._normalize_text(th.get_text(" ", strip=True)) for th in tr.find_all("th")]
                td_cells = [self._normalize_text(td.get_text(" ", strip=True)) for td in tr.find_all("td")]
                if th_cells and not headers:
                    headers = th_cells
                elif th_cells and not td_cells and not headers:
                    headers = th_cells
                elif td_cells:
                    rows.append(td_cells)
            caption_elem = tag.find("caption")
            caption = self._normalize_text(caption_elem.get_text(" ", strip=True)) if caption_elem else ""
            table_text = "\n".join([" | ".join(row) for row in rows])
            return ExtractedContentNode(
                node_type=ContentNodeType.TABLE,
                text=table_text,
                metadata={"headers": headers, "rows": rows, "caption": caption},
            )
        elif tag_name in ("ul", "ol", "dl"):
            items: List[str] = []
            for li in tag.find_all(["li", "dt", "dd"]):
                item_text = self._normalize_text(li.get_text(" ", strip=True))
                if item_text:
                    prefix = "* " if tag_name == "ul" else "- "
                    items.append(f"{prefix}{item_text}")
            return ExtractedContentNode(
                node_type=ContentNodeType.LIST,
                text="\n".join(items),
                metadata={"item_count": len(items)},
            )
        elif tag_name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            text = self._normalize_text(tag.get_text(" ", strip=True))
            if not text:
                return None
            level = int(tag_name[1])
            return ExtractedContentNode(
                node_type=ContentNodeType.HEADING,
                text=text,
                metadata={"level": level},
            )
        else:
            text = self._normalize_text(tag.get_text(" ", strip=True))
            if not text:
                return None
            is_bold = bool(
                tag.find(["b", "strong"])
                or "font-weight:bold" in tag.get("style", "").replace(" ", "").lower()
            )
            if is_bold and len(text) < 120 and not text.endswith("."):
                return ExtractedContentNode(
                    node_type=ContentNodeType.HEADING,
                    text=text,
                    metadata={"level": 3, "inferred": True},
                )
            return ExtractedContentNode(
                node_type=ContentNodeType.PARAGRAPH,
                text=text,
                metadata={},
            )

    def _is_note_heading(self, tag: Tag, text: str) -> Optional[Tuple[str, str]]:
        """Determine if block element represents an Item 8 Note disclosure heading.

        Must meet all constraints:
        1. Short text length: <= 120 characters
        2. Does NOT end with sentence-ending punctuation (., ;)
        3. Matches Note pattern: 'Note X - Title' or 'Note X. Title' or 'Note X: Title'
        4. Tag is either a heading tag (h1-h6) or bold element / bold style
        """
        if len(text) > 120 or text.endswith(".") or text.endswith(";"):
            return None

        m = re.match(r"^\s*Note\s+([0-9]{1,2})\b[.\s–—\-]*(.*?)$", text, re.IGNORECASE)
        if not m:
            return None

        tag_name = tag.name.lower() if tag.name else ""
        is_heading_tag = tag_name in ("h1", "h2", "h3", "h4", "h5", "h6")
        is_bold = bool(
            tag.find(["b", "strong"])
            or "font-weight:bold" in tag.get("style", "").replace(" ", "").lower()
            or (tag.parent and tag.parent.find(["b", "strong"]))
        )

        if is_heading_tag or is_bold:
            note_num = m.group(1)
            note_title = m.group(2).strip() or f"Note {note_num}"
            return note_num, note_title

        return None

    def extract(
        self,
        html_content: str,
        form: str = "10-K",
        ticker: Optional[str] = None,
        accession_number: Optional[str] = None,
    ) -> ExtractionReport:
        """Parse HTML, isolate TOC, resolve section boundaries, and extract structured content."""
        soup = BeautifulSoup(html_content, "lxml")
        self.sanitize_dom(soup)
        toc_elements, anchor_map = self.identify_toc_elements(soup)

        leaf_blocks = self._extract_leaf_blocks(soup.body if soup.body else soup, toc_elements)

        sections: List[ExtractedSection] = []
        current_part: Optional[str] = None
        current_section: Optional[ExtractedSection] = None

        def finalize_section(sec: Optional[ExtractedSection]) -> None:
            if not sec:
                return
            sec.char_count = len(sec.raw_text)
            if not sec.nodes:
                sec.status = SectionStatus.UNCERTAIN_BOUNDARY
            sections.append(sec)

        for block in leaf_blocks:
            text = self._normalize_text(block.get_text(" ", strip=True))
            if not text:
                continue

            m_part = PART_REGEX.match(text)
            if m_part:
                current_part = f"PART {m_part.group(1).upper()}"

            m_item = ITEM_REGEX.match(text)
            m_exec = EXECUTIVE_OFFICERS_REGEX.search(text)
            note_info = (
                self._is_note_heading(block, text)
                if (current_section and ("ITEM 8" in current_section.item or "ITEM_8_NOTE" in current_section.item))
                else None
            )

            is_new_section = False
            item_key = ""
            title = ""
            is_standard = True

            if m_item:
                raw_num = m_item.group(1).upper()
                item_key = f"ITEM {raw_num}"
                is_new_section = True
                if "10-K" in form.upper():
                    title = CANONICAL_10K_ITEMS.get(item_key, m_item.group(2).strip() or item_key)
                else:
                    if item_key in CANONICAL_10Q_ITEMS["PART II"] and item_key not in CANONICAL_10Q_ITEMS.get(current_part or "PART I", {}):
                        current_part = "PART II"
                    part_map = CANONICAL_10Q_ITEMS.get(current_part or "PART I", {})
                    title = part_map.get(item_key, m_item.group(2).strip() or item_key)
            elif m_exec and (not current_section or current_section.item != "EXECUTIVE_OFFICERS"):
                item_key = "EXECUTIVE_OFFICERS"
                title = "Executive Officers of the Registrant"
                is_standard = False
                is_new_section = True
            elif note_info:
                note_num, note_title = note_info
                item_key = f"ITEM_8_NOTE_{note_num}"
                title = f"Note {note_num} - {note_title}"
                is_standard = False
                is_new_section = True

            if is_new_section:
                finalize_section(current_section)
                anchor_id = anchor_map.get(item_key)
                current_section = ExtractedSection(
                    part=current_part,
                    item=item_key,
                    title=title,
                    status=SectionStatus.EXTRACTED if is_standard else SectionStatus.NON_STANDARD,
                    is_standard=is_standard,
                    start_anchor=anchor_id,
                )
                continue

            if current_section is not None:
                node = self._parse_content_node(block)
                if node:
                    current_section.nodes.append(node)
                    if node.node_type == ContentNodeType.TABLE:
                        current_section.table_count += 1
                    if current_section.raw_text:
                        current_section.raw_text += "\n\n" + node.text
                    else:
                        current_section.raw_text = node.text

        finalize_section(current_section)

        # Identify missing canonical items
        found_items = {s.item for s in sections}
        unidentified: List[str] = []

        if form.upper() == "10-K":
            for exp_item, exp_title in CANONICAL_10K_ITEMS.items():
                if exp_item not in found_items:
                    sections.append(
                        ExtractedSection(
                            part=None,
                            item=exp_item,
                            title=exp_title,
                            status=SectionStatus.NOT_FOUND,
                            is_standard=True,
                            nodes=[],
                            raw_text="",
                        )
                    )
                    unidentified.append(exp_item)

        return ExtractionReport(
            ticker=ticker,
            form=form,
            accession_number=accession_number,
            sections=sections,
            toc_found=len(toc_elements) > 0,
            toc_anchors_count=len(anchor_map),
            unidentified_sections=unidentified,
            diagnostics={
                "total_sections_extracted": len(
                    [s for s in sections if s.status in (SectionStatus.EXTRACTED, SectionStatus.NON_STANDARD)]
                ),
                "total_tables_extracted": sum(s.table_count for s in sections),
                "total_characters": sum(s.char_count for s in sections),
            },
        )
