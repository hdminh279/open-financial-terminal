"""Table-Aware Processing Module for SEC EDGAR HTML Filings.

Converts structured HTML tables into standardized Markdown representations,
preserves headers, captions, and cell alignments, and handles splitting long tables
into row groups with repeated headers to ensure semantic context in RAG.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import tiktoken
from src.data.processing.section_extractor import ContentNodeType, ExtractedContentNode

logger = logging.getLogger(__name__)


class TableProcessor:
    """Processes extracted HTML tables into structured Markdown tables with header repetition."""

    def __init__(self, target_tokens: int = 512, encoding_name: str = "cl100k_base") -> None:
        """Initialize TableProcessor with target token window and tokenizer."""
        self.target_tokens = target_tokens
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens using cl100k_base tokenizer."""
        return len(self.tokenizer.encode(text))

    def format_markdown_table(
        self,
        headers: List[str],
        rows: List[List[str]],
        caption: str = "",
        part_info: str = "",
    ) -> str:
        """Format table data into a clean Markdown table with optional caption and part tag."""
        lines: List[str] = []
        title = caption or "Financial Table"
        if part_info:
            title += f" ({part_info})"
        lines.append(f"[{title}]")

        clean_headers = [h.replace("|", "/").strip() for h in headers]
        if not clean_headers and rows:
            clean_headers = [f"Col_{i+1}" for i in range(len(rows[0]))]

        header_line = "| " + " | ".join(clean_headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(clean_headers)) + " |"
        lines.append(header_line)
        lines.append(sep_line)

        for row in rows:
            clean_row = [c.replace("|", "/").strip() for c in row]
            if len(clean_row) < len(clean_headers):
                clean_row.extend([""] * (len(clean_headers) - len(clean_row)))
            elif len(clean_row) > len(clean_headers):
                clean_row = clean_row[: len(clean_headers)]
            lines.append("| " + " | ".join(clean_row) + " |")

        return "\n".join(lines)

    def process_table(
        self,
        node: ExtractedContentNode,
        max_tokens: Optional[int] = None,
    ) -> List[Tuple[str, int]]:
        """Process a table node into one or more chunk strings, splitting long tables with repeated headers.

        Returns:
            List of tuples: [(formatted_markdown_text, token_count), ...]
        """
        budget = max_tokens or self.target_tokens
        metadata = node.metadata or {}
        headers: List[str] = metadata.get("headers", [])
        rows: List[List[str]] = metadata.get("rows", [])
        caption: str = metadata.get("caption", "")

        # Fallback if headers were inside first tr
        if not headers and rows:
            headers = rows[0]
            rows = rows[1:]

        if not rows:
            text = self.format_markdown_table(headers, [], caption=caption)
            return [(text, self.count_tokens(text))]

        # Check full table tokens
        full_text = self.format_markdown_table(headers, rows, caption=caption)
        full_tokens = self.count_tokens(full_text)
        if full_tokens <= budget:
            return [(full_text, full_tokens)]

        # Long table: Split into row groups and repeat headers on every part
        chunks: List[Tuple[str, int]] = []
        current_rows: List[List[str]] = []
        row_idx = 0
        total_rows = len(rows)

        while row_idx < total_rows:
            current_rows = []
            while row_idx < total_rows:
                test_rows = current_rows + [rows[row_idx]]
                test_text = self.format_markdown_table(headers, test_rows, caption=caption)
                if self.count_tokens(test_text) > budget and current_rows:
                    break
                current_rows.append(rows[row_idx])
                row_idx += 1

            part_str = f"Part {len(chunks) + 1}"
            chunk_text = self.format_markdown_table(
                headers, current_rows, caption=caption, part_info=part_str
            )
            chunks.append((chunk_text, self.count_tokens(chunk_text)))

        # Update part labels with total count: e.g., (Part 1/3)
        total_parts = len(chunks)
        final_chunks: List[Tuple[str, int]] = []
        for i, (txt, _) in enumerate(chunks):
            txt_updated = txt.replace(f"Part {i+1}", f"Part {i+1}/{total_parts}")
            final_chunks.append((txt_updated, self.count_tokens(txt_updated)))

        return final_chunks
