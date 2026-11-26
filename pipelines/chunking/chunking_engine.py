"""Structure-Aware Chunking Engine for SEC EDGAR Filings.

Implements semantic structure-first chunking respecting Item boundaries,
subsections, paragraphs, lists, and tables. Preserves rich provenance metadata.
"""

import hashlib
import logging
import re

import tiktoken
from pydantic import BaseModel

from pipelines.processing.section_extractor import (
    ContentNodeType,
    ExtractedSection,
)
from pipelines.processing.table_processor import TableProcessor

logger = logging.getLogger(__name__)


class TextChunk(BaseModel):
    """Normalized chunk data contract with full 18-field SEC EDGAR provenance metadata."""

    chunk_id: str
    cik: str
    ticker: str
    accession_number: str
    form: str
    fiscal_year: int
    filed_date: str
    period_end: str
    part: str | None = None
    item: str
    subsection: str | None = None
    content_type: str  # text, table, list
    chunk_index: int
    token_count: int
    text: str
    source_url: str
    source_content_hash: str


class StructureAwareChunkingEngine:
    """Chunks structured filing sections into token-bounded retrieval units."""

    def __init__(
        self,
        target_tokens: int = 512,
        overlap_tokens: int = 64,
        encoding_name: str = "cl100k_base",
    ) -> None:
        """Initialize chunking engine with target tokens and overlap."""
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.table_processor = TableProcessor(target_tokens=target_tokens, encoding_name=encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens using tokenizer."""
        return len(self.tokenizer.encode(text))

    def _split_long_text(self, text: str) -> list[str]:
        """Split a long text block exceeding target tokens at sentence boundaries with overlap."""
        sentences = re.split(r"(?<=[.?!;])\s+", text)
        chunks: list[str] = []
        current_sentences: list[str] = []
        current_len = 0

        for sent in sentences:
            sent_len = self.count_tokens(sent)
            if current_len + sent_len > self.target_tokens and current_sentences:
                chunks.append(" ".join(current_sentences))
                overlap: list[str] = []
                overlap_len = 0
                for s in reversed(current_sentences):
                    sl = self.count_tokens(s)
                    if overlap_len + sl <= self.overlap_tokens:
                        overlap.insert(0, s)
                        overlap_len += sl
                    else:
                        break
                current_sentences = overlap + [sent]
                current_len = overlap_len + sent_len
            else:
                current_sentences.append(sent)
                current_len += sent_len

        if current_sentences:
            chunks.append(" ".join(current_sentences))
        return chunks

    def chunk_section(
        self,
        section: ExtractedSection,
        cik: str,
        ticker: str,
        accession_number: str,
        form: str,
        fiscal_year: int,
        filed_date: str,
        period_end: str,
        source_url: str,
        start_chunk_index: int = 0,
    ) -> list[TextChunk]:
        """Chunk an extracted section into TextChunks adhering to Item and subsection boundaries."""
        chunks: list[TextChunk] = []
        current_subsection: str | None = None
        text_buffer: list[str] = []
        buffer_tokens = 0
        chunk_idx = start_chunk_index

        def flush_buffer(c_type: str = "text") -> None:
            nonlocal text_buffer, buffer_tokens, chunk_idx
            if not text_buffer:
                return
            combined_text = "\n\n".join(text_buffer).strip()
            if not combined_text:
                text_buffer = []
                buffer_tokens = 0
                return

            t_count = self.count_tokens(combined_text)
            content_hash = hashlib.sha256(combined_text.encode("utf-8")).hexdigest()
            chunk_id_str = (
                f"{ticker}/{accession_number}/{section.part or ''}/"
                f"{section.item}/{current_subsection or 'main'}/{chunk_idx}"
            )
            chunk_id = hashlib.sha256(chunk_id_str.encode("utf-8")).hexdigest()

            chunk = TextChunk(
                chunk_id=chunk_id,
                cik=cik,
                ticker=ticker,
                accession_number=accession_number,
                form=form,
                fiscal_year=fiscal_year,
                filed_date=filed_date,
                period_end=period_end,
                part=section.part,
                item=section.item,
                subsection=current_subsection,
                content_type=c_type,
                chunk_index=chunk_idx,
                token_count=t_count,
                text=combined_text,
                source_url=source_url,
                source_content_hash=content_hash,
            )
            chunks.append(chunk)
            chunk_idx += 1
            text_buffer = []
            buffer_tokens = 0

        for node in section.nodes:
            if node.node_type == ContentNodeType.HEADING:
                # Flush existing text when encountering a new subsection heading
                flush_buffer()
                current_subsection = node.text
                continue

            elif node.node_type == ContentNodeType.TABLE:
                # Flush text buffer before table
                flush_buffer()
                # Process table into row-grouped chunks with repeated headers
                table_chunks = self.table_processor.process_table(node, max_tokens=self.target_tokens)
                for t_text, t_count in table_chunks:
                    content_hash = hashlib.sha256(t_text.encode("utf-8")).hexdigest()
                    chunk_id_str = (
                        f"{ticker}/{accession_number}/{section.part or ''}/"
                        f"{section.item}/{current_subsection or 'main'}/{chunk_idx}"
                    )
                    chunk_id = hashlib.sha256(chunk_id_str.encode("utf-8")).hexdigest()
                    chunk = TextChunk(
                        chunk_id=chunk_id,
                        cik=cik,
                        ticker=ticker,
                        accession_number=accession_number,
                        form=form,
                        fiscal_year=fiscal_year,
                        filed_date=filed_date,
                        period_end=period_end,
                        part=section.part,
                        item=section.item,
                        subsection=current_subsection,
                        content_type="table",
                        chunk_index=chunk_idx,
                        token_count=t_count,
                        text=t_text,
                        source_url=source_url,
                        source_content_hash=content_hash,
                    )
                    chunks.append(chunk)
                    chunk_idx += 1
                continue

            # Paragraph or list nodes
            node_tokens = self.count_tokens(node.text)
            if node_tokens > self.target_tokens:
                flush_buffer()
                sub_chunks = self._split_long_text(node.text)
                for sc in sub_chunks:
                    text_buffer.append(sc)
                    flush_buffer(c_type="list" if node.node_type == ContentNodeType.LIST else "text")
            elif buffer_tokens + node_tokens > self.target_tokens:
                flush_buffer()
                text_buffer.append(node.text)
                buffer_tokens = node_tokens
            else:
                text_buffer.append(node.text)
                buffer_tokens += node_tokens

        flush_buffer()
        return chunks
