#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structural-fidelity metrics for a chunking strategy.

Claims about chunking quality are usually asserted, not measured.  These
metrics measure them, and they are deliberately computed from *chunk text
alone* -- no privileged access to a chunker's internal metadata -- so that any
strategy can be scored on equal terms.

Author: Zeteng Lin (Hong Kong University of Science and Technology, Guangzhou)
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

from .chunking import FENCE_RE, FIGURE_RE, HEADING_RE, Block, parse_blocks

__all__ = ["FidelityReport", "score_chunks"]

_SECTION_PREFIX_RE = re.compile(r"^Section:\s*\S", re.MULTILINE)
_HEADING_ANY_RE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)


@dataclass
class FidelityReport:
    """How much document structure survived chunking."""

    n_chunks: int
    mean_chunk_chars: float
    #: Fraction of source tables whose every body row sits in a chunk that also
    #: carries that table's header row.  A table that fails this is unreadable.
    table_integrity: float
    #: Same idea at row granularity -- less brittle on documents with one huge table.
    row_header_coverage: float
    #: Fraction of chunks that can name the section they came from.
    section_attribution: float
    #: Fraction of fenced blocks that are not cut mid-fence.
    code_integrity: float
    #: Fraction of figures still adjacent to their caption.
    figure_caption_adjacency: float
    n_tables: int
    n_table_rows: int
    n_code_blocks: int
    n_figures: int

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def _tables(blocks: Sequence[Block]) -> List[Block]:
    return [b for b in blocks if b.kind == "table" and len(b.lines) > 2]


def _chunks_containing(needle: str, chunks: Sequence[str]) -> List[int]:
    needle = needle.strip()
    if not needle:
        return []
    return [i for i, c in enumerate(chunks) if needle in c]


def score_chunks(markdown: str, chunks: Sequence[str]) -> FidelityReport:
    """Score ``chunks`` against the structure present in the source ``markdown``."""
    blocks = parse_blocks(markdown)
    tables = _tables(blocks)
    figures = [b for b in blocks if b.kind == "figure" and len(b.lines) > 1]
    codes = [b for b in blocks if b.kind == "code"]

    # -- tables ------------------------------------------------------------- #
    intact_tables = 0
    total_rows = 0
    covered_rows = 0
    for table in tables:
        header = table.lines[0].strip()
        rows = [r for r in table.lines[2:] if r.strip()]
        total_rows += len(rows)
        header_chunks = set(_chunks_containing(header, chunks))
        rows_ok = 0
        for row in rows:
            if set(_chunks_containing(row, chunks)) & header_chunks:
                rows_ok += 1
        covered_rows += rows_ok
        if rows and rows_ok == len(rows):
            intact_tables += 1

    # -- sections ----------------------------------------------------------- #
    attributed = sum(
        1
        for c in chunks
        if _SECTION_PREFIX_RE.search(c) or _HEADING_ANY_RE.search(c)
    )

    # -- code fences -------------------------------------------------------- #
    balanced = sum(
        1 for c in chunks if len([l for l in c.splitlines() if FENCE_RE.match(l)]) % 2 == 0
    )
    code_integrity = balanced / len(chunks) if chunks else 1.0

    # -- figures ------------------------------------------------------------ #
    adjacent = 0
    for fig in figures:
        image, caption = fig.lines[0].strip(), fig.lines[-1].strip()
        if set(_chunks_containing(image, chunks)) & set(_chunks_containing(caption, chunks)):
            adjacent += 1

    n = len(chunks) or 1
    return FidelityReport(
        n_chunks=len(chunks),
        mean_chunk_chars=sum(len(c) for c in chunks) / n,
        table_integrity=intact_tables / len(tables) if tables else 1.0,
        row_header_coverage=covered_rows / total_rows if total_rows else 1.0,
        section_attribution=attributed / n,
        code_integrity=code_integrity,
        figure_caption_adjacency=adjacent / len(figures) if figures else 1.0,
        n_tables=len(tables),
        n_table_rows=total_rows,
        n_code_blocks=len(codes),
        n_figures=len(figures),
    )
