#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for factor_rag.metrics: the fidelity scores must actually move.

Author: Zeteng Lin (Hong Kong University of Science and Technology, Guangzhou)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factor_rag.chunking import chunk_markdown, naive_chunk
from factor_rag.metrics import score_chunks

ROOT = Path(__file__).resolve().parents[1]


def _big_table(n_rows: int) -> str:
    lines = ["| a | b |", "| --- | --- |"]
    lines += [f"| {i} | value_{i} |" for i in range(n_rows)]
    return "\n".join(lines)


def test_structured_chunking_scores_at_least_as_well_as_naive_on_a_big_table():
    md = "# Report\n\n" + _big_table(150)
    naive = naive_chunk(md, chunk_size=400, chunk_overlap=40)
    structured = [c.text for c in chunk_markdown(md, chunk_size=400, chunk_overlap=40)]

    report_naive = score_chunks(md, naive)
    report_structured = score_chunks(md, structured)

    assert report_structured.table_integrity == 1.0
    assert report_structured.row_header_coverage == 1.0
    assert report_structured.table_integrity >= report_naive.table_integrity
    assert report_structured.row_header_coverage >= report_naive.row_header_coverage


def test_score_chunks_handles_a_document_with_no_tables():
    md = "# Report\n\nJust prose, no tables at all. " * 10
    chunks = [c.text for c in chunk_markdown(md, chunk_size=200, chunk_overlap=20)]
    report = score_chunks(md, chunks)
    assert report.n_tables == 0
    assert report.table_integrity == 1.0  # vacuously true, not zero


def test_sample_corpus_hits_perfect_fidelity():
    for path in sorted((ROOT / "samples").glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        md = path.read_text(encoding="utf-8")
        chunks = [c.text for c in chunk_markdown(md, source=str(path))]
        report = score_chunks(md, chunks)
        assert report.table_integrity == 1.0, path
        assert report.row_header_coverage == 1.0, path
        assert report.figure_caption_adjacency == 1.0, path
