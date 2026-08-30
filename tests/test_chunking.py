#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for factor_rag.chunking.

These exercise exactly the failure modes the legacy character-window
splitter had: tables split without their header, unterminated splitting
loops, and CJK sentence boundaries not being recognised. See
factor_rag/chunking.py's module docstring for the full rationale.

Author: Zeteng Lin (Hong Kong University of Science and Technology, Guangzhou)
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factor_rag.chunking import chunk_markdown, naive_chunk, parse_blocks

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def _big_table(n_rows: int) -> str:
    lines = ["| a | b | c |", "| --- | --- | --- |"]
    lines += [f"| {i} | value_{i} | {i * 2} |" for i in range(n_rows)]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# parse_blocks
# --------------------------------------------------------------------------- #


def test_parse_blocks_identifies_table():
    blocks = parse_blocks(_big_table(5))
    assert len(blocks) == 1
    assert blocks[0].kind == "table"
    assert len(blocks[0].lines) == 7  # header + delimiter + 5 rows


def test_parse_blocks_pairs_figure_with_caption():
    md = "![](fig.png)\n*Figure 1. A caption.*"
    blocks = parse_blocks(md)
    assert len(blocks) == 1
    assert blocks[0].kind == "figure"
    assert len(blocks[0].lines) == 2


def test_parse_blocks_does_not_treat_prose_pipes_as_a_table():
    md = "The ratio a|b is not a table, just prose with a pipe in it."
    blocks = parse_blocks(md)
    assert all(b.kind != "table" for b in blocks)


def test_parse_blocks_keeps_fence_intact():
    md = "```python\nx = 1\ny = 2\n```"
    blocks = parse_blocks(md)
    assert len(blocks) == 1
    assert blocks[0].kind == "code"
    assert blocks[0].lines[0].startswith("```") and blocks[0].lines[-1].startswith("```")


# --------------------------------------------------------------------------- #
# chunk_markdown -- the core guarantees
# --------------------------------------------------------------------------- #


def test_oversized_table_keeps_header_in_every_part():
    md = "# Report\n\n" + _big_table(200)
    chunks = chunk_markdown(md, chunk_size=800, chunk_overlap=100)
    table_chunks = [c for c in chunks if c.metadata["has_table"]]
    assert len(table_chunks) > 1, "200 rows at chunk_size=800 must split"
    for c in table_chunks:
        assert "| a | b | c |" in c.text
        assert "| --- | --- | --- |" in c.text


def test_every_table_row_appears_with_its_header():
    md = "# Report\n\n" + _big_table(150)
    chunks = [c.text for c in chunk_markdown(md, chunk_size=600, chunk_overlap=50)]
    header = "| a | b | c |"
    header_chunks = {i for i, c in enumerate(chunks) if header in c}
    for i in range(150):
        row = f"| {i} | value_{i} | {i * 2} |"
        holders = {j for j, c in enumerate(chunks) if row in c}
        assert holders & header_chunks, f"row {i} lost its header"


def test_heading_breadcrumb_present_in_every_chunk():
    md = "# Top\n\n## Sub\n\n" + "prose. " * 200
    chunks = chunk_markdown(md, chunk_size=300, chunk_overlap=30)
    assert all(c.metadata["heading_path"] for c in chunks[1:])  # first chunk may be the bare title


def test_cjk_sentences_are_split_on_cjk_punctuation():
    # The legacy splitter looked only for ". " and would hard-slice this
    # instead of respecting the (CJK) sentence boundaries.
    text = "价值因子在市场中长期有效。" * 60
    md = f"# 研究\n\n{text}"
    chunks = chunk_markdown(md, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        body = c.text.split("\n\n", 1)[-1]
        # every chunk boundary should land on a sentence terminator, not mid-sentence
        assert body.rstrip().endswith("。") or body == chunks[-1].text.split("\n\n", 1)[-1]


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_markdown("# x\n\nhello", chunk_size=100, chunk_overlap=100)


def test_deterministic_chunk_ids_across_runs():
    md = "# Report\n\n" + _big_table(30)
    first = [c.metadata["chunk_uid"] for c in chunk_markdown(md, source="a.md")]
    second = [c.metadata["chunk_uid"] for c in chunk_markdown(md, source="a.md")]
    assert first == second


def test_empty_and_whitespace_only_input():
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n\n   ") == []


# --------------------------------------------------------------------------- #
# naive_chunk -- the repaired baseline used for the benchmark comparison
# --------------------------------------------------------------------------- #


def test_naive_chunk_terminates_when_overlap_is_close_to_chunk_size():
    # The original implementation could move `start` backwards forever here.
    text = "x" * 5000
    start = time.time()
    chunks = naive_chunk(text, chunk_size=100, chunk_overlap=99)
    assert time.time() - start < 5.0
    assert "".join(chunks).replace("x", "") == ""  # sanity: still just x's


def test_naive_chunk_splits_a_table_without_repeating_its_header():
    # This is the behaviour structure-aware chunking exists to fix -- pin it
    # down on the baseline so a regression there doesn't silently make the
    # two strategies look identical.
    md = _big_table(200)
    chunks = naive_chunk(md, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    without_header = [c for c in chunks[1:] if "| a | b | c |" not in c]
    assert without_header, "expected the naive baseline to orphan at least one part"


# --------------------------------------------------------------------------- #
# End-to-end over the bundled sample corpus
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", sorted(SAMPLES.glob("*.md")))
def test_sample_corpus_round_trips(path: Path):
    if path.name.lower() == "readme.md":
        pytest.skip("not a layout-parser sample")
    md = path.read_text(encoding="utf-8")
    chunks = chunk_markdown(md, source=str(path))
    assert chunks, f"{path} produced no chunks"
    assert all(c.text.strip() for c in chunks)
