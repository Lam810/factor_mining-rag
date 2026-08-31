#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Invariants the chunker must never break.

Author: Zeteng Lin (Hong Kong University of Science and Technology, Guangzhou)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factor_rag.chunking import chunk_markdown, naive_chunk, parse_blocks  # noqa: E402
from factor_rag.metrics import score_chunks  # noqa: E402

SAMPLES = sorted((Path(__file__).resolve().parents[1] / "samples").glob("*.md"))
CORPUS = [p for p in SAMPLES if p.name.lower() != "readme.md"]

BIG_TABLE = "\n".join(
    ["# Report", "", "## Numbers", "", "| Name | Value | Note |", "| --- | --- | --- |"]
    + [f"| row {i} | {i * 3.7:.2f} | filler text to add width {i} |" for i in range(80)]
)


def test_parses_the_constructs_a_layout_parser_emits():
    kinds = {b.kind for b in parse_blocks(BIG_TABLE)}
    assert "table" in kinds and "heading" in kinds


def test_table_rows_always_travel_with_their_header():
    chunks = [c.text for c in chunk_markdown(BIG_TABLE, chunk_size=600)]
    assert len(chunks) > 1, "this fixture is meant to force a split"
    header = "| Name | Value | Note |"
    for chunk in chunks:
        if "| row " in chunk:
            assert header in chunk


def test_baseline_really_does_break_the_table():
    # The guarantee is only interesting if the naive approach fails here.
    report = score_chunks(BIG_TABLE, naive_chunk(BIG_TABLE, 600, 100))
    assert report.row_header_coverage < 1.0


def test_fenced_blocks_stay_balanced():
    md = "# T\n\n## S\n\n```\n" + "\n".join(f"line {i}" for i in range(400)) + "\n```\n"
    for chunk in (c.text for c in chunk_markdown(md, chunk_size=500)):
        assert chunk.count("```") % 2 == 0


def test_figure_keeps_its_caption():
    md = "# T\n\n## S\n\n![](figures/f1.png)\n\n*Figure 1. A caption.*\n"
    chunks = [c.text for c in chunk_markdown(md, chunk_size=300, chunk_overlap=60)]
    holder = [c for c in chunks if "figures/f1.png" in c]
    assert holder and "Figure 1." in holder[0]


def test_every_chunk_knows_its_section():
    md = "# Doc\n\n## Alpha\n\nText a.\n\n## Beta\n\nText b.\n"
    for chunk in chunk_markdown(md, chunk_size=200, chunk_overlap=40):
        assert chunk.metadata["heading_path"]


def test_metadata_is_flat_enough_for_a_vector_store():
    for chunk in chunk_markdown(BIG_TABLE, chunk_size=600):
        for key, value in chunk.metadata.items():
            assert isinstance(value, (str, int, float, bool)), f"{key} is {type(value)}"


def test_chunk_ids_are_stable_across_calls():
    first = [c.metadata["chunk_uid"] for c in chunk_markdown(BIG_TABLE, source="a.md")]
    second = [c.metadata["chunk_uid"] for c in chunk_markdown(BIG_TABLE, source="a.md")]
    assert first == second


@pytest.mark.parametrize("size,overlap", [(200, 50), (400, 100), (1000, 200), (2000, 400)])
def test_terminates_and_covers_the_corpus(size, overlap):
    for path in CORPUS:
        md = path.read_text(encoding="utf-8")
        chunks = chunk_markdown(md, source=str(path), chunk_size=size, chunk_overlap=overlap)
        assert chunks
        joined = "\n".join(c.text for c in chunks)
        for line in md.splitlines():
            if line.strip().startswith("|") and "---" not in line:
                assert line.strip() in joined, f"lost table row in {path.name}: {line[:40]}"


def test_overlap_must_be_smaller_than_the_chunk():
    with pytest.raises(ValueError):
        chunk_markdown("# T\n\ntext\n", chunk_size=100, chunk_overlap=100)


def test_structure_aware_beats_the_baseline_on_the_corpus():
    for path in CORPUS:
        md = path.read_text(encoding="utf-8")
        base = score_chunks(md, naive_chunk(md, 1000, 200))
        smart = score_chunks(md, [c.text for c in chunk_markdown(md, chunk_size=1000)])
        assert smart.row_header_coverage >= base.row_header_coverage
        assert smart.section_attribution >= base.section_attribution
