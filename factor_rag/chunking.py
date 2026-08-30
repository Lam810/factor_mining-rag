#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structure-aware chunking for Markdown converted from visually rich documents.

When a PDF full of tables and figures is converted to Markdown by a layout
parser (MinerU, Marker, Nougat, PP-StructureV2, ...), the resulting file is
mostly *structure*: pipe tables, fenced blocks, figure references, and a
heading hierarchy that carries the document's semantics.

A character-window chunker is blind to all of it.  It happily cuts a 40-row
financial table in half, leaving the second half as a wall of ``| 0.31 | 0.08 |``
rows with no header and no section title -- a chunk that is simultaneously
unretrievable (no lexical signal) and unusable (an LLM cannot read a headerless
table).  This module fixes that.

Guarantees provided by :func:`chunk_markdown`:

1. A table is never split across chunks without its header row being repeated.
2. A fenced code block is never split mid-fence.
3. A figure reference is never separated from its caption.
4. Every chunk carries the heading breadcrumb of the section it came from.
5. Chunking always terminates (the legacy character-window splitter did not).
6. Chunk ids are deterministic across processes and runs.

Author: Zeteng Lin (Hong Kong University of Science and Technology, Guangzhou)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

__all__ = [
    "Block",
    "Chunk",
    "parse_blocks",
    "chunk_markdown",
    "naive_chunk",
]

# --------------------------------------------------------------------------- #
# Block model
# --------------------------------------------------------------------------- #

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
FENCE_RE = re.compile(r"^\s*(```+|~~~+)(.*)$")
TABLE_DELIM_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
FIGURE_RE = re.compile(r"^\s*!\[[^\]]*\]\([^)]*\)\s*$")
LIST_RE = re.compile(r"^\s*([-*+]|\d+[.)])\s+")
CAPTION_RE = re.compile(r"^\s*(\*.*\*|_.*_|(Figure|Fig\.|Table|图|表)\s*\d+[.:：]?.*)\s*$")

#: Blocks that lose their meaning if cut at an arbitrary character offset.
ATOMIC_KINDS = frozenset({"table", "code", "figure"})


@dataclass
class Block:
    """A semantically indivisible run of Markdown lines."""

    kind: str
    lines: List[str]
    level: int = 0  # heading depth, 0 for non-headings

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def __len__(self) -> int:  # size in characters, as rendered
        return len(self.text)


@dataclass
class Chunk:
    """A retrievable unit of text plus the structural context it came from."""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def _is_table_start(lines: Sequence[str], i: int) -> bool:
    """A pipe table is a row line immediately followed by a delimiter line."""
    if "|" not in lines[i] or not lines[i].strip():
        return False
    if i + 1 >= len(lines):
        return False
    return bool(TABLE_DELIM_RE.match(lines[i + 1]))


def parse_blocks(markdown: str) -> List[Block]:
    """Split ``markdown`` into a flat list of semantic :class:`Block` objects.

    The parser is deliberately shallow: it recognises exactly the constructs
    that layout-parser output actually contains, and treats everything else as
    prose.  Nested structures (a table inside a list item) are not modelled,
    because layout parsers do not emit them.
    """
    lines = markdown.splitlines()
    blocks: List[Block] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # Blank lines are separators, not content.
        if not line.strip():
            i += 1
            continue

        # Fenced code / formula block ------------------------------------- #
        fence = FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)
            buf = [line]
            i += 1
            while i < n:
                buf.append(lines[i])
                if lines[i].strip().startswith(marker):
                    i += 1
                    break
                i += 1
            blocks.append(Block("code", buf))
            continue

        # Heading ----------------------------------------------------------- #
        heading = HEADING_RE.match(line)
        if heading:
            blocks.append(Block("heading", [line], level=len(heading.group(1))))
            i += 1
            continue

        # Pipe table --------------------------------------------------------- #
        if _is_table_start(lines, i):
            buf = [lines[i], lines[i + 1]]
            i += 2
            while i < n and lines[i].strip() and "|" in lines[i]:
                buf.append(lines[i])
                i += 1
            blocks.append(Block("table", buf))
            continue

        # Figure (image plus its optional caption) --------------------------- #
        if FIGURE_RE.match(line):
            buf = [line]
            j = i + 1
            if j < n and not lines[j].strip():
                j += 1  # a single blank line between image and caption is common style
            if j < n and lines[j].strip() and CAPTION_RE.match(lines[j]):
                buf.extend(lines[i + 1 : j + 1])  # keep any skipped blank line for fidelity
                i = j + 1
            else:
                i += 1
            blocks.append(Block("figure", buf))
            continue

        # List ---------------------------------------------------------------- #
        if LIST_RE.match(line):
            buf = []
            while i < n and lines[i].strip() and not HEADING_RE.match(lines[i]):
                if not LIST_RE.match(lines[i]) and not lines[i].startswith((" ", "\t")):
                    break
                buf.append(lines[i])
                i += 1
            blocks.append(Block("list", buf))
            continue

        # Paragraph ------------------------------------------------------------ #
        buf = []
        while i < n and lines[i].strip():
            if HEADING_RE.match(lines[i]) or FENCE_RE.match(lines[i]):
                break
            if _is_table_start(lines, i) or FIGURE_RE.match(lines[i]):
                break
            buf.append(lines[i])
            i += 1
        if buf:
            blocks.append(Block("paragraph", buf))

    return blocks


# --------------------------------------------------------------------------- #
# Splitting oversized blocks
# --------------------------------------------------------------------------- #

#: Sentence terminators for Latin *and* CJK text.  The legacy splitter looked
#: only for ``". "``, which never matches a Chinese document and silently
#: degraded to blind character slicing.
_SENTENCE_END = re.compile(r"(?<=[。．！？；])|(?<=[.!?;])(?=\s)")


def _split_table(block: Block, budget: int) -> List[Block]:
    """Split an oversized table into parts, repeating the header in each part.

    This is the single most valuable guarantee in this module.  A table row
    without its header is unreadable to a language model and unretrievable by
    an embedding model; repeating two short lines per part removes the failure
    mode entirely.
    """
    lines = block.lines
    header, delim, body = lines[0], lines[1], lines[2:]
    if not body:
        return [block]

    overhead = len(header) + len(delim) + 2
    parts: List[List[str]] = []
    current: List[str] = []

    for row in body:
        projected = overhead + sum(len(r) + 1 for r in current) + len(row) + 1
        if current and projected > budget:
            parts.append([header, delim, *current])
            current = []
        current.append(row)  # always appended: guarantees forward progress
    if current:
        parts.append([header, delim, *current])

    return [Block("table", p) for p in parts]


def _split_code(block: Block, budget: int) -> List[Block]:
    """Split an oversized fenced block, re-opening the fence in each part."""
    lines = block.lines
    opener = lines[0]
    closer = lines[-1] if len(lines) > 1 and FENCE_RE.match(lines[-1]) else "```"
    body = lines[1:-1] if len(lines) > 2 else lines[1:]

    overhead = len(opener) + len(closer) + 2
    parts: List[List[str]] = []
    current: List[str] = []

    for line in body:
        projected = overhead + sum(len(l) + 1 for l in current) + len(line) + 1
        if current and projected > budget:
            parts.append([opener, *current, closer])
            current = []
        current.append(line)
    if current:
        parts.append([opener, *current, closer])

    return [Block("code", p) for p in parts] or [block]


def _sentences(text: str) -> List[str]:
    parts = [s for s in _SENTENCE_END.split(text) if s]
    return parts or [text]


def _split_prose(text: str, budget: int, overlap: int) -> List[str]:
    """Sentence-aware sliding window over prose, CJK included."""
    if len(text) <= budget:
        return [text]

    out: List[str] = []
    current = ""
    for sentence in _sentences(text):
        # A single sentence longer than the budget is hard-sliced; without this
        # the loop could not make progress.
        while len(sentence) > budget:
            if current:
                out.append(current)
                current = ""
            out.append(sentence[:budget])
            sentence = sentence[budget - overlap:] if overlap < budget else sentence[budget:]
        if current and len(current) + len(sentence) > budget:
            out.append(current)
            tail = current[-overlap:] if overlap else ""
            current = tail + sentence
        else:
            current += sentence
    if current:
        out.append(current)
    return out


# --------------------------------------------------------------------------- #
# The chunker
# --------------------------------------------------------------------------- #


def _breadcrumb(stack: Sequence[str]) -> str:
    return " > ".join(stack)


def _context_line(stack: Sequence[str], note: str = "") -> str:
    if not stack:
        return f"Section: (document root){note}"
    return f"Section: {_breadcrumb(stack)}{note}"


def chunk_markdown(
    markdown: str,
    *,
    source: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    add_heading_context: bool = True,
    section_level: int = 2,
) -> List[Chunk]:
    """Chunk ``markdown`` without destroying its structure.

    Args:
        markdown: Document text, typically the output of a PDF layout parser.
        source: Value recorded as ``metadata["source"]``.
        chunk_size: Target chunk size in characters, including the heading
            breadcrumb prefix.
        chunk_overlap: Characters of prose carried between consecutive chunks.
            Overlap is never applied across a table or code boundary, because
            duplicating table rows adds retrieval noise rather than context.
        add_heading_context: Prefix each chunk with its heading breadcrumb.
        section_level: Start a new chunk whenever a heading at this level or
            shallower is reached.  Set to ``0`` to pack sections together.

    Returns:
        A list of :class:`Chunk`.  Metadata values are all scalars, so the list
        can be handed straight to Chroma / FAISS / Qdrant without flattening.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    blocks = parse_blocks(markdown)
    chunks: List[Chunk] = []

    stack: List[str] = []          # active heading breadcrumb
    pending: List[Block] = []      # blocks accumulated for the current chunk
    carry: str = ""                # prose overlap seeded from the previous chunk
    table_part = 0                 # position within a split table
    table_parts = 0

    def budget_for(path: Sequence[str]) -> int:
        if not add_heading_context:
            return chunk_size
        return max(chunk_size - len(_context_line(path)) - 2, chunk_size // 4)

    def flush(path: Sequence[str]) -> None:
        nonlocal pending, carry, table_part, table_parts
        if not pending:
            return
        body = "\n\n".join(b.text for b in pending)
        if carry:
            body = f"{carry}\n\n{body}"
        note = f" (table part {table_part} of {table_parts})" if table_parts > 1 else ""
        text = f"{_context_line(path, note)}\n\n{body}" if add_heading_context else body

        kinds = sorted({b.kind for b in pending})
        uid = hashlib.sha1(
            f"{source}|{_breadcrumb(path)}|{len(chunks)}|{body}".encode("utf-8")
        ).hexdigest()[:16]

        chunks.append(
            Chunk(
                text=text,
                metadata={
                    "source": str(source) if source is not None else "",
                    "chunk_id": len(chunks),
                    "chunk_uid": uid,
                    "heading_path": _breadcrumb(path),
                    "section": path[-1] if path else "",
                    "block_kinds": ",".join(kinds),
                    "has_table": "table" in kinds,
                    "has_code": "code" in kinds,
                    "has_figure": "figure" in kinds,
                    "table_part": table_part,
                    "table_parts": table_parts,
                    "char_len": len(text),
                },
            )
        )

        # Overlap is prose-only: carrying table rows forward is pure noise.
        last = pending[-1]
        if chunk_overlap and last.kind in ("paragraph", "list"):
            carry = last.text[-chunk_overlap:].lstrip()
        else:
            carry = ""
        pending = []
        table_part = table_parts = 0

    def size_of(blocks_: Iterable[Block]) -> int:
        parts = [b.text for b in blocks_]
        return sum(len(p) for p in parts) + 2 * max(len(parts) - 1, 0) + len(carry)

    for block in blocks:
        if block.kind == "heading":
            title = HEADING_RE.match(block.lines[0]).group(2).strip()
            if section_level and block.level <= section_level:
                flush(stack)
                carry = ""
            stack = stack[: block.level - 1]
            while len(stack) < block.level - 1:
                stack.append("")
            stack.append(title)
            continue

        budget = budget_for(stack)

        # Oversized blocks are split by a structure-aware splitter first.
        if len(block) > budget:
            flush(stack)
            if block.kind == "table":
                pieces = _split_table(block, budget)
            elif block.kind == "code":
                pieces = _split_code(block, budget)
            elif block.kind == "figure":
                pieces = [block]  # never split; oversized figures stay whole
            else:
                pieces = [Block(block.kind, p.split("\n")) for p in _split_prose(block.text, budget, chunk_overlap)]

            if block.kind == "table" and len(pieces) > 1:
                table_parts = len(pieces)
                for idx, piece in enumerate(pieces, start=1):
                    table_part = idx
                    pending = [piece]
                    saved = table_parts
                    flush(stack)
                    table_parts = saved
                table_parts = table_part = 0
            else:
                for piece in pieces:
                    pending = [piece]
                    flush(stack)
            carry = ""
            continue

        if pending and size_of([*pending, block]) > budget:
            flush(stack)

        pending.append(block)

    flush(stack)
    return chunks


# --------------------------------------------------------------------------- #
# Baseline, for benchmarking and visualisation
# --------------------------------------------------------------------------- #


def naive_chunk(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """The character-window baseline this project used to ship.

    Reproduced here (with the non-termination bug repaired) so that the cost of
    structure-blind chunking can be *measured* rather than asserted.  See
    ``benchmarks/run_benchmark.py``.
    """
    if len(text) < chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            para = text.find("\n\n", max(end - chunk_size // 2, start + 1), end + chunk_size // 2)
            sent = text.find(". ", max(end - chunk_size // 4, start + 1), end + chunk_size // 4)
            if para != -1:
                end = para + 2
            elif sent != -1:
                end = sent + 2
        chunks.append(text[start:end])
        nxt = end - chunk_overlap
        start = nxt if nxt > start else end  # the original could move backwards
    return chunks
