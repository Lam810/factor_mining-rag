#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dependency-free SVG rendering of what a chunker does to a document.

Two figures are produced, each in a light and a dark variant:

``structure-map``
    The same document rendered twice at an identical vertical scale -- once per
    chunking strategy -- so chunk boundaries can be compared line by line.
    Table rows that end up in a chunk without their header are drawn in the
    critical status colour and counted.

``fidelity-bars``
    Grouped bars over the structural-fidelity metrics in :mod:`factor_rag.metrics`.

No plotting library is involved; the output is plain SVG text, which keeps the
package installable with zero heavy dependencies and keeps the committed figures
diffable in review.

Author: Zeteng Lin (Hong Kong University of Science and Technology, Guangzhou)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .chunking import HEADING_RE, Block, parse_blocks

__all__ = ["Theme", "LIGHT", "DARK", "structure_map_svg", "fidelity_bars_svg"]

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


@dataclass(frozen=True)
class Theme:
    """Colour slots. Values come from a palette validated for CVD separation."""

    name: str
    surface: str
    text_primary: str
    text_secondary: str
    text_muted: str
    grid: str
    series_1: str
    series_2: str
    critical: str
    substrate: str


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    text_muted="#7c7b75",
    grid="#e4e3df",
    series_1="#2a78d6",
    series_2="#eb6834",
    critical="#d03b3b",
    substrate="#c9c8c2",
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    text_muted="#8f8e85",
    grid="#383835",
    series_1="#3987e5",
    series_2="#d95926",
    critical="#d03b3b",
    substrate="#54534e",
)


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _text(x, y, s, *, fill, size=12, weight=400, anchor="start", family=FONT, opacity=1.0):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-family="{family}" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'opacity="{opacity:g}">{_esc(s)}</text>'
    )


def _rect(x, y, w, h, fill, *, rx=0.0, opacity=1.0):
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0.5):.1f}" height="{max(h, 0.5):.1f}" '
        f'rx="{rx:g}" fill="{fill}" opacity="{opacity:g}"/>'
    )


@dataclass
class SourceLine:
    index: int
    text: str
    kind: str
    table_id: int = -1


def classify_lines(markdown: str) -> List[SourceLine]:
    """Label every non-blank source line by the structure it belongs to."""
    blocks = parse_blocks(markdown)
    lines = markdown.splitlines()
    kinds: Dict[int, Tuple[str, int]] = {}

    cursor = 0
    table_id = 0
    for block in blocks:
        first = block.lines[0]
        while cursor < len(lines) and lines[cursor] != first:
            cursor += 1
        for offset in range(len(block.lines)):
            idx = cursor + offset
            if block.kind == "heading":
                kinds[idx] = ("heading", -1)
            elif block.kind == "table":
                kinds[idx] = (("table_header" if offset < 2 else "table_row"), table_id)
            else:
                kinds[idx] = ("other", -1)
        if block.kind == "table":
            table_id += 1
        cursor += len(block.lines)

    out = []
    for i, raw in enumerate(lines):
        if not raw.strip():
            continue
        kind, tid = kinds.get(i, ("other", -1))
        out.append(SourceLine(index=i, text=raw, kind=kind, table_id=tid))
    return out


def _orphan_rows(source_lines: Sequence[SourceLine], chunks: Sequence[str]) -> Dict[int, bool]:
    """True for every table row that never co-occurs with its own header."""
    headers: Dict[int, List[str]] = {}
    for sl in source_lines:
        if sl.kind == "table_header":
            headers.setdefault(sl.table_id, []).append(sl.text.strip())

    header_chunks: Dict[int, set] = {}
    for tid, hdr_lines in headers.items():
        head = hdr_lines[0] if hdr_lines else ""
        header_chunks[tid] = {i for i, c in enumerate(chunks) if head and head in c}

    orphan: Dict[int, bool] = {}
    for sl in source_lines:
        if sl.kind != "table_row":
            continue
        holders = {i for i, c in enumerate(chunks) if sl.text.strip() in c}
        orphan[sl.index] = not (holders & header_chunks.get(sl.table_id, set()))
    return orphan


# --------------------------------------------------------------------------- #
# Figure 1 -- structure map
# --------------------------------------------------------------------------- #

_LINE_H = 4.0
_LINE_PITCH = 5.0
_SPINE_W = 6.0
_MAX_LINE_CHARS = 110.0


def _panel(
    x0: float,
    y0: float,
    width: float,
    title: str,
    subtitle: str,
    accent: str,
    source_lines: Sequence[SourceLine],
    chunks: Sequence[str],
    theme: Theme,
) -> Tuple[str, int]:
    """Render one strategy's column. Returns (svg, orphan_row_count)."""
    parts: List[str] = []
    parts.append(_text(x0, y0, title, fill=theme.text_primary, size=13, weight=600))
    parts.append(_text(x0, y0 + 17, subtitle, fill=theme.text_secondary, size=11.5))

    body_top = y0 + 34
    orphan = _orphan_rows(source_lines, chunks)

    # Which chunk each source line landed in.
    owner: List[int] = []
    for sl in source_lines:
        probe = sl.text.strip()
        found = -1
        for i, c in enumerate(chunks):
            if probe and probe in c:
                found = i
                break
        owner.append(found)

    bar_x = x0 + _SPINE_W + 10
    bar_max = width - _SPINE_W - 10

    # Chunk spine: one rounded capsule per chunk, 2px gap between capsules.
    run_start = 0
    for i in range(1, len(owner) + 1):
        if i == len(owner) or owner[i] != owner[run_start]:
            top = body_top + run_start * _LINE_PITCH
            height = (i - run_start) * _LINE_PITCH - 2
            shade = 0.55 if (owner[run_start] % 2 == 0) else 0.3
            parts.append(
                _rect(x0, top, _SPINE_W, height, theme.text_muted, rx=3, opacity=shade)
            )
            run_start = i

    # Content bars.
    for row, sl in enumerate(source_lines):
        y = body_top + row * _LINE_PITCH
        w = bar_max * min(len(sl.text.rstrip()), _MAX_LINE_CHARS) / _MAX_LINE_CHARS
        if sl.kind == "heading":
            fill, op, bw = theme.text_primary, 0.8, max(w * 0.55, 24)
        elif sl.kind == "table_header":
            fill, op, bw = accent, 1.0, w
        elif sl.kind == "table_row":
            if orphan.get(sl.index):
                fill, op, bw = theme.critical, 1.0, w
            else:
                fill, op, bw = accent, 0.5, w
        else:
            fill, op, bw = theme.substrate, 1.0, w
        parts.append(_rect(bar_x, y, bw, _LINE_H, fill, rx=1.5, opacity=op))

    # Bracket the longest run of orphaned rows and label it.
    best_len = best_start = 0
    cur_len = cur_start = 0
    for row, sl in enumerate(source_lines):
        if orphan.get(sl.index):
            if cur_len == 0:
                cur_start = row
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0

    if best_len >= 2:
        top = body_top + best_start * _LINE_PITCH - 1
        height = best_len * _LINE_PITCH
        bx = x0 + width + 6
        parts.append(
            f'<path d="M{bx:.1f} {top:.1f} h5 v{height:.1f} h-5" fill="none" '
            f'stroke="{theme.critical}" stroke-width="2" stroke-linejoin="round"/>'
        )
        parts.append(
            _text(
                bx + 9,
                top + height / 2 + 4,
                f"{best_len} rows, no header",
                fill=theme.critical,
                size=11,
                weight=600,
            )
        )

    return "\n".join(parts), sum(1 for v in orphan.values() if v)


def structure_map_svg(
    markdown: str,
    baseline_chunks: Sequence[str],
    structured_chunks: Sequence[str],
    *,
    theme: Theme = LIGHT,
    doc_name: str = "document.md",
    chunk_size: int = 1000,
) -> str:
    """Render both chunkings of one document at an identical vertical scale."""
    source_lines = classify_lines(markdown)

    pad = 32
    panel_w = 300.0
    col_gap = 190.0
    width = pad * 2 + panel_w * 2 + col_gap
    head_h = 74
    body_h = len(source_lines) * _LINE_PITCH + 34
    legend_h = 62
    height = head_h + body_h + legend_h + pad

    n_tables = len({sl.table_id for sl in source_lines if sl.table_id >= 0})
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'aria-label="Chunk boundaries over one document, character-window versus structure-aware">',
        f'<rect width="{width:.0f}" height="{height:.0f}" fill="{theme.surface}"/>',
        _text(pad, pad + 6, "Where the chunk boundaries fall", fill=theme.text_primary, size=17, weight=700),
        _text(
            pad,
            pad + 26,
            f"{doc_name} - {len(source_lines)} lines, {n_tables} tables - chunk size {chunk_size} chars - "
            "one bar per source line, same vertical scale in both columns",
            fill=theme.text_secondary,
            size=11.5,
        ),
    ]

    left, orphan_l = _panel(
        pad, head_h, panel_w,
        "Character-window chunking",
        f"{len(baseline_chunks)} chunks - the previous default",
        theme.series_1, source_lines, baseline_chunks, theme,
    )
    right, orphan_r = _panel(
        pad + panel_w + col_gap, head_h, panel_w,
        "Structure-aware chunking",
        f"{len(structured_chunks)} chunks - factor_rag.chunk_markdown",
        theme.series_1, source_lines, structured_chunks, theme,
    )
    out.extend([left, right])

    # Legend -- identity is never carried by colour alone; every swatch is labelled.
    ly = head_h + body_h + 20
    out.append(f'<line x1="{pad}" y1="{ly - 16:.1f}" x2="{width - pad:.1f}" y2="{ly - 16:.1f}" stroke="{theme.grid}" stroke-width="1"/>')
    legend = [
        ("Heading", theme.text_primary, 0.8),
        ("Prose / figure / formula", theme.substrate, 1.0),
        ("Table row with its header", theme.series_1, 0.5),
        ("Table row orphaned from its header", theme.critical, 1.0),
    ]
    lx = pad
    for label, colour, op in legend:
        out.append(_rect(lx, ly + 2, 14, 8, colour, rx=2, opacity=op))
        out.append(_text(lx + 20, ly + 10, label, fill=theme.text_secondary, size=11.5))
        lx += 24 + len(label) * 6.3
    out.append(
        _text(
            pad,
            ly + 32,
            f"Orphaned table rows: {orphan_l} of {sum(1 for s in source_lines if s.kind == 'table_row')} "
            f"on the left, {orphan_r} on the right.  Grey capsules on each column's left edge mark chunk extents.",
            fill=theme.text_muted,
            size=11,
        )
    )
    out.append("</svg>")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Figure 2 -- fidelity bars
# --------------------------------------------------------------------------- #


def _hbar(x: float, y: float, w: float, h: float, fill: str, r: float = 4.0) -> str:
    """Horizontal bar with rounded data-end, square against the baseline."""
    w = max(w, 1.0)
    r = min(r, w, h / 2)
    return (
        f'<path d="M{x:.1f} {y:.1f} H{x + w - r:.1f} A{r:.1f} {r:.1f} 0 0 1 {x + w:.1f} {y + r:.1f} '
        f'V{y + h - r:.1f} A{r:.1f} {r:.1f} 0 0 1 {x + w - r:.1f} {y + h:.1f} H{x:.1f} Z" fill="{fill}"/>'
    )


def fidelity_bars_svg(
    metrics: Sequence[Tuple[str, float, float]],
    *,
    theme: Theme = LIGHT,
    title: str = "Structural fidelity after chunking",
    subtitle: str = "",
) -> str:
    """Grouped horizontal bars: ``(metric_label, baseline, structured)`` as 0-1 rates."""
    pad = 32
    label_w = 232.0
    plot_w = 430.0
    row_h = 46.0
    bar_h = 15.0
    head_h = 92.0
    width = pad * 2 + label_w + plot_w + 62
    height = head_h + len(metrics) * row_h + 40

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="{_esc(title)}">',
        f'<rect width="{width:.0f}" height="{height:.0f}" fill="{theme.surface}"/>',
        _text(pad, pad + 4, title, fill=theme.text_primary, size=17, weight=700),
    ]
    if subtitle:
        out.append(_text(pad, pad + 24, subtitle, fill=theme.text_secondary, size=11.5))

    # Legend, always present for two series.
    lx = pad
    ly = pad + 44
    for label, colour in (
        ("Character-window baseline", theme.series_2),
        ("Structure-aware", theme.series_1),
    ):
        out.append(_rect(lx, ly - 8, 14, 8, colour, rx=2))
        out.append(_text(lx + 20, ly, label, fill=theme.text_secondary, size=11.5))
        lx += 26 + len(label) * 6.4

    plot_x = pad + label_w
    # Recessive gridlines at 0/25/50/75/100 %.
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        gx = plot_x + plot_w * frac
        out.append(
            f'<line x1="{gx:.1f}" y1="{head_h - 12:.1f}" x2="{gx:.1f}" '
            f'y2="{head_h + len(metrics) * row_h - 14:.1f}" stroke="{theme.grid}" stroke-width="1"/>'
        )
        out.append(
            _text(gx, head_h + len(metrics) * row_h + 2, f"{frac * 100:.0f}%",
                  fill=theme.text_muted, size=10.5, anchor="middle")
        )

    for i, (label, base, struct) in enumerate(metrics):
        top = head_h + i * row_h
        out.append(_text(plot_x - 14, top + 20, label, fill=theme.text_primary,
                         size=12.5, anchor="end", weight=500))
        # 2px gap between the two bars in a group.
        for j, (value, colour) in enumerate(((base, theme.series_2), (struct, theme.series_1))):
            by = top + j * (bar_h + 2)
            w = plot_w * max(min(value, 1.0), 0.0)
            out.append(_hbar(plot_x, by, w, bar_h, colour))
            out.append(
                _text(plot_x + w + 8, by + bar_h - 3, f"{value * 100:.0f}%",
                      fill=theme.text_secondary, size=11.5, weight=600, family=MONO)
            )

    out.append("</svg>")
    return "\n".join(out)
