#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure what structure-aware chunking buys, and regenerate the README figures.

    python benchmarks/run_benchmark.py                # score the bundled corpus
    python benchmarks/run_benchmark.py --docs mine/   # score your own corpus
    python benchmarks/run_benchmark.py --no-figures   # numbers only

Everything it reports is reproducible from a clean clone: the corpus lives in
``samples/`` and no model, API key, or GPU is involved.

Author: Zeteng Lin (Hong Kong University of Science and Technology, Guangzhou)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from factor_rag.chunking import chunk_markdown, naive_chunk  # noqa: E402
from factor_rag.metrics import FidelityReport, score_chunks  # noqa: E402
from factor_rag import viz  # noqa: E402

METRIC_ROWS = [
    ("Table integrity", "table_integrity"),
    ("Table rows keeping their header", "row_header_coverage"),
    ("Chunks that can name their section", "section_attribution"),
    ("Figures still beside their caption", "figure_caption_adjacency"),
]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate(docs: Sequence[Path], chunk_size: int, overlap: int):
    rows = []
    for path in sorted(docs):
        md = path.read_text(encoding="utf-8")
        base = naive_chunk(md, chunk_size, overlap)
        struct = [c.text for c in chunk_markdown(md, source=str(path), chunk_size=chunk_size, chunk_overlap=overlap)]
        rows.append((path, md, base, struct, score_chunks(md, base), score_chunks(md, struct)))
    return rows


def print_report(rows, chunk_size: int, overlap: int) -> Dict:
    print(f"\nCorpus: {len(rows)} document(s)   chunk_size={chunk_size}   chunk_overlap={overlap}\n")
    header = f"{'Document':<28}{'Tables':>7}{'Rows':>6}{'  baseline -> structure-aware'}"
    print(header)
    print("-" * len(header))
    for path, _md, base, struct, rb, rs in rows:
        print(
            f"{path.name:<28}{rb.n_tables:>7}{rb.n_table_rows:>6}"
            f"   table integrity {rb.table_integrity:>6.0%} -> {rs.table_integrity:>4.0%}"
            f"   chunks {len(base):>3} -> {len(struct):<3}"
        )

    agg = {}
    print()
    print(f"{'Metric':<42}{'Baseline':>10}{'Structure-aware':>18}")
    print("-" * 70)
    for label, field in METRIC_ROWS:
        b = _mean([getattr(r[4], field) for r in rows])
        s = _mean([getattr(r[5], field) for r in rows])
        agg[field] = (b, s)
        print(f"{label:<42}{b:>9.0%}{s:>17.0%}")
    print()
    return agg


def write_figures(rows, agg, chunk_size: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Pick the most table-dense document for the structure map.
    path, md, base, struct, rb, rs = max(rows, key=lambda r: r[4].n_table_rows)

    for theme in (viz.LIGHT, viz.DARK):
        (out_dir / f"structure-map-{theme.name}.svg").write_text(
            viz.structure_map_svg(md, base, struct, theme=theme, doc_name=path.name, chunk_size=chunk_size),
            encoding="utf-8",
        )
        (out_dir / f"fidelity-{theme.name}.svg").write_text(
            viz.fidelity_bars_svg(
                [(label, *agg[field]) for label, field in METRIC_ROWS],
                theme=theme,
                title="Structural fidelity after chunking",
                subtitle=f"Mean over the {len(rows)}-document sample corpus, chunk size {chunk_size} characters. Higher is better.",
            ),
            encoding="utf-8",
        )
    print(f"figures written to {out_dir}/")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", default=str(ROOT / "samples"), help="directory of .md files")
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--json", default="", help="also write raw results here")
    args = ap.parse_args()

    docs = [p for p in Path(args.docs).glob("*.md") if p.name.lower() != "readme.md"]
    if not docs:
        print(f"no .md files under {args.docs}", file=sys.stderr)
        return 1

    rows = evaluate(docs, args.chunk_size, args.overlap)
    agg = print_report(rows, args.chunk_size, args.overlap)

    if not args.no_figures:
        write_figures(rows, agg, args.chunk_size, ROOT / "assets")

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "chunk_size": args.chunk_size,
                    "overlap": args.overlap,
                    "documents": {
                        r[0].name: {"baseline": r[4].as_dict(), "structured": r[5].as_dict()} for r in rows
                    },
                    "aggregate": {k: {"baseline": v[0], "structured": v[1]} for k, v in agg.items()},
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
