<div align="center">

# factor-rag

**Structure-preserving RAG for visually rich documents.**
Tables, figures, and formulas survive chunking intact — instead of being sliced apart by a character counter.

[![tests](https://github.com/Lam810/factor_mining-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/Lam810/factor_mining-rag/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-informational.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![Zero-dependency core](https://img.shields.io/badge/core%20deps-tqdm%20only-brightgreen.svg)](pyproject.toml)

</div>

---

## The problem, in one picture

Feed a PDF through a layout parser (MinerU, Marker, PP-StructureV2, Nougat, ...) and you get Markdown that is
mostly *structure*: multi-page pipe tables, figure references, fenced formulas, a heading hierarchy. Almost
every RAG tutorial then chunks that Markdown with a character-window splitter — cut every *N* characters,
nudge the boundary to the nearest paragraph break. That splitter has no idea a table exists. It cuts a
40-row table in half, and the second half becomes a chunk full of bare numbers with no header, no units, no
name — unretrievable by an embedding model and unreadable by an LLM.

Below is the same real-world-shaped document (a 124-line factor-attribution report, one of the three sample
documents in [`samples/`](samples/)), chunked both ways at an identical, realistic setting
(`chunk_size=1000`). Every bar is one line of source text, both columns share the same vertical scale, and
red marks a table row that lost its header:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/structure-map-dark.svg">
  <img alt="Chunk boundaries over one document: character-window chunking orphans 60 of 94 table rows from their header; structure-aware chunking orphans zero." src="assets/structure-map-light.svg">
</picture>

That is not a contrived worst case — it is the default behavior of the chunking code in most RAG tutorials
and starter templates, on a table of perfectly ordinary size. This repository is the fix, plus the tooling
to measure it instead of assuming it.

## What's actually in here

- **`factor_rag.chunking`** — a structure-aware Markdown chunker. It parses the document into semantic
  blocks (heading, table, fenced code, figure+caption, list, paragraph) *before* deciding where to cut, so a
  cut only ever lands between blocks, never through one. An oversized table is split into parts that each
  repeat the header row; an oversized code block re-opens its fence in every part; a figure is never
  separated from its caption; every chunk is tagged with the heading breadcrumb of the section it came from.
  Zero dependencies beyond the standard library.
- **`factor_rag.metrics`** — turns "this chunking is better" from a vibe into a number: table integrity, the
  fraction of table rows that keep their header, section attribution, figure/caption adjacency — computed
  from chunk text alone, so any chunker (not just this one) can be scored on equal terms.
- **`factor_rag.viz`** — the two figures in this README, as dependency-free SVG generation. No matplotlib, no
  browser, no headless Chrome; `git diff` on a regenerated figure is a readable text diff.
- **`benchmarks/run_benchmark.py`** — runs the metrics over a sample corpus and regenerates both figures. One
  command, no GPU, no API key, fully reproducible from a clean clone.
- **A retrieval + generation pipeline** (`factor_rag.rag_system`, `vector_db`, `document_processor`,
  `models/`) wired to the chunker above, for going from a folder of Markdown to an answered question. This
  half needs a GPU (for the default local embedder) and an OpenRouter API key — see
  [Running without a GPU](#running-without-a-gpu) for the CPU path.

## Results

Mean over the 3-document sample corpus in [`samples/`](samples/), `chunk_size=1000`, `chunk_overlap=200` —
reproduce with `python benchmarks/run_benchmark.py`:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/fidelity-dark.svg">
  <img alt="Structural fidelity after chunking: table integrity 78% to 100%, table rows keeping their header 79% to 100%, chunks that can name their section 81% to 100%, figures still beside their caption 100% to 100%." src="assets/fidelity-light.svg">
</picture>

| Metric | Character-window baseline | Structure-aware | 
| --- | ---: | ---: |
| Table integrity (whole tables kept intact) | 78% | **100%** |
| Table rows that keep their header | 79% | **100%** |
| Chunks that can name their own section | 81% | **100%** |
| Figures still next to their caption | 100% | 100% |

Figure/caption adjacency ties at this chunk size: the sample corpus's captions sit close enough to their
images that even a blind character window rarely separates them. Tables are the load-bearing failure mode —
they run for dozens of rows, so *any* fixed-size window eventually lands inside one. That is also exactly
the content type "visually rich document" RAG exists to handle, which is why table integrity is the headline
number here rather than a footnote.

## Quickstart

```bash
pip install "factor-rag[serve] @ git+https://github.com/Lam810/factor_mining-rag.git"
# or, for just the chunker/metrics/viz (zero extra dependencies):
pip install "factor-rag @ git+https://github.com/Lam810/factor_mining-rag.git"
```

```python
from factor_rag import chunk_markdown, score_chunks

markdown = open("quarterly_report.md", encoding="utf-8").read()

chunks = chunk_markdown(markdown, source="quarterly_report.md", chunk_size=1000, chunk_overlap=200)
for c in chunks[:2]:
    print(c.metadata["heading_path"], "|", c.metadata["block_kinds"])
    print(c.text[:200], "...\n")

# Score it against your own baseline chunker -- score_chunks only looks at
# chunk text, so it works for any strategy, not just this one.
report = score_chunks(markdown, [c.text for c in chunks])
print(f"table integrity: {report.table_integrity:.0%}")
```

For the full retrieval + generation pipeline:

```python
import os
os.environ["OPENROUTER_API_KEY"] = "sk-or-..."

from factor_rag.rag_system import get_rag_system

rag = get_rag_system()
rag.index_directory("samples/", pattern="*.md")
print(rag.query("Which industries drove the active return?"))
```

## Running without a GPU

The chunker, metrics, and figure generation (the three modules above) need nothing but `tqdm` and run
anywhere. The retrieval pipeline's default embedder is a 7B instruction-tuned model served through vLLM,
which does need a GPU. To run the full pipeline on CPU, point it at a small `sentence-transformers`
checkpoint instead:

```bash
export FACTOR_RAG_VECTOR_MODEL=BAAI/bge-small-zh-v1.5
export FACTOR_RAG_VECTOR_BACKEND=sentence-transformers
```

(`sentence-transformers` backend wiring is a good first contribution — see [Roadmap](#roadmap).)

## How it works

```
                    ┌────────────────────┐
  Markdown  ───────▶│   parse_blocks()   │  heading / table / code / figure / list / paragraph
  (from a layout    └─────────┬──────────┘
   parser: MinerU,            │
   Marker, PP-Struct...)      ▼
                    ┌────────────────────┐
                    │  chunk_markdown()  │  packs whole blocks into a budget;
                    │                    │  splits an oversized block WITHOUT
                    │                    │  breaking it (repeat header / re-open
                    │                    │  fence / never split a figure)
                    └─────────┬──────────┘
                              ▼
                 Chunk(text, metadata={heading_path,
                       chunk_uid, has_table, has_figure, ...})
                              │
              ┌───────────────┼────────────────┐
              ▼                                 ▼
   score_chunks() / viz.py            embed → ChromaDB → OpenRouter LLM
   (measure any chunker,               (factor_rag.rag_system —
    no GPU, no deps)                    needs a GPU + API key)
```

## Roadmap

- [ ] `sentence-transformers` embedding backend, so the full pipeline runs end-to-end on CPU.
- [ ] Ascend (昇腾) NPU backend for the embedding step, via `torch_npu`. In progress in a sibling project
      ([`visual-document-rag`](https://github.com/Lam810/visual-document-rag)) — not yet verified on real
      Ascend hardware, so it is listed here as planned rather than claimed.
- [ ] A recursive variant that also parses nested structures (a table inside a list item), which current
      layout-parser output does not produce but hand-written Markdown sometimes does.
- [ ] FAISS / Qdrant backends alongside ChromaDB.

Contributions welcome — `tests/` has 19 fast, dependency-free tests (`pytest`) covering exactly the failure
modes described above; a PR that adds a case this suite doesn't catch is a very welcome PR.

## Citation

If this chunker or its benchmark methodology is useful in your own work, please cite it:

```bibtex
@software{lin2026factorrag,
  author  = {Lin, Zeteng},
  title   = {factor-rag: Structure-Preserving RAG for Visually Rich Documents},
  year    = {2026},
  url     = {https://github.com/Lam810/factor_mining-rag},
  note    = {Hong Kong University of Science and Technology (Guangzhou)}
}
```

(GitHub's "Cite this repository" button, top right, generates the same citation from
[`CITATION.cff`](CITATION.cff) in APA or BibTeX.)

## License

[MIT](LICENSE) © 2026 Zeteng Lin, Hong Kong University of Science and Technology (Guangzhou)
