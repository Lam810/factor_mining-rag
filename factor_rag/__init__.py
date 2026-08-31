#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""factor_rag -- structure-aware RAG for visually rich documents.

The chunking and metrics layers have no heavy dependencies, so this import is
cheap.  Retrieval and generation components are imported lazily, because they
pull in chromadb / vllm / an HTTP client that most users of the chunker do not
need.

Author: Zeteng Lin (Hong Kong University of Science and Technology, Guangzhou)
"""

from .chunking import Block, Chunk, chunk_markdown, naive_chunk, parse_blocks
from .metrics import FidelityReport, score_chunks

__version__ = "0.2.0"
__author__ = "Zeteng Lin"

__all__ = [
    "Block",
    "Chunk",
    "chunk_markdown",
    "naive_chunk",
    "parse_blocks",
    "FidelityReport",
    "score_chunks",
    "__version__",
]
