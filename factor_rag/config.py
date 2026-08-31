#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration, resolved from the environment rather than hardcoded paths.

The published version of this file hardcoded ``/root/autodl-fs/rag_system`` and
created those directories as an import side effect, so importing the package on
any machine other than the author's rented GPU box wrote directories into the
filesystem root.  Everything is now overridable and nothing is created until a
component actually needs it.

Author: Zeteng Lin (Hong Kong University of Science and Technology, Guangzhou)
"""

from __future__ import annotations

import os
from pathlib import Path

#: Root for anything this package persists.  Override with ``FACTOR_RAG_HOME``.
BASE_DIR = Path(os.environ.get("FACTOR_RAG_HOME", Path.home() / ".factor_rag"))
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = Path(os.environ.get("FACTOR_RAG_VECTOR_DB", DATA_DIR / "vector_db"))

# -- Language model ---------------------------------------------------------- #
# Read from the environment only; never write a key into this file.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.environ.get("FACTOR_RAG_LLM", "deepseek/deepseek-chat")

# -- Embeddings -------------------------------------------------------------- #
VECTOR_MODEL = os.environ.get("FACTOR_RAG_EMBED_MODEL", "intfloat/multilingual-e5-large")
VECTOR_MODEL_TASK = "embed"

# -- Chunking ---------------------------------------------------------------- #
CHUNK_SIZE = int(os.environ.get("FACTOR_RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("FACTOR_RAG_CHUNK_OVERLAP", "200"))
SECTION_LEVEL = int(os.environ.get("FACTOR_RAG_SECTION_LEVEL", "2"))

# -- Retrieval --------------------------------------------------------------- #
TOP_K_RETRIEVAL = int(os.environ.get("FACTOR_RAG_TOP_K", "5"))
MAX_WORKERS = int(os.environ.get("FACTOR_RAG_MAX_WORKERS", "4"))


def ensure_dirs() -> None:
    """Create the persistence directories. Called by components that write."""
    for path in (BASE_DIR, DATA_DIR, VECTOR_DB_DIR):
        path.mkdir(parents=True, exist_ok=True)
