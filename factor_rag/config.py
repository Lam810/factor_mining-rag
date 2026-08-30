#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Runtime configuration for the RAG pipeline.

Every path and credential is overridable via an environment variable, and the
default paths are portable (XDG-style, under the user's home). Earlier
revisions of this file hardcoded ``/root/autodl-fs/rag_system``, which only
existed on the author's rented AutoDL instance -- a fresh clone on any other
machine would fail at import time. That is fixed here: nothing in this module
assumes a specific host.

Author: Zeteng Lin (Hong Kong University of Science and Technology, Guangzhou)
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths -- all overridable, all portable by default
# --------------------------------------------------------------------------- #

BASE_DIR = Path(os.environ.get("FACTOR_RAG_HOME", Path.home() / ".cache" / "factor_rag"))
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
VECTOR_DB_DIR = Path(os.environ.get("FACTOR_RAG_VECTOR_DB_DIR", DATA_DIR / "vector_db"))

for _dir in (DATA_DIR, MODELS_DIR, VECTOR_DB_DIR):
    os.makedirs(_dir, exist_ok=True)

# --------------------------------------------------------------------------- #
# LLM generation, via OpenRouter's OpenAI-compatible endpoint
# --------------------------------------------------------------------------- #

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-chat")

# --------------------------------------------------------------------------- #
# Embedding model
# --------------------------------------------------------------------------- #

# The 7B instruction-tuned embedder gives the best retrieval quality but needs
# a GPU with ~16GB free for vLLM to serve it. Point FACTOR_RAG_VECTOR_MODEL at
# a small sentence-transformers checkpoint (e.g. "BAAI/bge-small-zh-v1.5") to
# run the pipeline end-to-end on CPU -- see README "Running without a GPU".
VECTOR_MODEL = os.environ.get("FACTOR_RAG_VECTOR_MODEL", "intfloat/e5-mistral-7b-instruct")
VECTOR_MODEL_TASK = "embed"
VECTOR_MODEL_BACKEND = os.environ.get("FACTOR_RAG_VECTOR_BACKEND", "vllm")  # "vllm" or "sentence-transformers"

# --------------------------------------------------------------------------- #
# Document processing
# --------------------------------------------------------------------------- #

MAX_WORKERS = int(os.environ.get("FACTOR_RAG_MAX_WORKERS", "4"))
CHUNK_SIZE = int(os.environ.get("FACTOR_RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("FACTOR_RAG_CHUNK_OVERLAP", "200"))

# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #

TOP_K_RETRIEVAL = int(os.environ.get("FACTOR_RAG_TOP_K", "5"))
