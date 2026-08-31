#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Embedding backends.

Two are provided.  ``SentenceTransformerEmbedder`` is the default: it runs on
CPU, needs no server, and is what the quickstart uses.  ``VLLMEmbedder`` keeps
the original vLLM path for GPU throughput, with the output unwrapping corrected
-- ``LLM.embed`` returns request objects, and the published code passed those
straight into NumPy instead of reading ``.outputs.embedding`` off them.

Author: Zeteng Lin (Hong Kong University of Science and Technology, Guangzhou)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

from ..config import VECTOR_MODEL, VECTOR_MODEL_TASK

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder:
    """CPU-friendly default backend."""

    def __init__(self, model: str = VECTOR_MODEL, device: Optional[str] = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self.model = SentenceTransformer(model, device=device)
        logger.info("loaded embedding model %s", model)

    def embed(self, text: Union[str, Sequence[str]], batch_size: int = 32):
        single = isinstance(text, str)
        payload = [text] if single else list(text)
        vectors = self.model.encode(
            payload, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors[0] if single else vectors

    def embed_documents(
        self, documents: List[Dict[str, Any]], text_key: str = "text"
    ) -> List[Dict[str, Any]]:
        vectors = self.embed([d[text_key] for d in documents])
        for doc, vector in zip(documents, vectors):
            doc["embedding"] = vector
        return documents

    def __call__(self, text):
        return self.embed(text)


class VLLMEmbedder:
    """vLLM backend, for GPU throughput on large corpora."""

    def __init__(self, model: str = VECTOR_MODEL, task: str = VECTOR_MODEL_TASK) -> None:
        from vllm import LLM

        self.model_name = model
        self.llm = LLM(model=model, task=task)
        logger.info("loaded vLLM embedding model %s", model)

    @staticmethod
    def _vector(output):
        # vLLM returns EmbeddingRequestOutput objects, not raw vectors.
        embedding = getattr(getattr(output, "outputs", None), "embedding", None)
        return embedding if embedding is not None else output

    def embed(self, text: Union[str, Sequence[str]], batch_size: int = 32):
        import numpy as np

        single = isinstance(text, str)
        payload = [text] if single else list(text)
        outputs = self.llm.embed(payload)
        vectors = np.asarray([self._vector(o) for o in outputs])
        return vectors[0] if single else vectors

    def embed_documents(
        self, documents: List[Dict[str, Any]], text_key: str = "text"
    ) -> List[Dict[str, Any]]:
        vectors = self.embed([d[text_key] for d in documents])
        for doc, vector in zip(documents, vectors):
            doc["embedding"] = vector
        return documents

    def __call__(self, text):
        return self.embed(text)


VectorEmbedder = SentenceTransformerEmbedder

_vector_model_instance = None


def get_vector_model(backend: str = "sentence-transformers"):
    """Return the shared embedder. ``backend`` is ``sentence-transformers`` or ``vllm``."""
    global _vector_model_instance
    if _vector_model_instance is None:
        _vector_model_instance = (
            VLLMEmbedder() if backend == "vllm" else SentenceTransformerEmbedder()
        )
    return _vector_model_instance
