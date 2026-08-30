#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Embedding model, served locally through vLLM.

This is the GPU-heavy half of the pipeline: the default checkpoint is a 7B
instruction-tuned embedder. See the project README ("Running without a GPU")
for a CPU-only alternative -- this module intentionally does not fall back to
one silently, so a missing GPU fails loudly at model load rather than
producing degraded embeddings without saying so.

Author: Zeteng Lin (Hong Kong University of Science and Technology, Guangzhou)
"""

from typing import Any, Dict, List, Union

import numpy as np
import torch
from tqdm import tqdm
from vllm import LLM

from ..config import MAX_WORKERS, VECTOR_MODEL, VECTOR_MODEL_TASK


class VectorEmbedder:
    """Text embedding model served through vLLM."""

    def __init__(self, model: str = VECTOR_MODEL, task: str = VECTOR_MODEL_TASK):
        """
        Args:
            model: Embedding model name or local path.
            task: vLLM task type -- ``"embed"`` for this pipeline.
        """
        self.model_name = model
        self.task = task
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            raise RuntimeError(
                f"No CUDA device visible; refusing to load {model!r} on CPU via vLLM. "
                "Set FACTOR_RAG_VECTOR_MODEL to a small sentence-transformers checkpoint "
                "and FACTOR_RAG_VECTOR_BACKEND=sentence-transformers to run without a GPU "
                "(see README: 'Running without a GPU')."
            )

        self.llm = LLM(model=model, task=task)

    def embed(self, text: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Embed a string or a list of strings.

        Args:
            text: A single string, or a list of strings.
            batch_size: Batch size used when ``text`` is a list.

        Returns:
            A single embedding vector, or a 2D array of embeddings.
        """
        if isinstance(text, str):
            return self.llm.embed(text)[0]

        results = []
        for i in tqdm(range(0, len(text), batch_size), desc="Embedding"):
            batch = text[i : i + batch_size]
            results.extend(self.llm.embed(t)[0] for t in batch)
        return np.array(results)

    def embed_documents(self, documents: List[Dict[str, Any]], text_key: str = "text") -> List[Dict[str, Any]]:
        """Attach an ``"embedding"`` key to each document in ``documents``."""
        texts = [doc[text_key] for doc in documents]
        embeddings = self.embed(texts)
        for doc, embedding in zip(documents, embeddings):
            doc["embedding"] = embedding
        return documents

    def __call__(self, text: Union[str, List[str]]) -> np.ndarray:
        return self.embed(text)


_vector_model_instance = None


def get_vector_model() -> VectorEmbedder:
    """Return the process-wide :class:`VectorEmbedder` singleton."""
    global _vector_model_instance
    if _vector_model_instance is None:
        _vector_model_instance = VectorEmbedder()
    return _vector_model_instance
