#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end pipeline: chunk, embed, store, retrieve, answer.

Components load lazily, so constructing a ``RAGSystem`` does not pull a model
into memory until something actually needs to embed.

Author: Zeteng Lin (Hong Kong University of Science and Technology, Guangzhou)
"""

from __future__ import annotations

import logging
import time
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .config import TOP_K_RETRIEVAL
from .document_processor import get_document_processor
from .models.llm_model import get_llm_model
from .models.vector_model import get_vector_model
from .vector_db import get_vector_db

logger = logging.getLogger(__name__)


class RAGSystem:
    """Retrieval-augmented QA over Markdown converted from visually rich documents."""

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_backend: str = "sentence-transformers",
    ) -> None:
        self.collection_name = collection_name
        self.embedding_backend = embedding_backend
        self.document_processor = get_document_processor()
        self._vector_model = None
        self._vector_db = None
        self._llm_model = None

    @property
    def vector_model(self):
        if self._vector_model is None:
            start = time.time()
            self._vector_model = get_vector_model(self.embedding_backend)
            logger.info("embedding model ready in %.2fs", time.time() - start)
        return self._vector_model

    @property
    def vector_db(self):
        if self._vector_db is None:
            self._vector_db = get_vector_db(collection_name=self.collection_name)
        return self._vector_db

    @property
    def llm_model(self):
        if self._llm_model is None:
            self._llm_model = get_llm_model()
        return self._llm_model

    def index_documents(self, file_paths: List[Union[str, Path]]) -> int:
        documents = self.document_processor.process_documents(file_paths)
        if not documents:
            logger.warning("no chunks produced from %d file(s)", len(file_paths))
            return 0
        logger.info("embedding %d chunks", len(documents))
        self.vector_db.add_documents(self.vector_model.embed_documents(documents))
        logger.info("collection now holds %d chunks", self.vector_db.get_collection_count())
        return len(documents)

    def index_directory(self, directory_path: Union[str, Path], pattern: str = "**/*.md") -> int:
        directory = Path(directory_path)
        if not directory.is_dir():
            raise ValueError(f"not a directory: {directory_path}")
        paths = glob(str(directory / pattern), recursive=True)
        if not paths:
            logger.warning("no files matching %r under %s", pattern, directory_path)
            return 0
        return self.index_documents(paths)

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        return self.vector_db.search(self.vector_model.embed(query), top_k=top_k)

    def query(self, query: str, top_k: int = TOP_K_RETRIEVAL, with_sources: bool = False):
        results = self.retrieve(query, top_k=top_k)
        if not results:
            answer = "No indexed content matched that question."
            return (answer, []) if with_sources else answer
        answer = self.llm_model.rag_generate(query, [r["text"] for r in results])
        return (answer, results) if with_sources else answer


_rag_system_instance: Optional[RAGSystem] = None


def get_rag_system(collection_name: str = "documents") -> RAGSystem:
    global _rag_system_instance
    if _rag_system_instance is None:
        _rag_system_instance = RAGSystem(collection_name=collection_name)
    return _rag_system_instance
