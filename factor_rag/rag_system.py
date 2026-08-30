#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""RAGSystem: the top-level class wiring embedding, storage, retrieval and generation.

Author: Zeteng Lin (Hong Kong University of Science and Technology, Guangzhou)
"""

import logging
import os
import time
from glob import glob
from pathlib import Path
from typing import List, Optional, Union

from .config import OPENROUTER_API_KEY
from .document_processor import get_document_processor
from .models.llm_model import get_llm_model
from .models.vector_model import get_vector_model
from .vector_db import get_vector_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RAGSystem:
    """Retrieval-augmented generation over a directory of Markdown documents."""

    def __init__(self, collection_name: str = "documents", api_key: Optional[str] = None):
        """
        Args:
            collection_name: Name of the vector database collection to use.
            api_key: OpenRouter API key. Falls back to ``OPENROUTER_API_KEY``.
        """
        if api_key:
            os.environ["OPENROUTER_API_KEY"] = api_key
        elif not OPENROUTER_API_KEY:
            logger.warning("No OpenRouter API key set; generation will fail until one is provided")

        start_time = time.time()
        self.vector_model = get_vector_model()
        logger.info(f"Embedding model ready in {time.time() - start_time:.2f}s")

        self.document_processor = get_document_processor()
        self.vector_db = get_vector_db(collection_name=collection_name)

        self._llm_model = None  # lazily loaded -- generation is not always needed

    @property
    def llm_model(self):
        """Lazily-loaded generation model."""
        if self._llm_model is None:
            start_time = time.time()
            self._llm_model = get_llm_model()
            logger.info(f"LLM ready in {time.time() - start_time:.2f}s")
        return self._llm_model

    def index_documents(self, file_paths: List[Union[str, Path]]) -> int:
        """Chunk, embed, and store a list of documents.

        Args:
            file_paths: Paths to the documents to index.

        Returns:
            The number of chunks produced.
        """
        documents = self.document_processor.process_documents(file_paths)
        logger.info(f"{len(file_paths)} document(s) -> {len(documents)} chunk(s)")

        start_time = time.time()
        documents = self.vector_model.embed_documents(documents)
        logger.info(f"Embedded {len(documents)} chunk(s) in {time.time() - start_time:.2f}s")

        self.vector_db.add_documents(documents)
        logger.info(f"Vector database now holds {self.vector_db.get_collection_count()} document(s)")
        return len(documents)

    def index_directory(self, directory_path: Union[str, Path], pattern: str = "*.md") -> int:
        """Index every file matching ``pattern`` under ``directory_path``.

        Args:
            directory_path: Directory to scan.
            pattern: Glob pattern, relative to ``directory_path``.

        Returns:
            The number of chunks produced (``0`` if nothing matched).
        """
        dir_path = Path(directory_path)
        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"Not a valid directory: {directory_path}")

        file_paths = glob(os.path.join(str(dir_path), pattern), recursive=True)
        if not file_paths:
            logger.warning(f"No files matching {pattern!r} under {directory_path}")
            return 0

        return self.index_documents(file_paths)

    def query(self, query: str, top_k: int = 5) -> str:
        """Answer ``query`` using the indexed corpus.

        Args:
            query: The question to answer.
            top_k: Number of chunks to retrieve as context.

        Returns:
            The generated answer, or an apology string if nothing was retrieved.
        """
        query_embedding = self.vector_model(query)
        results = self.vector_db.search(query_embedding, top_k=top_k)

        if not results:
            return "I could not find any information relevant to your question."

        contexts = [doc["text"] for doc in results]
        system_prompt = (
            "You are a careful assistant that answers only from the supplied documents. "
            "Do not invent information that is not in them."
        )
        return self.llm_model.rag_generate(query, contexts, system_prompt)


_rag_system_instance = None


def get_rag_system(collection_name: str = "documents", api_key: Optional[str] = None) -> RAGSystem:
    """Return the process-wide :class:`RAGSystem` singleton."""
    global _rag_system_instance
    if _rag_system_instance is None:
        _rag_system_instance = RAGSystem(collection_name=collection_name, api_key=api_key)
    return _rag_system_instance
