#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Loading and chunking of Markdown produced by a document layout parser.

Chunking is delegated to :mod:`factor_rag.chunking`; this module handles I/O and
parallelism only.

Author: Zeteng Lin (Hong Kong University of Science and Technology, Guangzhou)
"""

from __future__ import annotations

import logging
import multiprocessing
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .chunking import chunk_markdown
from .config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_WORKERS, SECTION_LEVEL

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Loads Markdown files and turns them into retrievable chunks."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        section_level: int = SECTION_LEVEL,
        add_heading_context: bool = True,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.section_level = section_level
        self.add_heading_context = add_heading_context

    def load_markdown(self, file_path: Union[str, Path]) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"file not found: {file_path}")
        return path.read_text(encoding="utf-8")

    def split_text(self, text: str, source: Optional[str] = None) -> List[str]:
        """Structure-aware split. Kept for API compatibility with 0.1.x."""
        return [c.text for c in self._chunks(text, source)]

    def _chunks(self, text: str, source: Optional[str]):
        return chunk_markdown(
            text,
            source=source,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_heading_context=self.add_heading_context,
            section_level=self.section_level,
        )

    def process_document(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        try:
            content = self.load_markdown(file_path)
        except Exception as exc:
            logger.error("failed to read %s: %s", file_path, exc)
            return []

        chunks = self._chunks(content, str(file_path))
        for chunk in chunks:
            chunk.metadata.setdefault("file_name", Path(file_path).name)
            chunk.metadata["total_chunks"] = len(chunks)
        return [{"text": c.text, "metadata": c.metadata} for c in chunks]

    def process_documents(self, file_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        if len(file_paths) <= 1:
            return [d for p in file_paths for d in self.process_document(p)]

        workers = min(MAX_WORKERS, len(file_paths), multiprocessing.cpu_count())
        logger.info("processing %d documents with %d workers", len(file_paths), workers)
        with Pool(workers) as pool:
            results = pool.map(self.process_document, file_paths)
        return [doc for batch in results for doc in batch]


_document_processor_instance: Optional[DocumentProcessor] = None


def get_document_processor() -> DocumentProcessor:
    global _document_processor_instance
    if _document_processor_instance is None:
        _document_processor_instance = DocumentProcessor()
    return _document_processor_instance
