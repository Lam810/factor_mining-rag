#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Document loading and chunking, backed by :mod:`factor_rag.chunking`.

This module used to implement its own character-window splitter directly. It
had two independent bugs: no awareness of Markdown structure (a table row
could end up in a chunk without its header -- see the project README for why
that matters for visually rich documents), and a loop that could fail to
terminate when ``chunk_overlap`` was close to ``chunk_size`` (``start`` could
move backwards forever). Both are fixed by delegating to
:func:`factor_rag.chunking.chunk_markdown`, which this module now does while
keeping the public API (``DocumentProcessor.process_document`` etc.) unchanged
so existing callers do not need to change.

Author: Zeteng Lin (Hong Kong University of Science and Technology, Guangzhou)
"""

import logging
import multiprocessing
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Union

from tqdm import tqdm

from .chunking import chunk_markdown
from .config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_WORKERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Loads Markdown files and splits them into retrievable chunks."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Characters of prose context carried between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_markdown(self, file_path: Union[str, Path]) -> str:
        """Read a Markdown file as UTF-8 text.

        Args:
            file_path: Path to the Markdown file.

        Returns:
            The file's contents.

        Raises:
            FileNotFoundError: If ``file_path`` does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def split_text(self, text: str, *, source: str = "") -> List[str]:
        """Split ``text`` into chunk strings using structure-aware chunking.

        Kept for backward compatibility with callers that only want raw
        strings back. Prefer :meth:`process_document` when you also want
        per-chunk metadata (heading path, table/figure flags, chunk id).
        """
        if not isinstance(text, str):
            raise ValueError(f"Expected a string, got {type(text)}")
        return [c.text for c in chunk_markdown(
            text, source=source, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )]

    def process_document(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load and chunk a single document.

        Args:
            file_path: Path to the document.

        Returns:
            A list of ``{"text": ..., "metadata": {...}}`` dicts, one per
            chunk. ``metadata`` includes ``source``, ``chunk_id``, the heading
            breadcrumb the chunk came from, and whether it contains a table,
            figure, or code block -- see :class:`factor_rag.chunking.Chunk`.
        """
        try:
            content = self.load_markdown(file_path)
            file_name = Path(file_path).name

            chunks = chunk_markdown(
                content, source=str(file_path), chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )

            documents = []
            for chunk in chunks:
                metadata = dict(chunk.metadata)
                metadata["file_name"] = file_name
                metadata["total_chunks"] = len(chunks)
                documents.append({"text": chunk.text, "metadata": metadata})

            return documents

        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {e}")
            return []

    def process_documents(self, file_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """Process multiple documents, in parallel once there are enough of them.

        Args:
            file_paths: Paths to the documents to process.

        Returns:
            The concatenation of :meth:`process_document` over every path.
        """
        if len(file_paths) <= 1:
            all_documents = []
            for file_path in tqdm(file_paths, desc="Processing documents"):
                all_documents.extend(self.process_document(file_path))
            return all_documents

        num_workers = min(MAX_WORKERS, len(file_paths), multiprocessing.cpu_count())
        logger.info(f"Processing {len(file_paths)} documents with {num_workers} worker(s)")

        with Pool(num_workers) as pool:
            results = list(
                tqdm(pool.imap(self.process_document, file_paths), total=len(file_paths), desc="Processing documents")
            )

        all_documents = []
        for documents in results:
            all_documents.extend(documents)
        return all_documents


_document_processor_instance = None


def get_document_processor() -> DocumentProcessor:
    """Return the process-wide :class:`DocumentProcessor` singleton."""
    global _document_processor_instance
    if _document_processor_instance is None:
        _document_processor_instance = DocumentProcessor()
    return _document_processor_instance
