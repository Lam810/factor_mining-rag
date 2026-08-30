#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Vector storage and retrieval, backed by ChromaDB.

Two correctness fixes relative to earlier revisions of this file:

1. Document ids used Python's built-in ``hash()``, which is randomised per
   process (``PYTHONHASHSEED``) unless explicitly disabled. Re-indexing the
   same corpus in a new process therefore minted new ids for identical
   content instead of overwriting the old ones, silently duplicating every
   document on every re-run. Ids are now a SHA-1 of the chunk's own
   ``chunk_uid`` (falling back to its text), which is stable across
   processes and machines.
2. ``client.get_collection`` raises ``chromadb.errors.NotFoundError`` on
   current ChromaDB, not the ``ValueError`` this module used to catch --  so
   the "create it if missing" branch never actually ran on a fresh database.
   This now goes through ``get_or_create_collection``, which does not depend
   on either exception type.

Author: Zeteng Lin (Hong Kong University of Science and Technology, Guangzhou)
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import chromadb
import numpy as np
from chromadb.config import Settings

from .config import TOP_K_RETRIEVAL, VECTOR_DB_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _stable_id(doc: Dict[str, Any], index: int) -> str:
    """A content-derived id that is identical across processes and runs."""
    key = doc.get("metadata", {}).get("chunk_uid") or doc["text"]
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"doc_{index}_{digest}"


class VectorDatabase:
    """Thin wrapper around a persistent ChromaDB collection."""

    def __init__(self, collection_name: str = "documents", persist_directory: Union[str, Path] = VECTOR_DB_DIR):
        """
        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: On-disk directory for the persistent client.
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add or overwrite documents in the collection.

        Args:
            documents: Dicts with ``text``, ``metadata``, and optionally a
                precomputed ``embedding``. Ids are derived from content, so
                re-adding the same chunk updates it in place instead of
                creating a duplicate.
        """
        if not documents:
            logger.warning("No documents to add")
            return

        ids, embeddings, metadatas, contents = [], [], [], []
        for i, doc in enumerate(documents):
            ids.append(_stable_id(doc, i))
            embeddings.append(doc.get("embedding"))
            metadatas.append(doc["metadata"])
            contents.append(doc["text"])

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings if all(e is not None for e in embeddings) else None,
            metadatas=metadatas,
            documents=contents,
        )
        logger.info(f"Upserted {len(documents)} document(s) into the vector database")

    def search(self, query_embedding: np.ndarray, top_k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        """Return the ``top_k`` documents nearest to ``query_embedding``."""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = []
        for i in range(len(results["documents"][0])):
            documents.append(
                {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )
        return documents

    def get_collection_count(self) -> int:
        """Return the number of documents currently stored."""
        return self.collection.count()

    def delete_collection(self) -> None:
        """Delete the underlying collection."""
        self.client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")


_vector_db_instance = None


def get_vector_db(collection_name: str = "documents") -> VectorDatabase:
    """Return the process-wide :class:`VectorDatabase` singleton."""
    global _vector_db_instance
    if _vector_db_instance is None:
        _vector_db_instance = VectorDatabase(collection_name=collection_name)
    return _vector_db_instance
