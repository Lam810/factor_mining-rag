#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chroma-backed vector store.

Two defects in the published version are fixed here:

* Document ids were built from Python's ``hash()``, which is salted per process.
  Re-indexing the same corpus therefore produced fresh ids every run and silently
  duplicated the collection instead of updating it.  Ids are now a SHA-1 of the
  chunk's own content and provenance, and writes go through ``upsert``.
* ``get_collection`` was guarded by ``except ValueError``; current Chroma raises
  ``NotFoundError``, so the very first run crashed instead of creating the
  collection.  The guard now covers both.

Author: Zeteng Lin (Hong Kong University of Science and Technology, Guangzhou)
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .config import TOP_K_RETRIEVAL, VECTOR_DB_DIR, ensure_dirs

logger = logging.getLogger(__name__)


def _chunk_id(doc: Dict[str, Any]) -> str:
    meta = doc.get("metadata", {})
    if meta.get("chunk_uid"):
        return str(meta["chunk_uid"])
    payload = f"{meta.get('source', '')}|{meta.get('chunk_id', '')}|{doc['text']}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


class VectorDatabase:
    """Persistent Chroma collection with deterministic, idempotent writes."""

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Union[str, Path] = VECTOR_DB_DIR,
    ) -> None:
        import chromadb
        from chromadb.config import Settings

        ensure_dirs()
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info("opened existing collection %r", collection_name)
        except Exception:
            self.collection = self.client.create_collection(name=collection_name)
            logger.info("created collection %r", collection_name)

    def add_documents(self, documents: Sequence[Dict[str, Any]]) -> None:
        if not documents:
            logger.warning("nothing to index")
            return

        ids = [_chunk_id(d) for d in documents]
        contents = [d["text"] for d in documents]
        metadatas = [d.get("metadata", {}) for d in documents]
        embeddings = [d.get("embedding") for d in documents]
        have_embeddings = all(e is not None for e in embeddings)

        payload: Dict[str, Any] = {"ids": ids, "documents": contents, "metadatas": metadatas}
        if have_embeddings:
            payload["embeddings"] = [
                e.tolist() if hasattr(e, "tolist") else list(e) for e in embeddings
            ]

        # upsert, so re-indexing a corpus updates rather than duplicates.
        self.collection.upsert(**payload)
        logger.info("upserted %d chunks into %r", len(documents), self.collection_name)

    def search(self, query_embedding, top_k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        vector = query_embedding.tolist() if hasattr(query_embedding, "tolist") else list(query_embedding)
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if not results.get("documents") or not results["documents"][0]:
            return []
        return [
            {
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
            for i in range(len(results["documents"][0]))
        ]

    def get_collection_count(self) -> int:
        return self.collection.count()

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection_name)
        logger.info("deleted collection %r", self.collection_name)


_vector_db_instance: Optional[VectorDatabase] = None


def get_vector_db(collection_name: str = "documents") -> VectorDatabase:
    global _vector_db_instance
    if _vector_db_instance is None:
        _vector_db_instance = VectorDatabase(collection_name=collection_name)
    return _vector_db_instance
