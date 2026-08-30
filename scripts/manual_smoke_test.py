#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Manual, opt-in smoke test for the full pipeline: embedding, LLM, indexing, query.

This is deliberately **not** named ``test_*.py`` and is not run by ``pytest`` /
CI: it needs a GPU (for the default embedder) and an OpenRouter API key, and it
talks to a real network endpoint. The automated, dependency-free test suite
lives in ``tests/`` and covers the chunking and metrics logic; this script is
for confirming the GPU-dependent half actually works on a given machine.

    python scripts/manual_smoke_test.py --api-key sk-... --test-all
    python scripts/manual_smoke_test.py --test-vector          # embedder only, no API key needed

By default it indexes the bundled ``samples/`` corpus, so it runs out of the
box on a fresh clone -- earlier revisions of this script pointed at
``/root/autodl-fs``, a path that only existed on the author's rented instance.

Author: Zeteng Lin (Hong Kong University of Science and Technology, Guangzhou)
"""

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DOCS_DIR = str(Path(__file__).resolve().parents[1] / "samples")


def test_vector_model() -> bool:
    """Load the embedder and embed a couple of strings."""
    from factor_rag.models.vector_model import get_vector_model

    logger.info("Loading embedding model...")
    vector_model = get_vector_model()

    embedding = vector_model("A test sentence used to check the embedding model.")
    logger.info(f"Embedding shape: {embedding.shape}")

    embeddings = vector_model(["first sentence", "second sentence", "third sentence"])
    logger.info(f"Batch embedding shape: {embeddings.shape}")
    return True


def test_llm_model(api_key: str) -> bool:
    """Send one prompt through the configured OpenRouter model."""
    from factor_rag.models.llm_model import get_llm_model

    if not api_key:
        logger.error("No OpenRouter API key provided; skipping LLM test")
        return False

    os.environ["OPENROUTER_API_KEY"] = api_key
    llm_model = get_llm_model()

    response = llm_model("Briefly introduce yourself.")
    logger.info(f"LLM reply: {response}")
    return True


def test_document_processing(docs_dir: str) -> bool:
    """Chunk every Markdown file in ``docs_dir``."""
    from factor_rag.document_processor import get_document_processor

    md_files = glob.glob(os.path.join(docs_dir, "*.md"))
    if not md_files:
        logger.error(f"No Markdown files found under {docs_dir}")
        return False

    logger.info(f"Found {len(md_files)} Markdown file(s)")
    processor = get_document_processor()
    documents = processor.process_documents(md_files)

    logger.info(f"Produced {len(documents)} chunk(s)")
    if documents:
        logger.info(f"First chunk: {documents[0]['text'][:100]}...")
    return True


def test_full_rag_system(api_key: str, docs_dir: str, query: str) -> bool:
    """Index ``docs_dir`` and answer ``query`` end to end."""
    from factor_rag.rag_system import get_rag_system

    if not api_key:
        logger.error("No OpenRouter API key provided; skipping full pipeline test")
        return False

    os.environ["OPENROUTER_API_KEY"] = api_key
    rag = get_rag_system(collection_name="smoke_test_collection", api_key=api_key)

    md_files = glob.glob(os.path.join(docs_dir, "*.md"))
    if not md_files:
        logger.error(f"No Markdown files found under {docs_dir}")
        return False

    logger.info(f"Indexing {len(md_files)} document(s)...")
    rag.index_documents(md_files)

    logger.info(f"Query: {query}")
    logger.info(f"Answer:\n{rag.query(query)}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-key", default="", help="OpenRouter API key")
    parser.add_argument("--docs-dir", default=DEFAULT_DOCS_DIR, help="directory of .md files to index")
    parser.add_argument("--query", default="What does this document discuss?")
    parser.add_argument("--test-vector", action="store_true")
    parser.add_argument("--test-llm", action="store_true")
    parser.add_argument("--test-doc", action="store_true")
    parser.add_argument("--test-rag", action="store_true")
    parser.add_argument("--test-all", action="store_true")
    args = parser.parse_args()

    if not (args.test_vector or args.test_llm or args.test_doc or args.test_rag):
        args.test_all = True

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")

    results = {}
    if args.test_vector or args.test_all:
        results["embedding model"] = test_vector_model()
    if args.test_llm or args.test_all:
        results["LLM"] = test_llm_model(api_key)
    if args.test_doc or args.test_all:
        results["document processing"] = test_document_processing(args.docs_dir)
    if args.test_rag or args.test_all:
        results["full pipeline"] = test_full_rag_system(api_key, args.docs_dir, args.query)

    logger.info("\nResults:")
    for name, ok in results.items():
        logger.info(f"  {name}: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
