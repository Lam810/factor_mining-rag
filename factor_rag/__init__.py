"""factor_rag: structure-preserving RAG for visually rich documents.

The parts that make this package worth reusing -- chunking, fidelity
metrics, and visualization -- have no dependency beyond ``tqdm`` and import
cleanly without a GPU, an API key, or a vector database. The retrieval and
generation classes (``RAGSystem`` et al.) need the ``serve`` extra
(``pip install "factor-rag[serve]"``) and a GPU for the default embedder.

Author: Zeteng Lin (Hong Kong University of Science and Technology, Guangzhou)
"""

from .chunking import Block, Chunk, chunk_markdown, naive_chunk, parse_blocks
from .metrics import FidelityReport, score_chunks

__all__ = [
    "Block",
    "Chunk",
    "chunk_markdown",
    "naive_chunk",
    "parse_blocks",
    "FidelityReport",
    "score_chunks",
]

__version__ = "0.1.0"
