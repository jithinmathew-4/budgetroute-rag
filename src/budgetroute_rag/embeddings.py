from typing import Any

import numpy as np

from budgetroute_rag.chunking import Chunk


def generate_embeddings(
    chunks: list[Chunk],
    model: Any,
    batch_size: int = 32,
) -> np.ndarray:
    """Generate normalized embeddings for document chunks."""

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    if not chunks:
        return np.empty(
            (0, model.get_sentence_embedding_dimension()),
            dtype=np.float32,
        )

    texts = [chunk.text for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return np.asarray(embeddings, dtype=np.float32)

QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)


def embed_query(query: str, model: Any) -> np.ndarray:
    """Convert a user query into a normalized BGE embedding."""

    if not query.strip():
        raise ValueError("Query must not be empty")

    embedding = model.encode(
        QUERY_INSTRUCTION + query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return np.asarray(embedding, dtype=np.float32)
