import numpy as np

from budgetroute_rag.chunking import Chunk


def search_chunks(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    chunks: list[Chunk],
    top_k: int = 5,
) -> list[tuple[Chunk, float]]:
    """Return the most similar chunks to a query embedding."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D matrix")

    if query_embedding.ndim != 1:
        raise ValueError("query_embedding must be a 1D vector")

    if embeddings.shape[1] != query_embedding.shape[0]:
        raise ValueError("Query and document dimensions must match")

    if len(embeddings) != len(chunks):
        raise ValueError("Number of embeddings must match number of chunks")

    if len(chunks) == 0:
        return []

    # Normalized vectors make dot product equivalent to cosine similarity.
    scores = embeddings @ query_embedding

    # Get indices of the highest-scoring chunks.
    best_indices = np.argsort(scores)[::-1][:top_k]

    return [
        (chunks[int(index)], float(scores[index]))
        for index in best_indices
    ]
