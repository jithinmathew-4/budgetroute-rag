
from typing import Any

from budgetroute_rag.chunking import Chunk


def rerank_chunks(
    query: str,
    candidates: list[tuple[Chunk, float]],
    model: Any,
    top_k: int = 5,
) -> list[tuple[Chunk, float]]:
    """Rerank retrieved chunks using a cross-encoder."""

    if not query.strip():
        raise ValueError("Query must not be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if not candidates:
        return []

    pairs = [
        (query, chunk.text)
        for chunk, _ in candidates
    ]

    scores = model.predict(pairs)

    reranked = [
        (chunk, float(score))
        for (chunk, _), score in zip(candidates, scores)
    ]

    reranked.sort(key=lambda item: item[1], reverse=True)

    return reranked[:top_k]
