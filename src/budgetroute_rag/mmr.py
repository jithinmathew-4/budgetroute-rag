
import numpy as np

from budgetroute_rag.chunking import Chunk


def search_chunks_mmr(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    chunks: list[Chunk],
    top_k: int = 5,
    lambda_param: float = 0.7,
    candidate_k: int = 50,
) -> list[tuple[Chunk, float]]:
    """Retrieve relevant and diverse chunks using MMR."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if candidate_k <= 0:
        raise ValueError("candidate_k must be greater than zero")

    if not 0 <= lambda_param <= 1:
        raise ValueError("lambda_param must be between 0 and 1")

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

    # Stage 1: Find the most relevant candidate chunks.
    relevance_scores = embeddings @ query_embedding

    candidate_count = min(candidate_k, len(chunks))

    candidate_indices = np.argsort(relevance_scores)[::-1][:candidate_count]

    candidate_embeddings = embeddings[candidate_indices]
    candidate_scores = relevance_scores[candidate_indices]

    # Stage 2: Calculate similarity only between candidates.
    similarity_matrix = candidate_embeddings @ candidate_embeddings.T

    selected_positions = []
    remaining_positions = list(range(candidate_count))

    # Stage 3: Select diverse chunks using MMR.
    while remaining_positions and len(selected_positions) < top_k:

        mmr_scores = []

        for position in remaining_positions:

            if selected_positions:
                redundancy = max(
                    similarity_matrix[position, selected]
                    for selected in selected_positions
                )
            else:
                redundancy = 0.0

            score = (
                lambda_param * candidate_scores[position]
                - (1 - lambda_param) * redundancy
            )

            mmr_scores.append((position, score))

        best_position, _ = max(
            mmr_scores,
            key=lambda item: item[1],
        )

        selected_positions.append(best_position)
        remaining_positions.remove(best_position)

    return [
        (
            chunks[int(candidate_indices[position])],
            float(candidate_scores[position]),
        )
        for position in selected_positions
    ]

