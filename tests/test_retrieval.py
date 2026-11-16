import numpy as np
import pytest

from budgetroute_rag.chunking import Chunk
from budgetroute_rag.retrieval import search_chunks


def test_returns_most_similar_chunks_first():
    chunks = [
        Chunk("a.rst", 0, "Unrelated content", 3),
        Chunk("b.rst", 0, "Relevant content", 3),
        Chunk("c.rst", 0, "Somewhat relevant", 3),
    ]

    embeddings = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.6, 0.8, 0.0],
    ], dtype=np.float32)

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    results = search_chunks(query, embeddings, chunks, top_k=2)

    assert results[0][0].text == "Relevant content"
    assert results[1][0].text == "Somewhat relevant"
    assert results[0][1] > results[1][1]


def test_returns_empty_for_no_chunks():
    results = search_chunks(
        np.array([1.0, 0.0]),
        np.empty((0, 2)),
        [],
    )

    assert results == []


def test_rejects_invalid_top_k():
    with pytest.raises(ValueError):
        search_chunks(
            np.array([1.0, 0.0]),
            np.array([[1.0, 0.0]]),
            [Chunk("a.rst", 0, "Test", 1)],
            top_k=0,
        )


def test_rejects_mismatched_dimensions():
    with pytest.raises(ValueError):
        search_chunks(
            np.array([1.0, 0.0]),
            np.array([[1.0, 0.0, 0.0]]),
            [Chunk("a.rst", 0, "Test", 1)],
        )


def test_rejects_mismatched_chunk_count():
    with pytest.raises(ValueError):
        search_chunks(
            np.array([1.0, 0.0]),
            np.array([[1.0, 0.0]]),
            [],
        )
