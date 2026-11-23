
import numpy as np
import pytest

from budgetroute_rag.chunking import Chunk
from budgetroute_rag.mmr import search_chunks_mmr


def test_mmr_selects_diverse_chunks():
    chunks = [
        Chunk("a.rst", 0, "First relevant chunk", 3),
        Chunk("b.rst", 0, "Almost duplicate chunk", 3),
        Chunk("c.rst", 0, "Different relevant chunk", 3),
    ]

    embeddings = np.array([
        [0.90, 0.43589],
        [0.89, 0.45607],
        [0.80, -0.60],
    ], dtype=np.float32)

    query = np.array([1.0, 0.0], dtype=np.float32)

    results = search_chunks_mmr(
        query,
        embeddings,
        chunks,
        top_k=2,
    )

    selected_texts = [chunk.text for chunk, score in results]

    assert selected_texts[0] == "First relevant chunk"
    assert selected_texts[1] == "Different relevant chunk"


def test_mmr_returns_empty_for_no_chunks():
    results = search_chunks_mmr(
        np.array([1.0, 0.0]),
        np.empty((0, 2)),
        [],
    )

    assert results == []


def test_mmr_rejects_invalid_top_k():
    with pytest.raises(ValueError):
        search_chunks_mmr(
            np.array([1.0, 0.0]),
            np.array([[1.0, 0.0]]),
            [Chunk("a.rst", 0, "Test", 1)],
            top_k=0,
        )


def test_mmr_rejects_invalid_lambda():
    with pytest.raises(ValueError):
        search_chunks_mmr(
            np.array([1.0, 0.0]),
            np.array([[1.0, 0.0]]),
            [Chunk("a.rst", 0, "Test", 1)],
            lambda_param=1.5,
        )


def test_mmr_rejects_mismatched_dimensions():
    with pytest.raises(ValueError):
        search_chunks_mmr(
            np.array([1.0, 0.0]),
            np.array([[1.0, 0.0, 0.0]]),
            [Chunk("a.rst", 0, "Test", 1)],
        )


def test_mmr_rejects_mismatched_chunk_count():
    with pytest.raises(ValueError):
        search_chunks_mmr(
            np.array([1.0, 0.0]),
            np.array([[1.0, 0.0]]),
            [],
        )

