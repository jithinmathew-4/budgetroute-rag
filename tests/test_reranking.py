
import numpy as np
import pytest

from budgetroute_rag.chunking import Chunk
from budgetroute_rag.reranking import rerank_chunks


class FakeCrossEncoder:
    def predict(self, pairs):
        return np.array([3.5, -2.0, 7.0])


@pytest.fixture
def candidates():
    return [
        (Chunk("tree.rst", 0, "Decision trees classify data.", 5), 0.85),
        (Chunk("python.rst", 0, "Python is a programming language.", 6), 0.70),
        (Chunk("forest.rst", 0, "Random forests combine trees.", 5), 0.80),
    ]


def test_reranks_by_cross_encoder_score(candidates):
    results = rerank_chunks(
        "How do decision trees work?",
        candidates,
        FakeCrossEncoder(),
    )

    assert [chunk.source for chunk, _ in results] == [
        "forest.rst",
        "tree.rst",
        "python.rst",
    ]


def test_returns_only_top_k(candidates):
    results = rerank_chunks(
        "How do decision trees work?",
        candidates,
        FakeCrossEncoder(),
        top_k=2,
    )

    assert len(results) == 2


def test_returns_empty_for_no_candidates():
    results = rerank_chunks(
        "What is machine learning?",
        [],
        FakeCrossEncoder(),
    )

    assert results == []


def test_rejects_empty_query(candidates):
    with pytest.raises(ValueError):
        rerank_chunks("", candidates, FakeCrossEncoder())


def test_rejects_invalid_top_k(candidates):
    with pytest.raises(ValueError):
        rerank_chunks(
            "What is machine learning?",
            candidates,
            FakeCrossEncoder(),
            top_k=0,
        )
