import numpy as np
import pytest

from budgetroute_rag.chunking import Chunk
from budgetroute_rag.embeddings import generate_embeddings


class FakeEmbeddingModel:
    """A lightweight model for testing embedding generation."""

    def get_sentence_embedding_dimension(self):
        return 384

    def encode(
        self,
        texts,
        batch_size,
        show_progress_bar,
        convert_to_numpy,
        normalize_embeddings,
    ):
        assert batch_size == 32
        assert normalize_embeddings is True

        return np.ones((len(texts), 384), dtype=np.float32)


def test_generates_embeddings_for_chunks():
    chunks = [
        Chunk("test.rst", 0, "First test chunk", 3),
        Chunk("test.rst", 1, "Second test chunk", 3),
    ]

    model = FakeEmbeddingModel()

    embeddings = generate_embeddings(chunks, model)

    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32


def test_returns_empty_array_for_no_chunks():
    model = FakeEmbeddingModel()

    embeddings = generate_embeddings([], model)

    assert embeddings.shape == (0, 384)


def test_rejects_invalid_batch_size():
    model = FakeEmbeddingModel()

    with pytest.raises(ValueError):
        generate_embeddings([], model, batch_size=0)
