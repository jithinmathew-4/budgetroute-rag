import pytest
import tiktoken

from budgetroute_rag.ingest import Document
from budgetroute_rag.chunking import chunk_documents


def test_short_document_creates_one_chunk():
    document = Document(
        source="guide.rst",
        text="Machine learning is interesting."
    )

    chunks = chunk_documents([document])

    assert len(chunks) == 1
    assert chunks[0].source == "guide.rst"
    assert chunks[0].chunk_index == 0


def test_chunk_size_does_not_exceed_limit():
    document = Document(
        source="guide.rst",
        text="machine learning " * 300
    )

    chunks = chunk_documents(
        [document],
        chunk_size=100,
        overlap=20
    )

    encoding = tiktoken.get_encoding("cl100k_base")

    for chunk in chunks:
        assert len(encoding.encode(chunk.text)) <= 100


def test_chunk_indexes_restart_for_each_document():
    documents = [
        Document(source="first.rst", text="machine learning " * 300),
        Document(source="second.rst", text="data science " * 300),
    ]

    chunks = chunk_documents(documents)

    first_indexes = [
        chunk.chunk_index for chunk in chunks
        if chunk.source == "first.rst"
    ]

    second_indexes = [
        chunk.chunk_index for chunk in chunks
        if chunk.source == "second.rst"
    ]

    assert first_indexes[0] == 0
    assert second_indexes[0] == 0


def test_rejects_invalid_chunk_size():
    document = Document(source="guide.rst", text="Some text")

    with pytest.raises(ValueError):
        chunk_documents([document], chunk_size=0)


def test_rejects_invalid_overlap():
    document = Document(source="guide.rst", text="Some text")

    with pytest.raises(ValueError):
        chunk_documents([document], chunk_size=100, overlap=100)