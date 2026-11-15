import re
import pytest

from budgetroute_rag.ingest import Document
from budgetroute_rag.chunking import chunk_documents



class FakeTokenizer:
    """A lightweight tokenizer for unit testing."""

    def encode(self, text, add_special_tokens=False):
        tokens = text.split()

        if add_special_tokens:
            return [101] + tokens + [102]

        return tokens

    def __call__(
        self,
        text,
        add_special_tokens=False,
        return_offsets_mapping=False,
    ):
        matches = list(re.finditer(r"\S+", text))

        result = {
            "input_ids": list(range(len(matches))),
        }

        if return_offsets_mapping:
            result["offset_mapping"] = [
                match.span() for match in matches
            ]

        return result

class WordPieceFakeTokenizer:
    """Simulates a tokenizer that splits one word into subword tokens."""

    def _get_offsets(self, text):
        offsets = []

        for match in re.finditer(r"\S+", text):
            word = match.group()

            if word == "beta":
                # Simulate beta being split into be + ta.
                offsets.append((match.start(), match.start() + 2))
                offsets.append((match.start() + 2, match.end()))
            else:
                offsets.append(match.span())

        return offsets

    def encode(self, text, add_special_tokens=False):
        tokens = self._get_offsets(text)
        token_ids = list(range(len(tokens)))

        if add_special_tokens:
            return [101] + token_ids + [102]

        return token_ids

    def __call__(
        self,
        text,
        add_special_tokens=False,
        return_offsets_mapping=False,
    ):
        offsets = self._get_offsets(text)

        result = {"input_ids": list(range(len(offsets)))}

        if return_offsets_mapping:
            result["offset_mapping"] = offsets

        return result

@pytest.fixture
def tokenizer():
    return FakeTokenizer()


def test_short_document_creates_one_chunk(tokenizer):
    document = Document(
        source="guide.rst",
        text="Machine learning is interesting.",
    )

    chunks = chunk_documents([document], tokenizer)

    assert len(chunks) == 1
    assert chunks[0].source == "guide.rst"
    assert chunks[0].chunk_index == 0


def test_chunk_size_does_not_exceed_limit(tokenizer):
    document = Document(
        source="guide.rst",
        text="machine " * 300,
    )

    chunks = chunk_documents(
        [document],
        tokenizer,
        chunk_size=100,
        overlap=20,
    )

    for chunk in chunks:
        assert chunk.token_count <= 100


def test_overlap_is_preserved(tokenizer):
    document = Document(
        source="guide.rst",
        text=" ".join(f"word{i}" for i in range(10)),
    )

    chunks = chunk_documents(
        [document],
        tokenizer,
        chunk_size=6,
        overlap=2,
    )

    first_tokens = chunks[0].text.split()
    second_tokens = chunks[1].text.split()

    assert first_tokens[-2:] == second_tokens[:2]


def test_chunk_indexes_restart_for_each_document(tokenizer):
    documents = [
        Document(source="first.rst", text="first " * 300),
        Document(source="second.rst", text="second " * 300),
    ]

    chunks = chunk_documents(documents, tokenizer)

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


def test_rejects_invalid_chunk_size(tokenizer):
    document = Document(source="guide.rst", text="Some text")

    with pytest.raises(ValueError):
        chunk_documents(
            [document],
            tokenizer,
            chunk_size=0,
        )


def test_rejects_invalid_overlap(tokenizer):
    document = Document(source="guide.rst", text="Some text")

    with pytest.raises(ValueError):
        chunk_documents(
            [document],
            tokenizer,
            chunk_size=100,
            overlap=100,
        )


def test_rejects_impossible_model_limit(tokenizer):
    document = Document(
        source="guide.rst",
        text="word " * 10,
    )

    

    with pytest.raises(ValueError):
        chunk_documents(
            [document],
            tokenizer,
            chunk_size=10,
            overlap=0,
            max_seq_length=2,
        )

def test_preserves_original_text_formatting(tokenizer):
    original_text = "Compare cv_splitters and fold-to-fold results."

    document = Document(
        source="guide.rst",
        text=original_text,
    )

    chunks = chunk_documents(
        [document],
        tokenizer,
        chunk_size=100,
        overlap=0,
    )

    assert len(chunks) == 1
    assert chunks[0].text == original_text
def test_does_not_split_words_at_subword_boundaries():
    tokenizer = WordPieceFakeTokenizer()

    document = Document(
        source="wordpiece_test.rst",
        text="alpha beta gamma delta",
    )

    chunks = chunk_documents(
        [document],
        tokenizer,
        chunk_size=2,
        overlap=0,
    )

    chunk_texts = [chunk.text for chunk in chunks]

    assert chunk_texts == [
        "alpha",
        "beta",
        "gamma delta",
    ]
