
from dataclasses import dataclass
from typing import Any

from budgetroute_rag.ingest import Document


@dataclass(frozen=True)
class Chunk:
    """Represents one chunk of a source document."""

    source: str
    chunk_index: int
    text: str
    token_count: int


def _align_end_to_word_boundary(
    text: str,
    offsets: list[tuple[int, int]],
    start: int,
    end: int,
) -> int:
    """Move the chunk end backwards to a whitespace boundary."""

    if end >= len(offsets):
        return end

    for candidate in range(end, start, -1):
        previous_end = offsets[candidate - 1][1]
        next_start = offsets[candidate][0]

        if text[previous_end:next_start].isspace():
            return candidate

    return end


def _align_start_to_word_boundary(
    text: str,
    offsets: list[tuple[int, int]],
    start: int,
    minimum_start: int,
    maximum_start: int,
) -> int:
    """Move the chunk start forward to a whitespace boundary."""

    candidate = max(start, minimum_start)

    while candidate < maximum_start:
        previous_end = offsets[candidate - 1][1]
        current_start = offsets[candidate][0]

        if text[previous_end:current_start].isspace():
            return candidate

        candidate += 1

    return candidate


def chunk_documents(
    documents: list[Document],
    tokenizer: Any,
    chunk_size: int = 400,
    overlap: int = 60,
    max_seq_length: int = 512,
) -> list[Chunk]:
    """Split documents into overlapping, token-limited chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    if max_seq_length <= 0:
        raise ValueError("max_seq_length must be greater than zero")

    # Calculate special-token overhead once.
    special_tokens = (
        len(tokenizer.encode("", add_special_tokens=True))
        - len(tokenizer.encode("", add_special_tokens=False))
    )

    max_content_tokens = min(
        chunk_size,
        max_seq_length - special_tokens,
    )

    if max_content_tokens <= 0:
        raise ValueError(
            "Model sequence limit is too small for special tokens"
        )

    chunks = []

    for document in documents:
        result = tokenizer(
            document.text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        offsets = result["offset_mapping"]

        start = 0
        chunk_index = 0

        while start < len(offsets):

            end = min(
                start + max_content_tokens,
                len(offsets),
            )

            end = _align_end_to_word_boundary(
                document.text,
                offsets,
                start,
                end,
            )

            # Prevent empty chunks.
            if end <= start:
                end = min(
                    start + max_content_tokens,
                    len(offsets),
                )

            start_char = offsets[start][0]
            end_char = offsets[end - 1][1]

            chunk_text = document.text[start_char:end_char]

            chunks.append(
                Chunk(
                    source=document.source,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    token_count=end - start,
                )
            )

            if end >= len(offsets):
                break

            previous_start = start

            next_start = max(
                end - overlap,
                previous_start + 1,
            )

            next_start = _align_start_to_word_boundary(
                document.text,
                offsets,
                next_start,
                previous_start + 1,
                end,
            )

            # Guarantee forward progress.
            start = max(
                next_start,
                previous_start + 1,
            )

            chunk_index += 1

    return chunks
