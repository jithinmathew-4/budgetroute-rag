
from dataclasses import dataclass

import tiktoken

from budgetroute_rag.ingest import Document


@dataclass(frozen=True)
class Chunk:
    """Represents one chunk of a source document."""

    source: str
    chunk_index: int
    text: str
    token_count: int


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 400,
    overlap: int = 60,
    encoding_name: str = "cl100k_base",
) -> list[Chunk]:
    """
    Split documents into token-based chunks with overlap.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    encoding = tiktoken.get_encoding(encoding_name)

    chunks = []
    step = chunk_size - overlap

    for document in documents:

        token_ids = encoding.encode(document.text)

        for chunk_index, start in enumerate(
    range(0, len(token_ids), step)
):

            end = min(start + chunk_size, len(token_ids))

            chunk_tokens = token_ids[start:end]

            chunk_text = encoding.decode(chunk_tokens)

            chunk = Chunk(
                source=document.source,
                chunk_index=chunk_index,
                text=chunk_text,
                token_count=len(chunk_tokens),
            )

            chunks.append(chunk)
                        # Stop if this chunk already reaches the end of the document.
            if end == len(token_ids):
                break

    return chunks
