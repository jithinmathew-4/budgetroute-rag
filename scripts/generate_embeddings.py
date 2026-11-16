import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from budgetroute_rag.ingest import load_rst_documents
from budgetroute_rag.cleaning import clean_rst_text
from budgetroute_rag.chunking import chunk_documents
from budgetroute_rag.embeddings import generate_embeddings


# Configuration
MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 400
OVERLAP = 60
BATCH_SIZE = 32

DOCUMENT_PATH = Path("data/raw/scikit-learn-source/doc")
CHUNK_PATH = Path("data/chunks/chunks.jsonl")
EMBEDDING_PATH = Path("data/index/embeddings.npy")


def main():

    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("\nLoading documents...")
    documents = load_rst_documents(DOCUMENT_PATH)
    print(f"Documents loaded: {len(documents)}")

    print("\nCleaning documents...")
    cleaned_documents = [
        type(doc)(
            source=doc.source,
            text=clean_rst_text(doc.text),
        )
        for doc in documents
    ]

    print("\nGenerating chunks...")
    chunks = chunk_documents(
        cleaned_documents,
        model.tokenizer,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
        max_seq_length=model.max_seq_length,
    )

    print(f"Chunks generated: {len(chunks)}")

    print("\nGenerating embeddings...")
    embeddings = generate_embeddings(
        chunks,
        model,
        batch_size=BATCH_SIZE,
    )

    print("\nSaving files...")

    CHUNK_PATH.parent.mkdir(parents=True, exist_ok=True)
    EMBEDDING_PATH.parent.mkdir(parents=True, exist_ok=True)

    with CHUNK_PATH.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(asdict(chunk), ensure_ascii=False) + "\n"
            )

    np.save(EMBEDDING_PATH, embeddings)

    print("\n" + "=" * 45)
    print("       EMBEDDING GENERATION COMPLETE")
    print("=" * 45)

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"Embedding data type: {embeddings.dtype}")
    print(f"Chunk metadata: {CHUNK_PATH}")
    print(f"Embeddings: {EMBEDDING_PATH}")


if __name__ == "__main__":
    main()

