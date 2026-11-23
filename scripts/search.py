
from budgetroute_rag.mmr import search_chunks_mmr
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from budgetroute_rag.chunking import Chunk
from budgetroute_rag.embeddings import embed_query
from budgetroute_rag.retrieval import search_chunks


MODEL_NAME = "BAAI/bge-small-en-v1.5"

CHUNK_PATH = Path("data/chunks/chunks.jsonl")
EMBEDDING_PATH = Path("data/index/embeddings.npy")


def main():
    print("Loading BGE model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading stored chunks...")
    with CHUNK_PATH.open("r", encoding="utf-8") as file:
        chunks = [
            Chunk(**json.loads(line))
            for line in file
        ]

    print("Loading document embeddings...")
    embeddings = np.load(EMBEDDING_PATH)

    query = input("\nEnter your question: ").strip()

    print("\nGenerating query embedding...")
    query_embedding = embed_query(query, model)

    print("Searching relevant chunks...")

    results = search_chunks_mmr(
    query_embedding,
    embeddings,
    chunks,
    top_k=5,
    lambda_param=0.7,
)

    print("\n" + "=" * 55)
    print("             TOP 5 SEARCH RESULTS")
    print("=" * 55)

    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"Similarity: {score:.4f}")
        print(f"Source: {chunk.source}")
        print(f"Chunk index: {chunk.chunk_index}")
        print("-" * 55)
        print(chunk.text[:700])
        print("-" * 55)


if __name__ == "__main__":
    main()

