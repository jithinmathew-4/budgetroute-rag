import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from budgetroute_rag.chunking import Chunk
from budgetroute_rag.embeddings import embed_query
from budgetroute_rag.retrieval import search_chunks
from budgetroute_rag.mmr import search_chunks_mmr


MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUESTIONS_PATH = Path("data/evaluation/questions.json")
CHUNKS_PATH = Path("data/chunks/chunks.jsonl")
EMBEDDINGS_PATH = Path("data/index/embeddings.npy")

TOP_K = 5


def load_chunks():
    with CHUNKS_PATH.open(encoding="utf-8") as file:
        return [Chunk(**json.loads(line)) for line in file]


def load_questions():
    with QUESTIONS_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def calculate_metrics(results, relevant_chunks):
    retrieved = {
        (chunk.source, chunk.chunk_index)
        for chunk, _ in results
    }
    relevant = {
        (item["source"], item["chunk_index"])
        for item in relevant_chunks
    }
    relevant_retrieved = retrieved & relevant

    precision = len(relevant_retrieved) / len(results) if results else 0.0
    recall = len(relevant_retrieved) / len(relevant) if relevant else 0.0

    reciprocal_rank = 0.0
    for rank, (chunk, _) in enumerate(results, start=1):
        if (chunk.source, chunk.chunk_index) in relevant:
            reciprocal_rank = 1 / rank
            break

    return precision, recall, reciprocal_rank


def main():
    print("\nLoading evaluation dataset...")
    questions = load_questions()
    chunks = load_chunks()
    embeddings = np.load(EMBEDDINGS_PATH)

    print(f"Questions: {len(questions)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")

    print("\nLoading BGE model...")
    model = SentenceTransformer(MODEL_NAME)

    methods = {
        "Baseline": search_chunks,
        "MMR": search_chunks_mmr,
    }

    totals = {
        name: {"precision": 0.0, "recall": 0.0, "mrr": 0.0}
        for name in methods
    }

    print("\nEvaluating retrieval methods...")
    print("\n--- Individual Question Results ---")

    for index, question in enumerate(questions, start=1):
        query_embedding = embed_query(question["question"], model)
        question_results = {}

        for name, search_function in methods.items():
            results = search_function(
                query_embedding,
                embeddings,
                chunks,
                top_k=TOP_K,
            )

            precision, recall, mrr = calculate_metrics(
                results,
                question["relevant_chunks"],
            )

            question_results[name] = (precision, recall, mrr)
            if index in (5, 7, 14):
                relevant_keys = {
                    (item["source"], item["chunk_index"])
                    for item in question["relevant_chunks"]
        }

                print(f"\n{name} — Q{index}: {question['question']}")

                for rank, (chunk, score) in enumerate(results, start=1):
                    key = (chunk.source, chunk.chunk_index)
                    marker = "RELEVANT" if key in relevant_keys else ""
                    print(
                        f"  Rank {rank}: {chunk.source} | "
            f"chunk {chunk.chunk_index} | "
            f"score {score:.4f} {marker}"
        )
            totals[name]["precision"] += precision
            totals[name]["recall"] += recall
            totals[name]["mrr"] += mrr

        baseline = question_results["Baseline"]
        mmr = question_results["MMR"]

        print(
            f"Q{index:02d} | "
            f"Baseline P={baseline[0]:.2f} R={baseline[1]:.2f} "
            f"MRR={baseline[2]:.2f} | "
            f"MMR P={mmr[0]:.2f} R={mmr[1]:.2f} "
            f"MRR={mmr[2]:.2f}"
        )

    print("\n" + "=" * 60)
    print("             RETRIEVAL EVALUATION RESULTS")
    print("=" * 60)

    for name, scores in totals.items():
        count = len(questions)
        print(f"\n{name}")
        print("-" * 30)
        print(f"Precision@{TOP_K}: {scores['precision'] / count:.4f}")
        print(f"Recall@{TOP_K}:    {scores['recall'] / count:.4f}")
        print(f"MRR:               {scores['mrr'] / count:.4f}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
