
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from budgetroute_rag.chunking import Chunk
from budgetroute_rag.embeddings import embed_query
from budgetroute_rag.retrieval import search_chunks
from budgetroute_rag.mmr import search_chunks_mmr
from budgetroute_rag.reranking import rerank_chunks


MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"

QUESTIONS_PATH = Path("data/evaluation/questions.json")
CHUNKS_PATH = Path("data/chunks/chunks.jsonl")
EMBEDDINGS_PATH = Path("data/index/embeddings.npy")

TOP_K = 5
CANDIDATE_K = 20


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
    embedding_model = SentenceTransformer(MODEL_NAME)

    print("\nLoading cross-encoder reranker...")
    reranker_model = CrossEncoder(RERANKER_NAME)

    totals = {
        name: {"precision": 0.0, "recall": 0.0, "mrr": 0.0}
        for name in ("Baseline", "MMR", "Reranking")
    }

    print("\nEvaluating retrieval methods...")

    for index, question in enumerate(questions, start=1):
        query = question["question"]
        query_embedding = embed_query(query, embedding_model)

        # Method 1: Baseline BGE retrieval
        baseline_results = search_chunks(
            query_embedding,
            embeddings,
            chunks,
            top_k=TOP_K,
        )

        # Method 2: MMR retrieval
        mmr_results = search_chunks_mmr(
            query_embedding,
            embeddings,
            chunks,
            top_k=TOP_K,
        )

        # Method 3: BGE retrieves candidates, cross-encoder reranks them
        candidates = search_chunks(
            query_embedding,
            embeddings,
            chunks,
            top_k=CANDIDATE_K,
        )

        reranked_results = rerank_chunks(
            query,
            candidates,
            reranker_model,
            top_k=TOP_K,
        )

        methods = {
            "Baseline": baseline_results,
            "MMR": mmr_results,
            "Reranking": reranked_results,
        }

        question_scores = {}

        for name, results in methods.items():
            precision, recall, mrr = calculate_metrics(
                results,
                question["relevant_chunks"],
            )

            question_scores[name] = (precision, recall, mrr)

            totals[name]["precision"] += precision
            totals[name]["recall"] += recall
            totals[name]["mrr"] += mrr

        print(
            f"Q{index:02d} | "
            f"Baseline P={question_scores['Baseline'][0]:.2f} "
            f"R={question_scores['Baseline'][1]:.2f} "
            f"MRR={question_scores['Baseline'][2]:.2f} | "
            f"MMR P={question_scores['MMR'][0]:.2f} "
            f"R={question_scores['MMR'][1]:.2f} "
            f"MRR={question_scores['MMR'][2]:.2f} | "
            f"Reranking P={question_scores['Reranking'][0]:.2f} "
            f"R={question_scores['Reranking'][1]:.2f} "
            f"MRR={question_scores['Reranking'][2]:.2f}"
        )

    print("\n" + "=" * 65)
    print("                  RETRIEVAL EVALUATION")
    print("=" * 65)

    for name, scores in totals.items():
        count = len(questions)

        print(f"\n{name}")
        print("-" * 35)
        print(f"Precision@{TOP_K}: {scores['precision'] / count:.4f}")
        print(f"Recall@{TOP_K}:    {scores['recall'] / count:.4f}")
        print(f"MRR:               {scores['mrr'] / count:.4f}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
