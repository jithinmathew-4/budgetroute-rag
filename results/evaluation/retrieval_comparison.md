# Retrieval Evaluation

## Dataset and Configuration

- Evaluation questions: 20
- Available chunks: 2,323
- Embedding model: BAAI/bge-small-en-v1.5
- Embedding dimensions: 384
- Retrieval top_k: 5
- Reranking candidate pool: 20

## Results

| Method | Precision@5 | Recall@5 | MRR |
|---|---:|---:|---:|
| Baseline BGE | 0.2000 | 0.8750 | 0.7742 |
| MMR (lambda=0.7) | 0.1700 | 0.7250 | 0.7417 |
| Cross-encoder L6 | 0.1500 | 0.6250 | 0.5250 |
| Cross-encoder L12 | 0.1700 | 0.7250 | 0.5558 |

## Models Tested

- L6: cross-encoder/ms-marco-MiniLM-L6-v2
- L12: cross-encoder/ms-marco-MiniLM-L-12-v2

Both cross-encoders reranked the top 20 candidates retrieved by BGE.

## Analysis

Baseline BGE similarity retrieval achieved the strongest overall
results across the tested configurations.

MMR did not improve retrieval performance over the baseline.

The L12 cross-encoder improved over L6, but neither reranker
outperformed baseline retrieval.

These experiments demonstrate that additional model complexity
does not necessarily improve retrieval quality.

## Limitations

The evaluation uses 20 manually labelled questions. Relevant
chunks may be missing from the ground-truth labels, so the metrics
should be treated as directional rather than definitive.

A larger evaluation dataset and further label review are needed
before making broader performance claims.

## Conclusion

Baseline BGE similarity retrieval remains the preferred default.

MMR and cross-encoder reranking are retained as experimental
approaches rather than enabled in the default retrieval pipeline.
