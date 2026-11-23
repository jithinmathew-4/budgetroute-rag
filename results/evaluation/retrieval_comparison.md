# Retrieval Evaluation

## Dataset
- Evaluation questions: 20
- Available chunks: 2,323
- Embedding model: BAAI/bge-small-en-v1.5
- Embedding dimensions: 384
- Retrieval top_k: 5

## Results

| Method | Precision@5 | Recall@5 | MRR |
|---|---:|---:|---:|
| Baseline | 0.2000 | 0.8750 | 0.7742 |
| MMR (lambda=0.7) | 0.1700 | 0.7250 | 0.7417 |
| MMR (lambda=0.9) | 0.1900 | 0.8250 | 0.7617 |
| MMR (lambda=1.0) | 0.2000 | 0.8750 | 0.7742 |

## Conclusion

Baseline similarity retrieval performed best among the tested
configurations.

Increasing MMR lambda improved retrieval performance, but did
not outperform baseline.

MMR is retained as an experimental retrieval strategy.
Baseline retrieval remains the preferred default.

## Validation
- Test suite: 38 passed
