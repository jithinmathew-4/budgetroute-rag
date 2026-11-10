# BudgetRoute-RAG Learning Log

## Day 1: Project Initialization

### Concepts learned
- Git and version control
- Python virtual environments
- Python packages
- Editable installation
- Automated testing with pytest

### Implementation
- Created the project structure
- Initialized Git
- Created a Python virtual environment
- Installed the project with pip
- Created the first smoke test

### Test result
- 1 test passed

### Questions to revisit
- How does Python package importing work?
- How does pytest discover tests?
- How do dependencies work?

---

## Stage 1: Document Ingestion and Cleaning

### 1. Knowledge source

- Dataset: Scikit-learn documentation
- Format: RST (reStructuredText)
- Total documents loaded: 220
- First document: `about.rst`
- First document size: 6,613 characters

### 2. Document ingestion

Implemented `ingest.py` containing:

- `Document` dataclass
- `load_rst_documents()` function
- Recursive RST file discovery
- UTF-8 text reading
- Source metadata

### 3. Text cleaning

Implemented `cleaning.py` containing:

- `remove_rst_roles()` to remove inline RST role markup while preserving visible content.
- `normalize_whitespace()` to reduce unnecessary whitespace.
- Preservation of indentation in RST code blocks.
- `clean_rst_text()` to combine cleaning operations.

### 4. Important design decisions

- Keep ingestion and cleaning in separate modules.
- Preserve source information for traceability and future citations.
- Avoid removing meaningful technical content such as code and mathematical expressions.
- Treat the current cleaner as a first pass, not a complete RST parser.

### 5. Concepts learned

- Dataclasses
- `pathlib.Path`
- `rglob()`
- `read_text()`
- Regular expressions
- Type hints
- Unit testing
- Error handling
- Document metadata
- RST markup

### 6. Testing

Tested:
- Recursive file loading
- Text preservation
- Missing directory handling
- Non-RST file exclusion
- RST role removal
- Whitespace normalization
- Code indentation preservation

---

## Stage 2: Token-aware Document Chunking

### 1. Why chunking is needed

A document can contain thousands of tokens. Sending the entire document to an LLM for every question would waste context space and potentially increase API costs and latency.

Chunking divides documents into smaller pieces. Later, the retrieval system can select only the chunks relevant to a user's question.

**Important:** Chunking alone does not reduce the total number of tokens in the knowledge base. Retrieval reduces how much context we send to the LLM.

### 2. Implementation

Created `chunking.py` containing:

- `Chunk` dataclass
- `chunk_documents()` function
- Token-based document splitting
- Configurable chunk size and overlap
- Token counting using `tiktoken`

### 3. Chunk dataclass

Each chunk stores:

- `source`: Original document path.
- `chunk_index`: Position of the chunk within its source document.
- `text`: Actual chunk content.
- `token_count`: Number of tokens in the chunk.

Keeping the source and index allows us to trace retrieved content back to its original document.

### 4. Tokenizer

Installed and added `tiktoken` as a project dependency.

We use the `cl100k_base` encoding.

A tokenizer converts text into tokens, which are smaller units of text used by language models. Token counts are more useful than character counts when managing LLM context limits and costs.

### 5. Chunking configuration

Initial configuration:

| Parameter | Value |
|---|---:|
| Chunk size | 400 tokens |
| Overlap | 60 tokens |
| Step size | 340 tokens |
| Encoding | `cl100k_base` |

Step size is calculated as:

`chunk_size - overlap = 400 - 60 = 340`

Overlap means consecutive chunks share some tokens. This helps preserve context when an important sentence or explanation crosses a chunk boundary.

These are initial engineering choices, not proven optimal values. We will evaluate different settings later using retrieval quality.

### 6. Example

For a document containing 1,000 tokens:

- Chunk 1: tokens 0–399
- Chunk 2: tokens 340–739
- Chunk 3: tokens 680–999

The shared regions provide 60 tokens of overlap between consecutive chunks.

### 7. Real dataset results

After ingestion, cleaning and chunking:

| Measurement | Result |
|---|---:|
| Documents | 220 |
| Chunks created | 1,904 |
| Average chunk size | 371.78 tokens |
| Smallest chunk | 6 tokens |
| Largest chunk | 400 tokens |
| Chunks below 100 tokens | 101 |

### 8. Bug discovered and fixed

During dataset analysis, we found that the chunking loop could create an unnecessary final chunk.

For example, if a document contained 733 tokens:

- First chunk contained 400 tokens.
- Second chunk contained 393 tokens and already reached the document's end.
- A third chunk contained 53 tokens, entirely duplicated from the second chunk.

We fixed this by stopping the loop when the current chunk reaches the end of the document.

This reduced the total from 1,925 to 1,904 chunks.

Not every small chunk is a bug. Some documents are naturally short, and some final chunks contain useful new content.

### 9. Testing

Chunking tests cover:

- Short documents producing one chunk.
- Chunk size staying within the configured limit.
- Chunk indexes restarting for each document.
- Invalid chunk sizes being rejected.
- Invalid overlap values being rejected.

**Current complete test suite: 20 tests passed.**

### 10. Concepts learned

- Tokenization
- Token-aware chunking
- Chunk overlap
- Token counting
- Dataclasses for structured chunks
- Python loops and range stepping
- Input validation
- Edge-case testing
- Debugging with real datasets

---

## Upcoming Stages

1. Embeddings: Convert text chunks into numerical vectors.
2. Vector indexing: Store vectors for efficient similarity search.
3. Retrieval: Find chunks relevant to a user question.
4. Generation: Use retrieved context to produce grounded answers.
5. Cost tracking: Measure tokens, latency and model costs.
6. Evaluation: Measure answer quality.
7. ML routing: Predict the cheapest model tier likely to answer correctly.
8. Cascade system: Escalate to a stronger tier when necessary.

