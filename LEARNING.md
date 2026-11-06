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
1 passed

### Questions to revisit
- How does Python package importing work?
- How does pytest discover tests?
- How do dependencies work?

## Stage 1: Document Ingestion

### Knowledge source
Scikit-learn documentation.

### Dataset details
- Format: RST (reStructuredText)
- Total documents loaded: 220
- First document: about.rst
- First document size: 6613 characters

### Implementation
Created ingest.py containing:
- Document dataclass
- load_rst_documents() function
- Recursive RST file discovery
- UTF-8 text reading
- Source metadata

### Testing
- Recursive file loading
- Text preservation
- Missing directory handling
- Non-RST file exclusion
- Total tests passed: 5

### Important concepts learned
- pathlib
- dataclasses
- rglob()
- read_text()
- Document metadata
- Unit testing
## Stage 1: Document Ingestion and Cleaning

### 1. Document ingestion

- Implemented `load_rst_documents()` to load RST files recursively.
- Used `pathlib.Path` to work with file paths.
- Used a `Document` dataclass to store the source path and document text.
- Preserved source metadata for future retrieval and citations.
- Added error handling for missing directories and invalid paths.

### 2. Text cleaning

- Implemented `remove_rst_roles()` to remove inline RST role markup while preserving visible content.
- Implemented `normalize_whitespace()` to reduce unnecessary whitespace.
- Preserved indentation in RST code blocks.
- Combined cleaning operations in `clean_rst_text()`.

### 3. Testing

- Used pytest to test ingestion and cleaning functions.
- Tested normal inputs, edge cases and expected errors.
- Current result: 15 tests passed.

### 4. Important design decisions

- Keep ingestion and cleaning in separate modules.
- Preserve source information for traceability.
- Avoid removing meaningful technical content such as code and mathematical expressions.
- Treat the current cleaner as a first pass, not a complete RST parser.

### 5. Concepts learned

- Dataclasses
- pathlib
- Regular expressions
- Type hints
- Unit testing
- Error handling
- Document metadata
- RST markup