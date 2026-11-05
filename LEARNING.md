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
