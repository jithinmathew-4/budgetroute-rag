
import pytest

from budgetroute_rag.ingest import load_rst_documents


def test_loads_rst_files_recursively(tmp_path):
    # Create a temporary documentation directory.
    docs_dir = tmp_path / "docs"
    modules_dir = docs_dir / "modules"

    modules_dir.mkdir(parents=True)

    # Create two example RST files.
    (docs_dir / "guide.rst").write_text(
        "Machine learning guide",
        encoding="utf-8"
    )

    (modules_dir / "clustering.rst").write_text(
        "KMeans clustering documentation",
        encoding="utf-8"
    )

    documents = load_rst_documents(docs_dir)

    assert len(documents) == 2

    sources = {doc.source for doc in documents}

    assert "guide.rst" in sources
    assert "modules/clustering.rst" in sources


def test_preserves_original_document_text(tmp_path):
    content = """
    KMeans clustering
    =================

    KMeans groups data into clusters.
    """

    (tmp_path / "clustering.rst").write_text(
        content,
        encoding="utf-8"
    )

    documents = load_rst_documents(tmp_path)

    assert documents[0].text == content


def test_raises_error_for_missing_directory(tmp_path):
    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        load_rst_documents(missing_dir)


def test_ignores_non_rst_files(tmp_path):
    (tmp_path / "guide.rst").write_text(
        "Documentation",
        encoding="utf-8"
    )

    (tmp_path / "notes.txt").write_text(
        "This is not RST",
        encoding="utf-8"
    )

    documents = load_rst_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].source == "guide.rst"
