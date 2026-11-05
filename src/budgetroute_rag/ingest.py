
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    """Represents one source document."""

    source: str
    text: str


def load_rst_documents(source_dir: str | Path) -> list[Document]:
    """
    Load all reStructuredText files from a directory recursively.

    Args:
        source_dir: Directory containing RST documentation.

    Returns:
        A list of Document objects.

    Raises:
        FileNotFoundError: If the source directory does not exist.
        NotADirectoryError: If the supplied path is not a directory.
    """

    root = Path(source_dir)

    if not root.exists():
        raise FileNotFoundError(
            f"Source directory does not exist: {root}"
        )

    if not root.is_dir():
        raise NotADirectoryError(
            f"Expected a directory, received: {root}"
        )

    documents = []

    rst_files = sorted(root.rglob("*.rst"))

    for file_path in rst_files:
        if not file_path.is_file():
            continue

        text = file_path.read_text(encoding="utf-8")

        relative_source = file_path.relative_to(root).as_posix()

        document = Document(
            source=relative_source,
            text=text
        )

        documents.append(document)

    return documents
