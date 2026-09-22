"""Turn an uploaded PDF into searchable chunks in PGVector.

Ingestion is the preparation half of RAG:

1. Read the uploaded PDF.
2. Split its pages into smaller pieces.
3. Ask the embedding model to convert each piece into a vector.
4. Store the text, vector, and source details in PostgreSQL.

The vector conversion happens inside ``add_documents`` because the PGVector
store already has our Gemini embedding client attached to it.
"""

import tempfile
from pathlib import Path
from uuid import uuid4

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.vector_store import get_vector_store


def load_pdf(pdf_bytes: bytes, filename: str) -> list[Document]:
    """Read uploaded PDF bytes and return one LangChain document per page.

    ``PyPDFLoader`` expects a file path rather than raw bytes. We therefore
    create a temporary PDF, let the loader read it, and remove it afterward.
    The ``finally`` block runs even if PDF reading fails, preventing old upload
    files from collecting on the server.
    """

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as file:
            file.write(pdf_bytes)
            temporary_path = Path(file.name)

        documents = PyPDFLoader(str(temporary_path)).load()

        # The temporary filename is meaningless to the user. Replace it with
        # the original upload name so search results show a useful source.
        for document in documents:
            document.metadata["source"] = filename

        return documents
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()


def split_documents(documents: list[Document]) -> list[Document]:
    """Split PDF pages into smaller, partly overlapping text chunks.

    Smaller chunks make retrieval more focused. The overlap repeats a little
    text between neighboring chunks so a sentence near a boundary does not
    lose its surrounding meaning.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return splitter.split_documents(documents)


def ingest_pdf(pdf_bytes: bytes, filename: str) -> int:
    """Run the complete ingestion flow and return the stored chunk count."""

    pages = load_pdf(pdf_bytes, filename)
    chunks = split_documents(pages)

    # One document ID groups every chunk created from this upload. Each chunk
    # also receives its own ID, which lets vector search and keyword search
    # recognize the same result when we combine their rankings.
    document_id = str(uuid4())
    chunk_ids: list[str] = []

    for chunk_number, chunk in enumerate(chunks):
        chunk_id = str(uuid4())
        chunk_ids.append(chunk_id)

        chunk.metadata["document_id"] = document_id
        chunk.metadata["chunk_id"] = chunk_id
        chunk.metadata["chunk_number"] = chunk_number

    # PGVector creates embeddings for the chunk text and stores the resulting
    # vectors, text, and metadata in PostgreSQL. Passing our IDs also makes the
    # underlying database row use the same stable chunk identity.
    get_vector_store().add_documents(chunks, ids=chunk_ids)

    return len(chunks)
