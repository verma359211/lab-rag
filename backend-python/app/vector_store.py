"""Connect LangChain to the PGVector database collection.

PGVector is PostgreSQL with support for vector columns and vector similarity
search. LangChain's PGVector class hides the low-level insert and search SQL,
which keeps this demonstration small and readable.
"""

from functools import lru_cache

from langchain_postgres import PGVector

from app.config import COLLECTION_NAME, get_database_url
from app.embeddings import get_embeddings


@lru_cache(maxsize=1)
def get_vector_store() -> PGVector:
    """Create one reusable connection to our LangChain vector collection.

    ``use_jsonb=True`` tells LangChain to store metadata such as the PDF name
    and page number as PostgreSQL JSONB data. That metadata is returned with
    search results and later shown to the frontend as a source.
    """

    return PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=get_database_url(),
        use_jsonb=True,
    )
