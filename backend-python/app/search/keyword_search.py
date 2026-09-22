"""Retrieve chunks with PostgreSQL's built-in full-text search.

Unlike vector search, keyword search rewards exact words and phrases. This is
especially useful for names, identifiers, abbreviations, and error codes.
"""

import psycopg
from langchain_core.documents import Document

from app.config import (
    COLLECTION_NAME,
    SEARCH_CANDIDATE_LIMIT,
    get_psycopg_database_url,
)
from app.search.models import SearchResult


# LangChain stores chunk text in ``langchain_pg_embedding.document`` and the
# original metadata in ``cmetadata``. Joining the collection table prevents a
# different LangChain collection from leaking into this application's results.
KEYWORD_SEARCH_SQL = """
WITH parsed_question AS (
    SELECT websearch_to_tsquery('english', %s) AS query
)
SELECT
    embedding.id,
    embedding.document,
    embedding.cmetadata,
    ts_rank_cd(
        to_tsvector('english', COALESCE(embedding.document, '')),
        parsed_question.query
    ) AS keyword_score
FROM langchain_pg_embedding AS embedding
JOIN langchain_pg_collection AS collection
    ON collection.uuid = embedding.collection_id
CROSS JOIN parsed_question
WHERE collection.name = %s
  AND to_tsvector('english', COALESCE(embedding.document, ''))
      @@ parsed_question.query
ORDER BY keyword_score DESC
LIMIT %s
"""


def search_by_keyword(question: str) -> list[SearchResult]:
    """Return chunks containing words that match the user's question.

    Parameters are passed separately from the SQL string. Psycopg safely sends
    them to PostgreSQL without building SQL through string concatenation.
    """

    with psycopg.connect(get_psycopg_database_url()) as connection:
        rows = connection.execute(
            KEYWORD_SEARCH_SQL,
            (question, COLLECTION_NAME, SEARCH_CANDIDATE_LIMIT),
        ).fetchall()

    return [
        SearchResult(
            document=Document(
                id=str(row[0]),
                page_content=row[1] or "",
                metadata=row[2] or {},
            ),
            keyword_score=float(row[3]),
        )
        for row in rows
    ]
