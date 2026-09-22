"""List, inspect, and delete stored chunks by their source filename.

This is a temporary compatibility layer for documents ingested before we
started assigning ``document_id`` values. Every old chunk still has a
``source`` metadata value containing its filename, so grouping by that value
lets the learning UI manage the existing data.

Important limitation: two different uploads with the same filename are treated
as one document. After the legacy chunks are removed, this module should group
and delete by ``document_id`` instead.
"""

import psycopg

from app.config import COLLECTION_NAME, get_psycopg_database_url


# Count all rows sharing the same source. The page count uses distinct page
# numbers because one PDF page can be split into several vector chunks.
LIST_DOCUMENTS_SQL = """
SELECT
    embedding.cmetadata ->> 'source' AS source,
    COUNT(*) AS chunk_count,
    COUNT(DISTINCT embedding.cmetadata ->> 'page') AS page_count
FROM langchain_pg_embedding AS embedding
JOIN langchain_pg_collection AS collection
    ON collection.uuid = embedding.collection_id
WHERE collection.name = %s
  AND COALESCE(embedding.cmetadata ->> 'source', '') <> ''
GROUP BY embedding.cmetadata ->> 'source'
ORDER BY LOWER(embedding.cmetadata ->> 'source')
"""


# Chunk text is stored in the document column. Metadata contains values such as
# the page, chunk number, and IDs added during ingestion.
GET_CHUNKS_SQL = """
SELECT
    embedding.id,
    embedding.document,
    embedding.cmetadata
FROM langchain_pg_embedding AS embedding
JOIN langchain_pg_collection AS collection
    ON collection.uuid = embedding.collection_id
WHERE collection.name = %s
  AND embedding.cmetadata ->> 'source' = %s
"""


# Deletion is limited by both collection name and source. The collection check
# ensures another LangChain application using the database is not affected.
DELETE_DOCUMENT_SQL = """
DELETE FROM langchain_pg_embedding AS embedding
USING langchain_pg_collection AS collection
WHERE collection.uuid = embedding.collection_id
  AND collection.name = %s
  AND embedding.cmetadata ->> 'source' = %s
RETURNING embedding.id
"""


def list_documents() -> list[dict]:
    """Return one summary for each source filename in the vector collection.

    PostgreSQL performs the grouping, so the API receives only one small row
    per filename instead of loading every chunk into Python.
    """

    with psycopg.connect(get_psycopg_database_url()) as connection:
        rows = connection.execute(
            LIST_DOCUMENTS_SQL,
            (COLLECTION_NAME,),
        ).fetchall()

    return [
        {
            "source": row[0],
            "chunkCount": row[1],
            "pageCount": row[2],
        }
        for row in rows
    ]


def get_document_chunks(source: str) -> list[dict]:
    """Return stored text and metadata for every chunk with one source name."""

    with psycopg.connect(get_psycopg_database_url()) as connection:
        rows = connection.execute(
            GET_CHUNKS_SQL,
            (COLLECTION_NAME, source),
        ).fetchall()

    chunks = [
        {
            "id": str(row[0]),
            "content": row[1] or "",
            "metadata": row[2] or {},
        }
        for row in rows
    ]

    # Old chunks do not have chunk_number, but they do have a page. The default
    # values keep sorting predictable for both old and newly ingested data.
    return sorted(
        chunks,
        key=lambda chunk: (
            chunk["metadata"].get("page", 0),
            chunk["metadata"].get("chunk_number", 0),
            chunk["id"],
        ),
    )


def delete_document(source: str) -> int:
    """Delete all chunk rows sharing a source and return the deleted count.

    PostgreSQL commits the deletion when the connection context exits without
    an error. If an error occurs, psycopg rolls the transaction back instead.
    """

    with psycopg.connect(get_psycopg_database_url()) as connection:
        deleted_rows = connection.execute(
            DELETE_DOCUMENT_SQL,
            (COLLECTION_NAME, source),
        ).fetchall()

    return len(deleted_rows)
