-- PostgreSQL must repeatedly turn chunk text into searchable word tokens.
-- A GIN index stores those tokens in advance, making keyword retrieval faster
-- as the number of uploaded chunks grows.

CREATE INDEX IF NOT EXISTS ix_langchain_pg_embedding_document_fts
ON langchain_pg_embedding
USING GIN (
    to_tsvector('english', COALESCE(document, ''))
);
