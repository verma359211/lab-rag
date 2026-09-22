# Hybrid search implementation plan

## Current status

The first hybrid-search version is now implemented:

- New chunks receive document, chunk, and sequence IDs.
- Vector and PostgreSQL keyword searches run independently.
- Reciprocal Rank Fusion combines their rankings.
- Only the best fused chunks are passed to Groq.
- A SQL migration adds the full-text GIN index.
- Unit tests cover the fusion behavior.

The remaining production-oriented work is evaluation, optional reranking, and
eventually owning our database schema instead of using LangChain's tables.

## Goal

Our current retrieval uses vector search only. Hybrid search will run vector
and keyword searches for the same question, then combine their rankings before
we send context to Groq.

```text
                         -> vector search  --\
User question -> prepare                       -> rank fusion -> best chunks -> Groq
                         -> keyword search --/
```

Vector search is useful when the question and document express the same idea
with different words. Keyword search is useful for exact product names, error
codes, people, abbreviations, and uncommon technical terms. Using both makes
retrieval less dependent on the strengths of only one search method.

## Keep the first version simple

The first version will continue using the documents that LangChain already
stores in PostgreSQL. PostgreSQL has built-in full-text search, so we do not
need Elasticsearch or another service for this demonstration.

We will add a stable `chunk_id` to every chunk's metadata during ingestion.
Both search paths will return that ID, allowing us to recognize the same chunk
when it appears in both result lists.

## Implementation stages

### 1. Give every stored chunk an identity — implemented

Update `ingestion.py` before saving chunks:

- Add a `document_id` shared by all chunks from one upload.
- Add a unique `chunk_id` to each individual chunk.
- Add a `chunk_number` so neighboring chunks can be identified later.
- Keep the existing source name and page number.

For this small project, we can re-ingest existing PDFs after making the change.
That is clearer than writing a one-off migration for old metadata.

### 2. Add a PostgreSQL keyword-search index — implemented

Create a small SQL migration that adds a GIN full-text index over the document
text stored by LangChain. PostgreSQL's `to_tsvector` converts document text
into searchable words, while the GIN index makes that search fast.

The keyword query will use:

- `websearch_to_tsquery` to interpret normal user wording.
- `ts_rank_cd` to rank chunks by keyword relevance.
- The LangChain collection name so results only come from `rag_documents`.

We will keep this SQL in a migration file instead of hiding database setup in
Python startup code.

### 3. Split retrieval into two small search functions — implemented

Create these focused modules:

```text
app/search/
|-- vector_search.py   Uses LangChain PGVector similarity search
|-- keyword_search.py  Uses PostgreSQL full-text search
`-- fusion.py          Combines the two ordered result lists
```

Each search will initially return about 20 candidates. This wider candidate
set gives the fusion step enough useful results to compare. It does not mean
all 20 chunks will be sent to the language model.

### 4. Combine rankings with Reciprocal Rank Fusion — implemented

Vector and keyword scores use different scales, so adding their raw scores
would be misleading. Reciprocal Rank Fusion (RRF) uses each chunk's position
instead:

```text
RRF score = 1 / (60 + vector rank) + 1 / (60 + keyword rank)
```

If a chunk appears in only one list, it receives only that part of the score.
If it ranks well in both lists, its two contributions are added and it moves
up. The constant `60` prevents the first result from overpowering everything
else and is a common starting value, not a value we need to tune immediately.

### 5. Send only the best fused chunks to generation — implemented

After fusion we will:

1. Sort chunks by their RRF score.
2. Remove duplicates using `chunk_id`.
3. Keep roughly the best 5 to 8 chunks.
4. Reuse the existing `build_context` and `generate_answer` flow.

The `/chat` request and response shape can stay unchanged, so the React
frontend will not need to change for the first version.

### 6. Add simple retrieval diagnostics — partly implemented

During development, return or log enough information to understand why a chunk
won:

- Vector rank.
- Keyword rank.
- Final RRF score.
- Source and page.

This should be behind a debug setting so normal API responses stay clean.

### 7. Evaluate before adding a reranker

Create a small set of questions with known relevant pages. Include semantic
questions as well as questions containing exact names or codes. Compare:

- Vector-only retrieval.
- Keyword-only retrieval.
- Hybrid retrieval.

Useful first measurements are Recall@5 (did the correct chunk reach the first
five?) and Mean Reciprocal Rank (how early did the first correct chunk appear?).
We should keep hybrid search only if it improves the examples we care about.

### 8. Add a reranker afterward

A reranker is a separate improvement. Once hybrid search works, it can examine
the best 20 to 40 fused candidates more carefully and return the best 5 to 8.
Keeping it as a later stage makes it easy to measure whether the extra model
call improves quality enough to justify its latency and cost.

## Later production cleanup

The simple version reads LangChain's existing PostgreSQL tables. For a larger
production system, we should own a `document_chunks` table instead of depending
on LangChain's internal table layout. A production table would explicitly hold
the chunk ID, document ID, content, metadata, embedding vector, and searchable
text, with HNSW, GIN, and normal database indexes added deliberately.

That production schema is a good later upgrade, but it is unnecessary
complexity for the first tutorial implementation.

## Expected implementation order

1. Add chunk IDs during ingestion.
2. Add and run the full-text index migration.
3. Implement and test keyword search by itself.
4. Implement RRF and unit-test it with small fake result lists.
5. Connect vector search and keyword search in `retrieval.py`.
6. Test the existing frontend against the unchanged `/chat` endpoint.
7. Evaluate retrieval quality before deciding on a reranker.

Hybrid search is enabled. Evaluation and reranking remain deliberate future
steps so we can measure the value of each addition instead of adding complexity
without evidence.
