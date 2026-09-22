# Python LangChain backend

This backend provides the same `/ingest` and `/chat` endpoints as the original
Node.js backend. It uses LangChain for PDF loading, text splitting, Gemini
embeddings, PGVector storage and retrieval, prompting, and Groq generation.

## Code map

The backend is split by responsibility so you can learn one part of RAG at a
time. The comments and docstrings inside these files explain the important
decisions in more detail.

```text
app/
|-- main.py          HTTP routes and request validation
|-- schemas.py       Shape of JSON requests from the frontend
|-- config.py        Environment variables and shared settings
|-- embeddings.py    Gemini text-to-vector client
|-- vector_store.py  LangChain connection to PostgreSQL/PGVector
|-- ingestion.py     PDF loading, splitting, embedding, and storage
|-- documents.py     Temporary source-based listing, inspection, and deletion
|-- search/           Vector search, keyword search, and rank fusion
|-- retrieval.py     Hybrid search coordinator and context preparation
|-- generation.py    Prompt, Groq model, and output parsing
`-- rag.py           Small coordinator for the answering flow
```

The two main flows are:

```text
Upload: PDF -> pages -> chunks -> embeddings -> PGVector

Chat: question -> vector + keyword search -> rank fusion -> Groq -> answer
```

`main.py` is kept small because HTTP handling and RAG logic are different
jobs. `rag.py` is also small so it reads like a high-level map instead of
hiding all the work in one large file.

## Run locally

Start PostgreSQL from the repository root:

```powershell
docker compose up -d
```

Create the Python environment and install dependencies:

```powershell
cd backend-python
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create the keyword-search index:

```powershell
python -m scripts.apply_migrations
```

This command first lets LangChain create its PGVector tables if necessary. It
then adds a PostgreSQL GIN full-text index. The migration is safe to run again.

Start the API on the same port used by the frontend proxy:

```powershell
python -m uvicorn app.main:app --reload --port 3000
```

The existing root `.env` file supplies `DATABASE_URL`, `GEMINI_API_KEY`,
`GROQ_API_KEY`, and the frontend origins allowed by CORS:

```text
FRONTEND_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

The React app calls FastAPI directly rather than using a Vite proxy. Create
`frontend/.env` from `frontend/.env.example`:

```text
VITE_API_URL=http://localhost:3000
```

For deployment, set `VITE_API_URL` to the Railway backend address and
`FRONTEND_ORIGINS` to the Vercel frontend address. Frontend environment
changes require a new Vite build because the backend URL is placed into the
generated browser JavaScript.

## Stored document management

The API currently groups chunks using their `source` filename so documents
ingested before `document_id` was introduced remain visible:

```text
GET    /documents         List stored source filenames
GET    /documents/chunks  Inspect the stored chunks for one source
DELETE /documents         Delete every chunk for one source
```

This is intentionally temporary. Two different uploads with the same filename
will be treated as one document. After the legacy chunks are deleted, these
operations should use `document_id`, and a separate documents table can hold
one summary row per upload.

## Hybrid search

Hybrid retrieval is now enabled. It runs semantic PGVector search and
PostgreSQL full-text search, then combines their rankings using Reciprocal Rank
Fusion. See [HYBRID_SEARCH_PLAN.md](./HYBRID_SEARCH_PLAN.md) for the design,
completed stages, and sensible future improvements.

Run the fast fusion tests without using the database or external APIs:

```powershell
python -m unittest discover -s tests
```
