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
|-- retrieval.py     Semantic search and context preparation
|-- generation.py    Prompt, Groq model, and output parsing
`-- rag.py           Small coordinator for the answering flow
```

The two main flows are:

```text
Upload: PDF -> pages -> chunks -> embeddings -> PGVector

Chat: question -> vector search -> relevant chunks -> Groq -> answer
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

Start the API on the same port used by the frontend proxy:

```powershell
python -m uvicorn app.main:app --reload --port 3000
```

The existing root `.env` file supplies `DATABASE_URL`, `GEMINI_API_KEY`, and
`GROQ_API_KEY`.

## Next improvement

See [HYBRID_SEARCH_PLAN.md](./HYBRID_SEARCH_PLAN.md) for the planned vector +
keyword search upgrade. It is a plan only; the current code still uses vector
search so we can add and test hybrid search as a separate learning step.
