# RAG frontend

This Vite and React frontend intentionally keeps each page section in a small
component:

```text
src/
|-- App.jsx                         Page composition and document refresh
|-- api.js                          All calls to the FastAPI backend
`-- components/
    |-- AppHeader.jsx               Page introduction
    |-- UploadSection.jsx           PDF upload flow
    |-- DocumentsPanel.jsx          Stored document inspection and deletion
    |-- ChatSection.jsx             Question and answer state
    |-- SearchModeSelector.jsx      Hybrid, reranked, vector, or keyword mode
    |-- RetrievalDetails.jsx        Per-chunk scores and ranks
    `-- MarkdownContent.jsx         Safe Markdown and table rendering
```

`App.jsx` does not contain the feature logic. This keeps future features from
turning one file into a large component.

## API URL

Copy `.env.example` to `.env` and set the complete FastAPI address:

```text
VITE_API_URL=http://localhost:3000
```

Vite places this public URL into the browser build. API keys and database
credentials must never use a `VITE_` variable.

## Retrieval scores

The chat can display four score types:

- Vector relevance shows semantic similarity from PGVector.
- Keyword score comes from PostgreSQL full-text ranking.
- RRF score combines vector and keyword positions in hybrid mode.
- Reranker score compares the question and chunk together with a cross-encoder.

These values use different scales. They are displayed separately and should
only be compared with scores of the same type.

## Assistant formatting

Assistant responses are rendered with `react-markdown` and `remark-gfm`.
GitHub-flavored Markdown support is what turns model-generated table syntax
into real HTML tables. Raw HTML is not enabled; user messages and errors remain
plain text.
## Evaluation dashboard

The frontend has two workspace views:

- **Chat** asks questions with vector, keyword, hybrid, or reranked hybrid retrieval.
- **Evaluations** starts resumable retrieval or full RAGAS benchmarks and displays live progress, logs, metrics, question evidence, and run comparisons.

The browser talks directly to FastAPI using `VITE_API_URL`. API keys always
remain in the backend `.env`; they are never sent to React.

Run the frontend from this directory:

```powershell
npm run dev
```

Open the URL printed by Vite, usually `http://127.0.0.1:5173`.
