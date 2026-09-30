import { useEffect, useState } from "react";
import {
  deleteDocument,
  getDocumentChunks,
  getDocuments,
} from "../api";

function getChunkLabel(metadata) {
  const chunkNumber = metadata.chunk_number;
  return Number.isInteger(chunkNumber) ? `Chunk ${chunkNumber + 1}` : "Legacy chunk";
}

function getPageLabel(metadata) {
  const page = metadata.page;
  return Number.isInteger(page) ? `Page ${page + 1}` : "Page unknown";
}

export default function DocumentsPanel({ refreshToken }) {
  const [documents, setDocuments] = useState([]);
  const [selectedSource, setSelectedSource] = useState("");
  const [chunks, setChunks] = useState([]);
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(true);
  const [deletingSource, setDeletingSource] = useState("");

  // The token changes after an upload, which reloads the document summaries
  // without coupling this component to the upload form.
  useEffect(() => {
    loadDocuments();
  }, [refreshToken]);

  async function loadDocuments() {
    try {
      setLoading(true);
      const result = await getDocuments();
      setDocuments(result.documents);
      setStatus("");
    } catch (error) {
      setStatus(error.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleViewChunks(source) {
    if (selectedSource === source) {
      setSelectedSource("");
      setChunks([]);
      return;
    }

    try {
      setSelectedSource(source);
      setChunks([]);
      setStatus("Loading chunks...");

      const result = await getDocumentChunks(source);
      setChunks(result.chunks);
      setStatus("");
    } catch (error) {
      setSelectedSource("");
      setStatus(error.message);
    }
  }

  async function handleDelete(source) {
    // Source-based deletion is temporary for legacy chunks. Confirmation is
    // important because every row sharing this filename will be removed.
    const confirmed = window.confirm(
      `Delete ${source} and all of its stored chunks?`,
    );

    if (!confirmed) return;

    try {
      setDeletingSource(source);
      const result = await deleteDocument(source);

      if (selectedSource === source) {
        setSelectedSource("");
        setChunks([]);
      }

      await loadDocuments();
      setStatus(result.message);
    } catch (error) {
      setStatus(error.message);
    } finally {
      setDeletingSource("");
    }
  }

  return (
    <section className="panel">
      <div className="section-heading">
        <div>
          <span className="step">02</span>
          <h2>Documents</h2>
        </div>
        <button className="text-button" onClick={loadDocuments} type="button">
          Refresh
        </button>
      </div>

      <p className="section-note">Temporarily grouped by source filename.</p>

      {loading && <p className="empty">Loading documents...</p>}

      {!loading && documents.length === 0 && (
        <p className="empty">No documents have been ingested yet.</p>
      )}

      <div className="document-list">
        {documents.map((document) => (
          <article className="document-item" key={document.source}>
            <div className="document-summary">
              <div>
                <strong>{document.source}</strong>
                <small>
                  {document.chunkCount} chunks · {document.pageCount} pages
                </small>
              </div>

              <div className="document-actions">
                <button
                  className="text-button"
                  onClick={() => handleViewChunks(document.source)}
                  type="button"
                >
                  {selectedSource === document.source ? "Hide" : "Inspect"}
                </button>
                <button
                  className="delete-button"
                  disabled={deletingSource === document.source}
                  onClick={() => handleDelete(document.source)}
                  type="button"
                >
                  {deletingSource === document.source ? "Deleting..." : "Delete"}
                </button>
              </div>
            </div>

            {selectedSource === document.source && (
              <div className="chunk-list">
                {chunks.length === 0 && <p className="empty">Loading chunks...</p>}

                {chunks.map((chunk) => (
                  <article className="chunk-item" key={chunk.id}>
                    <small>
                      {getChunkLabel(chunk.metadata)} · {getPageLabel(chunk.metadata)}
                    </small>
                    <p>{chunk.content}</p>
                  </article>
                ))}
              </div>
            )}
          </article>
        ))}
      </div>

      {status && <p className="status">{status}</p>}
    </section>
  );
}
