import { useEffect, useState } from "react";
import {
  deleteDocument,
  getDocumentChunks,
  getDocuments,
} from "./api";

export default function DocumentsPanel({ refreshToken }) {
  const [documents, setDocuments] = useState([]);
  const [selectedSource, setSelectedSource] = useState("");
  const [chunks, setChunks] = useState([]);
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(true);
  const [deletingSource, setDeletingSource] = useState("");

  // Reload when the component first appears and after App reports a new upload.
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
    // Clicking View again closes the currently open chunk list.
    if (selectedSource === source) {
      setSelectedSource("");
      setChunks([]);
      return;
    }

    try {
      setSelectedSource(source);
      setChunks([]);
      setStatus("Loading stored chunks...");

      const result = await getDocumentChunks(source);
      setChunks(result.chunks);
      setStatus("");
    } catch (error) {
      setSelectedSource("");
      setStatus(error.message);
    }
  }

  async function handleDelete(source) {
    // Deletion removes every chunk grouped under this filename. A confirmation
    // prevents an accidental click from immediately removing vector data.
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
    <section className="card">
      <div className="card-heading">
        <div>
          <h2>2. Stored documents</h2>
          <p>Documents are temporarily grouped by their source filename.</p>
        </div>
        <button className="secondary-button" onClick={loadDocuments} type="button">
          Refresh
        </button>
      </div>

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
                  className="secondary-button"
                  onClick={() => handleViewChunks(document.source)}
                  type="button"
                >
                  {selectedSource === document.source ? "Hide chunks" : "View chunks"}
                </button>
                <button
                  className="danger-button"
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
                {chunks.length === 0 && (
                  <p className="empty">Loading chunks...</p>
                )}

                {chunks.map((chunk) => (
                  <article className="chunk-item" key={chunk.id}>
                    <strong>
                      Chunk {chunk.metadata.chunk_number ?? "old"} · Page{" "}
                      {(chunk.metadata.page ?? 0) + 1}
                    </strong>
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
