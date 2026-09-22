import { useState } from "react";
import { askQuestion, uploadDocument } from "./api";
import DocumentsPanel from "./DocumentsPanel";

export default function App() {
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [answering, setAnswering] = useState(false);
  const [documentsRefreshToken, setDocumentsRefreshToken] = useState(0);

  async function handleUpload(event) {
    event.preventDefault();

    if (!file) {
      setUploadStatus("Please choose a PDF first.");
      return;
    }

    try {
      setUploading(true);
      setUploadStatus("Uploading and processing the PDF...");
      const result = await uploadDocument(file);
      setUploadStatus(result.message);
      setDocumentsRefreshToken((current) => current + 1);
    } catch (error) {
      setUploadStatus(error.message);
    } finally {
      setUploading(false);
    }
  }

  async function handleQuestion(event) {
    event.preventDefault();

    const text = question.trim();
    if (!text || answering) return;

    setMessages((current) => [...current, { role: "user", text }]);
    setQuestion("");

    try {
      setAnswering(true);
      const result = await askQuestion(text);
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          text: result.answer,
          sources: result.sources,
          similarity: result.topSimilarity,
        },
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        { role: "error", text: error.message },
      ]);
    } finally {
      setAnswering(false);
    }
  }

  return (
    <main className="page">
      <header>
        <p className="eyebrow">Document assistant</p>
        <h1>Ask questions about your PDF</h1>
        <p>Upload a document, then chat with its content.</p>
      </header>

      <section className="card">
        <h2>1. Upload a PDF</h2>
        <form className="upload-form" onSubmit={handleUpload}>
          <input
            type="file"
            accept="application/pdf"
            onChange={(event) => setFile(event.target.files[0])}
          />
          <button disabled={uploading} type="submit">
            {uploading ? "Uploading..." : "Upload document"}
          </button>
        </form>
        {uploadStatus && <p className="status">{uploadStatus}</p>}
      </section>

      <DocumentsPanel refreshToken={documentsRefreshToken} />

      <section className="card chat-card">
        <h2>3. Ask a question</h2>

        <div className="messages" aria-live="polite">
          {messages.length === 0 && (
            <p className="empty">Your conversation will appear here.</p>
          )}

          {messages.map((message, index) => (
            <article className={`message ${message.role}`} key={index}>
              <strong>{message.role === "user" ? "You" : "Assistant"}</strong>
              <p>{message.text}</p>
              {message.sources?.length > 0 && (
                <small>
                  Source: {message.sources.join(", ")}
                  {message.similarity && ` · Similarity: ${message.similarity}`}
                </small>
              )}
            </article>
          ))}

          {answering && <p className="empty">Looking through your document...</p>}
        </div>

        <form className="question-form" onSubmit={handleQuestion}>
          <input
            type="text"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask something about your document..."
          />
          <button disabled={answering} type="submit">
            Send
          </button>
        </form>
      </section>
    </main>
  );
}
