import { useState } from "react";
import { uploadDocument } from "../api";

export default function UploadSection({ onUploadComplete }) {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("");
  const [uploading, setUploading] = useState(false);

  async function handleUpload(event) {
    event.preventDefault();

    if (!file) {
      setStatus("Choose a PDF first.");
      return;
    }

    try {
      setUploading(true);
      setStatus("Processing the PDF...");

      const result = await uploadDocument(file);
      setStatus(result.message);

      // App uses this callback only to tell the document list to refresh.
      onUploadComplete();
    } catch (error) {
      setStatus(error.message);
    } finally {
      setUploading(false);
    }
  }

  return (
    <section className="panel">
      <div className="section-heading">
        <div>
          <span className="step">01</span>
          <h2>Upload</h2>
        </div>
        <p>PDF files only</p>
      </div>

      <form className="upload-form" onSubmit={handleUpload}>
        <input
          accept="application/pdf"
          onChange={(event) => setFile(event.target.files[0])}
          type="file"
        />
        <button disabled={uploading} type="submit">
          {uploading ? "Uploading..." : "Upload PDF"}
        </button>
      </form>

      {status && <p className="status">{status}</p>}
    </section>
  );
}
