export default function AppHeader({ activePage, onPageChange }) {
  return (
    <header className="app-header">
      <p className="eyebrow">Learning RAG workspace</p>
      <h1>{activePage === "chat" ? "Ask your documents" : "Evaluate your RAG"}</h1>
      <p>
        {activePage === "chat"
          ? "Upload a PDF, choose a retrieval strategy, and inspect how the answer was found."
          : "Run resumable benchmarks, watch every stage, and compare retrieval and answer quality."}
      </p>
      <nav className="page-tabs" aria-label="Workspace sections">
        <button
          className={activePage === "chat" ? "active" : ""}
          onClick={() => onPageChange("chat")}
          type="button"
        >
          Chat
        </button>
        <button
          className={activePage === "evaluations" ? "active" : ""}
          onClick={() => onPageChange("evaluations")}
          type="button"
        >
          Evaluations
        </button>
      </nav>
    </header>
  );
}
