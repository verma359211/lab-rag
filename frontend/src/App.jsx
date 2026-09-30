import { useState } from "react";
import AppHeader from "./components/AppHeader";
import ChatSection from "./components/ChatSection";
import DocumentsPanel from "./components/DocumentsPanel";
import EvaluationsPage from "./pages/EvaluationsPage";
import UploadSection from "./components/UploadSection";

export default function App() {
  const [documentsRefreshToken, setDocumentsRefreshToken] = useState(0);
  const [activePage, setActivePage] = useState("chat");

  function refreshDocuments() {
    setDocumentsRefreshToken((current) => current + 1);
  }

  return (
    <main className="page">
      <AppHeader activePage={activePage} onPageChange={setActivePage} />

      {activePage === "chat" ? (
        <div className="workspace">
          <aside className="workspace-sidebar">
            <UploadSection onUploadComplete={refreshDocuments} />
            <DocumentsPanel refreshToken={documentsRefreshToken} />
          </aside>

          <section className="workspace-main">
            <ChatSection />
          </section>
        </div>
      ) : (
        <EvaluationsPage />
      )}
    </main>
  );
}
