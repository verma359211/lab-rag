import { useState } from "react";
import { askQuestion } from "../api";
import MarkdownContent from "./MarkdownContent";
import RetrievalDetails from "./RetrievalDetails";
import SearchModeSelector from "./SearchModeSelector";

export default function ChatSection() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [searchMode, setSearchMode] = useState("hybrid");
  const [answering, setAnswering] = useState(false);

  async function handleQuestion(event) {
    event.preventDefault();

    const text = question.trim();
    if (!text || answering) return;

    setMessages((current) => [...current, { role: "user", text }]);
    setQuestion("");

    try {
      setAnswering(true);
      const result = await askQuestion(text, searchMode);

      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          text: result.answer,
          method: result.retrievalMethod,
          retrievalDetails: result.retrievalDetails,
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
    <section className="panel chat-panel">
      <div className="section-heading chat-heading">
        <div>
          <span className="step">03</span>
          <h2>Ask</h2>
        </div>
        <SearchModeSelector value={searchMode} onChange={setSearchMode} />
      </div>

      <div className="messages" aria-live="polite">
        {messages.length === 0 && (
          <p className="empty">Your answers and retrieval scores appear here.</p>
        )}

        {messages.map((message, index) => (
          <article className={`message ${message.role}`} key={index}>
            <strong>{message.role === "user" ? "You" : "Assistant"}</strong>
            {message.role === "assistant" ? (
              <MarkdownContent>{message.text}</MarkdownContent>
            ) : (
              <p>{message.text}</p>
            )}
            <RetrievalDetails
              method={message.method}
              results={message.retrievalDetails}
            />
          </article>
        ))}

        {answering && <p className="empty">Searching your documents...</p>}
      </div>

      <form className="question-form" onSubmit={handleQuestion}>
        <input
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Ask a question about your documents"
          type="text"
          value={question}
        />
        <button disabled={answering} type="submit">
          Send
        </button>
      </form>
    </section>
  );
}
