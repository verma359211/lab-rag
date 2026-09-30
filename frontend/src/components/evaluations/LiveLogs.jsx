import { useState } from "react";

export default function LiveLogs({ events = [] }) {
  const [filter, setFilter] = useState("all");
  const visible = events.filter((event) => {
    if (filter === "all") return true;
    if (filter === "errors") return event.level === "warning" || event.level === "error";
    return event.stage === filter;
  });

  return (
    <section className="panel logs-panel">
      <div className="section-heading">
        <div><span className="step">02</span><h2>Live activity</h2></div>
        <select onChange={(event) => setFilter(event.target.value)} value={filter}>
          <option value="all">All</option>
          <option value="retrieval">Retrieval</option>
          <option value="generation">Generation</option>
          <option value="ragas">RAGAS</option>
          <option value="errors">Warnings</option>
        </select>
      </div>
      <div className="log-list">
        {!visible.length && <p className="empty">Events will appear here.</p>}
        {visible.slice().reverse().map((event, index) => (
          <div className={`log-entry ${event.level}`} key={`${event.timestamp}-${index}`}>
            <time>{new Date(event.timestamp).toLocaleTimeString()}</time>
            <span>{event.question_id || "RUN"}</span>
            <p>{event.message}</p>
            {event.data?.score !== undefined && <strong>{Number(event.data.score).toFixed(3)}</strong>}
          </div>
        ))}
      </div>
    </section>
  );
}
