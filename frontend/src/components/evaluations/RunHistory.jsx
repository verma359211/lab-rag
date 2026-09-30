export default function RunHistory({ jobs, selectedId, compareId, onSelect, onCompare }) {
  return (
    <section className="panel run-history">
      <div className="section-heading"><div><span className="step">05</span><h2>Run history</h2></div></div>
      {!jobs.length && <p className="empty">No dashboard evaluations yet.</p>}
      {jobs.map((job) => (
        <div className={`history-row ${selectedId === job.id ? "selected" : ""}`} key={job.id}>
          <button className="history-select" onClick={() => onSelect(job.id)} type="button">
            <strong>{job.config?.experiment}</strong>
            <span>{job.status} · {job.config?.dataset} · {job.config?.search_mode}</span>
          </button>
          {job.status === "completed" && job.id !== selectedId && (
            <button className="text-button" onClick={() => onCompare(compareId === job.id ? null : job.id)} type="button">
              {compareId === job.id ? "Remove" : "Compare"}
            </button>
          )}
        </div>
      ))}
    </section>
  );
}
