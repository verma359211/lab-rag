const ACTIVE = new Set(["queued", "running", "waiting", "cancelling"]);

export default function EvaluationProgress({ job, onCancel, onResume }) {
  if (!job) {
    return <section className="panel evaluation-empty">Start or select an evaluation to inspect it.</section>;
  }

  const completed = job.progress?.completed || 0;
  const total = job.progress?.total || 0;
  const percent = total ? Math.round((completed / total) * 100) : 0;
  const resumable = ["paused", "paused_rate_limit", "cancelled", "failed", "completed_with_errors"].includes(job.status)
    || (job.status === "completed" && (job.aggregate?.failed_count || 0) > 0);

  return (
    <section className="panel progress-panel">
      <div className="progress-title">
        <div>
          <span className={`status-pill ${job.status}`}>{job.status.replaceAll("_", " ")}</span>
          <h2>{job.config?.experiment}</h2>
          <p>{job.status_message}</p>
        </div>
        <div className="progress-actions">
          {ACTIVE.has(job.status) && <button className="delete-button" onClick={onCancel}>Cancel</button>}
          {resumable && <button onClick={onResume}>Resume</button>}
        </div>
      </div>

      <div className="progress-track"><span style={{ width: `${percent}%` }} /></div>
      <div className="progress-meta">
        <span>{completed} / {total} questions</span>
        <span>{percent}%</span>
        <span>Question: {job.current || "—"}</span>
        <span>Metric: {job.current_metric?.replaceAll("_", " ") || "—"}</span>
      </div>

      {job.rate_limit && (
        <div className="rate-limit-card">
          <strong>Groq limit</strong>
          <span>{job.rate_limit.limit_type?.replaceAll("_", " ")}</span>
          <span>Remaining tokens: {job.rate_limit.remaining_tokens ?? "unknown"}</span>
          <span>Reset: {job.rate_limit.token_reset || "use provider retry time"}</span>
        </div>
      )}
      {job.provider_quota && !job.rate_limit && (
        <div className="rate-limit-card quota-ok">
          <strong>Groq token budget</strong>
          <span>Remaining: {job.provider_quota["x-ratelimit-remaining-tokens"] ?? "unknown"}</span>
          <span>Limit: {job.provider_quota["x-ratelimit-limit-tokens"] ?? "unknown"}</span>
          <span>Reset: {job.provider_quota["x-ratelimit-reset-tokens"] || "unknown"}</span>
        </div>
      )}
    </section>
  );
}
