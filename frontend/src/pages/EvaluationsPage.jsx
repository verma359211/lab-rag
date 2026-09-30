import { useCallback, useEffect, useState } from "react";
import {
  cancelEvaluation,
  getEvaluation,
  getEvaluations,
  resumeEvaluation,
  startEvaluation,
} from "../api";
import EvaluationForm from "../components/evaluations/EvaluationForm";
import EvaluationProgress from "../components/evaluations/EvaluationProgress";
import LiveLogs from "../components/evaluations/LiveLogs";
import MetricBars from "../components/evaluations/MetricBars";
import QuestionResults from "../components/evaluations/QuestionResults";
import RunHistory from "../components/evaluations/RunHistory";

const ACTIVE = new Set(["queued", "running", "waiting", "cancelling"]);

export default function EvaluationsPage() {
  const [jobs, setJobs] = useState([]);
  const [job, setJob] = useState(null);
  const [compareJob, setCompareJob] = useState(null);
  const [error, setError] = useState("");

  const refreshHistory = useCallback(async () => {
    const response = await getEvaluations();
    setJobs(response.evaluations);
  }, []);

  const selectJob = useCallback(async (runId) => {
    if (!runId) return;
    setJob(await getEvaluation(runId));
  }, []);

  useEffect(() => {
    refreshHistory().catch((requestError) => setError(requestError.message));
  }, [refreshHistory]);

  useEffect(() => {
    if (!job?.id || !ACTIVE.has(job.status)) return undefined;
    const timer = window.setInterval(async () => {
      try {
        await selectJob(job.id);
        await refreshHistory();
      } catch (requestError) {
        setError(requestError.message);
      }
    }, 2000);
    return () => window.clearInterval(timer);
  }, [job?.id, job?.status, refreshHistory, selectJob]);

  async function start(options) {
    try {
      setError("");
      const created = await startEvaluation(options);
      setJob(created);
      await refreshHistory();
    } catch (requestError) {
      setError(requestError.message);
    }
  }

  async function cancel() {
    setJob(await cancelEvaluation(job.id));
  }

  async function resume() {
    setJob(await resumeEvaluation(job.id));
  }

  async function compare(runId) {
    setCompareJob(runId ? await getEvaluation(runId) : null);
  }

  const hasActiveJob = jobs.some((item) => ACTIVE.has(item.status));

  return (
    <div className="evaluation-page">
      {error && <p className="api-error">{error}</p>}
      <div className="evaluation-top-grid">
        <EvaluationForm disabled={hasActiveJob} onStart={start} />
        <EvaluationProgress job={job} onCancel={cancel} onResume={resume} />
      </div>
      <LiveLogs events={job?.events} />
      <MetricBars
        aggregate={job?.aggregate}
        comparison={compareJob?.aggregate}
        comparisonConfig={compareJob?.config}
        config={job?.config}
      />
      <QuestionResults results={job?.results} />
      <RunHistory
        compareId={compareJob?.id}
        jobs={jobs}
        onCompare={compare}
        onSelect={selectJob}
        selectedId={job?.id}
      />
    </div>
  );
}
