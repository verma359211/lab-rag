import { useState } from "react";

const DEFAULTS = {
  experiment: "hybrid-rrf-dashboard",
  dataset: "core15",
  mode: "retrieval",
  search_mode: "hybrid",
  question_delay: 10,
  metric_delay: 20,
  max_retries: 3,
  initial_retry_delay: 10,
  groq_tpm_limit: 8000,
};

export default function EvaluationForm({ disabled, onStart }) {
  const [values, setValues] = useState(DEFAULTS);

  function update(event) {
    const { name, value, type } = event.target;
    setValues((current) => ({
      ...current,
      [name]: type === "number" ? Number(value) : value,
    }));
  }

  function submit(event) {
    event.preventDefault();
    onStart(values);
  }

  return (
    <form className="panel evaluation-form" onSubmit={submit}>
      <div className="section-heading">
        <div><span className="step">01</span><h2>Configure run</h2></div>
      </div>

      <label>
        Experiment name
        <input name="experiment" onChange={update} type="text" value={values.experiment} />
      </label>

      <div className="form-grid">
        <label>
          Dataset
          <select name="dataset" onChange={update} value={values.dataset}>
            <option value="core15">Core 15</option>
            <option value="full60">Full 60</option>
          </select>
        </label>
        <label>
          Evaluation
          <select name="mode" onChange={update} value={values.mode}>
            <option value="retrieval">Retrieval only</option>
            <option value="full">Full RAG + RAGAS</option>
          </select>
        </label>
        <label>
          Search mode
          <select name="search_mode" onChange={update} value={values.search_mode}>
            <option value="hybrid">Hybrid</option>
            <option value="hybrid_rerank">Hybrid + Reranker</option>
            <option value="vector">Vector</option>
            <option value="keyword">Keyword</option>
          </select>
        </label>
        <label>
          Question delay (seconds)
          <input name="question_delay" onChange={update} type="number" value={values.question_delay} />
        </label>
        <label>
          Metric delay (seconds)
          <input name="metric_delay" onChange={update} type="number" value={values.metric_delay} />
        </label>
        <label>
          Groq TPM limit
          <input name="groq_tpm_limit" onChange={update} type="number" value={values.groq_tpm_limit} />
        </label>
      </div>

      <button disabled={disabled} type="submit">Start evaluation</button>
      <p className="section-note">Only one run is allowed at a time to protect provider quotas.</p>
    </form>
  );
}
