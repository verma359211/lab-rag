function isExpectedEvidence(result, chunkId) {
  const evidenceSets = result.expected_evidence_sets || [];
  return (
    result.expected_chunk_ids?.includes(chunkId) ||
    evidenceSets.some((evidenceSet) => evidenceSet.includes(chunkId))
  );
}

export default function QuestionResults({ results = [] }) {
  if (!results.length) return null;

  return (
    <section className="panel question-results">
      <div className="section-heading"><div><span className="step">04</span><h2>Questions</h2></div></div>
      {results.map((result) => (
        <details key={result.id}>
          <summary>
            <strong>{result.id}</strong>
            <span>{result.category?.replaceAll("_", " ")}</span>
            <span>{result.evaluation_status || "pending"}</span>
          </summary>
          <div className="question-detail">
            <h3>Question</h3><p>{result.question}</p>
            {result.answer && <><h3>Generated answer</h3><p>{result.answer}</p></>}
            <h3>Reference answer</h3><p>{result.reference_answer}</p>

            {result.expected_evidence_sets?.length > 1 && (
              <p className="section-note">
                This question has {result.expected_evidence_sets.length} valid evidence paths.
              </p>
            )}

            {result.retrieval_metrics && (
              <div className="compact-metrics">
                {Object.entries(result.retrieval_metrics).map(([name, value]) => (
                  <span key={name}>{name.replaceAll("_", " ")}: <strong>{Number(value).toFixed(3)}</strong></span>
                ))}
              </div>
            )}

            {result.ragas?.scores && (
              <div className="compact-metrics">
                {Object.entries(result.ragas.scores).map(([name, value]) => (
                  <span key={name}>
                    {name.replaceAll("_", " ")}: <strong>{Number(value).toFixed(3)}</strong>
                    {result.ragas.sources?.[name] === "human_chunk_ids" && " · human labels"}
                    {result.ragas.cache_hits?.[name] && " · cached"}
                  </span>
                ))}
              </div>
            )}

            {!!Object.keys(result.ragas?.token_estimates || {}).length && (
              <div className="compact-metrics token-estimates">
                {Object.entries(result.ragas.token_estimates).map(([name, value]) => (
                  <span key={name}>{name.replaceAll("_", " ")}: <strong>~{value} tokens</strong></span>
                ))}
              </div>
            )}

            {!!result.final_contexts?.length && (
              <details className="nested-details">
                <summary>Retrieved chunks</summary>
                {result.final_contexts.map((chunk) => (
                  <article className={`result-chunk ${isExpectedEvidence(result, chunk.chunk_id) ? "expected" : ""}`} key={chunk.chunk_id}>
                    <strong>
                      #{chunk.rank} · {chunk.source} · page {chunk.page || "?"}
                      {isExpectedEvidence(result, chunk.chunk_id) ? " · valid evidence" : ""}
                    </strong>
                    <small>{chunk.chunk_id}</small>
                    <p>{chunk.content}</p>
                  </article>
                ))}
              </details>
            )}

            {!!Object.keys(result.ragas?.reasons || {}).length && (
              <details className="nested-details">
                <summary>Evaluator reasons</summary>
                {Object.entries(result.ragas.reasons).map(([name, reason]) => (
                  <article className="result-chunk" key={name}>
                    <strong>{name.replaceAll("_", " ")}</strong>
                    <p>{reason}</p>
                  </article>
                ))}
              </details>
            )}

            {!!Object.keys(result.ragas?.traces || {}).length && (
              <details className="nested-details">
                <summary>Judge decision trace</summary>
                {Object.entries(result.ragas.traces).map(([name, trace]) => (
                  <article className="result-chunk judge-trace" key={name}>
                    <strong>{name.replaceAll("_", " ")}</strong>
                    <pre>{JSON.stringify(trace, null, 2)}</pre>
                  </article>
                ))}
              </details>
            )}

            {!!result.ragas?.judge_contexts?.length && (
              <details className="nested-details">
                <summary>Context shown to the judge</summary>
                {result.ragas.judge_contexts.map((context, index) => (
                  <article className="result-chunk judge-trace" key={`${context.chunk_ids?.join("-")}-${index}`}>
                    <strong>
                      {context.source} · original ranks {context.original_retrieval_ranks?.join(", ")}
                    </strong>
                    <small>{context.chunk_ids?.join(", ")}</small>
                    <pre>{context.text}</pre>
                  </article>
                ))}
              </details>
            )}

            {!!result.errors?.length && <p className="result-error">{result.errors.join(" | ")}</p>}
          </div>
        </details>
      ))}
    </section>
  );
}
