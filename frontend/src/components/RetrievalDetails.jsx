const METHOD_LABELS = {
  hybrid: "Hybrid",
  hybrid_rerank: "Hybrid + Reranker",
  vector: "Vector only",
  keyword: "Keyword only",
};

function formatScore(score) {
  return typeof score === "number" ? score.toFixed(4) : null;
}

export default function RetrievalDetails({ method, results }) {
  if (!results?.length) return null;

  return (
    <details className="retrieval-details">
      <summary>
        {METHOD_LABELS[method] || method} retrieval · {results.length} chunks
      </summary>

      <div className="score-list">
        <p className="score-note">
          Higher is better. Compare values only within the same score type.
        </p>

        {results.map((result, index) => (
          <article className="score-item" key={result.chunkId || index}>
            <div className="score-source">
              <strong>{result.source}</strong>
              <span>{result.page ? `Page ${result.page}` : "Page unknown"}</span>
            </div>

            <div className="score-values">
              {typeof result.vectorScore === "number" && (
                <span>
                  Vector {formatScore(result.vectorScore)} · rank {result.vectorRank}
                </span>
              )}
              {typeof result.keywordScore === "number" && (
                <span>
                  Keyword {formatScore(result.keywordScore)} · rank{" "}
                  {result.keywordRank}
                </span>
              )}
              {typeof result.hybridScore === "number" && (
                <span>
                  RRF {formatScore(result.hybridScore)}
                  {result.hybridRank ? ` · rank ${result.hybridRank}` : ""}
                </span>
              )}
              {typeof result.rerankerScore === "number" && (
                <span>
                  Reranker {formatScore(result.rerankerScore)} · rank{" "}
                  {result.rerankerRank}
                </span>
              )}
            </div>
          </article>
        ))}
      </div>
    </details>
  );
}
