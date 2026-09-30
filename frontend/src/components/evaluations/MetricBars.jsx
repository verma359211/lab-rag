function MetricGroup({ title, metrics = {}, comparison = {} }) {
  const entries = Object.entries(metrics);
  if (!entries.length) return null;

  return (
    <div className="metric-group">
      <h3>{title}</h3>
      {entries.map(([name, value]) => {
        const other = comparison?.[name];
        const change = typeof other === "number" ? value - other : null;
        return (
          <div className="metric-row" key={name}>
            <span>{name.replaceAll("_", " ")}</span>
            <div className="metric-track"><span style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} /></div>
            <strong>{Number(value).toFixed(3)}</strong>
            {change !== null && <small className={change >= 0 ? "positive" : "negative"}>{change >= 0 ? "+" : ""}{change.toFixed(3)}</small>}
          </div>
        );
      })}
    </div>
  );
}

export default function MetricBars({ aggregate, comparison, config, comparisonConfig }) {
  if (!aggregate || !Object.keys(aggregate).length) return null;

  const comparedFields = ["dataset", "mode", "search_mode"];
  const differences = comparisonConfig
    ? comparedFields.filter((name) => config?.[name] !== comparisonConfig?.[name])
    : [];

  return (
    <section className="panel metrics-panel">
      <div className="section-heading"><div><span className="step">03</span><h2>Metrics</h2></div></div>
      {!!differences.length && (
        <p className="comparison-warning">
          Comparison warning: these runs use different {differences.join(", ").replaceAll("_", " ")} settings.
        </p>
      )}
      <div className="summary-cards">
        <div><span>Questions</span><strong>{aggregate.sample_count || 0}</strong></div>
        <div><span>Successful</span><strong>{aggregate.successful_count || 0}</strong></div>
        <div><span>Errors</span><strong>{aggregate.failed_count || 0}</strong></div>
        <div><span>Average latency</span><strong>{aggregate.latency_ms?.average ? `${Math.round(aggregate.latency_ms.average)} ms` : "—"}</strong></div>
      </div>
      <MetricGroup title="Retrieval" metrics={aggregate.retrieval_metrics} comparison={comparison?.retrieval_metrics} />
      <MetricGroup title="RAGAS" metrics={aggregate.ragas_metrics} comparison={comparison?.ragas_metrics} />
    </section>
  );
}
