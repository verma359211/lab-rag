const MODE_HELP = {
  hybrid: "Combines semantic and exact-word retrieval with RRF.",
  hybrid_rerank: "Reranks the strongest hybrid candidates with a cross-encoder.",
  vector: "Finds chunks with meaning similar to the question.",
  keyword: "Finds chunks containing matching words and phrases.",
};

export default function SearchModeSelector({ value, onChange }) {
  return (
    <label className="mode-selector">
      <span>Retrieval method</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="hybrid">Hybrid</option>
        <option value="hybrid_rerank">Hybrid + Reranker</option>
        <option value="vector">Vector only</option>
        <option value="keyword">Keyword only</option>
      </select>
      <small>{MODE_HELP[value]}</small>
    </label>
  );
}
