const { pool } = require("./db");
const { embedText, generateAnswer } = require("./embeddings");

async function queryDocuments(question) {
	const questionEmbedding = await embedText(question);

	const { rows } = await pool.query(
		`SELECT content, source,
            1 - (embedding <=> $1::vector) AS similarity
     FROM documents
     ORDER BY embedding <=> $1::vector
     LIMIT 5`,
		[JSON.stringify(questionEmbedding)],
	);

	if (rows.length === 0) {
		return { answer: "No relevant documents found.", sources: [] };
	}

	const context = rows.map((r) => r.content).join("\n\n---\n\n");
	const answer = await generateAnswer(context, question);

	return {
		answer,
		sources: [...new Set(rows.map((r) => r.source))],
		topSimilarity: parseFloat(rows[0].similarity).toFixed(3),
	};
}

module.exports = { queryDocuments };
