const GEMINI_KEY = process.env.GEMINI_API_KEY;
const GEMINI_BASE = "https://generativelanguage.googleapis.com/v1/models";

async function embedText(text) {
	const res = await fetch(
		`${GEMINI_BASE}/gemini-embedding-001:embedContent?key=${GEMINI_KEY}`,
		{
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ content: { parts: [{ text }] } }),
		},
	);
	const data = await res.json();
	if (!res.ok) throw new Error(JSON.stringify(data));
	return data.embedding.values;
}

async function generateAnswer(context, question) {
	const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
		},
		body: JSON.stringify({
			model: "openai/gpt-oss-20b",
			messages: [
				{
					role: "system",
					content:
						"You are a helpful assistant. Answer the question using only the context provided. If the context does not contain enough information, say so clearly.",
				},
				{
					role: "user",
					content: `Context:\n${context}\n\nQuestion: ${question}`,
				},
			],
		}),
	});
	const data = await res.json();
	if (!res.ok) throw new Error(JSON.stringify(data));
	return data.choices[0].message.content;
}

module.exports = { embedText, generateAnswer };
