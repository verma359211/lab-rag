const { PDFParse } = require("pdf-parse");
const { v4: uuidv4 } = require("uuid");
const { pool } = require("./db");
const { embedText } = require("./embeddings");

function chunkText(text, chunkSize = 500, overlap = 50) {
	const chunks = [];
	let start = 0;

	while (start < text.length) {
		const end = Math.min(start + chunkSize, text.length);
		chunks.push(text.slice(start, end).trim());
		start += chunkSize - overlap;
	}

	return chunks.filter((chunk) => chunk.length > 50);
}

async function ingestDocument(buffer, filename) {
	const parser = new PDFParse({ data: buffer });

	const result = await parser.getText();
	const text = result.text;

	const chunks = chunkText(text);

	console.log(`Processing ${chunks.length} chunks from "${filename}"`);

	for (const chunk of chunks) {
		const embedding = await embedText(chunk);

		await pool.query(
			`INSERT INTO documents (id, content, source, embedding)
			 VALUES ($1, $2, $3, $4::vector)`,
			[uuidv4(), chunk, filename, JSON.stringify(embedding)],
		);
	}

	await parser.destroy();

	return chunks.length;
}

module.exports = { ingestDocument };
