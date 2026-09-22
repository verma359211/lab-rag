const path = require("path");
require("dotenv").config({ path: path.join(__dirname, "../../.env") });
const express = require("express");
const multer = require("multer");
const { initDb } = require("./db");
const { ingestDocument } = require("./ingest");
const { queryDocuments } = require("./query");

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

app.use(express.json());

app.post("/ingest", upload.single("file"), async (req, res) => {
	if (!req.file) {
		return res.status(400).json({ error: "No file uploaded" });
	}

	if (!req.file.mimetype.includes("pdf")) {
		return res.status(400).json({ error: "Only PDF files are supported" });
	}

	try {
		const count = await ingestDocument(req.file.buffer, req.file.originalname);
		res.json({
			message: `Ingested ${count} chunks from "${req.file.originalname}"`,
		});
	} catch (err) {
		console.error(err);
		res.status(500).json({ error: "Ingestion failed", detail: err.message });
	}
});

app.post("/chat", async (req, res) => {
	const { question } = req.body;

	if (!question || typeof question !== "string") {
		return res.status(400).json({ error: "question is required" });
	}

	try {
		const result = await queryDocuments(question);
		res.json(result);
	} catch (err) {
		console.error(err);
		res.status(500).json({ error: "Query failed", detail: err.message });
	}
});

const PORT = process.env.PORT || 3000;

initDb().then(() => {
	app.listen(PORT, () => {
		console.log(`RAG chatbot running on port ${PORT}`);
	});
});
