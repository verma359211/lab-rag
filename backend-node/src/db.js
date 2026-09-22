const { Pool } = require("pg");

const pool = new Pool({
	connectionString: process.env.DATABASE_URL,
});

async function initDb() {
	await pool.query(`CREATE EXTENSION IF NOT EXISTS vector`);

	await pool.query(`
    CREATE TABLE IF NOT EXISTS documents (
      id UUID PRIMARY KEY,
      content TEXT NOT NULL,
      source TEXT NOT NULL,
      embedding VECTOR(3072)
    )
  `);

	console.log("Database ready");
}

module.exports = { pool, initDb };
