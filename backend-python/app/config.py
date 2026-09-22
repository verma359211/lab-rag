"""Application settings used by the Python RAG backend.

Keeping settings in one file makes the rest of the project easier to read.
Other modules can ask this file for a database URL or model name without
needing to know where those values came from.

The real secrets stay in the root ``.env`` file. This file only reads them.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


# ``config.py`` is inside ``backend-python/app``. Going up two folders takes us
# to the project root, where the shared .env file lives.
ROOT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

# Load the variables before any function tries to read them with os.getenv().
load_dotenv(ROOT_ENV_FILE)


# These values are kept here so they are easy to find and change later.
COLLECTION_NAME = "rag_documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVAL_LIMIT = 5


def require_env(name: str) -> str:
    """Return one required environment variable.

    A missing API key or database URL would cause a confusing error later.
    Checking it here gives us a short and useful message immediately.
    """

    value = os.getenv(name)

    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")

    return value


def get_database_url() -> str:
    """Return the database URL in the format expected by LangChain PGVector.

    The Node backend can use ``postgresql://`` directly. The Python PGVector
    package uses SQLAlchemy and psycopg, so we add ``+psycopg`` to the driver
    name while leaving the host, port, username, and database unchanged.
    """

    database_url = require_env("DATABASE_URL")

    if database_url.startswith("postgresql+psycopg://"):
        return database_url

    if database_url.startswith("postgresql://"):
        return database_url.replace(
            "postgresql://",
            "postgresql+psycopg://",
            1,
        )

    if database_url.startswith("postgres://"):
        return database_url.replace(
            "postgres://",
            "postgresql+psycopg://",
            1,
        )

    return database_url


def get_embedding_model_name() -> str:
    """Return the Gemini embedding model configured for this project."""

    return os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")


def get_chat_model_name() -> str:
    """Return the Groq chat model configured for answer generation."""

    return os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
