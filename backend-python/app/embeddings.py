"""Create the embedding model used by ingestion and retrieval.

An embedding model turns text into a list of numbers called a vector. Texts
with similar meanings usually receive nearby vectors. We use those vectors to
find PDF chunks that are semantically related to a user's question.
"""

from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import get_embedding_model_name, require_env


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create one Gemini embedding client and reuse it.

    FastAPI calls this function for many requests. ``lru_cache`` remembers the
    first client, so we do not repeatedly rebuild the same object.
    """

    return GoogleGenerativeAIEmbeddings(
        model=get_embedding_model_name(),
        # The project already calls this value GEMINI_API_KEY in its root .env.
        google_api_key=require_env("GEMINI_API_KEY"),
    )
