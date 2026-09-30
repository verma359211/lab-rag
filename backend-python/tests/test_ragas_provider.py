"""Test the Groq compatibility adapter without making provider API calls."""

from openai import AsyncOpenAI

from evals.evaluator_config import EvaluatorConfig
from evals.ragas_metrics import RagasEvaluator


def groq_config() -> EvaluatorConfig:
    """Create a minimal evaluator configuration containing a fake API key."""

    return EvaluatorConfig(
        llm_provider="groq",
        llm_model="openai/gpt-oss-20b",
        llm_api_key="test-key",
        llm_base_url=None,
        embedding_provider="google",
        embedding_model="models/gemini-embedding-001",
        embedding_api_key="test-key",
        embedding_base_url=None,
    )


def test_groq_uses_ragas_openai_adapter() -> None:
    """RAGAS must not send the Groq SDK through its broken generic adapter."""

    config = groq_config()

    assert RagasEvaluator._get_ragas_provider(config) == "openai"


def test_groq_client_uses_official_compatible_endpoint() -> None:
    """The compatibility client must still send requests to Groq."""

    client = RagasEvaluator._build_llm_client(groq_config())

    assert isinstance(client, AsyncOpenAI)
    assert str(client.base_url) == "https://api.groq.com/openai/v1/"
