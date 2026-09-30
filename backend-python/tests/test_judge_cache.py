"""Verify expensive evaluator work can be reused safely across runs."""

import asyncio

from evals.evaluator_config import EvaluatorConfig
from evals.evaluation_prompts import (
    ClaimVerdict,
    FactualCorrectnessJudgment,
    FaithfulnessJudgment,
)
from evals.judge_cache import JudgeCache
from evals.ragas_metrics import RagasEvaluator


def evaluator_config() -> EvaluatorConfig:
    """Create non-secret model identifiers for cache-key tests."""

    return EvaluatorConfig(
        llm_provider="groq",
        llm_model="openai/gpt-oss-20b",
        llm_api_key="unused",
        llm_base_url=None,
        embedding_provider="google",
        embedding_model="models/gemini-embedding-001",
        embedding_api_key="unused",
        embedding_base_url=None,
    )


def test_judgment_cache_round_trip_and_input_safety(tmp_path) -> None:
    cache = JudgeCache(tmp_path)
    first_inputs = {"metric": "faithfulness", "answer": "A"}
    changed_inputs = {"metric": "faithfulness", "answer": "B"}
    first_key = cache.judgment_key(first_inputs)

    cache.save_judgment(
        first_key,
        first_inputs,
        {"score": 1.0, "reason": None},
    )

    assert cache.get_judgment(first_key) == {"score": 1.0, "reason": None}
    assert cache.judgment_key(changed_inputs) != first_key
    assert cache.get_judgment(cache.judgment_key(changed_inputs)) is None


class FakeJudge:
    """Return local structured judgments while recording the exact prompt."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def agenerate(self, prompt: str, output_type):
        self.prompts.append(prompt)
        if output_type is FactualCorrectnessJudgment:
            return FactualCorrectnessJudgment(
                score=1.0,
                reason="Same meaning.",
                correct_points=["The meaning matches."],
                problems=[],
            )
        if output_type is FaithfulnessJudgment:
            return FaithfulnessJudgment(
                claims=[
                    ClaimVerdict(
                        statement="The result is 20.",
                        verdict=1,
                        reason="Simple arithmetic.",
                    )
                ]
            )
        raise AssertionError(f"Unexpected output type: {output_type}")


def make_evaluator(judge: FakeJudge) -> RagasEvaluator:
    """Build only the state needed by prompt-focused helper tests."""

    evaluator = RagasEvaluator.__new__(RagasEvaluator)
    evaluator.evaluator_llm = judge
    return evaluator


def test_metric_selection_separates_abstentions() -> None:
    """Unanswerable records must not receive noncommittal penalties."""

    evaluator = RagasEvaluator.__new__(RagasEvaluator)

    assert evaluator.metric_names_for(True) == [
        "faithfulness",
        "answer_relevancy",
        "factual_correctness",
    ]
    assert evaluator.metric_names_for(False) == ["abstention_correctness"]


def test_factual_correctness_is_question_aware_and_traceable() -> None:
    """Semantic correctness must receive the question and preserve its reason."""

    judge = FakeJudge()
    evaluator = make_evaluator(judge)

    result = asyncio.run(
        evaluator._factual_correctness_with_trace(
            "What is the result?",
            "generated answer",
            "reference answer",
        )
    )

    assert result["score"] == 1.0
    assert result["trace"]["reason"] == "Same meaning."
    assert "What is the result?" in judge.prompts[0]


def test_faithfulness_uses_question_and_context_in_one_call() -> None:
    """Scenario facts and retrieved rules must reach one compact judgment."""

    judge = FakeJudge()
    evaluator = make_evaluator(judge)

    result = asyncio.run(
        evaluator._faithfulness_with_trace(
            "Ten minus two equals what?",
            "The result is eight.",
            ["Subtract two from ten."],
        )
    )

    assert result["score"] == 1.0
    assert len(judge.prompts) == 1
    assert "Ten minus two equals what?" in judge.prompts[0]
    assert "Subtract two from ten." in judge.prompts[0]
