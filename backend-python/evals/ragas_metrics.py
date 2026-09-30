"""Run model-judged RAGAS metrics for a completed application response."""

import asyncio
import time
from importlib.metadata import version

from typing import Any

from evals.abstention_metric import (
    AbstentionInput,
    AbstentionOutput,
    build_abstention_prompt,
)
from evals.evaluator_config import EvaluatorConfig
from evals.evaluation_prompts import (
    FactualCorrectnessJudgment,
    FaithfulnessJudgment,
    RelevancyQuestion,
    answer_relevancy_prompt,
    factual_correctness_prompt,
    faithfulness_prompt,
)
from evals.judge_cache import JudgeCache
from evals.rate_limit import (
    RetrySettings,
    run_with_metric_error_retry,
    run_with_rate_limit_retry,
)
from evals.ragas_compat import install_vertex_import_shim
from evals.token_budget import TokenBudget, estimate_tokens


class RagasEvaluator:
    """Build evaluator models once and score many dataset examples."""

    IMPLEMENTATION_VERSION = "question-aware-efficient-ragas-v3"
    ANSWERABLE_METRICS = (
        "faithfulness",
        "answer_relevancy",
        "factual_correctness",
    )
    UNANSWERABLE_METRICS = ("abstention_correctness",)

    def __init__(
        self,
        config: EvaluatorConfig,
        retry_settings: RetrySettings | None = None,
        metric_delay_seconds: float = 0,
        tokens_per_minute: int | None = None,
        cache: JudgeCache | None = None,
    ):
        # The shim must run before the first RAGAS import. See its module-level
        # explanation for the exact upstream compatibility issue.
        install_vertex_import_shim()

        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerRelevancy

        self.latest_quota: dict[str, Any] = {}
        llm_client = self._build_llm_client(config, self._capture_quota)
        ragas_provider = self._get_ragas_provider(config)
        evaluator_llm = llm_factory(
            model=config.llm_model,
            provider=ragas_provider,
            client=llm_client,
            temperature=0,
            max_tokens=config.llm_max_tokens,
        )
        evaluator_embeddings = self._build_embeddings(config)
        self.evaluator_llm = evaluator_llm

        # Context precision and recall are calculated from the dataset's
        # human-reviewed chunk IDs. Only generation-quality metrics need a
        # judge model.
        self.metrics = {
            "answer_relevancy": AnswerRelevancy(
                evaluator_llm,
                evaluator_embeddings,
                strictness=1,
            ),
        }
        self.retry_settings = retry_settings or RetrySettings()
        self.metric_delay_seconds = metric_delay_seconds
        self.config = config
        self.cache = cache or JudgeCache()
        self.ragas_version = version("ragas")
        self.token_budget = (
            TokenBudget(tokens_per_minute)
            if tokens_per_minute
            else None
        )

    def _capture_quota(self, headers: Any) -> None:
        """Keep only safe Groq quota headers from the latest judge response."""

        names = (
            "x-ratelimit-limit-tokens",
            "x-ratelimit-remaining-tokens",
            "x-ratelimit-reset-tokens",
            "x-ratelimit-limit-requests",
            "x-ratelimit-remaining-requests",
            "x-ratelimit-reset-requests",
        )
        self.latest_quota = {
            name: headers.get(name)
            for name in names
            if headers.get(name) is not None
        }

    @property
    def metric_names(self) -> list[str]:
        """Return judge metrics used for ordinary answerable questions."""

        return list(self.ANSWERABLE_METRICS)

    def metric_names_for(self, answerable: bool) -> list[str]:
        """Choose normal metrics or the dedicated abstention judgment."""

        names = self.ANSWERABLE_METRICS if answerable else self.UNANSWERABLE_METRICS
        return list(names)

    def _cache_identity(self) -> dict[str, Any]:
        """Identify every model setting that can change a metric score."""

        evaluator_identity = self.config.public_values()
        # A larger completion ceiling fixes truncated responses but does not
        # change a successful judgment's intended inputs. Keep earlier valid
        # cache entries reusable after increasing this safety limit.
        evaluator_identity.pop("llm_max_tokens", None)
        return {
            "ragas_version": self.ragas_version,
            "evaluator": evaluator_identity,
            # Base URLs distinguish OpenAI itself from local/proxy-compatible
            # servers. API keys are intentionally never stored or hashed.
            "llm_base_url": self.config.llm_base_url,
            "embedding_base_url": self.config.embedding_base_url,
            # Changing this value creates fresh cache keys. Older numeric-only
            # judgments must not leak into the new traceable evaluation.
            "implementation": self.IMPLEMENTATION_VERSION,
        }

    def _metric_settings(self, name: str) -> dict[str, Any]:
        """Include score-changing metric options in persistent cache keys."""

        if name == "abstention_correctness":
            return {"policy": "correct_abstention_v1"}
        if name == "faithfulness":
            return {"policy": "question_and_context_claims_v1", "calls": 1}
        if name == "answer_relevancy":
            return {"strictness": 1, "prompt": "compact_relevancy_v1"}
        if name == "factual_correctness":
            return {"policy": "whole_answer_semantic_v1", "calls": 1}
        return {}

    def _judgment_cache_inputs(
        self,
        name: str,
        question: str,
        answer: str,
        reference_answer: str,
        contexts: list[str],
    ) -> dict[str, Any]:
        """Describe one judgment completely so cache reuse stays safe."""

        common = {
            **self._cache_identity(),
            "metric": name,
            "metric_settings": self._metric_settings(name),
        }
        if name == "faithfulness":
            return {
                **common,
                "question": question,
                "answer": answer,
                "contexts": contexts,
            }
        if name == "answer_relevancy":
            return {**common, "question": question, "answer": answer}
        if name == "factual_correctness":
            return {
                **common,
                "question": question,
                "answer": answer,
                "reference_answer": reference_answer,
            }
        if name == "abstention_correctness":
            return {
                **common,
                "question": question,
                "answer": answer,
                "reference_answer": reference_answer,
                "contexts": contexts,
            }
        raise ValueError(f"Unknown judge metric: {name}")

    @staticmethod
    def _get_ragas_provider(config: EvaluatorConfig) -> str:
        """Return the provider name understood by RAGAS 0.4.3.

        RAGAS 0.4.3 incorrectly expects the Groq SDK to expose an Anthropic-like
        ``messages`` property. Groq also provides an official OpenAI-compatible
        API, so the evaluator uses RAGAS's working OpenAI adapter while requests
        still go to Groq.
        """

        if config.llm_provider == "groq":
            return "openai"

        return config.llm_provider

    @staticmethod
    def _build_llm_client(config: EvaluatorConfig, on_response: Any = None) -> Any:
        """Create the client selected by the evaluator-only provider setting."""

        import httpx
        from openai import AsyncOpenAI

        async def capture_response(response: httpx.Response) -> None:
            if on_response:
                on_response(response.headers)

        http_client = httpx.AsyncClient(
            event_hooks={"response": [capture_response]},
        )

        if config.llm_provider == "groq":
            # This is Groq's documented OpenAI-compatible endpoint. A custom
            # base URL can still be supplied for a proxy or compatible gateway.
            return AsyncOpenAI(
                api_key=config.llm_api_key,
                http_client=http_client,
                base_url=(
                    config.llm_base_url
                    or "https://api.groq.com/openai/v1"
                ),
            )

        client_options: dict[str, str] = {"api_key": config.llm_api_key}
        if config.llm_base_url:
            client_options["base_url"] = config.llm_base_url
        return AsyncOpenAI(http_client=http_client, **client_options)

    @staticmethod
    def _build_embeddings(config: EvaluatorConfig) -> Any:
        """Create the embedding wrapper needed by answer relevancy."""

        install_vertex_import_shim()
        from ragas.embeddings import GoogleEmbeddings, OpenAIEmbeddings

        if config.embedding_provider == "google":
            from google import genai

            client = genai.Client(api_key=config.embedding_api_key)
            return GoogleEmbeddings(client=client, model=config.embedding_model)

        from openai import OpenAI

        client_options: dict[str, str] = {
            "api_key": config.embedding_api_key,
        }
        if config.embedding_base_url:
            client_options["base_url"] = config.embedding_base_url

        client = OpenAI(**client_options)
        return OpenAIEmbeddings(client=client, model=config.embedding_model)

    def evaluate(
        self,
        question: str,
        answer: str,
        reference_answer: str,
        contexts: list[str],
        answerable: bool = True,
    ) -> dict[str, Any]:
        """Score one response and preserve individual metric failures."""

        output: dict[str, Any] = {
            "evaluation_version": self.IMPLEMENTATION_VERSION,
            "scores": {},
            "reasons": {},
            "errors": {},
            "token_estimates": {},
            "cache_hits": {},
            "sources": {},
            "traces": {},
        }

        for index, name in enumerate(self.metric_names_for(answerable)):
            cached = self.get_cached_metric(
                name,
                question,
                answer,
                reference_answer,
                contexts,
            )
            if cached:
                output["scores"][name] = cached["score"]
                if cached.get("reason"):
                    output["reasons"][name] = cached["reason"]
                if cached.get("trace"):
                    output["traces"][name] = cached["trace"]
                output["cache_hits"][name] = True
                output["sources"][name] = "persistent_judge_cache"
                continue

            if index > 0 and self.metric_delay_seconds > 0:
                time.sleep(self.metric_delay_seconds)

            estimate = estimate_tokens(
                question,
                answer,
                reference_answer,
                *contexts,
            )
            output["token_estimates"][name] = estimate
            if self.token_budget:
                self.token_budget.reserve(
                    estimate,
                    on_wait=lambda seconds, _tokens: print(
                        f"RAGAS token budget waiting {seconds:.1f}s before {name}."
                    ),
                )

            try:
                result = self.evaluate_metric(
                    name,
                    question,
                    answer,
                    reference_answer,
                    contexts,
                    on_retry=lambda attempt, delay, _error: print(
                        f"RAGAS {name} rate limited. Retry {attempt}/"
                        f"{self.retry_settings.max_retries} in {delay:g}s."
                    ),
                    on_error_retry=lambda attempt, delay, _error: print(
                        f"RAGAS {name} returned an invalid/incomplete result. "
                        f"Retry {attempt}/2 in {delay:g}s."
                    ),
                )
                output["scores"][name] = result["score"]
                output["cache_hits"][name] = result.get("cached", False)
                output["sources"][name] = "judge_llm"
                if result["reason"]:
                    output["reasons"][name] = result["reason"]
                if result.get("trace"):
                    output["traces"][name] = result["trace"]
            except Exception as error:  # One judge failure should not lose a run.
                output["errors"][name] = str(error)

        return output

    def get_cached_metric(
        self,
        name: str,
        question: str,
        answer: str,
        reference_answer: str,
        contexts: list[str],
    ) -> dict[str, Any] | None:
        """Return an identical earlier judgment without making an API call."""

        inputs = self._judgment_cache_inputs(
            name,
            question,
            answer,
            reference_answer,
            contexts,
        )
        key = self.cache.judgment_key(inputs)
        cached = self.cache.get_judgment(key)
        if cached is None:
            return None
        return {**cached, "cached": True}

    async def _faithfulness_with_trace(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> dict[str, Any]:
        """Extract and judge substantive claims in one model call."""

        prompt = faithfulness_prompt(question, answer, contexts)
        judged = await self.evaluator_llm.agenerate(
            prompt,
            FaithfulnessJudgment,
        )
        verdicts = [claim.model_dump() for claim in judged.claims]
        if not verdicts:
            return {
                "score": 0.0,
                "reason": "The judge did not extract any answer statements.",
                "trace": {"statements": [], "verdicts": []},
            }

        supported = sum(item["verdict"] for item in verdicts)
        score = supported / len(verdicts)
        return {
            "score": float(score),
            "reason": None,
            "trace": {
                "statements": [item["statement"] for item in verdicts],
                "verdicts": verdicts,
                "supported_statement_count": supported,
                "statement_count": len(verdicts),
            },
        }

    async def _answer_relevancy_with_trace(
        self,
        question: str,
        answer: str,
    ) -> dict[str, Any]:
        """Run answer relevancy and retain its questions and hard-gate flags."""

        import numpy as np
        metric = self.metrics["answer_relevancy"]
        prompt = answer_relevancy_prompt(answer)
        judged = await self.evaluator_llm.agenerate(prompt, RelevancyQuestion)
        if not judged.question:
            return {
                "score": 0.0,
                "reason": "The judge did not generate a comparison question.",
                "trace": {
                    "generated_questions": [],
                    "noncommittal_flags": [],
                    "cosine_similarities": [],
                    "hard_zero_applied": False,
                },
            }

        question_vector = np.asarray(await metric.embeddings.aembed_text(question))
        generated_vector = np.asarray(
            await metric.embeddings.aembed_text(judged.question)
        )
        norm = np.linalg.norm(question_vector) * np.linalg.norm(generated_vector)
        similarity = (
            float(np.dot(question_vector, generated_vector) / norm)
            if norm
            else 0.0
        )
        hard_zero = bool(judged.noncommittal)
        score = similarity * int(not hard_zero)
        return {
            "score": score,
            "reason": None,
            "trace": {
                "generated_questions": [judged.question],
                "noncommittal_flags": [int(judged.noncommittal)],
                "cosine_similarities": [similarity],
                "hard_zero_applied": hard_zero,
            },
        }

    async def _abstention_correctness(
        self,
        question: str,
        answer: str,
        reference_answer: str,
        contexts: list[str],
    ) -> dict[str, Any]:
        """Judge an intentionally unanswerable example without penalizing it."""

        prompt_input = AbstentionInput(
            question=question,
            response=answer,
            reference=reference_answer,
            contexts="\n".join(contexts),
        )
        prompt = build_abstention_prompt(prompt_input)
        judged = await self.evaluator_llm.agenerate(prompt, AbstentionOutput)
        return {
            "score": float(judged.verdict),
            "reason": judged.reason,
            "trace": {
                "verdict": int(judged.verdict),
                "reason": judged.reason,
            },
        }

    async def _factual_correctness_with_trace(
        self,
        question: str,
        answer: str,
        reference_answer: str,
    ) -> dict[str, Any]:
        """Judge whole-answer semantic correctness in one model call."""

        prompt = factual_correctness_prompt(
            question,
            answer,
            reference_answer,
        )
        judged = await self.evaluator_llm.agenerate(
            prompt,
            FactualCorrectnessJudgment,
        )
        return {
            "score": float(judged.score),
            "reason": judged.reason,
            "trace": {
                "score": float(judged.score),
                "reason": judged.reason,
                "correct_points": judged.correct_points,
                "problems": judged.problems,
            },
        }

    def evaluate_metric(
        self,
        name: str,
        question: str,
        answer: str,
        reference_answer: str,
        contexts: list[str],
        on_retry: Any = None,
        on_error_retry: Any = None,
        sleep: Any = None,
    ) -> dict[str, Any]:
        """Run one named metric so a job can checkpoint between metrics."""

        cached = self.get_cached_metric(
            name,
            question,
            answer,
            reference_answer,
            contexts,
        )
        if cached:
            return cached

        known_metrics = set(self.ANSWERABLE_METRICS + self.UNANSWERABLE_METRICS)
        if name not in known_metrics:
            raise ValueError(f"Unknown RAGAS metric: {name}")

        if name == "faithfulness":
            operation = lambda: asyncio.run(
                self._faithfulness_with_trace(question, answer, contexts)
            )
        elif name == "answer_relevancy":
            operation = lambda: asyncio.run(
                self._answer_relevancy_with_trace(question, answer)
            )
        elif name == "factual_correctness":
            operation = lambda: asyncio.run(
                self._factual_correctness_with_trace(
                    question,
                    answer,
                    reference_answer,
                )
            )
        else:
            operation = lambda: asyncio.run(
                self._abstention_correctness(
                    question,
                    answer,
                    reference_answer,
                    contexts,
                )
            )

        retry_options = {
            "operation": operation,
            "settings": self.retry_settings,
            "on_retry": on_retry,
        }
        if sleep is not None:
            retry_options["sleep"] = sleep

        metric_retry_options = {
            "operation": lambda: run_with_rate_limit_retry(**retry_options),
            "max_retries": 2,
            "initial_delay_seconds": 2.0,
            "on_retry": on_error_retry,
        }
        if sleep is not None:
            metric_retry_options["sleep"] = sleep

        raw_result = run_with_metric_error_retry(
            **metric_retry_options,
        )
        if isinstance(raw_result, dict):
            result = raw_result
        else:
            result = {
                "score": raw_result.value,
                "reason": raw_result.reason,
            }

        cache_inputs = self._judgment_cache_inputs(
            name,
            question,
            answer,
            reference_answer,
            contexts,
        )
        cache_key = self.cache.judgment_key(cache_inputs)
        cached_value = {
            "score": result["score"],
            "reason": result.get("reason"),
            "trace": result.get("trace"),
        }
        self.cache.save_judgment(cache_key, cache_inputs, cached_value)
        return {**result, "cached": False}
