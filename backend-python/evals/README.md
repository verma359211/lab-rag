# Offline RAG evaluation

This folder evaluates the same retrieval and generation functions used by the
FastAPI application. It does not maintain a second RAG pipeline.

The two checked-in benchmarks serve different purposes:

- `datasets/v1/golden.core15.jsonl` is the fast 15-question development
  benchmark. Retrieval and full RAGAS runs use it by default.
- `datasets/v1/golden.mapped.jsonl` is the complete 60-question benchmark for
  final checks and major regression testing.

## 1. Create the golden dataset

Copy `datasets/v1/golden.example.jsonl` to `datasets/v1/golden.jsonl`. Replace
the example with questions and answers reviewed by a person who knows the
documents. Keep one complete JSON object on each line.

```json
{
  "id": "pricing-001",
  "question": "A real question",
  "reference_answer": "A human-reviewed answer",
  "reference_sources": [
    {"source": "guide.pdf", "page": 4, "location": "Plan limits table"}
  ],
  "expected_chunk_ids": ["an-id-copied-from-the-database"],
  "category": "fact_lookup",
  "answerable": true,
  "notes": "Why this example matters"
}
```

`expected_chunk_ids` is optional. Add it only after a person has inspected the
stored chunks and decided which ones are relevant. Hit@K, Recall@K, and MRR are
reported only for rows containing these labels. The evaluator never invents
chunk IDs from the reference answer.

### Map the Northstar dataset to stored chunks

After the 58-chunk Northstar corpus has been ingested, run:

```powershell
python -m evals.map_ground_truth
```

The mapper uses exact human-cited source pages and reviewed evidence phrases;
it does not call vector search, hybrid retrieval, Groq, or RAGAS. It preserves
`golden.jsonl` and writes:

- `datasets/v1/golden.mapped.jsonl`
- `results/ground_truth_mapping_report.json`

Run the complete 60-question regression benchmark explicitly:

```powershell
python -m evals.runner --mode retrieval --dataset evals/datasets/v1/golden.mapped.jsonl --experiment full-regression
```

Use stable categories because the summary groups failures by category. Useful
future categories include `fact_lookup`, `multi_hop`, `ambiguous`,
`unanswerable`, `conflicting_sources`, and `adversarial`.

If the correct response is that the documents do not contain an answer, set
`answerable` to `false` and write that expected behavior in `reference_answer`.

## 2. Install dependencies

From `backend-python`, with the virtual environment active:

```powershell
pip install -r requirements.txt
```

RAGAS and pytest are evaluation/development dependencies. They do not change
the FastAPI endpoints.

## 3. Run retrieval-only evaluation

Start PostgreSQL/pgvector and make sure the same documents used by the golden
dataset are already ingested. The backend server does not need to be running.

```powershell
python -m evals.runner --mode retrieval --experiment hybrid-rrf-baseline
```

With no `--dataset` argument, this uses `golden.core15.jsonl`.

The runner waits five seconds between questions by default. Temporary `429`,
`RESOURCE_EXHAUSTED`, RPM, and TPM errors are retried up to four times with
waits of 5, 10, 20, and 40 seconds. A provider `Retry-After` header takes
priority when available. These controls affect evaluation only, not normal
chat requests. Retrieval-only keyword runs skip the fixed delay because they
use local PostgreSQL and do not make model API calls.

Override the defaults when needed:

```powershell
python -m evals.runner --mode retrieval --request-delay 8 --max-retries 5 --initial-retry-delay 10
```

Use `--request-delay 0 --max-retries 0` to disable pacing and retries.

This calls the application's real vector search, keyword search, and RRF
fusion. It does not call the answer model or any RAGAS evaluator model.

Compare another retrieval strategy with the same dataset:

```powershell
python -m evals.runner --mode retrieval --search-mode vector --experiment vector-baseline
python -m evals.runner --mode retrieval --search-mode keyword --experiment keyword-baseline
```

Run the reranker against the same core questions without changing the saved
Hybrid + RRF baseline:

```powershell
python -m evals.runner --mode retrieval --search-mode hybrid_rerank --experiment hybrid-cross-encoder-rerank
```

The report stores the original RRF rank, reranker rank, reranker score, and
reranking latency for every candidate. Compare Hit@1, Recall@1, MRR, context
precision, and latency with the baseline before making reranking the default.

Some questions have more than one independently complete evidence path. Those
records use `expected_evidence_sets`, where each inner list is one valid chunk
combination. Recall uses the best-matched complete path, while precision and
Hit@K accept relevant chunks from any approved path. Records without this field
continue using `expected_chunk_ids` exactly as before.

## 4. Configure and run full evaluation

Full mode first runs the application's real RAG answer flow. It then uses
separately configured evaluator models for RAGAS. Add these values to the root
`.env` file:

```dotenv
EVAL_LLM_PROVIDER=openai
EVAL_LLM_MODEL=gpt-4o-mini
EVAL_LLM_API_KEY=...
EVAL_LLM_MAX_TOKENS=4096
EVAL_EMBEDDING_PROVIDER=openai
EVAL_EMBEDDING_MODEL=text-embedding-3-small
EVAL_EMBEDDING_API_KEY=...
```

Supported evaluator LLM providers are `openai` and `groq`. Supported evaluator
embedding providers are `openai` and `google`. Optional
`EVAL_LLM_BASE_URL` and `EVAL_EMBEDDING_BASE_URL` support compatible custom
OpenAI endpoints. The artifact records provider/model names but never keys.

```powershell
python -m evals.runner --mode full --experiment hybrid-rrf-full-baseline
```

Full mode also uses `golden.core15.jsonl` unless `--dataset` is provided.

For answerable records, full mode reports context precision, context recall,
faithfulness, answer relevancy, and factual correctness. Context precision and
recall use the human-reviewed chunk labels. The remaining three measurements
use the judge model and may vary when their exact result is not already cached.

Records marked `answerable: false` use `abstention_correctness` instead of the
three normal generation metrics. This prevents a correct "cannot determine"
answer from receiving an automatic zero merely because it is noncommittal.

## 5. Read the results

Each run creates new timestamped files under `evals/results`:

- JSON keeps the complete nested evidence and configuration.
- CSV flattens each question into one row for filtering or spreadsheet use.

Files contain vector candidates, keyword candidates, fused candidates,
reranked candidates, final contexts, IDs, ranks, scores, references, generated
answers, latency, and errors. Generated artifacts are ignored by Git by
default so local paid runs are not committed accidentally.

The terminal summary includes sample counts, deterministic retrieval metrics,
RAGAS averages in full mode, average/p50/p95 latency, and failures by category.
A failed question or individual RAGAS metric is recorded and does not erase the
rest of the experiment.

## Evaluation dashboard

The React application now has a separate **Evaluations** view. It calls the
FastAPI evaluation routes directly; it does not execute shell commands.

Start FastAPI and Vite normally, open the frontend, and choose Evaluations.
The dashboard can:

- start a core-15 or full-60 run;
- run retrieval-only or full RAGAS evaluation;
- show vector, keyword, fusion, generation, and metric events;
- display retrieval and RAGAS metric bars;
- inspect generated answers, references, chunks, evaluator reasons, generated
  judge claims, per-claim verdicts, and the exact context shown to the judge;
- cancel and resume a run;
- compare a selected run with another completed run.

Only one evaluation runs at a time. Local job state is checkpointed under
`evals/results/runs` after retrieval, generation, and every completed RAGAS
metric. That directory is ignored by Git.

### Token-aware Groq pacing

Full evaluation uses a conservative rolling token estimate before every RAGAS
metric. `groq_tpm_limit` is configured in the dashboard and defaults to 8,000;
use the exact limit shown in your Groq account.

Successful Groq responses update the dashboard with safe rate-limit headers:
remaining tokens, token limit, and reset time. If a 429 persists, the job saves
its current question and completed metric scores, then changes to
`paused_rate_limit`. Resume continues only unfinished work.

Daily token limits pause immediately because short retries cannot fix them.
Per-minute limits honor the provider `retry-after` header when it is available.
Token values shown before a request are estimates; provider header values are
authoritative.

Malformed or truncated structured judge responses are retried twice without
rerunning retrieval, answer generation, or already completed metrics. RAGAS's
1,024-token default is too small for some reasoning-model JSON responses, so
the evaluator uses `EVAL_LLM_MAX_TOKENS=4096` by default. If a metric still
fails, the dashboard marks the run `completed_with_errors`; Resume retries only
the missing metric and reuses every successful cached result.

### Token-saving evaluation cache

Full evaluation avoids judge calls that cannot improve the result:

- Context precision and context recall are calculated directly from the
  human-reviewed `expected_chunk_ids`. Unanswerable records have no positive
  evidence labels, so these two metrics are left unavailable for those rows.
- Adjacent retrieved chunks are restored to document order only for judge
  evaluation. Exact splitter overlap is included once instead of being sent to
  the judge twice. Retrieval ranks, RRF scores, generation inputs, and
  retrieval metrics remain unchanged.
- Faithfulness receives the question and retrieved context, then extracts and
  judges the answer's substantive claims in one model call. Its trace saves
  each claim, verdict, and short reason.
- Factual correctness compares the whole generated answer with the reference
  in light of the question. It returns a semantic score from 0 to 1 plus a
  short explanation in one model call.
- Answer relevancy generates one comparison question instead of three, then
  calculates its embedding similarity with the original question.
- Successful faithfulness, answer relevancy, factual correctness, and
  abstention correctness judgments
  are cached under `evals/results/cache/judgments`. The key includes the RAGAS
  version, evaluator and embedding model names, metric inputs, generated answer,
  and ordered contexts used by that metric.
For an answerable question this reduces the generation-quality evaluation to
three judge calls: one each for faithfulness, answer relevancy, and factual
correctness. Changing any key input creates a new cache entry. The
question-aware v3 evaluator has a new cache identity, so older judgments are
not reused. Errors and partial judgments are never cached. Cache files are
local generated artifacts and are ignored by Git. In the dashboard,
deterministic context scores are marked `human labels` and reused judge results
are marked `cached`.

The dashboard API is intentionally small:

```text
POST /evaluations
GET  /evaluations
GET  /evaluations/{run_id}
POST /evaluations/{run_id}/cancel
POST /evaluations/{run_id}/resume
```

## 6. Run offline tests

```powershell
python -m pytest
```

The metric and parser tests use artificial local data. They do not connect to
PostgreSQL and do not call an LLM.

## Current RAGAS compatibility note

RAGAS 0.4.3 still imports a Vertex AI path removed by modern
`langchain-community`, even when this project is not using Vertex AI. The
isolated `ragas_compat.py` shim supplies only that unused import during evaluator
startup. It does not change metric behavior and can be deleted when RAGAS fixes
the upstream import.

RAGAS 0.4.3 also sends its Groq SDK client through an adapter that expects an
Anthropic-style `messages` property. The evaluator avoids that upstream bug by
using Groq's official OpenAI-compatible endpoint with the asynchronous OpenAI
client required by RAGAS `ascore()`. Calls still go to Groq and still use
`EVAL_LLM_MODEL`; the normal application answer flow continues to use
`ChatGroq`.
