# Architecture Decision Log

Use this file for decisions that materially affect data sources, storage, models, frameworks, data contracts, deployment, or security. Add new entries instead of rewriting old decisions; mark superseded decisions and link to their replacements.

## Entry Template

### ADR-NNN: Short title

- **Status:** Proposed | Accepted | Superseded
- **Date:** YYYY-MM-DD

**Decision**

State what is being chosen.

**Context**

Describe the problem, constraints, and current project stage.

**Alternatives considered**

- Alternative and its main trade-off.

**Reason**

Explain why this choice best fits the current constraints.

**Consequences**

- Record both benefits and costs.

**Review or migration trigger**

State the observable condition that should cause this decision to be revisited.

---

## ADR-001: Use batch arXiv ingestion with local querying

- **Status:** Accepted
- **Date:** 2026-07-10

**Decision**

Fetch arXiv papers through manually triggered batches, persist them locally, and answer user questions without calling arXiv at query time.

**Context**

The first project stage prioritizes learning a visible data pipeline, stable experiments, and respectful use of the arXiv API. Only arXiv is in scope initially.

**Alternatives considered**

- Query arXiv for every user question: fresher results, but higher latency, rate-limit risk, and less reproducibility.
- Run scheduled background ingestion: better freshness, but adds scheduling and operational complexity too early.
- Manually batch-fetch and store locally: less fresh, but simple and reproducible.

**Reason**

Manual batch ingestion isolates data acquisition from retrieval and generation. This makes each stage easier to inspect, test, and understand while avoiding high-frequency API calls.

**Consequences**

- User queries remain fast and independent of arXiv availability.
- Experiments can run repeatedly against the same dataset.
- Results become stale until ingestion is run again.
- Deduplication, update handling, and a visible knowledge-base timestamp are required.

**Review or migration trigger**

Revisit this decision when users require predictable freshness, manual updates become burdensome, or the project moves to multi-user deployment. The next step would be scheduled incremental ingestion, not unrestricted per-question API access.

---

## ADR-002: Separate paper identity from content state

- **Status:** Accepted
- **Date:** 2026-07-11

**Decision**

Use the versionless arXiv ID as the stable paper identity. Preserve the versioned ID and timestamps, normalize title and abstract text, and compute a SHA-256 hash from the normalized embedding content.

**Context**

One paper can appear as `v1`, `v2`, and later revisions. Treating each revision as a new paper would create duplicates, while relying only on `updated_at` would trigger unnecessary embedding work for metadata-only changes.

**Alternatives considered**

- Use the full versioned ID as the key: simple, but duplicates revisions.
- Re-embed on every timestamp change: safe, but performs unnecessary work.
- Compare full text fields directly: valid, but less convenient to persist and audit across runs.

**Reason**

Stable identity supports upsert behavior, while the content hash records whether the exact normalized embedding input changed.

**Consequences**

- Revisions update one logical paper record.
- Embeddings must be regenerated when the content hash changes.
- Normalization rules become part of the data contract and must remain deterministic.

**Review or migration trigger**

Revisit the hash input when retrieval begins using fields beyond title and abstract, or when normalization rules change.

---

## ADR-003: Publish JSONL snapshots atomically

- **Status:** Accepted
- **Date:** 2026-07-11

**Decision**

Write each completed batch to a temporary file in the destination directory, flush it, and replace the final `.jsonl` path only after every record is serialized successfully.

**Context**

Downstream processing must not mistake a partially written batch for a complete, reproducible snapshot. Initial batches are small enough to refetch after failure.

**Alternatives considered**

- Append directly to the final file: preserves partial work, but exposes incomplete batches.
- Add per-paper checkpoints and ingestion-run state: supports resume, but adds complexity before batch sizes justify it.
- Temporary file plus atomic replacement: simple and prevents partial publication.

**Reason**

Atomic publication gives downstream modules one clear rule: consume final `.jsonl` files and ignore temporary files.

**Consequences**

- Failed batches do not replace an existing valid snapshot.
- Temporary files are cleaned after write failures.
- A failed small batch is fetched again and may contain slightly different latest results.

**Review or migration trigger**

Introduce per-paper checkpoints and ingestion-run records when batches become expensive to repeat or exact resume becomes a requirement.

---

## ADR-004: Use SQLite as the local current-state store

- **Status:** Accepted
- **Date:** 2026-07-11

**Decision**

Import completed JSONL snapshots into a local SQLite `papers` table. Use the versionless `arxiv_id` as the primary key, compare `updated_at` before replacing current content, and import each snapshot inside one transaction.

**Context**

JSONL snapshots preserve historical fetch results but do not provide one deduplicated view of the latest paper state. The project is local, single-user, and currently handles small batches.

**Alternatives considered**

- Query JSONL files directly: no database, but makes deduplication and current-state queries difficult.
- Use PostgreSQL immediately: stronger concurrency, but adds deployment and administration without current need.
- Use SQLite: local, transactional, and available in the Python standard library.

**Reason**

SQLite provides primary keys, transactions, and repeatable local queries with minimal operational complexity.

**Consequences**

- Reimporting the same snapshot is idempotent.
- Newer revisions update one row; older revisions cannot downgrade it.
- Invalid records roll back the complete snapshot import.
- SQLite is not intended for high-concurrency multi-user deployment.

**Review or migration trigger**

Move to PostgreSQL when concurrent writers, hosted multi-user access, or operational scaling make a local database insufficient.

---

## ADR-005: Start semantic retrieval with multilingual E5

- **Status:** Accepted
- **Date:** 2026-07-11

**Decision**

Use `intfloat/multilingual-e5-small` with `query:` and `passage:` prefixes. Normalize vectors and first calculate cosine similarity in memory before adding ChromaDB.

**Context**

Users will ask questions in Chinese while arXiv titles and abstracts are primarily English. The first dataset contains only a few papers, so an in-memory experiment can expose retrieval behavior without hiding it behind a vector database.

**Alternatives considered**

- Use an English-only embedding model: lighter in some cases, but unsuitable for the core cross-language requirement.
- Use a hosted embedding API: reduces local model setup, but introduces cost, network dependency, and private query transmission.
- Add ChromaDB immediately: provides persistence, but obscures the basic similarity calculation during learning.

**Reason**

Multilingual E5 supports Chinese-to-English retrieval locally, and normalized vectors make a dot product equivalent to cosine similarity.

**Consequences**

- The first model download and CPU load take time and disk space.
- Query and passage prefixes are part of the embedding contract.
- Similarity scores alone do not guarantee relevance; filtering, thresholds, evaluation, or reranking may still be required.
- Vectors are currently recomputed on every demo run.

**Review or migration trigger**

Revisit the model when evaluation shows inadequate cross-language accuracy, latency becomes unacceptable, or deployment hardware requires a different model size.

---

## ADR-006: Treat ChromaDB as a rebuildable derived index

- **Status:** Accepted
- **Date:** 2026-07-11

**Decision**

Persist paper vectors in a cosine-distance ChromaDB collection. Use the stable versionless `arxiv_id` as the vector ID, store `content_hash` and `embedding_model` in metadata, and upsert only when either value changes.

**Context**

The in-memory experiment recomputed all paper vectors on every run. SQLite already contains the authoritative current paper state and can rebuild any lost vector index.

**Alternatives considered**

- Recompute vectors for every query: simple, but wastes CPU and increases latency.
- Use versioned IDs: preserves revisions in the index, but duplicates one logical paper and complicates current-state retrieval.
- Make ChromaDB authoritative: removes duplication, but weakens transactional data management and rebuildability.

**Reason**

A stable ID allows v2 to replace v1. Comparing both content and model metadata prevents stale vectors after paper updates or embedding-model changes.

**Consequences**

- Unchanged synchronization does not load the embedding model.
- Metadata-only changes do not regenerate vectors.
- Per-paper failures leave successful index updates intact and can be retried safely.
- ChromaDB may temporarily lag behind SQLite, but later synchronization restores consistency.

**Review or migration trigger**

Revisit the vector store when corpus size, concurrent access, filtering needs, or deployment requirements exceed local ChromaDB capabilities.

---

## ADR-007: Separate answerable retrieval metrics from refusal analysis

- **Status:** Accepted
- **Date:** 2026-07-11

**Decision**

Evaluate answerable questions with strict and relaxed Precision@K, Recall@K, and strict MRR. Evaluate unanswerable questions separately by recording their highest similarity and returned candidates until a formal refusal policy exists.

**Context**

An unanswerable question has zero relevant papers, so Recall would require division by zero. The project also distinguishes directly supporting papers from partially related papers.

**Alternatives considered**

- Treat no-answer Recall as zero: incorrectly suggests the retriever missed available evidence.
- Treat no-answer Recall as one: can reward a retriever that still returns convincing irrelevant papers.
- Mix partial relevance with direct relevance only: hides the difference between supporting evidence and useful background.

**Reason**

Separate reporting keeps ranking quality and refusal behavior conceptually distinct. Strict and relaxed metrics preserve both citation safety and partial topical usefulness.

**Consequences**

- Baseline metrics remain comparable when a reranker is introduced.
- Refusal accuracy cannot be reported until a threshold, reranker, or decision gate exists.
- Human labels are tied to the current corpus and must be reviewed when the corpus changes materially.

**Review or migration trigger**

Add refusal precision and recall after the system implements an explicit answer-or-refuse decision, and consider graded ranking metrics when partial relevance becomes more important.

---

## ADR-008: Use the OpenAI SDK for DeepSeek model calls

- **Status:** Accepted
- **Date:** 2026-07-11

**Decision**

Use the official OpenAI Python SDK against DeepSeek's OpenAI-compatible endpoint. Configure the API key, base URL, and model through environment variables. Attempt a request at most five times with exponential backoff, retrying only transient connection, timeout, rate-limit, and server failures.

**Context**

The basic RAG stage needs a model client without hand-written HTTP handling. Users must see reconnecting status, while permanent configuration failures must fail immediately.

**Alternatives considered**

- Write direct HTTP requests: fewer dependencies, but duplicates authentication, response parsing, and error classification.
- Introduce LangChain: offers provider wrappers, but hides the basic RAG data flow and adds abstractions not currently needed.
- Use DeepSeek through the OpenAI SDK: a small, familiar interface with provider portability.

**Reason**

The compatible SDK handles transport details while a thin project module retains explicit control over retry policy and UI status callbacks.

**Consequences**

- Changing compatible providers mainly requires environment changes.
- The SDK's internal retries are disabled to avoid hidden retries.
- Retry delays can reach 1, 2, 4, and 8 seconds before the fifth attempt.
- Prompt construction, structured-output parsing, and citation validation remain separate RAG responsibilities.

**Review or migration trigger**

Revisit the client when DeepSeek compatibility changes, streaming becomes necessary, or multiple providers require meaningfully different behavior.

---

## ADR-009: Temporarily display grounded model text without structured parsing

- **Status:** Accepted
- **Date:** 2026-07-11

**Decision**

During the current learning milestone, ask the model for display-ready Chinese text with inline arXiv IDs and show that text without converting it into a structured answer object.

**Context**

The first live refusal experiment failed because the compatible model returned an unexpected JSON field type. The current priority is understanding the basic retrieval-to-answer flow before studying structured-output parsing and validation.

**Alternatives considered**

- Keep strict parsing: preserves a strong data contract, but introduces parser behavior before the user is ready to study it.
- Add tolerant coercion and repair retries: improves robustness, but still expands the parsing subsystem.
- Display grounded text directly: simplest for the current milestone, but relies on prompt compliance for citations and refusal.

**Reason**

Direct text output keeps the runnable pipeline focused on retrieval, prompt construction, model behavior, and evidence sufficiency.

**Consequences**

- Model formatting differences no longer crash the answer pipeline.
- Citation IDs and refusal behavior are not programmatically validated.
- Model text must be treated as untrusted presentation content in a future UI.
- The live evaluation must manually inspect whether claims are supported and citations belong to retrieved papers.

**Review or migration trigger**

Reintroduce structured output and validation before citations become clickable UI data, answers are consumed by another program, or automated citation/refusal metrics are required.

---

## ADR-010: Use named, category-bounded arXiv queries for Agent coverage

- **Status:** Superseded by ADR-011
- **Date:** 2026-07-11

**Decision**

Store named arXiv query expressions in `config/arxiv_queries.json`. Cover core Agent methods and selected AI-agent application domains while bounding every query to relevant computer-science categories.

**Context**

The initial `cs.AI AND agent` query produced a usable but narrow corpus. The project remains focused on AI/Agent topics and does not need general-purpose arXiv ingestion.

**Alternatives considered**

- Use one broad `all:agent` query: simple, but includes unrelated disciplines and makes coverage difficult to inspect.
- Keep long queries only in shell history: no new configuration, but they are hard to reproduce and review.
- Store named category-bounded queries: explicit, repeatable, and easy to expand one topic at a time.

**Reason**

Named queries make ingestion scope visible without introducing a scheduling system or another configuration dependency.

**Consequences**

- Different queries will return overlapping papers; SQLite identity-based upsert handles the duplicates.
- Query names become part of the manual ingestion workflow.
- Broader coverage increases the need to review retrieval quality and evaluation labels.

**Review or migration trigger**

Revisit the query set when real user questions repeatedly lack coverage, irrelevant-paper rates increase, or another data source is introduced.

---

## ADR-011: Expand globally with explicit AI/Agent phrases

- **Status:** Accepted
- **Date:** 2026-07-11

**Decision**

Allow arXiv queries across all categories, but require explicit AI/Agent phrases such as `AI agent`, `LLM agent`, `agentic AI`, `model context protocol`, or `retrieval augmented generation`.

**Context**

The category-bounded corpus reached 4,719 papers, and the desired next milestone is at least 10,000 AI/Agent papers. Relevant work may be cross-listed or primarily published outside the selected computer-science categories.

**Alternatives considered**

- Continue category-bounded queries: higher topical precision, but may miss cross-disciplinary AI-agent work.
- Use unrestricted `all:agent`: maximizes volume, but mixes AI agents with unrelated meanings of agent.
- Remove category limits while requiring explicit AI phrases: broader coverage with a visible semantic boundary.

**Reason**

Explicit phrases preserve the project topic better than the ambiguous word `agent` while allowing interdisciplinary coverage.

**Consequences**

- The corpus can include cross-disciplinary AI-agent applications.
- Query overlap and duplicate rates will increase.
- Retrieval quality must be sampled again after expansion.
- Some false positives remain possible because keyword matching is not semantic classification.

**Review or migration trigger**

Revisit the policy if irrelevant retrieval increases materially or a later classifier can enforce AI/Agent scope more accurately.

---

## ADR-012: Use hybrid retrieval with BM25 and Cross-encoder reranking

- **Status:** Accepted
- **Date:** 2026-07-11

**Decision**

Retrieve up to 30 candidates from multilingual E5 and 30 candidates from a
SQLite FTS5 BM25 index, combine their ranks with reciprocal rank fusion, and
rerank the fused candidates with
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` before sending the final results
to the language model.

**Context**

The corpus expanded from hundreds of papers to more than 10,000. A fixed
dense Top-5 increasingly loses specifically named methods and allows broadly
similar papers to displace directly relevant evidence. Chinese questions also
need semantic retrieval, while English acronyms, method names, and exact
technical phrases benefit from lexical matching.

**Alternatives considered**

- Keep dense Top-5 only: simplest and fastest, but its ranking quality degraded
  after the corpus expansion.
- Replace the embedding model first: may improve dense retrieval, but does not
  add exact lexical matching and requires rebuilding every vector.
- Use an LLM as the reranker: flexible, but slower, more expensive, and harder
  to reproduce offline.
- Combine dense retrieval, BM25, and a small multilingual Cross-encoder: adds
  one local model and an FTS index while keeping each stage inspectable.

**Reason**

Dense and lexical retrieval have complementary failure modes. Reciprocal rank
fusion avoids comparing incompatible cosine and BM25 scores. The Cross-encoder
can jointly inspect each question and candidate abstract, but runs only on the
small fused set so local CPU cost remains bounded.

**Consequences**

- SQLite gains a generated FTS5 index and triggers that follow title or
  abstract changes.
- The runtime needs a second locally cached Hugging Face model.
- First-query latency includes lazy model loading; each later query performs
  Cross-encoder inference over at most 60 unique candidates.
- Raw reranker scores are useful for ordering but are not yet calibrated as a
  refusal threshold.
- Chinese-only terms cannot directly match English abstracts through BM25;
  E5 continues to provide the cross-language recall path.

**Review or migration trigger**

Revisit candidate counts, the fusion method, or the reranker when a refreshed
evaluation set shows inadequate Recall@30 or MRR, CPU latency is unacceptable,
or query translation becomes necessary for stronger cross-language lexical
matching.

---

## ADR-013: Use one bounded DeepSeek retrieval decision before answering

- **Status:** Accepted
- **Date:** 2026-07-12

**Decision**

After the first hybrid retrieval, send the original question and at most three
bounded paper excerpts to the configured `deepseek-chat` compatible endpoint.
Require a structured JSON decision stating whether the evidence is sufficient
and, when it is not, one standalone English rewritten query. Allow at most one
second retrieval, deduplicate both result sets, and rerank their union against
the original question. If the decision request or JSON validation fails, keep
the first retrieval and continue the existing answer path.

**Context**

The fixed RAG pipeline always answers after one retrieval even when the top
papers match only broad topic words. Raw dense and Cross-encoder scores are not
calibrated as an evidence-sufficiency threshold, and the project does not yet
have enough refreshed human labels to derive one. The existing DeepSeek client
can judge the semantic coverage and generate a cross-language query without
adding another runtime.

**Alternatives considered**

- Use a fixed reranker threshold: cheap and deterministic, but currently
  uncalibrated and unable to judge whether several papers cover the complete
  question.
- Always run multiple rewritten queries: improves recall, but adds avoidable
  latency and noise for questions already supported by the first retrieval.
- Add a local instruction model now: avoids API calls, but introduces model
  hosting and GPU contention before the control flow is validated.
- Introduce LangGraph immediately: represents the branch explicitly, but adds
  framework complexity while the workflow has only one bounded decision.

**Reason**

A single structured model decision adds the smallest useful Agentic behavior:
observe retrieved evidence, choose whether another action is needed, and stop
after a fixed bound. Combining judgment and rewriting in one request keeps the
additional token and latency cost small.

**Consequences**

- A normal hybrid RAG question now makes one additional small model request.
- An insufficient result performs one additional local retrieval and rerank.
- Model output becomes control data and therefore requires strict JSON
  validation.
- Decision quality remains probabilistic, so failures are recorded and safely
  degraded to the previous single-retrieval behavior.
- The final union is reranked against the original question to limit semantic
  drift from the rewritten query.

**Review or migration trigger**

Revisit the judge model or prompt when real queries show repeated false
decisions, API latency or cost becomes material, or a local Qwen model is ready.
Introduce an explicit state-graph framework only when more actions, retries, or
conversation-memory branches make the current orchestration hard to inspect.

---

## ADR-014: Record an in-memory structured execution trace per request

- **Status:** Accepted
- **Date:** 2026-07-12

**Decision**

Record ordered, structured events as each retrieval and generation stage
finishes. Events contain a stable stage name, display label, status, elapsed
time, and bounded details such as the query, result count, top arXiv IDs,
retrieval decision, and rewritten query. Return the immutable event sequence in
`RagResult` and render it in a Streamlit expander after the request completes.
Do not record API keys, environment values, complete model prompts, or complete
paper abstracts.

**Context**

The first Agentic RAG branch was working, but the UI exposed only the final
answer. A correct refusal after two retrievals looked identical to a fixed
single-retrieval RAG refusal, so users could not verify which modules actually
ran or why the Agent stopped. The project needs visible execution state before
it can meaningfully evaluate Agent behavior across many questions.

**Alternatives considered**

- Print only to the terminal: simple, but invisible to Streamlit users and hard
  to associate with a specific request.
- Add individual UI fields directly to every module: quick initially, but
  couples retrieval code to Streamlit and cannot preserve a general event
  order.
- Add Langfuse or LangSmith now: provides persistence and dashboards, but adds
  an external service before the local trace contract is understood.
- Return one structured in-memory trace: keeps retrieval independent from UI,
  is testable offline, and provides the data contract later evaluation needs.

**Reason**

Per-stage events make the current conditional workflow inspectable without
introducing an orchestration or observability framework. The same trace can be
shown in Streamlit now and later persisted or aggregated for evaluation.

**Consequences**

- Hybrid retrieval records dense, keyword, fusion, and Cross-encoder stages
  separately for each retrieval round.
- Agentic orchestration records the DeepSeek decision, final union rerank, and
  answer generation, including safe degradation failures.
- Timings are request-local measurements and are not yet long-term performance
  metrics.
- The UI gains a factual execution view but no aggregate success score; this is
  observability infrastructure, not a complete Agent evaluation system.
- Trace detail fields must remain bounded and must not become a second copy of
  full prompts or documents.

**Review or migration trigger**

Persist or export traces when users need cross-request evaluation, latency
percentiles, token-cost reporting, or multi-user debugging. Consider Langfuse
or LangSmith only after those needs exceed the in-memory event contract.

---

## ADR-015: Keep six conversational turns and an active evidence set

- **Status:** Accepted
- **Date:** 2026-07-12

**Decision**

Keep at most six completed user-assistant turns and the previous answer's final
Top-5 papers in the current Streamlit session. Before processing a follow-up,
ask `deepseek-chat` for one structured evidence action:
`answer_from_existing`, `retrieve_missing`, or `fresh_retrieval`. Reusable
arXiv IDs must belong to the active evidence set. A partial result retrieves
only the missing aspect, merges it with reusable evidence, and reranks the
union for the resolved current question. A new topic discards old evidence.
The controller cannot directly refuse, and each user turn can perform at most
two new retrievals.

**Context**

Rewriting every follow-up and always searching again wastes retrieval work and
ignores evidence already obtained in the conversation. Conversely, sending
only chat text to the answer model does not resolve whether previous papers
fully, partially, or no longer support the user's intent. The project now has
bounded retrieval actions and structured execution traces, so it can maintain
a small evidence state without adding an orchestration framework.

**Alternatives considered**

- Always contextualize and run a fresh retrieval: simple, but repeats searches
  even when the previous papers directly answer the follow-up.
- Give the answer model the last six messages without an evidence decision:
  preserves conversational tone, but may treat earlier assistant text as fact
  or miss evidence gaps.
- Retain every dense, BM25, and fusion candidate from every turn: maximizes
  recall, but rapidly expands context and introduces unrelated evidence.
- Introduce LangGraph for the conversation loop now: represents state and
  actions explicitly, but plain bounded Python control flow remains inspectable.

**Reason**

The previous final papers are the smallest useful factual memory. A structured
action makes evidence reuse explicit, validates paper identity, targets only
missing information, and preserves the existing refusal boundary after a
bounded search. This is a controlled ReAct-like loop without requesting or
storing private chain-of-thought.

**Consequences**

- A follow-up makes one additional small DeepSeek decision request.
- Sufficient previous evidence can answer with zero new retrievals.
- Partial evidence triggers targeted retrieval and combines old and new papers.
- New topics replace the active evidence set after the answer completes.
- Conversation history is used only for intent; the current paper set remains
  the sole factual source in the answer prompt.
- Successful answers and grounded refusals enter history; provider failures do
  not. State is session-local and disappears when a new conversation starts.
- Every action and its evidence IDs are visible in the per-turn execution trace.

**Review or migration trigger**

Revisit the six-turn and Top-5 bounds when real sessions lose needed context or
become too expensive. Add summarization or persistent user memory only when
sessions require it. Consider LangGraph when more tools, parallel actions, or
conversation branches make the current bounded loop difficult to inspect.

---

## ADR-016: Evaluate Agent control actions without an additional Judge model

- **Status:** Accepted
- **Date:** 2026-07-12

**Decision**

Define a small tracked JSON dataset of conversational scenarios with
human-specified allowed actions, retrieval-count bounds, and evidence-reuse
policies. Run those scenarios through the same production `run_rag` function,
update six-turn history and active evidence exactly as Streamlit does, and use
deterministic Python comparisons against `RagResult` and its trace. Do not call
a second LLM to judge action correctness. Write generated reports under
`eval/results/agent_*.json` and keep them untracked.

**Context**

The Agent now chooses among reusing evidence, retrieving missing information,
and starting a fresh search. These control actions and their retrieval counts
are explicit machine-readable fields, so an LLM judge would add cost and
uncertainty without improving the first behavioral baseline. Large-scale paper
relevance annotation was intentionally deferred, but a few expected control
actions are inexpensive for a human to specify.

**Alternatives considered**

- Manually re-enter every conversation in Streamlit: verifies the UI, but is
  slow, inconsistent, and cannot produce repeatable metrics.
- Use DeepSeek as an evaluation judge: supports semantic answer scoring, but
  adds cost, model bias, and nondeterministic grading before it is necessary.
- Compare final answers to exact reference text: deterministic, but rejects
  valid paraphrases and does not isolate Agent control failures.
- Compare structured actions and trace fields in Python: narrow in scope, but
  transparent, repeatable, and directly connected to the current state machine.

**Reason**

The first evaluation question is whether the Agent chose the intended action,
respected the two-retrieval limit, and reused evidence only when permitted.
Those properties need no semantic judge. Calling the real Agent preserves all
DeepSeek, retrieval, reranking, and conversation behavior while keeping the
grading rule inspectable.

**Consequences**

- Scenario definitions are small reviewed fixtures; reports are generated data.
- A baseline run incurs the normal DeepSeek calls made by the real Agent but no
  additional evaluation-model calls.
- The first three real scenarios passed with 100% action accuracy, zero
  unnecessary retrievals, and no retrieval-budget violations.
- A passing action does not prove that the final prose answer is factually
  complete; semantic answer evaluation remains a separate later layer.
- Prompt changes should be made only after a scenario exposes a concrete
  failure, then validated by rerunning all scenarios.

**Review or migration trigger**

Add a carefully reviewed LLM-as-a-Judge layer when the project begins scoring
answer completeness, groundedness, or refusal quality that cannot be checked
with deterministic fields. Keep deterministic action checks as guardrails even
after semantic judging is introduced.
