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

- **Status:** Superseded by ADR-023
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

- **Status:** Superseded by ADR-023
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

---

## ADR-017: Add a thin FastAPI boundary before replacing Streamlit

- **Status:** Accepted
- **Date:** 2026-07-12

**Decision**

Expose the existing `run_rag` application service through FastAPI before
building a separate React frontend. Keep Streamlit available during migration.
Load SQLite, ChromaDB, the embedder, and the reranker once during API process
startup. The first `/chat` contract is synchronous and accepts at most six
completed turns plus at most five active arXiv IDs. The server reloads those
papers from SQLite instead of accepting client-provided evidence text.

**Context**

The RAG and conversational Agent behavior now have repeatable offline tests and
a real terminal evaluation baseline. Streamlit currently owns presentation,
session state, and direct in-process calls to `run_rag`, which prevents another
frontend from using the same backend contract. Replacing the UI and adding an
API in one change would make failures harder to localize.

**Alternatives considered**

- Continue with Streamlit only: fastest for experiments, but keeps UI and
  backend process boundaries coupled.
- Replace Streamlit directly with a Next.js full-stack application: provides a
  polished web stack, but duplicates backend responsibilities already owned by
  Python and changes too many boundaries at once.
- Add FastAPI and immediately introduce server-side session storage: hides
  state from clients, but adds lifecycle, persistence, and multi-worker design
  before they are required.
- Add a thin stateless FastAPI boundary first: creates a testable contract while
  preserving the existing application logic.

**Reason**

FastAPI gives React or any future client one explicit HTTP boundary without
moving retrieval, model, or conversation decisions out of `rag/application.py`.
Reloading active evidence by trusted IDs preserves the rule that only local
papers may become factual context.

**Consequences**

- FastAPI and Streamlit temporarily coexist.
- API tests can inject a lightweight runtime and remain offline.
- One API worker is recommended initially because each worker would otherwise
  load another copy of the local embedding and reranking models.
- Client conversation history is bounded but not persisted across devices.
- The first response is non-streaming, so users wait for the complete answer.
- CORS, authentication, streaming, and durable sessions remain out of scope.

**Review or migration trigger**

Add SSE streaming when complete-response latency harms the React experience.
Introduce durable server-side conversation state when sessions must survive
refreshes, be shared across devices, or cannot be trusted to the client. Retire
Streamlit only after the React client covers chat, citations, traces, reset, and
error handling.

---

## ADR-018: Use React, TypeScript, and Vite for the separate web client

- **Status:** Accepted
- **Date:** 2026-07-12

**Decision**

Build the separate browser client with React 19, TypeScript, and Vite. Keep the
client as a single-page application that calls FastAPI directly. Store visible
conversation turns and the previous response's arXiv IDs in browser memory.
Send only the last six completed turns and at most five evidence IDs on each
request. Render model answers as escaped Markdown and keep Streamlit available
as a temporary internal debugging interface.

**Context**

The FastAPI boundary is working against the real local models and stores. The
next limitation is Streamlit's combined UI and Python runtime, which makes it
difficult to build a responsive chat layout and independently evolve frontend
interaction. The product is currently a local research assistant, not a public
content website requiring SEO or server-rendered pages.

**Alternatives considered**

- Continue extending Streamlit: minimizes new technology, but retains the
  coupled frontend runtime and limited interaction model.
- Use Next.js: provides routing and server rendering, but duplicates server
  responsibilities already owned by FastAPI and adds complexity without a
  current SEO or full-stack hosting requirement.
- Use another Python UI framework: avoids TypeScript, but does not establish a
  conventional independent web-client boundary.
- Use React with Vite: provides a small SPA toolchain and keeps all backend
  behavior in FastAPI and Python.

**Reason**

React supports the required chat, evidence cards, expandable traces, loading
states, and responsive layout without moving domain logic into the browser.
TypeScript makes the FastAPI response contract explicit, while Vite provides a
minimal development and production build path.

**Consequences**

- Node.js and npm become development dependencies for the frontend.
- Browser refresh currently clears conversation state.
- FastAPI allows only the two local Vite origins during development.
- Complete answers remain synchronous; the UI displays a loading state during
  retrieval, judgment, reranking, and generation.
- The frontend API base URL can be changed with `VITE_API_BASE_URL` in an
  ignored `frontend/.env.local` file.
- Streamlit and React temporarily coexist until browser acceptance is complete.

**Review or migration trigger**

Add client routing or reconsider Next.js when the product needs public pages,
authentication flows, or server rendering. Add durable server-side sessions
when conversations must survive refreshes or move across devices. Add SSE when
real user tests show that complete-response latency makes the loading-only
experience unacceptable.

---

## ADR-019: Retire the Streamlit interface after React acceptance

- **Status:** Accepted
- **Date:** 2026-07-12

**Decision**

Delete the old Python UI modules and remove Streamlit from runtime dependencies.
Use React as the only user-facing web client and FastAPI as its backend
boundary. Preserve earlier ADR entries as migration history rather than
rewriting them.

**Context**

The React client now covers question input, multi-turn state, trusted evidence
IDs, Markdown answers, paper links, Agent traces, loading states, errors, and
conversation reset. Component tests, production builds, a real two-turn API
flow, CORS, and a desktop browser screenshot all passed. Keeping a second UI
would create duplicate presentation behavior and two places to debug state.

**Alternatives considered**

- Keep the Python UI as a permanent admin console: provides a fallback, but no
  current admin-only behavior justifies the duplicate dependency and code path.
- Keep the files but stop documenting them: makes rollback easy, but leaves
  unused code that can drift and confuse future changes.
- Remove the old interface now: leaves one frontend contract and one supported
  user experience.

**Reason**

The migration acceptance criteria have been met. The execution trace and RAG
behavior are already exposed through typed API responses, so the old interface
no longer owns a unique capability.

**Consequences**

- React on port 5173 is the supported local UI.
- FastAPI on port 8000 is required for browser use.
- Python installation no longer includes the UI framework dependency.
- UI changes and tests now live under `frontend/`.
- Historical architecture entries still describe the previous migration state.

**Review or migration trigger**

Add a separate administration interface only when ingestion, evaluation, or
operations require capabilities that do not belong in the user-facing React
client. Build that interface against FastAPI instead of coupling it directly to
Python application objects.

---

## ADR-020: Expose compact read-only tools over Streamable HTTP MCP

- **Status:** Accepted
- **Date:** 2026-07-12

**Decision**

Run a separate stateless Streamable HTTP MCP service at `/mcp`. Protect it with
an interim Bearer Token boundary and DNS-rebinding Host validation. Load one
shared lazy RAG runtime per MCP process. Expose only three read-only tools:
compact knowledge-base search, paper lookup by arXiv ID, and knowledge-base
counts. Keep FastAPI and MCP as peer adapters over shared `rag/` services.

**Context**

The website already serves human users through React and FastAPI. The next
horizontal capability is allowing Agents on other computers to use the local
arXiv knowledge base. Returning complete internal search objects would expose
irrelevant scores, traces, paths, and large texts while increasing bandwidth
and making the public contract difficult to change.

**Alternatives considered**

- Use stdio MCP: simplest locally, but cannot directly serve remote clients and
  may load one model runtime per client process.
- Use the legacy HTTP plus SSE transport: maintains compatibility with older
  clients, but new development should use Streamable HTTP.
- Make MCP call FastAPI over HTTP: avoids a shared service refactor, but adds an
  unnecessary network hop and couples two adapters.
- Return full `SearchResult` and database rows: easier initially, but leaks
  internal fields and creates an unstable oversized contract.

**Reason**

Streamable HTTP supports the planned VPS and domain deployment while sharing one
model runtime across clients. Compact tool-specific payloads provide enough
evidence for an external Agent to cite papers without exposing implementation
details. Stateless sessions keep the first remote deployment simple.

**Consequences**

- Search returns at most five papers and truncates each abstract excerpt to 600
  characters.
- Full normalized abstracts are available only through explicit paper lookup.
- Tools cannot modify data, trigger ingestion, or call the answer model.
- `MCP_AUTH_TOKEN` is required and is never committed.
- Static Bearer Tokens are suitable only for development and limited access;
  public onboarding requires OAuth 2.1, per-user revocation, and rate limits.
- The MCP process listens on port 8100 by default and should remain behind an
  HTTPS reverse proxy in production.

**Review or migration trigger**

Replace static tokens when access is offered to untrusted or self-registering
users. Add stateful sessions only when a tool needs resumable server-side work.
Add write tools only after defining authorization, audit logging, quotas, and a
separate approval boundary.

---

## ADR-021: Stream completed Trace events before the final chat result

- **Status:** Superseded by ADR-028
- **Date:** 2026-07-12

**Decision**

Keep the existing synchronous `POST /chat` contract and add
`POST /chat/stream` using newline-delimited JSON. Send one immediate
`run_started` event, one `trace` event whenever the shared `TraceRecorder`
records a completed or failed stage, and one terminal `result` or `error`
event. Stream execution visibility only; return the final model answer as one
complete value rather than token streaming.

**Context**

Real requests can include dense retrieval, keyword retrieval, rank fusion,
Cross-encoder reranking, DeepSeek retrieval judgment, a second retrieval, and
answer generation. The React client previously displayed only a generic
loading state until all stages finished, even though structured Trace events
were already recorded during execution. Users need to see factual progress
without exposing private chain-of-thought or changing the answer-generation
contract.

**Alternatives considered**

- Keep a loading animation until `/chat` finishes: simplest, but hides the
  Agent's actual work and makes normal latency look like a stalled request.
- Stream answer tokens as well as Trace events: provides earlier prose, but
  adds answer assembly, Markdown-boundary, cancellation, and partial-citation
  complexity that is not currently required.
- Use WebSockets: supports bidirectional sessions, but one request produces one
  ordered response and does not need a persistent duplex connection.
- Use native browser `EventSource`: standardizes SSE reconnect behavior, but it
  does not directly support the required POST JSON request body.
- Use NDJSON over `fetch`: keeps the POST contract, frames each typed event as
  complete JSON, and is small enough to parse without another dependency.

**Reason**

NDJSON provides the smallest explicit protocol for incremental Trace delivery.
An optional callback on the existing recorder guarantees that streamed events
and the final stored trace come from the same source. The final response still
contains the complete trace, so synchronous clients, MCP, tests, and error
inspection remain compatible.

**Consequences**

- The browser displays completed stages incrementally and shows a pending next
  step until the final result arrives.
- Completed Trace details remain available in a collapsed summary above the
  answer; paper evidence is collapsed by default below it.
- The streaming endpoint runs the synchronous RAG pipeline in a producer
  thread and passes bounded events through a queue.
- Resetting the browser conversation aborts client consumption, but work
  already running in a provider or local model may finish server-side.
- Reverse proxies must not buffer the stream; the endpoint sends
  `X-Accel-Buffering: no` and `Cache-Control: no-transform`.
- Stream errors are terminal protocol events because headers may already have
  been sent with HTTP 200.

**Review or migration trigger**

Add answer-token streaming only when real usage shows that final generation
latency still harms the experience. Reconsider SSE or WebSockets when automatic
reconnection, server-initiated updates, resumable jobs, or bidirectional tool
approval becomes necessary.

---

## ADR-022: Add a bounded research ReAct mode beside the reliable pipeline

- **Status:** Accepted
- **Date:** 2026-07-12

**Decision**

Add an optional `react` chat mode while retaining the existing `pipeline`
mode. In ReAct mode, DeepSeek returns a validated research plan containing one
to four evidence requirements and selects one action per round:
`search_papers`, `finish`, or `refuse`. The search action calls the existing
in-process hybrid retrieval stack and returns bounded observations to the next
model decision. Allow at most four searches, generate the final answer with the
existing grounded answer prompt, and fall back to the pipeline if planning or
tool execution fails.

**Context**

The fixed pipeline embeds one query and can rewrite it once, but it does not
explicitly decompose comparison or multi-hop questions. A question can require
separate evidence about two concepts even when no single abstract directly
compares them. The project already has modular retrieval, structured JSON model
decisions, grounded generation, real-time Trace delivery, and a tested pipeline
baseline.

**Alternatives considered**

- Add a one-time deterministic query decomposition workflow: improves recall,
  but the model cannot react to the evidence returned by each subquery.
- Force three or four searches for every question: creates visible activity,
  but wastes latency and tokens on simple questions and encourages duplicate
  evidence.
- Replace the pipeline with ReAct: simplifies the public choice, but removes the
  reliable baseline needed for fallback and quantitative comparison.
- Call the external MCP server from the internal Agent: reuses a protocol, but
  adds transport, authentication, and process overhead inside one application.
- Add LangGraph immediately: provides loop and state abstractions, but plain
  bounded Python remains inspectable at the current number of actions.

**Reason**

The model should decide what evidence is missing, while deterministic code
should retain the action set, parameter bounds, search implementation, maximum
steps, JSON validation, and fallback behavior. Reusing the hybrid search
function keeps model autonomy at the semantic level without exposing low-level
retrieval weights or duplicating MCP infrastructure.

**Consequences**

- The React client defaults to research Agent mode and lets users switch back
  to the reliable pipeline.
- The first decision produces both a visible research plan and the first
  action; later decisions keep stable subquestion IDs and update coverage.
- Search queries and `top_k` are model-selected, but `top_k` is restricted to
  one through five.
- Trace events distinguish planning, tool calls, observations, finish,
  refusal, and pipeline fallback while preserving lower-level retrieval spans.
- The final evidence selector keeps representation from separately searched
  subquestions before filling remaining slots by global rerank score.
- ReAct can make more model calls and take longer than the pipeline. Token usage
  comparison remains required before claiming a quality-efficiency win.
- The first internal Agent exposes only paper search. Batch paper reading and
  other tools should be added only when real Trace inspection demonstrates a
  concrete need.

**Review or migration trigger**

Revisit the action set after real multi-hop runs expose failures that search
alone cannot solve. Add cumulative token accounting to make individual Trace
runs easier to understand. Introduce LangGraph only when additional tools,
resumable actions, parallel branches, or human approval make the plain loop
difficult to reason about.

---

## ADR-023: Pause the standalone evaluation subsystem

- **Status:** Accepted
- **Date:** 2026-07-13

**Decision**

Remove the standalone `eval/` scripts, human-labeled retrieval datasets,
generated baseline reports, and their dedicated unit tests. Keep production
unit tests and use the existing per-request Trace plus a small number of real
questions for current manual inspection. Do not add an LLM Judge, RAGAS, or
DeepEval at the current project scale.

**Context**

The evaluation assets were created when the project had a much smaller corpus
and an earlier fixed pipeline. Maintaining paper relevance labels now requires
manual retrieval review whenever the corpus or retrieval stack changes. The
current project is a personal learning system, and its React Trace already
shows retrieval rounds, model decisions, tool calls, observations, fallbacks,
durations, and selected evidence for direct inspection.

**Alternatives considered**

- Refresh and expand the labeled datasets: preserves comparable metrics, but
  creates ongoing annotation work disproportionate to the current project.
- Add an LLM Judge: reduces some manual review, but introduces extra cost,
  nondeterminism, and another model whose decisions need inspection.
- Keep the unused evaluation files: preserves optional tooling, but leaves a
  stale subsystem that appears authoritative while no longer matching the
  production Agent.
- Remove the subsystem and rely on Trace for now: reduces maintenance while
  keeping the actual execution path observable.

**Reason**

At the current scale, understanding individual Agent runs provides more
learning value than maintaining a small benchmark whose labels quickly become
stale. Trace inspection is already part of the product and requires no second
grading pipeline.

**Consequences**

- The repository no longer reports Precision, Recall, MRR, action accuracy, or
  pipeline-versus-ReAct benchmark numbers.
- Trace inspection can explain behavior but does not prove broad correctness or
  prevent regressions across many questions.
- Production behavior remains covered by offline pytest tests with mocked model
  and retrieval boundaries.
- Token usage and estimated cost may be added later when request cost or
  latency becomes a concrete concern.

**Review or migration trigger**

Reintroduce a small automated evaluation suite when the project gains multiple
users or data sources, prompt and model changes become frequent, regressions
are difficult to notice from individual traces, or a deployment decision needs
repeatable quality comparisons.

---

## ADR-024: Defer full answer reflection and prioritize a weekly radar output

- **Status:** Accepted; weekly-report prioritization superseded by ADR-025
- **Date:** 2026-07-13

**Decision**

Do not add a separate LLM reflection call after every generated answer at the
current project scale. Treat the existing ReAct observation loop as sufficient
process reflection for now. Also defer LangGraph while the handwritten loop has
one tool, bounded searches, and a clear pipeline fallback. Prioritize a
single-Agent technical radar weekly report as the next product capability, and
introduce Supervisor-Worker collaboration only after the single-Agent version
exposes a concrete limitation.

**Context**

The research Agent already decomposes questions, observes each retrieval,
updates evidence coverage, and chooses to search again, finish, or refuse. A
post-generation reviewer would add another model call to every answer without
current evidence of recurring citation or overclaim failures. LangGraph would
replace readable bounded Python without yet providing interruption, parallel
branches, resumable work, or human approval. The project is named a technical
radar assistant, but it does not yet produce a reusable radar artifact.

**Alternatives considered**

- Add full answer reflection now: teaches the pattern, but increases latency
  and cost before a real output-quality problem has been observed.
- Refactor to LangGraph next: introduces a mainstream framework, but the
  current orchestration is not difficult to maintain.
- Build multi-Agent weekly reporting immediately: demonstrates collaboration,
  but provides no single-Agent baseline showing that multiple Agents are
  necessary.
- Build a single-Agent weekly report first: creates a tangible product output
  while reusing the current retrieval and grounded generation boundaries.

**Reason**

New abstractions and model calls should solve observed limitations. A weekly
report advances the product directly and can later provide the concrete task
that justifies reflection, LangGraph, or multi-Agent coordination.

**Consequences**

- Final generated prose is not reviewed by a second LLM call.
- Reflection remains an optional learning extension or a response to repeated
  citation and overclaim failures.
- The first weekly-report implementation must remain single-Agent and must not
  duplicate retrieval logic.
- A multi-Agent version requires evidence that topic decomposition, context
  size, or aggregation quality is limiting the single-Agent version.

**Review or migration trigger**

Add answer reflection after repeated unsupported claims or invalid citations.
Introduce LangGraph when the workflow needs interruption, resumability,
parallel branches, or approval. Introduce Supervisor-Worker reporting after a
working single-Agent report exposes a measurable coordination problem.

---

## ADR-025: Add a query-shaping web_search tool to the research agent

- **Status:** Accepted
- **Date:** 2026-07-13

**Decision**

Give the ReAct research agent one additional in-process tool, `web_search`,
backed by the Tavily REST API behind a small `WebSearchClient` protocol. The
tool exists only to turn vague, very new, or product-flavored terminology into
precise English queries for the local paper search. Web results are untrusted:
they are truncated to at most five title-and-snippet pairs, never enter the
answer prompt, never become citable evidence, and cannot mark a subquestion as
covered. Allow at most two web searches per request inside the existing
four-action budget. A failed web search becomes a tool observation that the
model sees, not an abort. The tool is disabled automatically when
`TAVILY_API_KEY` is absent from the environment. This decision also supersedes
the weekly-report prioritization in ADR-024: the project refocuses on
practicing agent capabilities, and the radar report is deferred.

Live verification also hardened the decision contract deterministically: model
rewrites of subquestion text are normalized back to the original text (the id
is the stable goal), searching a subquestion marked covered downgrades it to
pending instead of failing the run, and every decision request now receives an
explicit `allowed_actions` list computed from the remaining budgets so
exhausted tools disappear from the model's menu.

**Context**

The corpus is English arXiv abstracts while real questions contain product
terms such as `skill` and very new acronyms that embedding retrieval cannot
match and the model's training data may predate. The ReAct loop previously had
one tool, so the model never practiced choosing between tools. The user
dropped the weekly-report product goal in favor of agent-skill practice. Three
consecutive live runs each exposed a different contract violation (plan text
rewrites, covered-target searches, budget-exhausted tool insistence), showing
that negative prompt rules alone do not control the model reliably.

**Alternatives considered**

- Wrap the three MCP read-only tools as in-process tools: satisfies the
  original stage-9 checklist, but adds no genuinely new capability, so tool
  selection stays trivial.
- A fixed chain (web search, keyword extraction, LLM validation, retrieval):
  the user's first sketch, but it rebuilds a pipeline inside the agent loop;
  the ReAct decision step already is the judgment.
- A PDF full-text reading tool: deepens evidence rather than recall, deferred
  per ADR-022 until traces show abstracts are the binding constraint.
- No new tool: the model already knows pre-cutoff terminology, but fails
  exactly on the fast-moving vocabulary this project tracks.

**Reason**

A heterogeneous external tool with real latency, failure modes, and untrusted
output is the smallest change that makes tool selection, tool-failure
handling, and injection surfaces real. Restricting it to query shaping keeps
the grounding contract intact: facts still come only from local abstracts.

**Consequences**

- The runtime gains an optional external dependency, one environment variable,
  and up to two extra network calls per ReAct request.
- Untrusted web snippets now flow into decision prompts, creating a concrete
  indirect-prompt-injection surface that stage 14 guardrails must address.
- Contract violations that are safe to correct are now normalized in code;
  only genuinely ambiguous violations still fail to the pipeline fallback.
- The live acceptance run completed the intended loop: the model chose
  web_search for `mcp 和 skill 是什么？`, refined the query to `Model Context
  Protocol MCP agent skill`, retrieved five local papers, and answered citing
  only those papers.
- Tavily's free tier (1,000 calls/month) bounds cost; the protocol keeps the
  backend swappable.

**Review or migration trigger**

Revisit if web snippets ever need to become citable evidence (requires new
guardrails and provenance), if Tavily reachability or quota degrades (swap the
`WebSearchClient` backend), or when stage 14 adds input filtering that should
also wrap web observations.

---

## ADR-026: Persist bounded conversations on the server

- **Status:** Accepted
- **Date:** 2026-07-13

**Decision**

Persist browser conversations in the existing SQLite database with two new
tables: `conversations` for UUID, title, and timestamps, and
`conversation_turns` for complete user-assistant exchanges plus the final
paper IDs. Keep at most 100 complete turns per conversation, but continue to
inject only the six most recent turns into conversation decisions and answer
generation. Derive active evidence from the newest stored turn and reload the
corresponding papers from SQLite.

Replace the client-state `POST /chat` and `POST /chat/stream` contracts with
conversation-scoped endpoints under `/conversations/{conversation_id}`. Add
create, list, full-history, and delete endpoints. The client now sends only the
question, `top_k`, and mode. Do not keep the old contract in parallel. Name a
conversation from its first stored question, truncated to 50 characters. At
100 turns, reject the next request and ask the user to create a new
conversation.

Store successful answers and grounded refusals as complete turns. Do not store
provider or generation failures. In the streaming path, finish persistence in
the producer thread before emitting the terminal result, even if the browser
has stopped consuming the response. Persist text and paper IDs, not old Trace
events or ranking scores; history reloads current paper metadata from SQLite.

**Context**

ADR-017 and ADR-018 deliberately kept the first FastAPI and React boundary
stateless, with recent history and active evidence owned by browser memory.
That made the first migration small, but a refresh erased the conversation,
only one conversation could exist, and increasing stored history would have
made every request upload more client-controlled state. Both ADRs named durable
sessions as their migration trigger, which is now reached.

The 100-turn product requirement contains two separate limits. Durable storage
must retain history for navigation, while the model window must stay bounded
for cost and relevance. Sending all 100 turns to both the evidence controller
and answer model would increase token cost and dilute the current question.

**Alternatives considered**

- Store conversations in browser local storage: survives refreshes on one
  browser, but still trusts client history and cannot support later server-side
  summarization or long-term memory cleanly.
- Keep the old `/chat` contract beside the new one: avoids an immediate frontend
  migration, but preserves two trust models and two service assembly paths with
  no external consumer requiring compatibility.
- Store separate user and assistant message rows: supports editing and
  branching later, but complicates the current invariant that only complete,
  accepted turns enter history.
- Copy active evidence IDs onto the conversation row: makes reads direct, but
  duplicates data already determined by the newest complete turn.
- Inject all stored turns into the model: appears to provide long memory, but
  creates unbounded cost and attention noise rather than a controlled memory
  design.

**Reason**

Complete-turn persistence matches the existing `ConversationTurn` domain
model and ADR-015 failure semantics. A server-owned UUID contract removes the
client-history trust boundary, gives React durable multi-session behavior, and
provides the minimum source data needed for the next rolling-summary stage.
Separating the 100-turn storage cap from the six-turn model window makes both
limits explicit and independently testable.

**Consequences**

- Browser refresh, conversation switching, and service restart preserve up to
  100 turns per conversation.
- Chat requests perform SQLite reads before RAG execution and one transaction
  after a completed answer.
- The frontend can replay text and citations, but old request-local Trace and
  ranking scores are intentionally unavailable after reload.
- A seventh or later turn still gives the model only the six most recent
  complete turns and the newest turn's first five evidence papers.
- The old `/chat` routes return 404, and requests that try to upload history or
  active evidence to the new contract fail validation.
- This is local single-user persistence; authentication, ownership, editing,
  branching, and cross-device identity remain out of scope.
- A turn deleted concurrently with a still-running streamed request may cause
  that request's final persistence to fail because its conversation no longer
  exists.

**Review or migration trigger**

Add a rolling summary when important facts older than six turns must remain
visible inside one conversation. Revisit the 100-turn policy if real use needs
archival export, automatic rotation, or a larger limit. Introduce user and
ownership columns before authentication or remote multi-user deployment.
Stage 13 may read these completed turns to extract cross-conversation memory,
but must use a separate schema and recall policy rather than widening the chat
window.

---

## ADR-027: Make web-search failure recovery deterministic

- **Status:** Superseded by ADR-028
- **Date:** 2026-07-14

**Decision**

Classify Tavily failures at the web-search boundary with an error type, optional
HTTP status, and retryability flag. Authentication and configuration failures
are non-retryable and disable `web_search` for the remainder of the current
request. Timeouts, transport failures, HTTP 429, and HTTP 5xx failures may keep
the tool available for one remaining ReAct retry, bounded by the existing
two-web-search and four-action limits.

Every failed web call becomes a structured observation containing the error,
error type, retryability, and whether the tool remains available. After any
web failure, require at least one subsequent `search_papers` action before the
agent may respond, finish, or refuse. If the web tool is disabled, remove it
from `allowed_actions`. A research subquestion cannot remain `covered` when no
paper evidence exists; normalize that bookkeeping back to `pending`. Keep the
HTTP and frontend response contracts unchanged.

**Context**

A live run with an intentionally invalid Tavily key returned HTTP 401. The
exception was safely converted into an observation, but the agent then repeated
the identical web call because tool availability was still derived only from
whether a client object existed. After the second failure exhausted the web
budget, the model reclassified the technical question as conversation, marked
the unanswered subquestion covered, and asked the user for context without
trying the local paper tool. The request stayed healthy, but recovery did not
complete the research task.

**Alternatives considered**

- Rely only on stronger prompt wording: small, but prior live runs already show
  that negative instructions do not reliably control action selection or plan
  bookkeeping.
- Fall back to the fixed pipeline on every web failure: reliable, but discards
  the remaining ReAct tools and hides a recoverable tool-level failure as an
  agent-level failure.
- Retry every web error twice: simple, but authentication, permission, invalid
  request, and malformed-response failures cannot recover by repetition.
- Answer from model memory after Tavily failure: responsive, but violates the
  project contract that technical claims must be grounded in local papers.

**Reason**

Failure classification belongs at the network boundary, while allowed actions
belong at the orchestration boundary. Making both explicit preserves the small
hand-written ReAct loop, prevents known-dead tools from consuming action budget,
and guarantees a best-effort grounded path before clarification or refusal.

**Consequences**

- An invalid or unauthorized Tavily key produces one failed web Trace event,
  followed by a local paper search rather than another doomed web call.
- Transient Tavily failures may still use the second web-search allowance, but
  terminal actions remain unavailable until a later local search completes.
- Web observations expose bounded operational metadata but never include API
  keys, response bodies, or citable web evidence.
- A request with too little remaining action budget to perform the required
  local search may still fail to the existing reliable-pipeline boundary.

**Review or migration trigger**

Revisit the retry classification when a replacement web provider exposes
provider-specific quota or maintenance signals. Move retries into a shared
tool-execution policy only after another external tool needs the same behavior.
Reconsider the mandatory local-search rule if a future trusted documentation
tool can itself provide citable evidence.

---

## ADR-028: Replace the research classifier with a tool-calling harness

- **Status:** Accepted
- **Date:** 2026-07-14

**Decision**

Replace the ReAct `question_type` and structured `respond/search/finish`
decision contract with one OpenAI-compatible tool-calling harness. For every
model turn, expose the currently available function tools and let the model
either return function calls or a user-visible assistant message. The two
tools are `search_papers(query, top_k)`, which encapsulates the complete local
hybrid retrieval and reranking stack, and optional
`web_search(query, max_results)`, which remains non-citable query-shaping data.
Execute at most one function call from each assistant turn even when a provider
ignores `parallel_tool_calls=false`; return rejected extra calls as tool error
messages without executing them or consuming the tool budget.
Validate every primary function call against the exact tool menu sent in that
turn. A request for a removed tool becomes a `tool_unavailable` observation,
is never executed, and still consumes one of the five call allowances so an
uncooperative model cannot loop forever.

Allow at most five tool executions per request. After the fifth execution,
perform one additional model call with no tools so the request can end with an
answer, clarification, or grounded refusal. Tool failures always become tool
messages containing bounded error metadata. Remove non-retryable tools from
the next model turn; do not deterministically force a replacement tool. Keep
the reliable fixed pipeline as the boundary fallback for model or harness
failures.

Upgrade the POST streaming endpoint from NDJSON completed-stage delivery to
Server-Sent Events. Stream run lifecycle, model start/retry/completion, tool
start/completion/failure, temporary status text, assistant text deltas, the
complete assistant message, provider-reported token usage, and the final
persisted result. Record per-call usage on model events and aggregate usage on
the final response. Public ReAct Trace stops at the tool boundary; dense
retrieval, BM25, rank fusion, and reranking remain internal diagnostics.

This decision supersedes ADR-021 and the ReAct control-flow portions of
ADR-025 and ADR-027. The Tavily trust boundary and failure classification from
those decisions remain in force.

**Context**

Live traces showed that pre-classifying a message as conversation, ambiguous,
or research created a second state machine beside the ReAct loop. A failed web
search could cause the model to reclassify a technical question as
conversation, while a deterministic recovery rule could over-correct and
force retrieval after a mistaken tool choice. The user also saw internal BM25
and dense-retrieval stages even though the Agent conceptually called one paper
search tool.

Completed-stage NDJSON made retrieval observable but still held the entire
answer until generation ended and did not expose token usage. The product now
needs the model/tool message boundary itself to be observable, including
partial assistant text and operational failures.

**Alternatives considered**

- Keep the classifier and improve its prompt: preserves existing tests, but
  retains duplicate control flow and cannot eliminate classifier/tool-choice
  disagreement.
- Treat `should_search` as a third tool: makes routing explicit, but adds a
  tool that changes no external state and merely recreates the classifier
  inside the harness.
- Force local search after every web failure: grounded, but assumes the web
  call was appropriate and prevents the model from asking for clarification.
- Expose every retrieval substage as a public tool: maximizes detail, but the
  model cannot usefully control those implementation steps and the UI leaks
  storage-specific complexity.
- Keep NDJSON and add more event types: technically workable, but SSE provides
  standard event framing and clearer lifecycle semantics for a long-lived
  server-to-browser stream.

**Reason**

Tool calling gives one control surface: the assistant message itself. A normal
chat ends on the first model response, while a research question naturally
continues through tool messages without a separate router. Encapsulating
retrieval gives the tool one stable responsibility and permits the underlying
search implementation to evolve without changing the Agent or UI contract.
Usage and text deltas belong at the model boundary, so the same harness is the
smallest place to make latency, retries, token cost, and intermediate tool
state visible.

**Consequences**

- The deleted `research_plan.py` JSON contract and its question-type tests are
  replaced by streamed function-call assembly and harness tests.
- Simple conversation messages can finish in one model call without any tool.
- Research quality now depends more directly on model tool choice; grounding
  instructions and citation-ID validation remain necessary acceptance checks.
- A request can make up to six model calls: one after each of five tools plus a
  final no-tool call when the budget is exhausted.
- ReAct users see concise model/tool events instead of storage-specific search
  stages. Pipeline mode may still expose its fixed internal stages.
- Token usage is request-scoped and is not persisted with old conversation
  turns, matching the existing policy that historical Trace is not replayed.
- POST streaming clients must parse SSE frames and handle incremental text;
  the synchronous chat response remains available.

**Review or migration trigger**

Add deterministic policy checks only after repeated live traces show a
specific unsafe tool-choice pattern. Introduce parallel tool calls when the
tool set contains independent operations that benefit from concurrency.
Persist usage and Trace when cross-request cost analysis becomes a product
requirement. Revisit the five-call budget using observed latency and token
distributions rather than increasing it preemptively.

---

## ADR-029: Replace fixed conversation windows with token-triggered batch compaction

- **Status:** Accepted
- **Date:** 2026-07-15

**Decision**

Remove both the fixed six-turn model window and the 100-turn conversation
storage cap. Persist every original user message, assistant response, and
paper-ID association in `conversation_turns` without rewriting or deleting old
rows. Add a structured rolling summary and a `compacted_through_turn_id` to the
conversation row. The summary is derived working memory and can always be
rebuilt from the immutable turn log.

Estimate the token size of the summary plus every uncompacted complete turn.
When it exceeds `CONVERSATION_CONTEXT_TOKEN_THRESHOLD`, batch the oldest
complete turns until the remaining raw context is below
`CONVERSATION_CONTEXT_TARGET_TOKENS`, merge that batch with the previous
summary through one deterministic-temperature model call, validate the exact
JSON schema, and atomically advance the compaction boundary. If an existing
conversation requires several bounded summary calls, build those intermediate
summaries in memory and commit only the final summary and boundary after every
batch succeeds. The fixed pipeline
and ReAct harness both receive the same structured summary followed by all
uncompacted original turns. No fixed number of recent turns is retained.

The summary may contain only user goals, confirmed requirements, decisions,
important conversational context, and open questions. It must not summarize
paper contents or promote prior assistant claims into facts. Original paper
records remain in SQLite and are loaded separately by arXiv ID; only current
paper tool results or active evidence may support technical claims. A failed
summary call or invalid JSON leaves the previous summary, boundary, and all raw
turns unchanged and fails the request explicitly instead of silently truncating
history.

This decision supersedes the six-turn history bound in ADR-015 and the
100-turn storage/six-turn model-window portions of ADR-026. Their active
evidence and persistent-conversation decisions remain in force.

**Context**

The six-turn window made latency predictable but discarded early goals and
constraints as soon as the seventh later turn arrived. Increasing the fixed
window would only postpone the same loss and would tie semantic continuity to
turn count even though turns vary greatly in size. The 100-turn storage cap
also conflicts with the requirement that original user wording remain
available for audit and rebuilding.

The application already stores complete turns and loads trusted paper records
separately, so it has the source data needed for an incremental compaction
boundary. Both the reliable pipeline and the ReAct harness need the same
conversation representation; otherwise the selected mode would change what
the model remembers.

**Alternatives considered**

- Increase the window from six to a larger fixed number: easy, but still loses
  context according to turn count rather than actual prompt size.
- Send the complete raw conversation on every request: lossless until the
  provider context limit is reached, but token cost and Agent-loop headroom grow
  without bound.
- Rewrite a summary after every completed turn: keeps prompts stable, but adds
  latency and cost to every request and repeatedly rewrites memory before it is
  necessary.
- Retrieve old turns from a vector index: useful for future cross-session or
  episodic memory, but semantic retrieval alone does not preserve sequential
  decisions and explicit user constraints.
- Silently fall back to the newest raw turns when compaction fails: available,
  but recreates the truncation behavior this decision removes and can hide lost
  requirements from the user.

**Reason**

A token-triggered boundary responds to the actual resource constraint while
keeping recent raw context dynamic. Batch compaction avoids paying for a model
call on every turn. Separating the immutable event log from derived working
memory protects user wording, permits summary rebuilds and migration, and
keeps paper evidence outside the lossy memory channel.

**Consequences**

- Conversations can store more than 100 complete turns; API reads may return a
  large history and may need pagination if real sessions grow substantially.
- A request that crosses the threshold makes one extra model call before the
  normal pipeline or ReAct loop.
- The current token estimator is conservative and provider-independent, not an
  exact tokenizer for the configured model. The default 12,000/8,000 thresholds
  require calibration from real provider usage.
- Successful compaction emits a structured Trace event with the turn count,
  previous estimate, next estimate, and persisted boundary.
- Summary failure blocks that request but never mutates raw messages or silently
  drops context.
- Summary correctness becomes a long-conversation quality boundary. Structured
  fields and immutable raw turns make it testable and recoverable, but repeated
  live compactions still require acceptance testing.

**Review or migration trigger**

Calibrate or replace the estimator when provider-reported prompt usage shows a
consistent error large enough to waste context or risk overflow. Add summary
version history when operators need rollback or comparison across prompts.
Introduce retrieval-based episodic memory only when conversations become too
large for one rolling summary, and add API pagination when full-history reads
become a measurable latency or payload problem.

---

## ADR-030: Enforce deterministic citation validation before displaying research answers

- **Status:** Accepted
- **Date:** 2026-07-15

**Decision**

Treat every model-generated research answer as untrusted text. Validate it at a
shared application boundary used by both the fixed pipeline and the ReAct
harness before the answer is returned or persisted. A research answer must cite
at least one current evidence paper using the exact arXiv ID inside square
brackets. Any arXiv-shaped ID anywhere in the answer that is not present in the
current evidence set invalidates the complete answer. Paper cards, persisted
paper IDs, and the next active evidence set contain only papers actually cited
by a validated answer, in first-citation order.

When retrieval returns no evidence, return a deterministic Chinese
insufficient-evidence response without calling the answer model. When generated
research text has no valid citation or contains an unknown ID, replace it with
a deterministic citation-validation refusal, return no paper cards, and record
an `answer_validation` failure in the request Trace. These safe refusals are
normal completed turns rather than provider failures.

The ReAct harness buffers model text until the final message is complete and
validated. It emits only the validated answer or deterministic refusal to the
SSE client. A conservative allowlist permits greetings, thanks, and short
feedback to remain citation-free conversation responses; other no-evidence
messages fail closed as research refusals.

This decision partially supersedes ADR-009: citation identity and the
zero-evidence refusal path are now programmatically validated. Semantic
evidence sufficiency and claim-level entailment remain probabilistic. This does
not introduce the LLM reflection described in the deferred stage 10 design.

**Context**

The application now renders clickable paper cards, persists citation IDs, and
reuses the newest papers as active conversational evidence. Prompt instructions
alone could not guarantee that inline IDs belonged to retrieved papers. The
previous ReAct selector silently ignored unknown IDs and, when no valid citation
was found, attached ranked candidate papers anyway. The fixed pipeline returned
all candidates regardless of which papers the answer cited. A completely empty
retrieval also returned `answer=None`, which the frontend presented as a
generation failure instead of the product's promised insufficient-evidence
refusal.

Streaming made post-hoc validation especially unsafe because invalid text could
already reach the browser before the final result was checked.

**Alternatives considered**

- Continue relying on prompts: preserves token streaming, but cannot enforce
  citation identity or prevent invalid answers from entering history.
- Silently remove unknown citations: keeps more model prose, but leaves claims
  attached to the wrong or incomplete evidence and hides the model failure.
- Require structured JSON output for every final answer: can express citations
  explicitly, but reintroduces provider-format compatibility and repair logic
  beyond the smallest safety boundary.
- Add one reflection or regeneration model call after every answer: may repair
  more failures, but increases latency and cost before deterministic validation
  has produced real failure data.
- Return every retrieved candidate as a paper card: preserves previous UI
  behavior, but confuses retrieval candidates with citations and pollutes active
  evidence.

**Reason**

Exact-ID validation is deterministic, offline-testable, provider-independent,
and directly protects the API and persistence contracts. Failing closed with a
stable refusal avoids another model call and makes invalid model output
observable without exposing it to users. Sharing one validator prevents the
fixed pipeline and ReAct mode from developing different citation semantics.

**Consequences**

- Zero-result questions produce a normal, persistable refusal rather than a
  frontend generation-error state.
- Unknown, version-invented, bare, or mixed valid/invalid arXiv IDs invalidate
  the complete research answer.
- Research answers without at least one valid bracketed citation are replaced
  with a safe refusal.
- Paper cards, stored paper IDs, and active evidence now mean "validated cited
  papers", not "all final retrieval candidates".
- ReAct final prose is delivered only after validation, then divided into small
  SSE display chunks instead of exposing raw model-token deltas before the
  safety boundary.
- Direct conversational handling is intentionally conservative; uncommon
  feedback wording may fail closed until a real example justifies widening the
  allowlist.
- The validator proves citation identity, not that every sentence is actually
  entailed by the cited abstract. Claim-level groundedness and answer quality
  still require evaluation or a future bounded reflection step.

**Review or migration trigger**

Add one controlled regeneration when real traces show frequent formatting-only
validation failures with otherwise useful evidence. Replace the conservative
direct-conversation allowlist with an explicit validated response-intent
contract when legitimate feedback or rewriting requests are repeatedly
rejected. Introduce claim-level reflection only after real answers demonstrate
unsupported conclusions despite valid citation IDs.

---

## ADR-031: Link only verified citations and collapse Trace lifecycle noise in the UI

- **Status:** Accepted
- **Date:** 2026-07-15

**Decision**

Render an inline arXiv citation as a hyperlink only when its exact ID exists in
the completed response's validated `papers` list and that paper provides an
HTTP(S) `arxiv.org/abs/...` URL whose path resolves to the same versionless ID.
Unknown IDs, unsafe URLs, citations inside code, and citations not present in
the response evidence remain plain text. Completed and restored conversation
turns use this shared renderer; the pending answer has no paper whitelist and
therefore does not create citation links before the final response arrives.

Keep every raw backend `TraceEvent`, but derive a presentation-only step list in
React. Pair `model` started/completed events and `tool` started/completed events
into one product-level step, use stable labels such as "模型选择论文检索",
"检索本地论文" and "引用校验通过", and count these presented steps in the
summary. The original labels, stages, statuses, parameters, usage, outputs and
timings remain available inside a collapsed "查看技术详情" section. The live
Trace shows the same concise steps plus the current status, without expanding
debug payloads by default.

**Context**

Validated answers contain inline IDs such as `[2602.01129]`, while the paper
cards below already link to arXiv. Leaving the inline ID as plain text makes it
hard to move from a claim to its source. Linking every arXiv-shaped string would
recreate a trust problem because model text is untrusted and URLs can be
malformed or hostile.

The ReAct Trace exposes useful lifecycle events, but one model decision and one
tool call appear as separate start and completion rows. A normal one-search
answer therefore displayed six low-level entries instead of three or four
meaningful product steps. Removing raw events would improve readability but
weaken debugging and learning value.

**Alternatives considered**

- Link every arXiv-shaped ID by constructing a URL: simple, but allows unverified
  model text to look trusted and ignores the stored evidence contract.
- Ask the backend to inject Markdown links: centralizes formatting, but couples
  domain responses to one UI representation and requires escaping Markdown.
- Link only paper-card titles: already safe, but leaves claim-level citations
  inconvenient to inspect.
- Delete model/tool start events from the backend Trace: reduces payload size,
  but permanently loses lifecycle timing and retry context.
- Show every raw Trace event by default: maximizes detail, but obscures the
  Agent's actual decisions for normal users.

**Reason**

The completed response's paper list is already the validated citation
whitelist established by ADR-030. Reusing it lets the frontend create useful
links without trusting arbitrary model output. Presentation-only Trace merging
improves readability without changing the HTTP schema, persistence format, or
observability source data.

**Consequences**

- Clicking a verified inline citation opens the corresponding arXiv abstract in
  a new tab.
- Persisted turns regain the same links after reload because paper metadata is
  reloaded from SQLite.
- Pending text remains unlinkified until the final validated result is known.
- Unsafe or mismatched paper URLs never become citation links.
- Displayed Trace step counts may be smaller than raw event counts; technical
  details expose the exact original sequence when needed.
- Trace labels are a frontend presentation contract and can evolve without
  changing backend stage names or stored data.

**Review or migration trigger**

Move citation rendering to an AST plugin if answers begin using tables or other
Markdown containers not covered by the shared renderer. Add same-page citation
anchors when users need to jump to paper cards instead of opening arXiv. Revisit
Trace grouping when parallel tools, nested spans, or persisted cross-request
traces make sequential lifecycle pairing insufficient.
