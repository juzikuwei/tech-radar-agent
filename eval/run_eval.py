"""Command-line entry point for local retrieval and model-backed evaluations.

Retrieval evaluation needs only the local SQLite/ChromaDB/indexed models. Agent
answer, and memory suites additionally use the configured answer model and are
therefore explicit opt-in suites.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from config.environment import load_repository_env

# Hugging Face reads endpoint settings during module import.
load_repository_env()

from eval.memory_adapter import build_memory_runner
from eval.runner import (
    run_agent_cases,
    run_answer_cases,
    run_memory_cases,
    run_retrieval_cases,
    write_report,
)
from eval.schemas import (
    load_agent_cases,
    load_answer_cases,
    load_memory_cases,
    load_retrieval_cases,
)
from rag.application import run_rag
from rag.embedder import E5Embedder
from rag.hybrid_search import hybrid_search
from rag.keyword_search import DEFAULT_DATABASE_PATH, ensure_keyword_index
from rag.research_agent import run_research_agent
from rag.reranker import CrossEncoderReranker
from rag.runtime import RagRuntime, load_rag_runtime
from rag.vector_store import get_persistent_collection


CASES_DIR = Path(__file__).parent / "cases"
DEFAULT_OUTPUT = Path("eval/results/latest.json")


def _load_local_retrieval_runtime() -> tuple[Any, E5Embedder, CrossEncoderReranker]:
    """Load local retrieval resources without requiring an LLM API key."""
    ensure_keyword_index(DEFAULT_DATABASE_PATH)
    return (
        get_persistent_collection(),
        E5Embedder(),
        CrossEncoderReranker(),
    )


def _build_turn_runner(
    runtime: RagRuntime,
    mode: Literal["pipeline", "react"],
    *,
    top_k: int,
):
    """Adapt one production mode to the runner's common turn contract."""
    def run_turn(
        question: str,
        history: tuple[Any, ...],
        evidence: tuple[Any, ...],
    ) -> Any:
        if mode == "pipeline":
            return run_rag(
                question,
                top_k=top_k,
                collection=runtime.collection,
                embedder=runtime.embedder,
                settings=runtime.settings,
                reranker=runtime.reranker,
                database_path=runtime.database_path,
                conversation_history=history,
                active_evidence=evidence,
            )
        return run_research_agent(
            question,
            top_k=top_k,
            collection=runtime.collection,
            embedder=runtime.embedder,
            reranker=runtime.reranker,
            settings=runtime.settings,
            database_path=runtime.database_path,
            conversation_history=history,
            active_evidence=evidence,
            web_search_client=None,
        )

    return run_turn


def _corpus_fingerprint(collection: Any) -> str:
    """Hash stable paper IDs and content hashes without hashing full abstracts."""
    payload = collection.get(include=["metadatas"])
    records = zip(payload.get("ids", []), payload.get("metadatas", []) or [])
    digest = hashlib.sha256()
    for paper_id, metadata in sorted(records, key=lambda item: str(item[0])):
        content_hash = (metadata or {}).get("content_hash", "")
        digest.update(f"{paper_id}\0{content_hash}\n".encode("utf-8"))
    return digest.hexdigest()


def _metadata(
    *,
    suite: str,
    mode: str | None,
    collection: Any,
    models: dict[str, str],
) -> dict[str, Any]:
    """Return stable context needed to interpret a generated report."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite": suite,
        "mode": mode,
        "corpus_size": collection.count(),
        "corpus_fingerprint": _corpus_fingerprint(collection),
        "models": models,
        "cases_are_anchor_based": True,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("smoke", "retrieval", "agent", "answer", "memory"),
        default="smoke",
    )
    parser.add_argument("--mode", choices=("pipeline", "react"), default="pipeline")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--memory-token-threshold", type=int, default=500)
    parser.add_argument("--memory-target-tokens", type=int, default=220)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one local suite and write a JSON report."""
    args = build_parser().parse_args(argv)
    if args.top_k <= 0:
        print("Evaluation failed: --top-k must be greater than zero")
        return 1
    if args.case_limit is not None and args.case_limit <= 0:
        print("Evaluation failed: --case-limit must be greater than zero")
        return 1

    try:
        if args.suite in {"smoke", "retrieval"}:
            version, cases = load_retrieval_cases(CASES_DIR / "retrieval.json")
            if args.suite == "smoke":
                cases = cases[:3]
            if args.case_limit is not None:
                cases = cases[: args.case_limit]
            collection, embedder, reranker = _load_local_retrieval_runtime()
            results = run_retrieval_cases(
                cases,
                search=lambda query, top_k: hybrid_search(
                    query,
                    top_k=top_k,
                    collection=collection,
                    embedder=embedder,
                    reranker=reranker,
                    database_path=DEFAULT_DATABASE_PATH,
                ),
                top_k=args.top_k,
            )
            report = {
                "metadata": _metadata(
                    suite=args.suite,
                    mode=None,
                    collection=collection,
                    models={
                        "embedding": embedder.model_name,
                        "reranker": reranker.model_name,
                    },
                ),
                "dataset_version": version,
                **results,
            }
        else:
            runtime = load_rag_runtime()
            runner = _build_turn_runner(runtime, args.mode, top_k=args.top_k)
            if args.suite == "agent":
                version, cases = load_agent_cases(CASES_DIR / "agent.json")
                if args.case_limit is not None:
                    cases = cases[: args.case_limit]
                results = run_agent_cases(cases, run_turn=runner)
            elif args.suite == "answer":
                version, cases = load_answer_cases(CASES_DIR / "answer.json")
                if args.case_limit is not None:
                    cases = cases[: args.case_limit]
                results = run_answer_cases(cases, run_turn=runner)
            else:
                version, cases = load_memory_cases(CASES_DIR / "memory.json")
                if args.case_limit is not None:
                    cases = cases[: args.case_limit]
                results = run_memory_cases(
                    cases,
                    run_memory=build_memory_runner(
                        runtime,
                        mode=args.mode,
                        top_k=args.top_k,
                        token_threshold=args.memory_token_threshold,
                        target_tokens=args.memory_target_tokens,
                    ),
                )
            report = {
                "metadata": _metadata(
                    suite=args.suite,
                    mode=args.mode,
                    collection=runtime.collection,
                    models={
                        "embedding": runtime.embedder.model_name,
                        "reranker": runtime.reranker.model_name,
                        "answer": runtime.settings.model,
                    },
                ),
                "dataset_version": version,
                **results,
            }
        write_report(report, args.output)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Evaluation failed: {error}")
        return 1

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
