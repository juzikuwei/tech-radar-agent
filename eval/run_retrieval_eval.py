"""Evaluate the current ChromaDB retriever against human relevance labels."""

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from rag.embedder import DEFAULT_MODEL_NAME, E5Embedder
from rag.search import SearchResult, search_collection
from rag.vector_store import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION_NAME,
    get_persistent_collection,
)


DEFAULT_LABELS_PATH = Path("eval/retrieval_questions.json")
DEFAULT_OUTPUT_PATH = Path("eval/results/retrieval_baseline.json")


def calculate_question_metrics(
    retrieved_ids: Sequence[str],
    *,
    relevant_ids: set[str],
    partially_relevant_ids: set[str],
    answerable: bool,
) -> dict[str, float | None]:
    """Calculate strict and relaxed ranking metrics for one labeled query."""
    if not answerable:
        return {
            "strict_precision": None,
            "relaxed_precision": None,
            "strict_recall": None,
            "relaxed_recall": None,
            "mrr": None,
        }
    if not relevant_ids:
        raise ValueError("answerable questions need at least one relevant paper")

    result_count = len(retrieved_ids)
    relaxed_ids = relevant_ids | partially_relevant_ids
    strict_hits = sum(paper_id in relevant_ids for paper_id in retrieved_ids)
    relaxed_hits = sum(paper_id in relaxed_ids for paper_id in retrieved_ids)

    reciprocal_rank = 0.0
    for rank, paper_id in enumerate(retrieved_ids, start=1):
        if paper_id in relevant_ids:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "strict_precision": strict_hits / result_count if result_count else 0.0,
        "relaxed_precision": relaxed_hits / result_count if result_count else 0.0,
        "strict_recall": strict_hits / len(relevant_ids),
        "relaxed_recall": (
            relaxed_hits / len(relaxed_ids) if relaxed_ids else 0.0
        ),
        "mrr": reciprocal_rank,
    }


def aggregate_results(question_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate answerable ranking metrics separately from no-answer scores."""
    answerable_results = [result for result in question_results if result["answerable"]]
    refusal_results = [result for result in question_results if not result["answerable"]]

    metric_names = (
        "strict_precision",
        "relaxed_precision",
        "strict_recall",
        "relaxed_recall",
        "mrr",
    )
    answerable_metrics = {
        f"mean_{metric_name}": fmean(
            float(result["metrics"][metric_name]) for result in answerable_results
        )
        for metric_name in metric_names
    }

    refusal_scores = [
        float(result["top_similarity"])
        for result in refusal_results
        if result["top_similarity"] is not None
    ]
    refusal_summary = {
        "question_count": len(refusal_results),
        "mean_top_similarity": fmean(refusal_scores) if refusal_scores else None,
        "max_top_similarity": max(refusal_scores) if refusal_scores else None,
        "min_top_similarity": min(refusal_scores) if refusal_scores else None,
    }

    return {
        "answerable": {
            "question_count": len(answerable_results),
            **answerable_metrics,
        },
        "unanswerable": refusal_summary,
    }


def label_search_result(
    result: SearchResult,
    *,
    relevant_ids: set[str],
    partially_relevant_ids: set[str],
) -> str:
    """Attach the human relevance class to one retrieved paper."""
    if result.arxiv_id in relevant_ids:
        return "relevant"
    if result.arxiv_id in partially_relevant_ids:
        return "partially_relevant"
    return "irrelevant"


def load_and_validate_dataset(
    labels_path: Path,
    *,
    indexed_ids: set[str],
) -> dict[str, Any]:
    """Load labels and ensure every labeled paper exists in the current index."""
    dataset = json.loads(labels_path.read_text(encoding="utf-8"))
    questions = dataset.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("evaluation dataset must contain a non-empty questions list")

    seen_question_ids: set[str] = set()
    for question in questions:
        question_id = question.get("id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("every question needs a non-empty id")
        if question_id in seen_question_ids:
            raise ValueError(f"duplicate question id: {question_id}")
        seen_question_ids.add(question_id)

        query = question.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"question {question_id} needs a non-empty query")
        if not isinstance(question.get("answerable"), bool):
            raise ValueError(f"question {question_id} needs an answerable boolean")

        relevant_ids = _require_id_list(question, "relevant_arxiv_ids", question_id)
        partial_ids = _require_id_list(
            question, "partially_relevant_arxiv_ids", question_id
        )
        if set(relevant_ids) & set(partial_ids):
            raise ValueError(f"question {question_id} has overlapping relevance labels")
        if question["answerable"] and not relevant_ids:
            raise ValueError(f"answerable question {question_id} has no relevant paper")
        if not question["answerable"] and relevant_ids:
            raise ValueError(f"unanswerable question {question_id} has relevant papers")

        missing_ids = (set(relevant_ids) | set(partial_ids)) - indexed_ids
        if missing_ids:
            raise ValueError(
                f"question {question_id} references unindexed papers: "
                f"{sorted(missing_ids)}"
            )

    return dataset


def _require_id_list(
    question: dict[str, Any],
    field: str,
    question_id: str,
) -> list[str]:
    value = question.get(field)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"question {question_id} needs a valid {field} list")
    return value


def validate_index_model(metadatas: Sequence[dict[str, Any]], model_name: str) -> None:
    """Prevent evaluation with a query model different from indexed vectors."""
    indexed_models = {
        str(metadata.get("embedding_model")) for metadata in metadatas
    }
    if indexed_models != {model_name}:
        raise ValueError(
            f"index models {sorted(indexed_models)} do not match query model {model_name}"
        )


def build_parser() -> argparse.ArgumentParser:
    """Build arguments for reproducible baseline retrieval evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chroma-path", type=Path, default=DEFAULT_CHROMA_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    return parser


def main() -> int:
    """Run every labeled query and publish a machine-readable baseline report."""
    args = build_parser().parse_args()
    if args.top_k <= 0:
        print("Evaluation failed: top-k must be greater than zero")
        return 1

    try:
        collection = get_persistent_collection(args.chroma_path, args.collection)
        indexed = collection.get(include=["metadatas"])
        indexed_ids = set(indexed["ids"])
        metadatas = [metadata or {} for metadata in indexed["metadatas"] or []]
        if not indexed_ids:
            raise ValueError("ChromaDB collection is empty")
        validate_index_model(metadatas, args.model)
        dataset = load_and_validate_dataset(args.labels, indexed_ids=indexed_ids)

        embedder = E5Embedder(args.model)
        question_results: list[dict[str, Any]] = []
        for question in dataset["questions"]:
            relevant_ids = set(question["relevant_arxiv_ids"])
            partial_ids = set(question["partially_relevant_arxiv_ids"])
            search_results = search_collection(
                question["query"],
                top_k=args.top_k,
                collection=collection,
                embedder=embedder,
            )
            retrieved_ids = [result.arxiv_id for result in search_results]
            metrics = calculate_question_metrics(
                retrieved_ids,
                relevant_ids=relevant_ids,
                partially_relevant_ids=partial_ids,
                answerable=question["answerable"],
            )
            question_results.append(
                {
                    "id": question["id"],
                    "query": question["query"],
                    "answerable": question["answerable"],
                    "metrics": metrics,
                    "top_similarity": (
                        search_results[0].similarity if search_results else None
                    ),
                    "results": [
                        {
                            "rank": rank,
                            "arxiv_id": result.arxiv_id,
                            "title": result.title,
                            "similarity": result.similarity,
                            "label": label_search_result(
                                result,
                                relevant_ids=relevant_ids,
                                partially_relevant_ids=partial_ids,
                            ),
                        }
                        for rank, result in enumerate(search_results, start=1)
                    ],
                }
            )

        report = {
            "dataset_version": dataset.get("dataset_version"),
            "corpus": dataset.get("corpus"),
            "embedding_model": args.model,
            "collection_size": collection.count(),
            "top_k": args.top_k,
            "summary": aggregate_results(question_results),
            "questions": question_results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Evaluation failed: {exc}")
        return 1

    answerable = report["summary"]["answerable"]
    unanswerable = report["summary"]["unanswerable"]
    print(f"Questions: {len(question_results)}")
    print(f"Collection size: {report['collection_size']}")
    print(f"Top-K: {args.top_k}")
    print(f"Mean strict Precision@K: {answerable['mean_strict_precision']:.4f}")
    print(f"Mean relaxed Precision@K: {answerable['mean_relaxed_precision']:.4f}")
    print(f"Mean strict Recall@K: {answerable['mean_strict_recall']:.4f}")
    print(f"Mean relaxed Recall@K: {answerable['mean_relaxed_recall']:.4f}")
    print(f"MRR: {answerable['mean_mrr']:.4f}")
    print(
        "Unanswerable max top similarity: "
        f"{unanswerable['max_top_similarity']:.4f}"
    )
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
