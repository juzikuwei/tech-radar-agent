from rag.answer_validation import select_cited_papers, validate_research_answer
from rag.search import SearchResult


def paper(arxiv_id: str) -> SearchResult:
    return SearchResult(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        document="Grounded abstract.",
        entry_url=f"https://arxiv.org/abs/{arxiv_id}",
        primary_category="cs.AI",
        similarity=0.8,
    )


def test_accepts_valid_citations_and_preserves_first_citation_order() -> None:
    evidence = [paper("2501.00001"), paper("2502.00002")]

    validation = validate_research_answer(
        "两篇论文共同支持结论 [2502.00002，2501.00001]。",
        evidence,
    )

    assert validation.is_valid is True
    assert validation.cited_ids == ("2502.00002", "2501.00001")
    assert [item.arxiv_id for item in select_cited_papers(validation, evidence)] == [
        "2502.00002",
        "2501.00001",
    ]


def test_rejects_mixed_valid_and_unknown_arxiv_citations() -> None:
    validation = validate_research_answer(
        "有依据的部分 [2501.00001]，虚构部分 [9999.99999]。",
        [paper("2501.00001")],
    )

    assert validation.is_valid is False
    assert validation.reason == "unknown_citation"
    assert validation.cited_ids == ("2501.00001",)
    assert validation.unknown_citation_ids == ("9999.99999",)


def test_rejects_a_versioned_id_not_present_in_the_evidence_contract() -> None:
    validation = validate_research_answer(
        "结论来自一个未提供的具体版本 [2501.00001v2]。",
        [paper("2501.00001")],
    )

    assert validation.is_valid is False
    assert validation.reason == "unknown_citation"
    assert validation.unknown_citation_ids == ("2501.00001v2",)


def test_rejects_an_unknown_arxiv_id_outside_citation_brackets() -> None:
    validation = validate_research_answer(
        "正文声称另见 arXiv 9999.99999，但只给出合法引用 [2501.00001]。",
        [paper("2501.00001")],
    )

    assert validation.is_valid is False
    assert validation.reason == "unknown_citation"
    assert validation.unknown_citation_ids == ("9999.99999",)


def test_rejects_research_answer_without_a_valid_citation() -> None:
    validation = validate_research_answer(
        "这是一段没有论文引用的研究结论。",
        [paper("2501.00001")],
    )

    assert validation.is_valid is False
    assert validation.reason == "missing_citation"


def test_ignores_non_arxiv_markdown_link_labels() -> None:
    validation = validate_research_answer(
        "参考 [项目文档](https://example.test)，论文证据见 [2501.00001]。",
        [paper("2501.00001")],
    )

    assert validation.is_valid is True
    assert validation.unknown_citation_ids == ()
