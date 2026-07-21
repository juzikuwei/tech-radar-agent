from rag.answer_validation import (
    is_direct_conversation_message,
    select_cited_papers,
    validate_research_answer,
)
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
        "有依据的部分 [2501.00001]，虚构部分 [9912.99999]。",
        [paper("2501.00001")],
    )

    assert validation.is_valid is False
    assert validation.reason == "unknown_citation"
    assert validation.cited_ids == ("2501.00001",)
    assert validation.unknown_citation_ids == ("9912.99999",)


def test_accepts_a_versioned_citation_of_versionless_evidence() -> None:
    validation = validate_research_answer(
        "结论来自 [2501.00001v2]。",
        [paper("2501.00001")],
    )

    assert validation.is_valid is True
    assert validation.cited_ids == ("2501.00001",)


def test_rejects_a_versioned_id_whose_base_is_not_in_the_evidence() -> None:
    validation = validate_research_answer(
        "合法引用 [2501.00001]，但还引用了 [2502.00002v3]。",
        [paper("2501.00001")],
    )

    assert validation.is_valid is False
    assert validation.reason == "unknown_citation"
    assert validation.unknown_citation_ids == ("2502.00002",)


def test_rejects_an_unknown_arxiv_id_outside_citation_brackets() -> None:
    validation = validate_research_answer(
        "正文声称另见 arXiv 9912.99999，但只给出合法引用 [2501.00001]。",
        [paper("2501.00001")],
    )

    assert validation.is_valid is False
    assert validation.reason == "unknown_citation"
    assert validation.unknown_citation_ids == ("9912.99999",)


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


def test_accepts_plain_decimal_numbers_in_answer_text() -> None:
    validation = validate_research_answer(
        "基准得分为 1234.56789，结论见 [2501.00001]。",
        [paper("2501.00001")],
    )

    assert validation.is_valid is True
    assert validation.unknown_citation_ids == ()


def test_ignores_id_like_fragment_embedded_in_a_longer_number() -> None:
    validation = validate_research_answer(
        "样本量为 31201.56789 条，结论见 [2501.00001]。",
        [paper("2501.00001")],
    )

    assert validation.is_valid is True
    assert validation.unknown_citation_ids == ()


def test_ignores_id_like_token_with_an_invalid_month_segment() -> None:
    validation = validate_research_answer(
        "正文提到编号 2513.00001，结论见 [2501.00001]。",
        [paper("2501.00001")],
    )

    assert validation.is_valid is True
    assert validation.unknown_citation_ids == ()


def test_detects_an_unknown_id_followed_by_an_english_period() -> None:
    validation = validate_research_answer(
        "As shown in 2502.99999. 合法引用见 [2501.00001]。",
        [paper("2501.00001")],
    )

    assert validation.is_valid is False
    assert validation.reason == "unknown_citation"
    assert validation.unknown_citation_ids == ("2502.99999",)


def test_ignores_slash_paths_that_are_not_arxiv_archives() -> None:
    validation = validate_research_answer(
        "参见 github.com/1234567 上的实现，结论见 [2501.00001]。",
        [paper("2501.00001")],
    )

    assert validation.is_valid is True
    assert validation.unknown_citation_ids == ()


def test_still_detects_old_style_archive_identifiers() -> None:
    validation = validate_research_answer(
        "另见 cs/0301012，合法引用 [2501.00001]。",
        [paper("2501.00001")],
    )

    assert validation.is_valid is False
    assert validation.unknown_citation_ids == ("cs/0301012",)


def test_exempts_ids_the_user_mentioned_in_the_question() -> None:
    validation = validate_research_answer(
        "本地论文库没有收录 2503.99999，无法给出结论。相关背景见 [2501.00001]。",
        [paper("2501.00001")],
        question="2503.99999 这篇论文讲了什么？",
    )

    assert validation.is_valid is True
    assert validation.unknown_citation_ids == ()


def test_question_exemption_does_not_allow_other_fabricated_ids() -> None:
    validation = validate_research_answer(
        "库里没有 2503.99999，但可以参考 2504.88888 的说法。引用 [2501.00001]。",
        [paper("2501.00001")],
        question="2503.99999 这篇论文讲了什么？",
    )

    assert validation.is_valid is False
    assert validation.unknown_citation_ids == ("2504.88888",)


def test_common_gratitude_variants_count_as_direct_conversation() -> None:
    for message in ("谢谢", "谢谢你", "谢谢啦", "好的谢谢", "明白了，谢谢你！"):
        assert is_direct_conversation_message(message), message


def test_technical_questions_are_not_direct_conversation() -> None:
    for message in (
        "谢谢，那 ReAct 和固定管线的区别是什么？",
        "什么是 RAG？",
        "帮我比较两篇论文",
    ):
        assert not is_direct_conversation_message(message), message
