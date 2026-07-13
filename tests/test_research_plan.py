import json

import pytest

from rag.research_plan import (
    ResearchDecisionError,
    ResearchSubquestion,
    SYSTEM_PROMPT,
    build_research_messages,
    parse_research_decision,
)


def decision_payload(
    *,
    action_type: str = "search_papers",
    statuses: tuple[str, ...] = ("pending", "pending"),
) -> str:
    return json.dumps(
        {
            "question_type": "comparison",
            "reason_summary": "需要分别查找两个概念再比较",
            "subquestions": [
                {"id": f"sq{index}", "question": f"问题 {index}", "status": status}
                for index, status in enumerate(statuses, start=1)
            ],
            "next_action": {
                "type": action_type,
                "target_subquestion_id": "sq1" if action_type == "search_papers" else None,
                "query": (
                    "first concept architecture"
                    if action_type == "search_papers"
                    else "what is agent skill" if action_type == "web_search" else None
                ),
                "top_k": 3 if action_type == "search_papers" else None,
            },
        }
    )


def test_parses_initial_research_plan_and_search_action() -> None:
    decision = parse_research_decision(
        decision_payload(),
        remaining_searches=4,
        search_count=0,
        evidence_count=0,
    )

    assert decision.question_type == "comparison"
    assert [item.id for item in decision.subquestions] == ["sq1", "sq2"]
    assert decision.next_action.type == "search_papers"
    assert decision.next_action.top_k == 3


def test_finish_requires_evidence_and_complete_plan() -> None:
    with pytest.raises(ResearchDecisionError, match="paper evidence"):
        parse_research_decision(
            decision_payload(action_type="finish", statuses=("covered", "covered")),
            remaining_searches=3,
            search_count=1,
            evidence_count=0,
        )

    with pytest.raises(ResearchDecisionError, match="all subquestions covered"):
        parse_research_decision(
            decision_payload(action_type="finish", statuses=("covered", "pending")),
            remaining_searches=3,
            search_count=1,
            evidence_count=2,
        )


def test_followup_decision_must_keep_original_plan_ids() -> None:
    previous = (
        ResearchSubquestion("sqA", "原问题一", "pending"),
        ResearchSubquestion("sqB", "原问题二", "pending"),
    )

    with pytest.raises(ResearchDecisionError, match="original plan ids"):
        parse_research_decision(
            decision_payload(),
            previous_plan=previous,
            remaining_searches=3,
            search_count=1,
            evidence_count=1,
        )


def test_followup_question_rewrites_are_normalized_to_original_text() -> None:
    previous = (
        ResearchSubquestion("sq1", "原问题一", "pending"),
        ResearchSubquestion("sq2", "原问题二", "pending"),
    )

    decision = parse_research_decision(
        decision_payload(),
        previous_plan=previous,
        remaining_searches=3,
        search_count=1,
        evidence_count=1,
    )

    assert [item.question for item in decision.subquestions] == [
        "原问题一",
        "原问题二",
    ]


def test_search_is_rejected_after_budget_is_exhausted() -> None:
    with pytest.raises(ResearchDecisionError, match="budget is exhausted"):
        parse_research_decision(
            decision_payload(),
            remaining_searches=0,
            search_count=4,
            evidence_count=3,
        )


def test_comparison_prompt_allows_synthesis_from_separate_evidence() -> None:
    assert "不要求存在一篇直接比较两个对象的论文" in SYSTEM_PROMPT
    assert "不得为了寻找“直接对比论文”重复检索" in SYSTEM_PROMPT


def test_searching_a_covered_target_downgrades_it_to_pending() -> None:
    payload = json.loads(decision_payload(statuses=("covered", "pending")))

    decision = parse_research_decision(
        json.dumps(payload),
        remaining_searches=3,
        search_count=1,
        evidence_count=1,
    )

    assert decision.next_action.target_subquestion_id == "sq1"
    assert decision.subquestions[0].status == "pending"
    assert decision.subquestions[1].status == "pending"


def test_parses_web_search_action_when_tool_is_available() -> None:
    decision = parse_research_decision(
        decision_payload(action_type="web_search"),
        remaining_searches=3,
        search_count=1,
        evidence_count=1,
        remaining_web_searches=2,
        web_search_available=True,
    )

    assert decision.next_action.type == "web_search"
    assert decision.next_action.query == "what is agent skill"
    assert decision.next_action.target_subquestion_id is None
    assert decision.next_action.top_k is None


def test_web_search_is_rejected_when_unavailable_or_exhausted() -> None:
    with pytest.raises(ResearchDecisionError, match="not available"):
        parse_research_decision(
            decision_payload(action_type="web_search"),
            remaining_searches=3,
            search_count=0,
            evidence_count=0,
            remaining_web_searches=2,
            web_search_available=False,
        )

    with pytest.raises(ResearchDecisionError, match="web search budget"):
        parse_research_decision(
            decision_payload(action_type="web_search"),
            remaining_searches=3,
            search_count=0,
            evidence_count=0,
            remaining_web_searches=0,
            web_search_available=True,
        )


def test_web_search_action_must_not_carry_paper_search_arguments() -> None:
    payload = json.loads(decision_payload(action_type="web_search"))
    payload["next_action"]["top_k"] = 3

    with pytest.raises(ResearchDecisionError, match="web_search must not set"):
        parse_research_decision(
            json.dumps(payload),
            remaining_searches=3,
            search_count=0,
            evidence_count=0,
            remaining_web_searches=2,
            web_search_available=True,
        )


def test_research_messages_expose_web_search_budget() -> None:
    messages = build_research_messages(
        "MCP 和 skill 是什么？",
        history=(),
        evidence=(),
        observations=(),
        previous_plan=(),
        remaining_searches=4,
        search_count=0,
        remaining_web_searches=2,
        web_search_available=True,
    )

    payload = json.loads(messages[1]["content"])
    assert payload["web_search_available"] is True
    assert payload["remaining_web_searches"] == 2
    assert payload["allowed_actions"] == ["search_papers", "web_search"]
    assert "web_search" in messages[0]["content"]


def test_allowed_actions_shrink_with_budgets_and_state() -> None:
    messages = build_research_messages(
        "MCP 和 skill 是什么？",
        history=(),
        evidence=(),
        observations=(),
        previous_plan=(),
        remaining_searches=2,
        search_count=2,
        remaining_web_searches=0,
        web_search_available=True,
    )

    payload = json.loads(messages[1]["content"])
    assert payload["allowed_actions"] == ["search_papers", "refuse"]
