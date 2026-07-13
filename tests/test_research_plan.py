import json

import pytest

from rag.research_plan import (
    ResearchDecisionError,
    ResearchSubquestion,
    SYSTEM_PROMPT,
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
                "query": "first concept architecture" if action_type == "search_papers" else None,
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


def test_followup_decision_must_keep_original_plan_identity() -> None:
    previous = (
        ResearchSubquestion("sq1", "原问题一", "pending"),
        ResearchSubquestion("sq2", "原问题二", "pending"),
    )

    with pytest.raises(ResearchDecisionError, match="original ids and questions"):
        parse_research_decision(
            decision_payload(),
            previous_plan=previous,
            remaining_searches=3,
            search_count=1,
            evidence_count=1,
        )


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
