import json
from types import SimpleNamespace

import pytest

from config.model_settings import ModelSettings
from rag.retrieval_judge import (
    RetrievalDecisionError,
    build_judge_messages,
    judge_retrieval,
    parse_retrieval_decision,
)
from rag.search import SearchResult


SETTINGS = ModelSettings("key", "https://api.deepseek.com", "deepseek-chat")


def make_result(document: str = "A" * 1_000) -> SearchResult:
    return SearchResult(
        arxiv_id="2607.00001",
        title="Long-horizon agent memory",
        document=document,
        entry_url="https://arxiv.org/abs/2607.00001",
        primary_category="cs.AI",
        similarity=0.8,
    )


def test_judge_prompt_bounds_the_evidence_payload() -> None:
    messages = build_judge_messages(
        "Agent 如何避免状态遗忘？",
        [make_result()],
        abstract_char_limit=120,
    )
    payload = json.loads(messages[1]["content"])

    assert payload["question"] == "Agent 如何避免状态遗忘？"
    assert len(payload["retrieved_papers"][0]["abstract"]) == 120
    assert "不要回答问题本身" in messages[0]["content"]


def test_parses_an_insufficient_decision() -> None:
    decision = parse_retrieval_decision(
        json.dumps(
            {
                "sufficient": False,
                "reason": "缺少长任务状态信息",
                "missing_aspects": ["long-horizon state"],
                "rewritten_query": "long-horizon agent state memory decay",
            }
        )
    )

    assert decision.sufficient is False
    assert decision.missing_aspects == ("long-horizon state",)
    assert decision.rewritten_query == "long-horizon agent state memory decay"


@pytest.mark.parametrize(
    "content",
    [
        "not-json",
        json.dumps({"sufficient": "yes"}),
        json.dumps(
            {
                "sufficient": False,
                "reason": "不足",
                "missing_aspects": [],
                "rewritten_query": None,
            }
        ),
        json.dumps(
            {
                "sufficient": True,
                "reason": "足够",
                "missing_aspects": [],
                "rewritten_query": "unexpected query",
            }
        ),
    ],
)
def test_rejects_invalid_control_decisions(content: str) -> None:
    with pytest.raises(RetrievalDecisionError):
        parse_retrieval_decision(content)


class FakeCompletions:
    def create(self, **options: object) -> object:
        assert options["response_format"] == {"type": "json_object"}
        assert options["max_tokens"] == 300
        assert options["temperature"] == 0.0
        content = json.dumps(
            {
                "sufficient": True,
                "reason": "证据直接覆盖问题",
                "missing_aspects": [],
                "rewritten_query": None,
            }
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def test_judge_uses_bounded_json_generation() -> None:
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )

    decision = judge_retrieval(
        "Agent memory",
        [make_result()],
        settings=SETTINGS,
        client=client,
    )

    assert decision.sufficient is True
    assert decision.rewritten_query is None
