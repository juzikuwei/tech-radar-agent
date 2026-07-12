import json
from pathlib import Path

import pytest

from config.arxiv_queries import get_arxiv_query, load_arxiv_queries


def test_repository_queries_cover_agent_topics() -> None:
    queries = load_arxiv_queries()

    assert {
        "agent_core",
        "agentic_rag",
        "multi_agent",
        "tool_use",
        "agent_reasoning",
        "agent_memory",
        "web_agent",
        "code_agent",
        "mcp_agents",
        "agent_skills",
        "rag_core",
        "rag_evaluation",
        "embodied_agents",
        "robotics_agents",
        "multimodal_agents",
        "vision_language_agents",
        "medical_ai_agents",
        "finance_ai_agents",
        "security_ai_agents",
        "scientific_ai_agents",
        "education_ai_agents",
        "data_ai_agents",
        "software_engineering_agents",
        "autonomous_driving_agents",
        "game_ai_agents",
        "recommendation_agents",
        "simulation_agents",
        "planning_agents",
        "conversational_agents",
        "collaborative_agents",
        "reinforcement_learning_agents",
        "personal_ai_agents",
        "global_ai_agents",
        "global_llm_agents",
        "global_language_model_agents",
        "global_agentic_ai",
        "global_multi_agent_systems",
        "global_autonomous_ai_agents",
        "global_tool_agents",
        "global_mcp",
        "global_rag",
        "global_agent_memory",
        "global_agent_reasoning",
        "global_agent_planning",
        "global_agent_safety",
        "global_agent_evaluation",
        "global_research_agents",
        "global_browser_agents",
    } <= queries.keys()
    assert all(
        any(
            term in query.lower()
            for term in (
                "agent",
                "rag",
                "retrieval augmented",
                "retrieval evaluation",
                "model context protocol",
                "tool use",
                "tool calling",
            )
        )
        for query in queries.values()
    )


def test_rejects_unknown_query_name(tmp_path: Path) -> None:
    path = tmp_path / "queries.json"
    path.write_text(json.dumps({"agent_core": "cat:cs.AI"}), encoding="utf-8")

    with pytest.raises(ValueError, match="available"):
        get_arxiv_query("missing", path)


def test_rejects_empty_query_value(tmp_path: Path) -> None:
    path = tmp_path / "queries.json"
    path.write_text(json.dumps({"agent_core": ""}), encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty"):
        load_arxiv_queries(path)
