"""Streamlit rendering for structured Agent execution traces."""

from collections.abc import Sequence

import streamlit as st

from rag.execution_trace import TraceEvent


STATUS_DISPLAY = {
    "completed": ("✅", "完成"),
    "failed": ("⚠️", "失败后降级"),
    "skipped": ("⏭️", "跳过"),
}


def show_execution_trace(events: Sequence[TraceEvent]) -> None:
    """Render ordered module events without exposing secrets or full prompts."""
    with st.expander("Agent 执行过程", expanded=True):
        if not events:
            st.caption("当前请求没有可展示的执行记录。")
            return

        st.caption(
            "以下是本次请求实际完成的模块步骤；耗时包含本地模型或网络调用。"
        )
        for index, event in enumerate(events, start=1):
            icon, status_text = STATUS_DISPLAY[event.status]
            with st.container(border=True):
                title, timing = st.columns([4, 1])
                title.markdown(f"**{index}. {icon} {event.label}**")
                timing.caption(f"{status_text} · {_format_duration(event.duration_ms)}")
                _show_event_details(event)


def _show_event_details(event: TraceEvent) -> None:
    details = event.details
    retrieval_round = details.get("round")
    if retrieval_round is not None:
        st.caption(f"检索轮次：第 {retrieval_round} 轮")

    query = details.get("query")
    if isinstance(query, str) and query:
        st.markdown(f"查询：`{query}`")

    if "sufficient" in details:
        sufficient = bool(details["sufficient"])
        st.markdown(f"证据是否充分：**{'是' if sufficient else '否'}**")

    coverage = details.get("coverage")
    if isinstance(coverage, str) and coverage:
        coverage_labels = {
            "sufficient": "已有证据充分",
            "partial": "已有证据部分充分",
            "unrelated": "新话题，与已有证据无关",
            "none": "没有可复用证据",
        }
        st.markdown(
            f"证据覆盖：**{coverage_labels.get(coverage, coverage)}**"
        )

    next_action = details.get("next_action")
    if isinstance(next_action, str) and next_action:
        action_labels = {
            "answer_from_existing": "直接复用已有证据回答",
            "retrieve_missing": "保留已有证据并检索缺失信息",
            "fresh_retrieval": "放弃旧证据并开始全新检索",
        }
        st.markdown(
            f"下一步动作：**{action_labels.get(next_action, next_action)}**"
        )

    reason = details.get("reason")
    if isinstance(reason, str) and reason:
        st.write(f"判断理由：{reason}")

    rewritten_query = details.get("rewritten_query")
    if isinstance(rewritten_query, str) and rewritten_query:
        st.markdown(f"改写查询：`{rewritten_query}`")

    standalone_question = details.get("standalone_question")
    if isinstance(standalone_question, str) and standalone_question:
        st.markdown(f"完整问题：`{standalone_question}`")

    retrieval_query = details.get("retrieval_query")
    if isinstance(retrieval_query, str) and retrieval_query:
        st.markdown(f"检索查询：`{retrieval_query}`")

    missing_aspects = details.get("missing_aspects")
    if isinstance(missing_aspects, list) and missing_aspects:
        st.write("缺失方面：" + "、".join(str(item) for item in missing_aspects))

    counts = []
    count_labels = {
        "result_count": "输出结果",
        "candidate_count": "重排候选",
        "unique_candidate_count": "去重候选",
        "dense_count": "向量候选",
        "keyword_count": "关键词候选",
        "first_round_count": "首轮结果",
        "second_round_count": "次轮结果",
        "existing_count": "已有证据",
        "new_count": "新候选",
        "paper_count": "回答证据",
    }
    for key, label in count_labels.items():
        value = details.get(key)
        if isinstance(value, int):
            counts.append(f"{label} {value}")
    if counts:
        st.caption(" · ".join(counts))

    top_ids = details.get("top_arxiv_ids")
    if isinstance(top_ids, list) and top_ids:
        st.caption("Top arXiv IDs：" + ", ".join(str(item) for item in top_ids))

    reusable_ids = details.get("reusable_arxiv_ids")
    if isinstance(reusable_ids, list) and reusable_ids:
        st.caption("复用 arXiv IDs：" + ", ".join(str(item) for item in reusable_ids))

    error = details.get("error")
    if isinstance(error, str) and error:
        st.warning(error)


def _format_duration(duration_ms: float) -> str:
    if duration_ms < 1:
        return "<1 ms"
    if duration_ms < 1_000:
        return f"{duration_ms:.0f} ms"
    return f"{duration_ms / 1_000:.2f} s"
