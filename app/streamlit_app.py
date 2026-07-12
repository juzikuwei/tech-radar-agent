"""Streamlit interface for the local arXiv RAG assistant."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.environment import load_repository_env

load_repository_env()

import streamlit as st

from config.model_settings import ModelSettings
from rag.application import RagResult, run_rag
from rag.conversation import ConversationTurn, MAX_CONVERSATION_TURNS
from rag.embedder import E5Embedder
from rag.keyword_search import DEFAULT_DATABASE_PATH, ensure_keyword_index
from rag.llm_client import RetryNotice
from rag.reranker import CrossEncoderReranker
from rag.vector_store import get_persistent_collection
from app.trace_view import show_execution_trace


st.set_page_config(page_title="AI/Agent 技术雷达", page_icon="📡", layout="wide")


@st.cache_resource
def load_runtime() -> tuple[
    object,
    E5Embedder,
    CrossEncoderReranker,
    ModelSettings,
]:
    """Load persistent resources once for the Streamlit process."""
    ensure_keyword_index(DEFAULT_DATABASE_PATH)
    return (
        get_persistent_collection(),
        E5Embedder(),
        CrossEncoderReranker(),
        ModelSettings.from_env(),
    )


def show_papers(result: RagResult) -> None:
    """Render retrieved evidence independently from answer generation."""
    st.subheader("检索到的论文")
    for rank, paper in enumerate(result.papers, start=1):
        with st.container(border=True):
            st.markdown(f"**{rank}. [{paper.title}]({paper.entry_url})**")
            left, middle, right = st.columns(3)
            left.caption(f"arXiv ID：{paper.arxiv_id}")
            middle.caption(f"分类：{paper.primary_category}")
            if paper.rerank_score is not None:
                right.caption(f"重排分数：{paper.rerank_score:.4f}")
            elif paper.similarity is not None:
                right.caption(f"向量相似度：{paper.similarity:.4f}")


st.title("📡 AI/Agent 技术雷达助手")
st.caption("基于本地 arXiv 标题与摘要进行检索，并使用检索证据生成中文回答。")

try:
    collection, embedder, reranker, settings = load_runtime()
except Exception as error:
    st.error(f"运行环境加载失败：{error}")
    st.stop()

st.sidebar.metric("知识库论文数", collection.count())
st.sidebar.caption("数据来源：本地 SQLite 与 ChromaDB；提问阶段不访问 arXiv。")

if "conversation_turns" not in st.session_state:
    st.session_state.conversation_turns = []
if "active_evidence" not in st.session_state:
    st.session_state.active_evidence = ()

st.sidebar.metric(
    "当前会话轮数",
    f"{len(st.session_state.conversation_turns)}/{MAX_CONVERSATION_TURNS}",
)
if st.sidebar.button("新对话", use_container_width=True):
    st.session_state.conversation_turns = []
    st.session_state.active_evidence = ()
    st.rerun()

for turn in st.session_state.conversation_turns:
    with st.chat_message("user"):
        st.markdown(turn.user_message)
    with st.chat_message("assistant"):
        st.markdown(turn.assistant_message)

with st.form("question_form", clear_on_submit=True):
    question = st.text_area(
        "请输入技术问题",
        placeholder="例如：多 Agent 系统执行失败后，如何定位最早出错步骤？",
        height=100,
    )
    submitted = st.form_submit_button("提问", type="primary")

if submitted:
    if not question.strip():
        st.warning("请输入一个非空问题。")
        st.stop()

    retry_message = st.empty()
    with st.chat_message("user"):
        st.markdown(question.strip())

    def show_retry(notice: RetryNotice) -> None:
        retry_message.warning(
            f"模型连接暂时失败，{notice.wait_seconds:g} 秒后进行 "
            f"第 {notice.next_attempt}/{notice.max_attempts} 次尝试。"
        )

    try:
        with st.spinner("正在检索论文并生成回答……"):
            rag_result = run_rag(
                question,
                top_k=5,
                collection=collection,
                embedder=embedder,
                reranker=reranker,
                database_path=DEFAULT_DATABASE_PATH,
                settings=settings,
                on_retry=show_retry,
                conversation_history=tuple(
                    st.session_state.conversation_turns
                ),
                active_evidence=tuple(st.session_state.active_evidence),
            )
    except Exception as error:
        st.error(f"论文检索失败：{error}")
        st.stop()
    finally:
        retry_message.empty()

    if not rag_result.papers:
        st.warning("当前向量库中没有可供回答的论文。")
        st.stop()

    with st.chat_message("assistant"):
        if rag_result.answer is not None:
            st.markdown(rag_result.answer)
        else:
            st.error("回答生成失败，但检索结果仍可查看。请稍后重试。")
            if rag_result.generation_error:
                st.caption(rag_result.generation_error)

    if rag_result.answer is not None:
        turns = [
            *st.session_state.conversation_turns,
            ConversationTurn(
                user_message=question.strip(),
                assistant_message=rag_result.answer,
            ),
        ]
        st.session_state.conversation_turns = turns[-MAX_CONVERSATION_TURNS:]
        st.session_state.active_evidence = rag_result.papers

    show_execution_trace(rag_result.trace)
    show_papers(rag_result)
