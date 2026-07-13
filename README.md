# AI/Agent Tech Radar

一个面向 AI Agent 技术研究的本地知识助手：批量收集 arXiv 论文，使用混合检索与重排寻找证据，再由固定 RAG 管线或有界 ReAct Agent 生成带引用的中文回答。

这个仓库也是一个渐进式 Agent 工程实践项目，重点不是堆叠框架，而是把数据摄取、检索、工具调用、失败降级、引用约束和执行可观测性做成可运行、可测试的完整链路。

## 当前能力

- 手动批量抓取 arXiv，保存可追溯 JSONL 快照，并幂等导入 SQLite。
- 使用 multilingual E5、BM25、RRF 和 Cross-encoder 完成中英跨语言混合检索与重排。
- 支持固定 Agentic RAG 管线：查询改写、证据充分性判断、有限二次检索和资料不足拒答。
- 支持有界 ReAct 模式：模型维护研究计划并在本地论文检索与可选网页搜索之间选择工具。
- Tavily 网页搜索只用于澄清新术语和形成更准确的论文查询；网页内容不会进入回答证据，也不可引用。
- React 聊天界面实时展示 NDJSON Trace、论文引用、多轮会话和 Agent 降级状态。
- 提供 FastAPI HTTP API，以及带 Bearer Token 的只读 Streamable HTTP MCP Server。
- 关键网络边界均可替换或 mock；测试默认离线运行。

## 系统流程

```mermaid
flowchart LR
    subgraph Offline[离线知识库构建]
        A[arXiv API] --> B[JSONL 快照]
        B --> C[规范化与去重]
        C --> D[(SQLite)]
        D --> E[多语言 E5]
        E --> F[(ChromaDB)]
    end

    subgraph Online[在线问答]
        U[用户问题] --> P{Pipeline / ReAct}
        P --> H[BM25 + Vector]
        H --> R[RRF + Cross-encoder]
        R --> J[证据充分性判断]
        J --> G[带 arXiv 引用的中文回答]
        P -.术语澄清.-> W[Tavily Web Search]
        W -.仅生成检索词.-> H
    end

    D --> H
    F --> H
    G --> UI[React + Live Trace]
    D --> MCP[Read-only MCP tools]
    F --> MCP
```

系统的核心证据边界是：最终答案只能基于本地 SQLite/ChromaDB 中的 arXiv 论文。外部网页结果、模型既有知识和执行 Trace 都不能成为引用来源。

## 技术栈

| 层 | 技术 |
|---|---|
| 数据源 | arXiv API |
| 当前状态存储 | SQLite |
| 向量索引 | ChromaDB |
| 检索 | multilingual E5 + BM25 + RRF |
| 重排 | Cross-encoder |
| 模型接口 | OpenAI-compatible API |
| 后端 | FastAPI + NDJSON streaming |
| Agent | 手写有界 ReAct 循环 |
| 工具协议 | MCP Streamable HTTP |
| 前端 | React 19 + TypeScript + Vite |
| 测试 | pytest + Vitest |

## 快速开始

当前开发环境以 Windows、PowerShell 和 Python 3.12 为基准。

### 1. 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

cd frontend
npm install
cd ..
```

首次加载嵌入模型和 Cross-encoder 时可能需要下载模型文件。

### 2. 配置运行环境

在仓库根目录创建一个不会被 Git 跟踪的 `.env` 文件。项目故意不提供 `.env.example`，请按需要配置以下变量：

- `LLM_API_KEY`：必需，OpenAI-compatible 模型服务的密钥。
- `LLM_BASE_URL`：必需，模型服务的 API 地址。
- `LLM_MODEL`：必需，聊天模型名称。
- `TAVILY_API_KEY`：可选；缺失时自动禁用 ReAct 的 `web_search` 工具。
- `MCP_AUTH_TOKEN`：仅运行 MCP Server 时必需，至少 16 个字符。
- `MCP_HOST`、`MCP_PORT`、`MCP_ALLOWED_HOSTS`：可选的 MCP 网络设置。

前端默认连接 `http://127.0.0.1:8000`。如需修改，在被忽略的 `frontend/.env.local` 中设置 `VITE_API_BASE_URL`。

### 3. 准备本地论文数据

列出预设 arXiv 查询：

```powershell
python -m ingestion.run_arxiv_ingestion --list-queries
```

抓取一个小批次，并将命令输出的快照路径用于后续导入：

```powershell
python -m ingestion.run_arxiv_ingestion --query-name agent_core --max-results 3
python -m ingestion.import_snapshot data/raw/<snapshot>.jsonl
python -m rag.indexer
```

抓取是手动批处理；在线问答不会实时调用 arXiv。`data/` 中的快照、SQLite 数据库和 ChromaDB 索引默认不提交到 Git。

### 4. 启动 Web 应用

安装完成后，可以在仓库根目录运行：

```powershell
.\start_services.ps1
```

脚本会启动 FastAPI 和 Vite，并打开 `http://127.0.0.1:5173`。也可以分别启动：

```powershell
python -m uvicorn api.main:app --reload
```

```powershell
cd frontend
npm run dev
```

API 文档位于 `http://127.0.0.1:8000/docs`。

## HTTP API

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/health` | 进程健康检查 |
| `GET` | `/knowledge-base/stats` | 返回 SQLite 论文数与向量数 |
| `POST` | `/chat` | 返回完整问答结果与 Trace |
| `POST` | `/chat/stream` | 先流式发送 Trace，再发送最终结果 |

`/chat` 支持 `pipeline` 和 `react` 两种模式。ReAct 规划或工具执行出现不可恢复错误时，会降级到可靠的固定管线，并在响应中标记 `fallback_used`。

## MCP Server

项目还提供独立的只读 MCP 服务：

```powershell
python -m mcp_server.main
```

当前工具包括：

- `query_knowledge_base(query, top_k=3)`
- `get_paper_by_arxiv_id(arxiv_id)`
- `get_knowledge_base_stats()`

客户端连接 `/mcp` 时必须发送 Bearer Token。完整的数据边界、配置和部署说明见 [docs/mcp.md](docs/mcp.md)。

## 验证

在仓库根目录运行后端测试：

```powershell
python -m pytest
```

在 `frontend/` 中运行前端测试与生产构建：

```powershell
npm test
npm run build
```

测试覆盖数据规范化、幂等导入、检索、引用、拒答、对话状态、ReAct 决策契约、网页搜索失败和 HTTP/MCP 边界。

## 项目结构

```text
api/            FastAPI 路由、请求契约与运行时生命周期
config/         查询、模型、MCP 和网页搜索配置边界
frontend/       React 聊天界面、引用卡片与实时 Trace
ingestion/      arXiv 抓取、规范化、快照与 SQLite 导入
mcp_server/     只读 Streamable HTTP MCP 适配器
rag/            检索、重排、固定管线、ReAct Agent 与回答生成
tests/          默认离线运行的 Python 测试
docs/           ADR 决策日志和 MCP 使用说明
first.md        完整范围、学习路线与阶段验收记录
```

FastAPI、MCP 和命令行入口复用 `rag/` 中的领域能力；网络、存储、检索和 UI 逻辑保持分离。

## 当前限制与路线图

- 知识库目前只使用 arXiv 标题与摘要，不读取 PDF 全文。
- 数据更新由人工触发，没有定时抓取或在线增量更新。
- ReAct 受到每请求动作预算约束，尚未加入请求级 token 上限。
- 网页搜索带来间接 prompt injection 表面，目前通过“不进入回答证据”限制影响，完整 Guardrails 尚待实现。
- MCP 的共享 Token 适合本地开发和受控邀请；公开服务前需要 OAuth 2.1、按用户限流和 HTTPS 反向代理。
- 尚未引入 LangGraph、长期记忆、Human-in-the-Loop 或多 Agent 编排。

下一阶段计划是实现最小的跨会话长期记忆。完整里程碑见 [first.md](first.md)，重要架构取舍见 [docs/decision-log.md](docs/decision-log.md)。

## 开发约定

项目使用 Conventional Commits，并要求重要数据源、存储、模型、框架、数据契约、部署和安全决定记录 ADR。提交前请运行后端测试、前端测试和前端构建；详细协作约束见 [AGENTS.md](AGENTS.md)。
