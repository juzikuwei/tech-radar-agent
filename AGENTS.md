# Repository Guidelines

## Project Structure & Module Organization

`first.md` defines scope, architecture, milestones, and the directory layout (section 8). Create modules only when required. Keep generated data untracked except for small fixtures; never commit `.venv/`.

## Build, Test, and Development Commands

Run from the repository root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
python -m uvicorn api.main:app --reload
python -m mcp_server.main
cd frontend
npm install
npm test
npm run build
npm run dev
```

Prefer `python -m ...` so Python tools use the active interpreter. Run frontend
commands from `frontend/`.

## Coding Style & Naming Conventions

Use four spaces, UTF-8, public-function type hints, and brief docstrings. Use `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants. Separate network, storage, retrieval, and UI logic.

## Testing Guidelines

Use pytest. Name files `test_<module>.py` and tests `test_<behavior>()`. Cover normalization, deduplication, retrieval, citations, and refusal. Mock arXiv or use fixtures; tests must work offline. Run `python -m pytest` before changes.

## Decision Log & Abstractions

Record important data-source, storage, model, framework, data-contract, deployment, and security decisions in `docs/decision-log.md`. Each entry must state the decision, context, alternatives, reason, consequences, and review or migration trigger. Minor implementation choices need no entry.

Do not create abstractions merely because an LLM suggests them. Every abstraction needs a clear responsibility, current usage, and concrete maintenance benefit.

## Commit & Pull Request Guidelines

Use Conventional Commits, for example `feat: add arxiv query experiment`. Pull requests should describe the milestone, verification, and limitations; include UI screenshots.

## Security & Configuration

Keep keys in the ignored `.env` file and never commit secrets, databases, vector indexes, or bulk downloads. This repository intentionally has no `.env.example`; document variable names without values when configuration guidance is needed.

## Agent-Specific Learning Instructions

Use teaching-oriented co-development. Before coding, explain the concept and one runnable goal. Let the user implement or execute meaningful parts, then review results and failures. Do not generate the whole project; connect additions to limitations already observed.

For the extension stages in `first.md` (stage 9+), never start implementing a stage without first explaining its concept, runnable goal, and change scope, and receiving explicit user confirmation.

For every significant module, ensure the user can explain why it exists, the problem it solves, its inputs and outputs, its assumptions, and its failure modes. If not, pause that module and study or test the concept before continuing. Develop systematic thinking with one or two questions grounded in the current milestone, covering boundaries, dependencies, state, trade-offs, observability, or acceptance criteria. Connect answers to the complete ingestion-to-answer pipeline; questions must advance work, not test trivia.
