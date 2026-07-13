# React frontend

This client talks to the local FastAPI application and keeps only bounded
conversation state plus the previous response's arXiv IDs in the browser.

Main responsibilities:

- `src/hooks/`: knowledge-base loading and conversational request state.
- `src/components/chat/`: input, pending state, turns, and empty state.
- `src/components/results/`: paper evidence and result composition.
- `src/components/trace/`: Agent execution trace rendering.
- `src/api.ts` and `src/types.ts`: HTTP calls and shared response contracts.

From the repository root, start the backend:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --reload
```

In a second PowerShell window, start the frontend:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

The default API is `http://127.0.0.1:8000`. Set `VITE_API_BASE_URL` in an
ignored `frontend/.env.local` file only when the API runs elsewhere.

Verification commands:

```powershell
npm test
npm run build
```
