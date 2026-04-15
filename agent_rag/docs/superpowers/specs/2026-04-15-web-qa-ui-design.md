# Web QA UI Design

## Goal

Add a browser-based question-answering interface for the existing Graph RAG agent in `D:\Code\agent_learning\agent_rag`.

## Scope

The first version focuses on full question answering only. It does not include retrieval debugging, index building, web search tools, Scholar search tools, or visual graph inspection.

## User Experience

The user starts the UI from the project root with:

```powershell
python graph_rag.py ui --index-dir .\agent
```

The browser page lets the user:

- Enter a question.
- View or change the index directory.
- View or change the session ID.
- Toggle resume mode.
- Submit a question and see a loading state.
- Read the final answer or an error message.
- Keep recent question-answer pairs in the current browser session.

## Architecture

The UI uses Python standard library HTTP serving to avoid adding package dependencies to the current CLI project. A new `graph_rag_app.web_ui` module owns the HTTP handler, JSON request parsing, static file serving, and the call into existing runtime functions.

The existing `run_or_resume(...)` and `build_sqlite_checkpointer(...)` functions remain the only execution path for agent answers. The UI layer validates HTTP input, times execution, formats JSON, and delegates to the existing runtime.

## Components

- `graph_rag_app/web_ui.py`: Standard-library web server, API request handling, static asset resolution, and launch helper.
- `graph_rag_app/static/index.html`: Main browser interface.
- `graph_rag_app/static/styles.css`: Layout and visual styling.
- `graph_rag_app/static/app.js`: Browser-side form handling, loading state, API calls, and in-memory conversation rendering.
- `graph_rag_app/cli.py`: Adds the `ui` command and dispatches to `serve_ui`.
- `tests/test_web_ui.py`: Unit tests for request validation, answer API execution, static route behavior, and CLI dispatch.
- `README.md`: Documents how to launch the UI.

## API

`POST /api/ask`

Request JSON:

```json
{
  "question": "What does the knowledge base say about agents?",
  "index_dir": ".\\agent",
  "session_id": "graph_rag_default",
  "resume": false
}
```

Success response:

```json
{
  "answer": "Final answer text",
  "question": "What does the knowledge base say about agents?",
  "index_dir": ".\\agent",
  "session_id": "graph_rag_default",
  "resume": false,
  "elapsed_seconds": 1.23
}
```

Error response:

```json
{
  "error": "Question is required."
}
```

## Error Handling

Invalid JSON, missing question, and missing index directory return HTTP 400 with a JSON error. Runtime exceptions return HTTP 500 with a concise JSON error. Static files outside the static directory are not served.

## Testing

Tests use `unittest` and mocks, matching the existing project. The API tests exercise the handler helpers without making real LLM calls. CLI tests verify argument parsing and dispatch to `serve_ui`.
