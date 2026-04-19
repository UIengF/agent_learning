# graph-rag demo

This directory is the active Graph RAG demo project.

Use this directory as the working directory for development, testing, indexing,
and runtime commands.

## Entry point

The main entrypoint is
[graph_rag.py](D:/Code/agent_learning/agent_rag/graph_rag.py).

## Project layout

- [graph_rag.py](D:/Code/agent_learning/agent_rag/graph_rag.py): CLI entrypoint
- [graph_rag_app](D:/Code/agent_learning/agent_rag/graph_rag_app): application code
- [tests](D:/Code/agent_learning/agent_rag/tests): automated tests
- [agent](D:/Code/agent_learning/agent_rag/agent): local retrieval index files
- [runtime](D:/Code/agent_learning/agent_rag/runtime): generated logs and checkpoints

## Common commands

Run these commands from [agent_rag](D:/Code/agent_learning/agent_rag):

```powershell
cd D:\Code\agent_learning\agent_rag
```

Install runtime dependencies:

```powershell
python -m pip install -r requirements.txt
```

Build an index:

```powershell
python graph_rag.py index build --kb-path "<knowledge-base-path>" --output-dir .\agent
```

Inspect an existing index:

```powershell
python graph_rag.py index inspect --index-dir .\agent
```

Run direct retrieval without the agent:

```powershell
python graph_rag.py query run --index-dir .\agent --question "openai agent"
```

Run the agent with local retrieval and web fallback:

```powershell
python graph_rag.py ask --index-dir .\agent --question "What changed recently about OpenAI agents?"
```

Run the browser question-answering UI:

```powershell
python graph_rag.py ui --index-dir .\agent
```

The UI is served by the FastAPI backend and listens at `http://127.0.0.1:8765`
by default. Use `--host` and `--port` to change the bind address.

Run the FastAPI backend service explicitly:

```powershell
python graph_rag.py serve --index-dir .\agent --host 127.0.0.1 --port 8765
```

Configure web search providers in `.env`:

```env
RAG_WEB_SEARCH_PROVIDER=searxng
RAG_SEARXNG_URL=http://127.0.0.1:8080
RAG_SEARXNG_ENGINES=google,bing,duckduckgo
RAG_SEARXNG_CATEGORIES=general
RAG_SEARXNG_LANGUAGE=zh-CN
```

SearXNG must enable JSON output for `/search?q=...&format=json` to work. In a
self-hosted SearXNG instance, make sure `settings.yml` includes `json` in
`search.formats`. A local Docker setup is included:

```powershell
docker compose -f docker-compose.searxng.yml up -d
```

Then set `RAG_WEB_SEARCH_PROVIDER=searxng` in `.env` and restart the API. If
SearXNG is unavailable or fails, the backend falls back to DuckDuckGo HTML search
and records provider failures in search debug metadata.

Useful HTTP endpoints:

```text
GET  /healthz
GET  /api/config
GET  /api/status
POST /api/ask
POST /api/retrieve
GET  /api/index/inspect?index_dir=agent
POST /api/web/search
POST /api/web/fetch
POST /api/scholar/search
```

Every HTTP response includes an `X-Request-ID` header for log correlation. Pass
your own `X-Request-ID` header to reuse a caller-provided id.

Runtime status does not expose secret values:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8765/api/status"
```

Errors use a stable envelope:

```json
{
  "error": {
    "code": "validation_error",
    "message": "Request validation failed.",
    "details": {},
    "request_id": "..."
  }
}
```

Ask through the API:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8765/api/ask" `
  -ContentType "application/json" `
  -Body '{"question":"What does the knowledge base say about agents?","index_dir":"agent","session_id":"demo"}'
```

The ask response includes answer traceability. Local knowledge-base hits are
returned as `sources` entries with `source_type: "local"`, `source_path`,
optional section metadata, score, and snippet text. External web evidence is
returned as `source_type: "web"` entries with `title`, `url`, optional search
provider metadata, and snippet text. `retrieval_debug.source_count` reports the
number of extracted sources.

Run retrieval without invoking the LLM:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8765/api/retrieve" `
  -ContentType "application/json" `
  -Body '{"query":"agent tools","index_dir":"agent","top_k":3,"strategy":"hybrid"}'
```

Run Google Scholar search and export the results to Markdown:

```powershell
python graph_rag.py scholar search --topic "graph rag" --count 5 --save-md
```

Run tests:

```powershell
python -m unittest discover -s tests -v
```

## Runtime files

Generated runtime files are stored under
[runtime](D:/Code/agent_learning/agent_rag/runtime):

- [runtime/checkpoints.db](D:/Code/agent_learning/agent_rag/runtime/checkpoints.db)
- [runtime/logs/graph_rag.log](D:/Code/agent_learning/agent_rag/runtime/logs/graph_rag.log)

## Notes

- Use [agent](D:/Code/agent_learning/agent_rag/agent) for the local index directory.
- Use [runtime](D:/Code/agent_learning/agent_rag/runtime) for checkpoints and logs.
- If a command depends on relative paths, run it from
  [agent_rag](D:/Code/agent_learning/agent_rag).
