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

Use the existing adapted Conda environment for this project:

```powershell
conda run -n langgraph python --version
```

Do not install or upgrade dependencies until this environment has been checked.
If dependencies are genuinely missing, review [requirements.txt](D:/Code/agent_learning/agent_rag/requirements.txt)
first and install only after confirming the `langgraph` environment is insufficient.

Start, inspect, and stop the service with the project scripts:

```powershell
.\scripts\start.ps1
.\scripts\status.ps1
.\scripts\stop.ps1
```

Build an index:

```powershell
conda run -n langgraph python graph_rag.py index build --kb-path "<knowledge-base-path>" --output-dir .\agent
```

Inspect an existing index:

```powershell
conda run -n langgraph python graph_rag.py index inspect --index-dir .\agent
```

Run direct retrieval without the agent:

```powershell
conda run -n langgraph python graph_rag.py query run --index-dir .\agent --question "openai agent"
```

Run the agent with local retrieval and web fallback:

```powershell
conda run -n langgraph python graph_rag.py ask --index-dir .\agent --question "What changed recently about OpenAI agents?"
```

Configure the chat model in `.env` with OpenAI-compatible provider settings:

```env
RAG_MODEL_API_KEY=<provider-api-key>
RAG_MODEL_API_BASE=https://api.deepseek.com
RAG_MODEL_NAME=deepseek-v4-flash
```

`DASHSCOPE_API_KEY` is still supported as a backwards-compatible fallback, but
new model/provider switches should use the `RAG_MODEL_*` variables.

Run the browser question-answering UI:

```powershell
conda run -n langgraph python graph_rag.py ui --index-dir .\agent
```

The UI is served by the FastAPI backend and listens at `http://127.0.0.1:8765`
by default. Use `--host` and `--port` to change the bind address.

Run the FastAPI backend service explicitly:

```powershell
conda run -n langgraph python graph_rag.py serve --index-dir .\agent --host 127.0.0.1 --port 8765
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
conda run -n langgraph python graph_rag.py scholar search --topic "graph rag" --count 5 --save-md
```

Run tests:

```powershell
conda run -n langgraph python -m unittest discover -s tests -v
```

## Harness features

The agent keeps the existing LangGraph runtime and adds three optional harness
layers around it:

- Skill loading: put local skills under `skills/<name>/SKILL.md`. The agent sees
  an inventory of available skills and can call `load_skill` when a skill is
  relevant to the current research task.
- Research plan: the agent can call `research_plan_update` to record the current
  question, planned steps, gathered evidence, open gaps, and next action. The
  latest plan is re-injected into context on later model calls.
- Structured trace: runtime events are written as JSONL under `runtime/traces`.
  Trace events include context construction, LLM decisions, tool calls, cache
  hits, tool results, checkpoints, and final answer previews.

Configure these with `.env`:

```env
RAG_SKILLS_DIR=skills
RAG_STRUCTURED_TRACE_ENABLED=true
RAG_STRUCTURED_TRACE_DIR=runtime/traces
```

Example skill:

```text
skills/retrieval-debug/SKILL.md
```

```markdown
---
name: retrieval-debug
description: Diagnose retrieval failures and evidence gaps.
---

Inspect index metadata, compare retrieved source titles, and identify whether
the failure is chunking, query framing, or answer synthesis.
```

## Permissions and background jobs

The service applies a small tool permission policy before using caller-provided
paths or URLs. By default, index directories must stay under this project
directory, while local/private web fetches remain enabled for local development.
Tighten those boundaries in `.env` when exposing the API beyond localhost:

```env
RAG_ALLOWED_INDEX_ROOTS=agent;runtime
RAG_ALLOW_LOCAL_WEB_FETCH=false
RAG_ALLOW_PRIVATE_WEB_FETCH=false
RAG_JOB_RUNTIME_DIR=runtime/jobs
RAG_JOB_MAX_LOG_CHARS=12000
```

Submit a long index build job through the API:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8765/api/jobs/index-build" `
  -ContentType "application/json" `
  -Body '{"kb_path":"docs","output_dir":"agent/job-index"}'
```

Submit an evaluation job:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8765/api/jobs/eval-run" `
  -ContentType "application/json" `
  -Body '{"dataset":"evals/datasets/agent-smoke.jsonl","index_dir":"agent","output_dir":"runtime/evals"}'
```

Check status and read logs:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8765/api/jobs/<job_id>"
Invoke-RestMethod -Uri "http://127.0.0.1:8765/api/jobs/<job_id>/log"
```

The CLI can inspect persisted job records and logs from the same runtime
directory:

```powershell
conda run -n langgraph python graph_rag.py job status --job-id <job_id>
conda run -n langgraph python graph_rag.py job log --job-id <job_id>
```

## Quality checks

Install development-only quality tools:

```powershell
conda run -n langgraph python -m pip install -r requirements-dev.txt
```

Run the same checks locally that CI runs:

```powershell
conda run -n langgraph python -m ruff format --check .
conda run -n langgraph python -m ruff check .
conda run -n langgraph python -m pyright
conda run -n langgraph python -m coverage run -m unittest discover -s tests -v
conda run -n langgraph python -m coverage report
```

GitHub Actions runs these commands from this directory on changes under
`agent_rag/**`.

## Optional LangSmith tracing

The existing local evaluation flow remains the source of truth for pass/fail
reports under `runtime/evals/*`. LangSmith tracing is optional and adds run
observation for `ask` and `eval run` without changing the local report format.

Enable tracing by setting:

```powershell
$env:RAG_LANGSMITH_ENABLED="true"
$env:LANGCHAIN_TRACING_V2="true"
$env:LANGCHAIN_API_KEY="<langsmith-api-key>"
$env:LANGCHAIN_PROJECT="agent-rag"
```

Then run the usual commands:

```powershell
conda run -n langgraph python graph_rag.py ask --index-dir .\agent --question "What changed recently about OpenAI agents?"
conda run -n langgraph python graph_rag.py eval run --dataset evals\datasets\agent-smoke.jsonl --index-dir .\agent
```

Evaluation runs attach dataset metadata such as `dataset_name`, `case_id`,
`group`, and `tags` to the LangSmith run context.

Current LangSmith workflow:

1. Set `RAG_LANGSMITH_ENABLED=true`, `LANGCHAIN_TRACING_V2=true`, and the
   usual LangSmith credentials.
2. `ask` and `resume` runs create a LangSmith chain run with metadata such as
   `session_id`, `index_dir`, and `resume`.
3. `eval run` creates a dataset-level LangSmith run named
   `graph_rag.eval.dataset:<dataset_name>`.
4. Each evaluation case creates its own LangSmith run named
   `graph_rag.eval.case:<case_id>`.
5. Case runs inherit evaluation metadata and derived tags such as
   `mode:eval`, `dataset:<name>`, `group:<group>`, and `case:<id>`.
6. After a case completes, the local evaluation system syncs feedback back to
   the same LangSmith run:
   - `case_passed`
   - `layer_question`
   - `layer_retrieval`
   - `layer_trajectory`
   - `layer_sources`
   - `layer_web_search_quality`
   - `layer_source_quality`
   - `layer_answer`
   - `trajectory_summary`
   - `answer_judge_score`, `answer_grounded`, `answer_complete` when the
     answer judge is enabled
7. After the dataset run finishes, the dataset-level LangSmith run receives
   aggregate feedback such as `dataset_pass_rate` and
   `dataset_layer_<layer_name>`.

That means the local files under `runtime/evals/*` remain the release gate,
while LangSmith becomes the observation and analysis layer for:

- case trace debugging
- layer-by-layer evaluation feedback
- answer judge groundedness/completeness review
- dataset-level pass-rate tracking

## Runtime files

Generated runtime files are stored under
[runtime](D:/Code/agent_learning/agent_rag/runtime):

- [runtime/checkpoints.db](D:/Code/agent_learning/agent_rag/runtime/checkpoints.db)
- [runtime/logs/graph_rag.log](D:/Code/agent_learning/agent_rag/runtime/logs/graph_rag.log)
- `runtime/service-stdout.log`
- `runtime/service-stderr.log`

## Notes

- Use [agent](D:/Code/agent_learning/agent_rag/agent) for the local index directory.
- Use [runtime](D:/Code/agent_learning/agent_rag/runtime) for checkpoints and logs.
- Use `conda run -n langgraph` for project commands unless a task explicitly targets another environment.
- If a command depends on relative paths, run it from
  [agent_rag](D:/Code/agent_learning/agent_rag).
