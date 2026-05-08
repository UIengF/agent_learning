# Agentic Academic RAG

这是一个本地学术知识库 Agentic RAG 项目，基于 Python、LangGraph 和 FastAPI 构建，支持 CLI、HTTP API 和 Web UI 多入口访问。

请在仓库根目录执行开发、测试、索引构建和运行命令。

## 入口文件

主入口是 [graph_rag.py](D:/Code/agent_learning/graph_rag.py)。

## 项目结构

- [graph_rag.py](D:/Code/agent_learning/graph_rag.py)：CLI 入口
- [graph_rag_app](D:/Code/agent_learning/graph_rag_app)：应用核心代码
- [tests](D:/Code/agent_learning/tests)：自动化测试
- [agent](D:/Code/agent_learning/agent)：本地检索索引目录
- [runtime](D:/Code/agent_learning/runtime)：运行日志、checkpoint 和后台任务状态
- [evals](D:/Code/agent_learning/evals)：检索和 Agent 评测数据集
- [scripts](D:/Code/agent_learning/scripts)：启动、状态检查、停止和评测脚本
- [skills](D:/Code/agent_learning/skills)：可通过 `load_skill` 动态加载的默认 harness skills

## 常用命令

进入仓库根目录：

```powershell
cd D:\Code\agent_learning
```

检查项目使用的 Conda 环境：

```powershell
conda run -n langgraph python --version
```

启动、查看状态和停止服务：

```powershell
.\scripts\start.ps1
.\scripts\status.ps1
.\scripts\stop.ps1
```

构建知识库索引：

```powershell
conda run -n langgraph python graph_rag.py index build --kb-path "<knowledge-base-path>" --output-dir .\agent
```

检查已有索引：

```powershell
conda run -n langgraph python graph_rag.py index inspect --index-dir .\agent
```

直接检索，不调用大模型：

```powershell
conda run -n langgraph python graph_rag.py query run --index-dir .\agent --question "openai agent"
```

运行本地检索 + Web fallback 的 Agent 问答：

```powershell
conda run -n langgraph python graph_rag.py ask --index-dir .\agent --question "What changed recently about OpenAI agents?"
```

启动 Web UI：

```powershell
conda run -n langgraph python graph_rag.py ui --index-dir .\agent
```

默认访问地址是 `http://127.0.0.1:8765`。

## 模型配置

在 `.env` 中配置 OpenAI-compatible 模型服务：

```env
RAG_MODEL_API_KEY=<provider-api-key>
RAG_MODEL_API_BASE=https://api.deepseek.com
RAG_MODEL_NAME=deepseek-v4-flash
```

`DASHSCOPE_API_KEY` 仍作为兼容旧配置的 fallback，但新配置建议使用 `RAG_MODEL_*`。

## Web 搜索

可在 `.env` 中配置 Web 搜索提供方：

```env
RAG_WEB_SEARCH_PROVIDER=searxng
RAG_SEARXNG_URL=http://127.0.0.1:8080
RAG_SEARXNG_ENGINES=google,bing,duckduckgo
RAG_SEARXNG_CATEGORIES=general
RAG_SEARXNG_LANGUAGE=zh-CN
```

本地 SearXNG 可通过 Docker 启动：

```powershell
docker compose -f docker-compose.searxng.yml up -d
```

如果 SearXNG 不可用，后端会 fallback 到 DuckDuckGo HTML 搜索，并在检索 debug 信息中记录 provider 失败原因。

## HTTP API

常用接口：

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

问答示例：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8765/api/ask" `
  -ContentType "application/json" `
  -Body '{"question":"What does the knowledge base say about agents?","index_dir":"agent","session_id":"demo"}'
```

检索示例：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8765/api/retrieve" `
  -ContentType "application/json" `
  -Body '{"query":"agent tools","index_dir":"agent","top_k":3,"strategy":"hybrid"}'
```

所有 HTTP 响应都会包含 `X-Request-ID`，用于日志关联。运行状态接口不会暴露密钥。

## Agent Harness 能力

系统在 LangGraph Agent 外增加了几层工程化 harness：

- Skill loading：默认 skill 放在 `skills/<name>/SKILL.md`，Agent 会看到 skill 清单，并可按需调用 `load_skill` 加载具体策略。
- Research plan：Agent 可调用 `research_plan_update` 记录问题、计划步骤、已收集证据、证据缺口和下一步行动。
- Structured trace：运行事件以 JSONL 写入 `runtime/traces`，覆盖上下文构建、LLM 决策、工具调用、缓存命中、checkpoint 和答案摘要。
- Evidence cache：缓存工具证据，减少重复上下文和 token 消耗。
- Context compression：对历史上下文进行预算控制和摘要压缩。

配置示例：

```env
RAG_SKILLS_DIR=skills
RAG_STRUCTURED_TRACE_ENABLED=true
RAG_STRUCTURED_TRACE_DIR=runtime/traces
```

当前内置 skills：

- `retrieval-debug`：用于诊断本地检索、分块、索引、查询改写和元数据重排问题。
- `web-scholar-research`：用于规划 Web Search、Web Fetch、官方来源优先和 Google Scholar 学术检索。
- `agent-evaluation`：用于分层评估 retrieval、tool trajectory、sources 和最终答案。
- `grounded-answering`：用于基于本地、网页和 Scholar 证据合成有依据的最终回答。

## 权限与后台任务

系统在使用用户传入路径或 URL 前会执行权限策略：

- 索引目录默认限制在项目目录内
- 可限制本地或私有网络网页抓取
- 后台任务支持异步提交、状态查询和日志读取

配置示例：

```env
RAG_ALLOWED_INDEX_ROOTS=agent;runtime
RAG_ALLOW_LOCAL_WEB_FETCH=false
RAG_ALLOW_PRIVATE_WEB_FETCH=false
RAG_JOB_RUNTIME_DIR=runtime/jobs
RAG_JOB_MAX_LOG_CHARS=12000
```

提交索引构建任务：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8765/api/jobs/index-build" `
  -ContentType "application/json" `
  -Body '{"kb_path":"docs","output_dir":"agent/job-index"}'
```

查询任务状态和日志：

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8765/api/jobs/<job_id>"
Invoke-RestMethod -Uri "http://127.0.0.1:8765/api/jobs/<job_id>/log"
```

CLI 也支持查看任务状态：

```powershell
conda run -n langgraph python graph_rag.py job status --job-id <job_id>
conda run -n langgraph python graph_rag.py job log --job-id <job_id>
```

## 学术检索与评测

Google Scholar 搜索并导出 Markdown：

```powershell
conda run -n langgraph python graph_rag.py scholar search --topic "graph rag" --count 5 --save-md
```

运行评测：

```powershell
conda run -n langgraph python graph_rag.py eval run --dataset evals\datasets\agent-smoke.jsonl --index-dir .\agent
```

评测结果保存在 `runtime/evals/*` 下。LangSmith tracing 是可选观测层，本地评测文件仍是主要结果来源。

## 质量检查

安装开发依赖：

```powershell
conda run -n langgraph python -m pip install -r requirements-dev.txt
```

运行质量检查：

```powershell
conda run -n langgraph python -m ruff format --check .
conda run -n langgraph python -m ruff check .
conda run -n langgraph python -m pyright
conda run -n langgraph python -m coverage run -m unittest discover -s tests -v
conda run -n langgraph python -m coverage report
```

GitHub Actions 会在仓库根目录运行同一组检查。

## 运行产物

运行产物默认保存在 [runtime](D:/Code/agent_learning/runtime)：

- `runtime/checkpoints.db`
- `runtime/logs/graph_rag.log`
- `runtime/service-stdout.log`
- `runtime/service-stderr.log`
- `runtime/jobs`
- `runtime/evals`
- `runtime/traces`

本地索引默认保存在 [agent](D:/Code/agent_learning/agent)。
