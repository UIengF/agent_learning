# agent_learning

This repository now keeps the main academic Agentic RAG project under
[`agent_rag/`](agent_rag/).

The previous root-level learning scripts have been retired because their
LangGraph, ReAct, memory, retrieval, web search, and UI functionality is covered
by the maintained `agent_rag` project.

## Main Project

See [`agent_rag/README.md`](agent_rag/README.md) for setup, CLI usage, FastAPI
service commands, Web UI access, evaluation datasets, permission controls,
background jobs, and quality checks.

Useful entry points:

```powershell
cd agent_rag
.\scripts\start.ps1
.\scripts\status.ps1
.\scripts\stop.ps1
```

Quality checks:

```powershell
cd agent_rag
python -m pytest
python -m ruff check .
python -m pyright
```
