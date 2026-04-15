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

The UI listens at `http://127.0.0.1:8765` by default. Use `--host` and `--port`
to change the bind address.

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
