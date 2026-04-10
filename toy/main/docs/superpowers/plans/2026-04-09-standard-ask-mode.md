# Standard Ask Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate index building from question answering so runtime `ask` only uses an existing index while preserving session checkpoint and resume behavior.

**Architecture:** Keep `index build` and `query run` as retrieval-layer commands, add an explicit `ask` command for agentic answering, and move the runtime boundary from `kb_path` to `index_dir`. Preserve backward compatibility by routing the legacy no-subcommand entrypoint through the new `ask` path after resolving an existing index, but reject missing indexes instead of rebuilding them implicitly.

**Tech Stack:** Python 3.12, `argparse`, `unittest`, LangGraph/LangChain runtime integration, SQLite-backed retrieval index and checkpoint storage.

---

### Task 1: Lock the CLI contract with tests

**Files:**
- Modify: `D:\Code\agent_learning\toy\main\tests\test_agent_tool_limits.py`
- Create: `D:\Code\agent_learning\toy\main\tests\test_cli_standard_mode.py`
- Test: `D:\Code\agent_learning\toy\main\tests\test_cli_standard_mode.py`

- [ ] **Step 1: Write failing tests for the new command contract**

```python
def test_parse_args_supports_ask_command(self):
    args = parse_args(["ask", "--index-dir", "index", "--question", "What is this?"])
    self.assertEqual(args.command, "ask")
    self.assertEqual(args.index_dir, "index")
```

```python
def test_main_ask_uses_run_or_resume_with_index_dir(self):
    with patch("graph_rag_app.cli.run_or_resume", return_value="answer") as run_or_resume:
        exit_code = main(["ask", "--index-dir", "index", "--question", "Q"])
    self.assertEqual(exit_code, 0)
    run_or_resume.assert_called_once()
```

```python
def test_legacy_default_path_requires_existing_index(self):
    with patch("graph_rag_app.cli.resolve_existing_index_for_kb", return_value=Path("index")):
        exit_code = main(["--kb-path", "kb", "--question", "Q"])
    self.assertEqual(exit_code, 0)
```

- [ ] **Step 2: Run targeted tests and confirm they fail for the expected reasons**

Run: `python -m unittest D:\Code\agent_learning\toy\main\tests\test_cli_standard_mode.py -v`

Expected: failures showing `ask` parsing or index-only runtime behavior does not exist yet.

- [ ] **Step 3: Add a failing runtime test for no implicit rebuilds**

```python
def test_build_agent_loads_existing_index_without_ensure(self):
    with patch("graph_rag_app.agent.load_index") as load_index:
        with patch("graph_rag_app.agent.ensure_index_for_kb") as ensure_index_for_kb:
            build_agent(index_dir="index", app_config=config)
    ensure_index_for_kb.assert_not_called()
    load_index.assert_called_once_with("index", keyword_weight=config.retrieval.keyword_weight)
```

- [ ] **Step 4: Re-run the new tests and confirm the failure is still on missing implementation**

Run: `python -m unittest D:\Code\agent_learning\toy\main\tests\test_cli_standard_mode.py D:\Code\agent_learning\toy\main\tests\test_agent_standard_mode.py -v`

Expected: failures tied to missing `ask` command, missing resolver, or outdated `build_agent` signature.

### Task 2: Move runtime inputs from knowledge-base path to index path

**Files:**
- Modify: `D:\Code\agent_learning\toy\main\graph_rag_app\cli.py`
- Modify: `D:\Code\agent_learning\toy\main\graph_rag_app\runtime.py`
- Modify: `D:\Code\agent_learning\toy\main\graph_rag_app\agent.py`
- Modify: `D:\Code\agent_learning\toy\main\graph_rag_app\__init__.py`

- [ ] **Step 1: Add explicit `ask` parsing in the CLI**

```python
ask_parser = subparsers.add_parser("ask", help="Run the LangGraph agent against an existing index.")
ask_parser.add_argument("--index-dir", required=True)
ask_parser.add_argument("--question", required=False)
ask_parser.add_argument("--session-id", default=os.getenv("RAG_SESSION_ID", DEFAULT_SESSION_ID))
ask_parser.add_argument("--checkpoint-db", default=os.getenv("RAG_CHECKPOINT_DB", DEFAULT_CHECKPOINT_DB))
ask_parser.add_argument("--resume", action="store_true", default=parse_bool_env("RAG_RESUME", False))
ask_parser.add_argument("--interrupt-after", action="append", default=None)
```

- [ ] **Step 2: Route both `ask` and the legacy default path through one helper**

```python
def _handle_ask(args: argparse.Namespace) -> int:
    checkpointer = build_sqlite_checkpointer(args.checkpoint_db)
    answer = run_or_resume(
        question=args.question or DEFAULT_QUESTION,
        index_dir=args.index_dir,
        session_id=args.session_id,
        checkpointer=checkpointer,
        resume=args.resume,
        interrupt_after=args.interrupt_after,
    )
    print(answer)
    return 0
```

- [ ] **Step 3: Change runtime to accept `index_dir` instead of `docx_path`**

```python
def run_or_resume(
    question: str,
    index_dir: str | Path,
    session_id: str = DEFAULT_SESSION_ID,
    checkpointer: Any | None = None,
    resume: bool = False,
    interrupt_after: list[str] | None = None,
    *,
    app_config: AppConfig | None = None,
) -> str:
    config_obj = app_config or build_app_config(index_dir=index_dir, ...)
    agent = build_agent(index_dir=index_dir, checkpointer=checkpointer, app_config=config_obj, ...)
```

- [ ] **Step 4: Change agent construction to load an existing index only**

```python
def build_agent(
    index_dir: str | Path,
    checkpointer: Any | None = None,
    app_config: AppConfig | None = None,
    *,
    ensure_log_file: Any | None = None,
    append_log: Any | None = None,
    shorten_text: Any | None = None,
) -> Agent:
    config = app_config or build_app_config(index_dir=index_dir)
    retriever = load_index(index_dir, keyword_weight=config.retrieval.keyword_weight)
    tool = LocalRAGRetrieveTool(store=retriever, min_evidence_score=config.generation.min_evidence_score)
```

- [ ] **Step 5: Run the targeted tests and confirm they now pass**

Run: `python -m unittest D:\Code\agent_learning\toy\main\tests\test_cli_standard_mode.py D:\Code\agent_learning\toy\main\tests\test_agent_standard_mode.py -v`

Expected: PASS for `ask` parsing, runtime wiring, and no-implicit-rebuild assertions.

### Task 3: Keep compatibility while forbidding hidden rebuilds

**Files:**
- Modify: `D:\Code\agent_learning\toy\main\graph_rag_app\indexing.py`
- Modify: `D:\Code\agent_learning\toy\main\graph_rag_app\cli.py`
- Test: `D:\Code\agent_learning\toy\main\tests\test_cli_standard_mode.py`

- [ ] **Step 1: Add a resolver that finds an existing index without building it**

```python
def resolve_existing_index_for_kb(kb_path: str | Path) -> Path:
    if is_index_directory(kb_path):
        return Path(kb_path)
    index_dir = default_index_dir_for_kb(kb_path)
    if not is_index_directory(index_dir):
        raise FileNotFoundError("No built index found. Run `index build` first.")
    return index_dir
```

- [ ] **Step 2: Use the resolver in the legacy default path**

```python
docx_path = args.kb_path or args.docx_path
index_dir = resolve_existing_index_for_kb(docx_path)
args.index_dir = str(index_dir)
return _handle_ask(args)
```

- [ ] **Step 3: Add a failing-then-passing compatibility test**

```python
def test_legacy_default_path_uses_existing_index_without_building(self):
    with patch("graph_rag_app.cli.resolve_existing_index_for_kb", return_value=Path("index")) as resolver:
        with patch("graph_rag_app.cli.run_or_resume", return_value="answer") as run_or_resume:
            main(["--kb-path", "kb", "--question", "Q"])
    resolver.assert_called_once_with("kb")
    run_or_resume.assert_called_once()
```

- [ ] **Step 4: Run the CLI test file again**

Run: `python -m unittest D:\Code\agent_learning\toy\main\tests\test_cli_standard_mode.py -v`

Expected: PASS, including the compatibility path and missing-index error behavior.

### Task 4: Verify the whole project behavior

**Files:**
- Modify: `D:\Code\agent_learning\toy\main\graph_rag_app\graph_rag.py` (only if export surface needs sync)
- Test: `D:\Code\agent_learning\toy\main\tests\test_chunking.py`
- Test: `D:\Code\agent_learning\toy\main\tests\test_agent_tool_limits.py`
- Test: `D:\Code\agent_learning\toy\main\tests\test_cli_standard_mode.py`

- [ ] **Step 1: Run the project tests from the project root with the correct import path**

Run: `python -m unittest discover -s tests -v`

Expected: PASS for old chunking and tool-limit tests plus the new CLI/runtime tests.

- [ ] **Step 2: Smoke-test the CLI help output**

Run: `python graph_rag.py --help`

Expected: help text includes `index`, `query`, and `ask`.

Run: `python graph_rag.py ask --help`

Expected: help text includes `--index-dir`, `--session-id`, `--checkpoint-db`, `--resume`, and `--interrupt-after`.

- [ ] **Step 3: Capture any follow-up cleanup as a separate task instead of extending scope**

```text
If config stale-detection or explicit rebuild commands need refinement, track them separately.
Do not add auto-rebuild logic back into ask.
```
