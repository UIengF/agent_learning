from __future__ import annotations

from pathlib import Path

from .context import build_context
from .context_explain import explain_context
from .evidence_cache import EvidenceCache, compact_text, compress_text
from .agent_progress import progress as _progress
from .plan import CodingPlan, format_plan
from .repo_map import build_repo_map
from .tools import CodingTools


_OVERVIEW_TASK_MARKERS = (
    "overview",
    "summarize",
    "summary",
    "structure",
    "architecture",
    "capabilities",
    "what is this project",
    "介绍",
    "结构",
    "主要能力",
    "项目能力",
    "项目综述",
)

_OVERVIEW_FILES = (
    "README.zh-CN.md",
    "README.md",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "package.json",
)


class AgentContextMixin:
    def _persist_harness_state(
        self,
        plan: CodingPlan,
        evidence_cache: EvidenceCache,
        tools: CodingTools,
        *,
        capture_diff: bool = True,
        context_query: str = "",
    ) -> None:
        _progress("Persisting harness state")
        self.session.plan = plan.to_jsonable()
        self.session.evidence = evidence_cache.to_jsonable()
        try:
            diff = tools.git_diff() if capture_diff else self.session.latest_diff
        except Exception:
            diff = self.session.latest_diff if capture_diff else ""
        self.session.latest_diff = diff
        self.session.summary = build_context(
            history=self.session.history,
            plan=plan,
            evidence_context=evidence_cache.to_context(3000),
            latest_diff=diff,
            max_chars=2000,
            project_instructions=self.project_instructions.as_context(),
            repo_map=self._repo_context_summary(context_query),
            documentation_summary=self._documentation_context(),
            memory_summary=self.memory_store.format(),
        )
        _progress("Harness state persisted")

    def _repo_context_summary(self, query: str) -> str:
        _progress("Building repository context summary")
        repo_summary = build_repo_map(self.workspace).format(max_files=40, max_symbols=60)
        if not query.strip():
            _progress("Repository context summary ready")
            return repo_summary
        explained = explain_context(self.workspace, query)
        _progress("Repository context summary ready")
        return f"{explained}\n\nRepository map summary:\n{repo_summary}"

    def _documentation_context(self) -> str:
        docs: list[str] = []
        for path in sorted(self.workspace.rglob("README*")):
            if not path.is_file():
                continue
            try:
                relative = path.relative_to(self.workspace)
            except ValueError:
                continue
            first_heading = ""
            try:
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        first_heading = stripped.lstrip("#").strip()
                        break
                    if stripped and not first_heading:
                        first_heading = compact_text(stripped, 80)
                        break
            except OSError:
                first_heading = "unreadable"
            docs.append(f"- {relative.as_posix()}: {first_heading or 'no heading'}")
            if len(docs) >= 20:
                break
        return "\n".join(docs)

    def _overview_context(self, task: str) -> str:
        if not self._is_overview_task(task):
            return ""

        candidates: list[str] = list(_OVERVIEW_FILES)
        candidates.extend(
            path.name
            for path in sorted(self.workspace.glob("*.py"))
            if path.is_file() and path.name not in candidates
        )

        sections: list[str] = []
        for relative_path in candidates[:8]:
            path = self.policy.resolve_path(relative_path)
            if not path.is_file():
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = path.read_text(encoding="utf-8", errors="replace")
            sections.append(
                "\n".join(
                    [
                        f"## {relative_path}",
                        compress_text(content, 1800),
                    ]
                )
            )
        return "\n\n".join(sections)

    def _is_overview_task(self, task: str) -> bool:
        lowered = task.lower()
        return any(marker in lowered or marker in task for marker in _OVERVIEW_TASK_MARKERS)

    def _project_overview_summary(self, task: str) -> str:
        if not self._is_overview_task(task):
            return ""

        readme = self._read_first_existing_text(("README.zh-CN.md", "README.md"))
        title = self._extract_markdown_title(readme) or self.workspace.name
        description = self._extract_first_paragraph(readme) or "No README description found."
        top_level_dirs = sorted(
            path.name for path in self.workspace.iterdir() if path.is_dir() and not path.name.startswith(".")
        )
        entrypoints = sorted(path.name for path in self.workspace.glob("*.py") if path.is_file())
        dependency_files = [name for name in _OVERVIEW_FILES if (self.workspace / name).is_file()]

        lines = [
            f"- Project: {title}",
            f"- Description: {compact_text(description, 500)}",
            f"- Entry points: {', '.join(entrypoints) if entrypoints else 'none detected'}",
            f"- Top-level directories: {', '.join(top_level_dirs[:12]) if top_level_dirs else 'none detected'}",
            f"- Dependency/config files: {', '.join(dependency_files) if dependency_files else 'none detected'}",
            "- This summary is generated in ask mode from read-only README, config, and file-tree context.",
        ]
        return "\n".join(lines)

    def _read_first_existing_text(self, relative_paths: tuple[str, ...]) -> str:
        for relative_path in relative_paths:
            path = self.policy.resolve_path(relative_path)
            if not path.is_file():
                continue
            try:
                return path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return path.read_text(encoding="utf-8", errors="replace")
        return ""

    @staticmethod
    def _extract_markdown_title(content: str) -> str:
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
        return ""

    @staticmethod
    def _extract_first_paragraph(content: str) -> str:
        lines: list[str] = []
        in_heading = True
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                if lines:
                    break
                continue
            if in_heading and stripped.startswith("#"):
                continue
            in_heading = False
            if stripped.startswith("```"):
                break
            lines.append(stripped)
        return " ".join(lines)

    def _build_deterministic_plan_details(self, task: str, files: list[str]) -> CodingPlan:
        lowered = task.lower()
        files_set = set(files)
        selected: list[str] = []
        edit_steps: list[str] = []
        validation_steps = [
            "Run the narrowest relevant test file first.",
            "Run python -m pytest tests if tests are available.",
            "Run python -m ruff check . for style-sensitive Python edits.",
        ]
        risks = ["Plan mode has not read every candidate file and has not executed edits."]

        def add_file(relative_path: str) -> None:
            if relative_path in files_set and relative_path not in selected:
                selected.append(relative_path)

        wants_tool = any(marker in lowered or marker in task for marker in ("tool", "工具"))
        wants_cli = "cli" in lowered or "命令" in task
        wants_test = "test" in lowered or "pytest" in lowered or "测试" in task
        wants_docs = "readme" in lowered or "文档" in task or "说明" in task

        if wants_tool:
            for candidate in (
                "graph_rag_app/web_tools.py",
                "graph_rag_app/scholar_tools.py",
                "graph_rag_app/tools.py",
                "aicoding_app/tools.py",
            ):
                add_file(candidate)
            for candidate in files:
                if candidate.endswith("tools.py"):
                    add_file(candidate)
            edit_steps.extend(
                [
                    "Inspect the nearest existing tools module and follow its naming, return-value, and permission patterns.",
                    "Add the smallest standalone helper function in the selected tools module.",
                    "Export or wire the helper only if the existing module requires explicit registration.",
                ]
            )

        if wants_cli:
            for candidate in ("graph_rag_app/cli.py", "graph_rag.py", "aicoding_app/cli.py"):
                add_file(candidate)
            edit_steps.append("If the helper needs a user-facing command, add a focused CLI subcommand.")

        if wants_test or wants_tool:
            for candidate in files:
                if candidate.startswith("tests/") and (
                    "tool" in candidate.lower() or "cli" in candidate.lower()
                ):
                    add_file(candidate)
            if "tests" in {Path(file).parts[0] for file in files if Path(file).parts}:
                edit_steps.append("Add or update a focused test under tests/ for the new helper behavior.")

        if wants_docs:
            add_file("README.zh-CN.md")
            add_file("README.md")
            edit_steps.append("Update README only if the helper becomes user-facing.")

        if not selected:
            selected = files[:20]
            edit_steps.extend(
                [
                    "Read the most relevant source files from the repository map.",
                    "Choose a minimal implementation location before editing.",
                ]
            )

        assumptions = [
            "Plan mode may update session plan state, but it must not modify repository files.",
            "The exact implementation file should be confirmed by reading the selected candidates before edit mode.",
        ]
        if wants_tool:
            assumptions.append("A simple tool/helper should live near existing tool modules, not in the CLI entrypoint.")

        if not edit_steps:
            edit_steps = [
                "Inspect selected files.",
                "Prepare a focused patch only after switching to edit or agent mode.",
            ]

        return CodingPlan(
            goal=task,
            assumptions=assumptions,
            files_to_check=selected[:20],
            edit_steps=edit_steps,
            validation_steps=validation_steps,
            risks=risks,
        )

    def _trace_state(self, task_id: str, state: str, output: str) -> None:
        _progress(f"Task {task_id} state: {state} - {output}")
        self.trace_writer.append(
            "state_transition",
            task_id=task_id,
            input_summary=state,
            output_summary=output,
            payload={"state": state},
        )

    def _run_ask_task(self, task: str, task_id: str) -> str:
        _progress(f"Gathering ask-mode context for {task_id}")
        tools, plan, evidence_cache = self._new_tools(task_id)
        self._trace_state(task_id, "intake", "ask mode accepted")
        files = tools.list_files()
        repo_map = tools.repo_map()
        explained = tools.explain_context(task)
        overview_summary = self._project_overview_summary(task)
        overview = self._overview_context(task)
        instructions = self.project_instructions.as_context()
        memory = self.memory_store.format()
        self._trace_state(task_id, "explore", "read-only context gathered")
        self._persist_harness_state(plan, evidence_cache, tools, capture_diff=False, context_query=task)
        context = "\n\n".join(
            [
                f"Project instructions:\n{instructions}",
                f"Memory summary:\n{memory}",
                f"Relevant repository context:\n{compress_text(explained, 1800)}",
                "Project overview summary:\n"
                f"{compact_text(overview_summary, 1200) or 'No overview summary was generated.'}",
                f"Project overview context:\n{compress_text(overview, 3600) or 'No overview files were selected.'}",
                f"Repository map:\n{compress_text(repo_map, 1200)}",
                f"Files:\n{compact_text(files, 1200) or 'none'}",
            ]
        )
        model_response = self._call_read_only_model(
            mode="ask",
            task=task,
            task_id=task_id,
            context=context,
        )
        if model_response:
            _progress(f"Ask-mode task {task_id} complete")
            return "\n".join(
                [
                    "Ask mode: model-generated read-only analysis",
                    "",
                    model_response,
                    "",
                    "No files were modified and no commands were run.",
                ]
            )

        _progress(f"Ask-mode task {task_id} using fallback response")
        return "\n".join(
            [
                "Ask mode: read-only analysis",
                "Model fallback: "
                f"{self._last_read_only_model_error or 'model call failed without details'}.",
                "",
                context,
                "",
                "No files were modified and no commands were run.",
                f"Task: {task}",
            ]
        )

    def _run_plan_task(self, task: str, task_id: str) -> str:
        _progress(f"Gathering plan-mode context for {task_id}")
        tools, plan, evidence_cache = self._new_tools(task_id)
        self._trace_state(task_id, "intake", "plan mode accepted")
        files = tools.list_files()
        repo_map = tools.repo_map()
        explained = tools.explain_context(task)
        memory = self.memory_store.format()
        file_list = [line.strip() for line in files.splitlines() if line.strip()]
        planned = self._build_deterministic_plan_details(task, file_list)
        self._trace_state(task_id, "explore", "planning context gathered")
        tools.plan_update(
            goal=planned.goal,
            assumptions=planned.assumptions,
            files_to_check=planned.files_to_check,
            edit_steps=planned.edit_steps,
            validation_steps=planned.validation_steps,
            risks=planned.risks,
        )
        self._trace_state(task_id, "plan", "structured plan recorded")
        self._persist_harness_state(plan, evidence_cache, tools, capture_diff=False, context_query=task)
        context = "\n\n".join(
            [
                f"Memory summary:\n{memory}",
                f"Draft structured plan:\n{format_plan(plan)}",
                f"Repository map summary:\n{compress_text(repo_map, 1600)}",
                f"Query-specific context:\n{compress_text(explained, 1200)}",
            ]
        )
        model_response = self._call_read_only_model(
            mode="plan",
            task=task,
            task_id=task_id,
            context=context,
        )
        if model_response:
            _progress(f"Plan-mode task {task_id} complete")
            return "\n".join(
                [
                    "Plan mode: model-generated plan; no repository files modified",
                    "",
                    model_response,
                ]
            )

        _progress(f"Plan-mode task {task_id} using fallback response")
        return "\n".join(
            [
                "Plan mode: no repository files modified",
                "Model fallback: "
                f"{self._last_read_only_model_error or 'model call failed without details'}.",
                "",
                context,
            ]
        )
