from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shlex
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid
from typing import Any, cast

from .config import AppConfig, PROJECT_ROOT
from .context_explain import explain_context
from .context import build_context
from .evidence_cache import EvidenceCache, compact_text
from .harness_state import ToolRunState
from .memory import MemoryStore
from .permissions import PermissionDenied, WorkspacePolicy
from .plan import CodingPlan, format_plan
from .project_instructions import load_project_instructions
from .repo_map import build_repo_map
from .session_store import SessionStore, safe_session_id
from .skills import SkillRegistry
from .tools import CodingTools
from .trace import StructuredTraceWriter

try:
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    from langchain_openai import ChatOpenAI

    CHAT_MODEL_AVAILABLE = True
    CHAT_MODEL_IMPORT_ERROR = ""
except ImportError as exc:  # pragma: no cover
    HumanMessage = None
    SystemMessage = None
    ToolMessage = None
    ChatOpenAI = None
    CHAT_MODEL_AVAILABLE = False
    CHAT_MODEL_IMPORT_ERROR = str(exc)

try:
    from langchain_core.tools import StructuredTool
    from langgraph.graph import END, MessagesState, StateGraph

    LANGGRAPH_AVAILABLE = CHAT_MODEL_AVAILABLE
    LANGGRAPH_IMPORT_ERROR = ""
except ImportError as exc:  # pragma: no cover
    StructuredTool = None
    MessagesState = None
    StateGraph = None
    END = None
    LANGGRAPH_AVAILABLE = False
    LANGGRAPH_IMPORT_ERROR = str(exc)


PROMPT = """\
You are an AI coding agent operating inside one local workspace.

Use tools to inspect files before editing. Before any apply_patch call, call
plan_update with a concrete goal, edit steps, validation steps, and risks.
Before apply_patch, call preview_patch with the exact same patch and only apply
after the preview succeeds.
For long complete files such as README files or large test modules, prefer
write_text_file(path, content). It is a harness-controlled, traceable text file
write tool and must also be preceded by plan_update.
When adding documentation for a specific file, module, or feature, inspect the
target documentation first and verify it covers the same topic. If an existing
README clearly documents a different project, module, or domain, create a
dedicated nearby document instead of appending unrelated content.
If validation reports that Python, pytest, ruff, or pyright failed because the
validation environment itself is broken, treat it as an environment/tooling
blocker rather than an ordinary code failure. Do not retry the same failing validation command.
Call check_environment once to diagnose fixed Python validation tooling.
Do not automatically use install_python_package unless the task or configuration
explicitly asks you to repair the environment. Do not use
python -c, pip, or shell control operators to work around the command whitelist.
Do not call check_environment repeatedly. After one diagnosis, continue with
other allowed smoke tests, ruff, pyright, or report the
environment limitation clearly.
Never access paths outside the workspace. Only run commands through run_command.
Commands already run with the workspace as current directory. Do not prefix
commands with directory changes such as `cd`, `cd ..`, or `cd <path>`. Do not
combine commands with `&&`, `||`, `;`, or pipes. If a command is denied, simplify
to one bare allowed command such as `python script.py ...`, `python -m pytest ...`,
`python -m ruff check ...`, `python -m pyright ...`, `git status`, or `git diff`.
After edits, inspect git diff and run relevant allowed validation commands.
If a tool returns command_denied or validation_denied, do not retry the same
command. Use another allowed validation command or finish with the denial
listed as a validation limitation.
The apply_patch tool input must be an exact patch block starting with
`*** Begin Patch` and ending with `*** End Patch`.
Default to ASCII when creating or editing source files unless non-ASCII content is
explicitly required by the task or already established in the target file.

Final answers must include:
- Change summary
- Modified files
- Validation commands and results
- Remaining risks or follow-up work
Clearly distinguish code failures from environment/tooling blockers.
"""


AgentState = dict[str, Any]

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


@dataclass(frozen=True)
class TaskResult:
    session_id: str
    task_id: str
    response: str


@dataclass(frozen=True)
class CompletionStatus:
    status: str
    completed_items: list[str]
    missing_items: list[str]
    smoke_commands: list[str]
    blocked_items: list[str]
    skipped_commands: list[str]


class HarnessStop(RuntimeError):
    def __init__(self, message: str, run_state: ToolRunState):
        super().__init__(message)
        self.run_state = run_state


class CodingAgent:
    def __init__(self, *, config: AppConfig, workspace: str | Path, session_id: str):
        workspace_path = Path(workspace).resolve()
        self.config = config
        self.workspace = workspace_path
        self.session_id = safe_session_id(session_id)
        self.session_store = SessionStore(config.harness.runtime_dir)
        self.session = self.session_store.get_or_create(self.session_id, str(workspace_path))
        self.policy = WorkspacePolicy(
            workspace=workspace_path,
            allowed_commands=config.harness.allowed_commands,
        )
        self.trace_writer = StructuredTraceWriter(
            config.harness.runtime_dir / "traces",
            self.session_id,
            enabled=config.harness.trace_enabled,
        )
        self.skill_registry = SkillRegistry(PROJECT_ROOT / "skills")
        self.project_instructions = load_project_instructions(workspace_path)
        self.memory_store = MemoryStore(config.harness.runtime_dir)
        self._last_read_only_model_error = ""

    def _new_tools(
        self,
        task_id: str,
        *,
        run_state: ToolRunState | None = None,
        require_patch_preview: bool = False,
    ) -> tuple[CodingTools, CodingPlan, EvidenceCache]:
        plan = CodingPlan.from_jsonable(self.session.plan)
        evidence_cache = EvidenceCache.from_jsonable(self.session.evidence)
        tools = CodingTools(
            policy=self.policy,
            plan=plan,
            evidence_cache=evidence_cache,
            skill_registry=self.skill_registry,
            trace_writer=self.trace_writer,
            command_timeout_seconds=self.config.harness.command_timeout_seconds,
            task_id=task_id,
            run_state=run_state,
            require_patch_preview=require_patch_preview,
        )
        return tools, plan, evidence_cache

    def run_task(self, task: str) -> TaskResult:
        return self.run_mode_task("run", task)

    def run_mode_task(self, mode: str, task: str) -> TaskResult:
        task_id = uuid.uuid4().hex[:12]
        self.trace_writer.append(
            "task_started",
            task_id=task_id,
            input_summary=compact_text(task, 400),
            output_summary=f"{mode} task accepted",
            payload={"mode": mode},
        )
        self.session.add_message("user", task)

        normalized_mode = mode.lower()
        if normalized_mode == "ask":
            response = self._run_ask_task(task, task_id)
        elif normalized_mode == "plan":
            response = self._run_plan_task(task, task_id)
        elif normalized_mode == "run":
            response = self._run_model_edit_or_fallback(task, task_id, mode="run")
        elif normalized_mode == "edit":
            response = self._run_model_edit_or_fallback(task, task_id, mode="edit")
        elif normalized_mode == "agent":
            response = self._run_model_edit_or_fallback(task, task_id, mode="agent")
        elif self.config.model.configured and LANGGRAPH_AVAILABLE:
            run_state = ToolRunState()
            try:
                response = self._run_langgraph_task(task, task_id, run_state=run_state)
            except HarnessStop as exc:
                response = self._handle_harness_stop(task, task_id, exc)
            except Exception as exc:  # pragma: no cover - external model fallback
                fallback_reason = str(exc)
                run_state.reset_problem_counters_for_fallback()
                self.trace_writer.append(
                    "model_fallback",
                    task_id=task_id,
                    input_summary="LangGraph execution failed",
                    output_summary=fallback_reason,
                    status="failed",
                )
                response = self._append_harness_correction(
                    "\n".join(
                        [
                            "Model execution failed; direct patch edit was attempted before deterministic fallback.",
                            f"Model failure: {fallback_reason}",
                            "",
                            self._run_direct_patch_edit(
                                task,
                                task_id,
                                mode=normalized_mode,
                                run_state=run_state,
                            )
                            or self._run_deterministic_task(task, task_id, run_state=run_state),
                        ]
                    ),
                    run_state,
                )
        else:
            run_state = ToolRunState()
            response = self._append_harness_correction(
                self._run_deterministic_task(task, task_id, run_state=run_state),
                run_state,
            )

        self.session.add_message("assistant", response)
        self.session_store.save(self.session)
        self.trace_writer.append(
            "final_response",
            task_id=task_id,
            input_summary="response",
            output_summary=self._final_response_trace_summary(response),
        )
        return TaskResult(session_id=self.session_id, task_id=task_id, response=response)

    def _run_model_edit_or_fallback(self, task: str, task_id: str, *, mode: str) -> str:
        completion = self._check_task_completion(task, task_id)
        if completion.status in {"completed", "completed_with_environment_blocker"}:
            return self._format_completion_summary(completion)

        if self.config.model.configured and LANGGRAPH_AVAILABLE:
            before_files = self._workspace_file_set()
            run_state = ToolRunState()
            try:
                return self._run_langgraph_task(task, task_id, run_state=run_state)
            except HarnessStop as exc:
                return self._handle_harness_stop(task, task_id, exc)
            except Exception as exc:  # pragma: no cover - external model fallback
                reason = str(exc)
                run_state.reset_problem_counters_for_fallback()
                self.trace_writer.append(
                    "model_fallback",
                    task_id=task_id,
                    input_summary=f"{mode} LangGraph execution failed",
                    output_summary=reason,
                    status="failed",
                )
                partial_response = self._summarize_partial_model_changes(
                    task=task,
                    task_id=task_id,
                    before_files=before_files,
                    failure_reason=reason,
                    run_state=run_state,
                )
                if partial_response:
                    return self._append_harness_correction(partial_response, run_state)
                direct_response = self._run_direct_patch_edit(
                    task,
                    task_id,
                    mode=mode,
                    run_state=run_state,
                )
                if direct_response:
                    return self._append_harness_correction(
                        "\n".join(
                            [
                                "LangGraph model execution failed; direct patch edit was used.",
                                f"LangGraph failure: {reason}",
                                "",
                                direct_response,
                            ]
                        ),
                        run_state,
                    )
                return self._append_harness_correction(
                    "\n".join(
                        [
                            "Model execution failed; deterministic fallback was used.",
                            f"Model failure: {reason}",
                            "",
                            self._run_deterministic_task(
                                task,
                                task_id,
                                include_pr_summary=True,
                                validation_command=self._validation_command_for_task(task)
                                if mode == "agent"
                                else None,
                                auto_repair=mode == "agent",
                                run_state=run_state,
                            ),
                        ]
                    ),
                    run_state,
                )

        if self.config.model.configured and not LANGGRAPH_AVAILABLE:
            detail = LANGGRAPH_IMPORT_ERROR or CHAT_MODEL_IMPORT_ERROR or "unknown import error"
            self.trace_writer.append(
                "model_fallback",
                task_id=task_id,
                input_summary=f"{mode} LangGraph runtime unavailable",
                output_summary=detail,
                status="failed",
            )
            run_state = ToolRunState()
            direct_response = self._run_direct_patch_edit(
                task,
                task_id,
                mode=mode,
                run_state=run_state,
            )
            if direct_response:
                return self._append_harness_correction(
                    "\n".join(
                        [
                            "LangGraph tool runtime is unavailable; direct patch edit was used.",
                            f"LangGraph import failure: {detail}",
                            "",
                            direct_response,
                        ]
                    ),
                    run_state,
                )
            return self._append_harness_correction(
                "\n".join(
                    [
                        "Model execution unavailable; deterministic fallback was used.",
                        f"Model failure: LangGraph tool runtime is unavailable: {detail}",
                        "Direct patch edit also failed; install edit/agent dependencies, or use run with explicit create/set/append/replace syntax.",
                        "",
                        self._run_deterministic_task(
                            task,
                            task_id,
                            include_pr_summary=True,
                            validation_command=self._validation_command_for_task(task)
                            if mode == "agent"
                            else None,
                            auto_repair=mode == "agent",
                            run_state=run_state,
                        ),
                    ]
                ),
                run_state,
            )

        return self._run_deterministic_task(
            task,
            task_id,
            include_pr_summary=True,
            validation_command=self._validation_command_for_task(task) if mode == "agent" else None,
            auto_repair=mode == "agent",
        )

    def _handle_harness_stop(self, task: str, task_id: str, exc: HarnessStop) -> str:
        return self._build_partial_summary(
            task,
            task_id,
            exc.run_state,
            failure_reason=str(exc),
        )

    def _check_task_completion(self, task: str, task_id: str) -> CompletionStatus:
        required_files = self._required_files_for_task(task)
        smoke_commands = self._smoke_commands_for_task(task)
        if not required_files and not smoke_commands:
            return CompletionStatus("unknown", [], [], [], [], [])
        if required_files and not self._completion_file_check_requested(task):
            return CompletionStatus("unknown", [], [], [], [], [])

        completed: list[str] = []
        missing: list[str] = []
        blocked: list[str] = []
        skipped: list[str] = []
        for path in required_files:
            if (self.workspace / path).exists():
                completed.append(f"Required file exists: {path}")
            else:
                missing.append(f"Required file missing: {path}")

        if missing:
            return CompletionStatus("incomplete", completed, missing, [], blocked, skipped)
        if self._explicit_validation_commands(task):
            skipped.append("Explicit validation command present; completion check will not short-circuit.")
            return CompletionStatus("unknown", completed, [], smoke_commands, blocked, skipped)

        run_state = ToolRunState()
        tools, plan, evidence_cache = self._new_tools(task_id, run_state=run_state)
        for command in smoke_commands:
            if self._should_skip_idempotent_smoke_command(command):
                skipped.append(f"Skipped idempotent side-effect command already reflected in data: {command}")
                continue
            if not self._is_safe_completion_smoke_command(command):
                skipped.append(f"Skipped unsafe or mutating smoke command during completion check: {command}")
                return CompletionStatus("unknown", completed, [], smoke_commands, blocked, skipped)
            output = tools.run_command(command)
            if "returncode: 0" in output:
                completed.append(f"Smoke command passed: {command}")
            elif "Validation environment failure detected." in output:
                blocked.append(f"Smoke command blocked by environment: {command}")
            elif output.startswith("command_denied:"):
                blocked.append(f"Smoke command denied by policy: {command}: {output}")
            else:
                missing.append(f"Smoke command failed: {command}")

        self._persist_harness_state(plan, evidence_cache, tools, context_query=task)
        if missing:
            status = "incomplete"
        elif blocked:
            status = "completed_with_environment_blocker"
        else:
            status = "completed"
        return CompletionStatus(status, completed, missing, smoke_commands, blocked, skipped)

    def _completion_file_check_requested(self, task: str) -> bool:
        lowered = task.lower()
        positive_markers = (
            "already",
            "check",
            "complete",
            "ensure",
            "ready",
            "smoke",
            "validate",
            "verify",
            "确认",
            "完成",
            "就绪",
            "检查",
            "验证",
        )
        mutation_markers = (
            "append",
            "change",
            "create",
            "edit",
            "fix",
            "implement",
            "modify",
            "replace",
            "update",
            "创建",
            "修复",
            "实现",
            "替换",
            "更新",
            "添加",
            "编辑",
            "修改",
        )
        if any(marker in lowered for marker in mutation_markers):
            return False
        return any(marker in lowered for marker in positive_markers)

    def _format_completion_summary(self, completion: CompletionStatus) -> str:
        lines = [
            "Task status:",
            f"- {completion.status}",
            "",
            "Completed:",
        ]
        lines.extend(f"- {item}" for item in completion.completed_items)
        if completion.skipped_commands:
            lines.extend(f"- {item}" for item in completion.skipped_commands)
        if not completion.completed_items and not completion.skipped_commands:
            lines.append("- Existing workspace state satisfies the detectable task requirements.")
        lines.extend(["", "Smoke commands:"])
        if completion.smoke_commands:
            lines.extend(f"- {command}" for command in completion.smoke_commands)
        else:
            lines.append("- No smoke commands were detected.")
        lines.extend(["", "Blocked:"])
        if completion.blocked_items:
            lines.extend(f"- {item}" for item in completion.blocked_items)
        else:
            lines.append("- None.")
        lines.extend(
            [
                "",
                "Next action:",
                "- No coding fallback was run because the detectable task requirements are already satisfied.",
            ]
        )
        return "\n".join(lines)

    def _required_files_for_task(self, task: str) -> list[str]:
        files: list[str] = []
        for match in re.finditer(r"(?<![\w./\\-])([A-Za-z0-9_./\\-]+\.(?:py|md|json|toml|yaml|yml|txt|csv))", task):
            if match.start() > 0 and task[match.start() - 1] in {"/", "\\"}:
                continue
            raw_path = match.group(1).replace("\\", "/")
            if any(part == ".." for part in raw_path.split("/")):
                continue
            path = raw_path.strip("./")
            if not path or path in files:
                continue
            try:
                relative = self.policy.relative_path(path)
            except PermissionDenied:
                continue
            if relative not in files:
                files.append(relative)
        return files

    def _smoke_commands_for_task(self, task: str) -> list[str]:
        commands: list[str] = []
        unsafe_fragments = ("python -c", "python -m pip", " pip ", "&&", "||", ";", "|")
        validation_fragments = ("pytest", "ruff", "pyright")
        for raw_line in task.splitlines():
            line = raw_line.strip().strip("`")
            line = re.sub(r"^[-*]\s+", "", line).strip().strip("`")
            match = re.match(r"^(?:.*?:\s*)?(python\s+.+)$", line, flags=re.IGNORECASE)
            if not match:
                continue
            command = match.group(1).strip().strip("`")
            command_lower = command.lower()
            if any(fragment in command_lower for fragment in unsafe_fragments):
                continue
            if any(fragment in command_lower for fragment in validation_fragments):
                continue
            parts = self._split_command_for_inspection(command)
            if len(parts) < 2:
                continue
            script = parts[1]
            if script.startswith("-") or not script.endswith(".py"):
                continue
            if command not in commands:
                commands.append(command)
        return commands

    def _split_command_for_inspection(self, command: str) -> list[str]:
        def clean_native_token(token: str) -> str:
            cleaned = token.strip("\"'")
            if "=" in cleaned:
                key, value = cleaned.split("=", 1)
                stripped_value = value.strip("\"'")
                return f"{key}={stripped_value}"
            return cleaned

        try:
            parts = shlex.split(command)
            if "\\" not in command:
                return parts
            native_parts = shlex.split(command, posix=False)
            if len(native_parts) == len(parts):
                for index, native_part in enumerate(native_parts):
                    parts[index] = clean_native_token(native_part)
                if len(parts) >= 2:
                    parts[1] = parts[1].replace("\\", "/")
            elif len(parts) >= 2 and len(native_parts) >= 2:
                parts[0] = clean_native_token(native_parts[0])
                parts[1] = clean_native_token(native_parts[1]).replace("\\", "/")
            return parts
        except ValueError:
            return []

    def _is_safe_completion_smoke_command(self, command: str) -> bool:
        parts = self._split_command_for_inspection(command)
        if len(parts) < 2:
            return False
        executable = Path(parts[0]).name.lower()
        if executable not in {"python", "python.exe"}:
            return False
        script = Path(parts[1]).name.lower()
        if len(parts) > 2:
            return False
        return script.startswith("test_") and script.endswith(".py")

    def _should_skip_idempotent_smoke_command(self, command: str) -> bool:
        return False

    def _build_partial_summary(
        self,
        task: str,
        task_id: str,
        run_state: ToolRunState,
        *,
        failure_reason: str = "",
    ) -> str:
        del task_id
        status_by_reason = {
            "environment_blocker_detected": "completed_with_environment_blocker",
            "policy_denial_loop": "stopped_by_policy",
            "tool_misuse_loop": "stopped_by_tool_misuse",
            "too_many_dependency_violations": "stopped_by_harness",
            "too_many_consecutive_tool_failures": "stopped_by_harness",
        }
        task_status = status_by_reason.get(run_state.stop_reason, "stopped_by_harness")
        passed_validations = [
            attempt for attempt in run_state.validation_attempts if attempt.status == "passed"
        ]
        failed_validations = [
            attempt for attempt in run_state.validation_attempts if attempt.status == "failed"
        ]
        denied_validations = [
            attempt for attempt in run_state.validation_attempts if attempt.status == "denied"
        ]

        completed: list[str] = []
        completed.extend(f"Modified {path} (confirmed by harness state)." for path in run_state.written_files)
        completed.extend(
            f"Validation passed: {attempt.command}" for attempt in passed_validations
        )
        if not completed:
            completed.append("No completed code changes could be confirmed from harness state.")

        modified_files = run_state.written_files or ["No modified files were recorded by harness state."]
        validation_lines: list[str] = []
        validation_lines.extend(f"Passed: {attempt.command}" for attempt in passed_validations)
        validation_lines.extend(f"Failed: {attempt.command}" for attempt in failed_validations)
        validation_lines.extend(f"Denied: {attempt.command}" for attempt in denied_validations)
        if not validation_lines:
            validation_lines.append("No validation attempts were recorded by harness state.")

        blocked: list[str] = []
        blocked.extend(
            f"Environment blocker: {attempt.command}: {attempt.detail}"
            for attempt in run_state.environment_failures
        )
        blocked.extend(
            f"Policy denial: {attempt.command}: {attempt.detail}"
            for attempt in run_state.policy_denials
        )
        blocked.extend(
            f"Tool misuse: {violation.tool_name}: {violation.detail}"
            for violation in run_state.tool_misuses
        )
        blocked.extend(
            f"Dependency violation: {violation.tool_name}: {violation.detail}"
            for violation in run_state.dependency_violations
        )
        if not blocked:
            blocked.append(f"Harness stopped: {run_state.stop_reason or failure_reason or 'unknown reason'}")

        rerun_command = self._validation_command_for_task(task)
        if run_state.stop_reason == "environment_blocker_detected":
            next_action = "Repair local validation environment"
            if rerun_command:
                next_action = f"{next_action}, then rerun {rerun_command}."
            else:
                next_action = f"{next_action}, then rerun focused validation."
        elif run_state.stop_reason == "policy_denial_loop":
            next_action = "Use an allowed validation command or extend the whitelist deliberately."
        elif run_state.stop_reason == "tool_misuse_loop":
            next_action = "Retry with available tools only."
        else:
            next_action = "Inspect harness correction and rerun focused validation."

        return "\n".join(
            [
                "Task status:",
                f"- {task_status}",
                "",
                "Completed:",
                *[f"- {item}" for item in completed],
                "",
                "Modified files:",
                *[f"- {path}" for path in modified_files],
                "",
                "Validation:",
                *[f"- {item}" for item in validation_lines],
                "",
                "Blocked:",
                *[f"- {item}" for item in blocked],
                "",
                "Next action:",
                f"- {next_action}",
                "",
                "Harness stop:",
                f"- {failure_reason or run_state.stop_reason or 'unknown'}",
            ]
        )

    def _workspace_file_set(self) -> set[str]:
        files: set[str] = set()
        for path in self.workspace.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(self.workspace)
            if any(part in {".git", ".ruff_cache", ".pytest_cache", "__pycache__"} for part in relative.parts):
                continue
            files.add(relative.as_posix())
        return files

    def _summarize_partial_model_changes(
        self,
        *,
        task: str,
        task_id: str,
        before_files: set[str],
        failure_reason: str,
        run_state: ToolRunState | None = None,
    ) -> str | None:
        after_files = self._workspace_file_set()
        created_files = sorted(after_files - before_files)
        if not created_files:
            return None
        tools, plan, evidence_cache = self._new_tools(task_id, run_state=run_state)
        tools.plan_update(
            goal=task,
            assumptions=["The model created files before failing to produce a final response."],
            files_to_check=created_files,
            edit_steps=["Review files created before the model failure."],
            validation_steps=["Run the smallest relevant validation command."],
            risks=["The model did not complete its final summarization step."],
        )
        validation_result = tools.run_validation(self._validation_command_for_task(task) or "git diff -- .")
        pr_summary = tools.pr_summary(
            change_summary="model created files before final response failure",
            validation_result=validation_result,
            risks=["Review created files because the model hit an execution failure after editing."],
        )
        self._persist_harness_state(plan, evidence_cache, tools, context_query=task)
        return "\n".join(
            [
                "Model execution failed after applying changes.",
                f"Model failure: {failure_reason}",
                "",
                "Created files:",
                "\n".join(f"- {path}" for path in created_files),
                "",
                "Validation commands and results:",
                compact_text(validation_result, 1600),
                "",
                pr_summary,
            ]
        )

    def _run_direct_patch_edit(
        self,
        task: str,
        task_id: str,
        *,
        mode: str,
        run_state: ToolRunState | None = None,
    ) -> str | None:
        tools, plan, evidence_cache = self._new_tools(task_id, run_state=run_state)
        files = tools.list_files()
        repo_map = tools.repo_map()
        instructions = self.project_instructions.as_context()
        memory = self.memory_store.format()
        context = "\n\n".join(
            [
                f"Project instructions:\n{instructions}",
                f"Memory summary:\n{memory}",
                f"Repository map:\n{compact_text(repo_map, 3000)}",
                f"Files:\n{compact_text(files, 2000) or 'none'}",
            ]
        )
        patch = self._call_direct_patch_model(task=task, task_id=task_id, context=context)
        if not patch:
            self._persist_harness_state(plan, evidence_cache, tools, context_query=task)
            return None

        tools.plan_update(
            goal=task,
            assumptions=[
                "LangGraph tool runtime was unavailable, so a direct model-generated patch was used.",
                "The patch was validated by the local patch preview before applying.",
            ],
            files_to_check=files.splitlines()[:20],
            edit_steps=["Generate one exact patch block.", "Preview the patch.", "Apply the patch."],
            validation_steps=["Inspect changed files after applying the patch."],
            risks=["Direct patch edit cannot iterate with tools if the first generated patch is wrong."],
        )
        preview = tools.preview_patch(patch)
        if preview.startswith("patch_preview_failed"):
            self._persist_harness_state(plan, evidence_cache, tools, context_query=task)
            self.trace_writer.append(
                "model_fallback",
                task_id=task_id,
                input_summary="direct patch preview failed",
                output_summary=compact_text(preview, 600),
                status="failed",
            )
            return None
        apply_result = tools.apply_patch(patch)
        validation_result = tools.run_validation(self._validation_command_for_task(task) or "git diff -- .")
        pr_summary = tools.pr_summary(
            change_summary=apply_result,
            validation_result=validation_result,
            risks=["Review the diff before committing; no commit, push, or PR was created."],
        )
        self._persist_harness_state(plan, evidence_cache, tools, context_query=task)
        return "\n".join(
            [
                "Change summary:",
                f"- {apply_result}",
                "",
                "Patch preview:",
                f"- {preview}",
                "",
                "Validation commands and results:",
                compact_text(validation_result, 1600),
                "",
                pr_summary,
            ]
        )

    def _call_direct_patch_model(self, *, task: str, task_id: str, context: str) -> str | None:
        endpoint = self.config.model.api_base.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.model.model_name,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": "\n".join(
                        [
                            "You are a coding patch generator.",
                            "Return exactly one patch block and no prose.",
                            "The patch must start with *** Begin Patch and end with *** End Patch.",
                            "Put *** Begin Patch, file section headers, file content, and *** End Patch on separate lines.",
                            "Use only *** Add File:, *** Update File:, or *** Delete File: sections supported by the harness.",
                            "All paths must be relative to the workspace.",
                            "Do not wrap file content in Markdown code fences.",
                            "Use ASCII-only source text unless the task explicitly requires non-ASCII.",
                        ]
                    ),
                },
                {"role": "system", "content": f"Repository context:\n{context}"},
                {"role": "user", "content": task},
            ],
        }
        request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.model.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        self.trace_writer.append(
            "model_call",
            task_id=task_id,
            input_summary="direct patch edit",
            output_summary="calling OpenAI-compatible chat for patch",
        )
        try:
            with urlopen(request, timeout=self.config.harness.command_timeout_seconds) as response:
                raw = response.read().decode("utf-8", errors="replace")
            data = json.loads(raw)
            content = str(data["choices"][0]["message"]["content"])
        except (HTTPError, URLError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            self.trace_writer.append(
                "model_fallback",
                task_id=task_id,
                input_summary="direct patch model failed",
                output_summary=str(exc),
                status="failed",
            )
            return None
        return self._extract_patch_block(content)

    @staticmethod
    def _extract_patch_block(content: str) -> str | None:
        fence_match = re.search(
            r"```(?:diff|patch)?\s*(\*\*\* Begin Patch.*?\*\*\* End Patch)\s*```",
            content,
            re.DOTALL,
        )
        if fence_match:
            return CodingAgent._normalize_patch_block(fence_match.group(1))
        patch_match = re.search(r"\*\*\* Begin Patch.*?\*\*\* End Patch", content, re.DOTALL)
        if patch_match:
            return CodingAgent._normalize_patch_block(patch_match.group(0))
        return None

    @staticmethod
    def _normalize_patch_block(patch: str) -> str:
        normalized = patch.strip()
        normalized = re.sub(
            r"^\*\*\* Begin Patch\s+(?=(?:\*\*\* )?(?:Add File|Update File|Delete File): )",
            "*** Begin Patch\n",
            normalized,
        )
        normalized = re.sub(r"\s+\*\*\* End Patch$", "\n*** End Patch", normalized)
        lines = normalized.splitlines()
        cleaned: list[str] = []
        section_mode = ""
        for line in lines:
            stripped = line.strip()
            if not section_mode and re.match(r"^(Add File|Update File|Delete File): ", stripped):
                stripped = f"*** {stripped}"
            if stripped.startswith("*** Add File: "):
                section_mode = "add"
                cleaned.append(stripped.split("```", 1)[0].rstrip())
                continue
            if stripped.startswith("*** Update File: ") or stripped.startswith("*** Delete File: "):
                section_mode = "update" if stripped.startswith("*** Update File: ") else "delete"
                cleaned.append(stripped)
                continue
            if stripped == "*** End Patch":
                section_mode = ""
                cleaned.append(stripped)
                continue
            # Strip model-added Markdown fences around generated Add File content.
            if section_mode == "add" and stripped.startswith("```"):
                continue
            cleaned.append(line)
        return "\n".join(cleaned).strip()

    def _persist_harness_state(
        self,
        plan: CodingPlan,
        evidence_cache: EvidenceCache,
        tools: CodingTools,
        *,
        capture_diff: bool = True,
        context_query: str = "",
    ) -> None:
        self.session.plan = plan.to_jsonable()
        self.session.evidence = evidence_cache.to_jsonable()
        diff = tools.git_diff() if capture_diff else self.session.latest_diff
        self.session.latest_diff = diff
        self.session.summary = compact_text(
            build_context(
                history=self.session.history,
                plan=plan,
                evidence_context=evidence_cache.to_context(3000),
                latest_diff=diff,
                max_chars=self.config.harness.max_context_chars,
                project_instructions=self.project_instructions.as_context(),
                repo_map=self._repo_context_summary(context_query),
                documentation_summary=self._documentation_context(),
                memory_summary=self.memory_store.format(),
            ),
            2000,
        )

    def _repo_context_summary(self, query: str) -> str:
        repo_summary = build_repo_map(self.workspace).format(max_files=40, max_symbols=60)
        if not query.strip():
            return repo_summary
        explained = explain_context(self.workspace, query)
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
                        compact_text(content, 1800),
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
            path.name
            for path in self.workspace.iterdir()
            if path.is_dir() and not path.name.startswith(".")
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

    def _call_read_only_model(
        self,
        *,
        mode: str,
        task: str,
        task_id: str,
        context: str,
    ) -> str | None:
        self._last_read_only_model_error = ""
        if not self.config.model.configured:
            self._last_read_only_model_error = "model is not configured"
            return None
        if not CHAT_MODEL_AVAILABLE or ChatOpenAI is None or SystemMessage is None or HumanMessage is None:
            return self._call_openai_compatible_chat_model(
                mode=mode,
                task=task,
                task_id=task_id,
                context=context,
            )

        chat_openai = cast(Any, ChatOpenAI)
        system_message = cast(Any, SystemMessage)
        human_message = cast(Any, HumanMessage)
        model_options: dict[str, Any] = {}
        if self.config.model.model_name.startswith("deepseek-v4"):
            model_options["extra_body"] = {"thinking": {"type": "disabled"}}

        model = chat_openai(
            api_key=self.config.model.api_key,
            base_url=self.config.model.api_base,
            model=self.config.model.model_name,
            temperature=0,
            **model_options,
        )
        mode_instruction = (
            "You are in ASK mode. Answer the user's question from the supplied repository "
            "context. Do not propose file modifications unless the user asks for suggestions."
            if mode == "ask"
            else "You are in PLAN mode. Produce a concrete implementation plan with files to "
            "inspect or edit, ordered steps, validation commands, and risks. Do not claim that "
            "files were modified."
        )
        self.trace_writer.append(
            "model_call",
            task_id=task_id,
            input_summary=f"{mode} read-only context",
            output_summary="calling chat model without tools",
        )
        try:
            response = model.invoke(
                [
                    system_message(
                        content="\n".join(
                            [
                                mode_instruction,
                                "You may only use the context already provided.",
                                "No tools are available in this mode.",
                                "Do not say that commands were run or files were modified.",
                            ]
                        )
                    ),
                    system_message(content=f"Repository context:\n{context}"),
                    human_message(content=task),
                ]
            )
        except Exception as exc:  # pragma: no cover - external model fallback
            self._last_read_only_model_error = str(exc)
            self.trace_writer.append(
                "model_fallback",
                task_id=task_id,
                input_summary=f"{mode} read-only model failed",
                output_summary=str(exc),
                status="failed",
            )
            return None
        return str(getattr(response, "content", response))

    def _call_openai_compatible_chat_model(
        self,
        *,
        mode: str,
        task: str,
        task_id: str,
        context: str,
    ) -> str | None:
        mode_instruction = (
            "You are in ASK mode. Answer the user's question from the supplied repository "
            "context. Do not propose file modifications unless the user asks for suggestions."
            if mode == "ask"
            else "You are in PLAN mode. Produce a concrete implementation plan with files to "
            "inspect or edit, ordered steps, validation commands, and risks. Do not claim that "
            "files were modified."
        )
        endpoint = self.config.model.api_base.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.model.model_name,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": "\n".join(
                        [
                            mode_instruction,
                            "You may only use the context already provided.",
                            "No tools are available in this mode.",
                            "Do not say that commands were run or files were modified.",
                        ]
                    ),
                },
                {"role": "system", "content": f"Repository context:\n{context}"},
                {"role": "user", "content": task},
            ],
        }
        request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.model.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        self.trace_writer.append(
            "model_call",
            task_id=task_id,
            input_summary=f"{mode} direct OpenAI-compatible chat",
            output_summary="calling chat model without LangChain",
        )
        try:
            with urlopen(request, timeout=self.config.harness.command_timeout_seconds) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:  # pragma: no cover - external model fallback
            body = exc.read().decode("utf-8", errors="replace")
            self._last_read_only_model_error = f"HTTP {exc.code}: {body}"
            self.trace_writer.append(
                "model_fallback",
                task_id=task_id,
                input_summary=f"{mode} direct model failed",
                output_summary=self._last_read_only_model_error,
                status="failed",
            )
            return None
        except URLError as exc:  # pragma: no cover - external model fallback
            self._last_read_only_model_error = f"network error: {exc}"
            self.trace_writer.append(
                "model_fallback",
                task_id=task_id,
                input_summary=f"{mode} direct model failed",
                output_summary=self._last_read_only_model_error,
                status="failed",
            )
            return None

        try:
            data = json.loads(raw)
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            self._last_read_only_model_error = f"unexpected chat response: {exc}"
            self.trace_writer.append(
                "model_fallback",
                task_id=task_id,
                input_summary=f"{mode} direct model response parse failed",
                output_summary=self._last_read_only_model_error,
                status="failed",
            )
            return None

    def _trace_state(self, task_id: str, state: str, output: str) -> None:
        self.trace_writer.append(
            "state_transition",
            task_id=task_id,
            input_summary=state,
            output_summary=output,
            payload={"state": state},
        )

    def _select_skill(self, task: str) -> str:
        lowered = task.lower()
        if "review" in lowered or "审查" in task:
            return "code-review"
        if "test" in lowered or "pytest" in lowered or "失败" in task:
            return "test-debugging"
        if "edit" in lowered or "replace" in lowered or "修改" in task:
            return "safe-editing"
        return "repo-understanding"

    def _run_deterministic_task(
        self,
        task: str,
        task_id: str,
        *,
        include_pr_summary: bool = False,
        validation_command: str | None = None,
        auto_repair: bool = False,
        run_state: ToolRunState | None = None,
    ) -> str:
        tools, plan, evidence_cache = self._new_tools(task_id, run_state=run_state)
        skill_name = self._select_skill(task)
        tools.load_skill(skill_name)
        files = tools.list_files()
        tools.explain_context(task)
        status = tools.git_status()

        patch = self._build_simple_patch(task)
        changed_files: list[str] = []
        patch_result = "no edit pattern matched"
        validation_result = ""

        tools.plan_update(
            goal=task,
            assumptions=[
                "No configured model was used for this turn; deterministic fallback handled the task.",
            ],
            files_to_check=files.splitlines()[:20],
            edit_steps=["Apply a targeted patch if the task matches a supported edit pattern."],
            validation_steps=["Inspect git diff after any edit."],
            risks=["Fallback supports create, append, set, and replace tasks only."],
        )

        if patch:
            patch_result = tools.apply_patch(patch)
            changed_files = self._extract_changed_files(patch_result)
            if validation_command:
                validation_result = (
                    tools.auto_repair_loop(validation_command)
                    if auto_repair
                    else tools.run_validation(validation_command)
                )
            else:
                validation_result = tools.git_diff() if changed_files else status
        else:
            validation_result = (
                tools.auto_repair_loop(validation_command)
                if validation_command and auto_repair
                else tools.run_validation(validation_command)
                if validation_command
                else status
            )

        pr_summary = ""
        if include_pr_summary:
            pr_summary = tools.pr_summary(
                change_summary=patch_result,
                validation_result=validation_result,
                risks=["Review the diff before committing; no commit, push, or PR was created."],
            )
        self._persist_harness_state(plan, evidence_cache, tools, context_query=task)
        modified = ", ".join(changed_files) if changed_files else "none"
        validation_label = validation_command or ("git diff" if changed_files else "git status --short")
        lines = [
            "Change summary:",
            f"- {patch_result}",
            "",
            "Modified files:",
            f"- {modified}",
            "",
            "Validation commands and results:",
            f"- {validation_label}: {compact_text(validation_result, 1200) or 'no output'}",
            "",
            "Remaining risks or follow-up work:",
            "- Deterministic fallback was used; configure AICODING_MODEL_* for open-ended coding tasks.",
            "",
            "Current plan:",
            format_plan(plan),
        ]
        if auto_repair and validation_command:
            lines.extend(["", "Auto repair summary:", compact_text(validation_result, 2400)])
        if pr_summary:
            lines.extend(["", pr_summary])
        return "\n".join(lines)

    def _run_ask_task(self, task: str, task_id: str) -> str:
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
        self._persist_harness_state(
            plan, evidence_cache, tools, capture_diff=False, context_query=task
        )
        context = "\n\n".join(
            [
                f"Project instructions:\n{instructions}",
                f"Memory summary:\n{memory}",
                f"Relevant repository context:\n{compact_text(explained, 1800)}",
                "Project overview summary:\n"
                f"{compact_text(overview_summary, 1200) or 'No overview summary was generated.'}",
                f"Project overview context:\n{compact_text(overview, 3600) or 'No overview files were selected.'}",
                f"Repository map:\n{compact_text(repo_map, 1200)}",
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
            return "\n".join(
                [
                    "Ask mode: model-generated read-only analysis",
                    "",
                    model_response,
                    "",
                    "No files were modified and no commands were run.",
                ]
            )

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
        self._persist_harness_state(
            plan, evidence_cache, tools, capture_diff=False, context_query=task
        )
        context = "\n\n".join(
            [
                f"Memory summary:\n{memory}",
                f"Draft structured plan:\n{format_plan(plan)}",
                f"Repository map summary:\n{compact_text(repo_map, 1600)}",
                f"Query-specific context:\n{compact_text(explained, 1200)}",
            ]
        )
        model_response = self._call_read_only_model(
            mode="plan",
            task=task,
            task_id=task_id,
            context=context,
        )
        if model_response:
            return "\n".join(
                [
                    "Plan mode: model-generated plan; no repository files modified",
                    "",
                    model_response,
                ]
            )

        return "\n".join(
            [
                "Plan mode: no repository files modified",
                "Model fallback: "
                f"{self._last_read_only_model_error or 'model call failed without details'}.",
                "",
                context,
            ]
        )

    def _run_agent_mode_task(self, task: str, task_id: str) -> str:
        for state in ("intake", "explore", "plan", "edit", "verify", "summarize"):
            self._trace_state(task_id, state, f"agent mode entered {state}")
        return self._run_deterministic_task(
            task,
            task_id,
            include_pr_summary=True,
            validation_command=self._validation_command_for_task(task),
            auto_repair=True,
        )

    def _validation_command_for_task(self, task: str) -> str | None:
        commands = self._validation_commands_for_task(task)
        return commands[0] if commands else None

    def _validation_commands_for_task(self, task: str) -> list[str]:
        explicit = self._explicit_validation_commands(task)
        if explicit:
            return explicit
        lowered = task.lower()
        if "ruff" in lowered or "lint" in lowered:
            return ["python -m ruff check ."]
        if "pyright" in lowered or "type" in lowered or "类型" in task:
            return ["python -m pyright"]
        if "pytest" in lowered or "test" in lowered or "测试" in task:
            if (self.workspace / "tests").is_dir():
                return ["python -m pytest tests"]
            if any(self.workspace.glob("test_*.py")):
                return ["python -m pytest"]
            return []
        return []

    def _explicit_validation_commands(self, task: str) -> list[str]:
        commands: list[str] = []
        prefixes = (
            "python -m pytest",
            "pytest",
            "python -m ruff check",
            "ruff check",
            "python -m pyright",
            "pyright",
        )
        unsafe_fragments = ("python -c", "python -m pip", " pip ", "&&", "||", ";", "|")
        for raw_line in task.splitlines():
            line = raw_line.strip().strip("`")
            line = re.sub(r"^[-*]\s+", "", line).strip().strip("`")
            lowered = line.lower()
            if any(fragment in lowered for fragment in unsafe_fragments):
                continue
            starts: list[tuple[int, str]] = []
            for prefix in prefixes:
                match = re.search(
                    rf"^(?:run\s+|validate\s+with\s+|validation:\s*)?"
                    rf"(?P<command>{re.escape(prefix)})(?:\s|$)",
                    lowered,
                )
                if match:
                    starts.append((match.start("command"), prefix))
            if not starts:
                continue
            start, _ = min(starts, key=lambda item: item[0])
            command = line[start:].strip().strip("`")
            if command and command not in commands:
                commands.append(command)
        return commands

    def _extract_changed_files(self, patch_result: str) -> list[str]:
        prefix = "changed files: "
        if not patch_result.startswith(prefix):
            return []
        return [
            item.strip() for item in patch_result.removeprefix(prefix).split(",") if item.strip()
        ]

    def _build_simple_patch(self, task: str) -> str | None:
        create_match = re.search(
            r"create\s+([A-Za-z0-9_./\\-]+)\s+with\s+['\"](.+?)['\"]",
            task,
            re.IGNORECASE | re.DOTALL,
        )
        if create_match:
            path, content = create_match.groups()
            lines = "\n".join(f"+{line}" for line in content.split("\\n"))
            return f"*** Begin Patch\n*** Add File: {path}\n{lines}\n*** End Patch"

        set_match = re.search(
            r"set\s+([A-Za-z0-9_./\\-]+)\s+to\s+['\"](.+?)['\"]",
            task,
            re.IGNORECASE | re.DOTALL,
        )
        if set_match:
            path, content = set_match.groups()
            target = self.policy.resolve_path(path)
            if target.exists():
                current = target.read_text(encoding="utf-8")
                old_lines = "\n".join(f"-{line}" for line in current.splitlines())
                new_lines = "\n".join(f"+{line}" for line in content.split("\\n"))
                return f"*** Begin Patch\n*** Update File: {path}\n@@\n{old_lines}\n{new_lines}\n*** End Patch"
            lines = "\n".join(f"+{line}" for line in content.split("\\n"))
            return f"*** Begin Patch\n*** Add File: {path}\n{lines}\n*** End Patch"

        append_match = re.search(
            r"append\s+['\"](.+?)['\"]\s+to\s+([A-Za-z0-9_./\\-]+)",
            task,
            re.IGNORECASE | re.DOTALL,
        )
        if append_match:
            text, path = append_match.groups()
            return f"*** Begin Patch\n*** Update File: {path}\n@@\n+{text}\n*** End Patch"

        replace_match = re.search(
            r"replace\s+['\"](.+?)['\"]\s+with\s+['\"](.+?)['\"]\s+in\s+([A-Za-z0-9_./\\-]+)",
            task,
            re.IGNORECASE | re.DOTALL,
        )
        if replace_match:
            old, new, path = replace_match.groups()
            return f"*** Begin Patch\n*** Update File: {path}\n@@\n-{old}\n+{new}\n*** End Patch"

        return None

    def _run_langgraph_task(
        self,
        task: str,
        task_id: str,
        *,
        run_state: ToolRunState | None = None,
    ) -> str:
        if not LANGGRAPH_AVAILABLE or ChatOpenAI is None or StructuredTool is None:
            detail = LANGGRAPH_IMPORT_ERROR or CHAT_MODEL_IMPORT_ERROR
            raise RuntimeError(f"LangGraph runtime is not available: {detail}")

        run_state = run_state or ToolRunState()
        tools, plan, evidence_cache = self._new_tools(
            task_id,
            run_state=run_state,
            require_patch_preview=True,
        )
        structured_tool = cast(Any, StructuredTool)
        chat_openai = cast(Any, ChatOpenAI)
        messages_state = cast(Any, MessagesState)
        state_graph = cast(Any, StateGraph)
        system_message = cast(Any, SystemMessage)
        human_message = cast(Any, HumanMessage)
        tool_message = cast(Any, ToolMessage)
        end_node = cast(str, END)

        lc_tools = [
            structured_tool.from_function(tools.list_files),
            structured_tool.from_function(tools.search_text),
            structured_tool.from_function(tools.read_file),
            structured_tool.from_function(tools.write_text_file),
            structured_tool.from_function(tools.repo_map),
            structured_tool.from_function(tools.explain_context),
            structured_tool.from_function(tools.preview_patch),
            structured_tool.from_function(tools.apply_patch),
            structured_tool.from_function(tools.run_command),
            structured_tool.from_function(tools.run_validation),
            structured_tool.from_function(tools.check_environment),
            structured_tool.from_function(tools.install_python_package),
            structured_tool.from_function(tools.auto_repair_loop),
            structured_tool.from_function(tools.git_status),
            structured_tool.from_function(tools.git_diff),
            structured_tool.from_function(tools.git_branch),
            structured_tool.from_function(tools.git_diff_files),
            structured_tool.from_function(tools.git_commit_preview),
            structured_tool.from_function(tools.pr_summary),
            structured_tool.from_function(tools.text_hygiene_check),
            structured_tool.from_function(tools.text_hygiene_clean),
            structured_tool.from_function(tools.load_skill),
            structured_tool.from_function(tools.plan_update),
        ]
        tool_map = {tool.name: tool for tool in lc_tools}
        model_options: dict[str, Any] = {}
        if self.config.model.model_name.startswith("deepseek-v4"):
            model_options["extra_body"] = {"thinking": {"type": "disabled"}}

        model = chat_openai(
            api_key=self.config.model.api_key,
            base_url=self.config.model.api_base,
            model=self.config.model.model_name,
            temperature=0,
            **model_options,
        ).bind_tools(lc_tools)

        context = build_context(
            history=self.session.history,
            plan=plan,
            evidence_context=evidence_cache.to_context(3000),
            latest_diff=self.session.latest_diff,
            max_chars=self.config.harness.max_context_chars,
            project_instructions=self.project_instructions.as_context(),
            repo_map=self._repo_context_summary(task),
            documentation_summary=self._documentation_context(),
            memory_summary=self.memory_store.format(),
        )

        def call_model(state: AgentState) -> dict[str, list[Any]]:
            self.trace_writer.append(
                "model_call",
                task_id=task_id,
                input_summary="messages",
                output_summary="calling chat model",
            )
            return {"messages": [model.invoke(state["messages"])]}

        def take_action(state: AgentState) -> dict[str, list[Any]]:
            last_message = state["messages"][-1]
            messages: list[Any] = []
            for tool_call in getattr(last_message, "tool_calls", []) or []:
                name = tool_call["name"]
                args = tool_call.get("args", {})
                try:
                    if name not in tool_map:
                        available = ", ".join(sorted(tool_map))
                        result = (
                            f"unknown_tool: {name} is not available. "
                            "For long file writes use write_text_file(path, content). "
                            "For patch edits use preview_patch followed by apply_patch. "
                            f"Available tools: {available}"
                        )
                        run_state.record_tool_misuse(name, result)
                        self.trace_writer.append(
                            "tool_call",
                            task_id=task_id,
                            tool_name=name,
                            input_summary=str(args)[:400],
                            output_summary=compact_text(str(result), 600),
                            status="failed",
                        )
                    else:
                        result = tool_map[name].invoke(args)
                except Exception as exc:
                    result = f"tool_execution_failed: {exc}"
                    run_state.record_tool_result(name, "failed")
                    self.trace_writer.append(
                        "tool_call",
                        task_id=task_id,
                        tool_name=name,
                        input_summary=str(args)[:400],
                        output_summary=str(exc),
                        status="failed",
                    )
                messages.append(
                    tool_message(
                        tool_call_id=tool_call["id"],
                        name=name,
                        content=str(result),
                    )
                )
                if run_state.should_stop():
                    stop_result = (
                        "dependency_stop: harness stopped tool execution after repeated "
                        f"tool dependency or execution problems ({run_state.stop_reason})"
                    )
                    self.trace_writer.append(
                        "model_fallback",
                        task_id=task_id,
                        input_summary="tool dependency state machine stopped execution",
                        output_summary=stop_result,
                        status="failed",
                    )
                    messages.append(
                        tool_message(
                            tool_call_id=tool_call["id"],
                            name=name,
                            content=stop_result,
                        )
                    )
                    self._persist_harness_state(plan, evidence_cache, tools, context_query=task)
                    raise HarnessStop(stop_result, run_state)
            return {"messages": messages}

        def has_action(state: AgentState) -> bool:
            last_message = state["messages"][-1]
            return bool(getattr(last_message, "tool_calls", []) or [])

        graph = state_graph(messages_state)
        graph.add_node("llm", call_model)
        graph.add_node("action", take_action)
        graph.add_conditional_edges("llm", has_action, {True: "action", False: end_node})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        compiled = graph.compile()
        result = compiled.invoke(
            {
                "messages": [
                    system_message(content=f"{PROMPT}\n\n{self.skill_registry.format_inventory()}"),
                    system_message(content=f"Current harness context:\n{context}"),
                    human_message(content=task),
                ]
            },
            {"recursion_limit": self.config.harness.max_tool_rounds},
        )
        final_message = result["messages"][-1]
        response = str(getattr(final_message, "content", final_message))
        response = self._append_harness_correction(response, run_state)
        self._persist_harness_state(plan, evidence_cache, tools)
        return response

    def _append_harness_correction(self, response: str, run_state: ToolRunState) -> str:
        denied_commands = run_state.denied_commands
        failed_validations = [
            attempt for attempt in run_state.validation_attempts if attempt.status == "failed"
        ]
        denied_validations = [
            attempt for attempt in run_state.validation_attempts if attempt.status == "denied"
        ]
        dependency_violations = run_state.dependency_violations
        environment_failures = run_state.environment_failures
        policy_denials = run_state.policy_denials
        tool_misuses = run_state.tool_misuses
        code_failures = run_state.code_failures
        if (
            not denied_commands
            and not failed_validations
            and not denied_validations
            and not dependency_violations
            and not environment_failures
            and not policy_denials
            and not tool_misuses
            and not code_failures
            and not run_state.stop_reason
        ):
            return response

        passed_validations = [
            attempt for attempt in run_state.validation_attempts if attempt.status == "passed"
        ]
        lines = [
            response.rstrip(),
            "",
            "Harness correction:",
        ]
        if environment_failures:
            lines.append("- Environment blockers:")
            lines.extend(
                f"  - {attempt.command}: {attempt.detail}"
                for attempt in environment_failures
            )
        if policy_denials:
            lines.append("- Policy denials:")
            lines.extend(
                f"  - {attempt.command}: {attempt.detail}"
                for attempt in policy_denials
            )
        if tool_misuses:
            lines.append("- Tool misuses:")
            lines.extend(
                f"  - {violation.tool_name}: {violation.detail}"
                for violation in tool_misuses
            )
        if code_failures:
            lines.append("- Code validation failures:")
            lines.extend(f"  - {attempt.command}" for attempt in code_failures)
        if denied_commands and not policy_denials:
            lines.append("- Denied commands:")
            lines.extend(
                f"  - {attempt.command}: not run because it was outside the whitelist ({attempt.detail})"
                for attempt in denied_commands
            )
        if dependency_violations:
            lines.append("- Dependency violations:")
            lines.extend(
                f"  - {violation.tool_name}: {violation.detail}"
                for violation in dependency_violations
            )
        if run_state.stop_reason:
            lines.append(f"- Harness stop reason: {run_state.stop_reason}")
        if failed_validations:
            lines.append("- Failed validation commands:")
            lines.extend(f"  - {attempt.command}" for attempt in failed_validations)
        if denied_validations:
            lines.append("- Denied validation commands:")
            lines.extend(
                f"  - {attempt.command}: not run because it was outside the whitelist ({attempt.detail})"
                for attempt in denied_validations
            )
        if passed_validations:
            lines.append("- Passed validation commands:")
            lines.extend(f"  - {attempt.command}" for attempt in passed_validations)
        return "\n".join(lines)

    def _final_response_trace_summary(self, response: str) -> str:
        marker = "Harness correction:"
        matches = list(re.finditer(rf"(?m)^{re.escape(marker)}\s*$", response))
        if not matches:
            return compact_text(response, 1200)
        marker_match = matches[-1]
        before = response[: marker_match.start()]
        correction = response[marker_match.end() :]
        return "\n".join(
            [
                compact_text(before, 800),
                "",
                marker,
                compact_text(correction.strip(), 1600),
            ]
        ).strip()


def generate_session_id() -> str:
    return f"session-{uuid.uuid4().hex[:8]}"
