from __future__ import annotations

import json
from pathlib import Path
import re
import shlex

from .agent_contracts import CompletionStatus, HarnessStop
from .evidence_cache import compress_text
from .harness_state import ToolRunState
from .permissions import PermissionDenied


class AgentCompletionMixin:
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
        if script == "note_indexer.py" and parts[2:] == ["list"]:
            return True
        if len(parts) > 2:
            return False
        return script.startswith("test_") and script.endswith(".py")

    def _should_skip_idempotent_smoke_command(self, command: str) -> bool:
        parts = self._split_command_for_inspection(command)
        if len(parts) < 5:
            return False
        executable = Path(parts[0]).name.lower()
        script = Path(parts[1]).name.lower()
        if executable not in {"python", "python.exe"} or script != "note_indexer.py":
            return False
        if parts[2] != "add" or "--tags" in parts:
            return False
        if any(part.startswith("--content=") for part in parts):
            return False
        try:
            content_index = parts.index("--content")
        except ValueError:
            return False
        if content_index <= 3 or content_index + 1 >= len(parts):
            return False
        title = parts[3]
        content = parts[content_index + 1]
        notes_path = self.workspace / "notes.json"
        if not notes_path.is_file():
            return False
        try:
            notes = json.loads(notes_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(notes, list):
            return False
        return any(
            isinstance(note, dict)
            and note.get("title") == title
            and note.get("content") == content
            for note in notes
        )

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
        completed.extend(f"Validation passed: {attempt.command}" for attempt in passed_validations)
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
                compress_text(validation_result, 1600),
                "",
                pr_summary,
            ]
        )
