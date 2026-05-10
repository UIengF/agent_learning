from __future__ import annotations

import re

from .agent_progress import progress as _progress
from .evidence_cache import compress_text
from .harness_state import ToolRunState
from .plan import format_plan


class AgentFallbackMixin:
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
        _progress(f"Starting deterministic task flow for {task_id}")
        tools, plan, evidence_cache = self._new_tools(task_id, run_state=run_state)
        skill_name = self._select_skill(task)
        _progress(f"Loading skill context: {skill_name}")
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
            edit_steps=["Apply a targeted patch if the task matches a supported fallback pattern."],
            validation_steps=["Inspect git diff after any edit."],
            risks=[
                "Fallback patterns are intentionally narrow; configure model-backed agent mode for open-ended coding tasks.",
            ],
        )

        if patch:
            _progress(f"Applying deterministic patch for {task_id}")
            patch_result = tools.apply_patch(patch)
            changed_files = self._extract_changed_files(patch_result)
            if validation_command:
                _progress(f"Validating deterministic changes for {task_id}")
                validation_result = (
                    tools.auto_repair_loop(validation_command)
                    if auto_repair
                    else tools.run_validation(validation_command)
                )
            else:
                validation_result = tools.git_diff() if changed_files else status
        else:
            _progress(f"No deterministic patch matched for {task_id}; running validation or status")
            validation_result = (
                tools.auto_repair_loop(validation_command)
                if validation_command and auto_repair
                else tools.run_validation(validation_command)
                if validation_command
                else status
            )

        pr_summary = ""
        if include_pr_summary:
            _progress(f"Building PR summary for {task_id}")
            pr_summary = tools.pr_summary(
                change_summary=patch_result,
                validation_result=validation_result,
                risks=["Review the diff before committing; no commit, push, or PR was created."],
            )
        self._persist_harness_state(plan, evidence_cache, tools, context_query=task)
        _progress(f"Deterministic task flow complete for {task_id}")
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
            f"- {validation_label}: {compress_text(validation_result, 1200) or 'no output'}",
            "",
            "Remaining risks or follow-up work:",
            "- Deterministic fallback was used; configure AICODING_MODEL_* for open-ended coding tasks.",
            "",
            "Current plan:",
            format_plan(plan),
        ]
        if auto_repair and validation_command:
            lines.extend(["", "Auto repair summary:", compress_text(validation_result, 2400)])
        if pr_summary:
            lines.extend(["", pr_summary])
        return "\n".join(lines)

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
        return [item.strip() for item in patch_result.removeprefix(prefix).split(",") if item.strip()]

    def _build_simple_patch(self, task: str) -> str | None:
        return self.fallback_patterns.build_patch(task, policy=self.policy)
