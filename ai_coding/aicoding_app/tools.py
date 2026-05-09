from __future__ import annotations

from dataclasses import dataclass
import subprocess
from typing import Any

from .evidence_cache import EvidenceCache, compact_text
from .context_explain import explain_context
from .hooks import preview_hook
from .patching import apply_agent_patch, preview_agent_patch, validate_patch_paths
from .permissions import WorkspacePolicy
from .plan import CodingPlan
from .pr_summary import build_commit_preview, build_pr_summary
from .repair_loop import build_repair_candidate
from .repo_map import build_repo_map
from .skills import SkillRegistry
from .text_hygiene import clean_text_hygiene, format_text_hygiene_report
from .trace import StructuredTraceWriter
from .validation import run_validation as execute_validation


_SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "runtime"}


def _stdout_lines(command_text: str) -> list[str]:
    lines = command_text.splitlines()
    try:
        start = lines.index("stdout:") + 1
    except ValueError:
        return []
    try:
        end = lines.index("stderr:")
    except ValueError:
        end = len(lines)
    return [line.strip() for line in lines[start:end] if line.strip()]


@dataclass(frozen=True)
class CommandResult:
    command: str
    returncode: int
    stdout: str
    stderr: str

    def as_text(self) -> str:
        return "\n".join(
            [
                f"$ {self.command}",
                f"returncode: {self.returncode}",
                "stdout:",
                self.stdout.strip(),
                "stderr:",
                self.stderr.strip(),
            ]
        ).strip()


class CodingTools:
    def __init__(
        self,
        *,
        policy: WorkspacePolicy,
        plan: CodingPlan,
        evidence_cache: EvidenceCache,
        skill_registry: SkillRegistry,
        trace_writer: StructuredTraceWriter,
        command_timeout_seconds: int = 60,
        task_id: str | None = None,
    ):
        self.policy = policy
        self.plan = plan
        self.evidence_cache = evidence_cache
        self.skill_registry = skill_registry
        self.trace_writer = trace_writer
        self.command_timeout_seconds = command_timeout_seconds
        self.task_id = task_id

    def _trace(
        self,
        tool_name: str,
        input_summary: str,
        output_summary: str,
        *,
        status: str = "ok",
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.trace_writer.append(
            "tool_call",
            task_id=self.task_id,
            tool_name=tool_name,
            input_summary=input_summary,
            output_summary=output_summary,
            status=status,
            payload=payload,
        )

    def list_files(self, pattern: str = "**/*") -> str:
        """List workspace files matching a glob pattern."""
        files: list[str] = []
        for path in sorted(self.policy.workspace.glob(pattern)):
            if path.is_dir():
                continue
            if any(part in _SKIP_DIRS for part in path.relative_to(self.policy.workspace).parts):
                continue
            files.append(path.relative_to(self.policy.workspace).as_posix())
        result = "\n".join(files)
        self.evidence_cache.add("files", pattern, result)
        self._trace("list_files", pattern, f"{len(files)} files")
        return result

    def search_text(self, query: str, glob: str = "**/*") -> str:
        """Search text in workspace files and return matching lines."""
        matches: list[str] = []
        for path in sorted(self.policy.workspace.glob(glob)):
            if path.is_dir():
                continue
            rel_parts = path.relative_to(self.policy.workspace).parts
            if any(part in _SKIP_DIRS for part in rel_parts):
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(lines, start=1):
                if query.lower() in line.lower():
                    rel = path.relative_to(self.policy.workspace).as_posix()
                    matches.append(f"{rel}:{line_number}: {line.strip()}")
                    if len(matches) >= 50:
                        break
            if len(matches) >= 50:
                break
        result = "\n".join(matches)
        self.evidence_cache.add("search", f"{glob}:{query}", result)
        self._trace("search_text", f"{query} in {glob}", f"{len(matches)} matches")
        return result

    def repo_map(self) -> str:
        """Return a compact repository map with files, tests, dependencies, and Python symbols."""
        result = build_repo_map(self.policy.workspace).format()
        self.evidence_cache.add("repo_map", ".", result)
        self._trace("repo_map", ".", compact_text(result, 500))
        return result

    def explain_context(self, query: str) -> str:
        """Explain a symbol, file, or pytest failure using repository intelligence."""
        result = explain_context(self.policy.workspace, query)
        self.evidence_cache.add("context_explain", query, result)
        self._trace("explain_context", compact_text(query, 300), compact_text(result, 500))
        return result

    def read_file(self, path: str, start: int | None = None, end: int | None = None) -> str:
        """Read a workspace file or line range with line numbers."""
        resolved = self.policy.resolve_path(path)
        lines = resolved.read_text(encoding="utf-8").splitlines()
        start_line = max(1, start or 1)
        end_line = min(len(lines), end or len(lines))
        selected = lines[start_line - 1 : end_line]
        numbered = "\n".join(
            f"{line_number}: {line}" for line_number, line in enumerate(selected, start=start_line)
        )
        key = f"{self.policy.relative_path(path)}:{start_line}-{end_line}"
        self.evidence_cache.add("file", key, numbered)
        self._trace("read_file", key, f"{len(selected)} lines")
        return numbered

    def apply_patch(self, patch: str) -> str:
        """Apply an exact *** Begin Patch block after a coding plan has been recorded."""
        if not self.plan.ready_for_edit:
            result = "plan_required: call plan_update with edit_steps before apply_patch"
            self._trace("apply_patch", compact_text(patch, 300), result, status="denied")
            return result
        try:
            changed_paths = validate_patch_paths(self.policy, patch)
            result = apply_agent_patch(self.policy, patch)
        except Exception as exc:
            result_text = f"patch_failed: {exc}"
            self._trace(
                "apply_patch",
                compact_text(patch, 300),
                result_text,
                status="failed",
            )
            return result_text
        output = "changed files: " + ", ".join(result.changed_files)
        self.evidence_cache.add("patch", ",".join(changed_paths), patch)
        self._trace("apply_patch", compact_text(patch, 300), output)
        return output

    def preview_patch(self, patch: str) -> str:
        """Dry-run an exact *** Begin Patch block without modifying files."""
        try:
            changed_paths = validate_patch_paths(self.policy, patch)
            result = preview_agent_patch(self.policy, patch)
        except Exception as exc:
            result_text = f"patch_preview_failed: {exc}"
            self._trace(
                "preview_patch",
                compact_text(patch, 300),
                result_text,
                status="failed",
            )
            return result_text
        output = "preview changed files: " + ", ".join(result.changed_files)
        self.evidence_cache.add("patch_preview", ",".join(changed_paths), patch)
        self._trace("preview_patch", compact_text(patch, 300), output)
        return output

    def run_command(self, command: str) -> str:
        """Run an allowed command inside the workspace and return its output."""
        try:
            validated = self.policy.validate_command(command)
        except Exception as exc:
            self._trace("run_command", command, str(exc), status="denied")
            return f"command_denied: {exc}"

        completed = subprocess.run(
            validated,
            cwd=self.policy.workspace,
            shell=True,
            capture_output=True,
            text=True,
            timeout=self.command_timeout_seconds,
        )
        result = CommandResult(
            command=validated,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        text = result.as_text()
        status = "ok" if completed.returncode == 0 else "failed"
        self.evidence_cache.add("command", validated, text)
        self._trace("run_command", validated, compact_text(text, 500), status=status)
        return text

    def run_validation(self, command: str) -> str:
        """Run an allowed validation command and include structured failure context."""
        try:
            result = execute_validation(
                self.policy,
                command,
                timeout_seconds=self.command_timeout_seconds,
            )
        except Exception as exc:
            self._trace("run_validation", command, str(exc), status="denied")
            return f"validation_denied: {exc}"
        text = result.format()
        status = "ok" if result.ok else "failed"
        self.evidence_cache.add("validation", result.command, text)
        self._trace("run_validation", result.command, compact_text(text, 500), status=status)
        hook_text = preview_hook(self.policy, "after_verify").format()
        return f"{text}\n\n{hook_text}"

    def auto_repair_loop(self, validation_command: str, max_attempts: int = 2) -> str:
        """Run validation, attempt conservative repairs, and re-run validation."""
        attempts = max(1, min(max_attempts, 3))
        lines = [f"Auto repair loop: {validation_command}", f"Max attempts: {attempts}"]
        for attempt in range(1, attempts + 1):
            validation_text = self.run_validation(validation_command)
            lines.extend([f"Attempt {attempt} validation:", compact_text(validation_text, 1600)])
            if "Status: passed (0)" in validation_text:
                lines.append("Auto repair result: validation passed.")
                result = "\n\n".join(lines)
                self.evidence_cache.add("auto_repair", validation_command, result)
                self._trace("auto_repair_loop", validation_command, compact_text(result, 500))
                return result
            candidate = build_repair_candidate(self.policy, validation_text)
            if candidate is None:
                lines.append("Auto repair result: no safe deterministic repair matched.")
                break
            preview = self.preview_patch(candidate.patch)
            lines.extend([f"Repair candidate: {candidate.reason}", preview])
            if not preview.startswith("preview changed files:"):
                lines.append("Auto repair result: repair preview failed.")
                break
            applied = self.apply_patch(candidate.patch)
            lines.append(applied)
            if not applied.startswith("changed files:"):
                lines.append("Auto repair result: repair patch was not applied.")
                break
        result = "\n\n".join(lines)
        self.evidence_cache.add("auto_repair", validation_command, result)
        self._trace("auto_repair_loop", validation_command, compact_text(result, 500), status="failed")
        return result

    def git_status(self) -> str:
        """Return git status for the workspace."""
        return self.run_command("git status --short -- .")

    def git_diff(self) -> str:
        """Return git diff for the workspace."""
        return self.run_command("git diff -- .")

    def git_branch(self) -> str:
        """Return the current git branch."""
        return self.run_command("git status -b --short -- .")

    def git_diff_files(self) -> str:
        """Return changed file names from git diff."""
        return self.run_command("git status --short -- .")

    def git_commit_preview(self) -> str:
        """Return a commit message preview and changed-file summary without committing."""
        changed_text = self.git_diff_files()
        changed_files = _stdout_lines(changed_text)
        diff = self.git_diff()
        result = build_commit_preview(changed_files, diff)
        self.evidence_cache.add("git_commit_preview", ".", result)
        self._trace("git_commit_preview", ".", compact_text(result, 500))
        return result

    def pr_summary(
        self,
        change_summary: str,
        validation_result: str = "",
        risks: list[str] | None = None,
    ) -> str:
        """Build a PR-ready summary without pushing or creating a PR."""
        changed_text = self.git_diff_files()
        changed_files = _stdout_lines(changed_text)
        result = build_pr_summary(
            change_summary=change_summary,
            changed_files=changed_files,
            validation_result=validation_result,
            risks=risks or [],
        )
        self.evidence_cache.add("pr_summary", ".", result)
        self._trace("pr_summary", compact_text(change_summary, 300), compact_text(result, 500))
        return result

    def text_hygiene_check(self) -> str:
        """Report non-ASCII text in common source/document files without modifying files."""
        result = format_text_hygiene_report(self.policy.workspace)
        self.evidence_cache.add("text_hygiene", "check", result)
        self._trace("text_hygiene_check", ".", compact_text(result, 500))
        return result

    def text_hygiene_clean(self) -> str:
        """Apply conservative replacements for common mojibake/typographic characters."""
        result = clean_text_hygiene(self.policy.workspace)
        self.evidence_cache.add("text_hygiene", "clean", result)
        self._trace("text_hygiene_clean", ".", compact_text(result, 500))
        return result

    def load_skill(self, name: str) -> str:
        """Load full instructions for a named local skill."""
        payload = self.skill_registry.load_json(name)
        self.evidence_cache.add("skill", name, payload)
        status = "failed" if "unknown_skill" in payload else "ok"
        self._trace("load_skill", name, compact_text(payload, 300), status=status)
        return payload

    def plan_update(
        self,
        goal: str,
        assumptions: list[str] | None = None,
        files_to_check: list[str] | None = None,
        edit_steps: list[str] | None = None,
        validation_steps: list[str] | None = None,
        risks: list[str] | None = None,
    ) -> str:
        """Record the current coding plan before edits."""
        self.plan.update(
            goal=goal,
            assumptions=assumptions or [],
            files_to_check=files_to_check or [],
            edit_steps=edit_steps or [],
            validation_steps=validation_steps or [],
            risks=risks or [],
        )
        self._trace("plan_update", goal, "plan updated")
        return "plan updated"
