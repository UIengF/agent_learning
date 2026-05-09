from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib

from .evidence_cache import compact_text


class TaskPhase(str, Enum):
    STARTED = "started"
    PLANNED = "planned"
    PATCH_PREVIEWED = "patch_previewed"
    PATCH_APPLIED = "patch_applied"
    FILE_WRITTEN = "file_written"
    VALIDATED = "validated"
    SUMMARIZING = "summarizing"
    STOPPED = "stopped"


class ToolProblemKind(str, Enum):
    CODE_FAILURE = "code_failure"
    ENVIRONMENT_FAILURE = "environment_failure"
    POLICY_DENIED = "policy_denied"
    TOOL_MISUSE = "tool_misuse"
    DEPENDENCY_VIOLATION = "dependency_violation"


@dataclass(frozen=True)
class ValidationAttempt:
    command: str
    status: str
    stdout_summary: str = ""
    stderr_summary: str = ""
    detail: str = ""


@dataclass(frozen=True)
class DependencyViolation:
    tool_name: str
    detail: str


@dataclass
class ToolRunState:
    phase: TaskPhase = TaskPhase.STARTED
    successful_patch_previews: set[str] = field(default_factory=set)
    applied_patches: list[str] = field(default_factory=list)
    written_files: list[str] = field(default_factory=list)
    denied_commands: list[ValidationAttempt] = field(default_factory=list)
    validation_attempts: list[ValidationAttempt] = field(default_factory=list)
    dependency_violations: list[DependencyViolation] = field(default_factory=list)
    environment_failures: list[ValidationAttempt] = field(default_factory=list)
    policy_denials: list[ValidationAttempt] = field(default_factory=list)
    tool_misuses: list[DependencyViolation] = field(default_factory=list)
    code_failures: list[ValidationAttempt] = field(default_factory=list)
    failed_tool_count: int = 0
    denied_tool_count: int = 0
    consecutive_problem_count: int = 0
    stop_reason: str = ""
    terminal_status: str = ""
    environment_diagnosed: bool = False
    max_policy_denials: int = 4
    max_dependency_violations: int = 3
    max_consecutive_problems: int = 6

    def mark_patch_preview(self, patch: str) -> None:
        self.successful_patch_previews.add(patch_fingerprint(patch))
        self.phase = TaskPhase.PATCH_PREVIEWED

    def has_patch_preview(self, patch: str) -> bool:
        return patch_fingerprint(patch) in self.successful_patch_previews

    def mark_plan(self) -> None:
        self.phase = TaskPhase.PLANNED

    def mark_patch_applied(self, patch: str) -> None:
        self.applied_patches.append(patch_fingerprint(patch))
        self.phase = TaskPhase.PATCH_APPLIED

    def mark_file_written(self, path: str) -> None:
        self.written_files.append(path)
        self.phase = TaskPhase.FILE_WRITTEN

    def record_denied_command(self, command: str, detail: str) -> None:
        self.denied_commands.append(
            ValidationAttempt(command=command, status="denied", detail=compact_text(detail, 500))
        )

    def record_validation(
        self,
        *,
        command: str,
        status: str,
        stdout: str = "",
        stderr: str = "",
        detail: str = "",
    ) -> None:
        self.validation_attempts.append(
            ValidationAttempt(
                command=command,
                status=status,
                stdout_summary=compact_text(stdout, 500),
                stderr_summary=compact_text(stderr, 500),
                detail=compact_text(detail, 500),
            )
        )
        self.phase = TaskPhase.VALIDATED

    def has_validation_environment_failure(self) -> bool:
        attempts = [*self.validation_attempts, *self.environment_failures]
        return any(
            "Validation environment failure detected" in attempt.detail
            or "Validation environment failure detected" in attempt.stderr_summary
            or "Validation environment failure detected" in attempt.stdout_summary
            for attempt in attempts
        )

    def record_environment_failure(
        self,
        command: str,
        detail: str,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        self.environment_failures.append(
            ValidationAttempt(
                command=command,
                status="failed",
                stdout_summary=compact_text(stdout, 500),
                stderr_summary=compact_text(stderr, 500),
                detail=compact_text(detail, 500),
            )
        )
        if not self.environment_diagnosed:
            self.terminal_status = "environment_blocker_pending_diagnosis"

    def mark_environment_diagnosed(self) -> None:
        self.environment_diagnosed = True
        if self.environment_failures:
            self.terminal_status = "environment_blocker_detected"
        self._refresh_stop_reason()

    def record_policy_denial(self, command: str, detail: str) -> None:
        self.policy_denials.append(
            ValidationAttempt(command=command, status="denied", detail=compact_text(detail, 500))
        )
        self._refresh_stop_reason()

    def record_tool_misuse(self, tool_name: str, detail: str) -> None:
        self.tool_misuses.append(
            DependencyViolation(tool_name=tool_name, detail=compact_text(detail, 500))
        )
        self.record_tool_result(tool_name, "failed")

    def record_code_failure(
        self,
        command: str,
        stdout: str = "",
        stderr: str = "",
        detail: str = "",
    ) -> None:
        self.code_failures.append(
            ValidationAttempt(
                command=command,
                status="failed",
                stdout_summary=compact_text(stdout, 500),
                stderr_summary=compact_text(stderr, 500),
                detail=compact_text(detail, 500),
            )
        )

    def classify_validation_failure(
        self,
        command: str,
        stdout: str = "",
        stderr: str = "",
        detail: str = "",
    ) -> ToolProblemKind:
        combined = "\n".join([stdout, stderr, detail])
        if "Validation environment failure detected" in combined:
            return ToolProblemKind.ENVIRONMENT_FAILURE
        if detail.startswith("command is outside whitelist") or detail.startswith(
            "shell control operators are not allowed"
        ):
            return ToolProblemKind.POLICY_DENIED
        if "pytest" in command.lower() or "ruff" in command.lower() or "pyright" in command.lower():
            return ToolProblemKind.CODE_FAILURE
        return ToolProblemKind.CODE_FAILURE

    def check_dependency(
        self,
        tool_name: str,
        *,
        patch: str = "",
        plan_ready: bool = True,
        require_patch_preview: bool = False,
    ) -> str | None:
        if tool_name == "apply_patch" and not plan_ready:
            return "plan_required: call plan_update with edit_steps before apply_patch"
        if tool_name == "apply_patch" and require_patch_preview and not self.has_patch_preview(patch):
            return "preview_required: call preview_patch successfully before apply_patch"
        if tool_name == "write_text_file" and not plan_ready:
            return "plan_required: call plan_update with edit_steps before write_text_file"
        return None

    def record_dependency_violation(self, tool_name: str, detail: str) -> None:
        self.dependency_violations.append(
            DependencyViolation(tool_name=tool_name, detail=compact_text(detail, 500))
        )
        self.record_tool_result(tool_name, "denied")

    def record_tool_result(self, tool_name: str, status: str) -> None:
        if status == "ok":
            self.consecutive_problem_count = 0
            return
        if status == "denied":
            self.denied_tool_count += 1
        elif status == "failed":
            self.failed_tool_count += 1
        self.consecutive_problem_count += 1
        self._refresh_stop_reason()

    def _refresh_stop_reason(self) -> None:
        if self.stop_reason:
            return
        if self.environment_failures and self.environment_diagnosed:
            self.stop_reason = "environment_blocker_detected"
            self.terminal_status = "environment_blocker_detected"
            self.phase = TaskPhase.STOPPED
        elif len(self.policy_denials) >= self.max_policy_denials:
            self.stop_reason = "policy_denial_loop"
            self.terminal_status = "policy_denial_loop"
            self.phase = TaskPhase.STOPPED
        elif len(self.tool_misuses) >= 2:
            self.stop_reason = "tool_misuse_loop"
            self.terminal_status = "tool_misuse_loop"
            self.phase = TaskPhase.STOPPED
        elif len(self.dependency_violations) >= self.max_dependency_violations:
            self.stop_reason = "too_many_dependency_violations"
            self.terminal_status = "too_many_dependency_violations"
            self.phase = TaskPhase.STOPPED
        elif self.consecutive_problem_count >= self.max_consecutive_problems:
            self.stop_reason = "too_many_consecutive_tool_failures"
            self.terminal_status = "too_many_consecutive_tool_failures"
            self.phase = TaskPhase.STOPPED

    def should_stop(self) -> bool:
        self._refresh_stop_reason()
        return bool(self.stop_reason)

    def reset_problem_counters_for_fallback(self) -> None:
        self.dependency_violations.clear()
        self.tool_misuses.clear()
        self.failed_tool_count = 0
        self.denied_tool_count = 0
        self.consecutive_problem_count = 0
        if self.stop_reason in {
            "too_many_dependency_violations",
            "too_many_consecutive_tool_failures",
            "tool_misuse_loop",
        }:
            self.stop_reason = ""
            self.terminal_status = ""
            self.phase = TaskPhase.STARTED


def patch_fingerprint(patch: str) -> str:
    normalized = "\n".join(line.rstrip() for line in patch.strip().splitlines())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
