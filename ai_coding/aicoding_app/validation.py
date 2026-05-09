from __future__ import annotations

from dataclasses import dataclass
import subprocess

from .context_explain import explain_context
from .permissions import WorkspacePolicy
from .test_failures import format_pytest_failures


@dataclass(frozen=True)
class ValidationResult:
    command: str
    returncode: int
    stdout: str
    stderr: str
    failure_context: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def format(self) -> str:
        status = "passed" if self.ok else "failed"
        lines = [
            f"Validation command: {self.command}",
            f"Status: {status} ({self.returncode})",
            "stdout:",
            self.stdout.strip() or "<empty>",
            "stderr:",
            self.stderr.strip() or "<empty>",
        ]
        if self.failure_context:
            lines.extend(["Failure context:", self.failure_context])
        return "\n".join(lines)


def classify_validation_command(command: str) -> str:
    lowered = command.lower()
    if "pytest" in lowered:
        return "pytest"
    if "ruff" in lowered:
        return "ruff"
    if "pyright" in lowered:
        return "pyright"
    return "generic"


def run_validation(
    policy: WorkspacePolicy,
    command: str,
    *,
    timeout_seconds: int,
) -> ValidationResult:
    validated = policy.validate_command(command)
    completed = subprocess.run(
        validated,
        cwd=policy.workspace,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    combined_output = "\n".join([completed.stdout, completed.stderr])
    failure_context = ""
    if classify_validation_command(validated) == "pytest" and completed.returncode != 0:
        failure_context = "\n\n".join(
            [
                format_pytest_failures(combined_output),
                explain_context(policy.workspace, combined_output),
            ]
        )
    return ValidationResult(
        command=validated,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        failure_context=failure_context,
    )
