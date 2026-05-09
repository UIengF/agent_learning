from __future__ import annotations

from dataclasses import dataclass
import re


_FAILED_LINE_RE = re.compile(r"^FAILED\s+(?P<nodeid>\S+?)(?:\s+-\s+(?P<summary>.*))?$")
_LOCATION_RE = re.compile(r"^(?P<path>[^:\n]+\.py):(?P<line>\d+):")


@dataclass(frozen=True)
class PytestFailure:
    nodeid: str
    path: str
    line: int | None
    test_name: str
    summary: str

    def format(self) -> str:
        location = f"{self.path}:{self.line}" if self.line is not None else self.path
        return f"{location}: {self.test_name} - {self.summary or 'no summary'}"


def parse_pytest_failures(output: str) -> tuple[PytestFailure, ...]:
    failures: list[PytestFailure] = []
    seen: set[str] = set()
    location_by_path: dict[str, int] = {}

    for line in output.splitlines():
        location = _LOCATION_RE.match(line.strip())
        if location:
            location_by_path[location.group("path").replace("\\", "/")] = int(location.group("line"))

    for line in output.splitlines():
        match = _FAILED_LINE_RE.match(line.strip())
        if not match:
            continue
        nodeid = match.group("nodeid")
        if nodeid in seen:
            continue
        seen.add(nodeid)
        path, test_name = _split_nodeid(nodeid)
        failures.append(
            PytestFailure(
                nodeid=nodeid,
                path=path,
                line=location_by_path.get(path),
                test_name=test_name,
                summary=match.group("summary") or "",
            )
        )
    return tuple(failures)


def format_pytest_failures(output: str) -> str:
    failures = parse_pytest_failures(output)
    if not failures:
        return "No pytest failures found."
    lines = ["Pytest failures:"]
    lines.extend(f"- {failure.format()}" for failure in failures)
    return "\n".join(lines)


def _split_nodeid(nodeid: str) -> tuple[str, str]:
    parts = nodeid.split("::")
    path = parts[0].replace("\\", "/")
    test_name = "::".join(parts[1:]) if len(parts) > 1 else _path_like_name(path)
    return path, test_name


def _path_like_name(path: str) -> str:
    return path.rsplit("/", 1)[-1]
