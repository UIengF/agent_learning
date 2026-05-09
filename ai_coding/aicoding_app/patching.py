from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .permissions import PermissionDenied, WorkspacePolicy


@dataclass(frozen=True)
class PatchResult:
    changed_files: tuple[str, ...]


def _split_sections(lines: list[str]) -> list[tuple[str, str, list[str]]]:
    sections: list[tuple[str, str, list[str]]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("*** Add File: "):
            action = "add"
            path = line.removeprefix("*** Add File: ").strip()
        elif line.startswith("*** Update File: "):
            action = "update"
            path = line.removeprefix("*** Update File: ").strip()
        elif line.startswith("*** Delete File: "):
            action = "delete"
            path = line.removeprefix("*** Delete File: ").strip()
        else:
            index += 1
            continue

        index += 1
        body: list[str] = []
        while index < len(lines) and not lines[index].startswith("*** "):
            body.append(lines[index])
            index += 1
        sections.append((action, path, body))
    return sections


def _clean_unified_path(raw_path: str) -> str:
    path = raw_path.strip().split("\t", 1)[0].strip()
    if path in {"/dev/null", "dev/null"}:
        return path
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path


def _split_unified_sections(lines: list[str]) -> list[tuple[str, str, list[str]]]:
    sections: list[tuple[str, str, list[str]]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith("--- "):
            index += 1
            continue
        if index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
            index += 1
            continue

        old_path = _clean_unified_path(line.removeprefix("--- "))
        new_path = _clean_unified_path(lines[index + 1].removeprefix("+++ "))
        if old_path == "/dev/null":
            action = "add"
            path = new_path
        elif new_path == "/dev/null":
            action = "delete"
            path = old_path
        else:
            action = "update"
            path = new_path

        index += 2
        body: list[str] = []
        while index < len(lines):
            next_line = lines[index]
            if next_line.startswith("--- ") and index + 1 < len(lines):
                if lines[index + 1].startswith("+++ "):
                    break
            if next_line.startswith("diff --git "):
                break
            if next_line.startswith("index "):
                index += 1
                continue
            body.append(next_line)
            index += 1
        sections.append((action, path, body))
    return sections


def _sections_from_patch_body(lines: list[str]) -> list[tuple[str, str, list[str]]]:
    sections = _split_sections(lines)
    if sections:
        return sections
    return _split_unified_sections(lines)


def _parse_hunks(body: list[str]) -> list[tuple[list[str], list[str]]]:
    hunks: list[tuple[list[str], list[str]]] = []
    old: list[str] = []
    new: list[str] = []
    started = False

    def flush() -> None:
        nonlocal old, new, started
        if started:
            hunks.append((old, new))
        old = []
        new = []
        started = False

    for line in body:
        if line.startswith("@@"):
            flush()
            started = True
            continue
        started = True
        if line.startswith(" "):
            value = line[1:]
            old.append(value)
            new.append(value)
        elif line.startswith("-"):
            old.append(line[1:])
        elif line.startswith("+"):
            new.append(line[1:])
        elif line == "":
            old.append("")
            new.append("")
        else:
            old.append(line)
            new.append(line)
    flush()
    return hunks


def _added_content_lines(body: list[str]) -> list[str]:
    if any(line.startswith("@@") for line in body):
        lines: list[str] = []
        for _old_lines, new_lines in _parse_hunks(body):
            lines.extend(new_lines)
        return lines
    return [line[1:] if line.startswith("+") else line for line in body]


def _replace_once(content: str, old_lines: list[str], new_lines: list[str], path: Path) -> str:
    old_block = "\n".join(old_lines)
    new_block = "\n".join(new_lines)
    if old_block == "":
        return content + ("\n" if content and not content.endswith("\n") else "") + new_block + "\n"
    match_count = content.count(old_block)
    if match_count == 0:
        raise ValueError(f"patch hunk did not match file: {path}")
    if match_count > 1:
        raise ValueError(
            f"patch hunk matches {match_count} locations in file: {path}; "
            "cannot disambiguate without line numbers"
        )
    return content.replace(old_block, new_block, 1)


def validate_patch_paths(policy: WorkspacePolicy, patch: str) -> tuple[str, ...]:
    _validate_patch_envelope(patch)
    lines = patch.strip().splitlines()
    paths: list[str] = []
    for action, raw_path, _body in _sections_from_patch_body(lines[1:-1]):
        if not raw_path:
            raise PermissionDenied(f"patch {action} section is missing a path")
        resolved = policy.resolve_path(raw_path)
        paths.append(resolved.relative_to(policy.workspace).as_posix())
    return tuple(paths)


def _validate_patch_envelope(patch: str) -> None:
    lines = patch.strip().splitlines()
    if not lines or lines[0].strip() != "*** Begin Patch":
        raise ValueError("patch must start with *** Begin Patch")
    if lines[-1].strip() != "*** End Patch":
        raise ValueError("patch must end with *** End Patch")
    if not _sections_from_patch_body(lines[1:-1]):
        raise ValueError("patch contains no file sections")


def preview_agent_patch(policy: WorkspacePolicy, patch: str) -> PatchResult:
    _validate_patch_envelope(patch)
    lines = patch.strip().splitlines()
    changed: list[str] = []
    for action, raw_path, body in _sections_from_patch_body(lines[1:-1]):
        target = policy.resolve_path(raw_path)
        rel = target.relative_to(policy.workspace).as_posix()
        if action == "add":
            if target.exists():
                raise FileExistsError(f"file already exists: {rel}")
        elif action == "delete":
            if not target.exists():
                raise FileNotFoundError(f"file does not exist: {rel}")
        elif action == "update":
            if not target.exists():
                raise FileNotFoundError(f"file does not exist: {rel}")
            try:
                content = target.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise UnicodeDecodeError(
                    exc.encoding,
                    exc.object,
                    exc.start,
                    exc.end,
                    f"file is not valid UTF-8: {rel}",
                ) from exc
            for old_lines, new_lines in _parse_hunks(body):
                content = _replace_once(content, old_lines, new_lines, target)
        changed.append(rel)
    return PatchResult(changed_files=tuple(dict.fromkeys(changed)))


def apply_agent_patch(policy: WorkspacePolicy, patch: str) -> PatchResult:
    preview = preview_agent_patch(policy, patch)
    lines = patch.strip().splitlines()

    changed: list[str] = []
    sections = _sections_from_patch_body(lines[1:-1])

    for action, raw_path, body in sections:
        target = policy.resolve_path(raw_path)
        rel = target.relative_to(policy.workspace).as_posix()
        if action == "add":
            if target.exists():
                raise FileExistsError(f"file already exists: {rel}")
            content_lines = _added_content_lines(body)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("\n".join(content_lines) + "\n", encoding="utf-8")
            changed.append(rel)
        elif action == "delete":
            if not target.exists():
                raise FileNotFoundError(f"file does not exist: {rel}")
            target.unlink()
            changed.append(rel)
        elif action == "update":
            if not target.exists():
                raise FileNotFoundError(f"file does not exist: {rel}")
            content = target.read_text(encoding="utf-8")
            for old_lines, new_lines in _parse_hunks(body):
                content = _replace_once(content, old_lines, new_lines, target)
            target.write_text(content, encoding="utf-8")
            changed.append(rel)
    return PatchResult(changed_files=preview.changed_files or tuple(dict.fromkeys(changed)))
