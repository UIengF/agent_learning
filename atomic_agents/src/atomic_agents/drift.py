"""Changed-file drift detection for scheduler write atoms."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


SnapshotMode = Literal["git", "mtime"]

_EXCLUDED_DIRS = {".git", ".venv", "__pycache__", ".atomic-agents"}


@dataclass(frozen=True)
class Snapshot:
    """Workspace changed-file snapshot.

    In git mode ``files`` is the set returned by ``git status --porcelain``.
    In mtime mode ``files`` is the current workspace file set and ``metadata``
    stores ``mtime_ns`` and size for each path.
    """

    mode: SnapshotMode
    files: set[str]
    metadata: dict[str, tuple[int, int]]


def snapshot(workspace: str) -> Snapshot:
    """Capture the workspace's changed-file state.

    Git repositories use ``git status --porcelain`` so tracked and untracked
    changes are compared against git state. Non-git workspaces, or git command
    failures, fall back to a recursive mtime+size snapshot.
    """

    root = Path(workspace)
    git_files = _git_changed_files(root)
    if git_files is not None:
        return Snapshot(mode="git", files=git_files, metadata={})
    return _mtime_snapshot(root)


def detect_drift(
    workspace: str,
    pre: Snapshot,
    post: Snapshot,
    write_scope: list[str],
    ignore_paths: list[str] | None = None,
) -> list[str]:
    """Return changed files outside the declared ``write_scope``.

    ``write_scope`` entries may be exact files or directory prefixes. A trailing
    slash (``"src/"``) always marks a directory. An entry without a trailing
    slash is treated as a directory prefix when it names an existing directory
    in ``workspace`` (e.g. a bare ``"utils"``), otherwise as an exact file. A
    changed file below a declared directory is not drift.

    ``ignore_paths`` is used for parallel sibling writers. Changed files below
    those scopes are excluded from this node's drift decision because they are
    legal products of peers in the same scheduler batch.
    """

    root = Path(workspace)
    covered = [_normalize_scope_entry(item, root) for item in write_scope]
    ignored = [_normalize_scope_entry(item, root) for item in (ignore_paths or [])]
    changed = changed_files(pre, post)
    drift = [path for path in changed if not _is_covered(path, covered) and not _is_covered(path, ignored)]
    return sorted(drift)


def changed_files(pre: Snapshot, post: Snapshot) -> list[str]:
    """Return files whose changed state differs between two snapshots."""

    if pre.mode == "git" and post.mode == "git":
        return sorted(pre.files.symmetric_difference(post.files))

    all_paths = set(pre.metadata) | set(post.metadata)
    return sorted(path for path in all_paths if pre.metadata.get(path) != post.metadata.get(path))


def _git_changed_files(root: Path) -> set[str] | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, ValueError):
        return None

    if completed.returncode != 0:
        return None

    return _parse_porcelain(completed.stdout)


def _parse_porcelain(output: str) -> set[str]:
    files: set[str] = set()
    for raw_line in output.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if line.startswith("?? "):
            files.add(_normalize_relative_path(line[3:]))
            continue
        if len(line) < 4:
            continue
        path_part = line[3:]
        if " -> " in path_part:
            _old_path, path_part = path_part.rsplit(" -> ", 1)
        files.add(_normalize_relative_path(path_part))
    return {path for path in files if path}


def _mtime_snapshot(root: Path) -> Snapshot:
    metadata: dict[str, tuple[int, int]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in _EXCLUDED_DIRS]
        for filename in filenames:
            path = Path(dirpath) / filename
            try:
                stat = path.stat()
            except OSError:
                continue
            relative = _normalize_relative_path(os.path.relpath(path, root))
            if relative:
                metadata[relative] = (stat.st_mtime_ns, stat.st_size)
    return Snapshot(mode="mtime", files=set(metadata), metadata=metadata)


def _normalize_scope_entry(path: str, root: Path | None = None) -> tuple[str, bool] | None:
    normalized = path.strip().replace("\\", "/")
    if not normalized:
        return None
    while normalized.startswith("./"):
        normalized = normalized[2:]
    is_directory = normalized.endswith("/")
    normalized = normalized.strip("/")
    if not normalized or normalized == ".":
        return None
    # A bare entry (no trailing slash) is a directory prefix when it names an
    # existing directory in the workspace; otherwise it is an exact file path.
    if not is_directory and root is not None and (root / normalized).is_dir():
        is_directory = True
    return normalized, is_directory


def _normalize_relative_path(path: str) -> str:
    normalized = path.strip().strip('"').replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.strip("/")


def _is_covered(path: str, covered: list[tuple[str, bool] | None]) -> bool:
    normalized = _normalize_relative_path(path)
    for entry in covered:
        if entry is None:
            continue
        scope_path, is_directory = entry
        if normalized == scope_path:
            return True
        if is_directory and normalized.startswith(f"{scope_path}/"):
            return True
    return False


__all__ = ["Snapshot", "changed_files", "detect_drift", "snapshot"]
