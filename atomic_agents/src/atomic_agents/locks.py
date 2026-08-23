"""Consulting file locks for scheduler-level write coordination."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator


class ConsultingFileLock:
    """Advisory write mutex for scheduler-declared file scopes.

    This lock is a concurrency primitive for the scheduler: writers declare the
    paths in their ``write_scope`` and conflicting writers wait until those
    paths are released. Read-only atoms do not use this lock. It is not a
    permission or capability check and does not prevent code from writing files
    outside its declared scope.
    """

    def __init__(self) -> None:
        self._mutex = threading.Lock()
        self._condition = threading.Condition(self._mutex)
        self._occupied: set[str] = set()

    def acquire(self, write_scope: list[str]) -> "ConsultingFileLockHandle":
        """Block until all declared paths are free, then reserve them."""

        paths = _normalize_scope(write_scope)
        with self._condition:
            while self._conflicts(paths):
                self._condition.wait()
            self._occupied.update(paths)

        return ConsultingFileLockHandle(self, paths)

    def release(self, handle: "ConsultingFileLockHandle") -> None:
        """Release paths reserved by ``acquire`` and wake waiting writers."""

        with self._condition:
            if handle._released:
                return
            self._occupied.difference_update(handle.paths)
            handle._released = True
            self._condition.notify_all()

    @contextmanager
    def hold(self, write_scope: list[str]) -> Iterator["ConsultingFileLockHandle"]:
        """Context manager wrapper around ``acquire``/``release``."""

        handle = self.acquire(write_scope)
        try:
            yield handle
        finally:
            self.release(handle)

    def _conflicts(self, paths: set[str]) -> bool:
        return any(_paths_conflict(path, occupied) for path in paths for occupied in self._occupied)


class ConsultingFileLockHandle:
    """Handle returned by ``ConsultingFileLock.acquire``."""

    def __init__(self, owner: ConsultingFileLock, paths: set[str]) -> None:
        self._owner = owner
        self.paths = set(paths)
        self._released = False

    def release(self) -> None:
        """Release this handle's reserved paths."""

        self._owner.release(self)

    def __enter__(self) -> "ConsultingFileLockHandle":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


def _normalize_scope(write_scope: list[str]) -> set[str]:
    paths: set[str] = set()
    for raw_path in write_scope:
        path = raw_path.strip().replace("\\", "/")
        if not path:
            continue
        while path.startswith("./"):
            path = path[2:]
        path = path.rstrip("/")
        if path and path != ".":
            paths.add(path)
    return paths


def _paths_conflict(left: str, right: str) -> bool:
    return left == right or left.startswith(f"{right}/") or right.startswith(f"{left}/")


__all__ = ["ConsultingFileLock", "ConsultingFileLockHandle"]
