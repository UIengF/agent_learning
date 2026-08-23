from __future__ import annotations

import threading
import time
from pathlib import Path

from atomic_agents.locks import ConsultingFileLock


def test_hold_serializes_conflicting_write_paths(tmp_path: Path) -> None:
    lock = ConsultingFileLock()
    entered_first = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    entry_times: dict[str, float] = {}

    def first() -> None:
        with lock.hold([str(tmp_path / "same.txt")]):
            entry_times["first"] = time.monotonic()
            entered_first.set()
            assert release_first.wait(timeout=1.0)

    def second() -> None:
        assert entered_first.wait(timeout=1.0)
        with lock.hold([str(tmp_path / "same.txt")]):
            entry_times["second"] = time.monotonic()
            second_entered.set()

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)

    first_thread.start()
    second_thread.start()
    assert entered_first.wait(timeout=1.0)
    time.sleep(0.05)
    assert not second_entered.is_set()

    release_first.set()
    first_thread.join(timeout=1.0)
    second_thread.join(timeout=1.0)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert second_entered.is_set()
    assert entry_times["second"] >= entry_times["first"]


def test_disjoint_write_scopes_do_not_block_each_other(tmp_path: Path) -> None:
    lock = ConsultingFileLock()
    barrier = threading.Barrier(2)
    counter_lock = threading.Lock()
    current = 0
    max_concurrent = 0

    def worker(path: str) -> None:
        nonlocal current, max_concurrent
        with lock.hold([path]):
            with counter_lock:
                current += 1
                max_concurrent = max(max_concurrent, current)
            barrier.wait(timeout=1.0)
            time.sleep(0.02)
            with counter_lock:
                current -= 1

    threads = [
        threading.Thread(target=worker, args=(str(tmp_path / "a.txt"),)),
        threading.Thread(target=worker, args=(str(tmp_path / "b.txt"),)),
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1.0)

    assert all(not thread.is_alive() for thread in threads)
    assert max_concurrent == 2


def test_directory_prefix_conflicts_serialize() -> None:
    lock = ConsultingFileLock()
    entered_first = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def first() -> None:
        with lock.hold(["src/"]):
            entered_first.set()
            assert release_first.wait(timeout=1.0)

    def second() -> None:
        assert entered_first.wait(timeout=1.0)
        with lock.hold(["src/a.py"]):
            second_entered.set()

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)

    first_thread.start()
    second_thread.start()
    assert entered_first.wait(timeout=1.0)
    time.sleep(0.05)
    assert not second_entered.is_set()

    release_first.set()
    first_thread.join(timeout=1.0)
    second_thread.join(timeout=1.0)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert second_entered.is_set()


def test_empty_write_scope_is_read_only_and_does_not_reserve_paths() -> None:
    lock = ConsultingFileLock()

    with lock.hold([]) as read_only_handle:
        assert read_only_handle.paths == set()
        with lock.hold(["src/a.py"]) as write_handle:
            assert write_handle.paths == {"src/a.py"}


def test_same_path_queue_allows_only_one_holder_at_a_time(tmp_path: Path) -> None:
    lock = ConsultingFileLock()
    start = threading.Event()
    counter_lock = threading.Lock()
    current = 0
    max_concurrent = 0
    entered = 0

    def worker() -> None:
        nonlocal current, max_concurrent, entered
        assert start.wait(timeout=1.0)
        with lock.hold([str(tmp_path / "shared.txt")]):
            with counter_lock:
                current += 1
                entered += 1
                max_concurrent = max(max_concurrent, current)
            time.sleep(0.03)
            with counter_lock:
                current -= 1

    threads = [threading.Thread(target=worker) for _ in range(3)]

    for thread in threads:
        thread.start()
    start.set()
    for thread in threads:
        thread.join(timeout=1.0)

    assert all(not thread.is_alive() for thread in threads)
    assert entered == 3
    assert max_concurrent == 1
