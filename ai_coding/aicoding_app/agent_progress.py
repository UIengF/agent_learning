from __future__ import annotations

import sys


def progress(message: str) -> None:
    print(f"[aicoding] {message}", file=sys.stderr, flush=True)
