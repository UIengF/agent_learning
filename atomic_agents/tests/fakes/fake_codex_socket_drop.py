#!/usr/bin/env python3
from __future__ import annotations

import sys


def main() -> None:
    # Simulates ask_codex.sh surfacing a streaming socket disconnect: non-zero
    # exit with an [ERROR] line whose text is a connection-level transient
    # problem. The adapter must classify this as "transient", not "failed".
    print(
        "[ERROR] stream error: The socket connection was closed unexpectedly",
        file=sys.stderr,
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
