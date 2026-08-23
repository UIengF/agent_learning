#!/usr/bin/env python3
from __future__ import annotations

import sys


def main() -> None:
    print("[ERROR] boom", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
