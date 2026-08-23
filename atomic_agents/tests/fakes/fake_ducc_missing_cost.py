#!/usr/bin/env python3
from __future__ import annotations

import json


def main() -> None:
    print(
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "result": "ok",
                "session_id": "s",
                "is_error": False,
            }
        )
    )


if __name__ == "__main__":
    main()
