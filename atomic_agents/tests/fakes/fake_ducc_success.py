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
                "session_id": "sess-fake",
                "total_cost_usd": 0.07,
                "is_error": False,
            }
        )
    )


if __name__ == "__main__":
    main()
