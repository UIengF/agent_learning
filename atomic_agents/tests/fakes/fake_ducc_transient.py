#!/usr/bin/env python3
from __future__ import annotations

import json


def main() -> None:
    # Simulates the ducc CLI surfacing a transient inference-gateway error
    # (503 credentials exhausted). The adapter must classify this as
    # status=="transient", not "failed".
    print(
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": True,
                "api_error_status": 503,
                "result": (
                    "API Error: 503 All 133 credentials exhausted (0 available); "
                    "retry later. This is a server-side issue, usually temporary."
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
