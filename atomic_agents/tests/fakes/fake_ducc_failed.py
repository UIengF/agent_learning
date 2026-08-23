#!/usr/bin/env python3
from __future__ import annotations

import json


def main() -> None:
    print(
        json.dumps(
            {
                "type": "result",
                "subtype": "error",
                "result": "",
                "is_error": True,
            }
        )
    )


if __name__ == "__main__":
    main()
