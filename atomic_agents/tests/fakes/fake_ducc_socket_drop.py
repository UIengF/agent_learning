#!/usr/bin/env python3
from __future__ import annotations

import json
import sys


def main() -> None:
    # Simulates the ducc CLI dropping a streaming connection mid-task: the
    # process exits NON-ZERO and the JSON payload carries is_error=True with a
    # socket-level message. The adapter must classify this as "transient" (so the
    # scheduler backs off and retries) rather than "failed" — regression guard
    # for the socket-disconnect-misjudged-as-atom_failed bug.
    print(
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": True,
                "result": (
                    "API Error: The socket connection was closed unexpectedly. "
                    "For more information, pass `verbose: true` in the second "
                    "argument to fetch()"
                ),
            }
        )
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
