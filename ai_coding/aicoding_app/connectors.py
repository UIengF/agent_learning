from __future__ import annotations


def list_connectors() -> str:
    return "\n".join(
        [
            "Connectors:",
            "- github: disabled/read-only placeholder; no network calls are made.",
            "- mcp: disabled/read-only placeholder; no MCP servers are contacted.",
            "- docs: disabled/read-only placeholder; external document sources are not queried.",
        ]
    )
