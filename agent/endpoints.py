"""Explicit trusted peer configuration shared by the Agent and HIL CLI."""

from __future__ import annotations

import re


def parse_endpoints(entries: list[str]) -> dict[str, str]:
    result = {}
    for entry in entries:
        node_id, separator, endpoint = entry.partition("=")
        if (
            not separator
            or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", node_id)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+:[0-9]+", endpoint)
            or not 1 <= int(endpoint.rsplit(":", 1)[1]) <= 65535
        ):
            raise ValueError("endpoint must be node_id=hostname:port (IPv4 or DNS)")
        if node_id in result:
            raise ValueError(f"duplicate endpoint for {node_id}")
        result[node_id] = endpoint
    return result
