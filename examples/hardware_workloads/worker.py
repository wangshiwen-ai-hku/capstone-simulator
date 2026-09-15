"""One-shot JSON/stdin process boundary used by the real hardware Agent."""

from __future__ import annotations

import json
import sys

from .pipeline import MAX_PAYLOAD_BYTES, WorkloadError, execute

# Up to three input artifacts plus a small request envelope.
MAX_REQUEST_BYTES = 3 * MAX_PAYLOAD_BYTES + 4096


def main() -> int:
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            raise WorkloadError("worker request is too large")
        request = json.loads(raw)
        if not isinstance(request, dict) or set(request) != {
            "task_type",
            "inputs",
            "seed",
        }:
            raise WorkloadError(
                "request must contain exactly task_type, inputs, and seed"
            )
        result = execute(request["task_type"], request["inputs"], request["seed"])
        sys.stdout.write(
            json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        )
        return 0
    except (WorkloadError, ValueError, TypeError, UnicodeError, RecursionError) as exc:
        print(f"hardware workload failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
