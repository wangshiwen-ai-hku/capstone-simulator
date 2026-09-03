"""One-shot JSON worker; GPU dependencies may live in a separate Python env."""

from __future__ import annotations

import contextlib
import json
import os
import sys

from .pipeline import MAX_PAYLOAD_BYTES, WorkloadError, execute

MAX_REQUEST_BYTES = 2 * MAX_PAYLOAD_BYTES + 8192
MAX_RESPONSE_BYTES = MAX_PAYLOAD_BYTES + 1024


def main() -> int:
    # These are fixed policy safeguards, not request-configurable download flags.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            raise WorkloadError("worker request is too large")
        request = json.loads(raw)
        if not isinstance(request, dict) or set(request) != {
            "task_type",
            "inputs",
            "seed",
            "options",
        }:
            raise WorkloadError(
                "request must contain exactly task_type, inputs, seed, and options"
            )
        # LeRobot and its loaders can print progress; stdout remains one JSON value.
        with contextlib.redirect_stdout(sys.stderr):
            result = execute(
                request["task_type"],
                request["inputs"],
                request["seed"],
                request["options"],
            )
        encoded = json.dumps(
            result, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if len(encoded.encode("utf-8")) > MAX_RESPONSE_BYTES:
            raise WorkloadError("worker response is too large")
        sys.stdout.write(encoded + "\n")
        return 0
    except Exception as exc:
        # Fail closed on missing CUDA, missing assets, and all model/decode errors.
        print(f"GPU/VLA workload failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
