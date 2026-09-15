"""Bounded one-shot worker. CUDA paths are set only by the local Agent operator."""

from __future__ import annotations

import json
import sys

from .pipeline import execute


def main() -> int:
    try:
        limit = 8 * 1024 * 1024
        raw = sys.stdin.buffer.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("mixed worker request exceeds 8 MiB")
        request = json.loads(raw)
        if (
            not isinstance(request, dict)
            or set(request) - {"task_type", "inputs", "seed", "options"}
            or not {"task_type", "inputs", "seed"}.issubset(request)
        ):
            raise ValueError("invalid mixed worker request")
        options = request.get("options", {})
        if not isinstance(options, dict):
            raise ValueError("worker options must be an object")
        if request["task_type"] == "probe":
            from .native import probe_cuda

            result = {
                "gpu_info": probe_cuda(
                    options["cuda_binary"], device=options.get("device", "cuda:0")
                )
            }
        else:
            result = execute(
                request["task_type"],
                request["inputs"],
                request["seed"],
                options=options,
            )
        print(
            json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
        return 0
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        RuntimeError,
        RecursionError,
    ) as exc:
        print(f"mixed workload failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
