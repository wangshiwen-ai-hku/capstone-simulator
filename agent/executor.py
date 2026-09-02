"""Business invocation boundary; the scheduler never imports workload algorithms."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Protocol

from .artifacts import MAX_ARTIFACT_BYTES, canonical_json


@dataclass(frozen=True)
class ExecutionResult:
    outputs: dict[str, dict]
    elapsed_ms: float


class WorkloadExecutor(Protocol):
    ports: dict[str, dict[str, dict[str, str]]]

    async def execute(
        self, task_type: str, inputs: dict, seed: int
    ) -> ExecutionResult: ...


class NavigationExecutor:
    """Run a fixed, bundled worker in a killable subprocess, not the RPC loop.

    There is no arbitrary module/shell/command selection in dispatch messages.
    Each invocation has its own process, so cancellation kills actual work.
    """

    def __init__(self) -> None:
        from examples.hardware_workloads import PORT_TYPES

        self.ports = PORT_TYPES

    async def execute(self, task_type: str, inputs: dict, seed: int) -> ExecutionResult:
        if task_type not in self.ports:
            raise ValueError(f"unsupported business task: {task_type}")
        request = canonical_json(
            {"task_type": task_type, "inputs": inputs, "seed": seed}
        )
        if len(request) > 4 * MAX_ARTIFACT_BYTES:
            raise ValueError("business inputs exceed the invocation limit")
        started = perf_counter()
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "examples.hardware_workloads.worker",
            cwd=Path(__file__).resolve().parents[1],
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await process.communicate(request)
        except BaseException:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        elapsed_ms = (perf_counter() - started) * 1000
        if process.returncode:
            # Leave room for the prefix inside the Agent's 2,000-character
            # error limit, retaining the worker's final diagnostic line.
            detail = stderr.decode("utf-8", errors="replace")[-1800:]
            raise ValueError(f"business worker failed ({process.returncode}): {detail}")
        if len(stdout) > 4 * MAX_ARTIFACT_BYTES:
            raise ValueError("business outputs exceed the invocation limit")
        outputs = json.loads(stdout)
        expected = self.ports[task_type]["outputs"]
        if not isinstance(outputs, dict) or set(outputs) != set(expected):
            raise ValueError("business output ports do not match the contract")
        if any(not isinstance(value, dict) for value in outputs.values()):
            raise ValueError("business output payloads must be JSON objects")
        # Reject NaN/Infinity from a malformed worker before producing artifacts.
        canonical_json(outputs)
        return ExecutionResult(outputs, elapsed_ms)
