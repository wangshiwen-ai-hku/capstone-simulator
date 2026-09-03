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
        self.worker_python = sys.executable
        self.worker_module = "examples.hardware_workloads.worker"
        self.options: dict = {}

    async def execute(self, task_type: str, inputs: dict, seed: int) -> ExecutionResult:
        if task_type not in self.ports:
            raise ValueError(f"unsupported business task: {task_type}")
        request = canonical_json(
            {
                "task_type": task_type,
                "inputs": inputs,
                "seed": seed,
                **({"options": self.options} if self.options else {}),
            }
        )
        if len(request) > 4 * MAX_ARTIFACT_BYTES:
            raise ValueError("business inputs exceed the invocation limit")
        started = perf_counter()
        process = await asyncio.create_subprocess_exec(
            self.worker_python,
            "-m",
            self.worker_module,
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


class VlaExecutor(NavigationExecutor):
    """Fixed VLA worker; its optional ML environment is separate from gRPC.

    Paths and interpreter are selected by the local operator at Agent startup,
    never by a remote dispatch. CPU input/validation roles need no ML packages.
    """

    def __init__(
        self,
        role: str,
        *,
        worker_python: str | None = None,
        observation_file: str | Path | None = None,
        model_dir: str | Path | None = None,
        device: str = "cuda:0",
        repeats: int = 3,
    ) -> None:
        from examples.vla_workloads import PORT_TYPES

        if role not in {"io", "cuda"}:
            raise ValueError("VLA executor role must be io or cuda")
        if not 1 <= repeats <= 20:
            raise ValueError("VLA repeats must be between 1 and 20")
        tasks = (
            {"hil_vla_observe", "hil_vla_validate", "hil_cuda_validate"}
            if role == "io"
            else {"hil_cuda_smoke"}
            | ({"hil_vla_infer"} if model_dir is not None else set())
        )
        self.ports = {name: PORT_TYPES[name] for name in tasks}
        self.gpu_demands = {name: 1.0 for name in tasks} if role == "cuda" else {}
        # Keep a virtualenv's python symlink intact. Resolving it to the base
        # interpreter silently discards that environment's ML dependencies.
        self.worker_python = (
            str(Path(worker_python).expanduser().absolute())
            if worker_python
            else sys.executable
        )
        self.worker_module = "examples.vla_workloads.worker"
        self.options = {"device": device, "repeats": repeats}
        for key, value in (
            ("observation_file", observation_file),
            ("model_dir", model_dir),
        ):
            if value is not None:
                path = Path(value).resolve()
                if not path.exists():
                    raise ValueError(f"{key} does not exist: {path}")
                self.options[key] = str(path)

    async def probe_cuda(self) -> dict:
        """Check the exact worker interpreter/device before advertising CUDA."""
        process = await asyncio.create_subprocess_exec(
            self.worker_python,
            "-m",
            self.worker_module,
            cwd=Path(__file__).resolve().parents[1],
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(
                    canonical_json(
                        {
                            "task_type": "probe",
                            "inputs": {},
                            "seed": 0,
                            "options": {"device": self.options["device"]},
                        }
                    )
                ),
                timeout=60,
            )
        except BaseException:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        if process.returncode:
            raise ValueError(
                "CUDA preflight failed: " + stderr.decode(errors="replace")[-1800:]
            )
        if len(stdout) > 65536:
            raise ValueError("CUDA preflight returned oversized metadata")
        info = json.loads(stdout)["gpu_info"]
        if not isinstance(info, dict) or info.get("available") is not True:
            raise ValueError("worker did not verify a CUDA device")
        return info
