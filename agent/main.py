"""Explicit mock or real CPU Agent entry point for local/LAN validation."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
import re
import signal

from .endpoints import parse_endpoints
from .service import load_agent_configs, start_agent_server


async def run(args: argparse.Namespace) -> None:
    service = None
    if args.executor == "mock":
        if not args.config:
            raise ValueError("mock mode requires --config")
        config = next(
            (
                item
                for item in load_agent_configs(args.config)
                if item.agent_id == args.agent_id
            ),
            None,
        )
        if config is None:
            raise ValueError(f"agent config not found: {args.agent_id}")
        server, _ = await start_agent_server(config)
        description = "MOCK (no business computation)"
    else:
        if args.config:
            raise ValueError(
                "navigation mode auto-detects hardware; do not use synthetic --config"
            )
        from .artifacts import ArtifactFiles
        from .executor import NavigationExecutor, VlaExecutor
        from .real_service import ExecutionAgentService, start_execution_server
        from .service import AgentConfig
        from .telemetry import detected_node

        gpu_info = None
        if args.executor == "navigation":
            executor = NavigationExecutor()
            node = detected_node(args.kind)
            description = "REAL CPU navigation"
        else:
            role = "cuda" if args.executor == "vla-cuda" else "io"
            executor = VlaExecutor(
                role,
                worker_python=args.worker_python,
                observation_file=args.observation_file,
                model_dir=args.model_dir,
                device=args.device,
                repeats=args.inference_repeats,
            )
            capabilities = ["hil_vla_io_v1"]
            models = []
            if role == "cuda":
                capabilities = ["hil_cuda_v1"]
                if args.model_dir:
                    from examples.vla_workloads.bundle import validate_bundle

                    manifest = validate_bundle(args.model_dir)
                    capabilities.append("hil_smolvla_v1")
                    models.append(manifest["policy"]["repo_id"])
                gpu_info = await executor.probe_cuda()
            node = detected_node(
                args.kind,
                gpu_info=gpu_info,
                capabilities=capabilities,
                supported_models=models,
            )
            description = (
                "REAL CUDA VLA" if role == "cuda" else "REAL CPU VLA input/validation"
            )
        config = AgentConfig(args.agent_id, args.listen, node, {})
        files = ArtifactFiles(args.artifact_dir or Path(".mars-hil") / args.agent_id)
        service = ExecutionAgentService(
            config,
            executor,
            files,
            parse_endpoints(args.peer),
            task_timeout_seconds=args.task_timeout,
        )
        server, _ = await start_execution_server(service)
        description += f"; artifacts={files.directory}"
    print(f"{description}: {args.agent_id} listening on {config.listen}", flush=True)
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stopped.set)
    try:
        await stopped.wait()
    finally:
        if service is not None:
            await service.close()
        await server.stop(2.0)
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(signum)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--executor",
        choices=("mock", "navigation", "vla-io", "vla-cuda"),
        default="mock",
    )
    parser.add_argument("--config", help="Mock-only node/snapshot JSON")
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--kind", choices=("robot", "edge"), default="robot")
    parser.add_argument("--listen", default="127.0.0.1:50051")
    parser.add_argument("--peer", action="append", default=[], metavar="NODE=HOST:PORT")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--task-timeout", type=float, default=30.0)
    parser.add_argument(
        "--worker-python", help="Local VLA worker interpreter (separate ML environment)"
    )
    parser.add_argument(
        "--observation-file",
        type=Path,
        help="Prepared VLA observation JSON, on the IO host",
    )
    parser.add_argument(
        "--model-dir", type=Path, help="Verified local SmolVLA bundle, on the GPU host"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inference-repeats", type=int, default=3)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", args.agent_id):
        parser.error(
            "agent-id must use lowercase letters, digits, underscores or hyphens"
        )
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    try:
        asyncio.run(run(args))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"agent startup/run failed: {exc}\n")


if __name__ == "__main__":
    main()
