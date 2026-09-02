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
        from .executor import NavigationExecutor
        from .real_service import ExecutionAgentService, start_execution_server
        from .service import AgentConfig
        from .telemetry import detected_node

        config = AgentConfig(args.agent_id, args.listen, detected_node(args.kind), {})
        files = ArtifactFiles(args.artifact_dir or Path(".mars-hil") / args.agent_id)
        service = ExecutionAgentService(
            config,
            NavigationExecutor(),
            files,
            parse_endpoints(args.peer),
            task_timeout_seconds=args.task_timeout,
        )
        server, _ = await start_execution_server(service)
        description = f"REAL CPU navigation; artifacts={files.directory}"
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
    parser.add_argument("--executor", choices=("mock", "navigation"), default="mock")
    parser.add_argument("--config", help="Mock-only node/snapshot JSON")
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--kind", choices=("robot", "edge"), default="robot")
    parser.add_argument("--listen", default="127.0.0.1:50051")
    parser.add_argument("--peer", action="append", default=[], metavar="NODE=HOST:PORT")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--task-timeout", type=float, default=30.0)
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
