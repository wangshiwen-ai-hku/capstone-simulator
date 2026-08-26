"""Command-line entry point for one configured mock agent."""

from __future__ import annotations

import argparse
import asyncio

from .service import load_agent_configs, start_agent_server


async def run(config_path: str, agent_id: str) -> None:
    configs = load_agent_configs(config_path)
    config = next(
        (item for item in configs if item.agent_id == agent_id),
        None,
    )
    if config is None:
        raise ValueError(f"agent config not found: {agent_id}")
    server, _ = await start_agent_server(config)
    print(f"mock agent {agent_id} listening on {config.listen}", flush=True)
    await server.wait_for_termination()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--agent-id", required=True)
    args = parser.parse_args()
    asyncio.run(run(args.config, args.agent_id))


if __name__ == "__main__":
    main()
