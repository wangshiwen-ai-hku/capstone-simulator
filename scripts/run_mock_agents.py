"""Start the three localhost mock-agent processes."""

# ruff: noqa: E402 -- direct script execution bootstraps repository imports.

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    root_path = str(ROOT)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)

from agent.service import load_agent_configs


CONFIG = ROOT / "configs" / "mars" / "agents.local.json"


def main() -> None:
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "agent.main",
                "--config",
                str(CONFIG),
                "--agent-id",
                config.agent_id,
            ],
            cwd=ROOT,
        )
        for config in load_agent_configs(CONFIG)
    ]
    try:
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
        for process in processes:
            process.wait()


if __name__ == "__main__":
    main()
