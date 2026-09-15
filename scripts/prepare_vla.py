"""Prepare pinned SmolVLA weights and a real dataset frame in the ML environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    model = commands.add_parser(
        "model", help="Download and verify the pinned offline model bundle"
    )
    model.add_argument("--output", type=Path, required=True)
    sample = commands.add_parser(
        "sample", help="Export a recorded SO100 observation (~470 MB source download)"
    )
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--cache", type=Path, default=Path(".mars-vla/datasets"))
    args = parser.parse_args()
    if args.command == "model":
        from examples.vla_workloads.bundle import download_bundle

        manifest = download_bundle(args.output)
        result = {
            "model_dir": str(args.output.resolve()),
            "policy": manifest["policy"],
            "files": len(manifest["files"]),
        }
    else:
        from examples.vla_workloads.preparation import export_sample

        result = export_sample(args.output, args.cache)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
