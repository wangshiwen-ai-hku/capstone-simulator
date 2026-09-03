"""Check a CUDA ML environment and install VLA dependencies without replacing Torch.

Run this in a dedicated virtual environment, not the MARS gRPC environment.
Without --install this command only checks the installed inference stack.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def torch_constraints(torch_version: str, vision_version: str) -> str:
    for name, version, lower, upper in (
        ("torch", torch_version, (2, 2), (2, 11)),
        ("torchvision", vision_version, (0, 21), (0, 26)),
    ):
        match = re.match(r"^(\d+)\.(\d+)(?:\.\d+)?[a-zA-Z0-9.+_-]*$", version)
        if not match or not lower <= tuple(map(int, match.groups())) < upper:
            raise ValueError(
                f"{name} {version!r} is outside LeRobot 0.4.4's supported range"
            )
    return f"torch==={torch_version}\ntorchvision==={vision_version}\n"


def check_cuda() -> dict:
    import torch
    import torchvision

    if not torch.cuda.is_available() or not torch.version.cuda:
        raise ValueError(
            "CUDA PyTorch is unavailable; install the build matching JetPack before continuing"
        )
    # Verify both kernels and the Torch/TorchVision compiled-extension pairing.
    device = torch.device("cuda:0")
    value = torch.ones((16, 16), device=device) @ torch.ones((16, 16), device=device)
    if not torch.all(value == 16).item():
        raise ValueError("CUDA matrix check failed")
    torchvision.ops.nms(
        torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=device),
        torch.tensor([1.0], device=device),
        0.5,
    )
    torch.cuda.synchronize()
    versions = {
        name: importlib.metadata.version(name) for name in ("torch", "torchvision")
    }
    torch_constraints(versions["torch"], versions["torchvision"])
    return {
        **versions,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        parser.error(
            "activate a dedicated VLA virtual environment first; system Python is not modified"
        )
    # Prevent the known protobuf conflict even if this happens to have working CUDA.
    try:
        grpc_version = importlib.metadata.version("grpcio")
        protobuf_version = importlib.metadata.version("protobuf")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        if int(protobuf_version.split(".")[0]) >= 7:
            parser.error(
                f"this environment has MARS-compatible grpcio {grpc_version}/protobuf {protobuf_version}; "
                "create a separate .venv-vla for LeRobot"
            )
    before = check_cuda()
    if args.install:
        with tempfile.TemporaryDirectory(prefix="mars-vla-install-") as directory:
            constraints = Path(directory) / "torch-constraints.txt"
            constraints.write_text(
                torch_constraints(before["torch"], before["torchvision"])
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--constraint",
                    str(constraints),
                    "-r",
                    str(
                        Path(__file__).resolve().parents[1]
                        / "agent/requirements-vla.txt"
                    ),
                ],
                check=True,
            )
    after = check_cuda()
    if before != after:
        raise RuntimeError("Torch/CUDA changed while installing VLA dependencies")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: F401
    from lerobot.policies.factory import make_pre_post_processors  # noqa: F401

    lerobot_version = importlib.metadata.version("lerobot")
    transformers_version = importlib.metadata.version("transformers")
    if (lerobot_version, transformers_version) != ("0.4.4", "4.57.1"):
        raise ValueError(
            "expected lerobot 0.4.4 and transformers 4.57.1; run with --install"
        )
    print(
        json.dumps(
            {
                "status": "ready",
                **after,
                "lerobot": lerobot_version,
                "transformers": transformers_version,
                "worker_python": sys.executable,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
