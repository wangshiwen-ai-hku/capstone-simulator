"""Check a CUDA ML environment and install VLA dependencies without replacing Torch.

Run this in a dedicated virtual environment, not the MARS gRPC environment.
Without --install this command only checks the installed inference stack.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import site
import subprocess
import sys
import tempfile

from examples.vla_workloads.versions import (
    release_tuple,
    validate_vla_framework_versions,
)


def torch_constraints(torch_version: str, vision_version: str) -> str:
    validate_vla_framework_versions(torch_version, vision_version)
    return f"torch==={torch_version}\ntorchvision==={vision_version}\n"


def _compute_capability(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"([0-9]+)\.([0-9]+)", value)
    if match is None:
        raise argparse.ArgumentTypeError("compute capability must look like 8.7")
    return tuple(map(int, match.groups()))


def _inside_environment(path: str | Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(sys.prefix).resolve())
    except (OSError, ValueError):
        return False
    return True


def _installed_framework_version(module, distribution_name: str) -> str:
    module_file = getattr(module, "__file__", None)
    distribution = importlib.metadata.distribution(distribution_name)
    distribution_root = distribution.locate_file("")
    if not module_file or not _inside_environment(module_file) or not _inside_environment(
        distribution_root
    ):
        raise ValueError(
            f"{distribution_name} was imported outside the active VLA environment"
        )
    imported = str(getattr(module, "__version__", ""))
    installed = distribution.version
    if not imported or release_tuple(imported) != release_tuple(installed):
        raise ValueError(
            f"imported {distribution_name} {imported!r} does not match installed metadata {installed!r}"
        )
    return installed


def check_cuda(required_compute_capability: tuple[int, int] | None = None) -> dict:
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
    nms = torchvision.ops.nms(
        torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=device),
        torch.tensor([1.0], device=device),
        0.5,
    )
    torch.cuda.synchronize()
    capability = tuple(torch.cuda.get_device_capability(device))
    architectures = list(torch.cuda.get_arch_list())
    if required_compute_capability is not None:
        expected_arch = f"sm_{required_compute_capability[0]}{required_compute_capability[1]}"
        if capability != required_compute_capability:
            raise ValueError(
                f"CUDA device capability is {capability}, expected {required_compute_capability}"
            )
        if expected_arch not in architectures:
            raise ValueError(
                f"PyTorch does not contain {expected_arch}; compiled architectures: {architectures}"
            )
    if nms.detach().cpu().tolist() != [0]:
        raise ValueError("TorchVision CUDA NMS check failed")
    versions = {
        "torch": _installed_framework_version(torch, "torch"),
        "torchvision": _installed_framework_version(torchvision, "torchvision"),
    }
    torch_constraints(versions["torch"], versions["torchvision"])
    return {
        **versions,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(capability),
        "torch_arch_list": architectures,
        "python": platform.python_version(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    parser.add_argument(
        "--require-compute-capability",
        type=_compute_capability,
        help="also require the selected CUDA device and Torch build (AGX Orin: 8.7)",
    )
    parser.add_argument(
        "--require-python",
        help="also require the worker Python major.minor (JetPack 7.2.1: 3.12)",
    )
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        parser.error(
            "activate a dedicated VLA virtual environment first; system Python is not modified"
        )
    overrides = [name for name in ("PYTHONHOME", "PYTHONPATH") if os.environ.get(name)]
    if overrides or site.ENABLE_USER_SITE:
        detail = ", ".join(overrides) if overrides else "user site-packages"
        parser.error(
            f"worker package lookup is not isolated ({detail}); unset Python overrides and retry"
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
    if args.require_python and ".".join(map(str, sys.version_info[:2])) != args.require_python:
        parser.error(
            f"worker Python is {platform.python_version()}, expected {args.require_python}.x"
        )
    before = check_cuda(args.require_compute_capability)
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
    after = check_cuda(args.require_compute_capability)
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
