"""Build and probe the standalone occupancy-inflation CUDA helper on the target.

AGX Orin / JetPack 7.2.1 / CUDA 13.2:
    python -m scripts.build_cuda_smoke
Build without claiming runtime verification:
    python -m scripts.build_cuda_smoke --compile-only

An adjacent .manifest.json binds the binary to the current inflate.cu. Build and
probe failures exit nonzero with diagnostics on stderr; success emits JSON only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

from examples.mixed_workloads.native import (
    DEFAULT_ARCH,
    DEFAULT_BINARY,
    MANIFEST_SCHEMA,
    SOURCE_PATH,
    SOURCE_RELATIVE,
    NativeCudaError,
    _device_ordinal,
    file_sha256,
    manifest_path,
    probe_cuda,
    validate_arch,
)

DEFAULT_NVCC = "/usr/local/cuda/bin/nvcc"


def _compiler_run(command: list[str], *, timeout: int) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    # These nvcc variables silently inject flags not present in the build manifest.
    environment.pop("NVCC_PREPEND_FLAGS", None)
    environment.pop("NVCC_APPEND_FLAGS", None)
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            shell=False,
            timeout=timeout,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise NativeCudaError(
            f"Could not run CUDA compiler {command[0]}: {error}"
        ) from error
    if result.returncode:
        diagnostic = (result.stderr or result.stdout).strip()[:8192]
        raise NativeCudaError(
            f"CUDA compiler failed (exit {result.returncode}): {diagnostic}"
        )
    return result


def build_cuda_smoke(
    nvcc=DEFAULT_NVCC,
    output=DEFAULT_BINARY,
    *,
    arch=DEFAULT_ARCH,
    compile_only=False,
    device="cuda:0",
) -> dict:
    arch = validate_arch(arch)
    _device_ordinal(device)
    # Preserve a leading './': --nvcc ./nvcc must not turn into a PATH lookup.
    compiler_name = os.path.expanduser(os.fspath(nvcc))
    resolved_compiler = shutil.which(compiler_name)
    if resolved_compiler is None:
        raise NativeCudaError(
            f"CUDA compiler not found or not executable: {compiler_name}. "
            "Build on the AGX Orin with the CUDA toolkit installed, or pass --nvcc /path/to/nvcc. "
            "No binary was built and no GPU execution was verified."
        )
    compiler = Path(resolved_compiler).resolve()
    version_result = _compiler_run([str(compiler), "--version"], timeout=15)
    version = (version_result.stdout + version_result.stderr).strip()
    if not version or len(version) > 8192:
        raise NativeCudaError(
            "CUDA compiler returned missing or oversized version metadata"
        )
    binary = Path(output).expanduser().resolve()
    if binary in (SOURCE_PATH, Path(__file__).resolve(), compiler) or binary.suffix in (
        ".py",
        ".cu",
    ):
        raise ValueError(
            "output must be a binary artifact path, not a source file or compiler"
        )
    binary.parent.mkdir(parents=True, exist_ok=True)
    source = SOURCE_PATH.read_bytes()
    source_sha = hashlib.sha256(source).hexdigest()
    flags = ["-O2", "-std=c++17", f"-arch={arch}"]
    # Compile a private snapshot: concurrent source edits cannot change nvcc's input.
    # Publishing the binary and then manifest makes concurrent readers fail closed.
    with tempfile.TemporaryDirectory(
        prefix=".inflate-build-", dir=binary.parent
    ) as staging:
        staging_path = Path(staging)
        snapshot = staging_path / "inflate.cu"
        snapshot.write_bytes(source)
        candidate = staging_path / "inflate_cuda"
        _compiler_run(
            [str(compiler), *flags, str(snapshot), "-o", str(candidate)], timeout=180
        )
        if (
            not candidate.is_file()
            or candidate.stat().st_size == 0
            or not os.access(candidate, os.X_OK)
        ):
            raise NativeCudaError(
                "nvcc exited successfully without producing a nonempty executable"
            )
        if file_sha256(SOURCE_PATH) != source_sha:
            raise NativeCudaError(
                "inflate.cu changed during compilation; rerun the build"
            )
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "source": SOURCE_RELATIVE,
            "source_sha256": source_sha,
            "binary_name": binary.name,
            "binary_sha256": file_sha256(candidate),
            "arch": arch,
            "flags": flags,
            "compiler": {
                "path": str(compiler),
                "version": version,
                "sha256": file_sha256(compiler),
            },
            "build_host": {"system": platform.system(), "machine": platform.machine()},
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
        staged_manifest = staging_path / "manifest.json"
        staged_manifest.write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.replace(candidate, binary)
        os.replace(staged_manifest, manifest_path(binary))
    result = {
        "compiled": True,
        "runtime_verified": False,
        "binary": str(binary),
        "manifest": str(manifest_path(binary)),
        "build": manifest,
        "gpu_info": None,
        "measurement": None,
    }
    if not compile_only:
        try:
            info = probe_cuda(binary, device=device)
        except NativeCudaError as error:
            raise NativeCudaError(
                f"Compiled {binary}, but CUDA runtime probe failed: {error}. "
                "Check NVIDIA driver/device access and that --arch matches the GPU; "
                "--compile-only skips runtime verification explicitly."
            ) from error
        result["measurement"] = info["measurement"]
        result["gpu_info"] = {
            key: value for key, value in info.items() if key != "measurement"
        }
        result["runtime_verified"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nvcc", default=DEFAULT_NVCC, help="CUDA compiler (default: %(default)s)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_BINARY,
        help="Binary destination (default: %(default)s)",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile without claiming GPU runtime verification",
    )
    parser.add_argument(
        "--arch",
        default=DEFAULT_ARCH,
        help="CUDA target architecture; override for CI (default: %(default)s)",
    )
    parser.add_argument(
        "--device", default="cuda:0", help="CUDA device to probe (default: %(default)s)"
    )
    args = parser.parse_args()
    try:
        result = build_cuda_smoke(**vars(args))
    except (NativeCudaError, OSError, ValueError) as error:
        print(f"CUDA smoke build/probe failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
