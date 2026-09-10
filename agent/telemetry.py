"""Host measurements for hardware tests; unavailable sensors stay explicit."""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
import platform
import socket
import subprocess
from time import monotonic

import psutil

from interfaces.proto.mars.v1 import topology_pb2


def _read_identity_file(path: str) -> str | None:
    try:
        return Path(path).read_text().replace("\x00", "").strip() or None
    except (OSError, UnicodeError):
        return None


def _runtime_identity() -> dict:
    """Record deployment identity once, outside timed task execution.

    Only a hash of the OS machine identifier is exposed. These local reports
    support trusted-LAN deployment checks, not cryptographic attestation.
    """
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for directory in ("agent", "examples", "interfaces", "mars", "scripts"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".cu", ".proto"}:
                digest.update(path.relative_to(root).as_posix().encode() + b"\0")
                content = path.read_bytes()
                digest.update(len(content).to_bytes(8, "big"))
                digest.update(content)
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    machine_id = _read_identity_file("/etc/machine-id")
    return {
        "machine_id_sha256": hashlib.sha256(machine_id.encode()).hexdigest()
        if machine_id
        else None,
        "runtime_source_sha256": digest.hexdigest(),
        "git_revision": revision,
        "jetson_model": _read_identity_file("/proc/device-tree/model"),
        "jetson_linux": _read_identity_file("/etc/nv_tegra_release"),
        "python_version": platform.python_version(),
    }


class _CpuSampleNotReady(RuntimeError):
    pass


class HostTelemetry:
    MIN_SAMPLE_SECONDS = 0.1
    WARMUP_TIMEOUT_SECONDS = 5.0

    def __init__(self, *, gpu_info: dict | None = None) -> None:
        self.gpu_info = _checked_gpu_info(gpu_info) if gpu_info is not None else None
        self.runtime_identity = _runtime_identity()
        self.started = monotonic()
        self._baseline_at = self.started
        self._baseline_cpu = self._cpu_counters()
        self._cached: dict | None = None

    @staticmethod
    def _cpu_counters() -> tuple[float, float]:
        times = psutil.cpu_times()
        # Linux guest counters are already included in user/nice. I/O wait is
        # idle time, matching psutil's host utilization convention.
        total = (
            sum(times) - getattr(times, "guest", 0) - getattr(times, "guest_nice", 0)
        )
        idle = times.idle + getattr(times, "iowait", 0)
        return total, idle

    async def warmup(self) -> None:
        """Collect a genuine initial interval before advertising host state."""
        deadline = monotonic() + self.WARMUP_TIMEOUT_SECONDS
        while self._cached is None:
            remaining = self.MIN_SAMPLE_SECONDS - (monotonic() - self._baseline_at)
            if remaining > 0:
                await asyncio.sleep(remaining)
            try:
                self.observe()
            except _CpuSampleNotReady:
                # Some operating systems refresh their counters more slowly
                # than the minimum window. Wait for real progress, not fake 0%.
                if monotonic() >= deadline:
                    raise RuntimeError(
                        "host CPU counters did not advance during warmup"
                    )
                await asyncio.sleep(self.MIN_SAMPLE_SECONDS)

    def observe(self) -> dict:
        """Non-blocking host observations, not process-attributed consumption.

        Independent raw-counter baselines avoid psutil.cpu_percent's shared
        state. Calls less than 100 ms apart reuse the last complete observation
        with its original sample/window times, never a pretend fresh idle value.
        Timestamps use this Agent's clock and cannot be compared between hosts.
        """

        sampled_at = monotonic()
        window_seconds = sampled_at - self._baseline_at
        if window_seconds < self.MIN_SAMPLE_SECONDS:
            if self._cached is None:
                raise _CpuSampleNotReady("host telemetry requires initial warmup")
            return dict(self._cached)
        total, idle = self._cpu_counters()
        total_delta = total - self._baseline_cpu[0]
        idle_delta = idle - self._baseline_cpu[1]
        if total_delta == 0 and idle_delta == 0:
            if self._cached is not None:
                return dict(self._cached)
            raise _CpuSampleNotReady("host CPU counters have not advanced")
        if total_delta < 0 or idle_delta < 0 or idle_delta > total_delta + 1e-6:
            raise RuntimeError("host CPU counters did not yield a valid sample")
        cpu_ratio = (total_delta - min(idle_delta, total_delta)) / total_delta
        memory = psutil.virtual_memory()
        observation = {
            "scope": "host",
            "clock": "agent_monotonic_elapsed",
            "sampled_at_ms": (sampled_at - self.started) * 1000,
            "cpu_sample_window_start_ms": (self._baseline_at - self.started) * 1000,
            "cpu_sample_window_ms": window_seconds * 1000,
            "cpu_utilization_ratio": cpu_ratio,
            "memory_utilization_ratio": memory.percent / 100.0,
            "memory_total_bytes": memory.total,
            "memory_available_bytes": memory.available,
        }
        self._baseline_at = sampled_at
        self._baseline_cpu = total, idle
        self._cached = observation
        return observation

    def sample(self, agent_id: str, sequence: int, active: int):
        observation = self.observe()
        return topology_pb2.NodeSnapshot(
            node_id=agent_id,
            cpu_utilization_ratio=observation["cpu_utilization_ratio"],
            memory_utilization_ratio=observation["memory_utilization_ratio"],
            # v1 has no presence for these scalars. Zero is a compatibility
            # placeholder; diagnostics explicitly mark them unavailable.
            gpu_utilization_ratio=0.0,
            temperature_celsius=0.0,
            power_watts=0.0,
            network_latency_ms=0.0,
            online=True,
            sampled_at_ms=observation["sampled_at_ms"],
            snapshot_sequence=sequence,
            active_task_count=active,
        )

    def identity(self) -> dict:
        identity = {
            **self.runtime_identity,
            "hostname": socket.gethostname(),
            "architecture": platform.machine(),
            "platform": platform.platform(),
            "agent_pid": os.getpid(),
            "cpu_count": psutil.cpu_count() or 1,
            "memory_bytes": psutil.virtual_memory().total,
            "measured": ["cpu_utilization", "memory_utilization", "task_elapsed_ms"],
            "unavailable": ["gpu_utilization", "temperature", "power", "energy"],
        }
        if self.gpu_info is not None:
            # Preflight identity is not a live utilization or energy reading.
            identity["cuda_device"] = dict(self.gpu_info)
        return identity


def _checked_gpu_info(gpu_info: dict) -> dict:
    """Validate metadata supplied by the caller's successful CUDA preflight.

    This module never imports torch. Device execution must be checked before
    calling detected_node; a machine name or a configured GPU count is not
    sufficient evidence to advertise a CUDA execution node.
    """

    device = gpu_info.get("device")
    count = gpu_info.get("device_count")
    capability = gpu_info.get("compute_capability")
    backend = gpu_info.get("backend", "torch")
    if backend == "cuda_runtime":
        backend_valid = (
            gpu_info.get("kernel_execution_verified") is True
            and type(gpu_info.get("cuda_runtime_version")) is int
            and gpu_info["cuda_runtime_version"] > 0
            and type(gpu_info.get("cuda_driver_version")) is int
            and gpu_info["cuda_driver_version"] > 0
        )
    elif backend == "torch":
        backend_valid = isinstance(gpu_info.get("torch_version"), str) and bool(
            gpu_info["torch_version"].strip()
        )
    else:
        backend_valid = False
    if (
        gpu_info.get("available") is not True
        or not isinstance(device, str)
        or not device.startswith("cuda:")
        or not device[5:].isdigit()
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or int(device[5:]) >= count
        or not isinstance(gpu_info.get("device_name"), str)
        or not gpu_info["device_name"].strip()
        or not backend_valid
        or not isinstance(capability, (tuple, list))
        or len(capability) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in capability
        )
        or capability[0] < 1
        or capability[1] < 0
    ):
        raise ValueError("GPU metadata must come from a successful CUDA preflight")
    checked = {
        "available": True,
        "device": device,
        "device_count": count,
        "device_name": gpu_info["device_name"],
        "compute_capability": list(capability),
    }
    if backend == "torch":
        checked["torch_version"] = gpu_info["torch_version"]
    else:
        checked.update(
            {
                name: gpu_info[name]
                for name in (
                    "backend",
                    "kernel_execution_verified",
                    "cuda_runtime_version",
                    "cuda_driver_version",
                )
            }
        )
        for name in ("source_sha256", "binary_sha256"):
            if name in gpu_info:
                checked[name] = gpu_info[name]
    return checked


def detected_node(
    kind: str,
    *,
    gpu_info: dict | None = None,
    capabilities: list[str] | None = None,
    supported_models: list[str] | None = None,
) -> dict:
    """Describe this host, optionally using a verified CUDA device.

    GPU capacity is one exclusive worker slot on the selected device, including
    on hosts with multiple GPUs. Custom workload capabilities replace the
    navigation default; cpu/cuda capabilities reflect actual host support.
    """

    checked_gpu = _checked_gpu_info(gpu_info) if gpu_info is not None else None
    workload_capabilities = (
        ["hil_navigation_v1"] if capabilities is None else list(capabilities)
    )
    if "cuda" in workload_capabilities and checked_gpu is None:
        raise ValueError("cuda capability requires a successful CUDA preflight")
    node = {
        "kind": kind,
        "architecture": platform.machine(),
        "cpu_capacity": float(psutil.cpu_count() or 1),
        "gpu_capacity": 1.0 if checked_gpu is not None else 0.0,
        "memory_gb": psutil.virtual_memory().total / 1_000_000_000,
        # Link capacity is an initial planning assumption, not a measurement.
        "bandwidth_mbps": 100.0,
        "base_latency_ms": 0.0,
        "safety_capable": False,
        "capabilities": list(
            dict.fromkeys(
                ["cpu"]
                + (["cuda"] if checked_gpu is not None else [])
                + workload_capabilities
            )
        ),
        "supported_models": list(supported_models or []),
        "max_concurrency": 1,
    }
    if checked_gpu is not None:
        node["gpu_info"] = checked_gpu
    return node
