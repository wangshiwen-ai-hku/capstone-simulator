"""Host measurements for CPU hardware tests; unavailable sensors stay explicit."""

from __future__ import annotations

import asyncio
import os
import platform
import socket
from time import monotonic

import psutil

from interfaces.proto.mars.v1 import topology_pb2


class _CpuSampleNotReady(RuntimeError):
    pass


class HostTelemetry:
    MIN_SAMPLE_SECONDS = 0.1
    WARMUP_TIMEOUT_SECONDS = 5.0

    def __init__(self) -> None:
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
        return {
            "hostname": socket.gethostname(),
            "architecture": platform.machine(),
            "platform": platform.platform(),
            "agent_pid": os.getpid(),
            "cpu_count": psutil.cpu_count() or 1,
            "memory_bytes": psutil.virtual_memory().total,
            "measured": ["cpu_utilization", "memory_utilization", "task_elapsed_ms"],
            "unavailable": ["gpu_utilization", "temperature", "power", "energy"],
        }


def detected_node(kind: str) -> dict:
    return {
        "kind": kind,
        "architecture": platform.machine(),
        "cpu_capacity": float(psutil.cpu_count() or 1),
        "gpu_capacity": 0.0,
        "memory_gb": psutil.virtual_memory().total / 1_000_000_000,
        # Link capacity is an initial planning assumption, not a measurement.
        "bandwidth_mbps": 100.0,
        "base_latency_ms": 0.0,
        "safety_capable": False,
        "capabilities": ["cpu", "hil_navigation_v1"],
        "supported_models": [],
        "max_concurrency": 1,
    }
