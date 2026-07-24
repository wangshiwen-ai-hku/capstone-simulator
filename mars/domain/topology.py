"""Static topology declarations and time-specific resource observations."""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass


class NodeKind(str, enum.Enum):
    ROBOT = "robot"
    EDGE = "edge"
    CLOUD = "cloud"


@dataclass(frozen=True)
class NodeSpec:
    """Static node identity and declared execution capacity."""

    node_id: str
    kind: NodeKind
    cpu_capacity: float
    gpu_capacity: float
    memory_gb: float
    bandwidth_mbps: float
    base_latency_ms: float
    architecture: str = "generic"
    battery_capacity_wh: float | None = None
    safety_capable: bool = True
    capabilities: tuple[str, ...] = ()
    supported_models: tuple[str, ...] = ()
    max_concurrency: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capabilities",
            tuple(self.capabilities),
        )
        object.__setattr__(
            self,
            "supported_models",
            tuple(self.supported_models),
        )
        if not self.node_id.strip():
            raise ValueError("node_id must be non-blank")
        capacities = (
            self.cpu_capacity,
            self.gpu_capacity,
            self.memory_gb,
            self.bandwidth_mbps,
            self.base_latency_ms,
        )
        if not all(math.isfinite(value) for value in capacities):
            raise ValueError("node capacity values must be finite")
        if (
            self.cpu_capacity <= 0
            or self.gpu_capacity < 0
            or self.memory_gb <= 0
            or self.bandwidth_mbps <= 0
            or self.base_latency_ms < 0
        ):
            raise ValueError("node capacities are outside valid ranges")
        if self.battery_capacity_wh is not None and (
            not math.isfinite(self.battery_capacity_wh) or self.battery_capacity_wh <= 0
        ):
            raise ValueError("battery_capacity_wh must be positive when provided")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least one")


@dataclass(frozen=True)
class NodeSnapshot:
    """Dynamic resource and health state reported for a registered node."""

    node_id: str
    cpu_util: float = 0.0
    gpu_util: float = 0.0
    memory_util: float = 0.0
    temperature_c: float = 0.0
    power_w: float = 0.0
    network_latency_ms: float = 0.0
    online: bool = True

    def __post_init__(self) -> None:
        if not self.node_id.strip():
            raise ValueError("snapshot node_id must be non-blank")
        values = (
            self.cpu_util,
            self.gpu_util,
            self.memory_util,
            self.temperature_c,
            self.power_w,
            self.network_latency_ms,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("node snapshot values must be finite")
        if not all(
            0.0 <= value <= 1.0
            for value in (
                self.cpu_util,
                self.gpu_util,
                self.memory_util,
            )
        ):
            raise ValueError("node utilization values must be in [0, 1]")
        if self.power_w < 0 or self.network_latency_ms < 0:
            raise ValueError("node power and network latency must be non-negative")


@dataclass(frozen=True)
class LinkSpec:
    """Static declaration for one directed network link."""

    link_id: str
    source_node_id: str
    target_node_id: str
    bandwidth_mbps: float
    base_latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.link_id.strip():
            raise ValueError("link_id must be non-blank")
        if not self.source_node_id.strip() or not self.target_node_id.strip():
            raise ValueError("link endpoints must be non-blank")
        if self.source_node_id == self.target_node_id:
            raise ValueError("network links must connect two different nodes")
        if not math.isfinite(self.bandwidth_mbps):
            raise ValueError("link bandwidth_mbps must be finite")
        if self.bandwidth_mbps <= 0:
            raise ValueError("link bandwidth_mbps must be positive")
        if not math.isfinite(self.base_latency_ms) or self.base_latency_ms < 0:
            raise ValueError("link base_latency_ms must be non-negative")


@dataclass(frozen=True)
class LinkSnapshot:
    """Dynamic state for a directed link."""

    link_id: str
    available_bandwidth_mbps: float
    latency_ms: float = 0.0
    jitter_ms: float = 0.0
    packet_loss_rate: float = 0.0
    online: bool = True

    def __post_init__(self) -> None:
        if not self.link_id.strip():
            raise ValueError("link_id must be non-blank")
        values = (
            self.available_bandwidth_mbps,
            self.latency_ms,
            self.jitter_ms,
            self.packet_loss_rate,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("link snapshot values must be finite")
        if self.available_bandwidth_mbps < 0:
            raise ValueError("link available_bandwidth_mbps must be non-negative")
        if self.latency_ms < 0 or self.jitter_ms < 0:
            raise ValueError("link latency and jitter must be non-negative")
        if not 0.0 <= self.packet_loss_rate < 1.0:
            raise ValueError("link packet_loss_rate must be in [0, 1)")


__all__ = [
    "LinkSnapshot",
    "LinkSpec",
    "NodeKind",
    "NodeSnapshot",
    "NodeSpec",
]
