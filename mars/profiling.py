"""Execution-profile catalog for measured or synthetic MARS workloads."""

from __future__ import annotations

import json
from dataclasses import dataclass
import math
from pathlib import Path
from typing import TYPE_CHECKING

from .domain.task import TaskClass
from .domain.topology import NodeKind

if TYPE_CHECKING:
    from .synthetic_workloads import SyntheticWorkloadCatalog


DEFAULT_PROFILE_PATH = Path(__file__).resolve().parents[1] / "configs" / "mars" / "profiles.synthetic.json"


@dataclass(frozen=True)
class ExecutionProfile:
    task_type: str
    task_class: TaskClass
    node_kind: NodeKind
    model_variant: str
    input_shape: str
    precision: str
    batch_size: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_per_s: float
    peak_memory_mb: float
    energy_j: float
    output_size_mb: float
    failure_rate: float = 0.0
    supported: bool = True
    provenance: str = "synthetic_placeholder"
    cpu_units: float | None = None
    gpu_units: float | None = None

    def __post_init__(self) -> None:
        non_negative = (
            self.p50_ms,
            self.p95_ms,
            self.p99_ms,
            self.throughput_per_s,
            self.peak_memory_mb,
            self.energy_j,
            self.output_size_mb,
        )
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in non_negative
        ):
            raise ValueError(
                "execution profile measurements must be finite and "
                "non-negative"
            )
        if self.batch_size < 1:
            raise ValueError("execution profile batch_size must be positive")
        if not 0.0 <= self.failure_rate <= 1.0:
            raise ValueError("execution profile failure_rate must be in [0, 1]")
        if any(
            value is not None
            and (not math.isfinite(value) or value < 0.0)
            for value in (self.cpu_units, self.gpu_units)
        ):
            raise ValueError(
                "profile CPU/GPU demands must be finite and non-negative"
            )


class ProfileCatalog:
    def __init__(self, profiles: list[ExecutionProfile]) -> None:
        self.profiles = tuple(profiles)
        self._items = {(item.task_type, item.node_kind): item for item in profiles}
        sources = {item.provenance.strip() or "unknown" for item in profiles}
        self.provenance = next(iter(sources)) if len(sources) == 1 else ("mixed" if sources else "empty")

    def lookup(self, task_type: str, node_kind: NodeKind) -> ExecutionProfile | None:
        return self._items.get((task_type, node_kind))

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PROFILE_PATH) -> "ProfileCatalog":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            [
                ExecutionProfile(
                    task_type=item["task_type"],
                    task_class=TaskClass(item["task_class"]),
                    node_kind=NodeKind(item["node_kind"]),
                    model_variant=item["model_variant"],
                    input_shape=item["input_shape"],
                    precision=item["precision"],
                    batch_size=item["batch_size"],
                    p50_ms=item["p50_ms"],
                    p95_ms=item["p95_ms"],
                    p99_ms=item["p99_ms"],
                    throughput_per_s=item["throughput_per_s"],
                    peak_memory_mb=item["peak_memory_mb"],
                    energy_j=item["energy_j"],
                    output_size_mb=item["output_size_mb"],
                    failure_rate=item.get("failure_rate", 0.0),
                    supported=item.get("supported", True),
                    provenance=item.get("provenance", raw.get("provenance", "unknown")),
                    cpu_units=item.get(
                        "cpu_units",
                        item.get("resources", {}).get("cpu_cores"),
                    ),
                    gpu_units=item.get(
                        "gpu_units",
                        item.get("resources", {}).get("gpu_units"),
                    ),
                )
                for item in raw["profiles"]
            ]
        )


def profile_catalog_from_workloads(
    catalog: SyntheticWorkloadCatalog,
) -> ProfileCatalog:
    """Convert the canonical workload catalog without losing target facts."""

    from .synthetic_workloads import ExecutionTarget

    profiles: list[ExecutionProfile] = []
    for workload in catalog:
        for target in ExecutionTarget:
            profile = workload.profile_for(target)
            profiles.append(
                ExecutionProfile(
                    task_type=workload.task_type,
                    task_class=workload.task_class,
                    node_kind=(
                        NodeKind.ROBOT
                        if target is ExecutionTarget.ORIN
                        else NodeKind.EDGE
                    ),
                    model_variant=workload.model_variant,
                    input_shape="synthetic",
                    precision="synthetic",
                    batch_size=1,
                    p50_ms=profile.latency.p50_ms,
                    p95_ms=profile.latency.p95_ms,
                    p99_ms=profile.latency.p99_ms,
                    throughput_per_s=(
                        1000.0
                        * profile.max_concurrency
                        / profile.latency.p50_ms
                    ),
                    peak_memory_mb=profile.resources.memory_mb,
                    energy_j=profile.energy_j.typical,
                    output_size_mb=profile.output_size_mb.typical,
                    failure_rate=profile.failure_rate,
                    supported=profile.supported,
                    provenance="synthetic_workload_catalog",
                    cpu_units=profile.resources.cpu_cores,
                    # Accelerator demand belongs to the workload, not the
                    # target. Keep it in absolute sparse INT8 TOPS so moving
                    # a task between Jetson and edge hardware never changes
                    # the amount of work being scheduled.
                    gpu_units=workload.accelerator_demand_tops,
                )
            )
    return ProfileCatalog(profiles)


def load_default_catalog() -> ProfileCatalog | None:
    try:
        return ProfileCatalog.load()
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
