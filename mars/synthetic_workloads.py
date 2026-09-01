"""Synthetic workload definitions for scheduler and local-agent experiments.

Definitions contain task capabilities and target profiles without transport or
execution-framework dependencies. The coordinator and process-local agents
consume the same definitions.
"""

from __future__ import annotations

import enum
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .domain.task import DataPort, ResourceClass, TaskClass, TaskSpec


DEFAULT_SYNTHETIC_WORKLOAD_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "mars" / "workloads.synthetic.json"
)


class ExecutionTarget(str, enum.Enum):
    """Hardware roles used by the local simulated-agent environment."""

    ORIN = "orin"
    EDGE = "edge"


class UnsupportedTargetError(ValueError):
    """Raised when a workload target profile has ``supported=false``."""


@dataclass(frozen=True)
class ValueRange:
    minimum: float
    typical: float
    maximum: float

    def __post_init__(self) -> None:
        values = (self.minimum, self.typical, self.maximum)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("range values must be finite")
        if self.minimum < 0 or not self.minimum <= self.typical <= self.maximum:
            raise ValueError("range must satisfy 0 <= minimum <= typical <= maximum")

    @classmethod
    def from_dict(cls, item: Mapping[str, Any]) -> "ValueRange":
        return cls(float(item["min"]), float(item["typical"]), float(item["max"]))


@dataclass(frozen=True)
class LatencyDistribution:
    p50_ms: float
    p95_ms: float
    p99_ms: float

    def __post_init__(self) -> None:
        if self.p50_ms <= 0 or not self.p50_ms <= self.p95_ms <= self.p99_ms:
            raise ValueError("latency must satisfy 0 < p50_ms <= p95_ms <= p99_ms")


@dataclass(frozen=True)
class ResourceDemand:
    cpu_cores: float
    gpu_units: float
    memory_mb: float

    def __post_init__(self) -> None:
        if min(self.cpu_cores, self.gpu_units, self.memory_mb) < 0:
            raise ValueError("resource demand cannot be negative")


@dataclass(frozen=True)
class PortDefinition:
    name: str
    semantic_type: str

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.semantic_type.strip():
            raise ValueError("port name and semantic_type are required")


@dataclass(frozen=True)
class SyntheticRuntimeProfile:
    target: ExecutionTarget
    latency: LatencyDistribution
    resources: ResourceDemand
    input_size_mb: ValueRange
    output_size_mb: ValueRange
    energy_j: ValueRange
    failure_rate: float
    accuracy: ValueRange
    max_concurrency: int
    supported: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.failure_rate <= 1:
            raise ValueError("failure_rate must be between 0 and 1")
        if self.accuracy.maximum > 1:
            raise ValueError("accuracy values must be between 0 and 1")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least one")


@dataclass(frozen=True)
class SyntheticWorkload:
    """Task definition with target-specific execution profiles."""

    task_type: str
    display_name: str
    task_class: TaskClass
    description: str
    model_variant: str
    accelerator_demand_tops: float
    inputs: tuple[PortDefinition, ...]
    outputs: tuple[PortDefinition, ...]
    profiles: tuple[SyntheticRuntimeProfile, ...]

    def __post_init__(self) -> None:
        if not self.task_type.strip() or not self.display_name.strip():
            raise ValueError("task_type and display_name are required")
        if (
            not math.isfinite(self.accelerator_demand_tops)
            or self.accelerator_demand_tops < 0
        ):
            raise ValueError(
                "accelerator_demand_tops must be finite and non-negative"
            )
        targets = [profile.target for profile in self.profiles]
        if len(targets) != len(set(targets)):
            raise ValueError(f"duplicate target profile for {self.task_type}")
        missing = set(ExecutionTarget) - set(targets)
        if missing:
            names = ", ".join(sorted(target.value for target in missing))
            raise ValueError(f"{self.task_type} is missing target profiles: {names}")
        for label, ports in (("input", self.inputs), ("output", self.outputs)):
            names = [port.name for port in ports]
            if len(names) != len(set(names)):
                raise ValueError(f"duplicate {label} port in {self.task_type}")

    def profile_for(self, target: ExecutionTarget | str) -> SyntheticRuntimeProfile:
        resolved = ExecutionTarget(target)
        return next(profile for profile in self.profiles if profile.target is resolved)

    def to_task_spec(
        self,
        target: ExecutionTarget | str = ExecutionTarget.ORIN,
        *,
        latency_budget_ms: float | None = None,
        allow_local_fallback: bool | None = None,
    ) -> TaskSpec:
        """Convert the workload definition into a scheduler-facing TaskSpec."""
        profile = self.profile_for(target)
        dominant = (
            ResourceClass.GPU
            if self.accelerator_demand_tops > 0
            else ResourceClass.CPU
        )
        fallback = self.task_class is not TaskClass.LOCAL_SAFETY
        if allow_local_fallback is not None:
            fallback = allow_local_fallback
        return TaskSpec(
            task_type=self.task_type,
            task_class=self.task_class,
            compute_demand=max(0.1, profile.resources.cpu_cores),
            gpu_demand=self.accelerator_demand_tops,
            latency_budget_ms=latency_budget_ms or profile.latency.p95_ms * 1.25,
            model_requirement=self.model_variant,
            input_size_mb=profile.input_size_mb.typical,
            output_size_mb=profile.output_size_mb.typical,
            energy_budget_j=profile.energy_j.maximum,
            dominant_resource=dominant,
            allow_local_fallback=fallback,
            input_ports=tuple(
                DataPort(port.name, port.semantic_type) for port in self.inputs
            ),
            output_ports=tuple(
                DataPort(port.name, port.semantic_type) for port in self.outputs
            ),
        )


@dataclass(frozen=True)
class SyntheticExecution:
    task_type: str
    target: ExecutionTarget
    latency_ms: float
    cpu_cores: float
    gpu_units: float
    memory_mb: float
    input_size_mb: float
    output_size_mb: float
    energy_j: float
    accuracy: float
    failed: bool
    max_concurrency: int


class SyntheticSampler:
    """Samples repeatable simulated executions without global random state."""

    def __init__(
        self,
        catalog: "SyntheticWorkloadCatalog",
        *,
        seed: int | None = None,
        deterministic: bool = False,
    ) -> None:
        self.catalog = catalog
        self.deterministic = deterministic
        self._rng = random.Random(seed)

    def sample(self, task_type: str, target: ExecutionTarget | str) -> SyntheticExecution:
        resolved = ExecutionTarget(target)
        workload = self.catalog.get(task_type)
        profile = workload.profile_for(resolved)
        if not profile.supported:
            raise UnsupportedTargetError(f"{task_type} is not supported on {resolved.value}")

        if self.deterministic:
            latency = profile.latency.p50_ms
            input_size = profile.input_size_mb.typical
            output_size = profile.output_size_mb.typical
            energy = profile.energy_j.typical
            accuracy = profile.accuracy.typical
            failed = False
        else:
            latency = self._sample_latency(profile.latency)
            input_size = self._sample_range(profile.input_size_mb)
            output_size = self._sample_range(profile.output_size_mb)
            energy = self._sample_range(profile.energy_j)
            accuracy = self._sample_range(profile.accuracy)
            failed = self._rng.random() < profile.failure_rate

        return SyntheticExecution(
            task_type=task_type,
            target=resolved,
            latency_ms=round(latency, 4),
            cpu_cores=profile.resources.cpu_cores,
            gpu_units=workload.accelerator_demand_tops,
            memory_mb=profile.resources.memory_mb,
            input_size_mb=round(input_size, 6),
            output_size_mb=round(output_size, 6),
            energy_j=round(energy, 6),
            accuracy=round(accuracy, 6),
            failed=failed,
            max_concurrency=profile.max_concurrency,
        )

    def _sample_range(self, values: ValueRange) -> float:
        return self._rng.triangular(values.minimum, values.maximum, values.typical)

    def _sample_latency(self, latency: LatencyDistribution) -> float:
        # Piecewise inverse CDF preserves the configured percentile landmarks.
        quantile = self._rng.random()
        if quantile <= 0.5:
            floor = max(0.001, latency.p50_ms * 0.7)
            return floor + (latency.p50_ms - floor) * quantile / 0.5
        if quantile <= 0.95:
            return latency.p50_ms + (latency.p95_ms - latency.p50_ms) * (quantile - 0.5) / 0.45
        if quantile <= 0.99:
            return latency.p95_ms + (latency.p99_ms - latency.p95_ms) * (quantile - 0.95) / 0.04
        return latency.p99_ms * (1 + 0.15 * (quantile - 0.99) / 0.01)


@dataclass(frozen=True)
class SimulatedComponent:
    """Catalog-backed execution facade for simulated agents."""

    workload: SyntheticWorkload
    target: ExecutionTarget
    sampler: SyntheticSampler

    @property
    def max_concurrency(self) -> int:
        return self.workload.profile_for(self.target).max_concurrency

    def can_accept(self, active_executions: int) -> bool:
        return active_executions < self.max_concurrency

    def execute_sample(self) -> SyntheticExecution:
        return self.sampler.sample(self.workload.task_type, self.target)


class SyntheticWorkloadCatalog:
    """Registry for workload definitions loaded from code or JSON."""

    def __init__(self, workloads: Iterable[SyntheticWorkload] = ()) -> None:
        self._workloads: dict[str, SyntheticWorkload] = {}
        for workload in workloads:
            self.register(workload)

    def __len__(self) -> int:
        return len(self._workloads)

    def __iter__(self):
        return iter(self._workloads.values())

    def get(self, task_type: str) -> SyntheticWorkload:
        try:
            return self._workloads[task_type]
        except KeyError as exc:
            raise KeyError(f"unknown synthetic workload: {task_type}") from exc

    def register(self, workload: SyntheticWorkload, *, replace: bool = False) -> None:
        if workload.task_type in self._workloads and not replace:
            raise ValueError(f"synthetic workload already registered: {workload.task_type}")
        self._workloads[workload.task_type] = workload

    def register_dict(self, item: Mapping[str, Any], *, replace: bool = False) -> SyntheticWorkload:
        workload = workload_from_dict(item)
        self.register(workload, replace=replace)
        return workload

    def by_class(self, task_class: TaskClass | str) -> tuple[SyntheticWorkload, ...]:
        resolved = TaskClass(task_class)
        return tuple(item for item in self if item.task_class is resolved)

    def component(
        self,
        task_type: str,
        target: ExecutionTarget | str,
        *,
        seed: int | None = None,
        deterministic: bool = False,
    ) -> SimulatedComponent:
        resolved = ExecutionTarget(target)
        workload = self.get(task_type)
        profile = workload.profile_for(resolved)
        if not profile.supported:
            raise UnsupportedTargetError(f"{task_type} is not supported on {resolved.value}")
        return SimulatedComponent(workload, resolved, SyntheticSampler(self, seed=seed, deterministic=deterministic))

    @classmethod
    def load(cls, path: str | Path = DEFAULT_SYNTHETIC_WORKLOAD_PATH) -> "SyntheticWorkloadCatalog":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(workload_from_dict(item) for item in raw["workloads"])


def workload_from_dict(item: Mapping[str, Any]) -> SyntheticWorkload:
    """Build a validated workload from a dictionary or JSON object."""

    profiles = []
    for target_name, value in item["profiles"].items():
        resources = value["resources"]
        latency = value["latency_ms"]
        profiles.append(
            SyntheticRuntimeProfile(
                target=ExecutionTarget(target_name),
                latency=LatencyDistribution(
                    float(latency["p50"]), float(latency["p95"]), float(latency["p99"])
                ),
                resources=ResourceDemand(
                    float(resources["cpu_cores"]),
                    float(resources["gpu_units"]),
                    float(resources["memory_mb"]),
                ),
                input_size_mb=ValueRange.from_dict(value["input_size_mb"]),
                output_size_mb=ValueRange.from_dict(value["output_size_mb"]),
                energy_j=ValueRange.from_dict(value["energy_j"]),
                failure_rate=float(value["failure_rate"]),
                accuracy=ValueRange.from_dict(value["accuracy"]),
                max_concurrency=int(value["max_concurrency"]),
                supported=bool(value.get("supported", True)),
            )
        )
    def ports(values: Iterable[Mapping[str, Any]]) -> tuple[PortDefinition, ...]:
        return tuple(
            PortDefinition(value["name"], value["semantic_type"])
            for value in values
        )

    return SyntheticWorkload(
        task_type=item["task_type"],
        display_name=item["display_name"],
        task_class=TaskClass(item["task_class"]),
        description=item.get("description", ""),
        model_variant=item.get("model_variant", "synthetic"),
        # Resource units changed from target-relative ``gpu_units`` to a
        # workload-level absolute TOPS demand. Missing data is ambiguous and
        # must fail closed instead of silently becoming a CPU-only workload.
        accelerator_demand_tops=float(item["accelerator_demand_tops"]),
        inputs=ports(item.get("inputs", [])),
        outputs=ports(item.get("outputs", [])),
        profiles=tuple(profiles),
    )


def load_default_synthetic_workloads() -> SyntheticWorkloadCatalog:
    return SyntheticWorkloadCatalog.load()
