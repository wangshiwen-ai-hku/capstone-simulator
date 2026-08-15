"""Immutable, data-only capture of one coordinator workflow run."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass
import enum
import math
from types import MappingProxyType
from typing import TYPE_CHECKING

from .coordinator import CoordinatorReport
from .domain.topology import LinkSnapshot, LinkSpec, NodeSnapshot, NodeSpec
from .domain.workflow import WorkflowSpec
from .profiling import ExecutionProfile

if TYPE_CHECKING:
    from .optimizers.base import SchedulingPlan
    from .optimizers.formulation import SchedulingFormulation


class _FrozenList(tuple):
    """Immutable list snapshot that preserves its serialized container type."""


@dataclass(frozen=True, slots=True)
class RunArtifact:
    """Self-contained inputs and raw outputs for one coordinator run.

    This is deliberately pre-evaluation evidence.  It captures the initial
    declared facts, every final per-epoch scheduling plan retained by the
    coordinator, and the raw coordinator report without defining or attaching
    any later benchmark/evaluation semantics.
    """

    run_id: str
    workflow: WorkflowSpec
    node_specs: tuple[NodeSpec, ...]
    node_snapshots: tuple[NodeSnapshot, ...]
    link_specs: tuple[LinkSpec, ...]
    link_snapshots: tuple[LinkSnapshot, ...]
    profiles: tuple[ExecutionProfile, ...]
    raw_report: CoordinatorReport
    algorithm: str
    formulation: str
    seed: int
    deterministic: bool
    max_attempts: int
    network_jitter: float
    resource_noise: float
    schema_version: str = "mars.run-artifact.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_specs", tuple(self.node_specs))
        object.__setattr__(
            self,
            "node_snapshots",
            tuple(self.node_snapshots),
        )
        object.__setattr__(self, "link_specs", tuple(self.link_specs))
        object.__setattr__(
            self,
            "link_snapshots",
            tuple(self.link_snapshots),
        )
        object.__setattr__(self, "profiles", tuple(self.profiles))
        object.__setattr__(self, "run_id", self.run_id.strip())
        object.__setattr__(self, "algorithm", self.algorithm.strip())
        object.__setattr__(self, "formulation", self.formulation.strip())

        if not self.schema_version.strip():
            raise ValueError("run artifact schema_version must be non-blank")
        if not self.run_id:
            raise ValueError("run artifact run_id must be non-blank")
        if not self.algorithm:
            raise ValueError("run artifact algorithm must be non-blank")
        if not isinstance(self.workflow, WorkflowSpec):
            raise TypeError("run artifact workflow must be a WorkflowSpec")
        if not isinstance(self.raw_report, CoordinatorReport):
            raise TypeError(
                "run artifact raw_report must be a CoordinatorReport"
            )
        object.__setattr__(
            self,
            "raw_report",
            _snapshot_report(self.raw_report),
        )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("run artifact seed must be an integer")
        if self.seed < 0:
            raise ValueError("run artifact seed must be non-negative")
        if not isinstance(self.deterministic, bool):
            raise TypeError("run artifact deterministic must be bool")
        if (
            not isinstance(self.max_attempts, int)
            or isinstance(self.max_attempts, bool)
            or self.max_attempts < 1
        ):
            raise ValueError("run artifact max_attempts must be positive")
        if (
            not math.isfinite(self.network_jitter)
            or self.network_jitter < 0.0
        ):
            raise ValueError(
                "run artifact network_jitter must be non-negative"
            )
        if (
            not math.isfinite(self.resource_noise)
            or not 0.0 <= self.resource_noise <= 1.0
        ):
            raise ValueError(
                "run artifact resource_noise must be in [0, 1]"
            )

        report_workflow_id = str(
            self.raw_report.workflow.get("workflow_id", "")
        )
        if report_workflow_id != self.workflow.workflow_id:
            raise ValueError(
                "run artifact workflow does not match the coordinator report"
            )

    @property
    def scheduling_plans(self) -> tuple[SchedulingPlan, ...]:
        """The validated final plan retained for every scheduling epoch."""

        return self.raw_report.scheduling_plans

    def as_dict(self) -> dict[str, object]:
        """Return a complete JSON-compatible representation."""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "workflow": _to_data(self.workflow),
            "node_specs": _to_data(self.node_specs),
            "node_snapshots": _to_data(self.node_snapshots),
            "link_specs": _to_data(self.link_specs),
            "link_snapshots": _to_data(self.link_snapshots),
            "profiles": _to_data(self.profiles),
            "raw_report": self.raw_report.as_dict(),
            "scheduling_plans": _to_data(self.scheduling_plans),
            "algorithm": self.algorithm,
            "formulation": self.formulation,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "max_attempts": self.max_attempts,
            "network_jitter": self.network_jitter,
            "resource_noise": self.resource_noise,
        }


def build_run_artifact(
    *,
    run_id: str,
    workflow: WorkflowSpec,
    node_specs: Iterable[NodeSpec],
    node_snapshots: Iterable[NodeSnapshot],
    link_specs: Iterable[LinkSpec],
    link_snapshots: Iterable[LinkSnapshot],
    profiles: Iterable[ExecutionProfile],
    raw_report: CoordinatorReport,
    algorithm: str,
    formulation: str | SchedulingFormulation | None,
    seed: int,
    deterministic: bool,
    max_attempts: int,
    network_jitter: float,
    resource_noise: float,
) -> RunArtifact:
    """Snapshot one completed coordinator run without evaluating it."""

    return RunArtifact(
        run_id=run_id,
        workflow=workflow,
        node_specs=tuple(node_specs),
        node_snapshots=tuple(node_snapshots),
        link_specs=tuple(link_specs),
        link_snapshots=tuple(link_snapshots),
        profiles=tuple(profiles),
        raw_report=raw_report,
        algorithm=algorithm,
        formulation=_formulation_id(formulation),
        seed=seed,
        deterministic=deterministic,
        max_attempts=max_attempts,
        network_jitter=network_jitter,
        resource_noise=resource_noise,
    )


def _formulation_id(
    formulation: str | SchedulingFormulation | None,
) -> str:
    if formulation is None:
        return ""
    if isinstance(formulation, str):
        return formulation
    spec = getattr(formulation, "spec", None)
    formulation_id = getattr(spec, "formulation_id", None)
    if not isinstance(formulation_id, str):
        raise TypeError(
            "run artifact formulation must be an id or SchedulingFormulation"
        )
    return formulation_id


def _snapshot_report(report: CoordinatorReport) -> CoordinatorReport:
    """Own a recursively read-only snapshot of mutable report payloads."""

    return CoordinatorReport(
        workflow=_freeze_data(report.workflow),
        metrics=_freeze_data(report.metrics),
        task_results=tuple(
            _freeze_data(item) for item in report.task_results
        ),
        agents=tuple(_freeze_data(item) for item in report.agents),
        data_edges=tuple(
            _freeze_data(item) for item in report.data_edges
        ),
        events=tuple(report.events),
        logs=tuple(report.logs),
        scheduling_plans=tuple(report.scheduling_plans),
    )


def _freeze_data(value: object) -> object:
    """Copy JSON-shaped report data into immutable containers."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _freeze_data(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, list):
        return _FrozenList(_freeze_data(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_data(item) for item in value)
    return value


def _to_data(value: object) -> object:
    """Recursively serialize frozen contracts without deepcopying proxies."""

    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _to_data(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _to_data(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_to_data(item) for item in value]
    return value


__all__ = ["RunArtifact", "build_run_artifact"]
