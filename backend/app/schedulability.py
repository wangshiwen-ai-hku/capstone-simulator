"""Generation-time schedulability checks for synthetic Studio scenes."""

from __future__ import annotations

from dataclasses import dataclass, replace

from mars.profiling import profile_catalog_from_workloads
from mars.network import NetworkTopology
from mars.scheduler import allowed_nodes, estimate_candidate
from mars.synthetic_workloads import load_default_synthetic_workloads

from .mars_adapter import (
    build_link_snapshots,
    build_link_specs,
    build_node_snapshots,
    build_node_specs,
    build_workflow,
)
from .schemas import BenchmarkScene


@dataclass(frozen=True)
class TaskSchedulability:
    task_id: str
    feasible_node_ids: tuple[str, ...]
    rejection_reasons: tuple[str, ...]

    @property
    def feasible(self) -> bool:
        return bool(self.feasible_node_ids)


class SceneSchedulabilityError(ValueError):
    """A generated scene contains a task with no executable candidate."""


def audit_scene_schedulability(
    scene: BenchmarkScene,
) -> tuple[TaskSchedulability, ...]:
    """Check every task independently against initial resources and links.

    Dependencies are intentionally removed from each probe. The check asks
    whether the task can execute once its parents have produced their data;
    DAG correctness remains the responsibility of ``validate_scene``.
    """

    workflow = build_workflow(scene)
    node_specs = build_node_specs(scene)
    node_snapshots = build_node_snapshots(scene)
    specs_by_id = {item.node_id: item for item in node_specs}
    snapshots_by_id = {item.node_id: item for item in node_snapshots}
    profiles = profile_catalog_from_workloads(
        load_default_synthetic_workloads()
    )

    topology = NetworkTopology(
        (item.node_id for item in node_specs),
        build_link_specs(scene),
        build_link_snapshots(scene),
        node_online={
            item.node_id: item.online for item in node_snapshots
        },
    )
    reports: list[TaskSchedulability] = []
    for task in workflow.tasks:
        probe = replace(task, dependency_task_ids=())
        feasible: list[str] = []
        reasons: list[str] = []
        eligible = allowed_nodes(
            probe,
            node_specs,
            snapshots_by_id,
        )
        if not eligible:
            reasons.append("placement_constraints_reject_all_nodes")
        for node in eligible:
            estimate = estimate_candidate(
                probe,
                node,
                ready_time_ms=task.arrival_time_ms,
                node_available_ms=task.arrival_time_ms,
                node_specs=specs_by_id,
                node_snapshots=snapshots_by_id,
                parent_artifacts=(),
                profiles=profiles,
                topology=topology,
            )
            if not estimate.feasible:
                reasons.append(f"{node.node_id}:{estimate.reason}")
                continue
            snapshot = snapshots_by_id[node.node_id]
            demand = estimate.resource_demand
            available = (
                node.cpu_capacity * (1.0 - snapshot.cpu_util),
                node.gpu_capacity * (1.0 - snapshot.gpu_util),
                node.memory_gb * (1.0 - snapshot.memory_util),
            )
            required = (
                demand.cpu_units,
                demand.gpu_units,
                demand.memory_gb,
            )
            labels = ("cpu", "gpu", "memory")
            deficits = [
                f"{label}_background_capacity_insufficient"
                for label, need, free in zip(labels, required, available)
                if need > free + 1e-9
            ]
            if deficits:
                reasons.extend(
                    f"{node.node_id}:{reason}" for reason in deficits
                )
                continue
            feasible.append(node.node_id)
        reports.append(
            TaskSchedulability(
                task_id=task.task_id,
                feasible_node_ids=tuple(feasible),
                rejection_reasons=tuple(dict.fromkeys(reasons)),
            )
        )
    return tuple(reports)


def ensure_generated_scene_schedulable(
    scene: BenchmarkScene,
) -> tuple[str, ...]:
    """Repair synthetic background load, then reject hard infeasibility."""

    repairs: list[str] = []
    resources = {item.node_id: item for item in scene.initial_resources}
    workflow = build_workflow(scene)
    task_by_id = {item.task_id: item for item in workflow.tasks}
    node_specs = {item.node_id: item for item in build_node_specs(scene)}
    profiles = profile_catalog_from_workloads(
        load_default_synthetic_workloads()
    )
    topology = NetworkTopology(
        node_specs,
        build_link_specs(scene),
        build_link_snapshots(scene),
        node_online={
            item.node_id: item.online
            for item in build_node_snapshots(scene)
        },
    )

    for report in audit_scene_schedulability(scene):
        if report.feasible:
            continue
        task = task_by_id[report.task_id]
        # Only generated utilization is repairable. Task demand and declared
        # hardware capacity remain immutable absolute facts.
        probe = replace(task, dependency_task_ids=())
        snapshots = {
            item.node_id: item for item in build_node_snapshots(scene)
        }
        for node in allowed_nodes(
            probe,
            node_specs.values(),
            snapshots,
        ):
            profile = profiles.lookup(task.spec.task_type, node.kind)
            if profile is not None and not profile.supported:
                continue
            estimate = estimate_candidate(
                probe,
                node,
                ready_time_ms=task.arrival_time_ms,
                node_available_ms=task.arrival_time_ms,
                node_specs=node_specs,
                node_snapshots=snapshots,
                parent_artifacts=(),
                profiles=profiles,
                topology=topology,
            )
            if not estimate.feasible:
                continue
            cpu = (
                profile.cpu_units
                if profile is not None and profile.cpu_units is not None
                else max(0.05, task.spec.compute_demand * 0.15)
            )
            gpu = estimate.resource_demand.gpu_units
            memory = (
                profile.peak_memory_mb / 1024.0
                if profile is not None
                else max(0.05, task.spec.compute_demand * 0.08)
            )
            if (
                cpu > node.cpu_capacity + 1e-9
                or gpu > node.gpu_capacity + 1e-9
                or memory > node.memory_gb + 1e-9
            ):
                continue
            state = resources[node.node_id]
            before = (state.cpu_util, state.gpu_util, state.memory_util)
            state.cpu_util = min(
                state.cpu_util,
                max(0.0, 1.0 - cpu / node.cpu_capacity - 0.05),
            )
            state.gpu_util = min(
                state.gpu_util,
                max(
                    0.0,
                    1.0 - gpu / max(node.gpu_capacity, 1e-9) - 0.05,
                ),
            )
            state.memory_util = min(
                state.memory_util,
                max(0.0, 1.0 - memory / node.memory_gb - 0.05),
            )
            after = (state.cpu_util, state.gpu_util, state.memory_util)
            if after != before:
                repairs.append(f"{report.task_id}:{node.node_id}")
            break

    remaining = [
        item for item in audit_scene_schedulability(scene)
        if not item.feasible
    ]
    if remaining:
        detail = "; ".join(
            f"{item.task_id} ({', '.join(item.rejection_reasons)})"
            for item in remaining
        )
        raise SceneSchedulabilityError(
            "generated scene has no executable candidate for: " + detail
        )
    return tuple(repairs)


__all__ = [
    "SceneSchedulabilityError",
    "TaskSchedulability",
    "audit_scene_schedulability",
    "ensure_generated_scene_schedulable",
]
